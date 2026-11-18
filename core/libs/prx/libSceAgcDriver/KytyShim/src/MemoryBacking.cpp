#include "kernel/memory.h"

#include "MemoryShim.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libkernel/DirectMemory/DirectMemory.hpp"
#include "prx/libkernel/Pthread/include/Pthread.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>
#include <x86intrin.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#include <sys/mman.h>
#include <sys/uio.h>
#include <unistd.h>
#endif

namespace Libs::LibKernel::Memory {

namespace {

#ifdef _WIN32
constexpr int PROT_NONE = 0;
constexpr int PROT_READ = 1;
constexpr int PROT_WRITE = 2;
constexpr int PROT_EXEC = 4;
#endif

constexpr uint64_t PageBits = 14;
constexpr uint64_t PageSize = uint64_t {1} << PageBits;
constexpr uint64_t PageMask = PageSize - 1u;
constexpr uint64_t BucketPages = uint64_t {1} << 16u;
constexpr uint64_t BucketCount = ((kExtendedMemoryBase + kExtendedMemorySize) >> PageBits) / BucketPages;
constexpr uint32_t PrivateEntry = ~uint32_t {0};
constexpr uint64_t HostPageSize = 0x1000;
constexpr uint64_t HostPageMask = HostPageSize - 1u;
constexpr size_t ProbeBatch = 512;
#ifndef _WIN32
constexpr uint64_t SeekMinBytes = uint64_t {64} << 10u;
constexpr uint64_t SeekFirstChunk = uint64_t {64} << 10u;
constexpr uint64_t SeekLastChunk = uint64_t {1} << 20u;
#endif

class PageTable {
public:
	[[nodiscard]] uint32_t Load(uint64_t page) const {
		if (page / BucketPages >= BucketCount) return 0;
		const auto* entries = m_buckets[page / BucketPages].load(std::memory_order_acquire);
		return entries != nullptr ? entries[page % BucketPages].load(std::memory_order_relaxed) : 0;
	}

	[[nodiscard]] uint32_t LoadBacking(uint64_t page) const {
		const auto entry = Load(page);
		return entry == PrivateEntry ? 0 : entry;
	}

	void Store(uint64_t vaddr, uint64_t size, uint32_t first, bool step) {
		const auto end = vaddr + size;
		auto entry = first;
		for (auto page = vaddr >> PageBits; page < end >> PageBits; ++page) {
			auto& bucket = m_buckets[page / BucketPages];
			auto* entries = bucket.load(std::memory_order_acquire);
			if (entries == nullptr) {
				if (first == 0) {
					page = (page / BucketPages + 1u) * BucketPages - 1u;
					continue;
				}
				auto* created = new std::atomic<uint32_t>[BucketPages] {};
				if (bucket.compare_exchange_strong(entries, created, std::memory_order_acq_rel)) {
					entries = created;
				} else {
					delete[] created;
				}
			}
			entries[page % BucketPages].store(entry, std::memory_order_relaxed);
			if (step) ++entry;
		}
	}

	[[nodiscard]] bool AnyMapped(uint64_t vaddr, uint64_t size) const {
		for (auto page = vaddr >> PageBits; page < (vaddr + size + PageMask) >> PageBits; ++page) {
			if (page / BucketPages >= BucketCount) return false;
			if (m_buckets[page / BucketPages].load(std::memory_order_acquire) == nullptr) {
				page = (page / BucketPages + 1u) * BucketPages - 1u;
				continue;
			}
			if (Load(page) != 0) return true;
		}
		return false;
	}

	void StoreExecutable(uint64_t start, uint64_t end, bool executable) {
		for (auto host = start & ~HostPageMask; host < end; host += HostPageSize) {
			const auto page = host >> PageBits;
			if (page / BucketPages >= BucketCount) return;
			auto& bucket = m_exec[page / BucketPages];
			auto* bits = bucket.load(std::memory_order_acquire);
			if (bits == nullptr) {
				if (!executable) {
					host = (((page / BucketPages + 1u) * BucketPages) << PageBits) - HostPageSize;
					continue;
				}
				auto* created = new std::atomic<uint8_t>[BucketPages] {};
				if (bucket.compare_exchange_strong(bits, created, std::memory_order_acq_rel)) {
					bits = created;
				} else {
					delete[] created;
				}
			}
			const auto bit = static_cast<uint8_t>(1u << ((host >> 12u) & 3u));
			if (executable) {
				bits[page % BucketPages].fetch_or(bit, std::memory_order_relaxed);
			} else {
				bits[page % BucketPages].fetch_and(static_cast<uint8_t>(~bit), std::memory_order_relaxed);
			}
		}
	}

	[[nodiscard]] bool Executable(uint64_t vaddr) const {
		const auto page = vaddr >> PageBits;
		if (page / BucketPages >= BucketCount) return false;
		const auto* bits = m_exec[page / BucketPages].load(std::memory_order_acquire);
		return bits != nullptr && (bits[page % BucketPages].load(std::memory_order_relaxed) & (1u << ((vaddr >> 12u) & 3u))) != 0;
	}

	[[nodiscard]] uint8_t Probed(uint64_t host) const {
		const auto page = host >> PageBits;
		if (page / BucketPages >= BucketCount) return ProbeUnknown;
		const auto* bits = m_probe[page / BucketPages].load(std::memory_order_acquire);
		if (bits == nullptr) return ProbeUnknown;
		return static_cast<uint8_t>((bits[page % BucketPages].load(std::memory_order_acquire) >> ProbeShift(host)) & ProbeMask);
	}

	void StoreProbe(uint64_t host, bool readable) {
		const auto page = host >> PageBits;
		if (page / BucketPages >= BucketCount) return;
		auto& bucket = m_probe[page / BucketPages];
		auto* bits = bucket.load(std::memory_order_acquire);
		if (bits == nullptr) {
			auto* created = new std::atomic<uint8_t>[BucketPages] {};
			if (bucket.compare_exchange_strong(bits, created, std::memory_order_acq_rel)) {
				bits = created;
			} else {
				delete[] created;
			}
		}
		const auto shift = ProbeShift(host);
		const auto value = static_cast<uint8_t>((readable ? ProbeReadable : ProbeUnreadable) << shift);
		auto& slot = bits[page % BucketPages];
		auto current = slot.load(std::memory_order_relaxed);
		while (!slot.compare_exchange_weak(current, static_cast<uint8_t>((current & ~(ProbeMask << shift)) | value), std::memory_order_acq_rel)) {}
	}

	void ClearProbes(uint64_t start, uint64_t end) {
		for (auto host = start & ~HostPageMask; host < end; host += HostPageSize) {
			const auto page = host >> PageBits;
			if (page / BucketPages >= BucketCount) return;
			auto* bits = m_probe[page / BucketPages].load(std::memory_order_acquire);
			if (bits == nullptr) {
				host = (((page / BucketPages + 1u) * BucketPages) << PageBits) - HostPageSize;
				continue;
			}
			if ((host & PageMask) == 0 && end - host >= PageSize) {
				bits[page % BucketPages].store(0, std::memory_order_release);
				host += PageSize - HostPageSize;
				continue;
			}
			bits[page % BucketPages].fetch_and(static_cast<uint8_t>(~(ProbeMask << ProbeShift(host))), std::memory_order_acq_rel);
		}
	}

	static constexpr uint8_t ProbeUnknown = 0;
	static constexpr uint8_t ProbeUnreadable = 1;
	static constexpr uint8_t ProbeReadable = 3;

	template <typename Run>
	void ForEachMappedRun(Run&& run) const {
		uint64_t start = 0;
		bool open = false;
		for (uint64_t bucket = 0; bucket < BucketCount; ++bucket) {
			const auto* entries = m_buckets[bucket].load(std::memory_order_acquire);
			for (uint64_t index = 0; index < BucketPages; ++index) {
				const bool mapped = entries != nullptr && entries[index].load(std::memory_order_relaxed) != 0;
				const auto address = (bucket * BucketPages + index) << PageBits;
				if (mapped && !open) {
					start = address;
					open = true;
				} else if (!mapped && open) {
					run(start, address - start);
					open = false;
				}
				if (entries == nullptr && !open) break;
			}
		}
		if (open) run(start, (BucketCount * BucketPages << PageBits) - start);
	}

private:
	static constexpr uint8_t ProbeMask = 3;

	[[nodiscard]] static unsigned ProbeShift(uint64_t host) {
		return static_cast<unsigned>(((host >> 12u) & 3u) * 2u);
	}

	std::array<std::atomic<std::atomic<uint32_t>*>, BucketCount> m_buckets {};
	std::array<std::atomic<std::atomic<uint8_t>*>, BucketCount> m_exec {};
	std::array<std::atomic<std::atomic<uint8_t>*>, BucketCount> m_probe {};
};

struct ViewState {
	uint8_t* base = nullptr;
	uint64_t guest_bytes = 0;
	uint64_t view_bytes = 0;
	uint64_t arena_start = 0;
	uint64_t arena_end = 0;
#ifndef _WIN32
	pid_t pid = 0;
	int guest_file = -1;
	int direct_file = -1;
#endif
	std::atomic<bool> enabled {false};
};

PageTable& Pages() {
	static auto* pages = new PageTable;
	return *pages;
}

ViewState& State() {
	static auto* state = new ViewState;
	return *state;
}

std::atomic<uint64_t> g_probe_epoch {0};
std::atomic<uint64_t> g_probe_calls {0};
std::atomic<bool> g_any_probe {false};

void ForgetProbes(uint64_t start, uint64_t size) {
	g_probe_epoch.fetch_add(1, std::memory_order_seq_cst);
	if (g_any_probe.load(std::memory_order_seq_cst)) Pages().ClearProbes(start, start + size);
}

std::atomic<uint64_t> g_backing_epoch {0};
constexpr uint64_t BackingChangeHistory = 4096;
std::mutex g_backing_changes_mutex;
std::array<std::pair<uint64_t, uint64_t>, BackingChangeHistory> g_backing_changes {};

void BumpBackingEpoch(uint64_t start, uint64_t size) noexcept {
	std::lock_guard lock(g_backing_changes_mutex);
	const auto epoch = g_backing_epoch.load(std::memory_order_relaxed) + 1;
	g_backing_changes[epoch % BackingChangeHistory] = {start, size};
	g_backing_epoch.store(epoch, std::memory_order_release);
}

bool FirstClamp(uint64_t vaddr, uint64_t size, uint64_t clamped) {
	static std::mutex mutex;
	static std::set<std::tuple<uint64_t, uint64_t, uint64_t>> reported;
	std::lock_guard lock(mutex);
	return reported.emplace(vaddr, size, clamped).second;
}

struct RegCeilConfig {
	int      schedule   = 0;
	bool     query      = false;
	int      stillvalid = RegCeil::LeverExact;
	int      rearm      = RegCeil::LeverExact;
	uint64_t start_s    = 0;
};

[[noreturn]] void RegCeilBadValue(const char* name, const char* value, const char* allowed) {
	std::fprintf(stderr, "[reg] %s=%s is not one of %s\n", name, value, allowed);
	std::fflush(stderr);
	std::abort();
}

const char* RegCeilEnv(const char* name) {
	const char* value = std::getenv(name);
	return value == nullptr || *value == 0 ? nullptr : value;
}

const RegCeilConfig& RegCfg() {
	static const RegCeilConfig config = [] {
		RegCeilConfig c;
		if (const char* v = RegCeilEnv("KYTY_REG_CEIL")) {
			const std::string_view text(v);
			if (text == "0") {
			} else if (text == "stats") {
				c.schedule = 1;
			} else if (text == "on" || text == "1" || text == "ab") {
				c.schedule   = text == "ab" ? 3 : 2;
				c.query      = true;
				c.stillvalid = RegCeil::LeverSkip;
			} else if (text == "all" || text == "all-ab") {
				c.schedule   = text == "all-ab" ? 3 : 2;
				c.query      = true;
				c.stillvalid = RegCeil::LeverSkip;
				c.rearm      = RegCeil::LeverSkip;
			} else {
				RegCeilBadValue("KYTY_REG_CEIL", v, "0, stats, on, ab, all, all-ab");
			}
		}
		if (const char* v = RegCeilEnv("KYTY_STILLVALID_CEIL")) {
			const std::string_view text(v);
			if (text == "0") {
				c.stillvalid = RegCeil::LeverExact;
			} else if (text == "memo") {
				c.stillvalid = RegCeil::LeverMemo;
			} else if (text == "skip") {
				c.stillvalid = RegCeil::LeverSkip;
			} else {
				RegCeilBadValue("KYTY_STILLVALID_CEIL", v, "0, memo, skip");
			}
		}
		if (const char* v = RegCeilEnv("KYTY_REARM_CEIL")) {
			const std::string_view text(v);
			if (text == "0") {
				c.rearm = RegCeil::LeverExact;
			} else if (text == "protected") {
				c.rearm = RegCeil::LeverProtected;
			} else if (text == "skip") {
				c.rearm = RegCeil::LeverSkip;
			} else {
				RegCeilBadValue("KYTY_REARM_CEIL", v, "0, protected, skip");
			}
		}
		if (c.schedule == 0 && (c.stillvalid != RegCeil::LeverExact || c.rearm != RegCeil::LeverExact)) {
			c.schedule = 2;
		}
		if (const char* v = RegCeilEnv("KYTY_CEIL_AB")) {
			const std::string_view text(v);
			if (text != "0" && text != "1") RegCeilBadValue("KYTY_CEIL_AB", v, "0, 1");
			if (c.schedule >= 2) c.schedule = text == "1" ? 3 : 2;
		}
		if (const char* v = RegCeilEnv("KYTY_CEIL_START_S")) {
			char* end     = nullptr;
			const auto n  = std::strtoull(v, &end, 10);
			if (end == v || *end != 0 || n > 3600) RegCeilBadValue("KYTY_CEIL_START_S", v, "0 to 3600");
			c.start_s = n;
		}
		return c;
	}();
	return config;
}

int RegCeilMode() {
	return RegCfg().schedule;
}

std::atomic<bool>     g_reg_on {false};
std::atomic<uint64_t> g_reg_generation {1};
std::atomic<uint64_t> g_reg_frames {0};
uint64_t              g_reg_origin_ns = 0;

struct RegCounters {
	int      tid = 0;
	uint64_t calls[RegCeil::FnCount] {};
	uint64_t hits[RegCeil::FnCount] {};
	uint64_t valid[RegCeil::FnCount] {};
	uint64_t cycles[RegCeil::FnCount] {};
};

std::mutex                g_reg_threads_mutex;
std::vector<RegCounters*> g_reg_threads;
thread_local RegCounters* t_reg = nullptr;

RegCounters& RegC() {
	if (t_reg == nullptr) {
		t_reg = new RegCounters;
#ifndef _WIN32
		t_reg->tid = static_cast<int>(::gettid());
#endif
		std::lock_guard lock(g_reg_threads_mutex);
		g_reg_threads.push_back(t_reg);
	}
	return *t_reg;
}

struct RegMemoEntry {
	uint64_t vaddr      = 0;
	uint64_t size       = 0;
	uint64_t result     = 0;
	uint64_t backing    = 0;
	uint64_t probe      = 0;
	uint64_t generation = 0;
	int      kind       = 0;
};
constexpr size_t           RegMemoSlots = 4096;
thread_local RegMemoEntry* t_reg_memo   = nullptr;
thread_local int           t_reg_depth  = 0;

RegMemoEntry& RegMemoSlot(int kind, uint64_t vaddr, uint64_t size) {
	if (t_reg_memo == nullptr) t_reg_memo = new RegMemoEntry[RegMemoSlots] {};
	const auto hash = (vaddr * 0x9e3779b97f4a7c15ull) ^ (size * 0xc2b2ae3d27d4eb4full) ^ static_cast<uint64_t>(kind);
	return t_reg_memo[(hash >> 32u) % RegMemoSlots];
}

template <typename Compute>
uint64_t RegMemo(int kind, uint64_t vaddr, uint64_t size, Compute&& compute) {
	const auto& cfg = RegCfg();
	if (cfg.schedule == 0) return compute();
	auto& c = RegC();
	c.calls[kind]++;
	if (cfg.schedule == 1) {
		if (t_reg_depth != 0) return compute();
		t_reg_depth++;
		const auto start  = __rdtsc();
		const auto result = compute();
		c.cycles[kind] += __rdtsc() - start;
		t_reg_depth--;
		return result;
	}
	if (!cfg.query || !g_reg_on.load(std::memory_order_relaxed)) return compute();
	auto&      entry      = RegMemoSlot(kind, vaddr, size);
	const auto backing    = g_backing_epoch.load(std::memory_order_relaxed);
	const auto probe      = g_probe_epoch.load(std::memory_order_relaxed);
	const auto generation = g_reg_generation.load(std::memory_order_relaxed);
	if (entry.kind == kind + 1 && entry.vaddr == vaddr && entry.size == size && entry.generation == generation) {
		c.hits[kind]++;
		if (entry.backing == backing && entry.probe == probe) c.valid[kind]++;
		return entry.result;
	}
	const auto result = compute();
	entry             = {vaddr, size, result, backing, probe, generation, kind + 1};
	return result;
}

#ifndef _WIN32
constexpr uint64_t HostPage = 0x1000;

struct HostRange {
	uint64_t first = 0;
	uint64_t end = 0;
};

HostRange HostPages(uint64_t address, uint64_t bytes) {
	return {address & ~(HostPage - 1u), (address + bytes + HostPage - 1u) & ~(HostPage - 1u)};
}

struct HostWriteHold {
	HostRange pages;
	bool overridden = false;
};

std::mutex g_host_write_holds_mutex;
std::vector<HostWriteHold> g_host_write_holds;

enum class PageKeep : uint8_t { Writable, Untouched };

struct KeptRange {
	HostRange pages;
	PageKeep keep = PageKeep::Writable;
};

#if defined(__linux__)
void KeepThreadStack(void* context, std::uintptr_t first, std::uintptr_t end) {
	static_cast<std::vector<KeptRange>*>(context)->push_back({HostPages(first, end - first), PageKeep::Untouched});
}
#endif

bool ProtectHostPages(uint64_t start, uint64_t stop, int protection, const std::vector<KeptRange>& kept) {
	if (kept.empty()) return ::mprotect(reinterpret_cast<void*>(start), stop - start, protection) == 0;
	std::vector<uint64_t> bounds {start, stop};
	for (const auto& range : kept) {
		if (range.pages.first > start && range.pages.first < stop) bounds.push_back(range.pages.first);
		if (range.pages.end > start && range.pages.end < stop) bounds.push_back(range.pages.end);
	}
	std::sort(bounds.begin(), bounds.end());
	bounds.erase(std::unique(bounds.begin(), bounds.end()), bounds.end());
	const auto protection_at = [&](uint64_t page) {
		auto applied = protection;
		for (const auto& range : kept) {
			if (page < range.pages.first || page >= range.pages.end) continue;
			if (range.keep == PageKeep::Untouched) return -1;
			applied = protection | PROT_READ | PROT_WRITE;
		}
		return applied;
	};
	for (std::size_t index = 0; index + 1 < bounds.size();) {
		const auto applied = protection_at(bounds[index]);
		auto next = index + 1;
		while (next + 1 < bounds.size() && protection_at(bounds[next]) == applied) ++next;
		if (applied >= 0 && ::mprotect(reinterpret_cast<void*>(bounds[index]), bounds[next] - bounds[index], applied) != 0) return false;
		index = next;
	}
	return true;
}
#endif

std::recursive_mutex& HookMutex() {
	static auto* mutex = new std::recursive_mutex;
	return *mutex;
}

Shim::RangeHook g_map_hook = nullptr;
Shim::RangeHook g_unmap_hook = nullptr;

bool ValidRange(uint64_t vaddr, uint64_t size) {
	return size != 0 && UINT64_MAX - vaddr >= size;
}

#ifndef _WIN32
bool SparseReadsEnabled() {
	const char* value = std::getenv("ANYPS5_KYTY_SPARSE_READS");
	if (value == nullptr || *value == '\0') return true;
	const std::string_view text(value);
	if (text != "0" && text != "1") Shim::Fail("KytyShim memory: ANYPS5_KYTY_SPARSE_READS=%s is neither 0 nor 1\n", value);
	return text == "1";
}
#endif

void CopyFromView(uint8_t* destination, uint64_t offset, uint64_t bytes) {
	const auto& state = State();
#ifdef _WIN32
	std::memcpy(destination, state.base + offset, bytes);
#else
	if (bytes < SeekMinBytes || state.guest_file < 0) {
		std::memcpy(destination, state.base + offset, bytes);
		return;
	}
	const auto end = offset + bytes;
	auto chunk = SeekFirstChunk;
	for (auto cursor = offset; cursor < end;) {
		const bool guest = cursor < state.guest_bytes;
		const auto file = guest ? state.guest_file : state.direct_file;
		const auto file_start = guest ? 0 : state.guest_bytes;
		const auto file_stop = guest ? std::min(end, state.guest_bytes) : end;
		auto* out = destination + (cursor - offset);
		const auto data = ::lseek(file, static_cast<off_t>(cursor - file_start), SEEK_DATA);
		if (data < 0 && errno != ENXIO) {
			std::memcpy(out, state.base + cursor, file_stop - cursor);
			cursor = file_stop;
			continue;
		}
		const auto hole_end = data < 0 ? file_stop : std::min(file_stop, file_start + static_cast<uint64_t>(data));
		if (hole_end > cursor) {
			std::memset(out, 0, hole_end - cursor);
			cursor = hole_end;
			chunk = SeekFirstChunk;
			continue;
		}
		const auto copy_end = std::min(file_stop, (cursor & ~(chunk - 1u)) + chunk);
		std::memcpy(out, state.base + cursor, copy_end - cursor);
		cursor = copy_end;
		chunk = std::min(chunk * 2u, SeekLastChunk);
	}
#endif
}

template <typename Region, typename Gap>
void ForEachRun(uint64_t vaddr, uint64_t size, Region&& region, Gap&& gap) {
	const auto& pages = Pages();
	const auto end = vaddr + size;
	for (auto start = vaddr; start < end;) {
		const auto entry = pages.LoadBacking(start >> PageBits);
		const uint32_t step = entry != 0 ? 1 : 0;
		auto expected = entry + step;
		auto stop = (start | PageMask) + 1u;
		while (stop < end && pages.LoadBacking(stop >> PageBits) == expected) {
			stop += PageSize;
			expected += step;
		}
		stop = std::min(stop, end);
		if (entry != 0) {
			region((uint64_t {entry - 1u} << PageBits) + (start & PageMask), start, stop - start);
		} else {
			gap(start, stop - start);
		}
		start = stop;
	}
}

#ifdef _WIN32
bool HostReadable(const MEMORY_BASIC_INFORMATION& memory) {
	constexpr DWORD Readable = PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
	return memory.State == MEM_COMMIT && (memory.Protect & PAGE_GUARD) == 0 && (memory.Protect & Readable) != 0;
}

uint64_t ReadablePrefix(uint64_t start, uint64_t end) {
	for (auto cursor = start; cursor < end;) {
		MEMORY_BASIC_INFORMATION memory {};
		if (VirtualQuery(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(cursor)), &memory, sizeof(memory)) != sizeof(memory) || !HostReadable(memory)) {
			return cursor - start;
		}
		cursor = reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize;
	}
	return end - start;
}

void CopyReadable(uint64_t start, uint8_t* destination, uint64_t bytes) {
	const auto end = start + bytes;
	const HANDLE process = GetCurrentProcess();
	for (auto cursor = start; cursor < end;) {
		SIZE_T read = 0;
		const bool complete = ReadProcessMemory(process, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(cursor)), destination + (cursor - start), end - cursor, &read) != 0;
		if (complete) return;
		cursor += read;
		if (cursor < end) {
			const auto page_end = std::min(end, (cursor | HostPageMask) + 1u);
			SIZE_T page_read = 0;
			if (ReadProcessMemory(process, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(cursor)), destination + (cursor - start), page_end - cursor, &page_read) == 0) {
				std::memset(destination + (cursor - start), 0, page_end - cursor);
			}
			cursor = page_end;
		}
	}
}
#else
size_t ReadOwnMemory(const iovec& local, const iovec* remote, size_t count) {
	const auto read = ::process_vm_readv(State().pid, &local, 1, remote, count, 0);
	if (read >= 0) return static_cast<size_t>(read);
	if (errno != EFAULT) Shim::Fail("KytyShim memory: process_vm_readv on guest memory failed: %s\n", std::strerror(errno));
	return 0;
}

uint64_t ReadablePrefix(uint64_t start, uint64_t end) {
	auto& pages = Pages();
	for (auto cursor = start; cursor < end;) {
		const auto known = pages.Probed(cursor);
		if (known == PageTable::ProbeUnreadable) return cursor - start;
		if (known == PageTable::ProbeReadable) {
			cursor = (cursor | HostPageMask) + 1u;
			continue;
		}
		std::array<iovec, ProbeBatch> remote {};
		std::array<char, ProbeBatch> sink {};
		size_t count = 0;
		for (auto page = cursor; page < end && count < ProbeBatch && (count == 0 || pages.Probed(page) == PageTable::ProbeUnknown); page = (page | HostPageMask) + 1u) {
			remote[count++] = {reinterpret_cast<void*>(static_cast<std::uintptr_t>(page)), 1};
		}
		const auto epoch = g_probe_epoch.load(std::memory_order_seq_cst);
		g_any_probe.store(true, std::memory_order_seq_cst);
		g_probe_calls.fetch_add(1, std::memory_order_relaxed);
		const iovec local {sink.data(), count};
		const auto read = ReadOwnMemory(local, remote.data(), count);
		const auto known_end = std::min(read + 1u, count);
		for (size_t i = 0; i < known_end; ++i) pages.StoreProbe(reinterpret_cast<std::uintptr_t>(remote[i].iov_base), i < read);
		const auto probed_end = (reinterpret_cast<std::uintptr_t>(remote[known_end - 1u].iov_base) | HostPageMask) + 1u;
		if (g_probe_epoch.load(std::memory_order_seq_cst) != epoch) pages.ClearProbes(cursor, probed_end);
		if (read < count) return reinterpret_cast<std::uintptr_t>(remote[read].iov_base) - start;
		cursor = probed_end;
	}
	return end - start;
}
#endif

uint64_t UnreportedMappedPrefix(uint64_t start, uint64_t end) {
	const auto& state = State();
	if (start < state.arena_start || start >= state.arena_end) return 0;
	return ReadablePrefix(start, std::min(end, state.arena_end));
}

bool AnyUnreportedMapped(uint64_t start, uint64_t end) {
	const auto& state = State();
	end = std::min(end, state.arena_end);
	for (auto cursor = std::max(start, state.arena_start); cursor < end; cursor = (cursor | HostPageMask) + 1u) {
		if (ReadablePrefix(cursor, end) != 0) return true;
	}
	return false;
}

#ifndef _WIN32
void CopyReadable(uint64_t start, uint8_t* destination, uint64_t bytes) {
	const auto end = start + bytes;
	for (auto cursor = start; cursor < end;) {
		std::array<iovec, ProbeBatch> remote {};
		size_t count = 0;
		auto stop = cursor;
		while (stop < end && count < ProbeBatch) {
			const auto next = std::min(end, (stop | HostPageMask) + 1u);
			remote[count++] = {reinterpret_cast<void*>(static_cast<std::uintptr_t>(stop)), next - stop};
			stop = next;
		}
		const iovec local {destination + (cursor - start), stop - cursor};
		cursor += ReadOwnMemory(local, remote.data(), count);
		if (cursor < stop) {
			const auto skipped = std::min(end, (cursor | HostPageMask) + 1u);
			std::memset(destination + (cursor - start), 0, skipped - cursor);
			cursor = skipped;
		}
	}
}
#endif

void CopyUnreported(uint64_t start, uint8_t* destination, uint64_t bytes) {
	const auto& state = State();
	const auto end = start + bytes;
	const auto low = std::max(start, state.arena_start);
	const auto high = std::min(end, state.arena_end);
	if (low >= high) {
		std::memset(destination, 0, bytes);
		return;
	}
	std::memset(destination, 0, low - start);
	CopyReadable(low, destination + (low - start), high - low);
	std::memset(destination + (high - start), 0, end - high);
}

void CopyGuestPages(uint64_t start, uint8_t* destination, uint64_t bytes) {
	const auto& pages = Pages();
	const auto end = start + bytes;
	for (auto cursor = start; cursor < end;) {
		const bool reported = pages.Load(cursor >> PageBits) == PrivateEntry;
		auto stop = std::min(end, (cursor | PageMask) + 1u);
		while (stop < end && (pages.Load(stop >> PageBits) == PrivateEntry) == reported) stop = std::min(end, stop + PageSize);
		if (reported) {
			std::memcpy(destination + (cursor - start), reinterpret_cast<const void*>(cursor), stop - cursor);
		} else {
			CopyUnreported(cursor, destination + (cursor - start), stop - cursor);
		}
		cursor = stop;
	}
}

uint64_t MappedPrefixNow(uint64_t vaddr, uint64_t size) {
	const auto& pages = Pages();
	const auto end = vaddr + size;
	auto cursor = vaddr;
	while (cursor < end) {
		if (pages.Load(cursor >> PageBits) != 0) {
			cursor = std::min(end, (cursor | PageMask) + 1u);
			continue;
		}
		auto stop = std::min(end, (cursor | PageMask) + 1u);
		while (stop < end && pages.Load(stop >> PageBits) == 0) stop = std::min(end, stop + PageSize);
		cursor += UnreportedMappedPrefix(cursor, stop);
		if (cursor < stop) break;
	}
	return cursor - vaddr;
}

uint64_t MappedPrefix(uint64_t vaddr, uint64_t size) {
	return RegMemo(RegCeil::FnMappedPrefix, vaddr, size, [&] { return MappedPrefixNow(vaddr, size); });
}

int HostProtection(Common::VirtualMemory::Mode mode) {
	switch (mode) {
		case Common::VirtualMemory::Mode::Read: return PROT_READ;
		case Common::VirtualMemory::Mode::Write:
		case Common::VirtualMemory::Mode::ReadWrite: return PROT_READ | PROT_WRITE;
		case Common::VirtualMemory::Mode::Execute: return PROT_EXEC;
		case Common::VirtualMemory::Mode::ExecuteRead: return PROT_EXEC | PROT_READ;
		case Common::VirtualMemory::Mode::ExecuteWrite:
		case Common::VirtualMemory::Mode::ExecuteReadWrite: return PROT_EXEC | PROT_READ | PROT_WRITE;
		case Common::VirtualMemory::Mode::NoAccess:
		default: return PROT_NONE;
	}
}

int GuestProtectionBits(Common::VirtualMemory::Mode mode) {
	const auto host = HostProtection(mode);
	return ((host & PROT_READ) != 0 ? 0x01 : 0) | ((host & PROT_WRITE) != 0 ? 0x02 : 0) | ((host & PROT_EXEC) != 0 ? 0x04 : 0);
}

void OnViewChange(std::uintptr_t address, std::size_t bytes, std::uint64_t offset) {
	if (((address | bytes) & PageMask) != 0) {
		Shim::Fail("KytyShim memory: libkernel reported a view change 0x%016" PRIx64 "+0x%zx that is not 16 KiB aligned\n", static_cast<uint64_t>(address), bytes);
	}
	std::lock_guard lock(HookMutex());
	auto& pages = Pages();
	ForgetProbes(address, bytes);
	if (offset == DirectMemoryViewReplacing) {
		if (pages.AnyMapped(address, bytes)) {
			if (g_unmap_hook != nullptr) g_unmap_hook(address, bytes);
			pages.Store(address, bytes, 0, false);
			BumpBackingEpoch(address, bytes);
		}
		GuestArena::GuestArenaNoteHostProtection_nid_no_patch(address, bytes, true);
		return;
	}
	if (offset == DirectMemoryViewUnmapped) {
		const bool mapped = pages.AnyMapped(address, bytes);
		if (mapped && g_unmap_hook != nullptr) g_unmap_hook(address, bytes);
		pages.Store(address, bytes, 0, false);
		BumpBackingEpoch(address, bytes);
		GuestArena::GuestArenaNoteHostProtection_nid_no_patch(address, bytes, true);
		return;
	}
	uint32_t first = PrivateEntry;
	bool step = false;
	if (offset != DirectMemoryViewPrivate) {
		if ((offset & PageMask) != 0 || (offset >> PageBits) + (bytes >> PageBits) >= PrivateEntry) {
			Shim::Fail("KytyShim memory: libkernel reported view offset 0x%016" PRIx64 " for 0x%016" PRIx64 "+0x%zx\n", offset, static_cast<uint64_t>(address), bytes);
		}
		first = static_cast<uint32_t>(offset >> PageBits) + 1u;
		step = true;
	}
	pages.Store(address, bytes, first, step);
	BumpBackingEpoch(address, bytes);
	if (g_map_hook != nullptr) g_map_hook(address, bytes);
}

void OnProtectionChange(std::uintptr_t address, std::size_t bytes, int protection) {
	if (UINT64_MAX - address < bytes) {
		Shim::Fail("KytyShim memory: libkernel reported a protection change 0x%016" PRIx64 "+0x%zx past the address space\n", static_cast<uint64_t>(address), bytes);
	}
	Pages().StoreExecutable(address, address + bytes, protection >= 0 && (protection & 0x04) != 0);
	ForgetProbes(address, bytes);
}

void EnableViewOnce() {
	auto& state = State();
	uint8_t* view = nullptr;
	std::size_t guest_bytes = 0;
	std::size_t view_bytes = 0;
	try {
		if (!DirectMemoryEnableView_nid_no_patch(&OnViewChange, &view, &guest_bytes, &view_bytes)) {
			Shim::Fail("KytyShim memory: the guest arena is not available, so guest memory has no second view\n");
		}
	} catch (const std::exception& error) {
		Shim::Fail("KytyShim memory: enabling the second view of guest memory failed: %s\n", error.what());
	}
	std::uintptr_t arena_base = 0;
	std::size_t arena_bytes = 0;
	GuestArena::GuestArenaRange_nid_postfix(&arena_base, &arena_bytes);
	state.arena_start = arena_base;
	state.arena_end = arena_base + arena_bytes;
#ifndef _WIN32
	state.pid = ::getpid();
#endif
	state.base = view;
	state.guest_bytes = guest_bytes;
	state.view_bytes = view_bytes;
	std::fprintf(stderr, "[kyty] second view of guest memory at %p: guest range 0x%zx bytes, direct memory after it, 0x%zx bytes in all\n", static_cast<void*>(view), guest_bytes, view_bytes);
#ifndef _WIN32
	if (SparseReadsEnabled()) {
		int guest_file = -1;
		int direct_file = -1;
		if (!DirectMemoryViewFiles_nid_no_patch(&guest_file, &direct_file) || guest_file < 0 || direct_file < 0) {
			Shim::Fail("KytyShim memory: libkernel did not give the files behind the second view of guest memory\n");
		}
		state.guest_file = guest_file;
		state.direct_file = direct_file;
	}
#endif
	DirectMemoryWatchProtections_nid_no_patch(&OnProtectionChange);
	state.enabled.store(true, std::memory_order_release);
}

}

namespace Shim {

void Fail(const char* format, ...) {
	va_list args;
	va_start(args, format);
	std::vfprintf(stderr, format, args);
	va_end(args);
	std::fflush(stderr);
	std::abort();
}

bool ViewEnabled() {
	return State().enabled.load(std::memory_order_acquire);
}

#ifndef _WIN32
void HoldHostWrite(std::uintptr_t address, std::size_t bytes) {
	std::lock_guard lock(g_host_write_holds_mutex);
	g_host_write_holds.push_back({HostPages(address, bytes)});
}

bool ReleaseHostWrite(std::uintptr_t address, std::size_t bytes) {
	const auto pages = HostPages(address, bytes);
	std::lock_guard lock(g_host_write_holds_mutex);
	const auto hold = std::find_if(g_host_write_holds.begin(), g_host_write_holds.end(), [&pages](const auto& held) { return held.pages.first == pages.first && held.pages.end == pages.end; });
	if (hold == g_host_write_holds.end()) return false;
	const bool overridden = hold->overridden;
	*hold = g_host_write_holds.back();
	g_host_write_holds.pop_back();
	return overridden;
}
#endif

static bool IsBackedNow(uint64_t vaddr, uint64_t size) {
	if (!ValidRange(vaddr, size)) return false;
	const auto& pages = Pages();
	for (auto page = vaddr >> PageBits; page <= (vaddr + size - 1u) >> PageBits; ++page) {
		if (pages.LoadBacking(page) == 0) return false;
	}
	return true;
}

bool IsBacked(uint64_t vaddr, uint64_t size) {
	return RegMemo(RegCeil::FnIsBacked, vaddr, size, [&] { return uint64_t {IsBackedNow(vaddr, size)}; }) != 0;
}

bool InGuestSpace(uint64_t vaddr, uint64_t size) {
	return ValidRange(vaddr, size) && ViewEnabled() && vaddr < State().guest_bytes && size <= State().guest_bytes - vaddr;
}

bool OverlapsPrivate(uint64_t vaddr, uint64_t size) {
	const auto& pages = Pages();
	const auto end = vaddr + size;
	for (auto page = vaddr & ~PageMask; page < end; page += PageSize) {
		const auto entry = pages.Load(page >> PageBits);
		if (entry == PrivateEntry) return true;
		if (entry == 0 && AnyUnreportedMapped(std::max(page, vaddr & ~HostPageMask), std::min(page + PageSize, (end + HostPageMask) & ~HostPageMask))) return true;
	}
	return false;
}

bool TryWriteBackedRuns(uint64_t vaddr, const void* data, uint64_t size) {
	if (data == nullptr || !InGuestSpace(vaddr, size) || OverlapsPrivate(vaddr, size)) return false;
	const auto* source = static_cast<const uint8_t*>(data);
	auto* base = State().base;
	ForEachRun(
	    vaddr, size,
	    [&](uint64_t offset, uint64_t start, uint64_t bytes) { std::memcpy(base + offset, source + (start - vaddr), bytes); },
	    [](uint64_t, uint64_t) {});
	return true;
}

bool TryWriteBackedQwords(uint64_t vaddr, const uint64_t* values, uint32_t count, uint64_t stride) {
	if (count == 0 || values == nullptr) return false;
	const auto size = uint64_t {count - 1u} * stride + sizeof(uint64_t);
	if (!InGuestSpace(vaddr, size) || OverlapsPrivate(vaddr, size)) return false;
	if (const auto* backing = static_cast<const uint8_t*>(FindBackingPointer(vaddr, size)); backing != nullptr) {
		auto* target = const_cast<uint8_t*>(backing);
		for (uint32_t i = count; i-- > 0;) {
			*reinterpret_cast<volatile uint64_t*>(target + i * stride) = values[i];
		}
		return true;
	}
	for (uint32_t i = count; i-- > 0;) {
		(void)TryWriteBackedRuns(vaddr + i * stride, &values[i], sizeof(uint64_t));
	}
	return true;
}

void SetGpuRangeHooks(RangeHook map, RangeHook unmap) {
	std::lock_guard lock(HookMutex());
	g_map_hook = map;
	g_unmap_hook = unmap;
	if (map != nullptr) Pages().ForEachMappedRun([&](uint64_t start, uint64_t size) { map(start, size); });
}

uint64_t ProbeCalls() {
	return g_probe_calls.load(std::memory_order_relaxed);
}

void EnableView() {
	EnableViewOnce();
}

uint64_t DirectMappedBytes() {
	uint64_t bytes = 0;
	const auto guest_bytes = State().guest_bytes;
	Pages().ForEachMappedRun([&](uint64_t run_start, uint64_t run_size) {
		for (auto page = run_start; page < run_start + run_size; page += PageSize) {
			const auto entry = Pages().LoadBacking(page >> PageBits);
			if (entry != 0 && (uint64_t {entry - 1u} << PageBits) >= guest_bytes) bytes += PageSize;
		}
	});
	return bytes;
}

}

bool TryReadBacking(uint64_t vaddr, void* data, uint64_t size) {
	if (data == nullptr || !Shim::IsBacked(vaddr, size)) return false;
	auto* destination = static_cast<uint8_t*>(data);
	ForEachRun(
	    vaddr, size,
	    [&](uint64_t offset, uint64_t start, uint64_t bytes) { CopyFromView(destination + (start - vaddr), offset, bytes); },
	    [&](uint64_t start, uint64_t bytes) { std::memset(destination + (start - vaddr), 0, bytes); });
	return true;
}

static const void* FindBackingPointerNow(uint64_t vaddr, uint64_t size);

const void* FindBackingPointer(uint64_t vaddr, uint64_t size) {
	return reinterpret_cast<const void*>(static_cast<std::uintptr_t>(
	    RegMemo(RegCeil::FnFindBacking, vaddr, size, [&] { return static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(FindBackingPointerNow(vaddr, size))); })));
}

static const void* FindBackingPointerNow(uint64_t vaddr, uint64_t size) {
	if (!ValidRange(vaddr, size) || !Shim::ViewEnabled()) return nullptr;
	const uint8_t* result = nullptr;
	bool split = false;
	const auto* base = State().base;
	ForEachRun(
	    vaddr, size,
	    [&](uint64_t offset, uint64_t start, uint64_t) {
		    if (start == vaddr) {
			    result = base + offset;
		    } else {
			    split = true;
		    }
	    },
	    [&](uint64_t, uint64_t) { split = true; });
	return split ? nullptr : result;
}

bool TryGetBackingPieces(uint64_t vaddr, uint64_t size, std::vector<std::pair<const uint8_t*, uint64_t>>& pieces) {
	pieces.clear();
	if (!ValidRange(vaddr, size) || !Shim::ViewEnabled()) return false;
	bool complete = true;
	const auto* base = State().base;
	ForEachRun(
	    vaddr, size,
	    [&](uint64_t offset, uint64_t, uint64_t bytes) { pieces.emplace_back(base + offset, bytes); },
	    [&](uint64_t, uint64_t) { complete = false; });
	return complete && !pieces.empty();
}

bool TryGetSparseBackingPieces(uint64_t vaddr, uint64_t size, std::vector<std::pair<const uint8_t*, uint64_t>>& pieces) {
	if (TryGetBackingPieces(vaddr, size, pieces)) return true;
	pieces.clear();
	if (!Shim::InGuestSpace(vaddr, size) || Shim::OverlapsPrivate(vaddr, size)) return false;
	const auto* base = State().base;
	ForEachRun(
	    vaddr, size,
	    [&](uint64_t offset, uint64_t, uint64_t bytes) { pieces.emplace_back(base + offset, bytes); },
	    [&](uint64_t, uint64_t bytes) { pieces.emplace_back(nullptr, bytes); });
	return true;
}

bool IsBackingReadable(uint64_t vaddr, uint64_t size) {
	return Shim::IsBacked(vaddr, size);
}

uint64_t BackingEpoch() noexcept {
	return g_backing_epoch.load(std::memory_order_acquire);
}

bool BackingChangesSince(uint64_t epoch, std::vector<std::pair<uint64_t, uint64_t>>& changes) {
	std::lock_guard lock(g_backing_changes_mutex);
	const auto current = g_backing_epoch.load(std::memory_order_relaxed);
	if (current - epoch > BackingChangeHistory) return false;
	for (auto change = epoch + 1; change <= current; ++change) {
		changes.push_back(g_backing_changes[change % BackingChangeHistory]);
	}
	return true;
}

bool IsFullyMapped(uint64_t vaddr, uint64_t size) {
	if (!ValidRange(vaddr, size)) return size == 0;
	return RegMemo(RegCeil::FnIsFullyMapped, vaddr, size,
	               [&] { return uint64_t {Shim::IsBacked(vaddr, size) || MappedPrefix(vaddr, size) == size}; }) != 0;
}

uint64_t ClampRangeSize(uint64_t vaddr, uint64_t size) {
	if (!ValidRange(vaddr, size)) {
		Shim::Fail("KytyShim memory: attempted to access invalid address 0x%016" PRIx64 " with size 0x%016" PRIx64 "\n", vaddr, size);
	}
	const auto clamped = RegMemo(RegCeil::FnClampRange, vaddr, size, [&] { return IsFullyMapped(vaddr, size) ? size : MappedPrefix(vaddr, size); });
	if (clamped == size) return size;
	if (clamped == 0) {
		Shim::Fail("KytyShim memory: attempted to access invalid address 0x%016" PRIx64 " with size 0x%016" PRIx64 "\n", vaddr, size);
	}
	if (FirstClamp(vaddr, size, clamped)) std::fprintf(stderr, "KytyShim memory: clamped buffer range addr=0x%016" PRIx64 " size=0x%016" PRIx64 " to 0x%016" PRIx64 "\n", vaddr, size, clamped);
	return clamped;
}

template <bool CopyPrivate>
static bool CopySparse(uint64_t vaddr, void* data, uint64_t size) {
	if (TryReadBacking(vaddr, data, size)) return true;
	if (data == nullptr || !Shim::InGuestSpace(vaddr, size)) return false;
	if constexpr (!CopyPrivate) {
		if (Shim::OverlapsPrivate(vaddr, size)) return false;
	}
	auto* destination = static_cast<uint8_t*>(data);
	ForEachRun(
	    vaddr, size,
	    [&](uint64_t offset, uint64_t start, uint64_t bytes) { CopyFromView(destination + (start - vaddr), offset, bytes); },
	    [&](uint64_t start, uint64_t bytes) {
		    if constexpr (CopyPrivate) {
			    CopyGuestPages(start, destination + (start - vaddr), bytes);
		    } else {
			    std::memset(destination + (start - vaddr), 0, bytes);
		    }
	    });
	return true;
}

bool TryReadSparseBacking(uint64_t vaddr, void* data, uint64_t size) {
	return CopySparse<false>(vaddr, data, size);
}

bool CopySparseMemory(uint64_t vaddr, void* data, uint64_t size) {
	return CopySparse<true>(vaddr, data, size);
}

bool ProtectGuestHostMemory(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode) {
	if (!ValidRange(vaddr, size)) return false;
	const auto protection = HostProtection(mode);
	const bool writable = (protection & PROT_WRITE) != 0;
#ifdef _WIN32
	const DWORD host_protection = protection == PROT_NONE                               ? PAGE_NOACCESS
	                              : (protection & PROT_EXEC) != 0 && writable             ? PAGE_EXECUTE_READWRITE
	                              : (protection & PROT_EXEC) != 0 && (protection & PROT_READ) != 0 ? PAGE_EXECUTE_READ
	                              : (protection & PROT_EXEC) != 0                         ? PAGE_EXECUTE
	                              : writable                                              ? PAGE_READWRITE
	                                                                                      : PAGE_READONLY;
	if (!writable) GuestArena::GuestArenaNoteHostProtection_nid_no_patch(vaddr, size, false);
	const auto& mapped_pages = Pages();
	const auto host_first = vaddr & ~uint64_t {0xfff};
	const auto host_end = (vaddr + size + 0xfff) & ~uint64_t {0xfff};
	for (auto start = host_first; start < host_end;) {
		auto stop = std::min(host_end, (start | PageMask) + 1u);
		if (mapped_pages.Load(start >> PageBits) != 0) {
			DWORD previous = 0;
			if (VirtualProtect(reinterpret_cast<void*>(static_cast<std::uintptr_t>(start)), stop - start, host_protection, &previous) == 0) {
				for (auto cursor = start; cursor < stop;) {
					MEMORY_BASIC_INFORMATION memory {};
					if (VirtualQuery(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(cursor)), &memory, sizeof(memory)) != sizeof(memory)) return false;
					const auto region_end = std::min<uint64_t>(stop, reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize);
					if (memory.State == MEM_COMMIT && VirtualProtect(reinterpret_cast<void*>(static_cast<std::uintptr_t>(cursor)), region_end - cursor, host_protection, &previous) == 0) return false;
					cursor = region_end;
				}
			}
		}
		start = stop;
	}
	if (writable) GuestArena::GuestArenaNoteHostProtection_nid_no_patch(vaddr, size, true);
	return true;
#else
	if (!writable) GuestArena::GuestArenaNoteHostProtection_nid_no_patch(vaddr, size, false);
	const auto first = vaddr & ~uint64_t {0xfff};
	const auto end = (vaddr + size + 0xfff) & ~uint64_t {0xfff};
	std::vector<KeptRange> kept;
#if defined(__linux__)
	GuestThreadStackMappings_nid_no_patch(first, end, &KeepThreadStack, &kept);
#endif
	std::unique_lock<std::mutex> holds;
	if (!writable) {
		holds = std::unique_lock(g_host_write_holds_mutex);
		for (auto& hold : g_host_write_holds) {
			if (hold.pages.first >= end || hold.pages.end <= first) continue;
			hold.overridden = true;
			kept.push_back({hold.pages, PageKeep::Writable});
		}
	}
	const auto& pages = Pages();
	for (auto start = first; start < end;) {
		const bool mapped = pages.Load(start >> PageBits) != 0;
		auto stop = std::min(end, (start | PageMask) + 1u);
		while (stop < end && (pages.Load(stop >> PageBits) != 0) == mapped) {
			stop = std::min(end, stop + PageSize);
		}
		if (mapped && !ProtectHostPages(start, stop, protection, kept)) return false;
		start = stop;
	}
	if (writable) GuestArena::GuestArenaNoteHostProtection_nid_no_patch(vaddr, size, true);
	return true;
#endif
}

bool IsGuestExecutable(uint64_t vaddr) {
	return Pages().Executable(vaddr);
}

uint64_t WaitForTransientUnmap(uint64_t) noexcept {
	return 0;
}

bool RetryTransientUnmapFault(uint64_t, uint64_t) noexcept {
	return false;
}

uint64_t AllocateRuntimeMemory(uint64_t search_addr, uint64_t size, Common::VirtualMemory::Mode mode, const char* name, bool fixed) {
	constexpr int GuestMapFixed = 0x10;
	if (size == 0 || size > UINT64_MAX - PageMask || name == nullptr || (fixed && (search_addr == 0 || (search_addr & PageMask) != 0))) return 0;
	void* address = reinterpret_cast<void*>(static_cast<std::uintptr_t>(search_addr));
	try {
		if (DoMapAnon_nid_no_patch(&address, (size + PageMask) & ~PageMask, GuestProtectionBits(mode), fixed ? GuestMapFixed : 0, PageSize, fixed) != 0) return 0;
	} catch (const std::exception& error) {
		std::fprintf(stderr, "KytyShim memory: runtime memory \"%s\" of 0x%" PRIx64 " bytes not mapped: %s\n", name, size, error.what());
		return 0;
	}
	return reinterpret_cast<std::uintptr_t>(address);
}

bool FreeGuestMemory(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0 || size == 0 || size > UINT64_MAX - PageMask) return false;
	try {
		return DoMunmap_nid_no_patch(reinterpret_cast<void*>(static_cast<std::uintptr_t>(vaddr)), (size + PageMask) & ~PageMask) == 0;
	} catch (const std::exception& error) {
		Shim::Fail("KytyShim memory: freeing runtime memory 0x%016" PRIx64 "+0x%" PRIx64 " failed: %s\n", vaddr, size, error.what());
	}
}


namespace RegCeil {

namespace {

struct Period {
	bool     started     = false;
	bool     on          = false;
	int      execute_tid = 0;
	int      record_tid  = 0;
	uint64_t ops = 0, next_check = 0, draws = 0, disp = 0;
	uint64_t wall = 0, cpu = 0, exec = 0, record = 0, tsc = 0, frames = 0, period_draws = 0, period_disp = 0;
	std::vector<std::array<uint64_t, 4 * FnCount>> last;
};

thread_local Period t_period;

const char* const kFnNames[FnCount] = {"backed",   "prefix", "fullmap",  "clamp",    "stillvalid", "mayfault", "cpumod",
                                       "findback", "rearm",  "wunwatch", "rwatch",   "runwatch",   "otherprot"};
const char* const kScheduleNames[4] = {"off", "stats", "on", "ab"};

std::mutex g_restore_mutex;
void (*g_restore)(void*) = nullptr;
void* g_restore_context  = nullptr;

const char* LeverName(int lever, const char* middle) {
	return lever == LeverSkip ? "skip" : lever == LeverExact ? "0" : middle;
}

uint64_t MonoNs() {
	timespec ts {};
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
}

uint64_t ThreadNs() {
	timespec ts {};
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
	return uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
}

std::string ThreadName(int tid) {
	char path[96];
	std::snprintf(path, sizeof(path), "/proc/self/task/%d/comm", tid);
	FILE* file = std::fopen(path, "r");
	if (file == nullptr) return "?";
	char comm[64] = {};
	if (std::fgets(comm, sizeof(comm), file) == nullptr) comm[0] = 0;
	std::fclose(file);
	comm[std::strcspn(comm, "\n")] = 0;
	return comm;
}

int FindTid(const char* name) {
#ifndef _WIN32
	DIR* dir = ::opendir("/proc/self/task");
	if (dir == nullptr) return 0;
	int found = 0;
	while (const dirent* entry = ::readdir(dir)) {
		const int tid = std::atoi(entry->d_name);
		if (tid > 0 && ThreadName(tid) == name) {
			found = tid;
			break;
		}
	}
	::closedir(dir);
	return found;
#else
	(void)name;
	return 0;
#endif
}

uint64_t TidCpuNs(int tid) {
	if (tid == 0) return 0;
	char path[96];
	std::snprintf(path, sizeof(path), "/proc/self/task/%d/schedstat", tid);
	FILE* file = std::fopen(path, "r");
	if (file == nullptr) return 0;
	unsigned long long ns = 0;
	if (std::fscanf(file, "%llu", &ns) != 1) ns = 0;
	std::fclose(file);
	return ns;
}

bool PhaseOn(uint64_t now) {
	if (now < g_reg_origin_ns + RegCfg().start_s * 1000000000ull) return false;
	switch (RegCeilMode()) {
		case 2: return true;
		case 3: return ((now / 10000000000ull) % 2) == 0;
		default: return false;
	}
}

void PrintCounters(Period& p, bool on, double end) {
	std::lock_guard lock(g_reg_threads_mutex);
	p.last.resize(g_reg_threads.size());
	for (size_t t = 0; t < g_reg_threads.size(); t++) {
		const auto* c = g_reg_threads[t];
		auto& last = p.last[t];
		std::array<uint64_t, 4 * FnCount> now {};
		for (int f = 0; f < FnCount; f++) {
			now[f] = c->calls[f];
			now[FnCount + f] = c->hits[f];
			now[2 * FnCount + f] = c->valid[f];
			now[3 * FnCount + f] = c->cycles[f];
		}
		bool any = false;
		for (size_t i = 0; i < now.size(); i++) any = any || now[i] != last[i];
		if (any) {
			char line[2048];
			int pos = std::snprintf(line, sizeof(line), "[reg] counts on=%d end=%.2f thread=%s", on ? 1 : 0, end, ThreadName(c->tid).c_str());
			for (int f = 0; f < FnCount && pos < static_cast<int>(sizeof(line)) - 160; f++) {
				pos += std::snprintf(line + pos, sizeof(line) - pos, " %s=%llu/%llu/%llu/%llu", kFnNames[f],
				                     static_cast<unsigned long long>(now[f] - last[f]),
				                     static_cast<unsigned long long>(now[FnCount + f] - last[FnCount + f]),
				                     static_cast<unsigned long long>(now[2 * FnCount + f] - last[2 * FnCount + f]),
				                     static_cast<unsigned long long>(now[3 * FnCount + f] - last[3 * FnCount + f]));
			}
			std::fprintf(stderr, "%s\n", line);
		}
		last = now;
	}
}

void SwitchPhase(bool on) {
	g_reg_generation.fetch_add(1, std::memory_order_relaxed);
	const bool was_on = g_reg_on.exchange(on, std::memory_order_relaxed);
	if (on || !was_on || !SkipConfigured()) return;
	std::lock_guard lock(g_restore_mutex);
	if (g_restore == nullptr) return;
	const auto start = MonoNs();
	g_restore(g_restore_context);
	std::fprintf(stderr, "[reg] rearm restored ms=%.2f\n", static_cast<double>(MonoNs() - start) / 1e6);
}

}

bool Enabled() noexcept {
	return RegCeilMode() != 0;
}

bool Timing() noexcept {
	return RegCeilMode() == 1;
}

bool On() noexcept {
	return RegCfg().query && g_reg_on.load(std::memory_order_relaxed);
}

int StillValidLever() noexcept {
	const auto& cfg = RegCfg();
	if (cfg.schedule == 0) return LeverExact;
	auto& c = RegC();
	c.calls[FnStillValid]++;
	if (cfg.stillvalid == LeverExact || !g_reg_on.load(std::memory_order_relaxed)) return LeverExact;
	if (cfg.stillvalid == LeverSkip) c.hits[FnStillValid]++;
	return cfg.stillvalid;
}

int RearmLever() noexcept {
	const auto& cfg = RegCfg();
	if (cfg.rearm == LeverExact || !g_reg_on.load(std::memory_order_relaxed)) return LeverExact;
	return cfg.rearm;
}

bool SkipConfigured() noexcept {
	return RegCfg().rearm == LeverSkip;
}

uint64_t Begin() noexcept {
	return __rdtsc();
}

void End(int fn, uint64_t start) noexcept {
	RegC().cycles[fn] += __rdtsc() - start;
}

void Count(int fn) noexcept {
	if (RegCeilMode() != 0) RegC().calls[fn]++;
}

void NoteProtect(int fn, uint64_t bytes, bool skipped, uint64_t cycles) noexcept {
	if (RegCeilMode() == 0) return;
	auto& c = RegC();
	c.calls[fn]++;
	if (skipped) c.hits[fn]++;
	c.valid[fn] += (bytes + 0xfffu) >> 12u;
	c.cycles[fn] += cycles;
}

void NoteFrame() noexcept {
	if (RegCeilMode() != 0) g_reg_frames.fetch_add(1, std::memory_order_relaxed);
}

void SetRearmRestore(void (*restore)(void*), void* context) noexcept {
	std::lock_guard lock(g_restore_mutex);
	if (restore == nullptr && g_restore_context != context) return;
	g_restore         = restore;
	g_restore_context = restore == nullptr ? nullptr : context;
}

bool MemoLookup(int fn, uint64_t vaddr, uint64_t size, uint64_t& value) noexcept {
	if (RegCeilMode() < 2 || !g_reg_on.load(std::memory_order_relaxed)) return false;
	const auto& entry = RegMemoSlot(fn, vaddr, size);
	if (entry.kind == fn + 1 && entry.vaddr == vaddr && entry.size == size &&
	    entry.generation == g_reg_generation.load(std::memory_order_relaxed)) {
		RegC().hits[fn]++;
		value = entry.result;
		return true;
	}
	return false;
}

void MemoStore(int fn, uint64_t vaddr, uint64_t size, uint64_t value) noexcept {
	if (RegCeilMode() < 2 || !g_reg_on.load(std::memory_order_relaxed)) return;
	RegMemoSlot(fn, vaddr, size) = {vaddr, size, value, 0, 0, g_reg_generation.load(std::memory_order_relaxed), fn + 1};
}

void EndOp(int kind) noexcept {
	if (RegCeilMode() == 0) return;
	auto& s = t_period;
	if (kind == 1) s.draws++;
	if (kind == 2) s.disp++;
	if (kind == 0) return;
	s.ops++;
	if (s.ops < s.next_check) return;
	s.next_check = s.ops + 64;
	const uint64_t now = MonoNs();
	if (!s.started) {
		s.started       = true;
		g_reg_origin_ns = now;
		s.execute_tid   = FindTid("kytyGpuExecute");
		s.record_tid   = FindTid("kytyGpuRecord");
		s.wall         = now;
		s.cpu          = ThreadNs();
		s.exec         = TidCpuNs(s.execute_tid);
		s.record       = TidCpuNs(s.record_tid);
		s.tsc          = __rdtsc();
		s.frames       = g_reg_frames.load(std::memory_order_relaxed);
		s.period_draws = s.draws;
		s.period_disp  = s.disp;
		s.on           = PhaseOn(now);
		const auto& cfg = RegCfg();
		std::fprintf(stderr,
		             "[reg] start schedule=%s query=%d stillvalid=%s rearm=%s start_s=%llu execute_tid=%d record_tid=%d\n",
		             kScheduleNames[cfg.schedule], cfg.query ? 1 : 0, LeverName(cfg.stillvalid, "memo"),
		             LeverName(cfg.rearm, "protected"), static_cast<unsigned long long>(cfg.start_s), s.execute_tid,
		             s.record_tid);
		PrintCounters(s, s.on, 0.0);
		SwitchPhase(s.on);
		return;
	}
	const bool want_on = PhaseOn(now);
	if (want_on == s.on && now - s.wall < 10000000000ull) return;
	if (s.execute_tid == 0) s.execute_tid = FindTid("kytyGpuExecute");
	if (s.record_tid == 0) s.record_tid = FindTid("kytyGpuRecord");
	const uint64_t cpu    = ThreadNs();
	const uint64_t exec   = TidCpuNs(s.execute_tid);
	const uint64_t record = TidCpuNs(s.record_tid);
	const uint64_t tsc    = __rdtsc();
	const uint64_t frames = g_reg_frames.load(std::memory_order_relaxed);
	const uint64_t draws  = s.draws - s.period_draws;
	const uint64_t disp   = s.disp - s.period_disp;
	const uint64_t wall   = now - s.wall;
	timespec real {};
	clock_gettime(CLOCK_REALTIME, &real);
	const double end = real.tv_sec + real.tv_nsec / 1e9;
	if (draws > 0 && wall > 0) {
		const double seconds   = wall / 1e9;
		const double exec_ns   = s.exec != 0 ? static_cast<double>(exec - s.exec) : 0.0;
		const double record_ns = s.record != 0 ? static_cast<double>(record - s.record) : 0.0;
		const double resolve_ns = static_cast<double>(cpu - s.cpu);
		const double frame_count = static_cast<double>(frames - s.frames);
		std::fprintf(stderr,
		             "[reg] period on=%d end=%.2f wall_s=%.2f fps=%.2f frames=%llu draws=%llu disp=%llu dps=%.0f "
		             "draws_per_frame=%.0f resolve_us_per_draw=%.3f exec_us_per_draw=%.3f record_us_per_draw=%.3f "
		             "resolve_busy=%.3f exec_busy=%.3f record_busy=%.3f tsc_ghz=%.4f\n",
		             s.on ? 1 : 0, end, seconds, frame_count / seconds, static_cast<unsigned long long>(frames - s.frames),
		             static_cast<unsigned long long>(draws), static_cast<unsigned long long>(disp), draws / seconds,
		             frame_count > 0 ? draws / frame_count : 0.0, resolve_ns / 1e3 / draws, exec_ns / 1e3 / draws,
		             record_ns / 1e3 / draws, resolve_ns / wall, exec_ns / wall, record_ns / wall,
		             static_cast<double>(tsc - s.tsc) / wall);
	}
	PrintCounters(s, s.on, end);
	s.wall         = now;
	s.cpu          = cpu;
	s.exec         = exec;
	s.record       = record;
	s.tsc          = tsc;
	s.frames       = frames;
	s.period_draws = s.draws;
	s.period_disp  = s.disp;
	s.on           = want_on;
	SwitchPhase(want_on);
}

}

}
