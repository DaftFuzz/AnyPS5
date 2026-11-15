#include "graphics/host_gpu/faultPath.h"

#include "common/liveAb.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionManager.h"
#include "graphics/host_gpu/renderer/perfCounters.h"
#include "kernel/memory.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <xxhash.h>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace Libs::Graphics::FaultPath {
namespace {

uint32_t ReadSwitch(const char* name) {
	const char* text = std::getenv(name);
	return text != nullptr ? static_cast<uint32_t>(std::strtoul(text, nullptr, 10)) : 0u;
}

uint64_t NowNs() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

void Pause() {
#if defined(_M_X64) || defined(__x86_64__)
	_mm_pause();
#endif
}

constexpr uint64_t RegionCount = TRACKER_ADDRESS_SIZE / TRACKER_REGION_SIZE;

struct ArJob {
	uint64_t                     seq     = 0;
	MemoryTracker*               tracker = nullptr;
	std::vector<RegionManager*>  regions;
	std::vector<ArCheck>         checks;
};

struct ArStats {
	std::atomic<uint64_t> jobs {0}, regions {0}, updated {0}, checks {0}, bytes {0}, staged_bytes {0}, changed {0};
	std::atomic<uint64_t> unreadable {0}, worker_ns {0}, lag_ns {0};
	std::atomic<uint64_t> joins[ArSiteCount] {}, waited[ArSiteCount] {}, wait_ns[ArSiteCount] {};
	std::atomic<uint64_t> exposed {0}, late {0}, changed_total {0};
	std::atomic<int64_t>  window_start {0};
};
ArStats g_ar_stats;

class ArWorker {
public:
	ArWorker() {
		m_region_seq     = std::make_unique<std::atomic<uint64_t>[]>(RegionCount);
		m_region_changed = std::make_unique<std::atomic<uint64_t>[]>(RegionCount);
		m_thread         = std::thread([this] { Run(); });
		m_thread.detach();
	}

	void Post(ArJob&& job) {
		const uint64_t seq = job.seq;
		for (auto* region: job.regions) NoteRegion(region->GetCpuAddr() / TRACKER_REGION_SIZE, seq);
		for (const auto& check: job.checks) {
			for (uint64_t r = check.address / TRACKER_REGION_SIZE; r <= (check.address + check.size - 1u) / TRACKER_REGION_SIZE; r++) {
				NoteRegion(r, seq);
			}
		}
		{
			std::scoped_lock lock(m_mutex);
			m_jobs.push_back(std::move(job));
		}
		g_ar_posted.store(seq, std::memory_order_release);
		m_cv.notify_one();
	}

	[[nodiscard]] uint64_t RegionSeq(uint64_t region) const { return m_region_seq[region].load(std::memory_order_relaxed); }
	[[nodiscard]] uint64_t RegionChanged(uint64_t region) const {
		return m_region_changed[region].load(std::memory_order_acquire);
	}

	void Wait(uint64_t seq, uint32_t site) {
		auto& s = g_ar_stats;
		s.joins[site].fetch_add(1, std::memory_order_relaxed);
		if (g_ar_done.load(std::memory_order_acquire) >= seq) return;
		const uint64_t t0 = NowNs();
		for (int spin = 0; spin < 4000 && g_ar_done.load(std::memory_order_acquire) < seq; spin++) Pause();
		if (g_ar_done.load(std::memory_order_acquire) < seq) {
			m_waiters.fetch_add(1, std::memory_order_seq_cst);
			for (uint64_t done; (done = g_ar_done.load(std::memory_order_seq_cst)) < seq;) {
				g_ar_done.wait(done, std::memory_order_seq_cst);
			}
			m_waiters.fetch_sub(1, std::memory_order_relaxed);
		}
		s.waited[site].fetch_add(1, std::memory_order_relaxed);
		s.wait_ns[site].fetch_add(NowNs() - t0, std::memory_order_relaxed);
	}

private:
	void NoteRegion(uint64_t region, uint64_t seq) {
		if (region < RegionCount) m_region_seq[region].store(seq, std::memory_order_relaxed);
	}

	void Run() {
		t_ar_worker = true;
		std::vector<uint8_t> scratch;
		for (;;) {
			ArJob job;
			{
				std::unique_lock lock(m_mutex);
				m_cv.wait(lock, [&] { return !m_jobs.empty(); });
				job = std::move(m_jobs.front());
				m_jobs.pop_front();
			}
			const uint64_t t0 = NowNs();
			auto&          s  = g_ar_stats;
			s.regions.fetch_add(job.regions.size(), std::memory_order_relaxed);
			s.updated.fetch_add(job.tracker->CommitDeferredProtections(job.regions), std::memory_order_relaxed);
			uint64_t bytes = 0;
			for (const auto& check: job.checks) {
				bytes += check.size;
				scratch.resize(check.size);
				const bool read = LibKernel::Memory::TryReadBacking(check.address, scratch.data(), check.size) ||
				                  LibKernel::Memory::GuardedGuestCopy(scratch.data(), reinterpret_cast<const void*>(check.address),
				                                                      check.size);
				if (!read) s.unreadable.fetch_add(1, std::memory_order_relaxed);
				if (read && XXH3_64bits(scratch.data(), check.size) == check.hash) continue;
				job.tracker->MarkRegionAsCpuModified(check.address, check.size);
				s.changed.fetch_add(1, std::memory_order_relaxed);
				s.changed_total.fetch_add(1, std::memory_order_release);
				for (uint64_t r = check.address / TRACKER_REGION_SIZE; r <= (check.address + check.size - 1u) / TRACKER_REGION_SIZE; r++) {
					if (r < RegionCount) m_region_changed[r].store(job.seq, std::memory_order_release);
				}
			}
			s.checks.fetch_add(job.checks.size(), std::memory_order_relaxed);
			s.bytes.fetch_add(bytes, std::memory_order_relaxed);
			const uint64_t t1 = NowNs();
			s.worker_ns.fetch_add(t1 - t0, std::memory_order_relaxed);
			g_ar_done.store(job.seq, std::memory_order_seq_cst);
			if (m_waiters.load(std::memory_order_seq_cst) != 0) g_ar_done.notify_all();
		}
	}

	std::unique_ptr<std::atomic<uint64_t>[]> m_region_seq;
	std::unique_ptr<std::atomic<uint64_t>[]> m_region_changed;
	std::mutex                               m_mutex;
	std::condition_variable                  m_cv;
	std::deque<ArJob>                        m_jobs;
	std::atomic<uint32_t>                    m_waiters {0};
	std::thread                              m_thread;
};

ArWorker* g_ar_worker_ptr = nullptr;

ArWorker& Worker() {
	static ArWorker* const worker = [] {
		auto* w         = new ArWorker();
		g_ar_worker_ptr = w;
		return w;
	}();
	return *worker;
}

} // namespace

uint32_t AsyncReprotectFixed() {
	static const uint32_t value = [] {
		const uint32_t v = ReadSwitch("KYTY_PERF_ASYNC_REPROTECT");
		if (v != 0) {
			std::fprintf(stderr, "KYTY_PERF_ASYNC_REPROTECT=%u: upload batch protections and re-checks on a worker%s\n", v,
			             v == 2 ? " (check: memo, peek and BDA observers of pending regions wait and count late changes)" : "");
		}
		return v;
	}();
	return value;
}

uint32_t AsyncReprotectMode() {
	const uint32_t fixed = AsyncReprotectFixed();
	const uint32_t live  = Common::LiveAb::Gate(Common::LiveAb::Switch::AsyncReprotect, fixed == 2u ? 1u : fixed);
	return live != 0u && fixed == 2u ? 2u : live;
}

bool AsyncReprotectPossible() {
	static const bool possible = Common::LiveAb::Possible(Common::LiveAb::Switch::AsyncReprotect, AsyncReprotectFixed());
	return possible;
}

void ArJoinRangeSlow(uint64_t vaddr, uint64_t size, uint32_t site) {
	if (!PerfIsGpuThread() || g_ar_worker_ptr == nullptr) return;
	auto&          w    = *g_ar_worker_ptr;
	const uint64_t last = std::min((vaddr + size - 1u) / TRACKER_REGION_SIZE, RegionCount - 1u);
	uint64_t       need = 0;
	for (uint64_t r = vaddr / TRACKER_REGION_SIZE; r <= last; r++) need = std::max(need, w.RegionSeq(r));
	if (need > g_ar_done.load(std::memory_order_acquire)) w.Wait(need, site);
}

void ArJoinAll(uint32_t site) {
	if (g_ar_worker_ptr == nullptr || !ArPending()) return;
	g_ar_worker_ptr->Wait(g_ar_posted.load(std::memory_order_acquire), site);
}

void ArObserveRegionsSlow(uint64_t first_region, uint32_t count, uint32_t site) {
	if (!PerfIsGpuThread() || g_ar_worker_ptr == nullptr) return;
	auto&          w    = *g_ar_worker_ptr;
	const uint64_t done = g_ar_done.load(std::memory_order_acquire);
	uint64_t       need = 0;
	for (uint32_t i = 0; i < count && first_region + i < RegionCount; i++) need = std::max(need, w.RegionSeq(first_region + i));
	if (need <= done) return;
	g_ar_stats.exposed.fetch_add(1, std::memory_order_relaxed);
	if (AsyncReprotectFixed() != 2u) return;
	w.Wait(need, site);
	for (uint32_t i = 0; i < count && first_region + i < RegionCount; i++) {
		const uint64_t changed = w.RegionChanged(first_region + i);
		if (changed > done && changed <= need) {
			g_ar_stats.late.fetch_add(1, std::memory_order_relaxed);
			break;
		}
	}
}

void ArObserveAllSlow(uint32_t site) {
	if (!PerfIsGpuThread() || g_ar_worker_ptr == nullptr) return;
	g_ar_stats.exposed.fetch_add(1, std::memory_order_relaxed);
	if (AsyncReprotectFixed() != 2u) return;
	const uint64_t done   = g_ar_done.load(std::memory_order_acquire);
	const uint64_t need   = g_ar_posted.load(std::memory_order_acquire);
	const uint64_t before = g_ar_stats.changed_total.load(std::memory_order_acquire);
	g_ar_worker_ptr->Wait(need, site);
	if (need > done && g_ar_stats.changed_total.load(std::memory_order_acquire) != before) g_ar_stats.late.fetch_add(1, std::memory_order_relaxed);
}

void ArPost(MemoryTracker& tracker, std::vector<RegionManager*>&& regions, std::vector<ArCheck>&& checks, uint64_t staged_bytes) {
	auto& w = Worker();
	ArJob job;
	job.seq     = g_ar_posted.load(std::memory_order_relaxed) + 1u;
	job.tracker = &tracker;
	job.regions = std::move(regions);
	job.checks  = std::move(checks);
	g_ar_stats.jobs.fetch_add(1, std::memory_order_relaxed);
	g_ar_stats.staged_bytes.fetch_add(staged_bytes, std::memory_order_relaxed);
	w.Post(std::move(job));
}

void ArReport() {
	if (!PerfEnabled()) return;
	auto&         s     = g_ar_stats;
	const int64_t now   = static_cast<int64_t>(NowNs() / 1000000u);
	int64_t       start = s.window_start.load(std::memory_order_relaxed);
	if (start == 0) {
		s.window_start.compare_exchange_strong(start, now, std::memory_order_relaxed);
		return;
	}
	if (now - start < 5000 || !s.window_start.compare_exchange_strong(start, now, std::memory_order_relaxed)) return;
	const auto take = [](std::atomic<uint64_t>& c) { return static_cast<unsigned long long>(c.exchange(0, std::memory_order_relaxed)); };
	const auto ms   = [](unsigned long long ns) { return static_cast<double>(ns) / 1e6; };
	const auto jobs = take(s.jobs), regions = take(s.regions), updated = take(s.updated), checks = take(s.checks);
	const auto kb = take(s.bytes) / 1024u, staged = take(s.staged_bytes) / 1024u, changed = take(s.changed), unreadable = take(s.unreadable);
	const auto worker = take(s.worker_ns);
	unsigned long long j[ArSiteCount], wt[ArSiteCount], wn[ArSiteCount];
	for (uint32_t i = 0; i < ArSiteCount; i++) {
		j[i]  = take(s.joins[i]);
		wt[i] = take(s.waited[i]);
		wn[i] = take(s.wait_ns[i]);
	}
	std::fprintf(stderr,
	             "PERF async_reprotect mode=%u jobs=%llu regions=%llu region_updates=%llu checks=%llu kb=%llu staged_kb=%llu "
	             "changed=%llu unreadable=%llu worker_ms=%.2f joins(range,order,observe,bda)=%llu/%llu/%llu/%llu "
	             "waited=%llu/%llu/%llu/%llu wait_ms=%.2f/%.2f/%.2f/%.2f exposed=%llu late=%llu (per ~5 s)\n",
	             AsyncReprotectMode(), jobs, regions, updated, checks, kb, staged, changed, unreadable, ms(worker), j[0], j[1], j[2],
	             j[3], wt[0], wt[1], wt[2], wt[3], ms(wn[0]), ms(wn[1]), ms(wn[2]), ms(wn[3]), take(s.exposed), take(s.late));
}

uint32_t DeferReadProtectFixed() {
	static const uint32_t value = [] {
		const uint32_t v = ReadSwitch("KYTY_PERF_DEFER_READ_PROTECT");
		if (v != 0) {
			std::fprintf(stderr, "KYTY_PERF_DEFER_READ_PROTECT=%u: no-access protections of GPU-written pages %s\n", v,
			             v == 2 ? "applied at once, faults in their would-be windows counted (check)"
			                    : "deferred and merged until the next observation point");
		}
		return v;
	}();
	return value;
}

bool DeferReadProtectPossible() {
	static const bool possible = Common::LiveAb::Possible(Common::LiveAb::Switch::DeferReadProtect, DeferReadProtectFixed());
	return possible;
}

void BeforeOperation(bool plain) {
	if (!DeferReadProtectPossible()) return;
	if (!plain) {
		PageManager::FlushDeferredReadProtect();
		PageManager::SetReadProtectDeferral(0);
		return;
	}
	const uint32_t fixed = DeferReadProtectFixed();
	const uint32_t live  = Common::LiveAb::Gate(Common::LiveAb::Switch::DeferReadProtect, fixed == 2u ? 1u : fixed);
	PageManager::SetReadProtectDeferral(live != 0u && fixed == 2u ? 2u : live);
}

void BeforeObservation() {
	if (!DeferReadProtectPossible()) return;
	PageManager::FlushDeferredReadProtect();
	PageManager::SetReadProtectDeferral(0);
}

void DrpNoteFault(uint64_t vaddr, bool is_write) {
	if (DeferReadProtectFixed() != 2u) return;
	PageManager::NoteWindowFault(vaddr, is_write, PerfIsGpuThread());
}

} // namespace Libs::Graphics::FaultPath
