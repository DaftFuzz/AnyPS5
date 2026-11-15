#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_FAULTPATH_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_FAULTPATH_H_

#include "graphics/host_gpu/regionDefinitions.h"

#include <atomic>
#include <cstdint>
#include <vector>

// Fault-path levers (C:\a5\fable-notes\erlou-diff-2026-11-15.md #2 and #4), each behind its own switch, default off.
//
// KYTY_PERF_ASYNC_REPROTECT=1 (live A/B switch; =2 check, start value only): the tail of an upload batch commit
// (BufferCache::CommitUploadBatch: write-protect the claimed pages, then hash the staged ranges again and mark a range
// whose bytes changed CPU-dirty) runs on a worker thread. The GPU thread has already staged the bytes, so the order
// "clear the dirty bits, copy, protect, re-check" is kept for every page; only who runs the last two steps changes.
// While a job is pending its regions' CPU-dirty state may lack a guest write made between the copy and the protection.
// Nobody on the GPU thread acts on that state before the job finished:
//  - every MemoryTracker call of the GPU thread over a region a pending job covers waits for that job (ArJoinRange);
//  - at every point where the command stream reads guest memory that orders the guest's earlier writes (a submission
//    starts or resumes, a WAIT_REG_MEM, COND_EXEC, predication: CommandProcessor::PublishDirtyRegions, the BDA sync
//    epoch's points) the GPU thread waits for every job posted before it (ArJoinAll), before the deferred CPU-dirty
//    regions are published. A guest write the guest ordered before later GPU work is therefore marked before that work
//    is translated, as with the synchronous commit; only a write the guest did not order (a race with its own GPU work
//    on the console too) can be seen one ordering point later by the obtain/write memos and the BDA region pass.
// =2 (check): as 1, and every memo hit, peek proof and BDA region pass over a pending region waits for it and counts
// the cases where the job then found changed bytes there (late=, must be 0 for the run to show no difference).
//
// KYTY_PERF_DEFER_READ_PROTECT=1 (live A/B switch; =2 check, start value only): on the GPU thread, between two operations
// that are plain draws or direct dispatches, the no-access protection of pages the GPU work will write (GPU-modified
// marks, the image read watch) is not applied at once: the page runs are kept, adjacent runs merged, and applied at the
// next point the guest or the GPU thread can observe that the work ran or read those pages itself: before any other
// operation (labels, memory writes, waits, flips, indirect packets), before a guest command, before the GPU thread goes
// idle, and before a plain guest read of the GPU thread over a pending run (ReadGuestPlain). Applying re-reads the pages'
// current watchers under their region lock, so a page opened meanwhile is not closed again. Until then such a page is
// still write-protected (it was CPU-clean before the GPU write was recorded), so a guest write faults as before; a guest
// read sees the bytes from before the work, as on the console before the guest learns that the work ran.
// =2 (check): protections are applied at once, the runs that would have been deferred are kept until the same points,
// and every fault in them is counted (window_faults=: reads that deferral would have let through with older bytes).

namespace Libs::Graphics {
class MemoryTracker;
class RegionManager;
} // namespace Libs::Graphics

namespace Libs::Graphics::FaultPath {

enum ArSite : uint32_t { ArSiteRange = 0, ArSiteOrder = 1, ArSiteObserve = 2, ArSiteBda = 3, ArSiteCount = 4 };

[[nodiscard]] uint32_t AsyncReprotectFixed();
[[nodiscard]] uint32_t AsyncReprotectMode();
[[nodiscard]] bool     AsyncReprotectPossible();

inline std::atomic<uint64_t>  g_ar_posted {0};
inline std::atomic<uint64_t>  g_ar_done {0};
inline thread_local bool      t_ar_worker = false;

void ArJoinRangeSlow(uint64_t vaddr, uint64_t size, uint32_t site);
void ArJoinAll(uint32_t site);
void ArObserveRegionsSlow(uint64_t first_region, uint32_t count, uint32_t site);
void ArObserveAllSlow(uint32_t site);

[[nodiscard]] inline bool ArPending() noexcept {
	return g_ar_posted.load(std::memory_order_relaxed) != g_ar_done.load(std::memory_order_acquire);
}

inline void ArJoinRange(uint64_t vaddr, uint64_t size, uint32_t site = ArSiteRange) {
	if (!ArPending() || t_ar_worker || size == 0) return;
	ArJoinRangeSlow(vaddr, size, site);
}

inline void ArObserveRegions(uint64_t first_region, uint32_t count) {
	if (!ArPending()) return;
	ArObserveRegionsSlow(first_region, count, ArSiteObserve);
}

inline void ArObserveAll() {
	if (!ArPending()) return;
	ArObserveAllSlow(ArSiteBda);
}

struct ArCheck {
	uint64_t address;
	uint64_t size;
	uint64_t hash;
};

void ArPost(MemoryTracker& tracker, std::vector<RegionManager*>&& regions, std::vector<ArCheck>&& checks, uint64_t staged_bytes);
void ArReport();

[[nodiscard]] uint32_t DeferReadProtectFixed();
[[nodiscard]] bool     DeferReadProtectPossible();

void BeforeOperation(bool plain);
void BeforeObservation();
void DrpNoteFault(uint64_t vaddr, bool is_write);

} // namespace Libs::Graphics::FaultPath

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_FAULTPATH_H_
