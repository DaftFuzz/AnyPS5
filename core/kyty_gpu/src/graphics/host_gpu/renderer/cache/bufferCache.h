#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/spscQueue.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/translatorGuard.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <thread>
#include <functional>
#include <map>
#include <optional>
#include <memory>
#include <span>
#include <utility>
#include <set>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <shared_mutex>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class TextureCache;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

namespace RingCensus {
struct Request; // KYTY_TRACE_RING_CENSUS (ringCensus.h)
} // namespace RingCensus

// The game frame whose data-ring half the game writes now (game threads, at each submission) and the one the GPU work
// being recorded writes (execution thread, in stream order); BufferCache::DropDeadRingBytes.
void RingGameWriteFrame(uint64_t frame);
void RingExecDataFrame(uint64_t frame);

// Counters of one SynchronizeBuffersInRange call (the BDA region pass's statistics).
struct BufferSyncStats {
	uint64_t buffers = 0;  // buffers visited
	uint64_t uploads = 0;  // buffers that issued a copy
	uint64_t bytes   = 0;  // bytes copied
};

struct PlanBuffer;

class BufferCache {
public:
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = (LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemorySize) >> CACHING_PAGEBITS;
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    CACHING_NUMPAGES * sizeof(vk::DeviceAddress);
	// The fault buffer: the read-fault page bitmap, the BDA store bitmap (KYTY_RECORD_KEYED_VERIFY), then a diagnostic
	// tail (KYTY_DIAG_RT_TRIPS, FaultDiagTripsEnabled). Indices are 32-bit words of the buffer.
	static constexpr uint64_t FAULT_BITMAP_BYTES = CACHING_NUMPAGES / 8;
	static constexpr uint32_t FAULT_DIAG_BASE    = static_cast<uint32_t>(FAULT_BITMAP_BYTES * 2 / 4);
	static constexpr uint32_t FAULT_DIAG_WORDS   = 32768;
	// Tail layout: [0] invocations that hit a loop watchdog cap, [1] watchdog reports taken, [2] BDA stores to a page
	// without a host buffer (dropped), [3] store-miss reports taken; watchdog reports of FAULT_DIAG_TRIP_WORDS words
	// from FAULT_DIAG_TRIP_FIRST, store-miss reports of 4 words from FAULT_DIAG_MISS_FIRST.
	static constexpr uint32_t FAULT_DIAG_TRIP_FIRST = 64;
	static constexpr uint32_t FAULT_DIAG_TRIP_WORDS = 128;
	static constexpr uint32_t FAULT_DIAG_TRIP_SLOTS = 128;
	static constexpr uint32_t FAULT_DIAG_TRIP_LAST  = 8; // reports per invocation: its last iterations before the cap
	static constexpr uint32_t FAULT_DIAG_TRIP_PHIS  = 56; // header phi values per lane half (words 16.., 72..)
	static constexpr uint32_t FAULT_DIAG_MISS_FIRST = FAULT_DIAG_TRIP_FIRST + FAULT_DIAG_TRIP_WORDS * FAULT_DIAG_TRIP_SLOTS;
	static constexpr uint32_t FAULT_DIAG_MISS_SLOTS = 1024;
	static_assert(FAULT_DIAG_MISS_FIRST + 4 * FAULT_DIAG_MISS_SLOTS <= FAULT_DIAG_WORDS);

	static constexpr uint64_t PageIndex(uint64_t address) {
		return (address < LOWER_ADDRESS_SIZE
		            ? address
		            : address - LibKernel::Memory::kExtendedMemoryBase + LOWER_ADDRESS_SIZE) >>
		       CACHING_PAGEBITS;
	}
	static constexpr uint64_t GuestAddress(uint64_t offset) {
		return offset < LOWER_ADDRESS_SIZE
		           ? offset
		           : offset - LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemoryBase;
	}

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	// `defer_direct` (the fault pass, a deferred operation; KYTY_PERF_DIRECT_RANGES): a direct range the request would
	// have to activate or grow is only marked wanted (ActivateWantedDirectRanges), and a null id is returned for it.
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size, bool defer_direct = false);
	// The resolve thread's lookups (stageFLookup.h). Resolve thread: the registry's lookup lock if it is free (never waits), and its
	// release. Between them LookupBuffer answers FindBuffer(vaddr, size) when FindBuffer would find a buffer (a null id
	// when it would create one), writing nothing; the buffer registry epoch it saw is DrawRepeat::g_buffer_registry_epoch,
	// read under the same lock. GPU thread: the registry's page table and slots change only under this lock.
	[[nodiscard]] bool     TryLockLookup() noexcept { return m_lookup_lock.try_lock(); }
	// Project 15 way B W2.1: a helper's plan peek tries the lookup lock 4 times, 64 pauses apart (as the resolve's source
	// build does): a single try lost every entry of the stage while the other helper held it (~8% of the acquisitions).
	[[nodiscard]] bool     TryLockLookupSpin() noexcept {
        for (uint32_t attempt = 0; attempt < 4u; attempt++) {
            if (m_lookup_lock.try_lock()) return true;
            for (uint32_t spin = 0; spin < 64u; spin++) Common::SpinPause();
        }
        return false;
	}
	void                   UnlockLookup() noexcept { m_lookup_lock.unlock(); }
	[[nodiscard]] BufferId LookupBuffer(uint64_t vaddr, uint64_t size) const;
	// Project 15 S2a (N3, the read-only translator; design: the analysis session's n3-projekt.md section 4.1): what
	// ObtainBuffer would answer, writing nothing. Ready: a registered buffer holds the range, it is CPU-clean and no image
	// is registered over it - ObtainBuffer has nothing to upload or sync and only touches (its writes' notes, for a
	// written binding, are the commit's); the proof is the obtain memo's (buffer registry epoch, the 4 MiB regions'
	// CPU-dirty epochs, the image page epochs). Direct: an active direct range holds it. Null: nothing to bind. The rest
	// are marks "in order" (the commit obtains that binding itself): Stream (a small CPU-dirty read the stream ring would
	// copy), Upload (CPU-dirty, or not proven clean), Create (no buffer holds the range), Texel (an image over it).
	struct BufferPeek {
		enum Class : uint8_t { Ready, Direct, Null, Stream, Upload, Create, Image, NoProof, Written, Count };
		Class    klass = Create;
		BufferId id {};
		uint64_t offset       = 0;
		uint64_t buffer_epoch = 0, image_sum = 0, region = 0;
		uint32_t epochs[2] {};
		uint32_t regions = 0;
		bool     bda     = false; // ObtainBdaRange's peek: no image in the proof
		bool     helper_direct = false; // way B W0: a helper's peek of a range overlapping a direct range (Upload, in order)
		uint64_t direct_gen    = 0;     // W2.2: a Direct answer's generation of the direct ranges (g_direct_generation)
		vk::Buffer handle {};           // P1: the buffer's handle (Ready, Direct)
	};
	// bda: ObtainBdaRange's peek (no direct ranges, no stream copy, no image copy).
	[[nodiscard]] BufferPeek PeekBuffer(uint64_t vaddr, uint64_t size, bool is_written, bool is_texel_buffer, BufferId hint,
	                                    bool bda = false);
	// KYTY_CHECK_PEEK=2: today's path after the peek, compared (bufferCache.cpp).
	template <typename Path>
	std::pair<Buffer*, uint64_t> PlanChecked(PlanBuffer& entry, uint64_t vaddr, uint64_t size, Path&& path);
	template <typename Path>
	std::pair<Buffer*, uint64_t> PeekChecked(const BufferPeek& peek, uint64_t vaddr, uint64_t size, bool is_written, bool bda, Path&& path);
	// Whether a Ready or Direct peek's proof still holds (the obtain memo's check); PeekProofFail names the failed part
	// (0 holds, 1 buffer registry, 2 CPU-dirty region epochs, 3 image pages, 4 not a proved class).
	[[nodiscard]] static uint32_t PeekProofFail(const BufferPeek& peek, uint64_t vaddr, uint64_t size);
	[[nodiscard]] static bool     PeekProofHolds(const BufferPeek& peek, uint64_t vaddr, uint64_t size) {
        return PeekProofFail(peek, vaddr, size) == 0;
	}
	// The translator guard (KYTY_CHECK_PEEK=2, translatorGuard.h): writes noted inside a peek (expected 0).
	static void TranslatorNote(int site) { TranslatorNoteSite(site); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer = false,
	                                                        BufferId id              = {},
	                                                        bool     image_source    = false);
	// ObtainBuffer's work (ObtainBuffer adds KYTY_CHECK_PEEK=2 around it).
	// N4.2b (KYTY_CHECK_PLAN_BUFFERS=2, diagnostic): the running draw's plan entries; every acquisition with an entry of
	// its key is checked against it (the entry's proof, today's path quiet and with the same buffer and offset).
	static void SetPlanBuffers(std::vector<PlanBuffer>* entries);
	[[nodiscard]] static bool PlanBufferCheck();
	// Project 15 way B P1: the running draw's plan entry of a stage's read-only buffer resource (the first unused one of that
	// program and resource from the shared cursor; null without a plan). The only path since batch 1 (10.11).
	[[nodiscard]] static PlanBuffer* PlanStageEntry(const void* program, uint32_t resource);
	// Batch 1: a direct-range word no listed GPU write may cover (WaitDirectGpuWrites would return at once with no overlay):
	// read plainly. False: today's wait and overlay.
	[[nodiscard]] bool DirectWordQuiet(uint64_t vaddr) const { return m_direct_gpu_writes.empty() || !DirectWritesMayOverlap(vaddr, 4); }
	// P1: the plan's answer for a binding (the same condition and work as ObtainBuffer's plan branch: the proof, Direct's
	// mode, the touch of a Ready buffer); true when it answered.
	[[nodiscard]] bool PlanAnswerBinding(PlanBuffer& entry, uint64_t vaddr, uint64_t size);
	// Project 15 N4.3/N5: a draw with a plan is committed from it (its pipeline, table sets and buffer entries: an
	// acquisition whose entry is Ready and whose proof holds only touches the buffer). Only path since 10.11 (checkpoint 2:
	// +2.8% fps, the execution thread -1.6 ms/frame); the checks that compare against today's way turn it off.
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferPath(uint64_t vaddr, uint64_t size, bool is_written, bool is_texel_buffer,
	                                                            BufferId id, bool image_source);
	// KYTY_CHECK_PEEK=2: ObtainBufferPath's work that a Ready peek promised away (uploads, stream copies, image syncs).
	inline static thread_local uint64_t t_obtain_work = 0;
	// Asks the CPU to fetch the memo entry a later ObtainBuffer of this binding reads (the write memo's for a written one on
	// the GPU thread, the obtain memo's for a read that is not texel), so their cache misses overlap instead of following
	// each other. Changes no data.
	static void PrefetchObtainMemo(uint64_t vaddr, uint64_t size, bool is_written, bool is_texel_buffer);
	// FindBuffer's and ObtainBuffer's first question, asked as they ask it (KYTY_PERF_DIRECT_RANGES: the direct range's
	// buffer holding the range, activating or growing it): a null id when none does.
	[[nodiscard]] BufferId DirectBufferAt(uint64_t vaddr, uint64_t size) { return DirectBufferFor(vaddr, size); }
	// Record-keyed V#s: [vaddr, vaddr + size) as a shader reaches it through buffer device addresses: held by
	// a cached buffer (found or created; never the stream shortcut of ObtainBuffer, whose copies the BDA
	// page table does not map), its CPU-modified bytes uploaded, and when written GPU-modified from now on, like a
	// written storage buffer.
	std::pair<Buffer*, uint64_t> ObtainBdaRange(uint64_t vaddr, uint64_t size, bool is_written, BufferId id = {});
	std::pair<Buffer*, uint64_t> ObtainBdaRangePath(uint64_t vaddr, uint64_t size, bool is_written, BufferId id);
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return Staging();
			case MemoryUsage::Stream: return Stream();
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	// KYTY_FIX_IMAGE_MEMORY_OWNER: the cached buffer that holds all of [vaddr, vaddr + size) and its offset there, or
	// none; nothing is created, synchronized or touched.
	[[nodiscard]] std::pair<Buffer*, uint64_t> CachedBufferFor(uint64_t vaddr, uint64_t size);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool IsRegionRegistered(uint64_t vaddr, uint64_t size);
	enum class DirtyReader : uint32_t { Texture, IndirectArgs, ImageHeap, GpuWord, CpWriteUnmark, Other, Count };
	[[nodiscard]] bool HasGpuDirtyBytes(uint64_t vaddr, uint64_t size, DirtyReader reader = DirtyReader::Other);
	void               ReportInFlightReadPerf();
	// HasGpuDirtyBytes for another thread (the resolve thread): the dirty-byte set changes only
	// through GpuDirtyAdd/GpuDirtySubtract, which take m_gpu_dirty_mutex exclusively; this always takes it shared. Bytes of
	// an unfinished readback copy (m_in_flight_ranges, changed under the same lock) count as dirty.
	[[nodiscard]] bool HasGpuDirtyBytesShared(uint64_t vaddr, uint64_t size);
	void               WaitCopyReadbacks(uint64_t vaddr, uint64_t size);
	// Publishes pending asynchronous (eager) readbacks overlapping the range: their bytes already left
	// the GPU-dirty set but reach guest memory only when the copy completes. True: the caller puts ApplyDirectCpWrites
	// over what it reads (WaitDirectGpuWrites with `cp_overlay`).
	bool WaitPendingReadbacks(uint64_t vaddr, uint64_t size);
	// True when an eager readback overlapping the range is pending (WaitPendingReadbacks would publish it):
	// guest memory there still holds the older bytes. Changes nothing; GPU thread only, like HasGpuDirtyBytes.
	[[nodiscard]] bool HasPendingReadback(uint64_t vaddr, uint64_t size) const;
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool PeekCpuClean(uint64_t vaddr, uint64_t size) { return m_memory_tracker.PeekCpuClean(vaddr, size); }
	// BAR reads: a draw or dispatch was recorded into the current command buffer; the
	// writable buffers obtained for it get that command buffer's tick as their write tick.
	void NoteCommandEmitted();
	// Upstream 4644c5292: zeroes the BDA page table and the fault buffer once, in the render scheduler's first command
	// buffer (CommandScheduler::Begin), before the first buffer registration writes into the table.
	void               ClearDeviceTablesOnce();
	void               ProcessFaultBuffer();
	// stats (optional) counts the visited buffers and the uploads.
	void               SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size, BufferSyncStats* stats = nullptr);
	// KYTY_UPSTREAM_GPU_INDIRECT_RECORDS: copies the cached GPU buffer contents of a range
	// (written by shaders, possibly through untracked BDA stores) after waiting for queued work.
	[[nodiscard]] bool ReadGpuCopy(uint64_t vaddr, uint64_t size, void* out);
	// The same copy (at most 256 bytes) recorded at this point of the command stream without
	// waiting; `done` gets the bytes on the priority-completion thread once the GPU finished the copy. False (nothing
	// recorded): no single cached buffer holds the range, or guest memory holds newer bytes (CPU-modified).
	[[nodiscard]] bool PeekGpuCopyAsync(uint64_t vaddr, uint64_t size, std::function<void(const uint8_t*)> done);

	// ---- D4b step 1: append windows (bufferCache.cpp) ----------------------------------------------------------------
	// An append pool (ShaderIR.h BufferResource::append_counter) is marked GPU-dirty whole, as every written binding, but
	// a dispatch writes only the records [counter before, counter after). The counter is copied (no CPU wait) before the
	// first tagged dispatch of a command buffer and when it is submitted; the pool bytes no window wrote, that were not
	// GPU-dirty when the pool's cycle began and that no other writer touched are released at the CPU's first request.
	struct AppendBinding {
		uint64_t pool = 0, pool_size = 0, counter = 0, threads = 0;
		uint32_t stride = 0, min_off = 0, max_end = 0;
	};
	// Only pools this large are tracked: whole-binding marks of small pools cost little, and the pool table and the
	// per-write checks stay short.
	static constexpr uint64_t AppendMinPoolBytes = 1024u * 1024u;
	// RebindBuffers of a dispatch with tagged pools: their counters (own marks, not foreign writes), cleared after.
	void AppendSetDispatchCounters(std::vector<uint64_t> counters) { m_append_self_counters = std::move(counters); }
	// NativeStorageBuffer, before ObtainBuffer marks a tagged pool: opens the pool's cycle (bytes GPU-dirty now are kept).
	void AppendBeforePoolMark(uint64_t vaddr, uint64_t size);
	void AppendSetTaggedObtain(bool on) { m_append_tagged_obtain = on; }
	// After RebindBuffers, before the dispatch is recorded: qualified bindings (empty: the tagged pools did not qualify).
	void AppendDispatch(const std::vector<AppendBinding>& bindings, const std::vector<std::pair<uint64_t, uint64_t>>& unqualified);
	void AppendCloseWindows(); // CommandScheduler::Submit (pre-submit hook)
	// The store report (D6 verification, KYTY_RECORD_KEYED_VERIFY): shaders set a bit per 16 KiB page they store to through
	// BDA, and the fault pass checks each reported page against the record-time listing.
	[[nodiscard]] static bool BdaStoreReportCompiled() { return RecordKeyedVerifyEnabled(); }
	// KYTY_RECORD_KEYED_VERIFY: a reported page no listing covers (direct ranges and unregistered memory aside); counted
	// (PERF d6_c0 would_mark) and logged (BDA_STORE_WOULD_MARK).
	void NoteBdaStoreWouldMark(uint64_t vaddr, uint64_t size);
	// KYTY_RECORD_KEYED_VERIFY=1 (D6 verification mode; compiles the store report by itself): every page the GPU reports
	// written (or targeted, when no buffer backs it) through a BDA pointer must overlap a written record-keyed range
	// listed in the frames of the work that fault-buffer pass covers (frames first..last, NoteRecordKeyedWrite); other
	// pages are counted and logged (RECORD_KEYED_VERIFY ... uncovered), passes spanning more frames than kept count as
	// unverified, and so do pages beyond the report's capacity.
	[[nodiscard]] static bool RecordKeyedVerifyEnabled();
	[[nodiscard]] uint64_t    RecordKeyedFrame() const noexcept { return m_rk_frame; }
	void                      NoteRecordKeyedWrite(uint64_t vaddr, uint64_t size);
	void                      NoteRecordKeyedIncomplete() noexcept {
		++m_rk_incomplete;
		m_rk_incomplete_frame = m_rk_frame + 1u; // + 1: 0 means "none yet"
	}
	// D6 step C0 (Stage C0), KYTY_D6_C0 (read once, logged):
	//   0 - the path without C0: every store report marks its pages GPU-modified;
	//   1 (default) - C0. The written record-keyed ranges own their bytes exactly (ObtainBdaRange
	//       makes them GPU-modified when the call is recorded); a store page the fault buffer reports for the work of a
	//       pass is skipped when a written record-keyed range of that work's frames lies on it and no call of those
	//       frames had an incomplete listing (the report repeats bytes the GPU already owns). Every other report is an
	//       unlisted store (the safety net): its pages that are not CPU-dirty become GPU-modified as on path 0; a
	//       CPU-dirty page has no exact answer (neither side's bytes are known) and stays CPU-owned, counted and logged
	//       as UNLISTED_BDA_STORE_CONFLICT (expected 0: a listing gap of step B);
	//   2 - C0 with the upload clobber check (verify mode, see UploadClobberCheck);
	//   3 - the path without C0 with the upload clobber check (the control of the check).
	[[nodiscard]] static uint32_t D6C0Mode();
	[[nodiscard]] static bool     D6C0Enabled() { return D6C0Mode() == 1u || D6C0Mode() == 2u; }
	// KYTY_D6_C0=2/3 (verify only, GPU thread + fault handlers): a guest page that a guest write fault makes CPU-dirty
	// from clean (no CPU-dirty, no GPU-modified mark: guest memory equals the GPU copy) keeps a copy of its bytes from
	// before that write. When an upload of the page is recorded, the GPU copy is copied out right before it (GPU order)
	// and compared once that work completed: a byte where the GPU copy differs from the copy (the GPU wrote it while the
	// tracker did not know) and the uploaded byte differs from the GPU copy is a byte the upload overwrote = a clobber.
	// PERF d6_c0 counts pages checked, pages and bytes clobbered, pages without a copy (unchecked).
	[[nodiscard]] static bool UploadClobberCheck() { return D6C0Mode() >= 2u; }
	void                      NoteGuestWriteFault(uint64_t vaddr);
	// KYTY_DIAG_GPU_DELAY_HASHES=<hex,...> KYTY_DIAG_GPU_DELAY_COPIES=<n> (measurement only, sensitivity test): after each
	// dispatch of a listed compute shader, n copies of 128 MiB (KYTY_DIAG_GPU_DELAY_KB=<size of one copy>) inside a scratch device buffer, ordered after the dispatch
	// and before everything recorded later - pure extra GPU time at that place; the game's data is not touched.
	void DiagGpuDelayAfter(vk::CommandBuffer cmd, uint64_t shader_hash);
	std::unique_ptr<Buffer> m_diag_delay_buffer;
	// D6 step C0: a written record-keyed range of the frames first..last lies on the 16 KiB store page and no call of
	// those frames (or later) had an incomplete listing; false when the frames are no longer kept.
	[[nodiscard]] bool BdaStoreListed(uint64_t page_address, uint64_t first_frame, uint64_t last_frame) const;
	void               NoteBdaStoreListedSkip() noexcept { ++m_c0_listed_skipped; }
	// D4 step K0 (KYTY_D6_C0=2/3, diagnostic): the last 8 GPU-side events of each 4 KiB page (any thread), printed with
	// each D6_C0_CLOBBER line; PERF d6_c0_writer counts clobbered pages by the newest event that changed the GPU copy
	// (a shader write binding, an image copy, a fill, a buffer copy, a direct-range write) before the check.
	enum class C0Event : uint8_t {
		Snapshot,    // the CPU's first write fault on a clean page (the clobber check's "before")
		Upload,      // CPU-dirty bytes copied into the GPU copy
		Mark,        // pages marked GPU-modified (tracker)
		CleanMark,   // clean pages marked GPU-modified (an unlisted BDA store report)
		Unmark,      // GPU-modified mark removed (tracker)
		Release,     // GPU-modified -> CPU-dirty in one step (tracker)
		ShaderWrite, // a written buffer binding (tag: the writer's shader hash)
		DirectWrite, // a written binding in a direct range (no tracker)
		ImageCopy,   // an image's contents copied into the cached buffer (tag: the image address)
		Fill,        // a GPU fill (tag: the value)
		Copy,        // a GPU buffer copy (destination; tag: the source address)
		RingDead,    // GPU-dirty bytes of the data ring dropped
		ListedSkip,  // a BDA store report skipped as listed (16 KiB page)
		BarRead,     // GPU bytes taken from the host mapping (tag: newest write tick waited for | finished GPU tick << 32)
		CopyRead,    // GPU bytes downloaded through a GPU copy (tag: the recording tick)
		BdaStore,    // a store through a pointer reported by the fault buffer (16 KiB page; tag: listed 1 / not 0)
		Join,        // an older buffer's contents copied into a new one (tag: the older buffer's first address)
		Create,      // a buffer created over the page (tag: its size)
		Count
	};
	void C0Note(uint64_t vaddr, uint64_t size, C0Event kind, uint64_t tag = 0);
	void                      NoteRecordKeyedUnverified(uint64_t pages) noexcept { m_rk_unverified += pages; }
	void                      VerifyBdaWrittenPage(uint64_t page_address, uint64_t first_frame, uint64_t last_frame);
	void               RunGarbageCollector();
	// Bytes of cached guest-memory buffers (KYTY_TRACE_PERF memory split).
	[[nodiscard]] uint64_t UsedMemory() const noexcept { return m_total_used_memory; }
	// Bytes of cached guest-memory buffers (KYTY_TRACE_PERF buffers=): unlike UsedMemory(), which the collector
	// replaces with the device usage of the whole process at every run, only these buffers count.
	[[nodiscard]] uint64_t RegisteredBytes() const noexcept { return m_registered_bytes; }
	// KYTY_TRACE_PERF / KYTY_TRACE_MEM_CENSUS (GPU thread): cached buffers by class ({count, bytes}), the fixed
	// buffers, host-side copies, and the collector's work since the previous census.
	struct Census {
		uint64_t count = 0, bytes = 0, gpu_dirty = 0, gpu_dirty_bytes = 0;
		uint64_t bar = 0, bar_bytes = 0, largest = 0;
		uint64_t bda_table_bytes = 0, staging_bytes = 0, stream_bytes = 0, download_bytes = 0, device_bytes = 0;
		uint64_t other_fixed_bytes = 0; // GDS, fault buffer, readback staging
		uint64_t gc_runs = 0, gc_retired = 0, gc_retired_bytes = 0, gc_kept_bda = 0, gc_dirty_skipped = 0;
		// KYTY_VRAM_BUDGET: buffers retired to make room for a new one over the target, and the bytes retired in
		// the last collector ticks (their deletion waits for the GPU).
		uint64_t alloc_retired = 0, alloc_retired_bytes = 0, recently_retired = 0;
	};
	void TakeCensus(Census& census);
	// KYTY_VRAM_BUDGET (GPU thread): bytes of buffers retired in command buffers the GPU has not retired yet
	// (deferred deletion): the measured usage still holds them.
	[[nodiscard]] uint64_t PendingReleaseBytes() { return RecentlyRetiredBytes(); }


	// Called (GPU thread) for every range a draw, dispatch or tracked BDA store writes; the texture
	// cache uses it to forget metadata keys it knows (KYTY_FIX_DCC_KNOWN_KEYS).
	void SetGpuWriteObserver(std::function<void(uint64_t, uint64_t)> observer) { m_gpu_write_observer = std::move(observer); }
	// The BDA fallback (M06): a mapped direct-memory page without a buffer gets a BDA table entry into the imported
	// direct-memory backing (bit 0 set = fallback), so a shader reads and writes the guest's real bytes instead of zeros /
	// a dropped store; the shader still reports the page (BDA_FAULT), and the fault pass gives it a device-local buffer as
	// before. The general allocator's pool is imported when mapped (the map-time import below); other 2 MiB granules of
	// the backing on the first read of one (which reads 0) or a direct download into it.
	// KYTY_PERF_BDA_ENTRY_SHARE (0 off, 1 on, 2 verify; spirvEmitterMemory.cpp LoadBdaDwordShared): a scalar load through
	// a GPU-built V# reuses the page-table entry of the invocation's previous such load when the page is the same.
	// Verify counts checks / mismatches in fault-buffer diag words FAULT_DIAG_ENTRY_SHARE (+0, +1).
	[[nodiscard]] static int BdaEntryShareMode();
	static constexpr uint32_t FAULT_DIAG_ENTRY_SHARE = 4;
	// (Words 8, 9 and 13 are free.)
	// KYTY_PERF_SRT_GPU_WORDS_VERIFY (SrtTape.cpp PlanSrtGpuWords, spirvEmitterMemory.cpp EmitReadGpuWord): shader-only SRT
	// words read from GPU memory compared with the CPU's value: checks / mismatches in diag words +0, +1.
	static constexpr uint32_t FAULT_DIAG_SRT_GPU_WORDS = 12;
	// +2: mismatch records taken; records of FAULT_DIAG_SRT_GPU_STRIDE words from FAULT_DIAG_SRT_GPU_FIRST (record
	// FAULT_DIAG_SRT_GPU_RECORDS is the spare one other lanes store into; all below FAULT_DIAG_TRIP_FIRST).
	static constexpr uint32_t FAULT_DIAG_SRT_GPU_FIRST   = 20;
	static constexpr uint32_t FAULT_DIAG_SRT_GPU_STRIDE  = 7;
	static constexpr uint32_t FAULT_DIAG_SRT_GPU_RECORDS = 5;
	static_assert(FAULT_DIAG_SRT_GPU_FIRST + (FAULT_DIAG_SRT_GPU_RECORDS + 1) * FAULT_DIAG_SRT_GPU_STRIDE <= FAULT_DIAG_TRIP_FIRST);
	// The guest addresses of the last GPU-word device addresses RebindBuffers bound (verify mode; any thread reads).
	static inline std::array<std::atomic<uint64_t>, (1u << 20u)> s_gpu_word_device {};
	static inline std::array<std::atomic<uint64_t>, (1u << 20u)> s_gpu_word_guest {};
	static inline std::atomic<uint32_t>                   s_gpu_word_next {0};
	static void NoteGpuWordBind(uint64_t device, uint64_t guest) {
		const uint32_t i = s_gpu_word_next.fetch_add(1, std::memory_order_relaxed) % s_gpu_word_device.size();
		s_gpu_word_device[i].store(device, std::memory_order_relaxed);
		s_gpu_word_guest[i].store(guest, std::memory_order_relaxed);
	}
	[[nodiscard]] static uint64_t GpuWordGuest(uint64_t device) {
		for (size_t i = 0; i < s_gpu_word_device.size(); ++i) {
			if (s_gpu_word_device[i].load(std::memory_order_relaxed) == device)
				return s_gpu_word_guest[i].load(std::memory_order_relaxed);
		}
		return 0;
	}
	void QueueFallbackMap(uint64_t vaddr, uint64_t size); // any thread (RenderContext::MapMemory)
	void ApplyFallbackMaps();                             // GPU thread (RenderContext::PrepareBda)
	void ClearFallback(uint64_t vaddr, uint64_t size);    // GPU thread (unmap, after InvalidateMemory)
	[[nodiscard]] bool IsFallbackPage(uint64_t address) const {
		return m_fallback_pages.contains(address & ~(CACHING_PAGESIZE - 1));
	}
	// D6 step K4: the page had a fallback entry until a buffer replaced it at most `frames` frames ago (a miss the fault
	// pass reports later was then served by the entry, not read as 0).
	[[nodiscard]] bool WasFallbackPage(uint64_t address, uint32_t frames) const;
	// The fault pass: `page` (backing memory) was read as 0 because its granule is not imported yet: the worker imports it
	// (MapTimeRequestGranule) and the install gives the granule's pages their fallback entries.
	void WantFallback(uint64_t page);
	// The BDA map-time import (AGENTS 19, notes\PROJEKT-BDA-2026-11-08.md): the game's general GPU allocator (fixed
	// 2 MiB maps at 0x80_..., prot 0xD3; 626 of 632 fallback reads of an entry into the game) is visible to the GPU from
	// the moment it is mapped, as on the console: the first map of a 256 MiB piece of its backing has a worker import the
	// piece whole (2 MiB view buffers = the fallback's granules); the GPU thread only installs them and writes their
	// entries. Pieces stay for the session (the game keeps the allocation and maps it again at loads). The check: the
	// pool's pages the fault pass saw read as 0 (PERF bda_maptime zero=, must stay 0).
	[[nodiscard]] static bool InBdaMapTimePool(uint64_t vaddr) {
		return vaddr >= BdaMapTimePoolStart && vaddr < BdaMapTimePoolStart + BdaMapTimePoolSize;
	}
	static constexpr uint64_t BdaMapTimePoolStart = 0x8000000000ull;
	static constexpr uint64_t BdaMapTimePoolSize  = uint64_t {16} << 30u; // the allocator holds ~3.9 GiB on a base PS5
	static constexpr uint64_t MapTimeImportBytes  = uint64_t {256} << 20u; // one import per piece of the pool's backing
	static constexpr size_t   MapTimeGranuleItem  = size_t {1} << 62u;     // a worker queue item: one granule, not a piece
	void NoteBdaMapTimeZero() { m_maptime.zero.fetch_add(1, std::memory_order_relaxed); }

	// KYTY_TRACE_PERF: the PERF fault_wait_on_caller line (the caller wait's counters).
	void ReportCallerWaitPerf();
	static void ReportBarCopyPerf(); // KYTY_TRACE_PERF: PERF bar_copy (BAR copy bytes/time, read window size)
	// The write-through (see the .cpp), GPU thread: a command-processor write (end-of-pipe label, WRITE_DATA,
	// reference clock) of [address, address + size) into a GPU-modified page, none of whose written bytes the GPU owns.
	// true: written to guest memory and to the GPU copy at this point of the command stream, the page left GPU-owned;
	// false: the caller writes it as before (the write faults and synchronizes the page).
	[[nodiscard]] bool TryCpWriteThrough(uint64_t address, const void* data, uint64_t size);
	// KYTY_PERF_DIRECT_RANGES (A1 stage 2), GPU thread. A direct range's guest memory is the
	// memory of one cache buffer: the host GPU reads and writes the guest's bytes in place, no page of it is tracked.
	//   - InDirectRange: [vaddr, vaddr + size) overlaps an active direct range;
	//   - WaitDirectGpuWrites: before the GPU thread reads guest memory itself (indirect arguments, waits, predicates,
	//     resource tables, DCC keys): waits for GPU writes recorded earlier into those bytes;
	//   - UnmapDirect: an unmap of guest memory (after Finish) ends the direct ranges it overlaps;
	//   - ActivateWantedDirectRanges (frame boundary): activates or grows the ranges a deferred operation asked for
	//     (FindBuffer with `defer_direct`); a deferred operation cannot do it itself, since both finish all recorded GPU
	//     work and download the GPU-written bytes of the buffers they replace.
	// With `cp_overlay`, command-processor writes (whose bytes the GPU
	// thread knows) are not waited for; WaitDirectGpuWrites then returns true when such writes are still pending over
	// the range, and the caller puts ApplyDirectCpWrites over the guest memory it reads right after.
	[[nodiscard]] bool InDirectRange(uint64_t vaddr, uint64_t size) const;
	bool               WaitDirectGpuWrites(uint64_t vaddr, uint64_t size, bool cp_overlay = false);
	// KYTY_PERF_VERIFY_RANGE=1 (default 0, off; live A/B switch), GPU thread. The resource walker's read-log check
	// (FastVerifyLoggedReads) and its ordinary reads ask WaitPendingReadbacks about a whole 4 KiB page and then remember
	// the page as clean for the rest of the walk. In a direct range the page-wide question waits for a GPU write of
	// unknown bytes anywhere in the page (flush, then the whole queue: ~8 ms once per frame in the river village),
	// although the read may not touch those bytes. With the switch a page that still has a pending GPU write is asked
	// about per read, with the read's own bytes, and is not remembered: a read no pending write overlaps takes guest
	// memory at once (no recorded GPU work will change those bytes), one that overlaps a write of unknown bytes waits
	// for it as before, one under a command-processor write goes to the readers with the overlay as before.
	//   - DirectPendingIn: the GPU writes recorded into a direct range that the host GPU has not finished (as far as
	//     the known GPU tick says) and that overlap [vaddr, vaddr + size): bit 0 any, bit 1 one of unknown bytes;
	//   - VerifyRangeOn: the switch's value now; NoteVerifyRange: PERF verify_range counters (KYTY_TRACE_PERF).
	[[nodiscard]] uint32_t DirectPendingIn(uint64_t vaddr, uint64_t size);
	[[nodiscard]] static bool VerifyRangeOn();
	// KYTY_PERF_VERIFY_RANGE_CHECK=1 (diagnostic, default off, read once; with the switch on): a run the read-log check
	// takes without the page-wide wait is copied, the page-wide question is then asked for each of its pages (waiting
	// as the path without the switch does) and the run is read again; different bytes are counted and the first 16
	// logged (PERF_VERIFY_RANGE_MISMATCH). The waits are back while it is on: it checks, it does not measure speed.
	[[nodiscard]] static bool VerifyRangeCheck();
	enum class VerifyRangeEvent : uint32_t {
		PagePending, PageUnknown, PageOnly, RunCheck, RunWait, RunOverlay, NotCached, Checked, Mismatched, Count
	};
	static void NoteVerifyRange(VerifyRangeEvent event);
	// The number of blocking waits WaitDirectGpuWrites made so far (monotonic).
	[[nodiscard]] static uint64_t DirectWaitTotal();
	// KYTY_PERF_DRAIN_FREE_WAIT (default 0; 1 on, live A/B switch; 2 check, start value only): a WAIT_REG_MEM on a
	// direct-range word that passes as read re-reads it after the command processor itself waited for the word's listed
	// GPU writes, instead of after a drain of the execution thread, when no operation in flight can write buffer memory
	// (guest_gpu/graphicsRun.cpp, GuestGpu::TryDrainFreeDirectWait). While the switch can be on, every listed write is also
	// published under a lock for that thread:
	//   - PublishedDirectTick (any thread): the newest tick of a published write over [vaddr, vaddr + size) the host GPU
	//     has not finished as far as the known GPU tick says (0: none);
	//   - DirectShaderWrites (GPU thread): writes of unknown bytes listed so far (monotonic), for the check that an
	//     operation judged unable to write lists none.
	[[nodiscard]] static uint32_t DrainFreeWaitMode();
	[[nodiscard]] static bool     DrainFreeWaitPossible();
	[[nodiscard]] uint64_t        PublishedDirectTick(uint64_t vaddr, uint64_t size);
	[[nodiscard]] uint64_t        DirectShaderWrites() const noexcept { return m_direct_shader_writes; }
	// KYTY_PERF_SERVE_IN_WAIT (G5, design-gpuwait 2.1; default 0; 1 on, live A/B switch; 2 check, start value only), GPU
	// thread. A compute program lookup (ServeInWaitAllow, before anything of the dispatch is recorded) lets its adoption
	// check (ResolvedReadsHold) run its passes in ServeInWaitPass scopes: a blocking wait of WaitDirectGpuWrites inside
	// one runs the queued guest commands (RunCommandsWhileWaiting) while the GPU works, and a pass in which any command
	// ran is discarded and the check runs again, so the verdict is that of a pass with no command inside it, as when
	// the commands are served between the previous operation and this one. Check: after a pass without commands the
	// check runs a second time without serving and differing verdicts are counted.
	[[nodiscard]] static uint32_t ServeInWaitMode();
	struct ServeInWaitAllow {
		ServeInWaitAllow();
		~ServeInWaitAllow();
		ServeInWaitAllow(const ServeInWaitAllow&)            = delete;
		ServeInWaitAllow& operator=(const ServeInWaitAllow&) = delete;
		uint32_t previous;
	};
	[[nodiscard]] static bool ServeInWaitAllowed();
	struct ServeInWaitPass {
		explicit ServeInWaitPass(bool on);
		~ServeInWaitPass();
		ServeInWaitPass(const ServeInWaitPass&)            = delete;
		ServeInWaitPass& operator=(const ServeInWaitPass&) = delete;
		[[nodiscard]] bool Served() const;
		bool               previous_active;
		bool               previous_served;
	};
	static void NoteServeInWaitCheck(uint32_t passes, bool checked, bool differ);
	// KYTY_PERF_RING_WORDS_GPU (G6, design-gpuwait 2.1; default 0; 1 on, live A/B switch; 2 check, start value only), GPU
	// thread. A shader-read SRT word in a direct range with a pending GPU write of unknown bytes over it is not read by
	// the CPU after waiting for that write: RecordDirectWordCopy records a 4-byte copy from the direct buffer into the
	// word's word-stream slot at this point of the stream, after every GPU write recorded before it (false: no direct
	// buffer at once; the caller reads as before). The shader sees guest memory as the GPU leaves it there, which differs
	// from the CPU read only if the guest writes the word after submitting the work that reads it. Check: the CPU value is
	// read as before too and compared with a second copy recorded right after the first (CheckRingWordGpu, on the
	// priority thread); a difference is the guest touching the word between the CPU read and the copy.
	[[nodiscard]] static uint32_t RingWordsGpuMode();
	// KYTY_PERF_COPY_PREDICT (declared with its state below): between operations, predicted copies to start or apply.
	[[nodiscard]] static bool PredictWork() noexcept { return s_predict_work; }
	void                      IssuePredictedCopies();
	bool                          RecordDirectWordCopy(uint64_t address, vk::Buffer destination, uint64_t destination_offset);
	void                          CheckRingWordGpu(uint64_t address, uint32_t cpu_value);
	static void                   NoteRingWordWait(uint64_t nanoseconds); // check: the CPU wait the copy replaces
	// KYTY_DIAG_PAGE_MARK_CHECK=1 (diagnostic, default off, read once), GPU thread. The resource walker remembers a
	// 4 KiB page as clean for the rest of one walk (OrdinaryReadPageCache in aps5/memoryShim.cpp and kernel/memory.cpp);
	// a read through that mark takes guest memory without a question. An ordinary read sets the mark after asking
	// WaitPendingReadbacks about its own bytes, and in a direct range that question is about those bytes only, so
	// another read of the page can lie over a GPU write that is still pending. With the check on, every read through a
	// mark is asked about with its own bytes (DirectPendingIn, a pending eager copy, GPU-dirty bytes); when something
	// is pending, the bytes the read returns are compared with the readers' answer for them (the wait for writes of
	// unknown bytes, then guest memory with the pending command-processor writes over it). PERF page_mark every ~5 s:
	// read_hits / verify_hits (reads and read-log runs through a mark), pending (of them with something pending over
	// their bytes: unknown, cp, eager, dirty), compared, mismatched, marked_pending (marks set while a direct write was
	// pending elsewhere on the page), not_marked (marks KYTY_FIX_DIRECT_PAGE_MARK withheld). The first 32 differences
	// are logged (DIAG_PAGE_MARK_MISMATCH). It waits where the path without it does not: it checks, it does not
	// measure speed.
	[[nodiscard]] static bool PageMarkCheck();
	// KYTY_FIX_DIRECT_PAGE_MARK=1 (the default from Common::EnvDefaults; =0 off; live A/B switch), GPU thread: the fix for the case above. An ordinary
	// read does not mark its page while a direct write is pending anywhere on the page (DirectPendingIn over the 4 KiB),
	// so every further read of that page in the walk asks about its own bytes again: it waits for a shader write over
	// them and takes a pending command-processor write over guest memory, as the first read does. A page with nothing
	// pending is marked as before (no GPU write is recorded during a walk, so it stays clean until the walk ends), and
	// the read-log check's own marks were always set after a page-wide question. Only waits and overlays the stream
	// order asks for are added.
	[[nodiscard]] static bool DirectPageMarkFix();
	enum class PageMarkEvent : uint32_t {
		ReadHit, VerifyHit, Pending, PendingUnknown, PendingCp, PendingEager, PendingDirty, Compared, Mismatched,
		MarkedPending, NotMarked, Count
	};
	static void NotePageMark(PageMarkEvent event);
	void               ApplyDirectCpWrites(uint64_t vaddr, void* data, uint64_t size) const;
	void               UnmapDirect(uint64_t vaddr, uint64_t size);
	void               ActivateWantedDirectRanges();
	// KYTY_FIX_VIRTUAL_GPU_CLOCK (virtualGpuClock.cpp), GPU thread: a value the GPU produced at an earlier, finished point
	// of the command stream (an end-of-pipe clock value resolved once its command buffer completed) for [address,
	// address + size).
	// Plain: the bytes are in no GPU-modified page; the caller stores them (at most the page becomes CPU-modified).
	// Stored: written to guest memory and to the GPU copy at the current point of the stream, the page left GPU-owned
	// (the TryCpWriteThrough path).
	// Retry: not now (no open command buffer, or a recorded write or pending copy over the bytes).
	// Drop: the GPU owns the bytes, or an image covers them; only a synchronizing store could write them.
	enum class CompletedWrite : uint32_t { Stored, Plain, Retry, Drop };
	[[nodiscard]] CompletedWrite WriteCompletedValue(uint64_t address, const void* data, uint64_t size);
	// KYTY_TRACE_PERF: the PERF cp_write_through line.
	void ReportCpWriteThroughPerf();

	// The upload batch (see the .cpp), GPU thread: while the outermost scope lives, read-only synchronizations of
	// cached buffers only claim their CPU-modified pages (MemoryTracker::ClaimUploadRanges). The scope's end, or the
	// first operation that must see the bytes in a cached buffer before that, commits the batch: one protection update
	// per tracker region for all claims, then the staging copies and upload commands in claim order. Inner scopes do
	// nothing.
	class UploadBatchScope {
	public:
		explicit UploadBatchScope(BufferCache& cache);
		~UploadBatchScope();
		UploadBatchScope(const UploadBatchScope&)            = delete;
		UploadBatchScope& operator=(const UploadBatchScope&) = delete;

	private:
		BufferCache* m_cache = nullptr;
	};
	// KYTY_TRACE_PERF: the PERF protect_batch line.
	void ReportProtectPerf();
	// GPU thread: an asynchronous publish of guest bytes [vaddr, vaddr + size), a priority operation queued at the current
	// tick (this cache's and the texture cache's download publishes, the only priority operations that write guest memory
	// on the priority thread). The publish sets the returned flag when it ended (RenderContext::UnmapMemory waits for it).
	[[nodiscard]] std::shared_ptr<std::atomic_bool> NotePendingGuestWrite(uint64_t vaddr, uint64_t size);
	// GPU thread: the newest tick of an unfinished publish into [vaddr, vaddr + size), 0 when none (finished ones go).
	[[nodiscard]] uint64_t PendingGuestWriteTick(uint64_t vaddr, uint64_t size);

private:
	struct PendingGuestWrite {
		uint64_t                          start = 0;
		uint64_t                          end   = 0;
		uint64_t                          tick  = 0;
		std::shared_ptr<std::atomic_bool> done;
	};
	std::vector<PendingGuestWrite> m_pending_guest_writes; // RenderContext::UnmapMemory's wait, GPU thread
	// The join tick: a host-mapped read waits for the buffer's creation tick too (see TryBarDownload).
	enum class CpWriteDecline : uint32_t { Overlap, Pending, Merge, Image, NoBuffer, Shape, Count };
	// The write-through itself, once the page is known GPU-modified: Count when written, else the reason it is not.
	// replace: also over GPU-owned bytes.
	CpWriteDecline CpWriteThroughOwnedPage(uint64_t address, const void* data, uint64_t size, bool replace);
	struct CpWriteStats {
		std::atomic<uint64_t> writes {0}, bytes {0}, replaced {0};
		std::array<std::atomic<uint64_t>, static_cast<uint32_t>(CpWriteDecline::Count)> declined {};
	};
	CpWriteStats m_cp_write_stats;
	enum class CallerFallback : uint32_t { None, UnknownTick, Outside, Large, NoMapping, Pending, Count };
	struct CallerPlan {
		uint64_t       tick     = 0; // the request would wait for the GPU up to this tick (0: it would not wait)
		CallerFallback fallback = CallerFallback::None; // it would wait, for a reason only the drain path serves
		// KYTY_PERF_COPY_HANDBACK: a hand-back copy of the bytes is under way (its value of the copy timeline), and, when
		// the request would wait for nothing, the host-mapped download it would make now (buffer: null when none).
		uint64_t                                   copy_tick = 0;
		Buffer*                                    buffer    = nullptr;
		uint64_t                                   lo = 0, hi = 0, newest = 0, total = 0;
		std::vector<std::pair<uint64_t, uint64_t>> intervals;
	};
	// GPU thread, before the request changes anything: what a guest fault's ReadMemory request would wait for.
	[[nodiscard]] CallerPlan PlanCallerWait(uint64_t vaddr, uint64_t size, bool is_write);
	struct CallerWaitStats {
		std::atomic<uint64_t> eligible {0};       // requests of retried guest faults from other threads than the GPU thread
		std::atomic<uint64_t> deferred {0};       // requests handed back to the faulting thread at least once
		std::atomic<uint64_t> deferrals {0};      // hand-backs (at most CallerMaxDeferrals per request)
		std::atomic<uint64_t> caller_wait_ns {0}; // time the faulting threads waited for the GPU themselves
		std::atomic<uint64_t> retries {0};        // attempts sent again after a wait
		std::atomic<uint64_t> retry_slow {0};     // of those, attempts that still ran >= 1 ms on the GPU thread
		std::atomic<uint64_t> new_writer {0};     // retries that found a newer unfinished writer of the bytes
		std::atomic<uint64_t> exhausted {0};      // requests served by the drain path after the last hand-back
		std::atomic<uint64_t> gpu_ns {0};         // GPU-thread time of all eligible attempts
		std::atomic<uint64_t> gpu_defer_ns {0};   // of it, attempts that were (1) / would have been (2) handed back
		std::atomic<uint64_t> gpu_retry_ns {0};   // of it, attempts after a wait
		std::atomic<uint64_t> fallback[static_cast<uint32_t>(CallerFallback::Count)] {};
		// KYTY_PERF_COPY_HANDBACK: copies started and handed back; requests handed back for a copy already under way;
		// requests it could have served but did not (small, no plan, no slot, dead ring, overlap); waits of the GPU
		// thread for such a copy (its own readers, a download over the window).
		std::atomic<uint64_t> copy_issued {0}, copy_joined {0}, copy_declined {0}, copy_exec_waits {0}, copy_chained {0};
	};
	CallerWaitStats m_caller_stats;
	// KYTY_RECORD_KEYED_VERIFY: pages of the written record-keyed ranges listed per frame (a ring of frames, rotated by
	// RunGarbageCollector), and the counts of checked and uncovered BDA-written pages.
	std::array<std::unordered_set<uint64_t>, 16> m_rk_listed_pages;
	uint64_t                                     m_rk_frame      = 0;
	uint64_t                                     m_rk_checks     = 0;
	uint64_t                                     m_rk_uncovered  = 0;
	uint64_t                                     m_rk_unverified = 0;
	uint64_t                                     m_rk_incomplete = 0;
	uint64_t                                     m_rk_incomplete_frame = 0; // last frame with one, + 1 (0: none)
	// D6 step C0 (KYTY_D6_C0), GPU thread: reported store pages skipped as listed, unlisted store pages marked, and
	// CPU-dirty pages an unlisted store reported (UNLISTED_BDA_STORE_CONFLICT).
	uint64_t m_c0_listed_skipped = 0, m_c0_unlisted_pages = 0, m_c0_conflicts = 0;
	// D6 step K2 (KYTY_RECORD_KEYED_VERIFY), GPU thread: reported store pages no listing covers (NoteBdaStoreWouldMark).
	uint64_t m_c0_would_mark = 0;
	// KYTY_D6_C0=2/3 (UploadClobberCheck): the guest bytes of pages from before the write fault that made them CPU-dirty
	// (m_c0_snap_mutex: fault handlers add, the GPU thread takes them at the upload), and the pages whose GPU-owned bytes
	// DropDeadRingBytes dropped without a copy since (their old GPU bytes are no clobber; GPU thread only).
	std::mutex                                                               m_c0_snap_mutex; // leaf
	std::unordered_map<uint64_t, std::unique_ptr<std::array<uint8_t, 4096>>> m_c0_snapshots;
	std::unordered_set<uint64_t>                                              m_c0_dead_pages;
	std::unique_ptr<Buffer>                                                   m_c0_ring;      // download ring
	uint64_t                                                                  m_c0_ring_head = 0;
	std::atomic<uint64_t>                                                     m_c0_ring_tail {0};
	std::atomic<uint64_t> m_c0_checked {0}, m_c0_clobber_pages {0}, m_c0_clobber_bytes {0}, m_c0_dead_bytes {0};
	std::atomic<uint64_t> m_c0_unchecked {0}, m_c0_snap_dropped {0}, m_c0_stale_bytes {0}, m_c0_no_base_bytes {0};
	std::atomic<uint64_t> m_c0_snap_raced {0}; // D4 step K0: later faults of a clean period (their copies not kept)
	// The bytes UploadCopies staged (guest address, bytes), taken by the RecordBufferUpload that follows, and per page
	// a hash of each 32-byte chunk as last uploaded: a chunk the GPU copy still holds as uploaded was not written by
	// the GPU since, so a difference to the snapshot there is guest memory that changed without the tracker (a stale GPU
	// copy the upload refreshes), not a clobber. GPU thread only.
	struct C0Baseline {
		std::array<uint32_t, 128> hash {};
		std::array<uint64_t, 2>   valid {};
	};
	std::vector<std::pair<uint64_t, std::vector<uint8_t>>> m_c0_staged;
	std::unordered_map<uint64_t, C0Baseline>               m_c0_baseline;
	void NoteC0Staged(uint64_t address, const uint8_t* bytes, uint64_t size);
	// D4 step K0: the page event log (C0Note) and the clobbered pages per newest changing event.
	struct C0PageEvent {
		uint64_t tag   = 0;
		uint64_t tick  = 0;
		uint32_t frame = 0;
		uint16_t lo    = 0; // the event's bytes in the page: [lo, hi]
		uint16_t hi    = 0;
		C0Event  kind  = C0Event::Count;
	};
	struct C0PageLog {
		std::array<C0PageEvent, 8> events {};
		uint32_t                   head = 0;
	};
	std::mutex                                                             m_c0_log_mutex; // leaf
	std::unordered_map<uint64_t, C0PageLog>                                m_c0_log;
	std::array<std::atomic<uint64_t>, static_cast<size_t>(C0Event::Count) + 1u> m_c0_writer_pages {}; // last: none
	[[nodiscard]] C0Event C0PageHistory(uint64_t page, std::string& text);
	void                  CheckUploadClobber(Buffer& buffer, std::span<const vk::BufferCopy> copies);
	void UnmarkPublished(uint64_t start, uint64_t end);
	// The read window unmarks only pages no unfinished eager copy still has to publish.
	void        UnmarkWindow(uint64_t lo, uint64_t hi);

	std::function<void(uint64_t, uint64_t)> m_gpu_write_observer;

	// D4b step 1, append windows (GPU thread)
	struct AppendWindowState {
		std::atomic<uint32_t> before {0}, after {0};
		std::atomic<uint32_t> done {0}; // bit 0: before copied, bit 1: after copied
	};
	struct AppendWindow {
		uint64_t                           counter = 0;
		std::shared_ptr<AppendWindowState> state;
		uint64_t                           max_records = 0; // threads of its dispatches: the counter moves at most this
		std::vector<uint32_t>              pools;           // indices into m_append_pools
		bool                               closed = false, poisoned = false;
	};
	struct AppendPool {
		uint64_t base = 0, size = 0;
		uint32_t stride = 0, min_off = 0, max_end = 0;
		bool     active = false, poisoned = false;
		uint32_t windows = 0; // windows of this cycle not resolved yet
		uint64_t last_mark = 0; // m_append_mark_clock at its last mark
		RangeSet keep;        // never released this cycle: GPU-dirty at its start, written by windows or other writers
	};
	struct AppendStats {
		uint64_t dispatches = 0, unqualified = 0, windows = 0, resolved = 0, copy_failed = 0, poison_foreign = 0,
		         poison_upload = 0, poison_bounds = 0, releases = 0, released_kb = 0, kept_kb = 0, pending = 0,
		         records = 0, marks = 0, no_slot = 0;
	};
	std::vector<AppendPool>   m_append_pools;
	uint64_t                  m_append_mark_clock = 0;
	std::vector<AppendWindow> m_append_windows;
	std::vector<uint64_t>     m_append_self_counters;
	bool                      m_append_tagged_obtain = false;
	AppendStats               m_append_stats;
	void AppendResolveWindows();
	void AppendPoisonWindowsOf(uint64_t vaddr, uint64_t size, uint64_t AppendStats::*cause);
	void AppendForeignWrite(uint64_t vaddr, uint64_t size); // a GPU write of pool or counter bytes not by the windows
	void AppendNoteUpload(uint64_t vaddr, uint64_t size);   // CPU bytes uploaded over a counter
	bool AppendRelease(uint64_t vaddr, uint64_t size);      // ReadMemory: true when the range holds no GPU byte now
	void AppendReport();

	friend struct BufferCacheTestAccess;

	bool IsBufferInvalid(BufferId id) const {
		const auto* buffer = m_slot_buffers.try_get(id);
		return buffer == nullptr || buffer->is_deleted;
	}

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 44, 20>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	void TouchBuffer(Buffer& buffer);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void Unregister(BufferId id);
	template <bool insert>
	void ChangeRegister(BufferId id);
	void DeleteBuffer(BufferId id);
	// Who asked for a synchronization (an image source and a sized request outside the region pass are not batched).
	enum class UploadCaller : uint8_t { Other, Obtain, BdaRange, Image, Region };
	[[nodiscard]] bool SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                     bool is_written, bool is_texel_buffer,
	                                     uint64_t* uploaded_bytes = nullptr,
	                                     UploadCaller caller = UploadCaller::Other);
	// Records the copies of `copies` from `source` into `buffer` at this point of the command stream, between barriers
	// (SynchronizeBuffer, CommitUploadBatch).
	void RecordBufferUpload(Buffer& buffer, vk::Buffer source, std::span<const vk::BufferCopy> copies);
	// The barrier, copy, barrier of an upload into `target` (deferredCommands.h may record them later).
	void RecordBufferUploadCommands(vk::Buffer target, uint64_t target_size, vk::Buffer source,
	                                std::span<const vk::BufferCopy> copies);

	// The upload batch (UploadBatchScope), GPU thread. Why a batch was committed (PERF protect_batch commits[]).
	enum class BatchCommit : uint8_t { ScopeEnd, Sync, CreateBuffer, DeleteBuffer, ImageCopy, BdaStore, CpWrite, Collector, Count };
	struct UploadBatchEntry {
		Buffer*      buffer      = nullptr;
		uint32_t     first_range = 0; // into UploadBatch::ranges
		uint32_t     range_count = 0;
	};
	struct UploadBatch {
		bool                                       active = false; // an UploadBatchScope is open (and not paused)
		std::vector<UploadBatchEntry>              entries;        // claimed synchronizations, in claim order
		std::vector<std::pair<uint64_t, uint64_t>> ranges;         // their claimed page runs (address, bytes)
		std::vector<RegionManager*>                regions;        // tracker regions whose protection waits
	};
	UploadBatch m_upload_batch;
	[[nodiscard]] UploadBatch& Batch() noexcept { return m_upload_batch; }
	// SynchronizeBuffer's read-only path inside an open batch: claims the CPU-modified pages of the range.
	void ClaimBatchedUpload(Buffer& buffer, uint64_t vaddr, uint64_t size, uint64_t* uploaded_bytes);
	// Write-protects every claimed page (one update per tracker region), then stages and records the claimed uploads.
	// Nothing when the batch is empty; safe to call again from inside (a staging wait that runs deferred work).
	void CommitUploadBatch(BatchCommit reason);
	// Commits the open batch and keeps it closed while it lives (an obtained range that is copied from right away).
	class UploadBatchPause {
	public:
		UploadBatchPause(BufferCache& cache, bool pause);
		~UploadBatchPause();
		UploadBatchPause(const UploadBatchPause&)            = delete;
		UploadBatchPause& operator=(const UploadBatchPause&) = delete;

	private:
		BufferCache* m_cache = nullptr;
	};
	struct ProtectBatchStats {
		std::atomic<uint64_t> scopes {0};         // outermost scopes opened
		std::atomic<uint64_t> claims {0};         // synchronizations that claimed pages
		std::atomic<uint64_t> claim_regions {0};  // tracker regions they cleared pages in (protection updates unbatched)
		std::atomic<uint64_t> claimed_bytes {0};  // bytes claimed
		std::atomic<uint64_t> commits {0};        // commits that had claims
		std::atomic<uint64_t> region_updates {0}; // protection updates the commits made
		std::atomic<uint64_t> commit_ns {0};      // time in commits (KYTY_TRACE_PERF only)
		std::array<std::atomic<uint64_t>, static_cast<size_t>(BatchCommit::Count)> reasons {};
	};
	ProtectBatchStats m_protect_batch_stats;
	// The bookkeeping of a range a draw or dispatch is about to write (ObtainBuffer, ObtainBdaRange).
	void NoteObtainedWrite(uint64_t vaddr, uint64_t size);
	// The write memo: NoteObtainedWrite of a range noted at this tick with nothing undone since (the GPU-dirty set,
	// the write ticks and the granule ticks already hold it): only its notifications.
	void NoteObtainedWriteRepeat(uint64_t vaddr, uint64_t size, bool same_tick);
	// KYTY_TRACE_RING_CENSUS (diagnostic, bufferCacheRingCensus.cpp; execution thread): the writer of a range a draw or
	// dispatch is about to write; a write request's start, its byte compare (before the page is released) and its record.
	void RingCensusWrite(uint64_t vaddr, uint64_t size, bool repeat);
	void RingCensusBegin(RingCensus::Request& request, uint64_t vaddr, uint64_t size);
	void RingCensusCompare(RingCensus::Request& request);
	void RingCensusEnd(RingCensus::Request& request);
	// KYTY_PERF_DIRECT_RANGES (see InDirectRange).
	struct DirectRange {
		uint64_t start   = 0;
		uint64_t end     = 0;
		BufferId id      = {};
		bool     active  = false;
		bool     refused = false; // cannot be imported / holds images: the range stays on the tracked path
		// A deferred operation's request to activate or grow the range, served at the frame boundary
		// (ActivateWantedDirectRanges): the union of the requested bytes.
		bool     wanted     = false;
		uint64_t want_start = 0;
		uint64_t want_end   = 0;
	};
	struct DirectGpuWrite {
		uint64_t start = 0;
		uint64_t end   = 0;
		uint64_t tick  = 0;
		// A command-processor write's bytes (cp_size = end - start), 0 for others.
		uint32_t                cp_size = 0;
		std::array<uint8_t, 64> cp_bytes {};
		int64_t                 noted_us = 0; // KYTY_TRACE_PERF: steady-clock microseconds when the write was recorded
	};
	std::vector<DirectRange>    m_direct_ranges;
	std::vector<DirectGpuWrite> m_direct_gpu_writes;
	// Per direct range: the span of its listed GPU writes (a superset after removals; rebuilt with them), so a read outside
	// every span needs no scan of the list (WaitDirectGpuWrites, ~1 M calls per second).
	std::vector<std::pair<uint64_t, uint64_t>> m_direct_write_spans;
	bool                                     m_direct_write_unspanned = false; // a listed write in no range: scan
	void                                     RebuildDirectWriteSpans();
	[[nodiscard]] bool                       DirectWritesMayOverlap(uint64_t vaddr, uint64_t size) const;
	// The pending command-processor writes the last WaitDirectGpuWrites left to the reader.
	std::vector<DirectGpuWrite> m_direct_cp_overlay;
	// KYTY_PERF_DRAIN_FREE_WAIT: the listed writes as the command processor sees them (start, end, tick).
	struct PublishedDirectWrite {
		uint64_t start = 0;
		uint64_t end   = 0;
		uint64_t tick  = 0;
	};
	std::mutex                        m_published_direct_mutex;
	std::vector<PublishedDirectWrite> m_published_direct;
	uint64_t                          m_direct_shader_writes = 0;
	uint32_t                    m_direct_mode_seen = 0;
	bool                        m_direct_parsed    = false;
	bool                        m_direct_wanted    = false; // a range is wanted, or a live A/B switch waits
	// W2.2: the execution thread's changes of the direct ranges' fields (start, end, id, active, refused) and of the mode
	// seen, nested: the outermost bumps the generation to odd before and to even after (a helper reads the fields between
	// two equal even generations; a Direct plan answer holds while the generation is the one it saw).
	uint32_t m_direct_change_depth = 0;
	void     DirectChangeBegin();
	void     DirectChangeEnd();
	struct DirectChange {
		BufferCache& cache;
		explicit DirectChange(BufferCache& c): cache(c) { cache.DirectChangeBegin(); }
		~DirectChange() { cache.DirectChangeEnd(); }
		DirectChange(const DirectChange&)            = delete;
		DirectChange& operator=(const DirectChange&) = delete;
	};
	[[nodiscard]] uint32_t DirectMode();
	// The active direct buffer holding [vaddr, vaddr + size) (activating or growing its range), else a null id.
	// `deferred` (non-null: the caller is a deferred operation, which must not wait): a range that would have to be
	// activated, grown or ended is not touched but marked wanted, *deferred is set and a null id returned.
	[[nodiscard]] BufferId DirectBufferFor(uint64_t vaddr, uint64_t size, bool* deferred = nullptr);
	void                   ActivateDirect(DirectRange& range);
	void                   DeactivateDirect(DirectRange& range, bool replace = true);
	void                   NoteDirectGpuWrite(uint64_t vaddr, uint64_t size, const void* cp_data = nullptr);
	[[nodiscard]] bool     DropDeadRingBytes(uint64_t vaddr);
	[[nodiscard]] bool     RingBytesAllDead(uint64_t vaddr); // its verdict without the changes
	[[nodiscard]] vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
	                                      uint64_t total_size);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// found_image (optional): whether an image lies over the range (the obtain memo keeps none of those).
	[[nodiscard]] bool SyncBufferFromWrittenImage(Buffer& buffer, uint64_t vaddr, uint64_t size, bool* found_image = nullptr);
	// Synchronous downloads publish before returning; asynchronous callers wait before reuse.
	template <bool async>
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size);

	// Eager readback: 64 KiB windows the CPU keeps reading after GPU writes ("hot")
	// are copied back asynchronously after each processed submission. When the copy has been
	// published and no newer GPU write was recorded for the window, its pages stop being
	// GPU-modified, so the CPU reads them without a fault or a GPU wait.
	struct HotRead {
		uint64_t start      = 0;
		uint64_t end        = 0;
		uint32_t hits       = 0;
		uint64_t last_use   = 0;
		uint64_t generation = 0;
	};
	struct EagerReadback {
		uint64_t                          start      = 0;
		uint64_t                          end        = 0;
		uint64_t                          generation = 0;
		uint64_t                          tick       = 0;
		std::shared_ptr<std::atomic_bool> done;
		// GPU-dirty byte intervals the copy took out of m_gpu_modified_ranges (a superset).
		std::vector<std::pair<uint64_t, uint64_t>> bytes;
		// KYTY_PERF_COPY_HANDBACK: not 0 for a hand-back copy. It wrote the backing itself on the second queue and is
		// finished when the copy timeline reached this value (asked there, then remembered in `done`).
		uint64_t copy_tick = 0;
		bool     predicted = false; // KYTY_PERF_COPY_PREDICT: issued ahead of any fault
		uint64_t source    = 0; // KYTY_PERF_BAR_GPU_COPY_VERIFY: the copied buffer's handle
		uint32_t frame     = 0; // and the presented frame of the copy's start
	};
	// KYTY_PERF_COPY_HANDBACK (barGpuCopy.h). CopyHandbackOn: the switch's value now. StartCopyReadback: GPU thread, in a
	// guest fault's request before it changed anything; starts the copy of the plan's download and lists it, returns
	// the copy timeline value to wait for (0: nothing started). EagerDone: whether a readback's copy finished.
	// WaitCopyReadbacks: waits for the hand-back copies under way over the range. UnmarkPublishedWindow: the pages of
	// the window without GPU-dirty bytes and without bytes of another unfinished readback leave the GPU-modified
	// state, run by run. DeadRingWouldFree: DropDeadRingBytes's answer for the page, without changing anything.
	[[nodiscard]] static bool CopyHandbackOn();
	[[nodiscard]] static bool CopyChainOn(); // KYTY_PERF_COPY_CHAIN (bufferCache.cpp, before StartCopyReadback)
	[[nodiscard]] uint64_t    StartCopyReadback(CallerPlan& plan, uint64_t vaddr, bool is_write, bool chained = false, bool predicted = false);
	[[nodiscard]] bool        EagerDone(const EagerReadback& entry) const;
	void                      UnmarkPublishedWindow(uint64_t lo, uint64_t hi);
	[[nodiscard]] bool        DeadRingWouldFree(uint64_t vaddr);
	// KYTY_PERF_BAR_GPU_COPY_VERIFY=1 with a hand-back copy: when the finished readback is applied, its bytes in the
	// backing are compared with the buffer's mapping, interval by interval; intervals a GPU write was recorded over
	// since are GPU-dirty again and are skipped (PERF bar_gpu_copy verified / mismatched).
	void                      VerifyCopyReadback(const EagerReadback& entry);
	void                      RedoHandbackWrittenDuringFlight(const EagerReadback& entry); // KYTY_FIX_HANDBACK_REDO
	// KYTY_PERF_COPY_PREDICT (G4, design-gpuwait 2.2; default 0; 1 on, 2 check; with KYTY_PERF_COPY_HANDBACK), GPU thread.
	// A 64 KiB granule that retried guest read faults took in at least 2 of the last 4 frames is predicted, with the
	// 4 KiB pages they touched. When a GPU write raises a predicted granule's write tick (NoteGpuWriteTick), the granule
	// is queued; between operations, once that tick's command buffer is submitted, IssuePredictedCopies starts a
	// hand-back copy of the touched pages' GPU-dirty bytes waiting for the tick (StartCopyReadback, the hand-back
	// state transitions), and applies predicted copies that finished (no fault needed). Applied predicted windows take
	// the redo of KYTY_FIX_HANDBACK_REDO always. A predicted granule stays predicted while its copies are issued and
	// decays after 8 frames without a write. Check: every applied predicted window is verified against the mapping.
	[[nodiscard]] static uint32_t CopyPredictMode();
	void                          NotePredictFault(uint64_t vaddr);
	void                          PredictFrame();
	void                          PredictNoteWrite(uint64_t first_key, uint64_t last_key);
	struct PredictGranule {
		uint8_t  frames    = 0; // bit 0: this frame, bit i: i frames ago (a fault, or a predicted copy issued)
		uint16_t pages     = 0; // touched 4 KiB pages
		uint8_t  idle      = 0; // frames without a write since predicted
		bool     predicted = false;
	};
	std::unordered_map<uint64_t, PredictGranule> m_predict;
	std::vector<uint64_t>                        m_predict_keys;  // predicted granules, sorted
	std::vector<uint64_t>                        m_predict_dirty; // written since their last copy
	static inline bool                           s_predict_work = false;
	// KYTY_PERF_BAR_GPU_COPY_VERIFY=2: a hand-back mismatch (its first differing byte and the tick the copy waited for),
	// classified 3 frames later by the page events (C0ByteChangedAfter: an event that changes the GPU copy, newer than
	// the tick, covering the byte - a late-learned GPU write - or none).
	struct HandbackLate {
		uint64_t address = 0;
		uint64_t tick    = 0;
		uint32_t frame   = 0;
	};
	std::vector<HandbackLate> m_handback_late;
	[[nodiscard]] bool        C0ByteChangedAfter(uint64_t address, uint64_t tick);
	// True when an unfinished eager readback is copying one of the bytes of the range.
	[[nodiscard]] bool PendingReadbackCovers(uint64_t vaddr, uint64_t size) const;
	[[nodiscard]] uint32_t InFlightReadbackKinds(uint64_t vaddr, uint64_t size) const;
	std::atomic<uint64_t>  m_in_flight_reads[2][static_cast<uint32_t>(DirtyReader::Count)] {};
	std::atomic<uint64_t>  m_in_flight_ahead_reads {0};
	void                   InFlightRebuild();
	void        RecordHotRead(uint64_t vaddr);
	void        BumpHotGeneration(uint64_t vaddr, uint64_t size);
	void        ApplyEagerReadbacks(uint64_t vaddr, uint64_t size, bool wait);
	void        IssueEagerReadbacks();
	// KYTY_TRACE_SYNC_TIMELINE: write tick and GPU-dirty bytes of what the running reader request needs.
	void        NoteSyncTimelineNeed(uint64_t vaddr, uint64_t size);

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	FaultManager                                      m_fault_manager;
	Buffer                                            m_gds_buffer;
	Buffer                                            m_bda_pagetable_buffer;
	// the BDA fallback (M06)
	bool    FallbackEntry(uint64_t page, vk::DeviceAddress* entry);
	// The imported backing granule (null: not imported, refused, or not wanted and `want` false); also the direct downloads'
	Buffer* FallbackChunk(size_t chunk, bool want, uint64_t vaddr = 0);
	void    SizeFallback();
	void    WriteFallbackRange(uint64_t vaddr, uint64_t size);
	// BDA map-time import (AGENTS 19): the pool's pages (granule + 1 per 16 KiB page, 0: not mapped), mapped pages per granule, the
	// granules asked of the worker, the worker's queue and its finished imports.
	void MapTimeMap(uint64_t vaddr, uint64_t size);   // RenderContext::MapMemory (game threads)
	void MapTimeInstall();                            // GPU thread (ApplyFallbackMaps)
	void MapTimeRequestGranule(size_t granule, uint64_t vaddr); // GPU thread: a download or a read as 0 wants it
	void MapTimeSize(uint64_t total);                           // under m_maptime.mutex: the tables' first sizing
	void MapTimeWorker();
	struct MapTimeState {
		std::mutex                                           mutex;
		std::condition_variable                              wake;
		std::vector<uint64_t>                                granule_vaddr; // where a page of the granule was mapped last
		std::vector<uint8_t>                                 requested; // per piece: 1 asked or imported, 2 refused
		std::vector<uint8_t>                                 granule_requested; // single granules (outside the pool)
		std::deque<size_t>                                   queue;
		std::vector<vk::DeviceMemory>                        memories; // the imported pieces (freed at the end)
		std::vector<std::pair<size_t, std::unique_ptr<Buffer>>> ready;
		std::thread                                          worker;
		bool                                                 stop = false;
		std::atomic<uint64_t>                                zero {0};
		uint64_t imports = 0, granule_imports = 0, imported_bytes = 0, import_us = 0, refused = 0, installed = 0, dropped = 0;
	} m_maptime;
	std::vector<std::unique_ptr<Buffer>>              m_fallback_chunks; // imported backing per granule, null: not yet
	bool                                              m_fallback_sized = false;
	std::mutex                                        m_fallback_pending_mutex;
	std::vector<std::pair<uint64_t, uint64_t>>        m_fallback_pending;
	std::unordered_set<uint64_t>                      m_fallback_pages; // 16 KiB guest pages with a fallback entry
	std::unordered_map<uint64_t, uint32_t>            m_fallback_replaced; // page -> frame a buffer replaced its entry
	uint32_t m_fallback_replaced_pruned = ~0u; // present count of the last pruning of m_fallback_replaced
	struct FallbackStats {
		uint64_t maps = 0, pages = 0, replaced = 0, restored = 0, cleared = 0, chunks = 0, applies = 0;
		uint64_t imported_bytes = 0;
		// D6b: ClearFallback's Fill calls, its GPU-thread time.
		uint64_t clear_fills = 0, clear_us = 0;
		bool operator==(const FallbackStats&) const = default;
	} m_fallback_stats;
	// D6b: the stats line is printed at most every 5 s and only when a value changed since the last one (applies aside).
	FallbackStats                         m_fallback_printed;
	std::chrono::steady_clock::time_point m_fallback_print_time;
	// DownloadBufferMemory<false>: the copy path's GPU copies go straight into the guest backing (imported per 4 MiB of
	// the backing view) instead of the download ring. Per GPU-dirty interval: its guest address, its offset in the
	// backing and its size.
	struct DownloadToBackingItem {
		uint64_t vaddr = 0, backing = 0, size = 0;
	};
	struct DownloadToBackingPlan {
		std::vector<DownloadToBackingItem>                             items;
		std::vector<std::pair<Buffer*, std::vector<vk::BufferCopy>>>   chunks; // copies per imported chunk
	};
	// False: some interval cannot go straight to the backing (no single mapping, import failed); the whole
	// download then takes the download ring as before.
	[[nodiscard]] bool    PlanDownloadToBacking(uint64_t buffer_address, const std::vector<vk::BufferCopy>& copies,
	                                            DownloadToBackingPlan& plan);
	// The second-queue GPU copy of a host-mapped download's intervals (barGpuCopy.h), after the caller's waits; with
	// KYTY_PERF_BAR_GPU_DIRECT straight into the imported backing when PlanDownloadToBacking can place them all. False:
	// no GPU copy was made (the caller reads the mapping).
	[[nodiscard]] bool    BarGpuCopyIntervals(Buffer& buffer, uint64_t base,
	                                          const std::vector<std::pair<uint64_t, uint64_t>>& intervals, uint64_t tick);
	Common::SlotVector<Buffer>                        m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                         m_buffers;
	PageTable                                         m_page_table;
	RangeSet                                          m_gpu_modified_ranges;
	RangeSet                                          m_in_flight_ranges;
	// Every mutation of m_gpu_modified_ranges goes through these two, under the lock (the resolve thread
	// read the set while the GPU thread changes it).
	void                                              GpuDirtyAdd(uint64_t vaddr, uint64_t size);
	void                                              GpuDirtySubtract(uint64_t vaddr, uint64_t size);
	std::shared_mutex                                 m_gpu_dirty_mutex;
	// Guards m_page_table and m_slot_buffers' insertions/erasures against the resolve thread's LookupBuffer; taken by the GPU thread
	// only for those changes.
	TrackingSpinLock                                  m_lookup_lock {PerfLockSiteBufferLookup};
	// The GPU thread's guard for one registry change.
	class LookupWriteGuard {
	public:
		explicit LookupWriteGuard(BufferCache& cache): m_cache(cache) {
			if (t_translator != 0) TranslatorNote(0); // project 15 S2a: a registry change inside a peek
			m_cache.m_lookup_lock.lock();
		}
		~LookupWriteGuard() { m_cache.m_lookup_lock.unlock(); }
		LookupWriteGuard(const LookupWriteGuard&)            = delete;
		LookupWriteGuard& operator=(const LookupWriteGuard&) = delete;

	private:
		BufferCache& m_cache;
	};
	MemoryTracker                                     m_memory_tracker;
	StreamBuffer                                      m_staging_buffer;
	StreamBuffer                                      m_stream_buffer;
	// KYTY_PERF_SRT_GPU_WORDS: device-local ring with a device address for the values of GPU-read SRT words that lie in a
	// direct range (host memory): shaders read them from here instead of host memory over the bus in every invocation.
	StreamBuffer                                      m_word_stream;
	StreamBuffer                                      m_download_buffer;
	StreamBuffer                                      m_device_buffer;
	bool                                              m_device_tables_cleared = false; // ClearDeviceTablesOnce
	[[nodiscard]] StreamBuffer& Staging() noexcept { return m_staging_buffer; }
	[[nodiscard]] StreamBuffer& Stream() noexcept { return m_stream_buffer; }

public:
	[[nodiscard]] StreamBuffer& WordStream() noexcept { return m_word_stream; }

private:
	TextureCache&                                     m_texture_cache;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	uint64_t m_gc_tick            = 0;
	uint64_t m_registered_bytes   = 0; // RegisteredBytes()
	// TakeCensus: the collector's work since the previous census.
	uint64_t m_gc_runs = 0, m_gc_retired = 0, m_gc_retired_bytes = 0, m_gc_kept_bda = 0, m_gc_dirty_skipped = 0;
	// KYTY_VRAM_BUDGET: bytes retired per command-buffer tick, kept until the GPU retired the tick (the buffers are
	// destroyed then); buffers retired before an allocation; the collector tick at which a make-room scan found
	// nothing to retire.
	struct ReleasingBytes {
		uint64_t tick  = 0;
		uint64_t bytes = 0;
	};
	std::vector<ReleasingBytes> m_releasing;
	uint64_t                    m_alloc_retired = 0, m_alloc_retired_bytes = 0;
	uint64_t                    m_make_room_dry_tick = ~uint64_t {0};
	void                        MakeRoomForBuffer(uint64_t vaddr, uint64_t size);
	void                        NoteRetiredBytes(uint64_t bytes);
	[[nodiscard]] uint64_t      RecentlyRetiredBytes();
	std::vector<HotRead>       m_hot_reads;
	std::vector<EagerReadback> m_eager_readbacks;

	// Write ticks are tracked per 64 KiB granule; buffers by creation tick.
	void NoteGpuWriteTick(uint64_t vaddr, uint64_t size);
	// The submission tick of the newest GPU write of each byte range (interval map).
	void                   NoteWriteTick(uint64_t start, uint64_t end, uint64_t tick);
	void                   NoteWriteFrame(uint64_t start, uint64_t end); // data ring only: the recorded work's game frame
	[[nodiscard]] uint64_t QueryWriteTick(uint64_t start, uint64_t end) const; // 0: part of it unknown
	struct WriteTick {
		uint64_t end;
		uint64_t tick;
	};
	std::map<uint64_t, WriteTick> m_write_ticks;
	std::map<uint64_t, WriteTick> m_write_frames; // data ring: game frame + 1 of the newest GPU write of each byte range
	static void InsertTickRange(std::map<uint64_t, WriteTick>& ticks, uint64_t start, uint64_t end, uint64_t tick);
	// The same answers with fewer entries (an insert already held is skipped, equal-tick neighbours are joined).
	static void InsertTickRangeJoined(std::map<uint64_t, WriteTick>& ticks, uint64_t start, uint64_t end, uint64_t tick);
	[[nodiscard]] static uint64_t QueryTicks(const std::map<uint64_t, WriteTick>& ticks, uint64_t start, uint64_t end);
	// KYTY_PERF_EXACT_WALKER / KYTY_PERF_EXACT_FAULTS (cache/exactTicks.h): the exact write ticks inside the learned
	// granules (GPU thread only), the learned granules (WriteTickKey, sorted; with the learn clock of their last read),
	// and set when a write was noted off the GPU thread (the ticks are then not used).
	std::map<uint64_t, WriteTick>                 m_exact_ticks;
	std::vector<std::pair<uint64_t, uint64_t>>    m_exact_granules;
	uint64_t                                      m_exact_clock = 0;
	std::atomic<bool>                             m_exact_broken {false};
	// [start, end) loses its entries (the parts of entries outside it keep their ticks).
	static void EraseTickRange(std::map<uint64_t, WriteTick>& ticks, uint64_t start, uint64_t end);
	void ExactLearn(uint64_t vaddr, uint64_t size);
	void ExactNote(uint64_t vaddr, uint64_t size, uint64_t tick);
	struct ExactInterval {
		uint64_t start = 0, end = 0;
		uint64_t tick  = 0; // exact, else the granules' newest; UINT64_MAX: unknown
		bool     need  = false; // overlaps the bytes the read needs
	};
	// The GPU-dirty intervals of the GPU-modified pages of [lo, hi), with `need` for those overlapping [n_lo, n_hi).
	// False: one lies outside the buffer.
	bool ExactScan(Buffer& buffer, uint64_t lo, uint64_t hi, uint64_t n_lo, uint64_t n_hi, std::vector<ExactInterval>& out);
	// The usual path's alternative for a read of [vaddr, vaddr + size) (whole_pages: a guest fault, its pages). True: the
	// needed bytes are published; `published` holds what was published (address order). False: nothing changed.
	bool TryExactDownload(Buffer& buffer, uint64_t vaddr, uint64_t size, bool whole_pages,
	                      std::vector<std::pair<uint64_t, uint64_t>>& published);
	// The tick such a read of a guest fault's pages would wait for (PlanCallerWait); false: the usual plan.
	bool ExactFaultTick(Buffer& buffer, uint64_t vaddr, uint64_t size, uint64_t& tick);
	// After TryExactDownload: the pages of `published` without GPU-dirty bytes are unmarked, except [keep_lo, keep_hi).
	void ExactUnmark(const std::vector<std::pair<uint64_t, uint64_t>>& published, uint64_t keep_lo, uint64_t keep_hi);
	// Ranges whose granule write ticks were raised to `tick` (NoteGpuWriteTick); the granule
	// values only rise, so a range inside one of them at the same tick needs no new loop.
	struct GranuleNote {
		uint64_t start = 0;
		uint64_t end   = 0;
		uint64_t tick  = 0;
	};
	std::array<GranuleNote, 8> m_granule_notes {};
	uint32_t                   m_granule_note_next = 0;
	[[nodiscard]] std::optional<uint64_t> GranuleWriteTick(uint64_t key) const; // by WriteTickKey; empty: never written
	// BAR reads: GPU-dirty bytes of a mapped buffer whose writes all have a known, submitted
	// write tick are copied from the mapping after waiting for that tick only. 1: copied and
	// published, -1: not possible (use the GPU copy path).
	int TryBarDownload(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// KYTY_PERF_COUNT_EXACT: ReadGpuCopy's read by the bytes' own write ticks (false: not served), and its recorded copy.
	bool ReadCountExact(Buffer& buffer, uint64_t vaddr, uint64_t size, void* out);
	// KYTY_PERF_COUNT_GUEST: ReadGpuCopy's bytes from guest memory when guest memory holds what the buffer holds after all
	// recorded work (false: not served).
	bool ReadCountGuest(uint64_t vaddr, uint64_t size, void* out);
	bool ReadGpuCopyFrom(Buffer& buffer, uint64_t vaddr, uint64_t size, void* out);
	bool ReadGpuCopyRecorded(Buffer& buffer, uint64_t vaddr, uint64_t size, void* out);
	void ServeWhileWaiting(uint64_t target); // KYTY_PERF_SERVE_IN_WAIT
	// The range a CPU read of [vaddr, vaddr + size) downloads (ReadMemory, PlanCallerWait).
	[[nodiscard]] std::pair<uint64_t, uint64_t> ReadRange(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// Its publish of the intervals from span reads of the mapping.
	void BarBatchRead(const uint8_t* mapped, uint64_t base, const std::vector<std::pair<uint64_t, uint64_t>>& intervals);
	// Its scan: the GPU-dirty intervals of the GPU-modified pages of [vaddr, vaddr + size), the newest 64 KiB granule
	// write tick of their pages and their size. -1: usable; 1 a page without a write tick, 5 outside the buffer,
	// 6 nothing GPU-written (TryBarDownload's outcomes).
	int ScanBarDownload(Buffer& buffer, uint64_t vaddr, uint64_t size, std::vector<std::pair<uint64_t, uint64_t>>& intervals,
	                    uint64_t& newest, uint64_t& total);
	// ReadRange's question (7.11): ScanBarDownload usable, at most BarMaxBytes and its newest write tick
	// (and `ready_tick`) before the current tick - answered without the intervals, stopping at the first fact that makes
	// it false, one write-tick question per 64 KiB granule.
	[[nodiscard]] bool BarServable(Buffer& buffer, uint64_t vaddr, uint64_t size, uint64_t ready_tick);
	// The GPU-written bytes a read takes from the host mapping (BAR, uncached PCIe reads, ~0.66 GB/s measured 18.10 with PERF
	// bar_copy) at most; more go through the GPU copy into cached host memory (the copy path), which is faster for them.
	// Only how the same bytes are copied changes.
	static constexpr uint64_t BarMaxBytes = uint64_t {1024} << 10u;
	// Writable ranges obtained since the last recorded draw/dispatch (write tick not final yet).
	std::vector<std::pair<uint64_t, uint64_t>> m_bar_pending_writes;
	static constexpr uint64_t WriteTickKey(uint64_t address) noexcept { return address >> 16u; }
	// The granule write ticks in two levels - 2^16 blocks
	// of 4096 granules (256 MiB of guest addresses each, 32 KiB), a block allocated at its first write, consecutive
	// granules side by side; 0 = never written (ticks start at 1). Keys beyond the tracked address space go to `high`.
	class GranuleTickTable {
	public:
		static constexpr uint32_t kBlockBits = 12;
		static constexpr uint32_t kTopBits   = 16;
		[[nodiscard]] uint64_t Get(uint64_t key) const {
			if ((key >> (kBlockBits + kTopBits)) != 0u) {
				const auto it = m_high.find(key);
				return it != m_high.end() ? it->second : 0u;
			}
			if (m_top == nullptr) return 0u;
			const auto& block = m_top[key >> kBlockBits];
			return block != nullptr ? block[key & ((1u << kBlockBits) - 1u)] : 0u;
		}
		// The granule raised to `tick` (inserted when it had none).
		void Raise(uint64_t key, uint64_t tick) {
			uint64_t* value = Slot(key);
			if (*value < tick) *value = tick;
		}
		// Raise for every key of [first, last].
		void RaiseRange(uint64_t first, uint64_t last, uint64_t tick) {
			constexpr uint64_t mask = (uint64_t {1} << kBlockBits) - 1u;
			for (uint64_t key = first; key <= last;) {
				if ((key >> (kBlockBits + kTopBits)) != 0u) {
					Raise(key, tick);
					if (key == UINT64_MAX) break;
					key++;
					continue;
				}
				uint64_t* const block = Slot(key) - (key & mask);
				const uint64_t  end   = std::min(last, key | mask);
				for (; key <= end; key++) {
					uint64_t& value = block[key & mask];
					if (value < tick) value = tick;
				}
			}
		}
	private:
		uint64_t* Slot(uint64_t key) {
			if ((key >> (kBlockBits + kTopBits)) != 0u) return &m_high[key];
			if (m_top == nullptr) m_top = std::make_unique<std::unique_ptr<uint64_t[]>[]>(size_t {1} << kTopBits);
			auto& block = m_top[key >> kBlockBits];
			if (block == nullptr) block = std::make_unique<uint64_t[]>(size_t {1} << kBlockBits); // (zeros)
			return &block[key & ((1u << kBlockBits) - 1u)];
		}
		std::unique_ptr<std::unique_ptr<uint64_t[]>[]> m_top;
		std::unordered_map<uint64_t, uint64_t>         m_high;
	};
	GranuleTickTable m_granule_ticks;
	std::unordered_map<uint64_t, uint64_t> m_buffer_ready_tick;
	uint64_t                   m_hot_clock = 0;
};

// Project 15 N4.2b: one buffer acquisition a draw's plan peeked ahead (PeekBuffer), keyed by the acquisition's arguments:
// kind 0 a stage buffer, 1 a record-keyed range (BDA), 2 a merged vertex range, 3 the index buffer.
struct PlanBuffer {
	uint64_t                vaddr = 0, size = 0;
	bool                    texel = false, bda = false, used = false;
	uint8_t                 kind     = 0;
	uint32_t                distance = 0;
	BufferCache::BufferPeek peek;
	// P1 (kind 0): the stage's program and resource, and the binding computed on the helper (ready: the storage range's
	// limits hold) - StorageBufferRange's aligned offset and range, the offset byte, the record stride
	const void* program  = nullptr;
	uint32_t    resource = UINT32_MAX;
	bool        ready    = false;
	uint64_t    bind_offset = 0, bind_range = 0;
	uint32_t    memory_offset = 0, stride = 0;
	// P1.2: the resolve record's source (its buffer id and the epochs BufferSourceHolds checks)
	BufferId    source_id {};
	uint64_t    source_epoch = 0, ranges_generation = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
