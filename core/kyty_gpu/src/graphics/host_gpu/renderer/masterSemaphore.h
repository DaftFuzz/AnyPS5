#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <atomic>

namespace Libs::Graphics {

struct GraphicContext;

class MasterSemaphore {
public:
	explicit MasterSemaphore(GraphicContext& graphics);
	~MasterSemaphore();
	KYTY_CLASS_NO_COPY(MasterSemaphore);

	[[nodiscard]] uint64_t CurrentTick() const noexcept {
		return m_current_tick.load(std::memory_order_acquire);
	}
	[[nodiscard]] uint64_t KnownGpuTick() const noexcept {
		return m_gpu_tick.load(std::memory_order_acquire);
	}
	[[nodiscard]] bool     IsFree(uint64_t tick) const noexcept { return KnownGpuTick() >= tick; }
	[[nodiscard]] uint64_t NextTick() noexcept {
		return m_current_tick.fetch_add(1, std::memory_order_release);
	}
	[[nodiscard]] vk::Semaphore Handle() const noexcept { return m_semaphore; }
	// The newest tick whose signalling submission vkQueueSubmit has accepted (the recorder's worker submits later than
	// the tick is taken). A submission on another queue that waits for a tick must not run ahead of it: the driver may
	// hold queue operations behind that wait while the signalling submission waits for the queue lock.
	[[nodiscard]] uint64_t QueueSubmittedTick() const noexcept { return m_queue_submitted.load(std::memory_order_acquire); }
	void                   NoteQueueSubmitted(uint64_t tick) noexcept {
		uint64_t seen = m_queue_submitted.load(std::memory_order_relaxed);
		while (seen < tick && !m_queue_submitted.compare_exchange_weak(seen, tick, std::memory_order_release, std::memory_order_relaxed)) {
		}
	}

	// Refresh queries the timeline semaphore (a kernel call) at most once per 100 microseconds; in between it keeps the
	// last known GPU tick, which only makes IsFree conservative for that long. RefreshNow and Wait always query.
	void Refresh();
	void RefreshNow();
	void Wait(uint64_t tick);

	// KYTY_TRACE_PERF (PERF semaphore): semaphore queries (calls/ns) by caller: 0 Refresh, 1 Wait before blocking, 2 Wait
	// after blocking, 3 other; blocking waits (calls/ns). GPU thread only; reset by TakeStats. polls/poll_ns: the GPU
	// thread's waits served by KYTY_PERF_GPU_POLL.
	struct Stats {
		std::atomic<uint64_t> queries[4] {}, query_ns[4] {}, blocks {0}, block_ns {0}, polls {0}, poll_ns {0};
	};
	static Stats& GetStats();

	// KYTY_PERF_GPU_POLL=1 (default 0, off; live A/B switch; lane v4e2 of AnyPS5's PLAN v4): the GPU thread waits for the
	// GPU by polling (here the timeline value, vkGetSemaphoreCounterValue, ~0.1 us on the NVIDIA driver; the BAR copy its
	// fence, vkGetFenceStatus) with a pause backoff of at most ~2 us, instead of blocking in the driver until the OS wakes
	// it. Only how the thread waits changes; the condition it waits for is the same. A wait still unmet after 100 ms
	// blocks as before. True on the GPU thread while the switch is on (read once per wait).
	[[nodiscard]] static bool PollWaits() noexcept;

private:
	GraphicContext&       m_graphics;
	vk::Semaphore         m_semaphore = nullptr;
	std::atomic<uint64_t> m_gpu_tick {0};
	std::atomic<uint64_t> m_current_tick {1};
	std::atomic<uint64_t> m_queue_submitted {0};
	std::atomic<int64_t>  m_last_refresh_ns {0};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_
