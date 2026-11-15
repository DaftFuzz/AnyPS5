#include "graphics/host_gpu/renderer/hTrace.h"

#include "graphics/guest_gpu/graphicsRun.h"
#include "kernel/pthread.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep
#endif

namespace Libs::Graphics::HTrace {

namespace {

struct Counter {
	std::atomic<uint64_t> count {0};
	std::atomic<uint64_t> us {0};
	void                  Add(int64_t us_spent) {
        count.fetch_add(1, std::memory_order_relaxed);
        us.fetch_add(static_cast<uint64_t>(std::max<int64_t>(us_spent, 0)), std::memory_order_relaxed);
	}
	std::pair<uint64_t, uint64_t> Take() {
		return {count.exchange(0, std::memory_order_relaxed), us.exchange(0, std::memory_order_relaxed)};
	}
};

struct State {
	std::atomic<int64_t>  ready_us {0};
	std::atomic<int64_t>  present_us {0};
	std::atomic<int64_t>  done_us {0};
	std::atomic<int64_t>  first_q0_after_ready {0};
	std::atomic<int64_t>  first_q0_after_done {0};
	std::atomic<int64_t>  first_any_after_ready {0};
	std::atomic<int64_t>  last_q0r {-1};
	std::atomic<int64_t>  last_anyr {-1};
	std::atomic<uint64_t> q0_submits {0};
	std::atomic<uint64_t> other_submits {0};
	std::atomic<uint64_t> flips {0};
	Counter               fault_guest;
	Counter               fault_exec;
	Counter               fault_other;
	Counter               fault_outer;
	Counter               fault_outer_write;
	Counter               flip_wait;
	std::atomic<uint64_t> pac_ranges {0};
	std::atomic<uint64_t> pac_changed {0};
	std::mutex                             pages_mutex;
	std::unordered_map<uint64_t, uint32_t> pages;
	int64_t                                pages_report_us = 0;
	struct FaultPage {
		uint32_t faults = 0;
		uint32_t writes = 0;
		uint32_t frames = 0;
		uint64_t last_flip = 0;
	};
	std::mutex                              fault_pages_mutex;
	std::unordered_map<uint64_t, FaultPage> fault_pages;
	uint64_t                                fault_pages_first_flip = 0;
	int64_t                                 fault_pages_report_us  = 0;
};

State& S() {
	static auto* state = new State;
	return *state;
}

void SetFirst(std::atomic<int64_t>& slot, int64_t now) {
	int64_t expected = 0;
	(void)slot.compare_exchange_strong(expected, now, std::memory_order_relaxed);
}

int64_t Since(int64_t later, int64_t earlier) {
	return later != 0 && earlier != 0 ? later - earlier : -1;
}

} // namespace

bool Enabled() {
	static const bool on = [] {
		const char* value   = std::getenv("KYTY_TRACE_H");
		const bool  enabled = value != nullptr && value[0] == '1';
		if (enabled) std::fprintf(stderr, "TRACE_H enabled\n");
		return enabled;
	}();
	return on;
}

int64_t NowUs() {
#if defined(_WIN32)
	static const int64_t frequency = [] {
		LARGE_INTEGER value {};
		QueryPerformanceFrequency(&value);
		return static_cast<int64_t>(value.QuadPart);
	}();
	LARGE_INTEGER counter {};
	QueryPerformanceCounter(&counter);
	return static_cast<int64_t>(static_cast<long double>(counter.QuadPart) * 1000000.0L / static_cast<long double>(frequency));
#else
	return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
#endif
}

void NoteSubmit(uint32_t queue) {
	if (!Enabled()) return;
	auto&      s   = S();
	const auto now = NowUs();
	if (queue == 0) {
		s.q0_submits.fetch_add(1, std::memory_order_relaxed);
		if (s.ready_us.load(std::memory_order_relaxed) != 0) SetFirst(s.first_q0_after_ready, now);
		if (s.done_us.load(std::memory_order_relaxed) != 0) SetFirst(s.first_q0_after_done, now);
	} else {
		s.other_submits.fetch_add(1, std::memory_order_relaxed);
	}
	if (s.ready_us.load(std::memory_order_relaxed) != 0) SetFirst(s.first_any_after_ready, now);
}

void NoteFlipReady() {
	if (!Enabled()) return;
	auto&      s        = S();
	const auto now      = NowUs();
	const auto previous = s.ready_us.exchange(now, std::memory_order_relaxed);
	// The previous frame's link: its ready to the first submissions after it (they came before this ready).
	s.last_q0r.store(Since(s.first_q0_after_ready.exchange(0, std::memory_order_relaxed), previous), std::memory_order_relaxed);
	s.last_anyr.store(Since(s.first_any_after_ready.exchange(0, std::memory_order_relaxed), previous), std::memory_order_relaxed);
}

void NoteFlipPresentBegin() {
	if (!Enabled()) return;
	S().present_us.store(NowUs(), std::memory_order_relaxed);
}

void NoteFlipDone() {
	if (!Enabled()) return;
	auto&      s        = S();
	const auto now      = NowUs();
	const auto ready    = s.ready_us.load(std::memory_order_relaxed);
	const auto present  = s.present_us.load(std::memory_order_relaxed);
	const auto previous = s.done_us.exchange(now, std::memory_order_relaxed);
	// q0d: the first queue-0 submission after the previous done, if it came before this one.
	const auto q0_done  = s.first_q0_after_done.exchange(0, std::memory_order_relaxed);
	const auto q0_ready = s.last_q0r.load(std::memory_order_relaxed);
	const auto any      = s.last_anyr.load(std::memory_order_relaxed);
	const auto fg       = s.fault_guest.Take();
	const auto fe       = s.fault_exec.Take();
	const auto fo       = s.fault_other.Take();
	const auto fx       = s.fault_outer.Take();
	const auto fxw      = s.fault_outer_write.Take();
	const auto fw       = s.flip_wait.Take();
	const auto flip     = s.flips.fetch_add(1, std::memory_order_relaxed) + 1;
	std::fprintf(stderr,
	             "HT f=%llu t=%lld rd=%lld rp=%lld pd=%lld q0r=%lld q0d=%lld anyr=%lld sub=%llu/%llu fg=%llu/%llu fe=%llu/%llu "
	             "fo=%llu/%llu fx=%llu/%llu fxw=%llu/%llu fw=%llu/%llu pac=%llu/%llu\n",
	             static_cast<unsigned long long>(flip), static_cast<long long>(now), static_cast<long long>(Since(now, ready)),
	             static_cast<long long>(Since(present, ready)), static_cast<long long>(Since(now, present)),
	             static_cast<long long>(q0_ready), static_cast<long long>(Since(q0_done, previous)),
	             static_cast<long long>(any),
	             static_cast<unsigned long long>(s.q0_submits.exchange(0, std::memory_order_relaxed)),
	             static_cast<unsigned long long>(s.other_submits.exchange(0, std::memory_order_relaxed)),
	             static_cast<unsigned long long>(fg.first), static_cast<unsigned long long>(fg.second),
	             static_cast<unsigned long long>(fe.first), static_cast<unsigned long long>(fe.second),
	             static_cast<unsigned long long>(fo.first), static_cast<unsigned long long>(fo.second),
	             static_cast<unsigned long long>(fx.first), static_cast<unsigned long long>(fx.second),
	             static_cast<unsigned long long>(fxw.first), static_cast<unsigned long long>(fxw.second),
	             static_cast<unsigned long long>(fw.first), static_cast<unsigned long long>(fw.second),
	             static_cast<unsigned long long>(s.pac_changed.exchange(0, std::memory_order_relaxed)),
	             static_cast<unsigned long long>(s.pac_ranges.exchange(0, std::memory_order_relaxed)));
	std::vector<std::pair<uint64_t, uint32_t>> top;
	{
		std::scoped_lock lock(s.pages_mutex);
		if (now - s.pages_report_us >= 5000000 && !s.pages.empty()) {
			s.pages_report_us = now;
			top.assign(s.pages.begin(), s.pages.end());
			s.pages.clear();
		}
	}
	if (!top.empty()) {
		std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
		std::fprintf(stderr, "HT_PAC pages=%zu", top.size());
		for (size_t i = 0; i < top.size() && i < 12u; i++) {
			std::fprintf(stderr, " 0x%llx0000x%u", static_cast<unsigned long long>(top[i].first), top[i].second);
		}
		std::fprintf(stderr, "\n");
	}
	std::vector<std::pair<uint64_t, State::FaultPage>> fault_top;
	uint64_t                                           window_frames = 0;
	{
		std::scoped_lock lock(s.fault_pages_mutex);
		if (s.fault_pages_first_flip == 0) {
			s.fault_pages_first_flip = flip;
			s.fault_pages_report_us  = now;
		} else if (now - s.fault_pages_report_us >= 5000000) {
			window_frames            = flip - s.fault_pages_first_flip;
			s.fault_pages_first_flip = flip;
			s.fault_pages_report_us  = now;
			fault_top.assign(s.fault_pages.begin(), s.fault_pages.end());
			s.fault_pages.clear();
		}
	}
	if (window_frames != 0) {
		uint64_t faults = 0;
		uint64_t every  = 0;
		uint64_t every_faults = 0;
		uint64_t band_pages[4] {};
		uint64_t band_faults[4] {};
		uint64_t band_writes[4] {};
		for (const auto& [page, stat]: fault_top) {
			faults += stat.faults;
			if (stat.frames * 10u >= window_frames * 9u) {
				every++;
				every_faults += stat.faults;
			}
			const uint64_t share = stat.frames * 100u / window_frames;
			const int      band  = share >= 90u ? 0 : share >= 40u ? 1 : share >= 10u ? 2 : 3;
			band_pages[band]++;
			band_faults[band] += stat.faults;
			band_writes[band] += stat.writes;
		}
		std::fprintf(stderr, "HT_FAULT_BANDS frames=%llu", static_cast<unsigned long long>(window_frames));
		static constexpr const char* BandNames[4] = {"ge90", "40to90", "10to40", "lt10"};
		for (int band = 0; band < 4; band++) {
			std::fprintf(stderr, " %s=pages:%llu,faults:%llu,writes:%llu", BandNames[band], static_cast<unsigned long long>(band_pages[band]),
			             static_cast<unsigned long long>(band_faults[band]), static_cast<unsigned long long>(band_writes[band]));
		}
		std::fprintf(stderr, "\n");
		std::sort(fault_top.begin(), fault_top.end(), [](const auto& a, const auto& b) { return a.second.faults > b.second.faults; });
		std::fprintf(stderr, "HT_FAULT_PAGES frames=%llu pages=%zu faults=%llu every_frame_pages=%llu every_frame_faults=%llu top",
		             static_cast<unsigned long long>(window_frames), fault_top.size(), static_cast<unsigned long long>(faults),
		             static_cast<unsigned long long>(every), static_cast<unsigned long long>(every_faults));
		for (size_t i = 0; i < fault_top.size() && i < 24u; i++) {
			std::fprintf(stderr, " 0x%llx:%u/%u:w%u", static_cast<unsigned long long>(fault_top[i].first << 12u),
			             fault_top[i].second.faults, fault_top[i].second.frames, fault_top[i].second.writes);
		}
		std::fprintf(stderr, "\n");
	}
}

void NoteFaultPage(uint64_t address, bool write) {
	if (!Enabled()) return;
	auto&            s    = S();
	const auto       flip = s.flips.load(std::memory_order_relaxed) + 1;
	std::scoped_lock lock(s.fault_pages_mutex);
	auto&            page = s.fault_pages[address >> 12u];
	page.faults++;
	if (write) page.writes++;
	if (page.last_flip != flip) {
		page.last_flip = flip;
		page.frames++;
	}
}

void NoteFault(FaultLayer layer, bool write, int64_t start_us) {
	if (!Enabled()) return;
	auto&      s     = S();
	const auto spent = NowUs() - start_us;
	if (layer == FaultLayer::Outer) {
		s.fault_outer.Add(spent);
		if (write) s.fault_outer_write.Add(spent);
	} else if (GuestGpu::IsGpuThread()) {
		s.fault_exec.Add(spent);
	} else if (LibKernel::PthreadSelfOrNull() != nullptr) {
		s.fault_guest.Add(spent);
	} else {
		s.fault_other.Add(spent);
	}
}

void NoteFlipWait(int64_t start_us) {
	if (!Enabled()) return;
	S().flip_wait.Add(NowUs() - start_us);
}

void NotePacRanges(uint64_t ranges) {
	if (!Enabled()) return;
	S().pac_ranges.fetch_add(ranges, std::memory_order_relaxed);
}

void NotePacChanged(uint64_t address, uint64_t size) {
	if (!Enabled()) return;
	auto& s = S();
	s.pac_changed.fetch_add(1, std::memory_order_relaxed);
	std::scoped_lock lock(s.pages_mutex);
	for (uint64_t page = address >> 16u; page <= ((address + std::max<uint64_t>(size, 1u) - 1u) >> 16u); page++) {
		s.pages[page]++;
	}
}

void NotePacSample(uint64_t address, uint64_t size, uint64_t offset, uint64_t before, uint64_t after) {
	static std::atomic<uint32_t> printed {0};
	if (!Enabled() || printed.fetch_add(1, std::memory_order_relaxed) >= 4000u) return;
	std::fprintf(stderr, "HT_PAC_SAMPLE addr=0x%llx size=0x%llx off=0x%llx before=%016llx after=%016llx\n",
	             static_cast<unsigned long long>(address), static_cast<unsigned long long>(size),
	             static_cast<unsigned long long>(offset), static_cast<unsigned long long>(before),
	             static_cast<unsigned long long>(after));
}

} // namespace Libs::Graphics::HTrace
