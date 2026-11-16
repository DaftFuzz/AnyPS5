#ifndef KYTY_COMMON_INORDERTRIM_H_
#define KYTY_COMMON_INORDERTRIM_H_

#include "common/envRead.h"
#include "common/liveAb.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace Common::InorderTrim {

[[nodiscard]] inline bool Check() {
	static const bool on = [] {
		const char* text = std::getenv("KYTY_CHECK_INORDER_TRIM");
		return text != nullptr && text[0] == '2' && text[1] == '\0';
	}();
	return on;
}

[[nodiscard]] inline uint32_t Fixed() {
	static const uint32_t value = [] {
		const char*    text = std::getenv("KYTY_PERF_INORDER_TRIM");
		const uint32_t v    = text != nullptr ? static_cast<uint32_t>(std::strtoul(text, nullptr, 10)) : 0u;
		if (v != 0u || Check()) {
			std::fprintf(stderr, "KYTY_PERF_INORDER_TRIM=%u: null-share table writes, GPU-dirty marks and per-operation clock reads trimmed%s\n", v,
			             Check() ? " (KYTY_CHECK_INORDER_TRIM=2: checked, not used)" : "");
		}
		return v;
	}();
	return value;
}

[[nodiscard]] inline bool Possible() {
	static const bool possible = Check() || LiveAb::Possible(LiveAb::Switch::InorderTrim, Fixed());
	return possible;
}

[[nodiscard]] inline bool On() {
	return Possible() && !Check() && LiveAb::Gate(LiveAb::Switch::InorderTrim, Fixed()) != 0u;
}

[[nodiscard]] inline uint32_t FaultEvery() {
	static const uint32_t every = [] {
		const char* text = std::getenv("KYTY_CHECK_INORDER_TRIM_FAULT");
		return text != nullptr ? static_cast<uint32_t>(std::strtoul(text, nullptr, 10)) : 997u;
	}();
	return every;
}

struct Stats {
	std::atomic<uint64_t> share_slots {0}, share_hits {0}, share_written {0}, share_compared {0}, share_bad {0};
	std::atomic<uint64_t> dirty_adds {0}, dirty_contained {0}, dirty_foreign {0}, dirty_compared {0}, dirty_bad {0};
	std::atomic<uint64_t> injected {0}, detected {0}, clock_skipped {0}, clock_read {0};
};

inline Stats g_stats;

[[nodiscard]] inline bool Inject(uint64_t compared) {
	const uint32_t every = FaultEvery();
	return every != 0u && compared % every == every - 1u;
}

inline void NoteClock(bool skipped) {
	(skipped ? g_stats.clock_skipped : g_stats.clock_read).fetch_add(1, std::memory_order_relaxed);
}

[[nodiscard]] inline bool ClockDue(uint32_t& counter, uint32_t mask) {
	if (!On()) return true;
	const bool due = (counter++ & mask) == 0u;
	NoteClock(!due);
	return due;
}

inline void Report() {
	if (!Possible() || !(EnvRead::PerfLines() || Check())) return;
	thread_local uint32_t calls = 0;
	if ((calls++ & 1023u) != 0u) return;
	static std::chrono::steady_clock::time_point window = std::chrono::steady_clock::now();
	const auto                                   now    = std::chrono::steady_clock::now();
	if (now - window < std::chrono::seconds(5)) return;
	window          = now;
	const auto load = [](const std::atomic<uint64_t>& a) { return static_cast<unsigned long long>(a.load(std::memory_order_relaxed)); };
	const auto& s   = g_stats;
	std::fprintf(stderr,
	             "PERF inorder_trim mode=%s share[slots|hits|written]=%llu|%llu|%llu dirty[adds|contained|foreign]=%llu|%llu|%llu "
	             "check[share_compared|share_bad|dirty_compared|dirty_bad]=%llu|%llu|%llu|%llu fault[injected|detected]=%llu|%llu "
	             "clock[skipped|read]=%llu|%llu\n",
	             Check() ? "check" : (On() ? "on" : "off"), load(s.share_slots), load(s.share_hits), load(s.share_written),
	             load(s.dirty_adds), load(s.dirty_contained), load(s.dirty_foreign), load(s.share_compared), load(s.share_bad),
	             load(s.dirty_compared), load(s.dirty_bad), load(s.injected), load(s.detected), load(s.clock_skipped),
	             load(s.clock_read));
}

} // namespace Common::InorderTrim

#endif // KYTY_COMMON_INORDERTRIM_H_
