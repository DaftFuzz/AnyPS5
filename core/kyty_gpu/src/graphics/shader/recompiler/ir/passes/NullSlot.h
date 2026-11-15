#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_NULLSLOT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_NULLSLOT_H_

#include <array>
#include <cstdint>

namespace Libs::Graphics::ShaderRecompiler::IR::NullSlot {

inline constexpr uint32_t RuntimeFormat = 0x100u;
inline constexpr uint32_t FormatField   = 12u;
inline constexpr uint32_t FormatBits    = 7u;

using Value = std::array<uint32_t, 2>;

constexpr bool IsRuntime(const Value& value) {
	return value[0] == RuntimeFormat;
}

constexpr bool SwizzleResolvable(uint32_t component_count, uint32_t swizzle) {
	if (component_count == 0u || (swizzle >> 12u) != 0u) return false;
	for (uint32_t channel = 0; channel < 4u; channel++) {
		const uint32_t selector = (swizzle >> (3u * channel)) & 7u;
		if (selector == 2u || selector == 3u) return false;
	}
	return true;
}

enum class Source : uint8_t { None, Own, Live, Seed, Runtime };

struct Decision {
	Source source = Source::Own;
	Value  value {0u, 0u};
};

struct Guesses {
	Value precise {0u, 0u};
	bool  precise_usable = false;
	Value coarse {0u, 0u};
	bool  coarse_usable = false;
};

constexpr Decision DecideNull(const Value& live, const Guesses& guesses, bool seed, bool runtime) {
	if (live[0] != 0u) return {IsRuntime(live) ? Source::Runtime : Source::Live, live};
	if (seed && guesses.precise_usable && guesses.precise[0] != 0u) return {Source::Seed, guesses.precise};
	if (runtime) return {Source::Runtime, {RuntimeFormat, 0u}};
	if (seed && guesses.coarse_usable && guesses.coarse[0] != 0u) return {Source::Seed, guesses.coarse};
	return {};
}

enum class Transition : uint8_t { None, Caused, Avoided };

constexpr Transition Classify(Source pending, const Value& taken, const Value& live_value) {
	switch (pending) {
		case Source::Own: return Transition::Caused;
		case Source::Seed: return taken == live_value ? Transition::Avoided : Transition::Caused;
		case Source::Runtime: return Transition::Avoided;
		default: return Transition::None;
	}
}

constexpr uint32_t RuntimeFormatOf(uint32_t word3) {
	return (word3 >> FormatField) & ((1u << FormatBits) - 1u);
}

} // namespace Libs::Graphics::ShaderRecompiler::IR::NullSlot

#endif
