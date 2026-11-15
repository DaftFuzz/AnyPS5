#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/passes/NullSlot.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>

namespace Format   = Libs::Graphics::ShaderRecompiler::Format;
namespace NullSlot = Libs::Graphics::ShaderRecompiler::IR::NullSlot;
namespace Prospero = Libs::Graphics::Prospero;

namespace {

constexpr uint32_t RuntimeFormatCases = static_cast<uint32_t>(Prospero::BufferFormat::k32_32_32_32Float) + 1u;

void Require(bool condition, const std::string& message) {
	if (!condition) throw std::runtime_error(message);
}

Format::BufferFormatInfo Info(uint32_t format) {
	return Format::GetFormatInfo(static_cast<Prospero::BufferFormat>(format));
}

uint32_t Specialized(const Format::BufferFormatInfo& info, uint32_t selector, const std::array<uint32_t, 4>& memory) {
	const auto source = Format::ResolveFormattedSource(info, selector);
	if (source.kind == Format::FormattedSourceKind::Memory) return memory[source.component];
	return Format::FormattedConstantBits(info, source.kind);
}

uint32_t RuntimeClass(const Format::BufferFormatInfo& info) {
	return info.type == Format::ComponentType::Uint ? 0u : info.type == Format::ComponentType::Sint ? 1u : 2u;
}

uint32_t RuntimeFormula(const Format::BufferFormatInfo& info, uint32_t selector, const std::array<uint32_t, 4>& memory) {
	const uint32_t count = info.component_count;
	const uint32_t one   = RuntimeClass(info) < 2u ? 1u : 0x3f800000u;
	const uint32_t index = (selector - 4u) % count;
	uint32_t       picked = memory[0];
	for (uint32_t candidate = 1; candidate < 4u; candidate++) picked = index == candidate ? memory[candidate] : picked;
	const uint32_t constant = selector == 1u ? one : 0u;
	return selector >= 4u ? picked : constant;
}

uint32_t Pick(uint32_t word3) {
	const uint32_t format = NullSlot::RuntimeFormatOf(word3);
	return format < RuntimeFormatCases ? format : 0u;
}

void runtime_selection_matches_specialized() {
	const std::array<uint32_t, 4> memory {0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u};
	const std::array<uint32_t, 4> zeros {};
	for (uint32_t format = 0; format < 128u; format++) {
		const auto info = Info(format);
		if (info.type == Format::ComponentType::Unknown) continue;
		for (uint32_t selector = 0; selector < 8u; selector++) {
			if (selector == 2u || selector == 3u) continue;
			Require(RuntimeFormula(info, selector, memory) == Specialized(info, selector, memory),
			        "in-bounds channel differs, format " + std::to_string(format) + " selector " + std::to_string(selector));
			Require(RuntimeFormula(info, selector, zeros) == Specialized(info, selector, zeros),
			        "out-of-bounds channel differs, format " + std::to_string(format) + " selector " + std::to_string(selector));
		}
	}
}

void format_cases_cover_every_vsharp_format() {
	Require(Info(0).type == Format::ComponentType::Unknown, "format 0 is not the unknown case");
	for (uint32_t format = RuntimeFormatCases; format < 128u; format++) {
		Require(Info(format).type == Format::ComponentType::Unknown,
		        "format " + std::to_string(format) + " is known but has no case of its own");
	}
	for (uint32_t format = 0; format < 128u; format++) {
		const uint32_t word3 = format << NullSlot::FormatField | 0xfacu | 0x3u << 28u;
		const uint32_t pick  = Pick(word3);
		Require(pick < RuntimeFormatCases, "format " + std::to_string(format) + " picks a case past the switch");
		Require(Format::FormattedElementBits(Info(pick)) == Format::FormattedElementBits(Info(format)),
		        "the case of format " + std::to_string(format) + " uses another element width");
		Require(Info(pick).type == Info(format).type && Info(pick).byte_size == Info(format).byte_size,
		        "the case of format " + std::to_string(format) + " decodes another format");
	}
	Require(NullSlot::RuntimeFormat >= 1u << NullSlot::FormatBits, "the run-time format marker is a V# format");
	Require(NullSlot::RuntimeFormat != static_cast<uint32_t>(Prospero::BufferFormat::k8Srgb) &&
	            NullSlot::RuntimeFormat != static_cast<uint32_t>(Prospero::BufferFormat::k8_8Srgb),
	        "the run-time format marker is a host format");
	Require(Info(NullSlot::RuntimeFormat).type == Format::ComponentType::Unknown, "the run-time format marker has a layout");
}

void element_bits_follow_the_format() {
	Require(Format::FormattedElementBits(Info(static_cast<uint32_t>(Prospero::BufferFormat::k8UNorm))) == 8u, "k8UNorm");
	Require(Format::FormattedElementBits(Info(static_cast<uint32_t>(Prospero::BufferFormat::k8_8_8_8UNorm))) == 8u, "k8_8_8_8UNorm");
	Require(Format::FormattedElementBits(Info(static_cast<uint32_t>(Prospero::BufferFormat::k16_16Float))) == 16u, "k16_16Float");
	Require(Format::FormattedElementBits(Info(static_cast<uint32_t>(Prospero::BufferFormat::k10_10_10_2UNorm))) == 32u,
	        "k10_10_10_2UNorm");
	Require(Format::FormattedElementBits(Info(static_cast<uint32_t>(Prospero::BufferFormat::k32_32Float))) == 32u, "k32_32Float");
	Require(Format::FormattedElementBits(Info(0u)) == 32u, "unknown format");
}

void swizzle_guesses() {
	Require(NullSlot::SwizzleResolvable(4u, 0xfacu), "XYZW");
	Require(NullSlot::SwizzleResolvable(1u, 0x204u), "X001");
	Require(NullSlot::SwizzleResolvable(3u, 0u), "0000");
	Require(!NullSlot::SwizzleResolvable(0u, 0xfacu), "unknown format");
	for (uint32_t channel = 0; channel < 4u; channel++) {
		Require(!NullSlot::SwizzleResolvable(4u, 2u << (3u * channel)), "reserved selector 2");
		Require(!NullSlot::SwizzleResolvable(4u, 3u << (3u * channel)), "reserved selector 3");
	}
	Require(!NullSlot::SwizzleResolvable(4u, 0x1000u), "bits past DST_SEL_W");
}

void null_decisions() {
	const NullSlot::Value none {0u, 0u};
	const NullSlot::Value live {14u, 0x22cu};
	const NullSlot::Value other {1u, 0x204u};
	const NullSlot::Value runtime {NullSlot::RuntimeFormat, 0u};
	NullSlot::Guesses guesses {.precise = other, .precise_usable = true, .coarse = live, .coarse_usable = true};
	NullSlot::Guesses unusable {.precise = other, .precise_usable = false, .coarse = live, .coarse_usable = false};
	NullSlot::Guesses coarse_only {.coarse = live, .coarse_usable = true};
	const NullSlot::Guesses no_guess {};

	auto decision = NullSlot::DecideNull(live, guesses, true, true);
	Require(decision.source == NullSlot::Source::Live && decision.value == live, "a known live value comes first");
	decision = NullSlot::DecideNull(runtime, guesses, true, false);
	Require(decision.source == NullSlot::Source::Runtime && decision.value == runtime, "a run-time binding stays run-time");
	decision = NullSlot::DecideNull(none, guesses, false, false);
	Require(decision.source == NullSlot::Source::Own && decision.value == none, "both switches off: own specialization");
	decision = NullSlot::DecideNull(none, guesses, true, true);
	Require(decision.source == NullSlot::Source::Seed && decision.value == other, "a precise guess comes before run-time");
	decision = NullSlot::DecideNull(none, coarse_only, true, true);
	Require(decision.source == NullSlot::Source::Runtime && decision.value == runtime, "run-time comes before a coarse guess");
	decision = NullSlot::DecideNull(none, coarse_only, true, false);
	Require(decision.source == NullSlot::Source::Seed && decision.value == live, "a coarse guess without run-time");
	decision = NullSlot::DecideNull(none, unusable, true, false);
	Require(decision.source == NullSlot::Source::Own, "an unusable guess is not taken");
	decision = NullSlot::DecideNull(none, no_guess, false, true);
	Require(decision.source == NullSlot::Source::Runtime && NullSlot::IsRuntime(decision.value), "run-time alone");
	decision = NullSlot::DecideNull(none, guesses, false, true);
	Require(decision.source == NullSlot::Source::Runtime, "a guess is not taken with the seed switch off");
}

void transitions() {
	const NullSlot::Value guess {14u, 0x22cu};
	const NullSlot::Value live {1u, 0x204u};
	Require(NullSlot::Classify(NullSlot::Source::Own, {0u, 0u}, live) == NullSlot::Transition::Caused, "own null then live");
	Require(NullSlot::Classify(NullSlot::Source::Seed, guess, guess) == NullSlot::Transition::Avoided, "right guess");
	Require(NullSlot::Classify(NullSlot::Source::Seed, guess, live) == NullSlot::Transition::Caused, "wrong guess");
	Require(NullSlot::Classify(NullSlot::Source::Runtime, {NullSlot::RuntimeFormat, 0u}, live) == NullSlot::Transition::Avoided,
	        "run-time binding");
	Require(NullSlot::Classify(NullSlot::Source::Live, guess, live) == NullSlot::Transition::None, "live first");
	Require(NullSlot::Classify(NullSlot::Source::None, guess, live) == NullSlot::Transition::None, "no null seen");
}

} // namespace

int main() {
	try {
		runtime_selection_matches_specialized();
		format_cases_cover_every_vsharp_format();
		element_bits_follow_the_format();
		swizzle_guesses();
		null_decisions();
		transitions();
	} catch (const std::exception& error) {
		std::fprintf(stderr, "kyty_null_slot: %s\n", error.what());
		return 1;
	}
	std::printf("kyty_null_slot: passed\n");
	return 0;
}
