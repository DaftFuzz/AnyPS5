#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/envRead.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/startupTrace.h"
#include "common/shaderEnv.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/perfCounters.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderPrecompile.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/rectListShader.h"
#include "graphics/shader/shader.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

// PIPELINE_REPEAT (diagnostic): a driver compile (50 ms or more)
// of a VS+PS pair (shader hashes) already compiled in this session, with the inputs that differ from its closest earlier
// variant - which pipeline state makes the same shaders compile again (dynamic state or a pipeline library, stage 2).
struct PipelineVariant {
	uint64_t                 vs_id = 0, ps_id = 0; // the translated programs (another permutation of the same hash)
	uint64_t                 vs_module = 0, ps_module = 0; // their VkShaderModule handles
	uint64_t                 layout_hash = 0;              // descriptor set 0 bindings (binding, type, count, stages)
	bool                     rect_list = false;
	std::array<uint32_t, ShaderVertexInputInfo::RES_MAX> attribute_formats {};
	PipelineStaticParameters static_params;
	PipelineRenderingState   rendering;
	PipelineVertexInputState vertex_input;
};

std::string PipelineVariantDiff(const PipelineVariant& a, const PipelineVariant& b) {
	std::string out;
	const auto  add = [&](bool differs, const char* name) {
        if (!differs) return;
        if (!out.empty()) out += ',';
        out += name;
	};
	const auto& x       = a.static_params;
	const auto& y       = b.static_params;
	const auto  differs = [](const auto& p, const auto& q) { return std::memcmp(&p, &q, sizeof(p)) != 0; };
	add(a.vs_id != b.vs_id, "vs_program");
	add(a.vs_module != b.vs_module, "vs_module");
	add(a.ps_module != b.ps_module, "ps_module");
	add(a.layout_hash != b.layout_hash, "layout");
	add(a.rect_list != b.rect_list, "rect_list");
	add(a.attribute_formats != b.attribute_formats, "vertex_formats");
	add(a.ps_id != b.ps_id, "ps_program");
	add(x.topology != y.topology, "topology");
	add(x.primitive_restart_enable != y.primitive_restart_enable, "primitive_restart");
	add(x.cull_front != y.cull_front || x.cull_back != y.cull_back, "cull");
	add(x.face != y.face, "front_face");
	add(x.polygon_mode != y.polygon_mode, "polygon_mode");
	add(x.depth_clip_enable != y.depth_clip_enable, "depth_clip");
	add(x.negative_one_to_one != y.negative_one_to_one, "clip_space");
	add(x.provoking_vtx_last != y.provoking_vtx_last, "provoking");
	add(x.samples != y.samples, "samples");
	add(x.sample_shading_enable != y.sample_shading_enable, "sample_shading");
	add(x.alpha_to_coverage_enable != y.alpha_to_coverage_enable, "alpha_to_coverage");
	add(differs(x.color_mask, y.color_mask), "color_mask");
	add(differs(x.blend_enable, y.blend_enable), "blend_enable");
	add(differs(x.color_srcblend, y.color_srcblend) || differs(x.color_destblend, y.color_destblend) ||
	        differs(x.color_comb_fcn, y.color_comb_fcn) || differs(x.alpha_srcblend, y.alpha_srcblend) ||
	        differs(x.alpha_destblend, y.alpha_destblend) || differs(x.alpha_comb_fcn, y.alpha_comb_fcn) ||
	        differs(x.separate_alpha_blend, y.separate_alpha_blend),
	    "blend_equation");
	add(a.rendering.color_count != b.rendering.color_count, "color_count");
	add(a.rendering.color_formats != b.rendering.color_formats, "color_formats");
	add(a.rendering.depth_format != b.rendering.depth_format || a.rendering.stencil_format != b.rendering.stencil_format,
	    "depth_format");
	const auto& v = a.vertex_input;
	const auto& w = b.vertex_input;
	add(v.binding_count != w.binding_count || v.attribute_count != w.attribute_count, "vertex_counts");
	bool strides = false, rates = false, offsets = false;
	for (size_t i = 0; i < v.bindings.size(); i++) {
		strides |= v.bindings[i].stride != w.bindings[i].stride;
		rates |= v.bindings[i].instance != w.bindings[i].instance;
	}
	for (size_t i = 0; i < v.attributes.size(); i++) offsets |= !(v.attributes[i] == w.attributes[i]);
	add(strides, "vertex_strides");
	add(rates, "vertex_rates");
	add(offsets, "vertex_offsets");
	return out.empty() ? std::string("none") : out;
}

void NotePipelineVariant(uint64_t vs_hash, uint64_t ps_hash, const PipelineVariant& variant, double ms, bool background) {
	static std::mutex                                                           mutex;
	static std::map<std::pair<uint64_t, uint64_t>, std::vector<PipelineVariant>> seen;
	std::string                                                                 diff;
	size_t                                                                      earlier = 0;
	{
		std::lock_guard lock(mutex);
		auto&           list = seen[{vs_hash, ps_hash}];
		earlier              = list.size();
		size_t best          = SIZE_MAX;
		for (const auto& old: list) {
			auto   candidate = PipelineVariantDiff(variant, old);
			size_t fields    = static_cast<size_t>(std::count(candidate.begin(), candidate.end(), ',')) + 1;
			if (fields < best) {
				best = fields;
				diff = std::move(candidate);
			}
		}
		if (list.size() < 64) list.push_back(variant);
	}
	if (earlier == 0 || ms < 50.0) return;
	std::fprintf(stderr, "PIPELINE_REPEAT %s vs=0x%016llx ps=0x%016llx %.0fms earlier=%zu diff=%s\n",
	             background ? "graphics-background" : "graphics", static_cast<unsigned long long>(vs_hash),
	             static_cast<unsigned long long>(ps_hash), ms, earlier, diff.c_str());
}

} // namespace

// IDK: maybe we can remove it?
constexpr uint8_t kTemporaryVertexAttribFormat113 =
    static_cast<uint8_t>(Prospero::VertexAttribFormat::k16_16SInt);
constexpr uint32_t kTemporaryPs5BufferFormat121 = 121u;

static bool NarrowInputFormat(vk::Format& format, uint32_t& size, uint32_t used_components) {
	if (used_components == 0 || used_components >= size) {
		return false;
	}

	switch (format) {
		case vk::Format::eR32G32B32A32Sfloat:
			switch (used_components) {
				case 1: format = vk::Format::eR32Sfloat; break;
				case 2: format = vk::Format::eR32G32Sfloat; break;
				case 3: format = vk::Format::eR32G32B32Sfloat; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR32G32B32Sfloat:
			switch (used_components) {
				case 1: format = vk::Format::eR32Sfloat; break;
				case 2: format = vk::Format::eR32G32Sfloat; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR16G16B16A16Sfloat:
			switch (used_components) {
				case 1: format = vk::Format::eR16Sfloat; break;
				case 2: format = vk::Format::eR16G16Sfloat; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR8G8B8A8Unorm:
			switch (used_components) {
				case 1: format = vk::Format::eR8Unorm; break;
				case 2: format = vk::Format::eR8G8Unorm; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR8G8B8A8Snorm:
			if (used_components != 2) {
				return false;
			}
			format = vk::Format::eR8G8Snorm;
			size   = 2;
			return true;
		case vk::Format::eR8G8B8A8Uint:
			switch (used_components) {
				case 1: format = vk::Format::eR8Uint; break;
				case 2: format = vk::Format::eR8G8Uint; break;
				default: return false;
			}
			size = used_components;
			return true;
		default: break;
	}

	return false;
}

static void GetInputFormat(const ShaderBufferResource& res, vk::Format& format, uint32_t& size,
                           uint32_t used_components) {
	const auto fmt        = res.Format();
	const auto raw_format = res.RawFormat();
	if (raw_format == kTemporaryVertexAttribFormat113) {
		// Atomic: background compiles (shaderPrecompile.h) create pipelines too.
		static std::atomic_bool logged_113 {false};
		if (!logged_113.exchange(true, std::memory_order_relaxed)) {
			LOGF("InputFormat: temporary: accepting invalid PS5 buffer format 113 as "
			     "vk::Format::eR32G32B32A32Sfloat\n");
		}
		format = vk::Format::eR32G32B32A32Sfloat;
		size   = 4;
		if (NarrowInputFormat(format, size, used_components)) {
			LOGF("InputFormat: narrowing fmt=%u to %s for used_components=%u\n", raw_format,
			     vk::to_string(format).c_str(), used_components);
		}
		return;
	}
	if (raw_format == kTemporaryPs5BufferFormat121) {
		static std::atomic_bool logged_121 {false};
		if (!logged_121.exchange(true, std::memory_order_relaxed)) {
			LOGF("InputFormat: accepting PS5 buffer format 121 as vk::Format::eR16G16Sfloat\n");
		}
		format = vk::Format::eR16G16Sfloat;
		size   = 2;
		return;
	}

	format = VulkanFormat(fmt);
	size   = ShaderRecompiler::Format::GetFormatInfo(fmt).component_count;
	if (format == vk::Format::eUndefined || size == 0) {
		EXIT("unknown vertex format: fmt = %u\n", raw_format);
	}

	if (NarrowInputFormat(format, size, used_components)) {
		static std::atomic<uint64_t> log_count = 0;
		auto                         log_id    = log_count.fetch_add(1);
		if (log_id < 32) {
			LOGF("VertexInput: narrowed vertex format to %" PRIu32
			     " component(s) for shader fetch\n",
			     used_components);
		}
	}
}

static vk::BlendFactor GetBlendFactor(uint32_t factor) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kZero: return vk::BlendFactor::eZero;
		case Prospero::BlendFactor::kOne: return vk::BlendFactor::eOne;
		case Prospero::BlendFactor::kSrcColor: return vk::BlendFactor::eSrcColor;
		case Prospero::BlendFactor::kOneMinusSrcColor: return vk::BlendFactor::eOneMinusSrcColor;
		case Prospero::BlendFactor::kSrcAlpha: return vk::BlendFactor::eSrcAlpha;
		case Prospero::BlendFactor::kOneMinusSrcAlpha: return vk::BlendFactor::eOneMinusSrcAlpha;
		case Prospero::BlendFactor::kDstAlpha: return vk::BlendFactor::eDstAlpha;
		case Prospero::BlendFactor::kOneMinusDstAlpha: return vk::BlendFactor::eOneMinusDstAlpha;
		case Prospero::BlendFactor::kDstColor: return vk::BlendFactor::eDstColor;
		case Prospero::BlendFactor::kOneMinusDstColor: return vk::BlendFactor::eOneMinusDstColor;
		case Prospero::BlendFactor::kSrcAlphaSaturate: return vk::BlendFactor::eSrcAlphaSaturate;
		case Prospero::BlendFactor::kConstantColor: return vk::BlendFactor::eConstantColor;
		case Prospero::BlendFactor::kOneMinusConstantColor:
			return vk::BlendFactor::eOneMinusConstantColor;
		case Prospero::BlendFactor::kSrc1Color: return vk::BlendFactor::eSrc1Color;
		case Prospero::BlendFactor::kOneMinusSrc1Color: return vk::BlendFactor::eOneMinusSrc1Color;
		case Prospero::BlendFactor::kSrc1Alpha: return vk::BlendFactor::eSrc1Alpha;
		case Prospero::BlendFactor::kOneMinusSrc1Alpha: return vk::BlendFactor::eOneMinusSrc1Alpha;
		case Prospero::BlendFactor::kConstantAlpha: return vk::BlendFactor::eConstantAlpha;
		case Prospero::BlendFactor::kOneMinusConstantAlpha:
			return vk::BlendFactor::eOneMinusConstantAlpha;
		default: EXIT("unknown factor: %u\n", factor);
	}
	return vk::BlendFactor::eZero;
}

static vk::BlendOp GetBlendOp(uint32_t op) {
	switch (static_cast<Prospero::BlendOp>(op)) {
		case Prospero::BlendOp::kAdd: return vk::BlendOp::eAdd;
		case Prospero::BlendOp::kSubtract: return vk::BlendOp::eSubtract;
		case Prospero::BlendOp::kMin: return vk::BlendOp::eMin;
		case Prospero::BlendOp::kMax: return vk::BlendOp::eMax;
		case Prospero::BlendOp::kReverseSubtract: return vk::BlendOp::eReverseSubtract;
		default: EXIT("unknown op: %u\n", op);
	}
	return vk::BlendOp::eAdd;
}

// KYTY_BINDLESS: a stage samples images through the set-1 bindless arrays.
// KYTY_TRACE_PIPELINE_STATS=<all|comma separated hash prefixes> (diagnostic): pipelines whose
// shader hash starts with a listed prefix are created with statistics capture, and the driver's
// per-executable statistics (registers, local memory/spills, ...) are printed as PIPELINE_STATS.
static bool PipelineStatsWanted(const GraphicContext& graphics, uint64_t a, uint64_t b) {
	if (!graphics.pipeline_stats_enabled) {
		return false;
	}
	static const std::vector<std::string> prefixes = [] {
		std::vector<std::string> list;
		const char*              value = Common::ShaderEnv::Read("KYTY_TRACE_PIPELINE_STATS");
		std::string              all   = value != nullptr ? value : "";
		size_t                   start = 0;
		while (start <= all.size()) {
			const size_t end = std::min(all.find(',', start), all.size());
			if (end > start) list.push_back(all.substr(start, end - start));
			start = end + 1;
		}
		return list;
	}();
	char text[2][17];
	std::snprintf(text[0], sizeof(text[0]), "%016llx", static_cast<unsigned long long>(a));
	std::snprintf(text[1], sizeof(text[1]), "%016llx", static_cast<unsigned long long>(b));
	for (const auto& prefix: prefixes) {
		if (prefix == "all" || prefix == "1") return true;
		for (const auto* hash: text) {
			if (std::strncmp(hash, prefix.c_str(), prefix.size()) == 0) return true;
		}
	}
	return false;
}

static void PrintPipelineStats(const GraphicContext& graphics, vk::Pipeline pipeline, const char* kind,
                               uint64_t a, uint64_t b) {
	vk::PipelineInfoKHR pipeline_info {};
	pipeline_info.pipeline = pipeline;
	uint32_t count         = 0;
	if (graphics.device.getPipelineExecutablePropertiesKHR(&pipeline_info, &count, nullptr) !=
	    vk::Result::eSuccess) {
		return;
	}
	std::vector<vk::PipelineExecutablePropertiesKHR> executables(count);
	if (graphics.device.getPipelineExecutablePropertiesKHR(&pipeline_info, &count, executables.data()) !=
	    vk::Result::eSuccess) {
		return;
	}
	for (uint32_t index = 0; index < count; index++) {
		vk::PipelineExecutableInfoKHR executable {};
		executable.pipeline        = pipeline;
		executable.executableIndex = index;
		uint32_t stat_count        = 0;
		if (graphics.device.getPipelineExecutableStatisticsKHR(&executable, &stat_count, nullptr) !=
		    vk::Result::eSuccess) {
			continue;
		}
		std::vector<vk::PipelineExecutableStatisticKHR> stats(stat_count);
		if (graphics.device.getPipelineExecutableStatisticsKHR(&executable, &stat_count, stats.data()) !=
		    vk::Result::eSuccess) {
			continue;
		}
		std::string line;
		for (const auto& stat: stats) {
			char value[64] = {};
			switch (stat.format) {
				case vk::PipelineExecutableStatisticFormatKHR::eBool32:
					std::snprintf(value, sizeof(value), "%u", static_cast<unsigned>(stat.value.b32));
					break;
				case vk::PipelineExecutableStatisticFormatKHR::eInt64:
					std::snprintf(value, sizeof(value), "%lld", static_cast<long long>(stat.value.i64));
					break;
				case vk::PipelineExecutableStatisticFormatKHR::eUint64:
					std::snprintf(value, sizeof(value), "%llu", static_cast<unsigned long long>(stat.value.u64));
					break;
				case vk::PipelineExecutableStatisticFormatKHR::eFloat64:
					std::snprintf(value, sizeof(value), "%g", stat.value.f64);
					break;
			}
			line += " ";
			line += stat.name.data();
			line += "=";
			line += value;
		}
		std::fprintf(stderr, "PIPELINE_STATS %s %016llx %016llx exe=%u(%s) subgroup=%u:%s\n", kind,
		             static_cast<unsigned long long>(a), static_cast<unsigned long long>(b), index,
		             executables[index].name.data(), executables[index].subgroupSize, line.c_str());
	}
	std::fflush(stderr);
}

static bool UsesBindless(const ShaderRecompiler::IR::CompiledShaderInfo& program) {
	return std::ranges::any_of(program.info.images, &ShaderRecompiler::IR::ImageResource::bindless);
}

// Set 0 is the pipeline's own descriptors; set 1, when a stage samples bindless images, is the
// shared bindless table.
static void FillSetLayouts(GraphicContext& graphics, const PipelineCache::Pipeline& pipeline,
                           std::array<vk::DescriptorSetLayout, 2>& layouts, uint32_t& count) {
	layouts[0] = pipeline.descriptor_set_layout;
	count      = 1;
	// Set 1 is our bindless table (KYTY_BINDLESS). Upstream's resident image heap (27dc7c281) would also take set 1
	// (graphics.image_heap_layout); it is never in use here (RenderContext::ImageHeapEnabled), so no program asks for it.
	EXIT_IF(pipeline.uses_image_heap);
	if (pipeline.uses_bindless) {
		EXIT_IF(graphics.bindless_layout == nullptr);
		layouts[1] = graphics.bindless_layout;
		count      = 2;
	}
}

static void AddLayoutBindings(std::vector<vk::DescriptorSetLayoutBinding>& descriptor_bindings,
                              const ShaderRecompiler::IR::CompiledShaderInfo& program,
                              vk::ShaderStageFlagBits              stage) {
	for (auto mask = program.bindings.descriptor_mask; mask != 0; mask &= mask - 1) {
		const auto index = std::countr_zero(mask);
		const auto kind = static_cast<ShaderRecompiler::IR::DescriptorBindingKind>(index);
		descriptor_bindings.push_back(
		    {ShaderRecompiler::IR::NativeBinding(program.stage, kind), NativeDescriptorType(kind),
		     program.bindings.descriptor_counts[index], stage, nullptr});
	}
}

// False only on a background compile (shaderPrecompile.h) the driver refused; the GPU thread exits then.
static bool CreateDescriptorLayout(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                                   std::span<const vk::DescriptorSetLayoutBinding> bindings,
                                   std::span<const vk::DescriptorSetLayoutBinding> set_bindings);

static bool CreateDescriptorLayout(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                                   std::span<const vk::DescriptorSetLayoutBinding> bindings) {
	return CreateDescriptorLayout(graphics, pipeline, bindings, bindings);
}

static bool CreateDescriptorLayout(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                                   std::span<const vk::DescriptorSetLayoutBinding> bindings,
                                   std::span<const vk::DescriptorSetLayoutBinding> set_bindings) {
	const auto& limits = graphics.GetPhysicalDeviceProperties().limits;
	uint64_t descriptor_count = 0;
	uint64_t storage_buffers = 0;
	uint64_t samplers = 0;
	std::array<uint64_t, 32> stage_buffers {}, stage_resources {}, stage_samplers {};
	for (const auto& binding: bindings) {
		descriptor_count += binding.descriptorCount;
		const bool storage = binding.descriptorType == vk::DescriptorType::eStorageBuffer;
		const bool sampler = binding.descriptorType == vk::DescriptorType::eSampler;
		if (storage) storage_buffers += binding.descriptorCount;
		if (sampler) samplers += binding.descriptorCount;
		for (auto stages = static_cast<uint32_t>(binding.stageFlags); stages != 0; stages &= stages - 1u) {
			const auto stage = std::countr_zero(stages);
			if (storage) stage_buffers[stage] += binding.descriptorCount;
			if (sampler) stage_samplers[stage] += binding.descriptorCount;
			else
				stage_resources[stage] += binding.descriptorCount;
			if (stage_buffers[stage] > limits.maxPerStageDescriptorStorageBuffers ||
			    stage_resources[stage] > limits.maxPerStageResources ||
			    stage_samplers[stage] > limits.maxPerStageDescriptorSamplers) {
				EXIT("native descriptor stage %u exceeds device limits: buffers=%" PRIu64
				     "/%u resources=%" PRIu64 "/%u samplers=%" PRIu64 "/%u\n", stage, stage_buffers[stage],
				     limits.maxPerStageDescriptorStorageBuffers, stage_resources[stage], limits.maxPerStageResources,
				     stage_samplers[stage], limits.maxPerStageDescriptorSamplers);
			}
		}
	}
	if (storage_buffers > limits.maxDescriptorSetStorageBuffers) {
		EXIT("native storage-buffer descriptors exceed device limit: %" PRIu64 "/%u\n",
		     storage_buffers, limits.maxDescriptorSetStorageBuffers);
	}
	if (samplers > limits.maxDescriptorSetSamplers) {
		EXIT("native sampler descriptors exceed device limit: %" PRIu64 "/%u\n",
		     samplers, limits.maxDescriptorSetSamplers);
	}
	uint64_t set_descriptor_count = 0;
	for (const auto& binding: set_bindings) set_descriptor_count += binding.descriptorCount;
	(void)descriptor_count;
	pipeline.uses_push_descriptors = set_descriptor_count <= graphics.max_push_descriptors;

	vk::DescriptorSetLayoutCreateInfo create {};
	create.flags        = pipeline.uses_push_descriptors
	                          ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR
	                          : vk::DescriptorSetLayoutCreateFlags {};
	create.bindingCount = static_cast<uint32_t>(set_bindings.size());
	create.pBindings    = set_bindings.data();
	const auto result =
	    graphics.device.createDescriptorSetLayout(&create, nullptr, &pipeline.descriptor_set_layout);
	if (result != vk::Result::eSuccess && IsBackgroundCompileThread()) {
		pipeline.descriptor_set_layout = nullptr;
		return false;
	}
	EXIT_IF(result != vk::Result::eSuccess);
	return true;
}

// The blend factors and operations GetBlendFactor and GetBlendOp map (they exit on any other).
static bool BlendFactorSupported(uint32_t factor) {
	return factor <= static_cast<uint32_t>(Prospero::BlendFactor::kSrcAlphaSaturate) ||
	       (factor >= static_cast<uint32_t>(Prospero::BlendFactor::kConstantColor) &&
	        factor <= static_cast<uint32_t>(Prospero::BlendFactor::kOneMinusConstantAlpha));
}

static bool BlendOpSupported(uint32_t op) {
	return op <= static_cast<uint32_t>(Prospero::BlendOp::kReverseSubtract);
}

// The vertex formats GetInputFormat maps (it exits on any other).
static bool InputFormatSupported(const ShaderBufferResource& res) {
	const auto raw_format = res.RawFormat();
	if (raw_format == kTemporaryVertexAttribFormat113 || raw_format == kTemporaryPs5BufferFormat121) {
		return true;
	}
	return VulkanFormat(res.Format()) != vk::Format::eUndefined &&
	       ShaderRecompiler::Format::GetFormatInfo(res.Format()).component_count != 0;
}

const char* GraphicsPipelineUnsupported(const GraphicContext& graphics, const PipelineRenderingState& rendering,
                                        const PipelineVertexInputState&        vertex_input,
                                        std::span<const ShaderVertexInputInfo> vertex_info,
                                        const ShaderPixelInputInfo*            ps_input_info,
                                        const PipelineStaticParameters&        static_params) {
	if (vertex_info.size() != 1 && vertex_info.size() != 3) {
		return "vertex stage count";
	}
	for (const auto& stage: vertex_info) {
		if (!stage.stage) {
			return "vertex stage program";
		}
	}
	if (ps_input_info != nullptr && !ps_input_info->stage) {
		return "pixel stage program";
	}
	const auto& vs_input_info = vertex_info.front();
	const bool  mesh          = vs_input_info.stage.program->stage == ShaderType::Mesh;
	if (mesh) {
		if (!graphics.mesh_shader_enabled) {
			return "mesh shaders unsupported";
		}
		// PipelineCache::GetGraphicsPrograms exits on a mesh draw beyond these.
		const auto& mesh_info = vs_input_info.mesh;
		const auto& limits    = graphics.mesh_shader_properties;
		if (mesh_info.wave_size == 0) {
			return "mesh wave size";
		}
		const auto logical_threads = mesh_info.threads_num[0] * mesh_info.threads_num[1] * mesh_info.threads_num[2];
		const auto host_threads    = ((logical_threads + mesh_info.wave_size - 1u) / mesh_info.wave_size) *
		                          std::min(mesh_info.host_subgroup_size, mesh_info.wave_size);
		if (host_threads > limits.maxMeshWorkGroupInvocations || host_threads > limits.maxMeshWorkGroupSize[0] ||
		    mesh_info.max_vertices > limits.maxMeshOutputVertices ||
		    mesh_info.max_primitives > limits.maxMeshOutputPrimitives ||
		    mesh_info.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
			return "mesh shader exceeds host limits";
		}
	}
	if (vertex_info.size() == 3 &&
	    (vs_input_info.tess.input_control_points == 0 ||
	     vs_input_info.tess.input_control_points > graphics.GetPhysicalDeviceProperties().limits.maxTessellationPatchSize)) {
		return "tessellation patch size";
	}
	if (static_params.provoking_vtx_last && !graphics.provoking_vertex_last_enabled) {
		return "provoking vertex last unsupported";
	}
	if (static_params.sample_shading_enable && !graphics.sample_rate_shading_enabled) {
		return "sample-rate shading unsupported";
	}
	if (vulkan_sample_count(static_params.samples) == vk::SampleCountFlagBits {}) {
		return "sample count";
	}
	if (static_cast<uint32_t>(static_params.topology) > static_cast<uint32_t>(vk::PrimitiveTopology::ePatchList) ||
	    static_cast<uint32_t>(static_params.polygon_mode) > static_cast<uint32_t>(vk::PolygonMode::ePoint)) {
		return "topology or polygon mode";
	}
	if (rendering.color_count > RENDER_COLOR_ATTACHMENTS_MAX) {
		return "color attachment count";
	}
	for (uint32_t i = 0; i < rendering.color_count; i++) {
		if ((static_params.color_mask[i] & ~0x0fu) != 0) {
			return "color write mask";
		}
		if (!BlendFactorSupported(static_params.color_srcblend[i]) ||
		    !BlendFactorSupported(static_params.color_destblend[i]) ||
		    !BlendOpSupported(static_params.color_comb_fcn[i]) ||
		    (static_params.separate_alpha_blend[i] &&
		     (!BlendFactorSupported(static_params.alpha_srcblend[i]) ||
		      !BlendFactorSupported(static_params.alpha_destblend[i]) ||
		      !BlendOpSupported(static_params.alpha_comb_fcn[i])))) {
			return "blend state";
		}
	}
	if (!mesh) {
		if (vertex_input.binding_count > ShaderVertexInputInfo::RES_MAX ||
		    vertex_input.attribute_count > ShaderVertexInputInfo::RES_MAX) {
			return "vertex input count";
		}
		for (uint32_t index = 0; index < vertex_input.attribute_count; index++) {
			const auto registers_num = vs_input_info.resources_dst[index].registers_num;
			if (vertex_input.attributes[index].binding >= vertex_input.binding_count) {
				return "vertex input binding";
			}
			if (registers_num < 1 || registers_num > 4) {
				return "vertex input registers";
			}
			if (!InputFormatSupported(vs_input_info.resources[index])) {
				return "vertex input format";
			}
		}
	}
	bool bindless = ps_input_info != nullptr && UsesBindless(*ps_input_info->stage.program);
	for (const auto& stage: vertex_info) {
		bindless = bindless || UsesBindless(*stage.stage.program);
	}
	if (bindless && graphics.bindless_layout == nullptr) {
		return "bindless table missing";
	}
	return nullptr;
}

namespace {

enum GplKind : uint32_t { GplVertexInput, GplPreRaster, GplFragment, GplOutput, GplLayout, GplKindCount };

struct GplKey {
	uint64_t low  = 0;
	uint64_t high = 0;
	uint32_t kind = 0;
	bool     operator==(const GplKey&) const = default;
};

struct GplKeyHash {
	std::size_t operator()(const GplKey& key) const { return static_cast<std::size_t>(key.low ^ (key.high * 31u) ^ key.kind); }
};

class GplKeyBuilder {
public:
	template <typename T>
	void Add(const T& value) {
		static_assert(std::is_trivially_copyable_v<T>);
		const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
		m_bytes.insert(m_bytes.end(), bytes, bytes + sizeof(T));
	}
	void AddBytes(const void* data, size_t size) {
		const auto* bytes = static_cast<const uint8_t*>(data);
		m_bytes.insert(m_bytes.end(), bytes, bytes + size);
	}
	[[nodiscard]] GplKey Finish(GplKind kind) const {
		const auto hash = XXH3_128bits(m_bytes.data(), m_bytes.size());
		return {hash.low64, hash.high64, kind};
	}

private:
	std::vector<uint8_t> m_bytes;
};

struct GplLayoutEntry {
	vk::DescriptorSetLayout set_layout = nullptr;
	vk::PipelineLayout      layout     = nullptr;
};

struct GplStats {
	std::atomic<uint64_t> probe_hit {0}, probe_miss {0}, fast {0}, fast_us {0}, link_us {0}, mono {0};
	std::array<std::atomic<uint64_t>, GplKindCount> made {}, hit {}, made_us {};
	std::atomic<uint64_t> fb_rect_list {0}, fb_mesh {0}, fb_stats {0}, fb_module {0}, fb_library {0};
	std::atomic<uint64_t> opt_queued {0}, opt_swapped {0}, opt_failed {0}, opt_discarded {0}, opt_us {0};
	std::atomic<int64_t>  window_ms {0};
};

GplStats g_gpl;

struct GplTemplateEntry {
	vk::DescriptorUpdateTemplate update_template = nullptr;
	uint32_t                     bytes           = 0;
};

struct GplLibraryCache {
	std::mutex                                                                     mutex;
	std::unordered_map<GplKey, std::shared_future<vk::Pipeline>, GplKeyHash> libraries;
	std::unordered_map<GplKey, GplLayoutEntry, GplKeyHash>                   layouts;
	std::unordered_map<GplKey, vk::DescriptorSetLayout, GplKeyHash>          set_layouts;
	std::unordered_map<GplKey, vk::PipelineLayout, GplKeyHash>               pipeline_layouts;
	std::unordered_map<VkDescriptorSetLayout, GplTemplateEntry>             templates;
};

GplLibraryCache& GplLibraries() {
	static GplLibraryCache cache;
	return cache;
}

void GplKeyBindings(GplKeyBuilder& builder, std::span<const vk::DescriptorSetLayoutBinding> bindings) {
	builder.Add(static_cast<uint32_t>(bindings.size()));
	for (const auto& b: bindings) {
		builder.Add(b.binding);
		builder.Add(b.descriptorType);
		builder.Add(b.descriptorCount);
		builder.Add(static_cast<uint32_t>(b.stageFlags));
	}
}

vk::DescriptorSetLayout GplSharedSetLayout(GraphicContext& graphics, std::span<const vk::DescriptorSetLayoutBinding> bindings,
                                           bool push) {
	GplKeyBuilder builder;
	GplKeyBindings(builder, bindings);
	builder.Add(push);
	const auto            key   = builder.Finish(GplLayout);
	auto&                 cache = GplLibraries();
	const std::lock_guard lock(cache.mutex);
	if (const auto found = cache.set_layouts.find(key); found != cache.set_layouts.end()) return found->second;
	vk::DescriptorSetLayoutCreateInfo create {};
	create.flags        = push ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR : vk::DescriptorSetLayoutCreateFlags {};
	create.bindingCount = static_cast<uint32_t>(bindings.size());
	create.pBindings    = bindings.data();
	vk::DescriptorSetLayout layout = nullptr;
	if (graphics.device.createDescriptorSetLayout(&create, nullptr, &layout) != vk::Result::eSuccess) return nullptr;
	cache.set_layouts.emplace(key, layout);
	return layout;
}

vk::PipelineLayout GplSharedPipelineLayout(GraphicContext& graphics, std::span<const vk::DescriptorSetLayout> sets,
                                           const vk::PushConstantRange& push_constants, vk::PipelineLayoutCreateFlags flags) {
	GplKeyBuilder builder;
	builder.Add(static_cast<uint32_t>(sets.size()));
	for (const auto set: sets) builder.Add(static_cast<VkDescriptorSetLayout>(set));
	builder.Add(static_cast<uint32_t>(push_constants.stageFlags));
	builder.Add(push_constants.offset);
	builder.Add(push_constants.size);
	builder.Add(static_cast<uint32_t>(flags));
	const auto            key   = builder.Finish(GplLayout);
	auto&                 cache = GplLibraries();
	const std::lock_guard lock(cache.mutex);
	if (const auto found = cache.pipeline_layouts.find(key); found != cache.pipeline_layouts.end()) return found->second;
	vk::PipelineLayoutCreateInfo create {};
	create.flags                  = flags;
	create.setLayoutCount         = static_cast<uint32_t>(sets.size());
	create.pSetLayouts            = sets.data();
	create.pushConstantRangeCount = 1;
	create.pPushConstantRanges    = &push_constants;
	vk::PipelineLayout layout     = nullptr;
	if (graphics.device.createPipelineLayout(&create, nullptr, &layout) != vk::Result::eSuccess) return nullptr;
	g_gpl.made[GplLayout].fetch_add(1, std::memory_order_relaxed);
	cache.pipeline_layouts.emplace(key, layout);
	return layout;
}

vk::ShaderStageFlags SplitPushStages(bool mesh) {
	return mesh ? vk::ShaderStageFlags(vk::ShaderStageFlagBits::eAllGraphics | vk::ShaderStageFlagBits::eMeshEXT |
	                                   vk::ShaderStageFlagBits::eTaskEXT)
	            : vk::ShaderStageFlags(vk::ShaderStageFlagBits::eAllGraphics);
}

const ShaderRecompiler::IR::ResourceSnapshot g_gpl_no_resources {};

uint64_t GplElapsedUs(std::chrono::steady_clock::time_point begin) {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin).count());
}

template <typename Create>
vk::Pipeline GplLibrary(const GplKey& key, Create&& create) {
	auto&                            cache = GplLibraries();
	std::shared_future<vk::Pipeline> future;
	std::promise<vk::Pipeline>       promise;
	bool                             owner = false;
	{
		const std::lock_guard lock(cache.mutex);
		if (const auto found = cache.libraries.find(key); found != cache.libraries.end()) {
			future = found->second;
		} else {
			future = promise.get_future().share();
			cache.libraries.emplace(key, future);
			owner = true;
		}
	}
	if (!owner) {
		g_gpl.hit[key.kind].fetch_add(1, std::memory_order_relaxed);
		return future.get();
	}
	const auto begin   = std::chrono::steady_clock::now();
	const auto library = create();
	g_gpl.made[key.kind].fetch_add(1, std::memory_order_relaxed);
	g_gpl.made_us[key.kind].fetch_add(GplElapsedUs(begin), std::memory_order_relaxed);
	promise.set_value(library);
	return library;
}

const char* GplDynamicSubset(vk::DynamicState state) {
	switch (state) {
		case vk::DynamicState::eViewportWithCount:
		case vk::DynamicState::eScissorWithCount:
		case vk::DynamicState::eLineWidth:
		case vk::DynamicState::eDepthBiasEnable:
		case vk::DynamicState::eDepthBias:
		case vk::DynamicState::eCullMode:
		case vk::DynamicState::eFrontFace: return "pre";
		case vk::DynamicState::eDepthTestEnable:
		case vk::DynamicState::eDepthWriteEnable:
		case vk::DynamicState::eDepthCompareOp:
		case vk::DynamicState::eStencilTestEnable:
		case vk::DynamicState::eStencilOp:
		case vk::DynamicState::eStencilCompareMask:
		case vk::DynamicState::eStencilReference:
		case vk::DynamicState::eStencilWriteMask:
		case vk::DynamicState::eDepthBoundsTestEnable:
		case vk::DynamicState::eDepthBounds: return "fs";
		case vk::DynamicState::eBlendConstants:
		case vk::DynamicState::eColorWriteEnableEXT:
		case vk::DynamicState::eAttachmentFeedbackLoopEnableEXT: return "fo";
		default: return nullptr;
	}
}

void GplReport(bool force) {
	if (!PerfEnabled()) return;
	const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
	                        std::chrono::steady_clock::now().time_since_epoch())
	                        .count();
	auto last = g_gpl.window_ms.load(std::memory_order_relaxed);
	if (!force && now_ms - last < 5000) return;
	if (!g_gpl.window_ms.compare_exchange_strong(last, now_ms, std::memory_order_relaxed)) return;
	const auto ms = [](const std::atomic<uint64_t>& us) { return static_cast<double>(us.load(std::memory_order_relaxed)) / 1000.0; };
	const auto n  = [](const std::atomic<uint64_t>& v) { return static_cast<unsigned long long>(v.load(std::memory_order_relaxed)); };
	std::fprintf(stderr,
	             "PERF gpl mode=%u%s probe_hit=%llu probe_miss=%llu fast=%llu fast_ms=%.0f link_ms=%.0f mono=%llu "
	             "vi=%llu/%llu pre=%llu/%llu (%.0fms) fs=%llu/%llu (%.0fms) fo=%llu/%llu layouts=%llu "
	             "fallback rect_list=%llu mesh=%llu stats=%llu module=%llu library=%llu "
	             "opt queued=%llu swapped=%llu failed=%llu discarded=%llu opt_ms=%.0f\n",
	             Common::EnvRead::Gpl(), Common::EnvRead::GplSplit() ? " split=1" : "", n(g_gpl.probe_hit), n(g_gpl.probe_miss), n(g_gpl.fast), ms(g_gpl.fast_us),
	             ms(g_gpl.link_us), n(g_gpl.mono), n(g_gpl.made[GplVertexInput]), n(g_gpl.hit[GplVertexInput]),
	             n(g_gpl.made[GplPreRaster]), n(g_gpl.hit[GplPreRaster]), ms(g_gpl.made_us[GplPreRaster]),
	             n(g_gpl.made[GplFragment]), n(g_gpl.hit[GplFragment]), ms(g_gpl.made_us[GplFragment]),
	             n(g_gpl.made[GplOutput]), n(g_gpl.hit[GplOutput]), n(g_gpl.made[GplLayout]), n(g_gpl.fb_rect_list),
	             n(g_gpl.fb_mesh), n(g_gpl.fb_stats), n(g_gpl.fb_module), n(g_gpl.fb_library), n(g_gpl.opt_queued),
	             n(g_gpl.opt_swapped), n(g_gpl.opt_failed), n(g_gpl.opt_discarded), ms(g_gpl.opt_us));
	std::fflush(stderr);
}

GplLayoutEntry GplLibraryLayout(GraphicContext& graphics, const PipelineCache::Pipeline& pipeline,
                                std::span<const vk::DescriptorSetLayoutBinding> bindings,
                                const vk::PushConstantRange& push_constants) {
	GplKeyBuilder builder;
	for (const auto& b: bindings) {
		builder.Add(b.binding);
		builder.Add(b.descriptorType);
		builder.Add(b.descriptorCount);
		builder.Add(static_cast<uint32_t>(b.stageFlags));
	}
	builder.Add(pipeline.uses_push_descriptors);
	builder.Add(pipeline.uses_bindless);
	builder.Add(static_cast<uint32_t>(push_constants.stageFlags));
	builder.Add(push_constants.offset);
	builder.Add(push_constants.size);
	const auto key   = builder.Finish(GplLayout);
	auto&      cache = GplLibraries();
	const std::lock_guard lock(cache.mutex);
	if (const auto found = cache.layouts.find(key); found != cache.layouts.end()) return found->second;
	GplLayoutEntry                    entry;
	vk::DescriptorSetLayoutCreateInfo set_create {};
	set_create.flags        = pipeline.uses_push_descriptors ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR
	                                                         : vk::DescriptorSetLayoutCreateFlags {};
	set_create.bindingCount = static_cast<uint32_t>(bindings.size());
	set_create.pBindings    = bindings.data();
	if (graphics.device.createDescriptorSetLayout(&set_create, nullptr, &entry.set_layout) != vk::Result::eSuccess) {
		return {};
	}
	std::array<vk::DescriptorSetLayout, 2> set_layouts {entry.set_layout, graphics.bindless_layout};
	vk::PipelineLayoutCreateInfo           layout_create {};
	layout_create.setLayoutCount         = pipeline.uses_bindless ? 2u : 1u;
	layout_create.pSetLayouts            = set_layouts.data();
	layout_create.pushConstantRangeCount = 1;
	layout_create.pPushConstantRanges    = &push_constants;
	if (graphics.device.createPipelineLayout(&layout_create, nullptr, &entry.layout) != vk::Result::eSuccess) {
		graphics.device.destroyDescriptorSetLayout(entry.set_layout, nullptr);
		return {};
	}
	g_gpl.made[GplLayout].fetch_add(1, std::memory_order_relaxed);
	cache.layouts.emplace(key, entry);
	return entry;
}

} // namespace

void GplCountOptimize(GplOptimizeEvent event, uint64_t us) {
	switch (event) {
		case GplOptimizeEvent::Queued: g_gpl.opt_queued.fetch_add(1, std::memory_order_relaxed); break;
		case GplOptimizeEvent::Swapped: g_gpl.opt_swapped.fetch_add(1, std::memory_order_relaxed); break;
		case GplOptimizeEvent::Failed: g_gpl.opt_failed.fetch_add(1, std::memory_order_relaxed); break;
		case GplOptimizeEvent::Discarded: g_gpl.opt_discarded.fetch_add(1, std::memory_order_relaxed); break;
	}
	g_gpl.opt_us.fetch_add(us, std::memory_order_relaxed);
	GplReport(false);
}

void GplDestroyLibraries(GraphicContext& graphics) {
	auto&                 cache = GplLibraries();
	const std::lock_guard lock(cache.mutex);
	GplReport(true);
	for (auto& [key, future]: cache.libraries) {
		(void)key;
		if (future.valid() && future.get() != nullptr) graphics.device.destroyPipeline(future.get(), nullptr);
	}
	for (auto& [key, entry]: cache.layouts) {
		(void)key;
		graphics.device.destroyPipelineLayout(entry.layout, nullptr);
		graphics.device.destroyDescriptorSetLayout(entry.set_layout, nullptr);
	}
	for (auto& [layout, entry]: cache.templates) {
		(void)layout;
		graphics.device.destroyDescriptorUpdateTemplate(entry.update_template, nullptr);
	}
	for (auto& [key, layout]: cache.pipeline_layouts) {
		(void)key;
		graphics.device.destroyPipelineLayout(layout, nullptr);
	}
	for (auto& [key, layout]: cache.set_layouts) {
		(void)key;
		graphics.device.destroyDescriptorSetLayout(layout, nullptr);
	}
	cache.libraries.clear();
	cache.layouts.clear();
	cache.templates.clear();
	cache.pipeline_layouts.clear();
	cache.set_layouts.clear();
}

vk::DescriptorUpdateTemplate GplSetTemplate(GraphicContext& graphics, vk::DescriptorSetLayout layout,
                                            std::span<const vk::DescriptorUpdateTemplateEntry> entries, uint32_t& bytes) {
	auto&                 cache = GplLibraries();
	const std::lock_guard lock(cache.mutex);
	if (const auto found = cache.templates.find(static_cast<VkDescriptorSetLayout>(layout)); found != cache.templates.end()) {
		bytes = found->second.bytes;
		return found->second.update_template;
	}
	EXIT_IF(entries.empty());
	vk::DescriptorUpdateTemplateCreateInfo info {};
	info.descriptorUpdateEntryCount = static_cast<uint32_t>(entries.size());
	info.pDescriptorUpdateEntries   = entries.data();
	info.templateType               = vk::DescriptorUpdateTemplateType::eDescriptorSet;
	info.descriptorSetLayout        = layout;
	GplTemplateEntry entry;
	EXIT_IF(graphics.device.createDescriptorUpdateTemplate(&info, nullptr, &entry.update_template) != vk::Result::eSuccess);
	entry.bytes = entries.back().offset + entries.back().descriptorCount * entries.back().stride;
	cache.templates.emplace(static_cast<VkDescriptorSetLayout>(layout), entry);
	bytes = entry.bytes;
	return entry.update_template;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const PipelineRenderingState&          rendering,
                            const PipelineVertexInputState&        vertex_input,
                            std::span<const ShaderVertexInputInfo> vertex_info,
                            const ShaderPixelInputInfo*            ps_input_info,
                            const PipelineCache::GraphicsPrograms& programs,
                            const PipelineStaticParameters&        static_params,
                            vk::PipelineCache                      driver_cache) {
	const auto& vs_input_info  = vertex_info.front();
	const auto& vertex_program = programs.vertex[0];
	const auto& pixel_program  = programs.pixel;
	const bool  tessellation   = vertex_info.size() == 3;
	const bool ps_active = ps_input_info != nullptr;
	EXIT_IF(!vertex_program || (ps_active && !pixel_program));
	const bool with_depth = rendering.depth_format != vk::Format::eUndefined ||
	                        rendering.stencil_format != vk::Format::eUndefined;
	EXIT_IF(!vs_input_info.stage);
	pipeline.color_count  = rendering.color_count;
	pipeline.cull_dynamic = SharedPipelineState(graphics);
	// A background compile (shaderPrecompile.h) gives up where the GPU thread exits: what it created is destroyed
	// (the caller destroys the layouts and pipeline left in `pipeline`). It checks the state first
	// (GraphicsPipelineUnsupported); the driver calls can still fail.
	const bool background = IsBackgroundCompileThread();
	const bool mesh = vs_input_info.stage.program->stage == ShaderType::Mesh;
	if (background && mesh && !graphics.mesh_shader_enabled) {
		return;
	}
	EXIT_NOT_IMPLEMENTED(mesh && !graphics.mesh_shader_enabled);
	const bool rect_list =
	    !mesh && !tessellation && static_params.topology == vk::PrimitiveTopology::ePatchList;

	vk::ShaderModule tess_control_shader_module = nullptr;
	vk::ShaderModule tess_eval_shader_module    = nullptr;

	if (rect_list) {
		const auto shaders =
		    BuildRectListShaders(vs_input_info, ps_active ? ps_input_info : nullptr);
		tess_control_shader_module = background ? TryCompileSPV(shaders.control, graphics.device)
		                                        : CompileSPV(shaders.control, graphics.device);
		if (graphics_debug_dump_enabled()) {
			LOGF("PipelineTrace: vkCreateShaderModule RectList TCS done module=%p\n",
			     static_cast<void*>(tess_control_shader_module));
		}

		tess_eval_shader_module = background ? TryCompileSPV(shaders.evaluation, graphics.device)
		                                     : CompileSPV(shaders.evaluation, graphics.device);
		if (graphics_debug_dump_enabled()) {
			LOGF("PipelineTrace: vkCreateShaderModule RectList TES done module=%p\n",
			     static_cast<void*>(tess_eval_shader_module));
		}
	}
	const auto destroy_rect_list_modules = [&] {
		if (tess_control_shader_module != nullptr) {
			graphics.device.destroyShaderModule(tess_control_shader_module, nullptr);
		}
		if (tess_eval_shader_module != nullptr) {
			graphics.device.destroyShaderModule(tess_eval_shader_module, nullptr);
		}
	};

	if (background && rect_list &&
	    (tess_control_shader_module == nullptr || tess_eval_shader_module == nullptr)) {
		destroy_rect_list_modules();
		return;
	}
	EXIT_NOT_IMPLEMENTED(
	    rect_list && (tess_control_shader_module == nullptr || tess_eval_shader_module == nullptr));

	vk::PipelineShaderStageCreateInfo shader_stages[4] {};
	uint32_t                          shader_stage_count = 0;
	for (uint32_t i = 0; i < vertex_info.size(); i++) {
		shader_stages[shader_stage_count++] = {.stage =
		                                           NativeShaderStage(vertex_info[i].logical_stage),
		                                       .module = programs.vertex[i].module,
		                                       .pName  = "main"};
	}
	if (rect_list) {
		shader_stages[shader_stage_count++] = {.stage =
		                                           vk::ShaderStageFlagBits::eTessellationControl,
		                                       .module = tess_control_shader_module,
		                                       .pName  = "main"};
		shader_stages[shader_stage_count++] = {.stage =
		                                           vk::ShaderStageFlagBits::eTessellationEvaluation,
		                                       .module = tess_eval_shader_module,
		                                       .pName  = "main"};
	}
	if (ps_active) {
		shader_stages[shader_stage_count++] = {.stage  = vk::ShaderStageFlagBits::eFragment,
		                                       .module = pixel_program.module,
		                                       .pName  = "main"};
	}

	vk::VertexInputAttributeDescription input_attr[ShaderVertexInputInfo::RES_MAX] {};
	vk::VertexInputBindingDescription   input_desc[ShaderVertexInputInfo::RES_MAX] {};

	for (uint32_t binding = 0; binding < vertex_input.binding_count; binding++) {
		input_desc[binding].binding   = binding;
		input_desc[binding].stride    = vertex_input.bindings[binding].stride;
		input_desc[binding].inputRate = vertex_input.bindings[binding].instance
		                                    ? vk::VertexInputRate::eInstance
		                                    : vk::VertexInputRate::eVertex;
	}
	for (uint32_t index = 0; index < vertex_input.attribute_count; index++) {
		input_attr[index].binding  = vertex_input.attributes[index].binding;
		input_attr[index].location = index;
		input_attr[index].offset   = vertex_input.attributes[index].offset;

		uint32_t   attr_size     = 4;
		const auto registers_num = vs_input_info.resources_dst[index].registers_num;
		const auto compiled_components =
		    vs_input_info.stage.program->info.vertex_fetch_components[index];
		const auto used_components =
		    compiled_components > 0 ? static_cast<int>(compiled_components) : registers_num;
		GetInputFormat(vs_input_info.resources[index], input_attr[index].format, attr_size,
		               static_cast<uint32_t>(used_components));

		if (graphics_debug_dump_enabled()) {
			static std::atomic_uint log_count = 0;
			const auto              log_id    = log_count.fetch_add(1, std::memory_order_relaxed);
			if (log_id < 128) {
				LOGF("VertexInputState[%u]: attr=%u binding=%u offset=%u stride=%u fmt=%d "
				     "src_fmt=%u dst=v%u regs=%u"
				     " fetched_components=%u attr_size=%u swizzle=%u,%u,%u,%u\n",
				     log_id, index, input_attr[index].binding, input_attr[index].offset,
				     input_desc[input_attr[index].binding].stride,
				     static_cast<int>(input_attr[index].format),
				     static_cast<uint32_t>(vs_input_info.resources[index].Format()),
				     static_cast<uint32_t>(vs_input_info.resources_dst[index].register_start),
				     static_cast<uint32_t>(registers_num), static_cast<uint32_t>(used_components),
				     attr_size, static_cast<uint32_t>(vs_input_info.resources[index].DstSelX()),
				     static_cast<uint32_t>(vs_input_info.resources[index].DstSelY()),
				     static_cast<uint32_t>(vs_input_info.resources[index].DstSelZ()),
				     static_cast<uint32_t>(vs_input_info.resources[index].DstSelW()));
			}
		}

		if (vs_input_info.resources[index].OutOfBounds() != 0) {
			static std::atomic_bool logged {false};
			if (!logged.exchange(true, std::memory_order_relaxed)) {
				LOGF("VertexInput: temporary: accepting PS5 out-of-bounds behavior %" PRIu8 "\n",
				     vs_input_info.resources[index].OutOfBounds());
			}
		}

		EXIT_IF(registers_num < 1 || registers_num > 4);
	}

	vk::PipelineVertexInputStateCreateInfo vertex_input_info {};
	vertex_input_info.vertexBindingDescriptionCount   = vertex_input.binding_count;
	vertex_input_info.pVertexBindingDescriptions      = input_desc;
	vertex_input_info.vertexAttributeDescriptionCount = vertex_input.attribute_count;
	vertex_input_info.pVertexAttributeDescriptions    = input_attr;

	vk::PipelineInputAssemblyStateCreateInfo input_assembly {};
	input_assembly.topology = static_params.topology;
	input_assembly.primitiveRestartEnable =
	    static_params.primitive_restart_enable ? VK_TRUE : VK_FALSE;

	vk::PipelineViewportDepthClipControlCreateInfoEXT depth_clip_control {};
	depth_clip_control.negativeOneToOne = (static_params.negative_one_to_one ? VK_TRUE : VK_FALSE);

	vk::PipelineViewportStateCreateInfo viewport_state {};
	viewport_state.pNext = &depth_clip_control;

	vk::CullModeFlags cull_mode = vk::CullModeFlagBits::eNone;
	if (static_params.cull_back) {
		cull_mode |= vk::CullModeFlagBits::eBack;
	}
	if (static_params.cull_front) {
		cull_mode |= vk::CullModeFlagBits::eFront;
	}

	vk::FrontFace front_face =
	    (static_params.face ? vk::FrontFace::eClockwise : vk::FrontFace::eCounterClockwise);

	vk::PipelineRasterizationDepthClipStateCreateInfoEXT clip_ext {};
	clip_ext.depthClipEnable = static_params.depth_clip_enable ? VK_TRUE : VK_FALSE;

	vk::PipelineRasterizationStateCreateInfo rasterizer {};
	// MoltenVK lacks VK_EXT_depth_clip_enable; omit the depth-clip struct on macOS and accept
	// Vulkan's default depth clipping (enabled) instead of the PS5's clamp behavior.
#if !defined(__APPLE__)
	// The DB clamps depth to the viewport range after polygon offset is applied.
	rasterizer.depthClampEnable = VK_TRUE;
	rasterizer.pNext = &clip_ext;
#endif
	vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT provoking_vertex {};
	if (background && static_params.provoking_vtx_last && !graphics.provoking_vertex_last_enabled) {
		destroy_rect_list_modules();
		return;
	}
	EXIT_NOT_IMPLEMENTED(static_params.provoking_vtx_last &&
	                     !graphics.provoking_vertex_last_enabled);
	if (graphics.provoking_vertex_last_enabled) {
		provoking_vertex.provokingVertexMode = static_params.provoking_vtx_last
		    ? vk::ProvokingVertexModeEXT::eLastVertex : vk::ProvokingVertexModeEXT::eFirstVertex;
		provoking_vertex.pNext = rasterizer.pNext;
		rasterizer.pNext = &provoking_vertex;
	}
	rasterizer.cullMode  = cull_mode;
	rasterizer.frontFace = front_face;
	rasterizer.polygonMode = static_params.polygon_mode;
	rasterizer.lineWidth = 1.0f;

	vk::PipelineMultisampleStateCreateInfo multisampling {};
	multisampling.sampleShadingEnable  = static_params.sample_shading_enable ? VK_TRUE : VK_FALSE;
	multisampling.rasterizationSamples = vulkan_sample_count(static_params.samples);
	multisampling.minSampleShading     = 1.0f;
	multisampling.alphaToCoverageEnable =
	    static_params.alpha_to_coverage_enable ? VK_TRUE : VK_FALSE;

	vk::PipelineColorBlendAttachmentState color_blend_attachment[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	for (uint32_t i = 0; i < rendering.color_count; i++) {
		EXIT_NOT_IMPLEMENTED((static_params.color_mask[i] & ~0x0fu) != 0);
		color_blend_attachment[i].colorWriteMask =
		    vk::ColorComponentFlags {static_params.color_mask[i]};
		color_blend_attachment[i].blendEnable = static_params.blend_enable[i] ? VK_TRUE : VK_FALSE;
		color_blend_attachment[i].srcColorBlendFactor =
		    GetBlendFactor(static_params.color_srcblend[i]);
		color_blend_attachment[i].dstColorBlendFactor =
		    GetBlendFactor(static_params.color_destblend[i]);
		color_blend_attachment[i].colorBlendOp = GetBlendOp(static_params.color_comb_fcn[i]);
		color_blend_attachment[i].srcAlphaBlendFactor =
		    (static_params.separate_alpha_blend[i]
		         ? GetBlendFactor(static_params.alpha_srcblend[i])
		         : color_blend_attachment[i].srcColorBlendFactor);
		color_blend_attachment[i].dstAlphaBlendFactor =
		    (static_params.separate_alpha_blend[i]
		         ? GetBlendFactor(static_params.alpha_destblend[i])
		         : color_blend_attachment[i].dstColorBlendFactor);
		color_blend_attachment[i].alphaBlendOp =
		    (static_params.separate_alpha_blend[i] ? GetBlendOp(static_params.alpha_comb_fcn[i])
		                                           : color_blend_attachment[i].colorBlendOp);
	}

	vk::Bool32 color_write_enable[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	for (uint32_t i = 0; i < rendering.color_count; i++) {
		color_write_enable[i] = VK_TRUE;
	}

	vk::PipelineColorWriteCreateInfoEXT color_write {};
	color_write.attachmentCount    = rendering.color_count;
	color_write.pColorWriteEnables = color_write_enable;

	vk::PipelineColorBlendStateCreateInfo color_blending {};
	// MoltenVK lacks VK_EXT_color_write_enable; drop the dynamic color-write struct on macOS
	// and rely on each attachment's static colorWriteMask (all channels enabled by default).
#if !defined(__APPLE__)
	color_blending.pNext = &color_write;
#endif
	color_blending.logicOp         = vk::LogicOp::eCopy;
	color_blending.attachmentCount = rendering.color_count;
	color_blending.pAttachments    = color_blend_attachment;

	std::vector<vk::DescriptorSetLayoutBinding> descriptor_bindings;
	vk::ShaderStageFlags graphics_stages = vk::ShaderStageFlagBits::eFragment;
	pipeline.uses_bindless = false;
	for (const auto& stage: vertex_info) {
		const auto native_stage = NativeShaderStage(stage.logical_stage);
		AddLayoutBindings(descriptor_bindings, *stage.stage.program, native_stage);
		pipeline.uses_image_heap |= stage.stage.program->info.uses_image_heap;
		graphics_stages |= native_stage;
		pipeline.uses_bindless |= UsesBindless(*stage.stage.program);
	}
	if (ps_active) {
		EXIT_IF(!ps_input_info->stage);
		AddLayoutBindings(descriptor_bindings, *ps_input_info->stage.program,
		                  vk::ShaderStageFlagBits::eFragment);
		pipeline.uses_bindless |= UsesBindless(*ps_input_info->stage.program);
		pipeline.uses_image_heap |= ps_input_info->stage.program->info.uses_image_heap;
	}
	const bool split = Common::EnvRead::GplSplit();
	pipeline.split   = split;
	std::vector<vk::DescriptorSetLayoutBinding> vertex_bindings, pixel_bindings;
	bool                                        vertex_bindless = false, pixel_bindless = false;
	if (split) {
		for (const auto& stage: vertex_info) {
			AddLayoutBindings(vertex_bindings, *stage.stage.program, NativeShaderStage(stage.logical_stage));
			vertex_bindless |= UsesBindless(*stage.stage.program);
		}
		if (ps_active) {
			AddLayoutBindings(pixel_bindings, *ps_input_info->stage.program, vk::ShaderStageFlagBits::eFragment);
			pixel_bindless = UsesBindless(*ps_input_info->stage.program);
		}
	}
	if (!CreateDescriptorLayout(graphics, pipeline, descriptor_bindings, split ? vertex_bindings : descriptor_bindings) ||
	    (background && pipeline.uses_bindless && graphics.bindless_layout == nullptr)) {
		destroy_rect_list_modules();
		return;
	}
	const vk::PushConstantRange push_constants {split ? SplitPushStages(mesh) : graphics_stages, 0,
	                                            ShaderRecompiler::IR::NativePushConstantSize};
	pipeline.push_stages = push_constants.stageFlags;

	std::array<vk::DescriptorSetLayout, 3> set_layouts {};
	uint32_t                               set_layout_count = 0;
	vk::PipelineLayoutCreateFlags          layout_flags {};
	if (split) {
		pipeline.pixel_set_layout = GplSharedSetLayout(graphics, pixel_bindings, false);
		EXIT_IF(pipeline.pixel_set_layout == nullptr);
		EXIT_IF(pipeline.uses_image_heap);
		set_layouts[ShaderRecompiler::IR::SplitVertexSet] = pipeline.descriptor_set_layout;
		set_layouts[ShaderRecompiler::IR::SplitPixelSet]  = pipeline.pixel_set_layout;
		set_layout_count                                  = 2;
		if (pipeline.uses_bindless) {
			EXIT_IF(graphics.bindless_layout == nullptr);
			set_layouts[ShaderRecompiler::IR::SplitBindlessSet] = graphics.bindless_layout;
			set_layout_count                                    = 3;
		}
		if (graphics.graphics_pipeline_library_enabled) layout_flags = vk::PipelineLayoutCreateFlagBits::eIndependentSetsEXT;
	} else {
		std::array<vk::DescriptorSetLayout, 2> merged {};
		FillSetLayouts(graphics, pipeline, merged, set_layout_count);
		set_layouts[0] = merged[0];
		set_layouts[1] = merged[1];
	}
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.flags                  = layout_flags;
	pipeline_layout_info.setLayoutCount         = set_layout_count;
	pipeline_layout_info.pSetLayouts            = set_layouts.data();
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_constants;

	EXIT_IF(pipeline.pipeline_layout != nullptr);

	if (graphics_debug_dump_enabled()) {
		LOGF("PipelineTrace: vkCreatePipelineLayout begin VS=%" PRIu64 " PS=%" PRIu64
		     " set_layouts=1 push_constants=%" PRIu32 "\n",
		     vertex_program.id, ps_active ? pixel_program.id : 0, 1u);
	}
	auto result = graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                   &pipeline.pipeline_layout);
	if (graphics_debug_dump_enabled()) {
		LOGF("PipelineTrace: vkCreatePipelineLayout done result=%s layout=%p\n",
		     vk::to_string(result).c_str(), static_cast<void*>(pipeline.pipeline_layout));
	}
	if (background && (result != vk::Result::eSuccess || pipeline.pipeline_layout == nullptr)) {
		pipeline.pipeline_layout = nullptr;
		destroy_rect_list_modules();
		return;
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	EXIT_NOT_IMPLEMENTED(pipeline.pipeline_layout == nullptr);

	vk::PipelineDepthStencilStateCreateInfo depth_stencil_info {};

	std::vector<vk::DynamicState> dynamic_states {
	    vk::DynamicState::eViewportWithCount,
	    vk::DynamicState::eScissorWithCount,
	    vk::DynamicState::eLineWidth,
	    vk::DynamicState::eDepthTestEnable,
	    vk::DynamicState::eDepthWriteEnable,
	    vk::DynamicState::eDepthCompareOp,
	    vk::DynamicState::eDepthBiasEnable,
	    vk::DynamicState::eDepthBias,
	    vk::DynamicState::eStencilTestEnable,
	    vk::DynamicState::eStencilOp,
	    vk::DynamicState::eStencilCompareMask,
	    vk::DynamicState::eStencilReference,
	    vk::DynamicState::eStencilWriteMask,
	    vk::DynamicState::eBlendConstants,
	};
#if !defined(__APPLE__)
	dynamic_states.push_back(vk::DynamicState::eDepthBoundsTestEnable);
	dynamic_states.push_back(vk::DynamicState::eDepthBounds);
	if (rendering.color_count != 0) {
		dynamic_states.push_back(vk::DynamicState::eColorWriteEnableEXT);
	}
#endif
	if (graphics.attachment_feedback_loop_enabled) {
		dynamic_states.push_back(vk::DynamicState::eAttachmentFeedbackLoopEnableEXT);
	}
	if (pipeline.cull_dynamic) {
		dynamic_states.push_back(vk::DynamicState::eCullMode);
		dynamic_states.push_back(vk::DynamicState::eFrontFace);
	}

	vk::PipelineDynamicStateCreateInfo dynamic_state {};
	dynamic_state.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
	dynamic_state.pDynamicStates    = dynamic_states.data();

	vk::GraphicsPipelineCreateInfo  pipeline_info {};
	vk::PipelineRenderingCreateInfo rendering_info {};
	rendering_info.colorAttachmentCount    = rendering.color_count;
	rendering_info.pColorAttachmentFormats = rendering.color_formats.data();
	rendering_info.depthAttachmentFormat   = rendering.depth_format;
	rendering_info.stencilAttachmentFormat = rendering.stencil_format;
	pipeline_info.pNext                    = &rendering_info;
	pipeline_info.stageCount               = shader_stage_count;
	pipeline_info.pStages                  = shader_stages;
	pipeline_info.pVertexInputState        = mesh ? nullptr : &vertex_input_info;
	pipeline_info.pInputAssemblyState      = mesh ? nullptr : &input_assembly;
	vk::PipelineTessellationStateCreateInfo tessellation_state {};
	tessellation_state.patchControlPoints =
	    tessellation ? vs_input_info.tess.input_control_points : 3u;
	pipeline_info.pTessellationState = (rect_list || tessellation) ? &tessellation_state : nullptr;
	pipeline_info.pViewportState          = &viewport_state;
	pipeline_info.pRasterizationState     = &rasterizer;
	pipeline_info.pMultisampleState       = &multisampling;
	pipeline_info.pDepthStencilState      = (with_depth ? &depth_stencil_info : nullptr);
	pipeline_info.pColorBlendState        = &color_blending;
	pipeline_info.pDynamicState           = &dynamic_state;
	pipeline_info.layout                  = pipeline.pipeline_layout;
	pipeline_info.basePipelineIndex       = -1;

	EXIT_IF(pipeline.pipeline != nullptr);

	if (graphics_debug_dump_enabled()) {
		LOGF("PipelineTrace: vkCreateGraphicsPipelines begin VS=%" PRIu64 " PS=%" PRIu64
		     " topology=%" PRIu32 " color_mask=0x%08" PRIx32
		     " depth=%s blend=%s dyn_states=%" PRIu32 "\n",
		     vertex_program.id, ps_active ? pixel_program.id : 0,
		     static_cast<uint32_t>(static_params.topology), static_params.color_mask[0],
		     (with_depth ? "true" : "false"), (static_params.blend_enable[0] ? "true" : "false"),
		     dynamic_state.dynamicStateCount);
	}
	const uint64_t stats_vs   = vs_input_info.stage.program->shader_hash;
	const uint64_t stats_ps   = ps_active && ps_input_info->stage ? ps_input_info->stage.program->shader_hash : 0;
	const bool     want_stats = PipelineStatsWanted(graphics, stats_vs, stats_ps);
	if (want_stats) {
		pipeline_info.flags |= vk::PipelineCreateFlagBits::eCaptureStatisticsKHR;
	}
	const auto create_begin = std::chrono::steady_clock::now();
	const uint32_t gpl_mode =
	    graphics.graphics_pipeline_library_enabled && !t_gpl_monolithic ? Common::EnvRead::Gpl() : 0u;
	const char* gpl_form  = t_gpl_monolithic ? "opt" : "mono";
	bool        gpl_made  = false;
	if (gpl_mode != 0u) {
		if (rect_list) {
			g_gpl.fb_rect_list.fetch_add(1, std::memory_order_relaxed);
		} else if (mesh) {
			g_gpl.fb_mesh.fetch_add(1, std::memory_order_relaxed);
		} else if (want_stats) {
			g_gpl.fb_stats.fetch_add(1, std::memory_order_relaxed);
		} else {
			if (gpl_mode == 1u) {
				pipeline_info.flags |= vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequired;
				result = graphics.device.createGraphicsPipelines(driver_cache, 1, &pipeline_info, nullptr, &pipeline.pipeline);
				pipeline_info.flags &= ~vk::PipelineCreateFlags(vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequired);
				if (result == vk::Result::ePipelineCompileRequired) {
					pipeline.pipeline = nullptr;
					g_gpl.probe_miss.fetch_add(1, std::memory_order_relaxed);
				} else {
					gpl_made = true;
					gpl_form = "probe";
					if (result == vk::Result::eSuccess) g_gpl.probe_hit.fetch_add(1, std::memory_order_relaxed);
				}
			}
			if (!gpl_made) {
				const uint32_t pre_stage_count = static_cast<uint32_t>(vertex_info.size());
				std::array<ShaderModuleContentHash, 4> module_hashes {};
				bool                                   modules_known = true;
				for (uint32_t i = 0; i < shader_stage_count; i++) {
					module_hashes[i] = ModuleContentHash(shader_stages[i].module);
					modules_known    = modules_known && module_hashes[i].Known();
				}
				vk::PipelineLayout pre_layout = nullptr;
				vk::PipelineLayout fs_layout  = nullptr;
				if (modules_known && split) {
					const auto independent = vk::PipelineLayoutCreateFlags(vk::PipelineLayoutCreateFlagBits::eIndependentSetsEXT);
					const auto vertex_set  = GplSharedSetLayout(graphics, vertex_bindings, pipeline.uses_push_descriptors);
					const std::array<vk::DescriptorSetLayout, 3> pre_sets {vertex_set, nullptr,
					                                                       vertex_bindless ? graphics.bindless_layout : nullptr};
					const std::array<vk::DescriptorSetLayout, 3> fs_sets {nullptr, pipeline.pixel_set_layout,
					                                                      pixel_bindless ? graphics.bindless_layout : nullptr};
					if (vertex_set != nullptr) {
						pre_layout = GplSharedPipelineLayout(
						    graphics, std::span<const vk::DescriptorSetLayout>(pre_sets.data(), vertex_bindless ? 3u : 1u),
						    push_constants, independent);
					}
					fs_layout = GplSharedPipelineLayout(
					    graphics, std::span<const vk::DescriptorSetLayout>(fs_sets.data(), pixel_bindless ? 3u : 2u),
					    push_constants, independent);
				} else if (modules_known) {
					pre_layout = fs_layout = GplLibraryLayout(graphics, pipeline, descriptor_bindings, push_constants).layout;
				}
				struct {
					vk::PipelineLayout layout;
				} library_layout {pre_layout != nullptr && fs_layout != nullptr ? pre_layout : nullptr};
				if (!modules_known) {
					g_gpl.fb_module.fetch_add(1, std::memory_order_relaxed);
				} else if (library_layout.layout == nullptr) {
					g_gpl.fb_library.fetch_add(1, std::memory_order_relaxed);
				} else {
					std::vector<vk::DynamicState> pre_dynamic, fs_dynamic, fo_dynamic;
					for (const auto state: dynamic_states) {
						const char* subset = GplDynamicSubset(state);
						EXIT_IF(subset == nullptr);
						(subset[0] == 'p' ? pre_dynamic : subset[1] == 's' ? fs_dynamic : fo_dynamic).push_back(state);
					}
					const auto dynamic_info = [](const std::vector<vk::DynamicState>& states) {
						vk::PipelineDynamicStateCreateInfo info {};
						info.dynamicStateCount = static_cast<uint32_t>(states.size());
						info.pDynamicStates    = states.data();
						return info;
					};
					const auto pre_dynamic_info = dynamic_info(pre_dynamic);
					const auto fs_dynamic_info  = dynamic_info(fs_dynamic);
					const auto fo_dynamic_info  = dynamic_info(fo_dynamic);
					vk::PipelineRenderingCreateInfo     shader_rendering {};
					vk::PipelineDepthStencilStateCreateInfo library_depth_stencil {};
					const auto create_library = [&](vk::GraphicsPipelineLibraryFlagsEXT subset, vk::GraphicsPipelineCreateInfo info,
					                                const void* rendering_chain) {
						vk::GraphicsPipelineLibraryCreateInfoEXT library_info {};
						library_info.flags = subset;
						library_info.pNext = rendering_chain;
						info.pNext         = &library_info;
						info.flags         = vk::PipelineCreateFlagBits::eLibraryKHR;
						info.basePipelineIndex = -1;
						vk::Pipeline library = nullptr;
						if (graphics.device.createGraphicsPipelines(driver_cache, 1, &info, nullptr, &library) != vk::Result::eSuccess) {
							return vk::Pipeline {};
						}
						return library;
					};
					const auto multisample_key = [&](GplKeyBuilder& b) {
						b.Add(multisampling.rasterizationSamples);
						b.Add(multisampling.sampleShadingEnable);
						b.Add(multisampling.minSampleShading);
						b.Add(multisampling.alphaToCoverageEnable);
						b.Add(multisampling.alphaToOneEnable);
					};
					const auto dynamic_key = [](GplKeyBuilder& b, const std::vector<vk::DynamicState>& states) {
						b.Add(static_cast<uint32_t>(states.size()));
						b.AddBytes(states.data(), states.size() * sizeof(vk::DynamicState));
					};
					GplKeyBuilder vi_builder;
					vi_builder.Add(vertex_input_info.vertexBindingDescriptionCount);
					vi_builder.AddBytes(input_desc, sizeof(input_desc[0]) * vertex_input_info.vertexBindingDescriptionCount);
					vi_builder.Add(vertex_input_info.vertexAttributeDescriptionCount);
					vi_builder.AddBytes(input_attr, sizeof(input_attr[0]) * vertex_input_info.vertexAttributeDescriptionCount);
					vi_builder.Add(input_assembly.topology);
					vi_builder.Add(input_assembly.primitiveRestartEnable);
					GplKeyBuilder pre_builder;
					pre_builder.Add(static_cast<VkPipelineLayout>(pre_layout));
					pre_builder.Add(pre_stage_count);
					for (uint32_t i = 0; i < pre_stage_count; i++) {
						pre_builder.Add(shader_stages[i].stage);
						pre_builder.Add(module_hashes[i]);
					}
					pre_builder.Add(tessellation ? tessellation_state.patchControlPoints : 0u);
					pre_builder.Add(depth_clip_control.negativeOneToOne);
					pre_builder.Add(rasterizer.depthClampEnable);
					pre_builder.Add(rasterizer.pNext != nullptr ? clip_ext.depthClipEnable : VK_TRUE);
					pre_builder.Add(graphics.provoking_vertex_last_enabled ? provoking_vertex.provokingVertexMode
					                                                       : vk::ProvokingVertexModeEXT::eFirstVertex);
					pre_builder.Add(static_cast<uint32_t>(rasterizer.cullMode));
					pre_builder.Add(rasterizer.frontFace);
					pre_builder.Add(rasterizer.polygonMode);
					pre_builder.Add(rasterizer.lineWidth);
					dynamic_key(pre_builder, pre_dynamic);
					GplKeyBuilder fs_builder;
					fs_builder.Add(static_cast<VkPipelineLayout>(fs_layout));
					fs_builder.Add(ps_active);
					fs_builder.Add(ps_active ? module_hashes[shader_stage_count - 1u] : ShaderModuleContentHash {});
					multisample_key(fs_builder);
					dynamic_key(fs_builder, fs_dynamic);
					GplKeyBuilder fo_builder;
					fo_builder.Add(rendering.color_count);
					fo_builder.AddBytes(rendering.color_formats.data(), sizeof(vk::Format) * rendering.color_count);
					fo_builder.Add(rendering.depth_format);
					fo_builder.Add(rendering.stencil_format);
					fo_builder.AddBytes(color_blend_attachment, sizeof(color_blend_attachment[0]) * rendering.color_count);
					fo_builder.Add(color_blending.pNext != nullptr ? color_write.attachmentCount : 0xffffffffu);
					fo_builder.Add(color_blending.logicOp);
					multisample_key(fo_builder);
					dynamic_key(fo_builder, fo_dynamic);

					const auto make_vi = [&] {
						vk::GraphicsPipelineCreateInfo info {};
						info.pVertexInputState   = &vertex_input_info;
						info.pInputAssemblyState = &input_assembly;
						return create_library(vk::GraphicsPipelineLibraryFlagBitsEXT::eVertexInputInterface, info, nullptr);
					};
					const auto make_pre = [&] {
						vk::GraphicsPipelineCreateInfo info {};
						info.stageCount          = pre_stage_count;
						info.pStages             = shader_stages;
						info.pTessellationState  = tessellation ? &tessellation_state : nullptr;
						info.pViewportState      = &viewport_state;
						info.pRasterizationState = &rasterizer;
						info.pDynamicState       = &pre_dynamic_info;
						info.layout              = pre_layout;
						return create_library(vk::GraphicsPipelineLibraryFlagBitsEXT::ePreRasterizationShaders, info,
						                      &shader_rendering);
					};
					const auto make_fs = [&] {
						vk::GraphicsPipelineCreateInfo info {};
						info.stageCount         = ps_active ? 1u : 0u;
						info.pStages            = ps_active ? &shader_stages[shader_stage_count - 1u] : nullptr;
						info.pMultisampleState  = &multisampling;
						info.pDepthStencilState = &library_depth_stencil;
						info.pDynamicState      = &fs_dynamic_info;
						info.layout             = fs_layout;
						return create_library(vk::GraphicsPipelineLibraryFlagBitsEXT::eFragmentShader, info, &shader_rendering);
					};
					const auto make_fo = [&] {
						vk::GraphicsPipelineCreateInfo info {};
						info.pMultisampleState = &multisampling;
						info.pColorBlendState  = &color_blending;
						info.pDynamicState     = &fo_dynamic_info;
						return create_library(vk::GraphicsPipelineLibraryFlagBitsEXT::eFragmentOutputInterface, info,
						                      &rendering_info);
					};
					const auto fast_begin = std::chrono::steady_clock::now();
					std::future<vk::Pipeline> fs_async;
					const auto fs_key = fs_builder.Finish(GplFragment);
					if (!background) {
						fs_async = std::async(std::launch::async, [&] { return GplLibrary(fs_key, make_fs); });
					}
					const vk::Pipeline libraries[4] = {
					    GplLibrary(vi_builder.Finish(GplVertexInput), make_vi), GplLibrary(pre_builder.Finish(GplPreRaster), make_pre),
					    fs_async.valid() ? fs_async.get() : GplLibrary(fs_key, make_fs), GplLibrary(fo_builder.Finish(GplOutput), make_fo)};
					if (std::ranges::any_of(libraries, [](vk::Pipeline p) { return p == nullptr; })) {
						g_gpl.fb_library.fetch_add(1, std::memory_order_relaxed);
					} else {
						vk::PipelineLibraryCreateInfoKHR link_libraries {};
						link_libraries.libraryCount = 4;
						link_libraries.pLibraries   = libraries;
						link_libraries.pNext        = &rendering_info;
						vk::GraphicsPipelineCreateInfo link {};
						link.pNext             = &link_libraries;
						link.layout            = pipeline.pipeline_layout;
						link.basePipelineIndex = -1;
						const auto link_begin  = std::chrono::steady_clock::now();
						result = graphics.device.createGraphicsPipelines(driver_cache, 1, &link, nullptr, &pipeline.pipeline);
						g_gpl.link_us.fetch_add(GplElapsedUs(link_begin), std::memory_order_relaxed);
						if (result == vk::Result::eSuccess && pipeline.pipeline != nullptr) {
							gpl_made = true;
							gpl_form = "fast";
							g_gpl.fast.fetch_add(1, std::memory_order_relaxed);
							g_gpl.fast_us.fetch_add(GplElapsedUs(fast_begin), std::memory_order_relaxed);
							auto pending           = std::make_shared<GplPending>();
							pending->rendering     = rendering;
							pending->vertex_input  = vertex_input;
							pending->static_params = static_params;
							pending->vertex_count  = static_cast<uint32_t>(vertex_info.size());
							for (uint32_t i = 0; i < pending->vertex_count; i++) {
								pending->vertex_info[i]                     = vertex_info[i];
								pending->vertex_info[i].stage.resources     = &g_gpl_no_resources;
								pending->vertex_info[i].stage.stagef_record = nullptr;
							}
							pending->ps_active = ps_active;
							if (ps_active) {
								pending->ps_input_info                     = *ps_input_info;
								pending->ps_input_info.stage.resources     = &g_gpl_no_resources;
								pending->ps_input_info.stage.stagef_record = nullptr;
							}
							pending->programs = programs;
							pipeline.gpl      = std::move(pending);
						} else {
							pipeline.pipeline = nullptr;
							g_gpl.fb_library.fetch_add(1, std::memory_order_relaxed);
						}
					}
				}
			}
		}
	}
	if (!gpl_made) {
		if (gpl_mode != 0u || t_gpl_monolithic) g_gpl.mono.fetch_add(1, std::memory_order_relaxed);
		result = graphics.device.createGraphicsPipelines(driver_cache, 1, &pipeline_info, nullptr, &pipeline.pipeline);
	}
	if (gpl_mode != 0u) GplReport(false);
	// KYTY_TRACE_STARTUP: every graphics pipeline creation (GPU thread or background), 50 ms+ ones also as slow.
	if (Common::StartupTrace::Enabled()) {
		const auto ms =
		    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - create_begin).count();
		using Common::StartupTrace::Counter;
		Common::StartupTrace::AddMs(background ? Counter::PipeBackground : Counter::PipeGraphics, ms);
		if (ms >= 50.0) Common::StartupTrace::Add(Counter::PipeSlow, 0);
	}
	// KYTY_TRACE_PERF: a driver pipeline compile over 50 ms (a pipeline-cache hit takes well under 1 ms); always, one on the
	// GPU thread over 200 ms (the game waits for it: 25.10, stalls of 4-70 s in play were all driver compiles there).
	if (const auto create_ms =
	        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - create_begin).count();
	    (PerfEnabled() && create_ms >= 50.0) || (!background && create_ms >= 200.0)) {
		if (PerfEnabled()) {
			const auto& vs       = *vs_input_info.stage.program;
			const auto* ps       = ps_active && ps_input_info->stage ? ps_input_info->stage.program : nullptr;
			double      miss_age = 0.0;
			const char* miss     = MissPairName(stats_vs, ps_active ? stats_ps : 0u, miss_age);
			std::fprintf(stderr,
			             "PIPELINE_TIME %s vs=0x%016llx ps=0x%016llx %.0fms created_ago=%.0f/%.0fms kind=graphics source=%s "
			             "vs_spirv_words=%u ps_spirv_words=%u vs_form=%s vs_lds=%s vs_inlined=%u/%u ps_inlined=%u/%u gpl=%s "
			             "miss=%s miss_age=%.0f\n",
			             background ? "graphics-background" : "graphics", static_cast<unsigned long long>(stats_vs),
			             static_cast<unsigned long long>(stats_ps), create_ms, ShaderCreatedMsAgo(stats_vs),
			             ShaderCreatedMsAgo(stats_ps), t_gpl_monolithic ? "gpl-optimize" : background ? "precompile" : "runtime",
			             vs.spirv_words, ps != nullptr ? ps->spirv_words : 0u, ShaderRecompiler::IR::EmitFormName(vs.emit_form),
			             ShaderRecompiler::IR::EmitLdsName(vs.emit_lds), static_cast<unsigned>(vs.helper_inline.inlined),
			             static_cast<unsigned>(vs.helper_inline.helpers),
			             ps != nullptr ? static_cast<unsigned>(ps->helper_inline.inlined) : 0u,
			             ps != nullptr ? static_cast<unsigned>(ps->helper_inline.helpers) : 0u, gpl_form, miss, miss_age);
		} else {
			std::fprintf(stderr, "PIPELINE_TIME %s vs=0x%016llx ps=0x%016llx %.0fms created_ago=%.0f/%.0fms\n",
			             background ? "graphics-background" : "graphics", static_cast<unsigned long long>(stats_vs),
			             static_cast<unsigned long long>(stats_ps), create_ms, ShaderCreatedMsAgo(stats_vs),
			             ShaderCreatedMsAgo(stats_ps));
		}
		std::fflush(stderr);
	}
	if (result == vk::Result::eSuccess) { // PIPELINE_REPEAT
		const auto create_ms =
		    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - create_begin).count();
		NotePipelineVariant(stats_vs, stats_ps,
		                    {.vs_id         = vertex_program.id,
		                     .ps_id         = ps_active ? pixel_program.id : 0,
		                     .vs_module     = reinterpret_cast<uint64_t>(static_cast<VkShaderModule>(vertex_program.module)),
		                     .ps_module     = ps_active ? reinterpret_cast<uint64_t>(static_cast<VkShaderModule>(pixel_program.module)) : 0,
		                     .layout_hash   = [&] {
			                     uint64_t h = 1469598103934665603ull;
			                     for (const auto& b: descriptor_bindings) {
				                     for (const uint64_t v: {uint64_t {b.binding}, uint64_t(b.descriptorType), uint64_t {b.descriptorCount},
				                                             uint64_t(static_cast<uint32_t>(b.stageFlags))}) {
					                     h = (h ^ v) * 1099511628211ull;
				                     }
			                     }
			                     return h;
		                     }(),
		                     .rect_list     = rect_list,
		                     .attribute_formats = [&] {
			                     std::array<uint32_t, ShaderVertexInputInfo::RES_MAX> f {};
			                     for (uint32_t i = 0; i < vertex_input.attribute_count; i++) f[i] = static_cast<uint32_t>(input_attr[i].format);
			                     return f;
		                     }(),
		                     .static_params = static_params,
		                     .rendering     = rendering,
		                     .vertex_input  = vertex_input},
		                    create_ms, background);
	}
	if (want_stats && result == vk::Result::eSuccess) {
		PrintPipelineStats(graphics, pipeline.pipeline, background ? "draw-background" : "draw", stats_vs, stats_ps);
	}
	if (graphics_debug_dump_enabled()) {
		LOGF("PipelineTrace: vkCreateGraphicsPipelines done result=%s pipeline=%p\n",
		     vk::to_string(result).c_str(), static_cast<void*>(pipeline.pipeline));
	}
	if (result != vk::Result::eSuccess) {
		std::fprintf(stderr, "vkCreateGraphicsPipelines failed%s: %s VS id=%llu PS id=%llu\n",
		             background ? " (background)" : "", vk::to_string(result).c_str(),
		             static_cast<unsigned long long>(vertex_program.id),
		             static_cast<unsigned long long>(ps_active ? pixel_program.id : 0));
		std::fflush(stderr);
	}
	if (background && (result != vk::Result::eSuccess || pipeline.pipeline == nullptr)) {
		pipeline.pipeline = nullptr;
		destroy_rect_list_modules();
		return;
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	EXIT_NOT_IMPLEMENTED(pipeline.pipeline == nullptr);

	if (tess_control_shader_module != nullptr) {
		graphics.device.destroyShaderModule(tess_control_shader_module, nullptr);
	}
	if (tess_eval_shader_module != nullptr) {
		graphics.device.destroyShaderModule(tess_eval_shader_module, nullptr);
	}
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const ShaderComputeInputInfo& input_info,
                            vk::ShaderModule compute_module, vk::PipelineCache driver_cache) {
	EXIT_IF(compute_module == nullptr);

	vk::PipelineShaderStageCreateInfo                     comp_shader_stage_info {};
	vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo comp_subgroup_size {};
	comp_shader_stage_info.stage  = vk::ShaderStageFlagBits::eCompute;
	comp_shader_stage_info.module = compute_module;
	comp_shader_stage_info.pName  = "main";
	EXIT_IF(!input_info.stage);
	const auto wave_size = input_info.stage.program->wave_size;
	if (graphics.compute_subgroup_size_control_enabled &&
	    wave_size >= graphics.min_subgroup_size && wave_size <= graphics.max_subgroup_size) {
		comp_subgroup_size.requiredSubgroupSize = wave_size;
		comp_shader_stage_info.pNext            = &comp_subgroup_size;
	}

	// A background compile (shaderPrecompile.h) gives up where the GPU thread exits (the caller destroys what
	// `pipeline` holds then).
	const bool background = IsBackgroundCompileThread();
	std::vector<vk::DescriptorSetLayoutBinding> descriptor_bindings;
	AddLayoutBindings(descriptor_bindings, *input_info.stage.program,
	                  vk::ShaderStageFlagBits::eCompute);
	pipeline.uses_image_heap = input_info.stage.program->info.uses_image_heap; // upstream 27dc7c281 (never set here)
	pipeline.uses_bindless = UsesBindless(*input_info.stage.program);
	if (!CreateDescriptorLayout(graphics, pipeline, descriptor_bindings) ||
	    (background && pipeline.uses_bindless && graphics.bindless_layout == nullptr)) {
		return;
	}
	const vk::PushConstantRange push_constants {vk::ShaderStageFlagBits::eCompute, 0,
	                                            ShaderRecompiler::IR::NativePushConstantSize};

	std::array<vk::DescriptorSetLayout, 2> set_layouts {};
	uint32_t                               set_layout_count = 0;
	FillSetLayouts(graphics, pipeline, set_layouts, set_layout_count);
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = set_layout_count;
	pipeline_layout_info.pSetLayouts            = set_layouts.data();
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_constants;

	EXIT_IF(pipeline.pipeline_layout != nullptr);

	LOGF("PipelineTrace: vkCreatePipelineLayout CS begin set_layouts=1 push_constants=%u\n",
	     1u);
	auto result = graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                  &pipeline.pipeline_layout);
	LOGF("PipelineTrace: vkCreatePipelineLayout CS done result=%s layout=%p\n",
	     vk::to_string(result).c_str(), static_cast<void*>(pipeline.pipeline_layout));
	if (background && (result != vk::Result::eSuccess || pipeline.pipeline_layout == nullptr)) {
		pipeline.pipeline_layout = nullptr;
		return;
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	EXIT_NOT_IMPLEMENTED(pipeline.pipeline_layout == nullptr);

	vk::ComputePipelineCreateInfo info {};
	if (input_info.stage.program->info.uses_gds) {
		info.flags = vk::PipelineCreateFlagBits::eDispatchBase;
	}
	info.stage             = comp_shader_stage_info;
	info.layout            = pipeline.pipeline_layout;
	info.basePipelineIndex = -1;

	EXIT_IF(pipeline.pipeline != nullptr);

	if (std::FILE* f = Common::EnvRead::ShaderStagesFile()) { std::fprintf(f, "STAGE cs-pipeline-begin\n"); std::fclose(f); }
	LOGF("PipelineTrace: vkCreateComputePipelines begin layout=%p\n",
	     static_cast<void*>(pipeline.pipeline_layout));
	const uint64_t stats_cs   = input_info.stage.program->shader_hash;
	const bool     want_stats = PipelineStatsWanted(graphics, stats_cs, stats_cs);
	if (want_stats) {
		info.flags |= vk::PipelineCreateFlagBits::eCaptureStatisticsKHR;
	}
	const auto create_begin = std::chrono::steady_clock::now();
	result = graphics.device.createComputePipelines(driver_cache, 1, &info, nullptr,
	                                                &pipeline.pipeline);
	// KYTY_TRACE_STARTUP: every compute pipeline creation (GPU thread or background), 50 ms+ ones also as slow.
	if (Common::StartupTrace::Enabled()) {
		const auto ms =
		    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - create_begin).count();
		using Common::StartupTrace::Counter;
		Common::StartupTrace::AddMs(background ? Counter::PipeBackground : Counter::PipeCompute, ms);
		if (ms >= 50.0) Common::StartupTrace::Add(Counter::PipeSlow, 0);
	}
	if (const auto create_ms =
	        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - create_begin).count();
	    (PerfEnabled() && create_ms >= 50.0) || (!background && create_ms >= 200.0)) {
		if (PerfEnabled()) {
			const auto&    program = *input_info.stage.program;
			const uint32_t threads = std::max(input_info.threads_num[0], 1u) * std::max(input_info.threads_num[1], 1u) *
			                         std::max(input_info.threads_num[2], 1u);
			std::fprintf(stderr,
			             "PIPELINE_TIME %s cs=0x%016llx %.0fms created_ago=%.0fms kind=compute source=%s spirv_words=%u "
			             "form=%s lds=%s lds_dwords=%u threads=%u wave=%u %s\n",
			             background ? "compute-background" : "compute", static_cast<unsigned long long>(stats_cs),
			             create_ms, ShaderCreatedMsAgo(stats_cs), background ? "precompile" : "runtime", program.spirv_words,
			             ShaderRecompiler::IR::EmitFormName(program.emit_form),
			             ShaderRecompiler::IR::EmitLdsName(program.emit_lds), input_info.lds_size_dwords, threads,
			             program.wave_size, ShaderRecompiler::IR::HelperInlineFields(program.helper_inline).data());
		} else {
			std::fprintf(stderr, "PIPELINE_TIME %s cs=0x%016llx %.0fms created_ago=%.0fms\n",
			             background ? "compute-background" : "compute",
			             static_cast<unsigned long long>(stats_cs), create_ms, ShaderCreatedMsAgo(stats_cs));
		}
		std::fflush(stderr);
	}
	if (want_stats && result == vk::Result::eSuccess) {
		PrintPipelineStats(graphics, pipeline.pipeline, "dispatch", stats_cs, 0);
	}
	if (std::FILE* f = Common::EnvRead::ShaderStagesFile()) { std::fprintf(f, "STAGE cs-pipeline-done\n"); std::fclose(f); }
	LOGF("PipelineTrace: vkCreateComputePipelines done result=%s pipeline=%p\n",
	     vk::to_string(result).c_str(), static_cast<void*>(pipeline.pipeline));
	if (background && (result != vk::Result::eSuccess || pipeline.pipeline == nullptr)) {
		pipeline.pipeline = nullptr;
		return;
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	EXIT_NOT_IMPLEMENTED(pipeline.pipeline == nullptr);
}

} // namespace Libs::Graphics
