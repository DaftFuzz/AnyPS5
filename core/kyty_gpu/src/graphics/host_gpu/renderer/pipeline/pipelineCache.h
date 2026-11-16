#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/renderer/dispatchCensus.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/shaderCompiler.h"

#include <shared_mutex>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <type_traits>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <vector>

namespace Common {
class Mutex;
} // namespace Common

namespace Libs::Graphics {

class RenderContext;

[[nodiscard]] uint32_t CompileJobsMode();
[[nodiscard]] bool CompileJobsEnabled();
[[nodiscard]] bool CompileScoutEnabled();
inline std::atomic<uint32_t>   g_exec_compiling {0};
inline std::atomic<void (*)()> g_scout_kick {nullptr};
inline thread_local bool       t_scouting = false;
struct CompileScoutStats {
	std::atomic<uint64_t> stalls {0}, scouted {0}, dispatches {0}, draws {0}, polls {0}, max_ahead {0};
};
inline CompileScoutStats g_compile_scout;
struct CompileStallScope {
	CompileStallScope() {
		if (g_exec_compiling.fetch_add(1, std::memory_order_acq_rel) == 0) {
			if (auto* kick = g_scout_kick.load(std::memory_order_acquire)) kick();
		}
	}
	~CompileStallScope() { g_exec_compiling.fetch_sub(1, std::memory_order_acq_rel); }
	CompileStallScope(const CompileStallScope&)            = delete;
	CompileStallScope& operator=(const CompileStallScope&) = delete;
};

// The stage materializations the resolve thread made for one operation from its register snapshot (opaque;
// pipelineCache.cpp). While the execution thread runs the operation, t_resolved_stages points at them, so a draw takes the
// whole selection (AdoptResolvedPrograms) or ProgramCache::Get takes a matching result; either only when everything the
// resolve thread read still reads the same.
struct ResolvedStages;
[[nodiscard]] ResolvedStages* NewResolvedStages();
void                          DeleteResolvedStages(ResolvedStages* stages);
[[nodiscard]] bool            ResolvedStagesEmpty(const ResolvedStages& stages) noexcept;
// KYTY_CHECK_RESOLVE=2 (project 15 N2.2): SetResolveCheck(true) once at start keeps each stage's active sources for the
// comparison; ResolvedStagesCompare compares two resolutions of one operation: 1 refused (stage count, or a stage not
// ok), 2 reads differ (the read logs), 3 equal, 4 different (`field` names the first difference). ResolvedStagesFault
// puts KYTY_CHECK_RESOLVE_FAULT's error `fault` into a result that is only compared.
void               SetResolveCheck(bool on);
[[nodiscard]] int  ResolvedStagesCompare(const ResolvedStages& a, const ResolvedStages& b, const char** field);
void               ResolvedStagesFault(ResolvedStages& stages, uint32_t fault);
inline thread_local ResolvedStages* t_resolved_stages = nullptr;
// KYTY_PERF_DRAIN_FREE_WAIT=3 (the full variant): the guest bytes a resolved operation can write and the guest bytes its
// resolution read, each as at most a few intervals [first, second) that cover them (supersets: neighbours are joined).
// `known`: every write has a range (no address write outside the record-keyed ranges, no DMA, every written buffer's
// source found ahead).
struct OpWriteRanges {
	static constexpr uint32_t                     MaxWrites = 8;
	static constexpr uint32_t                     MaxReads  = 16;
	bool                                          known     = false;
	uint8_t                                       writes    = 0;
	uint8_t                                       reads     = 0;
	std::array<std::pair<uint64_t, uint64_t>, 8>  write {};
	std::array<std::pair<uint64_t, uint64_t>, 16> read {};
};
// Joins sorted or unsorted intervals into at most `max` covering ones (overlaps and the smallest gaps first); returns the
// count written to `out`.
uint32_t CoverIntervals(std::vector<std::pair<uint64_t, uint64_t>>& in, std::pair<uint64_t, uint64_t>* out, uint32_t max);
// KYTY_TRACE_DISPATCH_CENSUS: how ProgramCache::Get<Compute> got the last compute stage's resources on this thread
// (2 materialized, 3 the resolve thread's result, 4 compiled: a program miss). Not touched when the selection is adopted.
inline thread_local uint32_t t_compute_program_path = 0;

struct GraphicContext;
class ShaderPrecompiler;
struct PrecompileGraphicsRecord;
class PrecompileWarmList;
struct RenderColorInfo;
struct RenderDepthInfo;
class CommandBuffer;
class CommandScheduler;

namespace ShaderRecompiler::IR {
struct ResourcePlan;
} // namespace ShaderRecompiler::IR

namespace HW {
class Context;
class Shader;
class UserConfig;
struct ComputeShaderInfo;
} // namespace HW

namespace ShaderRecompiler::IR {
class ImageHeapSource;
} // namespace ShaderRecompiler::IR
namespace ShaderRecompiler::Spirv {
struct Profile;
} // namespace ShaderRecompiler::Spirv

#pragma pack(push, 1)

struct PipelineStaticParameters {
	bool                       negative_one_to_one      = false;
	bool                       depth_clip_enable        = true;
	vk::PrimitiveTopology      topology                 = vk::PrimitiveTopology::ePointList;
	bool                       primitive_restart_enable = false;
	uint32_t                   samples                  = 1;
	bool                       sample_shading_enable    = false;
	bool                       alpha_to_coverage_enable = false;
	uint32_t                   color_mask[RENDER_COLOR_ATTACHMENTS_MAX]           = {};
	bool                       cull_front                                         = false;
	bool                       cull_back                                          = false;
	bool                       face                                               = false;
	bool                       provoking_vtx_last                                 = false;
	vk::PolygonMode            polygon_mode                                       = vk::PolygonMode::eFill;
	uint8_t                    color_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	uint8_t                    alpha_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	bool                       separate_alpha_blend[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	bool                       blend_enable[RENDER_COLOR_ATTACHMENTS_MAX]         = {};

	bool operator==(const PipelineStaticParameters& other) const noexcept;
};

#pragma pack(pop)

static_assert(std::is_trivially_copyable_v<PipelineStaticParameters>);
static_assert(std::is_standard_layout_v<PipelineStaticParameters>);
static_assert(alignof(PipelineStaticParameters) == 1);
static_assert(sizeof(PipelineStaticParameters) == 117);

struct PipelineRenderingState {
	std::array<vk::Format, RENDER_COLOR_ATTACHMENTS_MAX> color_formats {};
	vk::Format                                           depth_format   = vk::Format::eUndefined;
	vk::Format                                           stencil_format = vk::Format::eUndefined;
	uint32_t                                             color_count    = 0;

	bool operator==(const PipelineRenderingState&) const = default;
};

struct PipelineVertexInputState {
	struct Binding {
		uint32_t stride                           = 0;
		bool     instance                         = false;
		bool     operator==(const Binding&) const = default;
	};
	struct Attribute {
		uint32_t offset                             = 0;
		uint8_t  binding                            = 0;
		bool     operator==(const Attribute&) const = default;
	};

	std::array<Binding, ShaderVertexInputInfo::RES_MAX>   bindings {};
	std::array<Attribute, ShaderVertexInputInfo::RES_MAX> attributes {};
	uint8_t                                               binding_count   = 0;
	uint8_t                                               attribute_count = 0;

	bool operator==(const PipelineVertexInputState&) const = default;
};

struct ShaderProgram {
	uint64_t         id     = 0;
	vk::ShaderModule module = nullptr;

	explicit operator bool() const { return id != 0 && module != nullptr; }
};

struct GplPending;
// KYTY_PERF_GPL: set on a thread that creates the optimized (monolithic) pipeline of a fast-linked one.
inline thread_local bool t_gpl_monolithic = false;

// The owning renderer serializes access, including saves while the GPU is running.
class PipelineCache {
public:
	PipelineCache(GraphicContext& graphics, ShaderRecompiler::IR::ImageHeapSource* image_heap);
	~PipelineCache();
	KYTY_CLASS_NO_COPY(PipelineCache);
	// True when the driver cache was written; a background save returns true once its thread is started.
	bool Save(bool background = false);
	void SaveIfDue();
	void ExportHints(const char* when);
	// --precompile-only (WindowPrecompileOnly): starts the background compiles of the journals at once on `threads`
	// workers (0: KYTY_SHADER_PRECOMPILE_THREADS) and prints PRECOMPILE_* lines on stdout (start, progress every
	// second, periodic saves, done) until every job is through or `stop` is set (the running compiles finish; the queued
	// ones stay for the next run). The driver cache is saved in the background every 6 minutes while compiling (as
	// SaveIfDue) and at the end, each save under `save_mutex`; the warm list then lists what
	// was compiled, so the next run (and the game) leaves it out. Returns the exit code: 0 all done, 3 stopped (what was
	// done is saved), 2 nothing to run (no driver cache or KYTY_SHADER_PRECOMPILE=0), 4 the final save failed.
	[[nodiscard]] int RunPrecompileOnly(uint32_t threads, const std::atomic<bool>& stop, Common::Mutex& save_mutex);

	struct Pipeline {
		vk::PipelineLayout      pipeline_layout       = nullptr;
		vk::Pipeline            pipeline              = nullptr;
		vk::DescriptorSetLayout descriptor_set_layout = nullptr;
		// KYTY_BINDLESS: a stage samples bindless images; descriptor set 1 is the bindless table.
		bool                    uses_bindless         = false;
		bool                    uses_push_descriptors = false;
		// The record packets: the descriptor update template (push descriptors, or the descriptor set of a heap
		// pipeline) CommitBindings creates on first use, and the bytes of descriptor infos it reads (destroyed with the
		// pipeline).
		mutable vk::DescriptorUpdateTemplate push_template       = nullptr;
		mutable uint32_t                     push_template_bytes = 0;
		bool                    uses_image_heap       = false;
		uint32_t                color_count           = 0;
		bool                    cull_dynamic          = false;
		// EvictIdlePipelines: the presented frame of the last lookup that returned this pipeline. An idle pipeline's
		// Vulkan objects are destroyed (pipeline == nullptr) and the next lookup creates them again from the same inputs.
		uint32_t                last_used             = 0;
		// KYTY_PERF_GPL: fast-linked from libraries; its optimized pipeline is made in the background and swapped in.
		std::shared_ptr<GplPending> gpl;
		// KYTY_PERF_GPL_SPLIT: descriptor_set_layout is set 0 (the vertex stages; push descriptors when they fit),
		// pixel_set_layout set 1 (shared, never push; not owned), the bindless table set 2, and the push constants are
		// pushed for push_stages.
		bool                    split            = false;
		vk::DescriptorSetLayout pixel_set_layout = nullptr;
		vk::ShaderStageFlags    push_stages {};
	};

	struct GraphicsPrograms {
		std::array<ShaderProgram, 3> vertex;
		ShaderProgram pixel;

		[[nodiscard]] uint32_t VertexStageCount() const { return vertex[1] ? 3u : 1u; }
	};

	// indirect_draw: GPU indirect vertex variant (ShaderIndirectDrawInfo); it applies to
	// a plain vertex stage only, mesh and tessellation programs ignore it.
	GraphicsPrograms
	GetGraphicsPrograms(const HW::VertexShaderInfo& vertex_regs,
	                    const HW::PixelShaderInfo& pixel_regs, const HW::ShaderRegisters& sh,
	                    const HW::Context& context, const HW::UserConfig& user_config,
	                    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
	                    bool pixel_active, std::array<ShaderVertexInputInfo, 3>& vertex_info,
	                    ShaderPixelInputInfo& pixel_info,
	                    const ShaderIndirectDrawInfo& indirect_draw = {});
	ShaderProgram GetComputeProgram(const HW::ComputeShaderInfo& regs,
	                                const HW::ShaderRegisters&   sh,
	                                ShaderComputeInputInfo&      input_info);
	// KYTY_DIAG_TWIN_DISPATCH (diagnostic, renderCompute.cpp TwinDispatch): GetComputeProgram with `variant_xor` toggled in
	// the program variant (ShaderComputeInputInfo::compute_variant) - the other module of the same dispatch.
	ShaderProgram GetComputeProgramTwin(const HW::ComputeShaderInfo& regs, const HW::ShaderRegisters& sh,
	                                    ShaderComputeInputInfo& input_info, uint8_t variant_xor);

	// ---- The resolve thread ----
	// What GetGraphicsPrograms and GetComputeProgram would select for an operation, from its register snapshots, into
	// `vertex_info`, `pixel_info`, `programs` and `results`: the exact program of each stage, its materialization with the
	// logged reader, FindBuffers' sources (ResolveRecord) and its permutation. They only read the program cache: a stage
	// without a program stays with the execution thread, nothing compiles here (the cache's programs_mutex, taken
	// exclusively by inserts and permutation appends).
	void ResolveGraphicsPrograms(RenderContext& renderer, const HW::VertexShaderInfo& vertex_regs,
	                             const HW::PixelShaderInfo& pixel_regs, const HW::ShaderRegisters& sh,
	                             const HW::Context& context, const HW::UserConfig& user_config,
	                             std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
	                             std::array<ShaderVertexInputInfo, 3>& vertex_info, ShaderPixelInputInfo& pixel_info,
	                             const ShaderIndirectDrawInfo& indirect_draw, GraphicsPrograms& programs,
	                             ResolvedStages& results);
	void ResolveDispatchPrograms(RenderContext& renderer, const HW::ComputeShaderInfo& regs, uint32_t groups_x,
	                             uint32_t groups_y, uint32_t groups_z, uint32_t mode, bool indirect, bool async_compute,
	                             ShaderComputeInputInfo& input_info, ShaderProgram& program, ResolvedStages& results);
	// The execution thread: the whole selection stands in for GetGraphicsPrograms or GetComputeProgram (true), when
	// the resolve thread found every stage and everything it read reads the same now.
	[[nodiscard]] bool AdoptResolvedPrograms(ResolvedStages& results);
	[[nodiscard]] bool AdoptResolvedProgramsChecked(ResolvedStages& results); // KYTY_CHECK_PEEK=2 (project 15 S2d)
	// Project 15 N4.1 (KYTY_CHECK_PEEK=2): PeekAdopt for an operation ahead (the slot keeps the prediction, checked when it
	// is adopted); InstallAhead puts the predicted programs into the slot's state for the other peeks (saved), RestoreAhead
	// puts back what was there.
	struct AheadSaved {
		std::array<const ShaderRecompiler::IR::CompiledShaderInfo*, 4> program {};
		std::array<ShaderProgram, 4>                                   handle {};
		uint32_t                                                       count = 0;
	};
	[[nodiscard]] bool PeekAdoptAhead(ResolvedStages& results, uint32_t distance);
	static void        InstallAhead(ResolvedStages& results, AheadSaved& saved);
	static void        RestoreAhead(ResolvedStages& results, const AheadSaved& saved);
	[[nodiscard]] static bool     StagesAdoptable(const ResolvedStages& results);
	[[nodiscard]] static bool     StagePairMissing(const ResolvedStages& results, uint64_t& vs_hash, uint64_t& ps_hash);
	// KYTY_PERF_DRAIN_FREE_WAIT: whether the resolved stages can write buffer memory: 1 none of them can (every stage has
	// the permutation it will run and neither it nor its source plan writes a buffer, an image, a record-keyed range,
	// through an address or by DMA), 2 a stage can, 3 not known (not adoptable, a stage without its permutation).
	[[nodiscard]] static uint8_t  ResolvedWrites(const ResolvedStages& results);
	// =3: the write and read ranges of resolved stages ResolvedWrites called 2 (OpWriteRanges).
	static void                   ResolvedWriteRanges(const ResolvedStages& results, OpWriteRanges& out);
	[[nodiscard]] static bool     AheadProgramsOk(const ResolvedStages& results);
	[[nodiscard]] static uint32_t StageCount(const ResolvedStages& results);
	[[nodiscard]] static const ShaderRecompiler::IR::CompiledShaderInfo* AheadProgram(const ResolvedStages& results, uint32_t i);
	[[nodiscard]] static const ShaderRecompiler::IR::ResourceSnapshot*   StageResources(const ResolvedStages& results, uint32_t i);
	[[nodiscard]] static const ShaderRecompiler::IR::CompiledShaderInfo* AdoptedProgram(const ResolvedStages& results, uint32_t i);
	[[nodiscard]] static const StageF::ResolveRecord*                    StageRecord(const ResolvedStages& results, uint32_t i);

	Pipeline& GetGraphicsPipeline(std::span<const RenderColorInfo>       colors,
	                              const RenderDepthInfo&                 depth,
	                              std::span<const ShaderVertexInputInfo> vertex_info,
	                              CommandBuffer& command, const ShaderPixelInputInfo* ps_input_info,
	                              vk::PrimitiveTopology topology, bool primitive_restart_enable,
	                              const GraphicsPrograms& programs);
	Pipeline& GetComputePipeline(const ShaderComputeInputInfo& input_info,
	                             const ShaderProgram&          compute_program);

	// KYTY_PERF_TABLE_ASYNC (default on; =0: every draw creates its pipeline on the calling thread, as before 28.10): a
	// graphics pipeline made in the background (renderDraw.cpp ExecutePreparedDraw: every draw that writes no memory asks
	// here). Ready: the pipeline, as GetGraphicsPipeline returns it. Not created yet: its creation is queued on the
	// compile workers (background threads, KYTY_PERF_TABLE_ASYNC_THREADS, default 4) and null is returned; the caller
	// skips the draw. A later draw with the same key finds it ready (the GPU thread moves the finished pipeline in,
	// records it for KYTY_SHADER_PRECOMPILE as GetGraphicsPipeline does). A creation the driver refused (`failed`) makes
	// every draw of that key create it on the GPU thread as before (logged). Counters: "PERF table_async" every 5 s
	// (KYTY_TRACE_PERF). The name is project D's (21.10, the table-mode pipelines were the first made this way).
	// Based on chenxiao07/KytyPS5 v20261020b (src/graphics/host_gpu/renderer/pipeline/pipelineCache.cpp,
	// CompileWorkers, TableWorkers, the pending graphics pipelines of GetGraphicsPipeline), GPL-2.0.
	[[nodiscard]] static bool TableAsyncEnabled();
	void                      ReportCompileJobs() const;
	// `failed` (may be null): set when the worker could not create the pipeline (null is returned for good).
	[[nodiscard]] Pipeline* GetGraphicsPipelineAsync(std::span<const RenderColorInfo>       colors,
	                                                 const RenderDepthInfo&                 depth,
	                                                 std::span<const ShaderVertexInputInfo> vertex_info,
	                                                 CommandBuffer& command, const ShaderPixelInputInfo* ps_input_info,
	                                                 vk::PrimitiveTopology topology, bool primitive_restart_enable,
	                                                 const GraphicsPrograms& programs, bool* failed = nullptr);

private:
	struct ProgramCache;
	struct TableCompileJob;
	class TableCompileWorkers;

public:
	struct GraphicsPipelineKey {
		PipelineRenderingState   rendering;
		std::array<uint64_t, 3>  vertex_shader_ids {};
		uint64_t                 ps_shader_id = 0;
		PipelineVertexInputState vertex_input;
		PipelineStaticParameters static_params;

		bool operator==(const GraphicsPipelineKey& other) const {
			return rendering == other.rendering && vertex_shader_ids == other.vertex_shader_ids &&
			       ps_shader_id == other.ps_shader_id && vertex_input == other.vertex_input &&
			       static_params == other.static_params;
		}
	};

private:
	struct PipelineKeyHash {
		static void Mix(std::size_t& hash, std::size_t value) {
			hash ^= value + static_cast<std::size_t>(0x9e3779b97f4a7c15ull) + (hash << 6u) +
			        (hash >> 2u);
		}
	};

	struct GraphicsPipelineKeyHash {
		std::size_t operator()(const GraphicsPipelineKey& key) const;
	};

	// The key of a draw's graphics pipeline (GetGraphicsPipeline, GetGraphicsPipelineAsync): its programs and the state
	// of the live registers and render targets the pipeline is made with.
	GraphicsPipelineKey MakeGraphicsPipelineKey(std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
	                                            std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
	                                            const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
	                                            bool primitive_restart_enable, const GraphicsPrograms& programs);

public:
	// Project 15 S2c: the same key from the operation's registers. described: the targets come from their description
	// (DescribeRenderColorTarget / DescribeRenderDepthTarget, no image id): every color given is present, the depth is
	// present when its format is defined.
	[[nodiscard]] GraphicsPipelineKey MakeGraphicsPipelineKeyFrom(std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
	                                                              std::span<const ShaderVertexInputInfo> vertex_info,
	                                                              const HW::Context& ctx, const HW::UserConfig& user_config,
	                                                              const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
	                                                              bool primitive_restart_enable, const GraphicsPrograms& programs,
	                                                              bool described);
	// Project 15 S2c: the pipeline of a key without a write. Found: in the map with its objects (the pointer stays valid:
	// entries are never removed); the other classes go in order (today's GetGraphicsPipeline[Async]).
	enum class PipelineFind : uint8_t { Found, Evicted, PendingDone, Pending, Missing, Count };
	[[nodiscard]] std::pair<Pipeline*, PipelineFind> FindGraphicsPipeline(const GraphicsPipelineKey& key, bool read_object = true) const;
	// (read_object false, N5.1: a resolve helper - the entry only, Found when it is in the map; the commit checks the object)
	// N4.3: a plan's pipeline taken as GetGraphicsPipeline[Async] takes a found one (housekeeping, last_used); null when its
	// objects were given back (today's lookup recreates them).
	[[nodiscard]] Pipeline* TakePlannedPipeline(Pipeline* planned);
	[[nodiscard]] size_t GraphicsPipelineCount() const { return m_graphics_pipelines.size() + m_table_pending.size(); }
	void QueueAheadGraphicsPipeline(const GraphicsPipelineKey& key, std::span<const ShaderVertexInputInfo> vertex_info,
	                                const ShaderPixelInputInfo* ps_input_info, const GraphicsPrograms& programs);
	[[nodiscard]] bool PeekStandIn(ResolvedStages& results);
	void QueueAheadGraphicsMiss(const ResolvedStages& results, const GraphicsPipelineKey& key,
	                            std::span<const ShaderVertexInputInfo> vertex_info, const ShaderPixelInputInfo* ps_input_info);
	// KYTY_PERF_COMPILE_JOBS>=3: a draw whose stages include one without a program (results.first_use), planned on a
	// worker: its stages compiled, its key made from the copied registers and the targets described for every slot
	// (colors_by_slot[slot], valid where colors_ok has the bit), its pipeline created; the GPU thread takes it when the
	// pipeline it is about to create has exactly the same stage records and state (=4: both made and compared).
	[[nodiscard]] static bool FirstUsePrebuildWanted(const ResolvedStages& results);
	enum class PrebuildDrop : uint8_t { ShortLead, Shape, ColorOp, Topology, Stages, Code, Seen, Full, Count };
	void NotePrebuildDrop(PrebuildDrop reason);
	void QueueFirstUsePrebuild(const ResolvedStages& results, std::span<const ShaderVertexInputInfo> vertex_info,
	                           const ShaderPixelInputInfo* ps_input_info, const HW::Context& ctx, const HW::UserConfig& user_config,
	                           std::span<const RenderColorInfo> colors_by_slot, uint32_t colors_ok, uint32_t color_output_mask,
	                           const RenderDepthInfo& depth, bool depth_ok, vk::PrimitiveTopology topology, bool restart);

private:
	Pipeline* TakeAheadGraphicsPipeline(const GraphicsPipelineKey& key);
	std::unique_ptr<Pipeline> TakeFirstUsePrebuild(const PrecompileGraphicsRecord& record, uint64_t vs_hash, uint64_t ps_hash,
	                                               uint64_t vs_id, uint64_t ps_id);
	std::mutex m_ahead_workers_mutex;
	std::unordered_set<uint64_t> m_ahead_miss_seen;
	struct FirstUsePrebuilds;
	std::shared_ptr<FirstUsePrebuilds> m_prebuilds;
	void ReportFirstUsePrebuild() const;

	GraphicContext&               m_graphics;
	std::unique_ptr<ProgramCache> m_program_cache;
	vk::PipelineCache             m_driver_cache = nullptr;
	std::chrono::steady_clock::time_point m_last_save {};
	uint64_t                      m_pipelines_since_save = 0; // created since the last driver-cache save
	std::thread                   m_save_thread;              // background write of a periodic save
	// KYTY_SHADER_PRECOMPILE: background compiles of recorded compute permutations and graphics pipelines
	// (shaderPrecompile.h).
	std::unique_ptr<ShaderPrecompiler> m_precompiler;
	// --export-hints / --import-hints: the PRECOMPILE_HINTS line RunPrecompileOnly prints (empty: neither asked), and
	// whether the export was asked (the run then ends after it) or failed.
	std::string                        m_hints_line;
	bool                               m_hints_export = false;
	bool                               m_hints_failed = false;
	std::filesystem::path              m_hints_journal;
	std::filesystem::path              m_hints_graphics_journal;
	std::atomic<uint32_t>              m_precompiled_since_save {0}; // pipelines it created since the last save
	std::filesystem::path         m_driver_cache_path;
	uint64_t                      m_driver_cache_loaded_bytes = 0; // the payload loaded from its file (0: none)
	// The warm list: the recorded pipelines the saved driver cache holds (shaderPrecompile.h); the
	// background save commits it.
	std::shared_ptr<PrecompileWarmList> m_warm;
	std::unordered_map<GraphicsPipelineKey, std::unique_ptr<Pipeline>, GraphicsPipelineKeyHash>
	                                                        m_graphics_pipelines;
	bool m_share_attachments = false;
	std::unordered_map<GraphicsPipelineKey, GraphicsPipelineKey, GraphicsPipelineKeyHash>              m_shared_keys;
	std::unordered_map<GraphicsPipelineKey, std::vector<GraphicsPipelineKey>, GraphicsPipelineKeyHash> m_share_groups;
	struct ShareStats {
		uint64_t alias_hits = 0, session_shared = 0, journal_chosen = 0, created = 0;
		std::chrono::steady_clock::time_point window {};
	} m_share_stats;
	[[nodiscard]] static GraphicsPipelineKey ShareRest(const GraphicsPipelineKey& key);
	Pipeline* FindSharedPipeline(const GraphicsPipelineKey& key, std::span<const ShaderVertexInputInfo> vertex_info,
	                             const ShaderPixelInputInfo* ps_input_info, const GraphicsPrograms& programs);
	void      NoteSharedPipeline(const GraphicsPipelineKey& key);
	void      RestoreRecordedCull(PipelineStaticParameters& static_params, const HW::Context& ctx,
	                              const HW::UserConfig& user_config) const;
	void      ReportShare();
	// N5.1: every change of m_graphics_pipelines and m_table_pending (the execution thread) holds it exclusively; a resolve
	// helper's FindGraphicsPipeline holds it shared. The execution thread's own reads need no lock (it is the only writer).
	mutable std::shared_mutex m_pipelines_mutex;
	std::unordered_map<uint64_t, std::unique_ptr<Pipeline>> m_compute_pipelines;
	// KYTY_PERF_TABLE_ASYNC: the workers (started at the first queued creation) and the creations queued, running,
	// finished but not taken yet, or refused (kept: their draws stay on today's path).
	std::unique_ptr<TableCompileWorkers> m_table_workers;
	std::unordered_map<GraphicsPipelineKey, std::shared_ptr<TableCompileJob>, GraphicsPipelineKeyHash> m_table_pending;
	bool                                 m_table_workers_stopped = false;
	// KYTY_TRACE_MEM_CENSUS: driver-cache bytes loaded at start, and written by the last save (its thread stores it).
	uint64_t                               m_cache_loaded_bytes = 0;
	std::shared_ptr<std::atomic<uint64_t>> m_cache_saved_bytes  = std::make_shared<std::atomic<uint64_t>>(0);

	void InitializeDriverCache();
	void StartBackgroundCompiles();
	// What every request of a pipeline does first: StartBackgroundCompiles and SaveIfDue.
	void PipelineHousekeeping();

public:
	// KYTY_SHADER_PRECOMPILE: joins the background compiles; RenderContext calls it before it destroys the
	// bindless table they may use.
	void StopBackgroundCompiles();

	// KYTY_TRACE_MEM_CENSUS (GPU thread): shader modules this cache keeps for the whole session (each holds its
	// SPIR-V in the driver), pipelines, and the driver-cache bytes loaded and last saved.
	struct HostCensus {
		uint64_t shader_modules = 0, graphics_pipelines = 0, compute_pipelines = 0, cache_loaded = 0, cache_saved = 0;
	};
	[[nodiscard]] HostCensus TakeHostCensus() const;

	// GPU thread, once per frame (RenderContext::RunGarbageCollector): pipelines no lookup returned for
	// PipelineIdleFrames frames give their Vulkan objects back (destroyed once the GPU finished the open tick); a later
	// lookup creates them again (the driver cache has them). Each pipeline is ~0.7 MiB of host memory (24.10):
	// kept until the end of the session before: ~5 GB per hour of play.
	void EvictIdlePipelines(CommandScheduler& scheduler);

private:
	// KYTY_PERF_GPL: fast-linked pipelines of the maps whose optimized pipeline is not swapped in yet, and its workers.
	std::vector<Pipeline*>               m_gpl_live;
	std::unique_ptr<TableCompileWorkers> m_gpl_workers;
	std::unique_ptr<TableCompileWorkers> m_prebuild_workers; // KYTY_PERF_COMPILE_JOBS>=3 (KYTY_PERF_PREBUILD_THREADS, default 3)
	void                                 NoteGplPipeline(Pipeline& pipeline);
	void                                 SwapGplPipelines(CommandScheduler& scheduler);
	uint32_t m_last_eviction = 0;
	uint64_t m_evicted = 0, m_recreated = 0;
	void     RecreateComputePipeline(Pipeline& pipeline, const ShaderComputeInputInfo& input_info,
	                                 const ShaderProgram& compute_program);
	void     RecreateGraphicsPipeline(Pipeline& pipeline, const GraphicsPipelineKey& key,
	                                  std::span<const ShaderVertexInputInfo> vertex_info,
	                                  const ShaderPixelInputInfo* ps_input_info, const GraphicsPrograms& programs);
};

// KYTY_PERF_GPL: the inputs of a fast-linked pipeline (copies; resource snapshots are not read), and the optimized
// pipeline a worker made from them (state 0 waiting for the cache's thread, 1 queued, 2 done).
struct GplPending {
	PipelineRenderingState                 rendering;
	PipelineVertexInputState               vertex_input;
	PipelineStaticParameters               static_params;
	std::array<ShaderVertexInputInfo, 3>   vertex_info {};
	uint32_t                               vertex_count = 1;
	ShaderPixelInputInfo                   ps_input_info {};
	bool                                   ps_active = false;
	PipelineCache::GraphicsPrograms        programs;
	std::atomic<uint32_t>                  state {0};
	PipelineCache::Pipeline                optimized;
	uint64_t                               us = 0;
};
enum class GplOptimizeEvent : uint8_t { Queued, Swapped, Failed, Discarded };
void GplCountOptimize(GplOptimizeEvent event, uint64_t us);
// Destroys the stage libraries and their layouts (after every thread that creates pipelines stopped).
void GplDestroyLibraries(GraphicContext& graphics);
// KYTY_PERF_GPL_SPLIT: the descriptor-set update template of a shared set layout (made once from `entries`, kept until
// GplDestroyLibraries), and the bytes of descriptor infos it reads.
[[nodiscard]] vk::DescriptorUpdateTemplate GplSetTemplate(GraphicContext& graphics, vk::DescriptorSetLayout layout,
                                                          std::span<const vk::DescriptorUpdateTemplateEntry> entries,
                                                          uint32_t& bytes);

void LogPipelineTrace(const char* phase, uint64_t vertex_program_id, uint64_t pixel_program_id);

[[nodiscard]] bool SharedPipelineState(const GraphicContext& graphics);
void               ClearAttachmentSet(PipelineRenderingState& rendering, PipelineStaticParameters& static_params);
void               ClearDynamicCull(PipelineStaticParameters& static_params);
[[nodiscard]] uint32_t AttachmentSetSlots(const PipelineRenderingState& rendering);
[[nodiscard]] bool AttachmentSetCovers(const PipelineRenderingState& cover_rendering, const PipelineStaticParameters& cover_static,
                                       const PipelineRenderingState& rendering, const PipelineStaticParameters& static_params);
void CopyAttachmentSet(const PipelineRenderingState& from_rendering, const PipelineStaticParameters& from_static,
                       PipelineRenderingState& to_rendering, PipelineStaticParameters& to_static);

// The SPIR-V profile of this device's shader compiles (upstream 460a196fc, 286481889). Every compile of a program uses
// it: ProgramCache::Get, and the background compiles of KYTY_SHADER_PRECOMPILE, whose modules the GPU thread reuses.
[[nodiscard]] ShaderRecompiler::Spirv::Profile ShaderCompileProfile(const GraphicContext& graphics);
// KYTY_TRACE_PERF: one "PERF shader_reuse" line (totals since the start) while the shader reuse is on
// (shaderPrecompile.h).
void ReportShaderReusePerf();
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const PipelineRenderingState&          rendering,
                            const PipelineVertexInputState&        vertex_input,
                            std::span<const ShaderVertexInputInfo> vertex_info,
                            const ShaderPixelInputInfo*            ps_input_info,
                            const PipelineCache::GraphicsPrograms& programs,
                            const PipelineStaticParameters&        static_params,
                            vk::PipelineCache                      driver_cache);
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const ShaderComputeInputInfo& input_info,
                            vk::ShaderModule compute_module, vk::PipelineCache driver_cache);
// Why the graphics CreatePipelineInternal would exit on this pipeline on this device, or null: a device feature it
// needs or a value its state helpers reject. The stages' programs must be set. The background compiles
// (shaderPrecompile.h) skip such records: their journal may hold another build's records.
[[nodiscard]] const char* GraphicsPipelineUnsupported(const GraphicContext& graphics,
                                                      const PipelineRenderingState&          rendering,
                                                      const PipelineVertexInputState&        vertex_input,
                                                      std::span<const ShaderVertexInputInfo> vertex_info,
                                                      const ShaderPixelInputInfo*            ps_input_info,
                                                      const PipelineStaticParameters&        static_params);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
