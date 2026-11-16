#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERINFOCOLLECTION_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERINFOCOLLECTION_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

// Collects the shader interface, resource usage and instruction features after resource tracking.
void CollectShaderInfo(Program& program, ShaderStageInputInfo input_info);

// Whether every typed attribute read of a vertex or pixel program is inside the stage's input count (the condition
// CollectShaderInfo ends the emulator on). A background compile skips a record that fails it: `index` gets the first
// attribute out of range and `count` the input count it was recorded with.
[[nodiscard]] bool StageInputsFit(const Program& program, ShaderStageInputInfo input_info, uint32_t& index, uint32_t& count);

// A background compile's scope (calling thread only): a record CollectShaderInfo would end the emulator on throws
// ShaderInfoRejected instead, so the compile skips the record; the GPU thread's own compiles still stop as before.
struct ShaderInfoRejected {
	const char* message = nullptr;
};
class ShaderInfoSoftFailScope {
public:
	ShaderInfoSoftFailScope();
	~ShaderInfoSoftFailScope();
	ShaderInfoSoftFailScope(const ShaderInfoSoftFailScope&)            = delete;
	ShaderInfoSoftFailScope& operator=(const ShaderInfoSoftFailScope&) = delete;

private:
	bool m_previous = false;
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERINFOCOLLECTION_H_ */
