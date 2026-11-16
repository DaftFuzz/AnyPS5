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

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERINFOCOLLECTION_H_ */
