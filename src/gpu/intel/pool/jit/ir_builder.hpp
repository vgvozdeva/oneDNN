/*******************************************************************************
* Copyright 2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GPU_INTEL_POOL_JIT_IR_BUILDER_HPP
#define GPU_INTEL_POOL_JIT_IR_BUILDER_HPP

#include "gpu/intel/jit/ir/builder.hpp"
#include "gpu/intel/jit/ir/kernel_info.hpp"
#include "gpu/intel/pool/jit/config.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace pool {
namespace jit {

class builder_t : public ir_builder_t {
public:
    builder_t(config_t &cfg, const kernel_info_t &ki,
            const primitive_desc_t &pd) {
        while ((stmt_ = try_build(*this, ki, cfg, pd)).is_empty()) {
            gpu_warning() << "loop too large: cut and retry!";
            const bool cut_ok = cfg.cut();
            if (!cut_ok) gpu_error_not_expected() << "minimal loop too large!";
        }
    }

private:
    void build() override {}
    static stmt_t try_build(builder_t &pb, const kernel_info_t &ki,
            const config_t &cfg, const primitive_desc_t &pd);
};

} // namespace jit
} // namespace pool
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
