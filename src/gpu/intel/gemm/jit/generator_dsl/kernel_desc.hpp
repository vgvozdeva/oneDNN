/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#ifndef GEMMSTONE_GUARD_KERNEL_DESC_HPP
#define GEMMSTONE_GUARD_KERNEL_DESC_HPP

#include "gemmstone/problem.hpp"
#include "gemmstone/strategy.hpp"
#include "gpu/intel/jit/ir/kernel_desc.hpp"
#include "gpu/intel/jit/ir/kernel_info.hpp"

namespace gemmstone {

struct generator_dsl_desc_t {
    generator_dsl_desc_t(const GEMMProblem &problem,
            const GEMMStrategy &strategy,
            const ngen::InterfaceHandler &ngen_iface, const ir::hw_t &hw)
        : problem(problem)
        , strategy(strategy)
        , iface(ngen_iface)
        , options(hw, strategy.GRFs, strategy.subgroupSize) {}

    const std::string &kernel_name() const { return iface.kernel_name(); }
    const ir::kernel::iface_t &kernel_iface() const { return iface; }

    const GEMMProblem &problem;
    const GEMMStrategy &strategy;
    ir::kernel::iface_t iface;
    ir::kernel::options_t options;
};

// Not all strategies parameters are supported via DSL. This attempts to fixup
// strategies to enable inter-operation with existing strategies.
inline void fixup_dsl_strategy(GEMMStrategy &strategy) {
    if (strategy.kParallel) {
        strategy.kParallel = false;
        strategy.C.atomic = false;
        strategy.CO.atomic = false;
    }
    if (strategy.kParallelLocal) {
        strategy.kParallelLocal = false;
        strategy.kInterleave = false;
    }
};

} // namespace gemmstone

#endif
