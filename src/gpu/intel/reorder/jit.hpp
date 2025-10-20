/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef GPU_INTEL_REORDER_JIT_HPP
#define GPU_INTEL_REORDER_JIT_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "gpu/gpu_reorder_pd.hpp"
#include "gpu/intel/primitive.hpp"
#include "gpu/intel/reorder/config.hpp"
#include "gpu/intel/reorder/jit/config.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace reorder {

class gen_t : public primitive_t {
public:
    struct pd_t : public reorder::pd_t {
        using reorder::pd_t::pd_t;

        DECLARE_COMMON_PD_T("jit:ir", gen_t);

        status_t init(impl::engine_t *, impl::engine_t *, impl::engine_t *);
        status_t init_kernel_info();

        std::shared_ptr<jit::config_t> cfg;
        std::shared_ptr<jit::kernel_info_t> kernel_info;

    private:
        DECLARE_GPU_REORDER_CREATE();
    };

    using primitive_t::primitive_t;

    status_t init(impl::engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    compute::kernel_t kernel_;
    std::shared_ptr<impl::primitive_t> zp_precomp_conv_;
};

} // namespace reorder
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
