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

#ifndef GPU_INTEL_CONV_JIT_V2_HPP
#define GPU_INTEL_CONV_JIT_V2_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "gpu/gpu_convolution_pd.hpp"
#include "gpu/intel/conv/config.hpp"
#include "gpu/intel/conv/jit/v2/primitive_plan.hpp"
#include "gpu/intel/jit/ir/kernel_desc.hpp"
#include "gpu/intel/primitive.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace conv {
namespace v2 {

using namespace conv::jit;
using namespace conv::jit::v2;

class gen_t;

class gen_fwd_t : public primitive_t {
public:
    friend gen_t;
    friend primitive_init_plan_t;
    struct pd_t : public fwd_pd_t {
        friend gen_t;
        using fwd_pd_t::fwd_pd_t;

        DECLARE_COMMON_PD_T("jit:ir_v2", gen_fwd_t);
        status_t init(impl::engine_t *engine);

        std::shared_ptr<primitive_init_plan_t> init_plan;
    };

    using primitive_t::primitive_t;

    status_t init(impl::engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::shared_ptr<gen_t> impl_;
};

class gen_bwd_data_t : public primitive_t {
public:
    friend gen_t;
    friend primitive_init_plan_t;
    struct pd_t : public bwd_data_pd_t {
        friend gen_t;
        using bwd_data_pd_t::bwd_data_pd_t;

        DECLARE_COMMON_PD_T("jit:ir_v2", gen_bwd_data_t);
        status_t init(impl::engine_t *engine);

        std::shared_ptr<primitive_init_plan_t> init_plan;
    };

    using primitive_t::primitive_t;

    status_t init(impl::engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::shared_ptr<gen_t> impl_;
};

class gen_bwd_weights_t : public primitive_t {
public:
    friend gen_t;
    friend primitive_init_plan_t;
    struct pd_t : public bwd_weights_pd_t {
        friend gen_t;
        using bwd_weights_pd_t::bwd_weights_pd_t;

        DECLARE_COMMON_PD_T("jit:ir_v2", gen_bwd_weights_t);
        status_t init(impl::engine_t *engine);

        std::shared_ptr<primitive_init_plan_t> init_plan;
    };

    using primitive_t::primitive_t;

    status_t init(impl::engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::shared_ptr<gen_t> impl_;
};

} // namespace v2
} // namespace conv
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
