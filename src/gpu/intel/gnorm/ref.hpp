/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GPU_INTEL_GNORM_REF_HPP
#define GPU_INTEL_GNORM_REF_HPP

#include "common/group_normalization_pd.hpp"
#include "gpu/intel/gnorm/config.hpp"
#include "gpu/intel/primitive.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace gnorm {

struct ref_fwd_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public fwd_pd_t {
        using fwd_pd_t::fwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_fwd_t);

        // check data types compatibility and initialize `dispatch`
        status_t init(impl::engine_t *engine);

        status_t init_conf(impl::engine_t *engine);
        // define kernel compile time environment
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        // kernel runtime nd_range
        compute::dispatch_t dispatch;
    };

    status_t init(impl::engine_t *engine) override {
        if (pd()->has_zero_dim_memory()) return status::success;

        compute::kernel_ctx_t kernel_ctx;

        CHECK(pd()->init_kernel_ctx(kernel_ctx));
        CHECK(create_kernel(engine, &kernel_, "ref_gnorm_fwd", kernel_ctx));

        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    compute::kernel_t kernel_;
};

struct ref_bwd_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public bwd_pd_t {
        using bwd_pd_t::bwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_bwd_t);

        // check data types compatibility and initialize `dispatch`
        status_t init(impl::engine_t *engine);

        // define kernel compile time environment
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        // kernel runtime nd_range
        compute::dispatch_t dispatch;
    };

    status_t init(impl::engine_t *engine) override {
        if (pd()->has_zero_dim_memory()) return status::success;

        compute::kernel_ctx_t kernel_ctx;

        CHECK(pd()->init_kernel_ctx(kernel_ctx));
        CHECK(create_kernel(engine, &kernel_, "ref_gnorm_bwd", kernel_ctx));

        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    compute::kernel_t kernel_;
};

} // namespace gnorm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
