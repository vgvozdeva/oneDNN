/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef GPU_INTEL_RESAMPLING_REF_HPP
#define GPU_INTEL_RESAMPLING_REF_HPP

#include "gpu/intel/primitive.hpp"
#include "gpu/intel/resampling/config.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace resampling {

struct ref_fwd_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public fwd_pd_t {
        using fwd_pd_t::fwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_fwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            assert(engine->kind() == engine_kind::gpu);
            using sm = primitive_attr_t::skip_mask_t;
            const auto attr_skip_mask = sm::post_ops;

            VDISPATCH_RESAMPLING(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_RESAMPLING_SC(
                    set_default_params(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_RESAMPLING(attr()->has_default_values(attr_skip_mask),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_RESAMPLING(post_ops_with_binary_ok(attr(), *dst_md(), 5),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_RESAMPLING_SC(attr_.set_default_formats(dst_md(0)),
                    VERBOSE_UNSUPPORTED_TAG);
            CHECK(init_conf(engine));
            return status::success;
        }
        compute::dispatch_t dispatch;
        conf_t conf;

        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;
        status_t init_conf(impl::engine_t *engine);
    };

    status_t init(impl::engine_t *engine) override {
        using namespace alg_kind;

        compute::kernel_ctx_t kernel_ctx;
        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(status);

        CHECK(create_kernel(
                engine, &kernel_, "ref_resampling_fwd", kernel_ctx));
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

struct ref_bwd_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public bwd_pd_t {
        using bwd_pd_t::bwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_bwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            assert(engine->kind() == engine_kind::gpu);

            VDISPATCH_RESAMPLING(!is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_RESAMPLING_SC(
                    set_default_params(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_RESAMPLING(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            CHECK(init_conf(engine));
            return status::success;
        }
        conf_t conf;

        status_t init_conf(impl::engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;
    };

    status_t init(impl::engine_t *engine) override {
        using namespace alg_kind;

        compute::kernel_ctx_t kernel_ctx;
        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(status);

        CHECK(create_kernel(
                engine, &kernel_, "ref_resampling_bwd", kernel_ctx));
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward(ctx);
    }

private:
    status_t execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

} // namespace resampling
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
