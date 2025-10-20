/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef GPU_INTEL_REDUCTION_REF_HPP
#define GPU_INTEL_REDUCTION_REF_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"
#include "gpu/intel/primitive.hpp"
#include "gpu/intel/reduction/config.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace reduction {

struct ref_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public reduction::pd_t {
        using reduction::pd_t::pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_t);

        status_t init(impl::engine_t *engine) {
            using sm = primitive_attr_t::skip_mask_t;
            const auto attr_skip_mask = sm::post_ops | sm::gpu_attr;

            VDISPATCH_REDUCTION_SC(
                    set_default_params(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_REDUCTION(memory_desc_ndims_ok(src_md(), dst_md()),
                    VERBOSE_INCONSISTENT_NDIMS_WITH_VALS, "src", "dst",
                    src_md()->ndims, dst_md()->ndims);
            VDISPATCH_REDUCTION(attr()->has_default_values(attr_skip_mask),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_REDUCTION(post_ops_with_binary_ok(attr(), *dst_md(), 5),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_REDUCTION_SC(attr_.set_default_formats(dst_md(0)),
                    VERBOSE_UNSUPPORTED_TAG);
            CHECK(init_conf(engine));
            return status::success;
        }

        status_t init_conf(impl::engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        conf_t conf;
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(status);

        status = create_kernel(engine, &kernel, "ref_reduce", kernel_ctx);
        CHECK(status);

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_ref(ctx);
    }

private:
    status_t execute_ref(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    compute::kernel_t kernel;
};

} // namespace reduction
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
