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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/ref_io_helper.hpp"
#include "cpu/scale_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

namespace {
constexpr size_t scales_simd_w = 16;
}

void book_precomputed_scales(memory_tracking::registrar_t &scratchpad,
        const scales_t &attr_scales, size_t wei_scale_count,
        float scale_adjust_factor, bool req_transpose) {
    if (req_copy_scales(attr_scales, scale_adjust_factor, req_transpose)) {
        const int wei_mask = attr_scales.get_mask(DNNL_ARG_WEIGHTS);
        // Current infrastructure relies on the fact the precomputed scales
        // output will be float values. However, changes in this commit update
        // the expectation for matmul grouped scaling.
        // TODO: once precompute scales infrastructure remains to be used in
        // brgemm_matmul only (for x64), update the interface with requested
        // data type for scales, since it'll be no longer true that float values
        // are expected.
        const size_t precomputed_scales_size = wei_mask > 0
                ? nstl::max(static_cast<size_t>(wei_scale_count), scales_simd_w)
                : scales_simd_w;
        scratchpad.template book<float>(
                memory_tracking::names::key_precomputed_scales,
                precomputed_scales_size);
    }
}

bool req_copy_scales(const scales_t &attr_scales, float scale_adjust_factor,
        bool req_transpose) {
    const bool with_src_scales = !attr_scales.has_default_values(DNNL_ARG_SRC);
    const bool with_wei_scales
            = !attr_scales.has_default_values(DNNL_ARG_WEIGHTS);

    return (with_src_scales && with_wei_scales)
            || scale_adjust_factor != 1.0f
            // When scales are transposed, it must be handled before the kernel
            || req_transpose
            || !attr_scales.has_default_data_type(DNNL_ARG_WEIGHTS)
            || !attr_scales.get(DNNL_ARG_WEIGHTS).has_default_groups();
}

const float *precompute_scales(const memory_tracking::grantor_t &scratchpad,
        const float *src_scales, const float *wei_scales, dim_t oc,
        const primitive_attr_t *attr, float scale_adjust_factor) {
    // Note: per-ic-channel is no supported by default.
    const int wei_scale_mask = attr->scales_.get_mask(DNNL_ARG_WEIGHTS);
    return precompute_scales(scratchpad, src_scales, wei_scales, 1, oc, false,
            wei_scale_mask > 0, attr, scale_adjust_factor, false);
}

// Note: `wei_scale_per_ic` and `wei_scale_per_oc` could be identified in this
// function unless different primitives have same definition of `per_ic` and
// `per_oc` masks. Mostly, matmul is different from anybody else.
const float *precompute_scales(const memory_tracking::grantor_t &scratchpad,
        const float *src_scales, const float *wei_scales, dim_t IC, dim_t OC,
        const bool wei_scale_per_ic, const bool wei_scale_per_oc,
        const primitive_attr_t *attr, float scale_adjust_factor,
        bool req_transpose) {
    using namespace dnnl::impl::memory_tracking::names;

    const auto &attr_scales = attr->scales_;
    const bool with_src_scales = !attr_scales.has_default_values(DNNL_ARG_SRC);
    const auto wei_scale_count
            = (wei_scale_per_ic ? IC : 1) * (wei_scale_per_oc ? OC : 1);

    const float *scales = nullptr;
    if (req_copy_scales(attr_scales, scale_adjust_factor, req_transpose)) {
        size_t size = 0;
        auto loc_scales
                = scratchpad.template get<float>(key_precomputed_scales, &size);
        if (wei_scale_count == 1) {
            const size_t count = nstl::min(size / sizeof(float), scales_simd_w);
            utils::array_set(loc_scales,
                    src_scales[0] * wei_scales[0] * scale_adjust_factor, count);
        } else {
            const dim_t count = nstl::min(
                    static_cast<dim_t>(size / sizeof(float)), wei_scale_count);
            const auto wei_scale_dt
                    = attr_scales.get_data_type(DNNL_ARG_WEIGHTS);
            const auto wei_scale_groups_ic
                    = attr_scales.get_group(DNNL_ARG_WEIGHTS, 0);
            // Note: per-ic-channel scales is only supported for
            // weights decompression for now
            if ((wei_scale_per_ic && wei_scale_groups_ic > 1)
                    || req_transpose) {
                const auto wei_scale_stride_ic
                        = wei_scale_per_ic ? wei_scale_per_oc ? OC : 1 : 0;
                const auto wei_scale_stride_oc = wei_scale_per_oc ? 1 : 0;
                assert(count == wei_scale_count);
                PRAGMA_OMP_SIMD()
                for_(int ic = 0; ic < IC; ic++)
                for (int oc = 0; oc < wei_scale_stride_ic; oc++) {
                    const auto wei_scale_idx = wei_scale_stride_oc * oc
                            + wei_scale_stride_ic * (ic / wei_scale_groups_ic);
                    const auto loc_scale_idx
                            = req_transpose ? oc * IC + ic : ic * OC + oc;
                    const float wei_scales_val = io::load_float_value(
                            wei_scale_dt, wei_scales, wei_scale_idx);
                    loc_scales[loc_scale_idx] = src_scales[0] * wei_scales_val
                            * scale_adjust_factor;
                }
            } else {
                PRAGMA_OMP_SIMD()
                for (dim_t c = 0; c < count; c++) {
                    const float wei_scales_val
                            = io::load_float_value(wei_scale_dt, wei_scales, c);
                    loc_scales[c] = src_scales[0] * wei_scales_val
                            * scale_adjust_factor;
                }
            }
        }
        scales = loc_scales;
    } else if (with_src_scales) {
        scales = src_scales;
    } else {
        scales = wei_scales;
    }

    return scales;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
