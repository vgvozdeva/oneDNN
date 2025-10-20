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

#include <assert.h>
#include "opdesc.hpp"
#include "primitive_desc_iface.hpp"

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::types;

#define VCHECK_MATMUL(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, matmul, (cond), \
            status::invalid_arguments, msg, ##__VA_ARGS__);

#define VCHECK_MATMUL_UNIMPL(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, matmul, (cond), \
            status::unimplemented, msg, ##__VA_ARGS__);

namespace {
status_t matmul_attr_check(const matmul_desc_t &desc, const engine_t *engine,
        const primitive_attr_t *attr) {
    using smask_t = primitive_attr_t::skip_mask_t;

    if (attr == nullptr) return status::success;
    if (attr->has_default_values()) return status::success;

    // Check attributes
    const data_type_t src_dt = desc.src_desc.data_type;
    const data_type_t wei_dt = desc.weights_desc.data_type;
    const data_type_t dst_dt = desc.dst_desc.data_type;

    auto attr_mask = smask_t::post_ops | smask_t::sum_dt | smask_t::dropout
            | smask_t::rounding_mode;
    // Matmul supports scales for floating point data types
    attr_mask |= smask_t::scales_data_type;

    const bool src_is_int8
            = utils::one_of(src_dt, data_type::s8, data_type::u8);
    const bool src_is_fp8
            = utils::one_of(src_dt, data_type::f8_e5m2, data_type::f8_e4m3);
    const bool src_is_fp4
            = utils::one_of(src_dt, data_type::f4_e2m1, data_type::f4_e3m0);
    if (src_is_int8 || src_is_fp8 || src_is_fp4)
        attr_mask |= smask_t::zero_points;
    if (src_is_int8) attr_mask |= smask_t::precomputed_reductions;

    // Matmul supports zero points for floating point data types as part of
    // weights decompression.
    const bool wei_is_int = utils::one_of(
            wei_dt, data_type::s8, data_type::u8, data_type::s4, data_type::u4);
    const bool wei_is_fp8
            = utils::one_of(wei_dt, data_type::f8_e5m2, data_type::f8_e4m3);
    const bool wei_is_fp4
            = utils::one_of(wei_dt, data_type::f4_e2m1, data_type::f4_e3m0);
    if (wei_is_int || wei_is_fp8 || wei_is_fp4) {
        attr_mask |= smask_t::zero_points_data_type;
        attr_mask |= smask_t::zero_points_groups;
        attr_mask |= smask_t::scales_groups;
    }

    const bool dst_is_fp8
            = utils::one_of(dst_dt, data_type::f8_e5m2, data_type::f8_e4m3);
    const bool dst_is_fp4
            = utils::one_of(dst_dt, data_type::f4_e2m1, data_type::f4_e3m0);
    // grouped dst scales are supported for mxfp
    if (dst_is_fp8 || dst_is_fp4) attr_mask |= smask_t::scales_groups;

    // Matmul supports fpmath mode and accumulation mode
    attr_mask |= smask_t::fpmath_mode | smask_t::accumulation_mode;

    VCHECK_MATMUL_UNIMPL(attr->has_default_values(attr_mask, dst_dt),
            VERBOSE_UNSUPPORTED_ATTR);

    const int ndims_src = desc.src_desc.ndims;
    const int ndims_wei = desc.weights_desc.ndims;
    const int m_idx = ndims_src - 2;
    const int k_idx_wei = m_idx;
    const int n_idx = ndims_wei - 1;
    const dim_t K = desc.weights_desc.dims[k_idx_wei];
    const dim_t N = desc.weights_desc.dims[n_idx];

    assert(ndims_src >= 2);
    assert(ndims_wei >= 2);
    int src_qmask_M = 1 << (ndims_src - 2);
    int src_qmask_K = 1 << (ndims_src - 1);

    int wei_qmask_K = 1 << (ndims_wei - 2);
    int wei_qmask_N = 1 << (ndims_wei - 1);

    int dst_qmask_M = src_qmask_M;
    int dst_qmask_N = wei_qmask_N;

    int full_tensor_mask = (1 << ndims_src) - 1;

    const auto &quant_groups_are_divisible = [](dim_t g1, dim_t g2) -> bool {
        return IMPLICATION(g1 > 1 && g2 > 1, (g1 % g2 == 0) || (g2 % g1 == 0));
    };

    // Check scales
    if (!attr->scales_.has_default_values()) {
        const auto &sc = attr->scales_;

        dim_t src_scale_group_k = 1;
        if (!sc.has_default_values(DNNL_ARG_SRC)) {
            const int mask_src = sc.get_mask(DNNL_ARG_SRC);

            VCHECK_MATMUL_UNIMPL(
                    utils::one_of(mask_src, 0, src_qmask_K,
                            src_qmask_M + src_qmask_K, full_tensor_mask),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);

            if (!sc.get(DNNL_ARG_SRC).has_default_groups()) {
                if (mask_src & src_qmask_K)
                    src_scale_group_k = sc.get_group(DNNL_ARG_SRC, 1);
            }

            // Due to hardware specifics, groups, when more than 1, should be
            // multiple of 32.
            VCHECK_MATMUL_UNIMPL(
                    IMPLICATION(src_scale_group_k > 1 && src_scale_group_k < K,
                            src_scale_group_k % 32 == 0),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
        }

        dim_t wei_scale_group_k = 1;
        dim_t wei_scale_group_n = 1;
        if (!sc.has_default_values(DNNL_ARG_WEIGHTS)) {
            const int mask_wei = sc.get_mask(DNNL_ARG_WEIGHTS);

            // Masks for weights scales can be any - skipping them.

            if (!sc.get(DNNL_ARG_WEIGHTS).has_default_groups()) {
                if (mask_wei & wei_qmask_K)
                    wei_scale_group_k = sc.get_group(DNNL_ARG_WEIGHTS, 0);
                if (mask_wei & wei_qmask_N)
                    wei_scale_group_n = sc.get_group(DNNL_ARG_WEIGHTS, 1);
            }

            // Groups per N are solely for weights decompression as it's
            // impossible to get performant kernel for a single `k` element in
            // chain for regular quantized case.
            VCHECK_MATMUL_UNIMPL(IMPLICATION(wei_scale_group_n > 1,
                                         attr->fpmath_.apply_to_int_),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);

            // Due to hardware specifics, groups, when more than 1, should be
            // multiple of 16.
            VCHECK_MATMUL_UNIMPL(
                    IMPLICATION(wei_scale_group_k > 1 && wei_scale_group_k < K,
                            wei_scale_group_k % 16 == 0),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
            VCHECK_MATMUL_UNIMPL(
                    IMPLICATION(wei_scale_group_n > 1 && wei_scale_group_n < N,
                            wei_scale_group_n % 16 == 0),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
        }

        if (!sc.has_default_values(DNNL_ARG_DST)) {
            const int mask_dst = sc.get_mask(DNNL_ARG_DST);
            VCHECK_MATMUL_UNIMPL(
                    utils::one_of(mask_dst, 0, dst_qmask_N, dst_qmask_M,
                            dst_qmask_N + dst_qmask_M, full_tensor_mask),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
        }

        // Check dependency between scales.
        // Source scales groups are supported for int8 source and must divide
        // or be divided by weights groups when both are greater than 1.
        const bool groups_are_divisible = quant_groups_are_divisible(
                src_scale_group_k, wei_scale_group_k);
        VCHECK_MATMUL_UNIMPL(IMPLICATION(src_scale_group_k > 1,
                                     (src_is_int8 || src_is_fp8 || src_is_fp4)
                                             && groups_are_divisible),
                VERBOSE_UNSUPPORTED_SCALES_CFG);

        // For dynamic scaling, we support only OCP MX flavor
        if (sc.get(DNNL_ARG_DST).is_mx()) {
            // only group size of 32
            VCHECK_MATMUL_UNIMPL(sc.get_mask(DNNL_ARG_DST) == full_tensor_mask,
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
            VCHECK_MATMUL_UNIMPL(sc.get_group(DNNL_ARG_DST, -1) == 32
                            && sc.get_group(DNNL_ARG_DST, -2) == 1,
                    VERBOSE_UNSUPPORTED_SCALES_CFG);

            // only e8m0 scales
            VCHECK_MATMUL_UNIMPL(
                    sc.get_data_type(DNNL_ARG_DST) == data_type::e8m0,
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
        }
    }

    // Check zero points
    if (!attr->zero_points_.has_default_values()) {
        const auto &zp = attr->zero_points_;

        dim_t src_zero_point_group_k = 1;
        if (!zp.has_default_values(DNNL_ARG_SRC)) {
            const int mask_src = zp.get_mask(DNNL_ARG_SRC);

            VCHECK_MATMUL_UNIMPL(utils::one_of(mask_src, 0, src_qmask_K,
                                         src_qmask_M + src_qmask_K),
                    VERBOSE_UNSUPPORTED_ZP_CFG);

            if (!zp.get(DNNL_ARG_SRC).has_default_groups()) {
                if (mask_src & src_qmask_K)
                    src_zero_point_group_k = zp.get_group(DNNL_ARG_SRC, 1);
            }

            // Due to hardware specifics, groups, when more than 1, should be
            // multiple of 32.
            VCHECK_MATMUL_UNIMPL(IMPLICATION(src_zero_point_group_k > 1
                                                 && src_zero_point_group_k < K,
                                         src_zero_point_group_k % 32 == 0),
                    VERBOSE_UNSUPPORTED_ZP_CFG);
        }

        dim_t wei_zero_point_group_k = 1;
        dim_t wei_zero_point_group_n = 1;
        if (!zp.has_default_values(DNNL_ARG_WEIGHTS)) {
            const int mask_wei = zp.get_mask(DNNL_ARG_WEIGHTS);

            // Masks for weights zero_points can be any - skipping them.

            if (!zp.get(DNNL_ARG_WEIGHTS).has_default_groups()) {
                if (mask_wei & wei_qmask_K)
                    wei_zero_point_group_k = zp.get_group(DNNL_ARG_WEIGHTS, 0);
                if (mask_wei & wei_qmask_N)
                    wei_zero_point_group_n = zp.get_group(DNNL_ARG_WEIGHTS, 1);
            }

            // Groups per N are solely for weights decompression as it's
            // impossible to get performant kernel for a single `k` element in
            // chain for regular quantized case.
            VCHECK_MATMUL_UNIMPL(IMPLICATION(wei_zero_point_group_n > 1,
                                         attr->fpmath_.apply_to_int_),
                    VERBOSE_UNSUPPORTED_ZP_CFG);

            // Due to hardware specifics, groups, when more than 1, should be
            // multiple of 16.
            VCHECK_MATMUL_UNIMPL(IMPLICATION(wei_zero_point_group_k > 1
                                                 && wei_zero_point_group_k < K,
                                         wei_zero_point_group_k % 16 == 0),
                    VERBOSE_UNSUPPORTED_ZP_CFG);
            VCHECK_MATMUL_UNIMPL(IMPLICATION(wei_zero_point_group_n > 1
                                                 && wei_zero_point_group_n < N,
                                         wei_zero_point_group_n % 16 == 0),
                    VERBOSE_UNSUPPORTED_ZP_CFG);

            if (utils::one_of(zp.get_data_type(DNNL_ARG_WEIGHTS), data_type::s4,
                        data_type::u4)) {
                dim_t k = desc.weights_desc.dims[ndims_wei - 2];
                dim_t n = desc.weights_desc.dims[ndims_wei - 1];
                VCHECK_MATMUL_UNIMPL(
                        IMPLICATION(mask_wei & wei_qmask_K, k % 2 == 0),
                        VERBOSE_UNSUPPORTED_ZP_CFG);
                VCHECK_MATMUL_UNIMPL(
                        IMPLICATION(mask_wei & wei_qmask_N, n % 2 == 0),
                        VERBOSE_UNSUPPORTED_ZP_CFG);
            }
        }

        if (!zp.has_default_values(DNNL_ARG_DST)) {
            const int mask_dst = zp.get_mask(DNNL_ARG_DST);

            VCHECK_MATMUL_UNIMPL(mask_dst == 0
                            || (desc.dst_desc.ndims == 2 && mask_dst == 1 << 1),
                    VERBOSE_UNSUPPORTED_ZP_CFG);
        }

        // Check dependency between zero_points.
        // Source zero_points groups are supported for int8 source and must
        // divide or be divided by weights groups when both are greater than 1.
        const bool groups_are_divisible = quant_groups_are_divisible(
                src_zero_point_group_k, wei_zero_point_group_k);
        VCHECK_MATMUL_UNIMPL(IMPLICATION(src_zero_point_group_k > 1,
                                     src_is_int8 && groups_are_divisible),
                VERBOSE_UNSUPPORTED_ZP_CFG);
    }

    // Check precomputed reductions
    if (!attr->precomputed_reductions_.has_default_values()) {
        const auto &pr = attr->precomputed_reductions_;

        // Only SRC argument is supported so far.
        std::vector<int> supported_args = {DNNL_ARG_SRC};
        VCHECK_MATMUL_UNIMPL(pr.has_default_values(supported_args),
                VERBOSE_UNSUPPORTED_PR_CFG);

        if (!pr.has_default_values(DNNL_ARG_SRC)) {
            const auto &zp = attr->zero_points_;
            // Weights zero points must be specified.
            VCHECK_MATMUL_UNIMPL(!zp.get(DNNL_ARG_WEIGHTS).has_default_values(),
                    VERBOSE_UNSUPPORTED_PR_CFG);

            // Mask must be defined for a full tensor, no broadcasts.
            const int pr_mask_src = pr.get_mask(DNNL_ARG_SRC);
            VCHECK_MATMUL_UNIMPL(pr_mask_src == full_tensor_mask,
                    VERBOSE_UNSUPPORTED_PR_CFG);

            // Data type must be s32 so far.
            const auto pr_dt = pr.get_data_type(DNNL_ARG_SRC);
            VCHECK_MATMUL_UNIMPL(
                    pr_dt == data_type::s32, VERBOSE_UNSUPPORTED_PR_CFG);

            if (!pr.get(DNNL_ARG_SRC).has_default_groups()) {
                const dim_t src_pr_group_k = pr.get_group(DNNL_ARG_SRC, 1);

                dim_t wei_zero_point_group_k = 1;
                if (!zp.get(DNNL_ARG_WEIGHTS).has_default_groups()) {
                    const int mask_wei = zp.get_mask(DNNL_ARG_WEIGHTS);
                    if (mask_wei & wei_qmask_K)
                        wei_zero_point_group_k
                                = zp.get_group(DNNL_ARG_WEIGHTS, 0);
                }

                const bool groups_are_divisible = quant_groups_are_divisible(
                        src_pr_group_k, wei_zero_point_group_k);
                VCHECK_MATMUL_UNIMPL(
                        IMPLICATION(src_pr_group_k > 1,
                                src_is_int8 && groups_are_divisible),
                        VERBOSE_UNSUPPORTED_PR_CFG);
            }
        }
    }

    // Check post-ops
    if (!attr->post_ops_.has_default_values()) {
        const auto &po = attr->post_ops_;
        using namespace primitive_kind;
        VCHECK_MATMUL_UNIMPL(
                po.has_default_values({binary, eltwise, prelu, sum}),
                VERBOSE_UNSUPPORTED_POSTOP);

        // Check sum
        VCHECK_MATMUL_UNIMPL(
                po.check_sum_consistency(dst_dt, src_is_int8, true),
                VERBOSE_UNSUPPORTED_POSTOP);

        // Note: verbose support is inside the call.
        CHECK(po.validate_binary(engine->kind(), &desc.dst_desc));
    }

    return status::success;
}

} // namespace

namespace dnnl {
namespace impl {
status_t matmul_desc_init(matmul_desc_t *matmul_desc,
        const memory_desc_t *src_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *bias_desc, const memory_desc_t *dst_desc,
        const memory_desc_t *reduce_desc, matmul_reduce_kind_t reduce_kind) {
    VCHECK_MATMUL(
            !any_null(src_desc, weights_desc, dst_desc), VERBOSE_NULL_ARG);

    // Note: This is an artificial limitation for the internal `reduce` feature
    // to limit the scope to what is actually used.
    VCHECK_MATMUL(
            IMPLICATION(bias_desc, !reduce_desc), VERBOSE_UNSUPPORTED_BIAS_CFG);

    VCHECK_MATMUL(!any_memory_desc_host_scalar(src_desc, weights_desc,
                          bias_desc, dst_desc, reduce_desc),
            VERBOSE_UNSUPPORTED_FORMAT_KIND);

    auto op_d = matmul_desc_t();
    op_d.primitive_kind = primitive_kind::matmul;

    op_d.src_desc = *src_desc;
    op_d.weights_desc = *weights_desc;
    if (bias_desc) op_d.bias_desc = *bias_desc;
    op_d.dst_desc = *dst_desc;
    if (reduce_desc) {
        VCHECK_MATMUL(reduce_desc->format_kind != format_kind::any,
                VERBOSE_UNSUPPORTED_FORMAT_KIND);
        op_d.reduce_desc = *reduce_desc;
        op_d.reduce_kind = reduce_kind;
        VCHECK_MATMUL(op_d.reduce_kind != matmul_reduce_kind::undef,
                VERBOSE_BAD_PARAM, "reduce_kind");
    }

    const bool with_bias = op_d.bias_desc.ndims != 0;
    const bool with_reduce = op_d.reduce_desc.ndims != 0;
    const int ndims = dst_desc->ndims;
    VCHECK_MATMUL(ndims >= 2 && ndims <= DNNL_MAX_NDIMS, VERBOSE_BAD_NDIMS,
            "dst", ndims);
    VCHECK_MATMUL(everyone_is(ndims, src_desc->ndims, weights_desc->ndims),
            VERBOSE_INCONSISTENT_NDIMS_WITH_VALS, "src", "weights",
            src_desc->ndims, weights_desc->ndims);
    VCHECK_MATMUL(IMPLICATION(with_bias, op_d.bias_desc.ndims == ndims),
            VERBOSE_BAD_NDIMS, "bias", op_d.bias_desc.ndims);
    VCHECK_MATMUL(IMPLICATION(with_reduce, op_d.reduce_desc.ndims == ndims),
            VERBOSE_BAD_NDIMS, "reduce", op_d.reduce_desc.ndims);

    // check: m, n, k
    const int m_idx = ndims - 2;
    const int k_idx_src = m_idx + 1;
    const int k_idx_wei = m_idx;
    const int n_idx = ndims - 1;
    VCHECK_MATMUL(dst_desc->dims[m_idx] == src_desc->dims[m_idx],
            VERBOSE_INCONSISTENT_DIM, "dst", m_idx, "src", m_idx);
    VCHECK_MATMUL(dst_desc->dims[n_idx] == weights_desc->dims[n_idx],
            VERBOSE_INCONSISTENT_DIM, "dst", n_idx, "weights", n_idx);
    VCHECK_MATMUL(src_desc->dims[k_idx_src] == weights_desc->dims[k_idx_wei],
            VERBOSE_INCONSISTENT_DIM, "src", k_idx_src, "weights", k_idx_wei);
    VCHECK_MATMUL(IMPLICATION(with_bias,
                          one_of(op_d.bias_desc.dims[n_idx], 1,
                                  dst_desc->dims[n_idx])),
            VERBOSE_INCONSISTENT_DIM, "bias", n_idx, "dst", n_idx);
    VCHECK_MATMUL(IMPLICATION(with_bias,
                          one_of(op_d.bias_desc.dims[m_idx], 1,
                                  dst_desc->dims[m_idx])),
            VERBOSE_INCONSISTENT_DIM, "bias", m_idx, "dst", m_idx);

    VCHECK_MATMUL(IMPLICATION(with_reduce,
                          one_of(op_d.reduce_desc.dims[n_idx], 1,
                                  dst_desc->dims[n_idx])),
            VERBOSE_INCONSISTENT_DIM, "reduce", n_idx, "dst", n_idx);
    VCHECK_MATMUL(IMPLICATION(with_reduce,
                          one_of(op_d.reduce_desc.dims[m_idx], 1,
                                  dst_desc->dims[m_idx])),
            VERBOSE_INCONSISTENT_DIM, "reduce", m_idx, "dst", m_idx);

    const int bia_mask = with_bias
            ? utils::get_dims_mask(dst_desc->dims, op_d.bias_desc.dims, ndims)
            : 0;

    using namespace data_type;
    if (weights_desc->format_kind == format_kind::blocked
            && utils::one_of(
                    weights_desc->data_type, s4, u4, f4_e2m1, f4_e3m0)) {
        const auto &wei_strides = weights_desc->format_desc.blocking.strides;

        int n_unit_strides = 0;
        for (int d = 0; d < ndims; d++) {
            if (wei_strides[d] == 1) {
                n_unit_strides++;
                VCHECK_MATMUL(
                        n_unit_strides <= 1, VERBOSE_BAD_DIM, "weights", d);
            }
            VCHECK_MATMUL(
                    IMPLICATION(wei_strides[d] > 1, wei_strides[d] % 2 == 0),
                    VERBOSE_BAD_DIM, "weights", d);
        }
    }
    if (src_desc->format_kind == format_kind::blocked
            && utils::one_of(src_desc->data_type, s4, u4, f4_e2m1, f4_e3m0)) {
        const auto &src_strides = src_desc->format_desc.blocking.strides;

        int n_unit_strides = 0;
        for (int d = 0; d < ndims; d++) {
            if (src_strides[d] == 1) {
                n_unit_strides++;
                VCHECK_MATMUL(n_unit_strides <= 1, VERBOSE_BAD_DIM, "src", d);
            }
            VCHECK_MATMUL(
                    IMPLICATION(src_strides[d] > 1, src_strides[d] % 2 == 0),
                    VERBOSE_BAD_DIM, "src", d);
        }
    }

    // check if other dims match.
    for (int d = 0; d < ndims - 2; ++d) {
        const dim_t s_dim = src_desc->dims[d];
        const dim_t w_dim = weights_desc->dims[d];
        const dim_t d_dim = dst_desc->dims[d];
        const dim_t b_dim = with_bias ? op_d.bias_desc.dims[d] : 0;
        const dim_t r_dim = with_reduce ? op_d.reduce_desc.dims[d] : 0;

        if (one_of(DNNL_RUNTIME_DIM_VAL, s_dim, w_dim, d_dim, b_dim)) {

            VCHECK_MATMUL(everyone_is(DNNL_RUNTIME_DIM_VAL, s_dim, w_dim, d_dim)
                            && IMPLICATION((bia_mask & (1 << d)) && with_bias,
                                    b_dim == DNNL_RUNTIME_DIM_VAL),
                    VERBOSE_RUNTIMEDIM_INCONSISTENT, d);
        } else {
            // This follows numpy semantics of broadcasting when 0 is involved.
            VCHECK_MATMUL(IMPLICATION(!everyone_is(s_dim, w_dim, d_dim),
                                  one_of(1, s_dim, w_dim)),
                    VERBOSE_INVALID_BROADCAST, "dst", d);
            VCHECK_MATMUL(IMPLICATION(s_dim == 1, d_dim == w_dim),
                    VERBOSE_INVALID_BROADCAST, "weights", d);
            VCHECK_MATMUL(IMPLICATION(w_dim == 1, d_dim == s_dim),
                    VERBOSE_INVALID_BROADCAST, "src", d);
            VCHECK_MATMUL(IMPLICATION(with_bias, one_of(b_dim, 1, d_dim)),
                    VERBOSE_INCONSISTENT_DIM, "bias", d, "dst", d);
            VCHECK_MATMUL(IMPLICATION(with_reduce, one_of(r_dim, 1, d_dim)),
                    VERBOSE_INCONSISTENT_DIM, "reduce", d, "dst", d);
        }
    }

    op_d.accum_data_type = types::default_accum_data_type(src_desc->data_type,
            weights_desc->data_type, dst_desc->data_type, prop_kind::forward);
    VCHECK_MATMUL(op_d.accum_data_type != data_type::undef,
            VERBOSE_INVALID_DATATYPE, "accumulation");
    *matmul_desc = op_d;
    return status::success;
}

status_t matmul_desc_init(matmul_desc_t *matmul_desc,
        const memory_desc_t *src_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *bias_desc, const memory_desc_t *dst_desc) {
    return matmul_desc_init(matmul_desc, src_desc, weights_desc, bias_desc,
            dst_desc, nullptr, matmul_reduce_kind::undef);
}

} // namespace impl
} // namespace dnnl

status_t dnnl_matmul_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        const memory_desc_t *src_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *bias_desc, const memory_desc_t *dst_desc,
        const primitive_attr_t *attr) {
    auto matmul_desc = matmul_desc_t();
    CHECK(matmul_desc_init(
            &matmul_desc, src_desc, weights_desc, bias_desc, dst_desc));
    CHECK(matmul_attr_check(matmul_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&matmul_desc, nullptr, attr);
}
