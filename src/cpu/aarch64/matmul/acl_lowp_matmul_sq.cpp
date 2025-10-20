/*******************************************************************************
* Copyright 2025 Arm Ltd. and affiliates
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

#include "cpu/aarch64/matmul/acl_lowp_matmul_sq.hpp"

#include "arm_compute/core/utils/quantization/AsymmHelpers.h"

#include "cpu/aarch64/acl_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace matmul {

namespace {
// Keys are anonymous. So deduce the type automagically.
using lowp_matmul_key_t = decltype(memory_tracking::names::key_gemm_tmp_buffer);

const std::vector<lowp_matmul_key_t> lowp_matmul_keys = {
        memory_tracking::names::key_gemm_asm_tmp_buffer,
        memory_tracking::names::key_gemm_pretranspose_b,
        memory_tracking::names::key_gemm_pretranspose,
        memory_tracking::names::key_conv_gemm_col,
        memory_tracking::names::key_conv_gemm_row,
        memory_tracking::names::key_gemm_blocked_a,
        memory_tracking::names::key_gemm_blocked_b,
        memory_tracking::names::key_gemm_mm_result_s32,
        memory_tracking::names::key_gemm_mm_signed_a,
        memory_tracking::names::key_gemm_mm_signed_output,
};
} // namespace

status_t acl_lowp_matmul_sq_t::init(engine_t *engine) {

    auto almc = pd()->almc_;
    arm_compute::QuantizationInfo qi {1.0, 0, true};
    almc.src_tensor_info.set_quantization_info(qi);
    almc.wei_tensor_info.set_quantization_info(qi);
    almc.dst_tensor_info.set_quantization_info(qi);

    gemm_->configure(&almc.src_tensor_info, &almc.wei_tensor_info,
            almc.with_bias ? &almc.bia_tensor_info : nullptr,
            &almc.dst_tensor_info, almc.gemm_info);

    return status::success;
}

status_t acl_lowp_matmul_sq_t::pd_t::init(engine_t *engine) {

    VDISPATCH_MATMUL(set_default_formats(), "failed to set default formats");
    using smask_t = primitive_attr_t::skip_mask_t;
    VDISPATCH_MATMUL(attr()->has_default_values(smask_t::scales
                             | smask_t::zero_points | smask_t::post_ops),
            "only scale, zero point and post-ops attrs supported");
    VDISPATCH_MATMUL(is_dense_format_kind(), VERBOSE_UNSUPPORTED_SPARSE_CFG);

    static const std::vector<int> supported_args {
            DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST};
    for (int arg : supported_args) {
        if (attr()->scales_.has_default_values(arg)) continue;

        VDISPATCH_MATMUL(attr()->scales_.get_mask(arg) == 0,
                VERBOSE_UNSUPPORTED_SCALES_CFG);
    }

    for (int arg : supported_args) {
        if (attr()->zero_points_.has_default_values(arg)) continue;

        VDISPATCH_MATMUL(attr()->zero_points_.get_mask(arg) == 0,
                VERBOSE_UNSUPPORTED_SCALES_CFG);
    }

    VDISPATCH_MATMUL(
            !has_runtime_dims_or_strides(), VERBOSE_RUNTIMEDIM_UNSUPPORTED);
    const memory_desc_wrapper src_d(src_md_);
    const memory_desc_wrapper wei_d(weights_md_);
    const memory_desc_wrapper bia_d(bias_md_);
    const memory_desc_wrapper dst_d(dst_md_);

    cpu::matmul::matmul_helper_t helper(src_d, wei_d, dst_d);
    const dim_t M = helper.M();
    const dim_t N = helper.N();
    const dim_t K = helper.K();
    const dim_t dst_batch = helper.batch();
    const dim_t src_batch = helper.src_batch();
    const dim_t wei_batch = helper.wei_batch();

    using namespace data_type;
    VDISPATCH_MATMUL(utils::one_of(src_d.data_type(), s8, u8)
                    && wei_d.data_type() == s8
                    && (src_d.data_type() == s8 ? dst_d.data_type() == s8
                                                : dst_d.data_type() == u8),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_MATMUL(utils::one_of(bia_d.data_type(), f32, undef),
            VERBOSE_UNSUPPORTED_DT_CFG);

    // reject in case the op is running on a cpu that have i8mm instruction set.
    // this is a temporary fix until the issue is resolved.
    VDISPATCH_MATMUL(arm_compute::CPUInfo::get().has_i8mm(),
            "Op not supported on CPUs without i8mm instructions");

    // ACL batch dimension only support s32 for 3D and 4D
    VDISPATCH_MATMUL(
            wei_batch == 1, "Batch dimension must be 1 for the weights");

    using namespace format_tag;
    auto src_tag = memory_desc_matches_one_of_tag(src_md_, abcd, abc, ab);
    auto wei_tag = memory_desc_matches_one_of_tag(weights_md_, abcd, abc, ab);
    auto dst_tag = memory_desc_matches_one_of_tag(dst_md_, abcd, abc, ab);

    ACL_CHECK_SUPPORT(
            utils::one_of(format_tag::undef, src_tag, wei_tag, dst_tag),
            "Format tag is undefined");

    VDISPATCH_MATMUL_SC(memory_desc_init_by_tag(bias_md_, bias_md_.ndims,
                                bias_md_.dims, bias_md_.data_type, dst_tag),
            VERBOSE_UNSUPPORTED_BIAS_CFG);

    // We set the QuantizationInfo to be dynamic because it is re-set in run()
    almc_.src_tensor_info = arm_compute::TensorInfo(
            arm_compute::TensorShape(K, M, 1, src_batch), 1,
            acl_utils::get_acl_data_t(src_d.data_type(), true),
            arm_compute::QuantizationInfo(1.0, 0, true));
    almc_.src_tensor_info.set_are_values_constant(false);

    almc_.wei_tensor_info = arm_compute::TensorInfo(
            arm_compute::TensorShape(N, K, 1, wei_batch), 1,
            acl_utils::get_acl_data_t(wei_d.data_type(), true),
            arm_compute::QuantizationInfo(1.0, 0, true));
    almc_.wei_tensor_info.set_are_values_constant(false);

    almc_.dst_tensor_info = arm_compute::TensorInfo(
            arm_compute::TensorShape(N, M, 1, dst_batch), 1,
            acl_utils::get_acl_data_t(dst_d.data_type(), true),
            arm_compute::QuantizationInfo(1.0, 0, true));

    almc_.bia_tensor_info = arm_compute::TensorInfo(
            arm_compute::TensorShape(), 1, arm_compute::DataType::S32);
    almc_.with_bias = bia_d.format_kind() != format_kind::undef;

    if (almc_.with_bias) {
        switch (bia_d.ndims()) {
            case 2:
                VDISPATCH_MATMUL(bia_d.dims()[0] == 1 && bia_d.dims()[1] == N,
                        "Only 1xN bias is supported for 2D input");
                almc_.bia_tensor_info.set_tensor_shape(arm_compute::TensorShape(
                        bia_d.dims()[1], bia_d.dims()[0]));
                break;
            case 3:
                VDISPATCH_MATMUL(bia_d.dims()[0] == 1 && bia_d.dims()[1] == 1
                                && bia_d.dims()[2] == N,
                        "Only 1x1xN bias is supported for 3D input");
                almc_.bia_tensor_info.set_tensor_shape(
                        arm_compute::TensorShape(bia_d.dims()[2], 1, 1));
                break;
            case 4:
                VDISPATCH_MATMUL(bia_d.dims()[0] == 1 && bia_d.dims()[1] == 1
                                && bia_d.dims()[2] == 1 && bia_d.dims()[3] == N,
                        "Only 1x1x1xN bias is supported for 4D input");
                almc_.bia_tensor_info.set_tensor_shape(
                        arm_compute::TensorShape(bia_d.dims()[3], 1, 1, 1));
                break;
        }
    }

    arm_compute::GEMMLowpOutputStageInfo info;
    info.type = arm_compute::GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
    info.gemmlowp_multiplier = 1073741824;
    info.gemmlowp_shift = -1;
    info.gemmlowp_offset = 0;
    info.gemmlowp_min_bound = -128;
    info.gemmlowp_max_bound = 127;
    info.output_data_type = almc_.dst_tensor_info.data_type();
    almc_.gemm_info.set_gemmlowp_output_stage(info);

    arm_compute::experimental::op::CpuGEMMLowp gemm;
    gemm.configure(&almc_.src_tensor_info, &almc_.wei_tensor_info,
            almc_.with_bias ? &almc_.bia_tensor_info : nullptr,
            &almc_.dst_tensor_info, almc_.gemm_info);

    auto aux_mem_req = gemm.workspace();
    auto scratchpad = scratchpad_registry().registrar();
    const dnnl::impl::memory_desc_t dst_md_ {desc_.dst_desc};
    arm_compute::ActivationLayerInfo act_info;
    CHECK(init_scratchpad(engine, scratchpad, acl_post_ops, attr_.post_ops_,
            act_info, dst_md_, aux_mem_req));
    almc_.gemm_info.set_activation_info(act_info);

    ACL_CHECK_VALID(arm_compute::experimental::op::CpuGEMMLowp::validate(
            &almc_.src_tensor_info, &almc_.wei_tensor_info,
            almc_.with_bias ? &almc_.bia_tensor_info : nullptr,
            &almc_.dst_tensor_info, almc_.gemm_info));

    return status::success;
}

// Keys are anonymous with local linkage. So deduce the type automagically.
using matmul_key_t = decltype(memory_tracking::names::key_gemm_tmp_buffer);

status_t acl_lowp_matmul_sq_t::pd_t::init_scratchpad(engine_t *engine,
        memory_tracking::registrar_t &scratchpad, acl_post_ops_t &post_ops,
        dnnl::impl::post_ops_t &attr_post_ops,
        arm_compute::ActivationLayerInfo &act_info,
        const dnnl::impl::memory_desc_t &dst_md,
        const arm_compute::experimental::MemoryRequirements &aux_mem_req) {

    CHECK(post_ops.init(engine, attr_post_ops, dst_md, act_info));

    // Book temp mem.
    if (!aux_mem_req.empty()) {
        for (size_t id = 0; id < lowp_matmul_keys.size(); id++) {
            if (aux_mem_req[id].size > 0) {
                scratchpad.book(lowp_matmul_keys[id], aux_mem_req[id].size, 1,
                        aux_mem_req[id].alignment, aux_mem_req[id].alignment);
            }
        }
    }

    // ACL only accepts s32 bias for quantization and since
    // the current bias vector is f32 we need to convert.
    if (almc_.with_bias) {
        const memory_desc_wrapper bias_d(&bias_md_);
        scratchpad.book(memory_tracking::names::key_conv_bias_s32_convert,
                bias_d.nelems(), bias_d.data_type_size());
    }

    return status::success;
}

status_t acl_lowp_matmul_sq_t::execute(const exec_ctx_t &ctx) const {

    std::lock_guard<std::mutex> _lock {this->mtx_};
    const auto scratchpad = ctx.get_scratchpad_grantor();
    auto almc = pd()->almc_;
    bool with_bias = pd()->almc_.with_bias;

    auto src = CTX_IN_MEM(const int8_t *, DNNL_ARG_SRC);
    auto wei = CTX_IN_MEM(const int8_t *, DNNL_ARG_WEIGHTS);
    auto dst = CTX_OUT_MEM(const int8_t *, DNNL_ARG_DST);

    arm_compute::Tensor src_tensor, dst_tensor, wei_tensor, bia_tensor;
    src_tensor.allocator()->init(almc.src_tensor_info);
    wei_tensor.allocator()->init(almc.wei_tensor_info);
    dst_tensor.allocator()->init(almc.dst_tensor_info);
    src_tensor.allocator()->import_memory(const_cast<int8_t *>(src));
    wei_tensor.allocator()->import_memory(const_cast<int8_t *>(wei));
    dst_tensor.allocator()->import_memory(const_cast<int8_t *>(dst));

    DEFINE_ARG_SCALES_BUFFER(src_scale, DNNL_ARG_SRC);
    DEFINE_ARG_SCALES_BUFFER(wei_scale, DNNL_ARG_WEIGHTS);
    DEFINE_ARG_SCALES_BUFFER(dst_scale, DNNL_ARG_DST);

    const int32_t *src_zero_points = CTX_IN_MEM(
            const int32_t *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);
    const int32_t *wei_zero_points = CTX_IN_MEM(
            const int32_t *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS);
    const int32_t *dst_zero_points = CTX_IN_MEM(
            const int32_t *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST);

    const int32_t src_zero_point = src_zero_points ? src_zero_points[0] : 0;
    const int32_t wei_zero_point = wei_zero_points ? wei_zero_points[0] : 0;
    const int32_t dst_zero_point = dst_zero_points ? dst_zero_points[0] : 0;

    if (with_bias) {
        auto bia_s32_base = scratchpad.get<uint32_t>(
                memory_tracking::names::key_conv_bias_s32_convert);
        auto bia_f32_base = CTX_IN_MEM(const float32_t *, DNNL_ARG_BIAS);
        const float bias_scale = 1 / (*src_scale * (*wei_scale));
        const int num_elements
                = almc.bia_tensor_info.total_size() / sizeof(float32_t);
        parallel_nd(num_elements, [&](dim_t e) {
            const auto b = int32_t(std::round(bia_f32_base[e] * bias_scale));
            bia_s32_base[e] = b;
        });
        bia_tensor.allocator()->init(almc.bia_tensor_info);
        bia_tensor.allocator()->import_memory(bia_s32_base);
    }

    src_tensor.info()->set_quantization_info(
            arm_compute::QuantizationInfo(*src_scale, -src_zero_point, true));
    wei_tensor.info()->set_quantization_info(
            arm_compute::QuantizationInfo(*wei_scale, -wei_zero_point, true));
    // for efficiency reasons, oneDNN saves the inverse of the destination
    dst_tensor.info()->set_quantization_info(arm_compute::QuantizationInfo(
            1.0 / (*dst_scale), dst_zero_point, true));

    gemm_->update_quantization_parameters(
            src_tensor.info()->quantization_info(),
            wei_tensor.info()->quantization_info(),
            dst_tensor.info()->quantization_info(),
            dst_tensor.info()->data_type(), true, true);

    arm_compute::ITensorPack gemm_pack {
            {arm_compute::TensorType::ACL_SRC_0, &src_tensor},
            {arm_compute::TensorType::ACL_SRC_1, &wei_tensor},
            {arm_compute::TensorType::ACL_DST, &dst_tensor}};

    if (with_bias) {
        gemm_pack.add_tensor(arm_compute::TensorType::ACL_SRC_2, &bia_tensor);
    }

    // Hold onto temp tensors while we need run pack.
    auto aux_mem = gemm_->workspace();
    std::vector<arm_compute::Tensor> tmp_tensors(aux_mem.size());
    for (size_t id = 0; id < lowp_matmul_keys.size(); id++) {
        if (aux_mem[id].size > 0) {
            const auto info = arm_compute::TensorInfo(
                    arm_compute::TensorShape(aux_mem[id].size), 1,
                    arm_compute::DataType::U8);
            auto buffer = scratchpad.get<void>(lowp_matmul_keys[id]);
            tmp_tensors[id].allocator()->init(info, aux_mem[id].alignment);
            tmp_tensors[id].allocator()->import_memory(buffer);
            gemm_pack.add_tensor(aux_mem[id].slot, &tmp_tensors[id]);
        }
    }

    gemm_->run(gemm_pack);

    // free() here tells ACL it can no longer use it, it does not deallocate
    src_tensor.allocator()->free();
    wei_tensor.allocator()->free();
    if (with_bias) { bia_tensor.allocator()->free(); }
    dst_tensor.allocator()->free();

    return status::success;
};
} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
