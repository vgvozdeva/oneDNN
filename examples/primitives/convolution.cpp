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

/// @example convolution.cpp
/// > Annotated version: @ref convolution_example_cpp

/// @page convolution_example_cpp_brief
/// @brief This C++ API example demonstrates how to create and execute a
/// [Convolution](@ref dev_guide_convolution) primitive in forward propagation
/// mode in two configurations - with and without groups.

/// @page convolution_example_cpp Convolution Primitive Example
/// \copybrief convolution_example_cpp_brief
///
/// Key optimizations included in this example:
/// - Creation of optimized memory format from the primitive descriptor;
/// - Primitive attributes with fused post-ops.
///
/// @include convolution.cpp

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

void convolution_example(dnnl::engine::kind engine_kind) {

        std::cout << "[conv] Start convolution_example" << std::endl;

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);
        std::cout << "[conv] Engine created" << std::endl;

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);
        std::cout << "[conv] Stream created" << std::endl;

    // Tensor dimensions.
    const memory::dim N = 3, // batch size
            IH = 13, // input height
            IW = 13, // input width
            IC = 32, // input channels
            KH = 3, // weights height
            KW = 3, // weights width
            OC = 64, // output channels
            PH_L = 1, // height padding: left
            PH_R = 1, // height padding: right
            PW_L = 1, // width padding: left
            PW_R = 1, // width padding: right
            SH = 4, // height-wise stride
            SW = 4, // width-wise stride
            OH = (IH - KH + PH_L + PH_R) / SH + 1, // output height
            OW = (IW - KW + PW_L + PW_R) / SW + 1; // output width

    // Source (src), weights, bias, and destination (dst) tensors
    // dimensions.
    memory::dims src_dims = {N, IH, IW, IC, };
    memory::dims weights_dims = {OC, KH, KW, IC};
    // To simulate an empty bias use an empty initializer `{}`.
    memory::dims bias_dims = {OC};
    memory::dims dst_dims = {N, OH, OW, OC};

    // Primitive logical dimensions use canonical convolution ordering.
    memory::dims conv_src_dims = {N, IC, IH, IW};
    memory::dims conv_weights_dims = {OC, IC, KH, KW};
    memory::dims conv_dst_dims = {N, OC, OH, OW};

    // Strides, padding dimensions.
    memory::dims strides_dims = {SH, SW};
    memory::dims padding_dims_l = {PH_L, PW_L};
    memory::dims padding_dims_r = {PH_R, PW_R};

    // Allocate buffers.
    std::vector<float> src_data(product(src_dims));
    std::vector<float> weights_data(product(weights_dims));
    std::vector<float> bias_data(product(bias_dims));
    std::vector<float> dst_data(product(dst_dims));
        std::cout << "[conv] Buffers allocated" << std::endl;

    // Initialize src, weights, and dst tensors.
    std::generate(src_data.begin(), src_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });
    std::generate(weights_data.begin(), weights_data.end(), []() {
        static int i = 0;
        return std::sin(i++ * 2.f);
    });
    std::generate(bias_data.begin(), bias_data.end(), []() {
        static int i = 0;
        return std::tanh(float(i++));
    });
        std::cout << "[conv] Input data initialized" << std::endl;

    // Create memory objects for tensor data (src, weights, dst). In this
    // example, NHWC layout is assumed for src and dst, and OHWI for weights.
    auto user_src_mem = memory(
            {src_dims, memory::data_type::f32, memory::format_tag::nhwc},
            engine);
    auto user_weights_mem = memory(
            {weights_dims, memory::data_type::f32, memory::format_tag::ohwi},
            engine);
    auto user_dst_mem = memory(
            {dst_dims, memory::data_type::f32, memory::format_tag::nhwc},
            engine);
        std::cout << "[conv] User memory objects created" << std::endl;

    // Create memory descriptors with format_tag::any for the primitive. This
    // enables the convolution primitive to choose memory layouts for an
    // optimized primitive implementation, and these layouts may differ from the
    // ones provided by the user.
    auto conv_src_md = memory::desc(
            conv_src_dims, memory::data_type::f32, memory::format_tag::any);
    auto conv_weights_md = memory::desc(
            conv_weights_dims, memory::data_type::f32, memory::format_tag::any);
    auto conv_dst_md = memory::desc(
            conv_dst_dims, memory::data_type::f32, memory::format_tag::any);

    // Create memory descriptor and memory object for input bias.
    auto user_bias_md = bias_dims.empty()
            ? memory::desc()
            : memory::desc(
                      bias_dims, memory::data_type::f32, memory::format_tag::a);
    auto user_bias_mem = memory(user_bias_md, engine);
        std::cout << "[conv] Bias memory prepared" << std::endl;

    // Write data to memory object's handle.
    write_to_dnnl_memory(src_data.data(), user_src_mem);
    write_to_dnnl_memory(weights_data.data(), user_weights_mem);
    if (!bias_dims.empty())
        write_to_dnnl_memory(bias_data.data(), user_bias_mem);
        std::cout << "[conv] Data written to dnnl memory" << std::endl;

    // Create primitive post-ops (ReLU).
    const float alpha = 0.f;
    const float beta = 0.f;
    post_ops conv_ops;
    conv_ops.append_eltwise(algorithm::eltwise_relu, alpha, beta);
    primitive_attr conv_attr;
    conv_attr.set_post_ops(conv_ops);
        std::cout << "[conv] Primitive attributes created" << std::endl;

    // Create primitive descriptor.
    auto conv_pd = convolution_forward::primitive_desc(engine,
            prop_kind::forward_training, algorithm::convolution_direct,
            conv_src_md, conv_weights_md, user_bias_md, conv_dst_md,
            strides_dims, padding_dims_l, padding_dims_r, conv_attr);
        std::cout << "[conv] Primitive descriptor created" << std::endl;

    // For now, assume that the src, weights, and dst memory layouts generated
    // by the primitive and the ones provided by the user are identical.
    auto conv_src_mem = user_src_mem;
    auto conv_weights_mem = user_weights_mem;
    auto conv_dst_mem = user_dst_mem;

    // Reorder the data in case the src and weights memory layouts generated by
    // the primitive and the ones provided by the user are different. In this
    // case, we create additional memory objects with internal buffers that will
    // contain the reordered data. The data in dst will be reordered after the
    // convolution computation has finalized.
    if (conv_pd.src_desc() != user_src_mem.get_desc()) {
        conv_src_mem = memory(conv_pd.src_desc(), engine);
        reorder(user_src_mem, conv_src_mem)
                .execute(engine_stream, user_src_mem, conv_src_mem);
                std::cout << "[conv] Source reordered" << std::endl;
    }

    if (conv_pd.weights_desc() != user_weights_mem.get_desc()) {
        conv_weights_mem = memory(conv_pd.weights_desc(), engine);
        reorder(user_weights_mem, conv_weights_mem)
                .execute(engine_stream, user_weights_mem, conv_weights_mem);
                std::cout << "[conv] Weights reordered" << std::endl;
    }

    if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
        conv_dst_mem = memory(conv_pd.dst_desc(), engine);
                std::cout << "[conv] Destination memory adjusted for primitive" << std::endl;
    }

    // Create the primitive.
    auto conv_prim = convolution_forward(conv_pd);
        std::cout << "[conv] Primitive created" << std::endl;

    // Primitive arguments.
    std::unordered_map<int, memory> conv_args;
    conv_args.insert({DNNL_ARG_SRC, conv_src_mem});
    conv_args.insert({DNNL_ARG_WEIGHTS, conv_weights_mem});
    conv_args.insert({DNNL_ARG_BIAS, user_bias_mem});
    conv_args.insert({DNNL_ARG_DST, conv_dst_mem});
        std::cout << "[conv] Primitive arguments prepared" << std::endl;

    // Primitive execution: convolution with ReLU.
    conv_prim.execute(engine_stream, conv_args);
        std::cout << "[conv] Primitive executed" << std::endl;

    // Reorder the data in case the dst memory descriptor generated by the
    // primitive and the one provided by the user are different.
    if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
        reorder(conv_dst_mem, user_dst_mem)
                .execute(engine_stream, conv_dst_mem, user_dst_mem);
                std::cout << "[conv] Destination reordered back to user layout" << std::endl;
    } else
        user_dst_mem = conv_dst_mem;

    // Wait for the computation to finalize.
    engine_stream.wait();
        std::cout << "[conv] Stream wait completed" << std::endl;

    // Read data from memory object's handle.
    read_from_dnnl_memory(dst_data.data(), user_dst_mem);
        std::cout << "[conv] Output read complete" << std::endl;
}

void depthwise_convolution_example(dnnl::engine::kind engine_kind) {

        std::cout << "[dwconv] Start depthwise_convolution_example" << std::endl;

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);
        std::cout << "[dwconv] Engine created" << std::endl;

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);
        std::cout << "[dwconv] Stream created" << std::endl;

    // Tensor dimensions.
    const memory::dim N = 3, // batch size
            G = 32, // channel groups
            IC = 32, // input channels
            IH = 13, // input height
            IW = 13, // input width
            OC = 32, // output channels
            KH = 3, // weights height
            KW = 3, // weights width
            PH_L = 1, // height padding: left
            PH_R = 1, // height padding: right
            PW_L = 1, // width padding: left
            PW_R = 1, // width padding: right
            SH = 4, // height-wise stride
            SW = 4, // width-wise stride
            OH = (IH - KH + PH_L + PH_R) / SH + 1, // output height
            OW = (IW - KW + PW_L + PW_R) / SW + 1; // output width

    // Source (src), weights, bias, and destination (dst) tensors dimensions.
    memory::dims src_dims = {N, IH, IW, IC};
    memory::dims weights_dims = {G, KH, KW, IC / G, OC / G};
    memory::dims bias_dims = {OC};
    memory::dims dst_dims = {N, OH, OW, OC};

    // Primitive logical dimensions use canonical grouped convolution ordering.
    memory::dims conv_src_dims = {N, IC, IH, IW};
    memory::dims conv_weights_dims = {G, OC / G, IC / G, KH, KW};
    memory::dims conv_dst_dims = {N, OC, OH, OW};

    // Strides, padding dimensions.
    memory::dims strides_dims = {SH, SW};
    memory::dims padding_dims_l = {PH_L, PW_L};
    memory::dims padding_dims_r = {PH_R, PW_R};

    // Allocate buffers.
    std::vector<float> src_data(product(src_dims));
    std::vector<float> weights_data(product(weights_dims));
    std::vector<float> bias_data(OC);
    std::vector<float> dst_data(product(dst_dims));
        std::cout << "[dwconv] Buffers allocated" << std::endl;

    // Initialize src, weights, and dst tensors.
    std::generate(src_data.begin(), src_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });
    std::generate(weights_data.begin(), weights_data.end(), []() {
        static int i = 0;
        return std::sin(i++ * 2.f);
    });
    std::generate(bias_data.begin(), bias_data.end(), []() {
        static int i = 0;
        return std::tanh(float(i++));
    });
        std::cout << "[dwconv] Input data initialized" << std::endl;

    // Create memory objects for tensor data (src, weights, dst). In this
    // example, NHWC layout is assumed for src and dst, and OHWI for weights.
    auto user_src_mem = memory(
            {src_dims, memory::data_type::f32, memory::format_tag::nhwc},
            engine);
    auto user_weights_mem = memory(
            {weights_dims, memory::data_type::f32, memory::format_tag::gohwi},
            engine);
    auto user_dst_mem = memory(
            {dst_dims, memory::data_type::f32, memory::format_tag::nhwc},
            engine);
        std::cout << "[dwconv] User memory objects created" << std::endl;

    // Create memory descriptors with format_tag::any for the primitive. This
    // enables the convolution primitive to choose memory layouts for an
    // optimized primitive implementation, and these layouts may differ from the
    // ones provided by the user.
    auto conv_src_md = memory::desc(
            conv_src_dims, memory::data_type::f32, memory::format_tag::any);
    auto conv_weights_md = memory::desc(
            conv_weights_dims, memory::data_type::f32, memory::format_tag::any);
    auto conv_dst_md = memory::desc(
            conv_dst_dims, memory::data_type::f32, memory::format_tag::any);

    // Create memory descriptor and memory object for input bias.
    auto user_bias_md = memory::desc(
            bias_dims, memory::data_type::f32, memory::format_tag::a);
    auto user_bias_mem = memory(user_bias_md, engine);
    std::cout << "[dwconv] Bias memory prepared" << std::endl;

    // Write data to memory object's handle.
    write_to_dnnl_memory(src_data.data(), user_src_mem);
    write_to_dnnl_memory(weights_data.data(), user_weights_mem);
    write_to_dnnl_memory(bias_data.data(), user_bias_mem);
        std::cout << "[dwconv] Data written to dnnl memory" << std::endl;

    // Create primitive post-ops (ReLU).
    const float alpha = 0.f;
    const float beta = 0.f;
    post_ops conv_ops;
    conv_ops.append_eltwise(algorithm::eltwise_relu, alpha, beta);
    primitive_attr conv_attr;
    conv_attr.set_post_ops(conv_ops);
        std::cout << "[dwconv] Primitive attributes created" << std::endl;

    // Create primitive descriptor.
    auto conv_pd = convolution_forward::primitive_desc(engine,
            prop_kind::forward_training, algorithm::convolution_direct,
            conv_src_md, conv_weights_md, user_bias_md, conv_dst_md,
            strides_dims, padding_dims_l, padding_dims_r, conv_attr);
        std::cout << "[dwconv] Primitive descriptor created" << std::endl;

    // For now, assume that the src, weights, and dst memory layouts generated
    // by the primitive and the ones provided by the user are identical.
    auto conv_src_mem = user_src_mem;
    auto conv_weights_mem = user_weights_mem;
    auto conv_dst_mem = user_dst_mem;

    // Reorder the data in case the src and weights memory layouts generated by
    // the primitive and the ones provided by the user are different. In this
    // case, we create additional memory objects with internal buffers that will
    // contain the reordered data. The data in dst will be reordered after the
    // convolution computation has finalized.
    if (conv_pd.src_desc() != user_src_mem.get_desc()) {
        conv_src_mem = memory(conv_pd.src_desc(), engine);
        reorder(user_src_mem, conv_src_mem)
                .execute(engine_stream, user_src_mem, conv_src_mem);
                std::cout << "[dwconv] Source reordered" << std::endl;
    }

    if (conv_pd.weights_desc() != user_weights_mem.get_desc()) {
        conv_weights_mem = memory(conv_pd.weights_desc(), engine);
        reorder(user_weights_mem, conv_weights_mem)
                .execute(engine_stream, user_weights_mem, conv_weights_mem);
                std::cout << "[dwconv] Weights reordered" << std::endl;
    }

    if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
        conv_dst_mem = memory(conv_pd.dst_desc(), engine);
                std::cout << "[dwconv] Destination memory adjusted for primitive" << std::endl;
    }

    // Create the primitive.
    auto conv_prim = convolution_forward(conv_pd);
        std::cout << "[dwconv] Primitive created" << std::endl;

    // Primitive arguments.
    std::unordered_map<int, memory> conv_args;
    conv_args.insert({DNNL_ARG_SRC, conv_src_mem});
    conv_args.insert({DNNL_ARG_WEIGHTS, conv_weights_mem});
    conv_args.insert({DNNL_ARG_BIAS, user_bias_mem});
    conv_args.insert({DNNL_ARG_DST, conv_dst_mem});
        std::cout << "[dwconv] Primitive arguments prepared" << std::endl;

    // Primitive execution: convolution with ReLU.
    conv_prim.execute(engine_stream, conv_args);
        std::cout << "[dwconv] Primitive executed" << std::endl;

    // Reorder the data in case the dst memory descriptor generated by the
    // primitive and the one provided by the user are different.
    if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
        reorder(conv_dst_mem, user_dst_mem)
                .execute(engine_stream, conv_dst_mem, user_dst_mem);
                std::cout << "[dwconv] Destination reordered back to user layout" << std::endl;
    } else
        user_dst_mem = conv_dst_mem;

    // Wait for the computation to finalize.
    engine_stream.wait();
        std::cout << "[dwconv] Stream wait completed" << std::endl;

    // Read data from memory object's handle.
    read_from_dnnl_memory(dst_data.data(), user_dst_mem);
        std::cout << "[dwconv] Output read complete" << std::endl;
}

int main(int argc, char **argv) {
        std::cout << "[main] Running convolution_example" << std::endl;
    auto exit_code = handle_example_errors(
            convolution_example, parse_engine_kind(argc, argv));
    if (exit_code != 0) return exit_code;

        std::cout << "[main] Running depthwise_convolution_example" << std::endl;
    return handle_example_errors(
            depthwise_convolution_example, parse_engine_kind(argc, argv));
}
