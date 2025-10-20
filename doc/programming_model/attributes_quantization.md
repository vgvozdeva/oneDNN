Quantization {#dev_guide_attributes_quantization}
=================================================

@anchor dgaq_intro
## Introduction

Some primitives support input and output tensors with `int8` data types,
both signed and unsigned, enabling reduced-precision inference on
supported hardware.

Similarly, some primitives support
[Open Compute Project (OCP) 8-bit Floating Point (f8) data types][f8-spec]
designed to accelerate AI workloads, including training and inference
of large neural networks. Lowering precision to 8 bits with `f8` enables faster
computation and reduced memory usage.

See also:
- [Lower Numerical Precision Deep Learning Inference and Training](https://www.intel.com/content/dam/develop/external/us/en/documents/lower-numerical-precision-deep-learning-jan2018-754765.pdf)

## Quantization Model

oneDNN supports two main categories of quantization:
- Static Quantization (see @ref quantization_mode::dnnl_quantization_mode_static_sazp)
  with scales only (symmetric) or scales and zero-points (asymmetric),
  where scales are applied after zero-point.
- Dynamic Quantization (see @ref quantization_mode::dnnl_quantization_mode_dynamic_mx)
  compliant with the [OCP Microscaling (MX) Formats Specification][mx-spec].

To support quantization, primitives should be created and executed as
follows:
- During primitive descriptor creation source, weights or destination
  memory descriptors use low precision datatype (e.g., `s8` or
  `fp8_e4m3`).
- During primitive descriptor creation group size, data types, and
  broadcasting masks of the scaling factors and zero-point are
  provided using primitive attributes.
- During primitive execution the actual quantization parameters are
  provided as arguments to the execute function.

For performance reasons, each primitive implementation typically
supports only a subset of quantization parameter masks, group sizes
and data type combinations. Which combination is supported and
optimized is listed in each primitive documentation page.

This guide does not cover how the appropriate scaling factor can be found.
Refer to the materials in the [Introduction](@ref dgaq_intro).

### Static Quantization

The only formula for static quantization currently supported
by oneDNN is with scales applied after zero-point as follows:

\f[
x_{f32}[:] = scale_{x} \cdot (x_{quant}[:] - zp_{x})
\f]

where \f$x_{f32}\f$ and \f$x_{quant}\f$ are the non-quantized and
quantized representation of \f$x\f$ respectively, \f$scale_{x}\f$ is a
scaling factor in a floating-point format, \f$zp_{x}\f$ is a zero
point (typically in integral format), and \f$[:]\f$ is used to denote
element-wise application of the formula to the arrays.

In this model, oneDNN assumes that quantization parameters are inputs
provided by the user and the library does not compute those scaling
factors and zero-points as part of primitive computation.

These quantization parameters can either be computed ahead of time
using calibration tools or at runtime based on the actual minimum and
maximum values of a tensor. Either method can be used in conjunction
with oneDNN static quantization, as long as the quantization
parameters are passed as input to the oneDNN primitives at execution
time.

### Dynamic Quantization

The only formula for dynamic quantization currently supported by
oneDNN is with scales computed following the
[OCP MX Formats Specification][mx-spec], namely:

\f[
x_{f32}[:] = scale_{x} \cdot x_{quant}[:]
\f]

where \f$x_{f32}\f$ and \f$x_{quant}\f$ are the non-quantized and
quantized representation of \f$x\f$ respectively, and \f$scale_{x}\f$ is a
scaling factor:
- in `e8m0` format,
- computed for each group of size `32`,
- and computed as the largest power-of-two less than or equal to the
  maximum absolute value of the group divided by the largest
  power-of-two representable in the \f$x_{quant}\f$ data type,
  e.g., \f$E8M0(amax(x_{quant}[:])) / E8M0(MAX\_QUANT\_DT) \f$.

## General Numerical Behavior Notes

Primitive implementations are allowed to convert inputs to wider
data types (e.g., `int8` to `int16` or `int32`), when those conversions do not
impact accuracy.

During execution, primitives implementations avoid integer overflows
and maintain integer accuracy by using wider data types (e.g., `int32`)
for intermediate values and accumulators.

Results are then converted as
necessary before the result is written to the output memory objects.

The scales are applied in single precision floating point data type
(#dnnl::memory::data_type::f32) before downconversion to the
destination data type. When converting to integral data types,
implementations typically saturate, whereas for floating-point
data types, underflow/overflow can occur. To force saturation in
floating-point data types use @ref
dev_guide_attributes_post_ops_eltwise with clip algorithm. Rounding
happens according to [rounding mode attribute](@ref dev_guide_attributes_rounding_mode).

@warning
Depending on the architecture, the behavior of `int8` computations might slightly
vary. For more details, refer to @ref dev_guide_int8_computations.

When multiple operations are fused in a single primitive using the
[post ops attribute](@ref dev_guide_attributes_post_ops), those are assumed to be
computed in `f32` precision. As a result the destination quantization
parameters are applied after the post-ops as follows:

\f[
   \dst[:] = post\_ops(OP(src[:], weights[:], ...)) / scale_{\dst} + zp_{\dst}
\f]

Quantizing and dequantizing values between post-operations can be achieved
using one of [eltwise](@ref dev_guide_attributes_post_ops_eltwise),
[binary](@ref dev_guide_attributes_post_ops_binary), or the scale
parameter of the appropriate post-operation.

## Relevant APIs and Supported Granularity Levels

oneDNN provides APIs to set scales, zero-points, and precomputed reductions
for different quantization levels from global (per-tensor) to fine-grained block-wise.

### Argument Scaling

The library uses @ref dev_guide_attributes API for setting the scaling factors
for most of the primitives. The supporting attributes can be found in the
documentation for each primitive. The unsupported cases are handled according
to the
[attributes error handling section](@ref dev_guide_attributes_error_handling).

#### Available Scaling API Methods

oneDNN provides the following methods for setting scaling factors:

~~~cpp
// Legacy method with simple mask-based scaling
void dnnl::primitive_attr::set_scales_mask(int arg, int mask);

// Generic method with groups support
void dnnl::primitive_attr::set_scales(int arg, int mask,
                                      const dnnl::memory::dims &groups,
                                      dnnl::memory::data_type data_type = dnnl::memory::data_type::f32,
                                      bool is_on_host = false,
                                      quantization_mode qmode = quantization_mode::static_sazp);

// Convenience method for single host-side scalar
void dnnl::primitive_attr::set_host_scale(int arg,
                                          dnnl::memory::data_type data_type = dnnl::memory::data_type::f32);
~~~

Key parameters of the scaling API methods are summarized below:

| Parameter | Options* | Description |
|:----------|:--------|:------------|
| `arg` | `DNNL_ARG_SRC`, `DNNL_ARG_WEIGHTS`, `DNNL_ARG_DST`, `DNNL_ARG_BIAS` | Tensor to scale |
| `mask` | `0`, `1<<dim`, `(1<<d1)+(1<<d2)` | Scaling granularity: global, per-dimension, multi-dimensional |
| `groups` | `{}`, `{G}`, `{G1,G2,...}` | Block quantization: none, single-size, multi-dimensional blocks |
| `data_type` | `f32`, `bf16`, `f16`, `f8_e5m2`, `f8_e4m3`, `e8m0` | Scaling factor data type |
| `is_on_host` | `true`/`false` | Host vs device memory location of scaling factor |
| `qmode` | `static_sazp`, `dynamic_mx` | Quantization mode: static with scales and zero-points, dynamic (MXFP8 compatible) |

(*) Support for quantization options varies based on individual primitive and
target hardware. Refer to primitives documentation for the details.

#### Supported Scaling Granularity Levels

oneDNN supports the following scaling granularity levels to support different quantization
schemes:

- [Per-tensor scaling](#per-tensor-scaling) (`mask=0`) uses a single scaling factor for the entire
  tensor, making it the simplest approach.
- [Per-channel scaling](#per-channel-scaling) (`mask=1<<dim`) applies different scaling factors
  along a specific dimension, for instance commonly used for CNN weights.
- [Block scaling](#block-scaling) subdivides tensor dimensions into smaller
  blocks with individual scaling factors, important for large transformer
  models and advanced quantization techniques.
- [Multi-dimensional scaling](#multi-dimensional-scaling) (`mask=(1<<dim1)+(1<<dim2)`) provides
  independent scaling factors along multiple tensor dimensions, useful for complex
  activations where both batch and channel dimensions need separate scaling.

##### Per-tensor Scaling

In the simplest case, when there is only one common scaling factor the attribute changes
the op behavior from
\f[
    \dst[:] = Op(...)
\f]

to

\f[
    \dst[:] = scale \cdot Op(...).
\f]

~~~cpp
// Using full set_scales API (recommended)
attr.set_scales(DNNL_ARG_SRC, 0, {}, dnnl::memory::data_type::f32);

// Using convenience set_host_scale API for host-side scaling factor
attr.set_host_scale(DNNL_ARG_SRC, dnnl::memory::data_type::f32);

// Using legacy set_scales_mask API
attr.set_scales_mask(DNNL_ARG_SRC, 0);

// Scaling factors: 1 value
// Usage: All elements use same scaling factor
~~~

@note For more details on global scaling with a single scaling factor residing on
host, use @ref host-side-scalars-and-zero-points "host-side scalar scaling"
(`set_host_scale`) to avoid device memory transfer overhead.

See examples:
- [Convolution with Per-output-channel Quantization](#convolution-with-per-output-channel-quantization)

##### Per-Channel Scaling

Per-channel scaling applies different scaling factors along specific tensor
dimensions. For instance, it is commonly used for CNN weights where each
output channel has its own scaling factor.

~~~cpp
// Scaling factor per output channel (dimension 0 of weights)
attr.set_scales(DNNL_ARG_WEIGHTS, 1 << 0, {}, dnnl::memory::data_type::f32);

// Tensor: [OC, IC, H, W] = [64, 128, 3, 3]
// Scaling factors: 64 values (one per output channel)
// Usage: Each output channel gets its own scaling factor
~~~

See examples:
- [Weights Preparation with Per-output-channel Scaling](#weights-preparation-with-per-output-channel-scaling)
- [Convolution with Per-output-channel Quantization](#convolution-with-per-output-channel-quantization)
- @ref inference_int8_matmul_cpp

##### Block Scaling

Groups enable block-wise quantization by subdividing tensor dimensions into
smaller blocks, each with its own scaling factor. This might help balance accuracy
and efficiency by providing more granular quantization than per-tensor scaling.

~~~cpp
// Weight shape: [K, N] = [1024, 512] with groups [32, 1]
// Creates 32 groups along K dimension, each with its own scaling factor per N value
std::vector<dnnl::memory::dim_t> groups = {32, 1};
attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), groups,
                dnnl::memory::data_type::f32);

// Tensor: [K, N] = [1024, 512]
// Scaling factors: 32 × 512 = 16,384 values (one per group)
// Usage: Each (group_k, n) combination gets its own scaling factor
~~~

See examples:
- [Matmul with Advanced Quantization](#matmul-with-advanced-quantization)
- [Matmul with Precomputed Reductions and Advanced Quantization](#matmul-with-precomputed-reductions-and-advanced-quantization)
- @ref weights_decompression_matmul_cpp

###### Special Case: MX-compatible Block Scaling (or Dynamic Quantization)

MX-compatible block scaling uses `e8m0` data type for scaling factors
and `dynamic_mx` quantization mode to align with the [OCP MX Formats Specification][mx-spec].

~~~cpp
// Set MX-compatible block scaling for weights
attr.set_scales(DNNL_ARG_WEIGHTS, 1 << 0, {32}, dnnl::memory::data_type::e8m0,
                false /*on device*/, dnnl::quantization_mode::dynamic_mx);

// Tensor: [K, N] = [1024, 512]
// Scaling factors: 32 values (one per group of 32 in K dimension)
// Usage: Each group of 32 in K dimension gets its own scaling factor
~~~

##### Multi-Dimensional Scaling

Multi-dimensional scaling applies scaling factors across multiple tensor dimensions
simultaneously.

For scaling factors per dimensions \f$d_i\f$, set `mask = `\f$\sum_{d_i} 2^{d_i}\f$.

Resulting scaling factor count without groups: \f$\prod_{d_i} D_{d_i}\f$, with groups:
\f$\prod_{d_i} G_{d_i}\f$.

~~~cpp
// Scaling factors vary along batch and channel dimensions
attr.set_scales(DNNL_ARG_SRC, (1 << 0) + (1 << 1), {},
                dnnl::memory::data_type::f32, false);

// Tensor: [N, C, H, W] = [8, 64, 32, 32]
// Scaling factors needed: 8 * 64 = 512 values
// Usage: Each (batch, channel) combination gets its own scaling factor
~~~

See examples:
- [Matmul with Advanced Quantization](#matmul-with-advanced-quantization)
- [Matmul with Precomputed Reductions and Advanced Quantization](#matmul-with-precomputed-reductions-and-advanced-quantization)
- @ref weights_decompression_matmul_cpp

### Argument Zero-Points

Zero-points handle the quantization case where the quantized integer range
does not center around zero.

The library uses @ref dev_guide_attributes API for setting zero-points for
most primitives. The supporting attributes can be found in the documentation
for each primitive. The unsupported cases are handled according to the
[attributes error handling section](@ref dev_guide_attributes_error_handling).

#### Available Zero-Point API Methods

oneDNN provides the following methods for setting zero-points:

~~~cpp
// Legacy method with simple mask-based zero-points
void dnnl::primitive_attr::set_zero_points_mask(int arg, int mask);

// Generic method with groups support
void dnnl::primitive_attr::set_zero_points(int arg, int mask,
                                          const dnnl::memory::dims &groups,
                                          dnnl::memory::data_type data_type = dnnl::memory::data_type::s32,
                                          bool is_on_host = false);

// Convenience method for single host-side scalar
void dnnl::primitive_attr::set_host_zero_point(int arg,
                                              dnnl::memory::data_type data_type = dnnl::memory::data_type::s32);
~~~


Key parameters of the zero-point API methods are summarized below:

| Parameter | Options* | Description |
|:----------|:--------|:------------|
| `arg` | `DNNL_ARG_SRC`, `DNNL_ARG_WEIGHTS`, `DNNL_ARG_DST` | Tensor to apply zero-point |
| `mask` | `0`, `1<<dim`, `(1<<d1)+(1<<d2)` | Zero-point granularity: global, per-dimension, multi-dimensional |
| `groups` | `{}`, `{G}`, `{G1,G2,...}` | Block quantization: none, single-size, multi-dimensional blocks |
| `data_type` | `s32`, `s8`, `u8`, `s4`, `u4` | Zero-point data type |
| `is_on_host` | `true`/`false` | Host vs device memory location of zero-point |

(*) Support for quantization options varies based on individual primitive and
target hardware. Refer to primitives documentation for the details.

#### Supported Zero-Point Granularity Levels

Zero-point granularity mirrors the scaling factor granularity described above.
The same mask and groups concepts apply:

- **Per-tensor zero-point** (`mask=0`): Single zero-point for entire tensor
- **Per-channel zero-points** (`mask=1<<dim`): Different zero-points per
  channel
- **Block zero-points** (`mask` with `groups`): Block-wise zero-points
- **Multi-dimensional zero-points** (`mask=(1<<dim1)+(1<<dim2)`):
  Independent zero-points across multiple dimensions

~~~cpp
// Per-tensor zero-point
attr.set_zero_points(DNNL_ARG_SRC, 0, {}, dnnl::memory::data_type::s32);

// Per-channel zero-points
attr.set_zero_points(DNNL_ARG_WEIGHTS, 1 << 0, {}, dnnl::memory::data_type::s8);

// Block zero-points
std::vector<dnnl::memory::dim_t> groups = {64, 1};
attr.set_zero_points(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), groups,
                     dnnl::memory::data_type::s32);
~~~

See examples:
- [Convolution with Per-output-channel Quantization](#convolution-with-per-output-channel-quantization)
- [Matmul with Precomputed Reductions and Advanced Quantization](#matmul-with-precomputed-reductions-and-advanced-quantization)
- @ref inference_int8_matmul_cpp
- @ref weights_decompression_matmul_cpp

@anchor host-side-scalars-and-zero-points
### Special Case: Host-side Scalar Scaling Factor and Zero-point

When using the GPU engine and per-tensor quantization,
host-side scaling factor and zero-point are
supported to reduce copying of data from host to device.
A memory object for scaling factor or zero-point value should be created as a
host-side scalar (see @ref dev_guide_host_side_scalars for details) and passed
to the primitive execution function.

The host scaling factor or zero-point attributes could also
be set using the following convenience API:

~~~cpp
dnnl::primitive_attr attr;
attr.set_host_scale(DNNL_ARG_DST,
           dnnl::memory::data_type::f32);

attr.set_host_zero_point(DNNL_ARG_DST,
           dnnl::memory::data_type::s32);
~~~

See examples:
- @ref matmul_with_host_scalar_scale_cpp

### Precomputed Reductions

Precomputed reductions could help optimize performance for Large Language Models (LLM).

When using block-wise zero-points for quantized weights, the library must compute
reductions over the source tensor during matrix multiplication. This involves
summing source tensor values across groups along the reduction dimension:

\f[
\dst_{m,n}=\sum_{g=0}^{G-1}\sum_{k={K\over{G}}g}^{{K\over{G}}(g+1)-1}{\src_{m,k}(\weights_{k,n}-zp_{\weights}(g,n))}=\sum_{k=0}^{K-1}{\src_{m,k}\weights_{k,n}}-\sum_{g=0}^{G-1}zp_{\weights}(g,n)\underbrace{\sum_{k={K\over{G}}g}^{{K\over{G}}(g+1)-1}\src_{m,k}}_{R_{m,g}}
\f]

where `R` represents the precomputed reductions that can be calculated
externally when quantizing the source tensor,
therefore removing the need for the library to compute them at runtime.

The library uses @ref dev_guide_attributes API for setting precomputed reductions.
The supporting attributes can be found in the documentation for each primitive.
The unsupported cases are handled according to the
[attributes error handling section](@ref dev_guide_attributes_error_handling).

#### Available Precomputed Reductions API Method

oneDNN provides the following method for setting precomputed reductions:

~~~cpp
void dnnl::primitive_attr::set_precomputed_reductions(int arg, int mask,
        const dnnl::memory::dims &groups,
        dnnl::memory::data_type data_type = dnnl::memory::data_type::s32);
~~~

Key parameters of the precomputed reductions API method are summarized below:

| Parameter | Options* | Description |
|:----------|:--------|:------------|
| `arg` | `DNNL_ARG_SRC` | Tensor to apply precomputed reductions |
| `mask` | `0`, `1<<dim`, `(1<<d1)+(1<<d2)` | Reduction granularity: global, per-dimension, multi-dimensional |
| `groups` | `{}`, `{G}`, `{G1,G2,...}` | Block quantization: none, single-size, multi-dimensional blocks |
| `data_type` | `s32` | Reduction data type |

@note
The following limitations apply when using precomputed reductions:
- Requires weight zero-points: Cannot be used without weights zero-points specified.
- Full matrix mask required: Must have full A matrix mask, meaning broadcast is not supported.

(*) Support for quantization options varies based on individual primitive and
target hardware. Refer to primitives documentation for the details.

See examples:
- [Matmul with Precomputed Reductions and Advanced Quantization](#matrix-multiplication-with-precomputed-reductions-and-advanced-quantization)

## Quantization Workflows Examples

### Breakdown of Convolution with INT8 Quantization

Consider a convolution with bias. The tensors are represented as:

- \f$\src_{f32}[:] = scale_{\src} \cdot (\src_{int8}[:] - zp_{\src})\f$
- \f$\weights_{f32}[:] = scale_{\weights} \cdot \weights_{int8}[:]\f$
- \f$\dst_{f32}[:] = scale_{\dst} \cdot (\dst_{int8}[:] - zp_{\dst})\f$

Here the \f$\src_{f32}, \weights_{f32}, \dst_{f32}\f$ are not
computed at all, the whole work happens with int8 tensors. So the task
is to compute the \f$\dst_{int8}\f$ tensor, using the \f$\src_{int8}\f$,
\f$\weights_{int8}\f$ tensors passed at execution time, as well as the
corresponding quantization parameters \f$scale_{\src}\f$, \f$scale_{\weights}\f$,
\f$scale_{\dst}\f$, and \f$zp_{\src}\f$, \f$zp_{\dst}\f$.
Mathematically, the computations are:

\f[
   \dst_{int8}[:] =
      \operatorname{f32\_to\_int8}(
         (scale_{\src} \cdot scale_{\weights} \cdot
         \operatorname{s32\_to\_f32}(conv_{s32}(\src_{int8}, \weights_{int8})
	   - zp_{\src} \cdot comp_{s32}) + bias_{f32}) / scale_{\dst}
           + zp_{\dst} )
\f]

where

- \f$\operatorname{conv}_{s32}\f$ is just a regular convolution which takes source and
  weights with int8 data type and compute the result in int32 data type (int32
  is chosen to avoid overflows during the computations);

- \f$comp_{s32}\f$ is a compensation term to account for
  \f$\src\f$ non-zero zero-point. This term is computed by the oneDNN
  library and can typically be pre-computed ahead of time, for example
  during weights reorder.

- \f$\operatorname{f32\_to\_s8}()\f$ converts an `f32` value to `s8` with
  potential saturation if the values are out of the range of the int8 data
  type.

- \f$\operatorname{s32\_to\_f32}()\f$ converts an `int8` value to
  `f32` with potential rounding. This conversion is typically
  necessary to apply `f32` scaling factors.

#### Per-Channel Scaling Specifics

Some of the primitives have limited support of multiple scales for a quantized
tensor. The most popular use case is the @ref dev_guide_convolution primitive
that supports per-output-channel scaling factors for the weights, meaning that
the actual convolution computations would need to scale different output
channels differently. This is possible without significant performance loss
because the per-output-channel re-quantization is only required at the very end
of the computations. It seems impossible to implement the same trick for the
input channels, since that would require re-quantization for every input
data point.

- \f$\src_{f32}(n, ic, ih, iw) = scale_{\src} \cdot \src_{int8}(n, ic, ih, iw)\f$

- \f$\weights_{f32}(oc, ic, kh, kw) = scale_{\weights}(oc) \cdot \weights_{int8}(oc, ic, kh, kw)\f$

- \f$\dst_{f32}(n, oc, oh, ow) = scale_{\dst} \cdot \dst_{int8}(n, oc, oh, ow)\f$

Note that now the weights' scaling factor depends on \f$oc\f$.

To compute the \f$\dst_{int8}\f$ we need to perform the following:

\f[

    \dst_{int8}(n, oc, oh, ow) =
        \operatorname{f32\_to\_int8}(
            \frac{scale_{\src} \cdot scale_{\weights}(oc) \cdot
            conv_{s32}(\src_{int8}, \weights_{int8})|_{(n, oc, oh, ow)} + \bias_{f32}}{scale_{\dst}}
        ).
\f]

The user is responsible for preparing quantized weights accordingly. To do that,
oneDNN provides reorders that can perform per-channel scaling:

\f[

    \weights_{int8}(oc, ic, kh, kw) =
        \operatorname{f32\_to\_int8}(
            \weights_{f32}(oc, ic, kh, kw) / scale_{weights}(oc)
        ).
\f]

#### Weights Preparation with Per-output-channel Scaling

~~~cpp
   // weights dimensions
   const int OC, IC, KH, KW;

   // original f32 weights in plain format
   dnnl::memory::desc wei_plain_f32_md(
           {OC, IC, KH, KW},                 // dims
           dnnl::memory::data_type::f32,     // the data originally in f32
           dnnl::memory::format_tag::hwigo   // the plain memory format
           );

   // the scaling factors for quantized weights
   // An unique scale for each output-channel.
   std::vector<float> wei_scales(OC) = { /* values */ };
   dnnl::memory();

   // int8 convolution primitive descriptor
   dnnl::convolution_forward::primitive_desc conv_pd(/* see the convolution workflow section */);

   // query the convolution weights memory descriptor
   dnnl::memory::desc wei_conv_s8_md = conv_pd.weights_desc();

   // prepare the attributes for the reorder
   dnnl::primitive_attr attr;
   const int quantization_mask = 0
       | (1 << 0);  // scale per  OC dimension, which is the dim #0
   attr.set_scales_mask(DNNL_ARG_DST, quantization_mask);

   // create reorder that would perform:
   //   wei_s8(oc, ic, kh, kw) <- wei_f32(oc, ic, kh, kw) / scale(oc)
   // including the data format conversion.
   auto wei_reorder_pd = dnnl::reorder::primitive_desc(
           wei_plain_f32_md, engine, // source
           wei_conv_s8_md, engine, // destination,
           attr);
   auto wei_reorder = dnnl::reorder(wei_reorder_pd);

// ...
~~~

#### Convolution with Per-output-channel Quantization

Building upon the weights preparation shown above, this section shows
the complete workflow for an int8 convolution that combines per-output-channel
weight scaling with global source and destination scaling.

~~~cpp
   const float src_scale; // src_f32[:] = src_scale * src_s8[:]
   const float dst_scale; // dst_f32[:] = dst_scale * dst_s8[:]

   // the scaling factors for quantized weights (as declared above)
   // An unique scale for each output-channel.
   std::vector<float> wei_scales(OC) = {...};


   // Src, weights, and dst memory descriptors for convolution,
   // with memory format tag == any to allow a convolution implementation
   // to chose the appropriate memory format

   dnnl::memory::desc src_conv_s8_any_md(
           {BATCH, IC, IH, IW},          // dims
           dnnl::memory::data_type::s8,  // the data originally in s8
           dnnl::memory::format_tag::any // let convolution to choose
           );

   dnnl::memory::desc wei_conv_s8_any_md(
           {OC, IC, KH, KW},             // dims
           dnnl::memory::data_type::s8,  // the data originally in s8
           dnnl::memory::format_tag::any // let convolution to choose
           );

   dnnl::memory::desc dst_conv_s8_any_md(...);  // ditto

   // prepare the attributes for the convolution
   dnnl::primitive_attr attr;
   const int data_mask = 0; // scale and zero-point per tensor for source and destination
   const int wei_mask = 0
       | (1 << 0); // scale per OC dimension, which is the dim #0 on weights tensor:
                   // (   OC, IC, KH, KW)
                   //      0   1   2   3

   attr.set_scales_mask(DNNL_ARG_SRC, data_mask);
   attr.set_zero_points_mask(DNNL_ARG_SRC, data_mask);

   attr.set_scales_mask(DNNL_ARG_WEIGHTS, wei_mask);

   attr.set_scales_mask(DNNL_ARG_DST, data_mask);
   attr.set_zero_points_mask(DNNL_ARG_DST, data_mask);

   // create a convolution primitive descriptor
   auto conv_pd = dnnl::convolution_forward::primitive_desc(
           dnnl::prop_kind::forward_inference,
           dnnl::algorithm::convolution_direct,
           src_conv_s8_any_md,                     // what's important is that
           wei_conv_s8_any_md,                     // we specified that we want
           dst_conv_s8_any_md,                     // computations in s8
           strides, padding_l, padding_r,
           dnnl::padding_kind::zero
           attr);   // the attributes describe the quantization flow
// ...
~~~

### Matrix Multiplication with Weight-only Quantization (WoQ)

This example describes a process of weights decompression, or
weight-only quantization (WoQ), in matmul primitive which may be found when
running Large Language Models (LLM). The advanced quantization here implies
additional grouping introduced over reduction dimension besides traditional
per-N quantization.

~~~cpp
   // Src, weights, and dst memory descriptors for matmul.
   // Consider simple 2D matmul case.
   dnnl::memory::desc src_f16_any_md(...);
   dnnl::memory::desc wei_s8_any_md(
           {K (256), N (512)},           // dims
           dnnl::memory::data_type::s8,  // the data originally in s8
           dnnl::memory::format_tag::any // let matmul to choose
           );
   dnnl::memory::desc dst_f16_any_md(...);

   // prepare the attributes
   dnnl::primitive_attr attr;
   // scale per K and N dimensions:
   const int wei_mask = (1 << 0) | (1 << 1);
   // K dimension specifies the group size of `128`. It means that each 128
   // elements over K dimension will share a single value. For a given example,
   // there will be two groups, thus, two values referring to a single N value.
   std::vector<dim_t> wei_groups = {128, 1}

   // the scaling factors for quantized weights (as declared above)
   // A unique scale for each gK (256 / 128 = 2) times N, total 1024 elements.
   std::vector<half> wei_scales(gK, N) = {...};

   attr.set_scales(DNNL_ARG_WEIGHTS, wei_mask, wei_groups, dnnl::memory::data_type::f16);

   // Additionally, to instruct the library to perform weights decompression,
   // fpmath mode must be set with a flag set to `true`:
   attr.set_fpmath_mode(dnnl::fpmath_mode::f16, /* apply_to_int = */ true);

   // create a matmul primitive descriptor
   auto matmul_pd = dnnl::matmul::primitive_desc(
           engine,
           src_f16_any_md,
           wei_s8_any_md,
           dst_f16_any_md,
           attr);   // the attributes describe the quantization flow
// ...
~~~

### Matrix Multiplication with Precomputed Reductions and Advanced Quantization

This example extends the [Weight-only Quantization](#matrix-multiplication-with-weight-only-quantization-woq)
workflow by adding asymmetric weight quantization and external precomputed reductions.

This scenario occurs when quantizing the source tensor at runtime on the application-side,
while passing both quantized source and weights to the library.

Precomputed reductions are important when using `s8` zero-points for weights,
as applying them during computations would cause accuracy loss.

~~~cpp
   // Src, weights, and dst memory descriptors for matmul.
   // Consider simple 2D matmul case.
   dnnl::memory::desc src_u8_any_md(
           {M (64), K (256)},            // dims
           dnnl::memory::data_type::u8,  // the data originally in u8
           dnnl::memory::format_tag::any // let matmul to choose
           );
   dnnl::memory::desc wei_s8_any_md(
           {K (256), N (512)},           // dims
           dnnl::memory::data_type::s8,  // the data originally in s8
           dnnl::memory::format_tag::any // let matmul to choose
           );
   dnnl::memory::desc dst_f16_any_md(...);

   // prepare the attributes
   dnnl::primitive_attr attr;
   // scale per K and N dimensions:
   const int wei_mask = (1 << 0) | (1 << 1);
   // K dimension specifies the group size of `128`. It means that each 128
   // elements over K dimension will share a single value. For a given example,
   // there will be two groups, thus, two values referring to a single N value.
   std::vector<dim_t> wei_scales_groups = {128, 1}

   // The scaling factors for quantized weights (as declared above)
   // A unique scale for each scale_gK (256 / 128 = 2) times N, total 1024
   // elements.
   std::vector<half> wei_scales(scale_gK, N) = {...};

   attr.set_scales(DNNL_ARG_WEIGHTS, wei_mask, wei_scales_groups,
           dnnl::memory::data_type::f16);

   // Zero-points would have the same mask as grouping applies for them as well.
   // For example, let it use the different size of the group.
   std::vector<dim_t> wei_zp_groups = {64, 1};

   // The zero-point factors for quantized weights (as declared above)
   // A unique zero-point for each zp_gK (256 / 64 = 4) times N, total 2048
   // elements.
   std::vector<half> wei_zps(zp_gK, N) = {...};

   attr.set_zero_points(DNNL_ARG_WEIGHTS, wei_mask, wei_zp_groups,
           dnnl::memory::data_type::s8);

   // Now, specify the precomputed reductions.
   // Note that it's specified for source tensor.
   // It means it should have full-size source tensor mask (which in this
   // example coincides with `wei_mask`), and groups would be over another
   // dimension, same as zero-points group size.
   std::vector<dim_t> src_pr_groups = {1, 64};

   // The precomputed reduction factors for quantized sources.
   // A unique reduction for each M times pr_gK (256 / 64 = 4), total 256
   // elements.
   std::vector<half> src_prs(M, pr_gK) = {...};

   attr.set_precomputed_reductions(DNNL_ARG_SRC, src_tensor_mask,
           src_pr_groups);

   // fpmath mode is not required in case of dynamic quantization as it's
   // treated as classical quantization case.

   // create a matmul primitive descriptor
   auto matmul_pd = dnnl::matmul::primitive_desc(
           engine,
           src_s8_any_md,
           wei_s8_any_md,
           dst_f16_any_md,
           attr);   // the attributes describe the quantization flow
// ...
~~~

[f8-spec]: https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf

[mx-spec]: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf