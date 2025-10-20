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

#ifndef GPU_INTEL_INCLUDE_GENERIC_VECTOR_OPS_H
#define GPU_INTEL_INCLUDE_GENERIC_VECTOR_OPS_H

typedef half __attribute__((ext_vector_type(1))) half1;
typedef uint __attribute__((ext_vector_type(1))) uint1;
typedef float __attribute__((ext_vector_type(1))) float1;
typedef int __attribute__((ext_vector_type(1))) int1;

float1 __attribute__((overloadable)) vmad(float1 a, float1 b, float1 c) {
    c[0] = mad(a[0], b[0], c[0]);
    return c;
}
float2 __attribute__((overloadable)) vmad(float2 a, float2 b, float2 c) {
    return mad(a, b, c);
}
float4 __attribute__((overloadable)) vmad(float4 a, float4 b, float4 c) {
    return mad(a, b, c);
}
float8 __attribute__((overloadable)) vmad(float8 a, float8 b, float8 c) {
    return mad(a, b, c);
}
float16 __attribute__((overloadable)) vmad(float16 a, float16 b, float16 c) {
    return mad(a, b, c);
}

float1 __attribute__((overloadable)) native_vrecip(float1 x) {
    x[0] = native_recip(x[0]);
    return x;
}
float2 __attribute__((overloadable)) native_vrecip(float2 x) {
    return native_recip(x);
}
float4 __attribute__((overloadable)) native_vrecip(float4 x) {
    return native_recip(x);
}
float8 __attribute__((overloadable)) native_vrecip(float8 x) {
    return native_recip(x);
}
float16 __attribute__((overloadable)) native_vrecip(float16 x) {
    return native_recip(x);
}

float1 __attribute__((overloadable)) native_vexp2(float1 x) {
    x[0] = native_exp2(x[0]);
    return x;
}
float2 __attribute__((overloadable)) native_vexp2(float2 x) {
    return native_exp2(x);
}
float4 __attribute__((overloadable)) native_vexp2(float4 x) {
    return native_exp2(x);
}
float8 __attribute__((overloadable)) native_vexp2(float8 x) {
    return native_exp2(x);
}
float16 __attribute__((overloadable)) native_vexp2(float16 x) {
    return native_exp2(x);
}

float1 __attribute__((overloadable)) vselect(float1 x, float1 y, int1 c) {
    x[0] = select(x[0], y[0], c[0]);
    return x;
}
float2 __attribute__((overloadable)) vselect(float2 x, float2 y, int2 c) {
    return select(x, y, c);
}
float4 __attribute__((overloadable)) vselect(float4 x, float4 y, int4 c) {
    return select(x, y, c);
}
float8 __attribute__((overloadable)) vselect(float8 x, float8 y, int8 c) {
    return select(x, y, c);
}
float16 __attribute__((overloadable)) vselect(float16 x, float16 y, int16 c) {
    return select(x, y, c);
}

int1 __attribute__((overloadable)) visfinite(float1 x) {
    int1 out = isfinite(x[0]);
    return out;
}
int2 __attribute__((overloadable)) visfinite(float2 x) {
    return isfinite(x);
}
int4 __attribute__((overloadable)) visfinite(float4 x) {
    return isfinite(x);
}
int8 __attribute__((overloadable)) visfinite(float8 x) {
    return isfinite(x);
}
int16 __attribute__((overloadable)) visfinite(float16 x) {
    return isfinite(x);
}

#endif
