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

#ifndef GEMMSTONE_GUARD_COPY_PLAN_HPP
#define GEMMSTONE_GUARD_COPY_PLAN_HPP

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

#include "internal/ngen_includes.hpp"
#include "internal/utils.hpp"

GEMMSTONE_NAMESPACE_START

#ifndef GEMMSTONE_ENABLE_COPY_PLAN_DUMP
#if   defined(DNNL_DEV_MODE)
#define GEMMSTONE_ENABLE_COPY_PLAN_DUMP 1
#else
#define GEMMSTONE_ENABLE_COPY_PLAN_DUMP 0
#endif
#endif

class CopyPlan;

struct CopyOperand
{
    int16_t grf = 0;
    uint8_t offset = 0;
    int8_t stride = 1;
    uint8_t vs = 0, width = 0;                          // 2D region parameters.
    ngen::DataType type = ngen::DataType::invalid;
    ngen::DataType range = ngen::DataType::invalid;
    enum : uint8_t {GRF, Immediate, Flag, Null} kind = Null;
    bool temp = false;                                  // Operand is a temporary?
    bool overwrite = false;                             // Operand can be trashed?
    bool overwriteStride = false;                       // Padding area between strides can be trashed?
    bool neg = false;
    bool abs = false;
    uint64_t value = 0;                                 // Immediate value, or temporary index

    bool isNull() const { return kind == Null; }
    operator bool() const { return !isNull(); }
    bool operator!() const { return isNull(); }
    bool operator==(const CopyOperand &op) const;
    bool operator!=(const CopyOperand &op) const { return !operator==(op); }

    int byteOffset() const { return offset * getBytes(type); }
    int absByteOffset(ngen::HW hw) const { return byteOffset() + ngen::GRF::bytes(hw) * grf; }

    int byteStride() const { return stride * getBytes(type); }

    ngen::RegData ngen() const;
    ngen::Immediate ngenImmediate() const;
    ngen::FlagRegister ngenFlag() const;

    CopyOperand() = default;
    CopyOperand(ngen::RegData rd);
    CopyOperand(ngen::Immediate imm) : type(imm.getType()), kind(Immediate), value(imm) {}
    CopyOperand(int imm) : CopyOperand(ngen::Immediate(imm)) {}

    CopyOperand operator-() const;

    friend CopyOperand abs(CopyOperand o) { o.abs = true; return o; }

#if GEMMSTONE_ENABLE_COPY_PLAN_DUMP
    void dump() const;
#endif
};

struct CopyInstruction
{
    ngen::Opcode op;
    uint8_t ctrl;
    int simd = 0;
    int16_t cnumMin, cnumMax;
    uint16_t phase = 0, spread = 0;
    CopyOperand dst, src0, src1, src2, flag;
    ngen::ConditionModifier cmod = ngen::ConditionModifier::none;
    bool atomic = false, sat = false;
    int16_t cnumSub = 0;

    void invalidate()       { simd = 0; }
    bool isInvalid()  const { return (simd == 0); }

    operator bool() const { return !isInvalid(); }
    bool operator!() const { return isInvalid(); }

    bool hasCMod()    const { return cmod != ngen::ConditionModifier::none; }

    void moveToIntegerPipe();

    ngen::InstructionModifier ngenModifiers() const;

    template <typename Generator>
    inline void execute(Generator &g);

#if GEMMSTONE_ENABLE_COPY_PLAN_DUMP
    void dump(const CopyPlan &plan) const;
#endif
};

struct CopyTemporary
{
    friend class CopyPlan;

    int bytes = 0, align = 0, offset = 0;
    bool flag = false;
    int16_t cnumMin = 0x7FFF;
    int16_t cnumMax = -1;
    uint16_t phaseMin = 0xFFFF;
    int assignment = -1;

    explicit CopyTemporary(int bytes_, int align_, int offset_ = 0)
            : bytes(bytes_), align(align_), offset(offset_) {}

    static CopyTemporary createFlag(int bits = 16);

protected:
    void usedBy(const CopyInstruction &i) {
        cnumMin = std::min(cnumMin, i.cnumMin);
        cnumMax = std::max(cnumMax, i.cnumMax);
        phaseMin = std::min(phaseMin, i.phase);
    }

private:
    CopyTemporary() = default;
};

struct CopyResource
{
    friend class CopyPlan;

    enum Kind : uint64_t {
        null = 0,
        constantBase = 0x100000000,
    } kind;
    CopyOperand src;
    bool preinitialized = true;

    CopyResource(Kind kind_) : kind(kind_) {}

    template <typename Generator>
    inline void initialize(Generator &g);

    static Kind makeConstant32(uint32_t c);

protected:
    int getData(std::array<uint8_t, 64> &data) const;
};

class CopyPlan
{
public:
    using GRFAllocator = std::function<void(int count, ngen::GRFRange &range)>;
    using FlagAllocator = std::function<void(int bytes, ngen::FlagRegister &flag)>;

    CopyPlan(ngen::HW hw_, bool systolicAvailable_) : hw(hw_), systolicAvailable(systolicAvailable_) {}

    CopyInstruction &append(CopyInstruction &&i);
    CopyInstruction &append(ngen::Opcode op, int simd, const CopyOperand &dst, const CopyOperand &src0, const CopyOperand &src1 = CopyOperand(), const CopyOperand &src2 = CopyOperand());
    CopyInstruction &append(ngen::Opcode op, int simd, ngen::InstructionModifier mod, const CopyOperand &dst, const CopyOperand &src0, const CopyOperand &src1 = CopyOperand(), const CopyOperand &src2 = CopyOperand());
    CopyInstruction &append(int phase, ngen::Opcode op, int simd, const CopyOperand &dst, const CopyOperand &src0, const CopyOperand &src1 = CopyOperand(), const CopyOperand &src2 = CopyOperand());
    CopyInstruction &append(int phase, ngen::Opcode op, int simd, ngen::InstructionModifier mod, const CopyOperand &dst, const CopyOperand &src0, const CopyOperand &src1 = CopyOperand(), const CopyOperand &src2 = CopyOperand());
    CopyInstruction &appendDestructiveMov(int simd, const CopyOperand &dst, const CopyOperand &src0, bool overwriteStride = false);
    CopyInstruction &appendDestructiveMov(int simd, ngen::InstructionModifier mod, const CopyOperand &dst, const CopyOperand &src0, bool overwriteStride = false);

    void transform();
    void materializeTemps(const GRFAllocator &grfAllocator, const FlagAllocator &flagAllocator);

    template <typename Generator>
    inline void execute(Generator &g);

    int tempFlagBytes() const;

#if GEMMSTONE_ENABLE_COPY_PLAN_DUMP
    void dump() const;
    int cycleCount() const;
#endif

protected:
    ngen::HW hw;
    bool systolicAvailable;
    bool freezeCNums = false;
    std::vector<CopyInstruction> insns, newInsns;
    std::vector<CopyTemporary> temps;
    CopyInstruction invalidInsn;
    std::vector<CopyResource> resources;

    enum class SortType {
        PhaseOnly, Register, SourceOrder
    };

    CopyOperand newTemp(ngen::DataType type, int elems, int stride, int align = 0, int offset = 0);
    CopyOperand newFlag(int bits = 16);

    CopyOperand getResource(CopyResource::Kind kind);

    CopyInstruction &split(CopyInstruction &i, bool sequenced = true);
    template <int n>
    std::array<CopyInstruction*, n> splitMultiple(CopyInstruction &i);
    CopyInstruction &join(CopyInstruction &i1, CopyInstruction &i2, int maxGap = 3);
    void mergeChanges();

    void copyThrough(CopyInstruction &i, ngen::DataType type, int stride = 0, bool strideOff0 = false, bool movAfter = false);
    void restrideSrc0(CopyInstruction &i, int stride, bool strideOff0 = false);
    void restrideDst(CopyInstruction &i, int stride, bool strideOff0 = false);

    void repositionSrc(CopyInstruction &i, int n, int stride, int offset);
    void repositionDst(CopyInstruction &i, int stride, int offset);

    void checkNoSubbytes();
    void collapseCNums();
    bool trySwapCNumRanges(int16_t min0, int16_t max0, int16_t min1);

    void distributePhases();
    void split2DRegions();
    void planTypeConversions();
    void planEarlyInt4Upconversions();
    void planEmulatedHalveFloat(CopyInstruction &i);
    void planInt8ToHF(CopyInstruction &i);
    void planInt8ToBF(CopyInstruction &i);
    void planInt4ToF16(CopyInstruction &i);
    void planUnpack4To16(CopyInstruction &i);
    void planUnpack8To16High(CopyInstruction &i);
    void planInt4Upconversion(CopyInstruction &i);
    void plan4BitShifts(CopyInstruction &i);
    void planInt4Downconversion(CopyInstruction &i);
    void planEmulatedSIMD1(CopyInstruction &i);
    void planEmulatedBF8ToBF(CopyInstruction &i);
    void planEmulatedHF8ToHF(CopyInstruction &i);
    void planEmulatedHF8ToBF(CopyInstruction &i);
    void planEmulatedHFToHF8(CopyInstruction &i);
    void planEmulatedF4ToHF(CopyInstruction &i);
    void planEmulatedF4ToBF(CopyInstruction &i);
    void planEmulatedNF4ToHF(CopyInstruction &i);
    void planEmulatedHFToF4(CopyInstruction &i);
    void planE8M0ToF(CopyInstruction &i);
    void planBFNEmulation();
    void legalizeSIMD(bool initial = false);
    void legalizeRegions();
    void legalizeNegation();
    void legalizeImmediateTypes();
    void sort(SortType type);
    void optimizeZip(bool zip2DSrc0 = false);
    void optimizeZipAdjacent();
    void optimizeWidenIntegers();
    void optimizeConcatenate(bool initial = false);
    void optimizeWriteCombine();
    void optimizeWriteSpread();
    void optimizeIntegerDownconvert();
    void optimizeSaturate();
    void optimizeMoveToIntPipe();

    CopyOperand bfImmediate(uint16_t bits, bool ternary);
    CopyOperand zipImmediates(const CopyOperand &o1, const CopyOperand &o2);

    bool bfArithmeticOK(const CopyInstruction &i) const;
};


template <typename Generator>
void CopyPlan::execute(Generator &g)
{
    for (auto &r: resources) if (!r.preinitialized) r.initialize(g);
    for (auto &i: insns) i.execute(g);
}

template <typename Generator>
void CopyInstruction::execute(Generator &g)
{
    using namespace ngen;

#define UNARY_OP_CASE(o)                                                            \
    case Opcode::o:                                                           \
        if (src0.kind == CopyOperand::Immediate)                                    \
            g.o(ngenModifiers(), dst.ngen(), src0.ngenImmediate());                 \
        else                                                                        \
            g.o(ngenModifiers(), dst.ngen(), src0.ngen());                          \
        break;
#define BINARY_OP_CASE(o)                                                           \
    case Opcode::o:                                                           \
        if (src1.kind == CopyOperand::Immediate)                                    \
            g.o(ngenModifiers(), dst.ngen(), src0.ngen(), src1.ngenImmediate());    \
        else                                                                        \
            g.o(ngenModifiers(), dst.ngen(), src0.ngen(), src1.ngen());             \
        break;
#define TERNARY_OP_CASE(o)                                                          \
    case Opcode::o:                                                           \
        if (src0.kind == CopyOperand::Immediate) {                                  \
            if (src2.kind == CopyOperand::Immediate)                                \
                g.o(ngenModifiers(), dst.ngen(), src0.ngenImmediate(), src1.ngen(), src2.ngenImmediate()); \
            else                                                                    \
                g.o(ngenModifiers(), dst.ngen(), src0.ngenImmediate(), src1.ngen(), src2.ngen()); \
        } else {                                                                    \
            if (src2.kind == CopyOperand::Immediate)                                \
                g.o(ngenModifiers(), dst.ngen(), src0.ngen(), src1.ngen(), src2.ngenImmediate()); \
            else                                                                    \
                g.o(ngenModifiers(), dst.ngen(), src0.ngen(), src1.ngen(), src2.ngen()); \
        }                                                                           \
        break;

    switch (op) {
        UNARY_OP_CASE(mov)
        BINARY_OP_CASE(add)
        BINARY_OP_CASE(mul)
        BINARY_OP_CASE(cmp)
        BINARY_OP_CASE(and_)
        BINARY_OP_CASE(or_)
        BINARY_OP_CASE(xor_)
        BINARY_OP_CASE(shl)
        BINARY_OP_CASE(shr)
        BINARY_OP_CASE(asr)
        BINARY_OP_CASE(sel)
        TERNARY_OP_CASE(mad)
        TERNARY_OP_CASE(csel)
        case Opcode::bfn:
            if (src0.kind == CopyOperand::Immediate) {
                if (src2.kind == CopyOperand::Immediate)
                    g.bfn(ngenModifiers(), ctrl, dst.ngen(), src0.ngenImmediate(), src1.ngen(), src2.ngenImmediate());
                else
                    g.bfn(ngenModifiers(), ctrl, dst.ngen(), src0.ngenImmediate(), src1.ngen(), src2.ngen());
            } else {
                if (src2.kind == CopyOperand::Immediate)
                    g.bfn(ngenModifiers(), ctrl, dst.ngen(), src0.ngen(), src1.ngen(), src2.ngenImmediate());
                else
                    g.bfn(ngenModifiers(), ctrl, dst.ngen(), src0.ngen(), src1.ngen(), src2.ngen());
            }
            break;
        case Opcode::math: {
            auto fc = static_cast<MathFunction>(ctrl);
            if (mathArgCount(g.getHardware(), fc) == 1)
                g.math(ngenModifiers(), fc, dst.ngen(), src0.ngen());
            else if (src1.kind == CopyOperand::Immediate)
                g.math(ngenModifiers(), fc, dst.ngen(), src0.ngen(), src1.ngenImmediate());
            else
                g.math(ngenModifiers(), fc, dst.ngen(), src0.ngen(), src1.ngen());
            break;
        }
        default: stub("Unsupported opcode");
    }

#undef UNARY_OP_CASE
#undef BINARY_OP_CASE
#undef TERNARY_OP_CASE
}

template <typename Generator>
void CopyResource::initialize(Generator &g)
{
    using namespace ngen;

    std::array<uint8_t, 64> data;
    int n = getData(data);

    auto dataUQ = (const uint64_t *) &data[0];
    auto dataDF = (const double *)   &data[0];
    auto dataUD = (const uint32_t *) &data[0];
    auto dataF  = (const float *)    &data[0];
    int n32 = (n + 3) >> 2;

    bool do64 = (g.getHardware() >= HW::XeHPC);
    GRF r(src.ngen().getBase());
    for (int i = 0; 2*i+1 < n32; i++) {
        if (do64) {
            (i & 1) ? g.mov(1, r.uq(i), Immediate::uq(dataUQ[i]))
                    : g.mov(1, r.df(i), Immediate::df(dataDF[i]));
        } else {
            g.mov(1, r.ud(2*i),    Immediate::ud(dataUD[2*i]));
            g.mov(1, r.f(2*i + 1), Immediate::f(dataF[2*i + 1]));
        }
    }
    if (n32 & 1)
        g.mov(1, r.ud(n32 - 1), Immediate::ud(dataUD[n32 - 1]));
}

GEMMSTONE_NAMESPACE_END

#endif /* header guard */
