/*  This file is part of the Vc library.

    Copyright (C) 2010-2012 Matthias Kretz <kretz@kde.org>

    Vc is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    Vc is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with Vc.  If not, see <http://www.gnu.org/licenses/>.

*/

#include <Vc/global.h>
#include <Vc/cpuid.h>
#include <Vc/support.h>

#ifdef VC_MSVC
#include <intrin.h>
#endif

#if defined(VC_GCC) && VC_GCC >= 0x40400
#define VC_TARGET_NO_SIMD __attribute__((target("no-sse2,no-avx")))
#else
#define VC_TARGET_NO_SIMD
#endif

namespace ROOT {
namespace Vc
{

VC_TARGET_NO_SIMD
static inline bool xgetbvCheck(unsigned int bits)
{
#if defined(VC_MSVC) && VC_MSVC >= 160040219 // MSVC 2010 SP1 introduced _xgetbv
    unsigned long long xcrFeatureMask = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
    return (xcrFeatureMask & bits) == bits;
#elif defined(VC_GNU_ASM) && !defined(VC_NO_XGETBV)
    unsigned int eax;
    asm("xgetbv" : "=a"(eax) : "c"(0) : "edx");
    return (eax & bits) == bits;
#else
    // can't check, but if OSXSAVE is true let's assume it'll work
    return bits > 0; // ignore 'warning: unused parameter'
#endif
}

VC_TARGET_NO_SIMD
bool isImplementationSupported(Implementation impl)
{
    CpuId::init();

    switch (impl) {
    case ScalarImpl:
        return true;
    case SSE2Impl:
        return CpuId::hasSse2();
    case SSE3Impl:
        return CpuId::hasSse3();
    case SSSE3Impl:
        return CpuId::hasSsse3();
    case SSE41Impl:
        return CpuId::hasSse41();
    case SSE42Impl:
        return CpuId::hasSse42();
    case AVXImpl:
        return CpuId::hasOsxsave() && CpuId::hasAvx() && xgetbvCheck(0x6);
    case AVX2Impl:
        return false;
    case ImplementationMask:
        return false;
    }
    return false;
}

VC_TARGET_NO_SIMD
Vc::Implementation bestImplementationSupported()
{
    CpuId::init();

    if (!CpuId::hasSse2 ()) return Vc::ScalarImpl;
    if (!CpuId::hasSse3 ()) return Vc::SSE2Impl;
    if (!CpuId::hasSsse3()) return Vc::SSE3Impl;
    if (!CpuId::hasSse41()) return Vc::SSSE3Impl;
    if (!CpuId::hasSse42()) return Vc::SSE41Impl;
    if (CpuId::hasAvx() && CpuId::hasOsxsave() && xgetbvCheck(0x6)) {
        return Vc::AVXImpl;
    }
    return Vc::SSE42Impl;
}

VC_TARGET_NO_SIMD
unsigned int extraInstructionsSupported()
{
    unsigned int flags = 0;
    if (CpuId::hasF16c()) flags |= Vc::Float16cInstructions;
    if (CpuId::hasFma4()) flags |= Vc::Fma4Instructions;
    if (CpuId::hasXop ()) flags |= Vc::XopInstructions;
    if (CpuId::hasPopcnt()) flags |= Vc::PopcntInstructions;
    if (CpuId::hasSse4a()) flags |= Vc::Sse4aInstructions;
    if (CpuId::hasFma ()) flags |= Vc::FmaInstructions;
    //if (CpuId::hasPclmulqdq()) flags |= Vc::PclmulqdqInstructions;
    //if (CpuId::hasAes()) flags |= Vc::AesInstructions;
    //if (CpuId::hasRdrand()) flags |= Vc::RdrandInstructions;
    return flags;
}

} // namespace Vc
} // namespace ROOT

#undef VC_TARGET_NO_SIMD

// vim: sw=4 sts=4 et tw=100
