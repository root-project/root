/*  This file is part of the Vc library.

    Copyright (C) 2009-2012 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_GLOBAL_H
#define VC_GLOBAL_H

#ifndef DOXYGEN

// Compiler defines
#ifdef __INTEL_COMPILER
#define VC_ICC __INTEL_COMPILER_BUILD_DATE
#elif defined(__OPENCC__)
#define VC_OPEN64 1
#elif defined(__clang__)
#define VC_CLANG (__clang_major__ * 0x10000 + __clang_minor__ * 0x100 + __clang_patchlevel__)
#elif defined(__GNUC__)
#define VC_GCC (__GNUC__ * 0x10000 + __GNUC_MINOR__ * 0x100 + __GNUC_PATCHLEVEL__)
#elif defined(_MSC_VER)
#define VC_MSVC _MSC_FULL_VER
#else
#define VC_UNSUPPORTED_COMPILER 1
#endif

// Features/Quirks defines
#if defined VC_MSVC && defined _WIN32
// the Win32 ABI can't handle function parameters with alignment >= 16
#define VC_PASSING_VECTOR_BY_VALUE_IS_BROKEN 1
#endif
#if defined(__GNUC__) && !defined(VC_NO_INLINE_ASM)
#define VC_GNU_ASM 1
#endif
#if defined(VC_GCC) && (VC_GCC <= 0x40405 || (VC_GCC >= 0x40500 && VC_GCC <= 0x40502)) && !(VC_GCC == 0x40502 && defined(__GNUC_UBUNTU_VERSION__) && __GNUC_UBUNTU_VERSION__ == 0xb0408)
// GCC 4.6.0 / 4.5.3 / 4.4.6 switched to the interface as defined by ICC
// (Ubuntu 11.04 ships a GCC 4.5.2 with the new interface)
#define VC_MM256_MASKSTORE_WRONG_MASK_TYPE 1
#endif
#if defined(VC_GCC) && VC_GCC >= 0x40300
#define VC_HAVE_ATTRIBUTE_ERROR 1
#define VC_HAVE_ATTRIBUTE_WARNING 1
#endif

#if (defined(__GXX_EXPERIMENTAL_CXX0X__) && VC_GCC >= 0x40600) || __cplusplus >= 201103
#  define VC_CXX11 1
#  ifdef VC_GCC
#    if VC_GCC >= 0x40700 // && VC_GCC < 0x408000)
//     ::max_align_t was introduced with GCC 4.7. std::max_align_t took a bit longer.
#      define VC_HAVE_MAX_ALIGN_T 1
#    endif
#  elif defined(VC_ICC)
#      define VC_HAVE_MAX_ALIGN_T 1
#  elif !defined(VC_CLANG)
//   Clang doesn't provide max_align_t at all
#    define VC_HAVE_STD_MAX_ALIGN_T 1
#  endif
#endif

// ICC ships the AVX2 intrinsics inside the AVX1 header.
// FIXME: the number 20120731 is too large, but I don't know which one is the right one
#if (defined(VC_ICC) && VC_ICC >= 20120731) || (defined(VC_MSVC) && VC_MSVC >= 170000000)
#define VC_UNCONDITIONAL_AVX2_INTRINSICS 1
#endif

/* Define the following strings to a unique integer, which is the only type the preprocessor can
 * compare. This allows to use -DVC_IMPL=SSE3. The preprocessor will then consider VC_IMPL and SSE3
 * to be equal. Of course, it is important to undefine the strings later on!
 */
#define Scalar 0x00100000
#define SSE    0x00200000
#define SSE2   0x00300000
#define SSE3   0x00400000
#define SSSE3  0x00500000
#define SSE4_1 0x00600000
#define SSE4_2 0x00700000
#define AVX    0x00800000

#define XOP    0x00000001
#define FMA4   0x00000002
#define F16C   0x00000004
#define POPCNT 0x00000008
#define SSE4a  0x00000010
#define FMA    0x00000020

#define IMPL_MASK 0xFFF00000
#define EXT_MASK  0x000FFFFF

#ifdef VC_MSVC
# ifdef _M_IX86_FP
#  if _M_IX86_FP >= 1
#   ifndef __SSE__
#    define __SSE__ 1
#   endif
#  endif
#  if _M_IX86_FP >= 2
#   ifndef __SSE2__
#    define __SSE2__ 1
#   endif
#  endif
# elif defined(_M_AMD64)
// If the target is x86_64 then SSE2 is guaranteed
#  ifndef __SSE__
#   define __SSE__ 1
#  endif
#  ifndef __SSE2__
#   define __SSE2__ 1
#  endif
# endif
#endif

#ifndef VC_IMPL

#  if defined(__AVX__)
#    define VC_IMPL_AVX 1
#  else
#    if defined(__SSE4_2__)
#      define VC_IMPL_SSE 1
#      define VC_IMPL_SSE4_2 1
#    endif
#    if defined(__SSE4_1__)
#      define VC_IMPL_SSE 1
#      define VC_IMPL_SSE4_1 1
#    endif
#    if defined(__SSE3__)
#      define VC_IMPL_SSE 1
#      define VC_IMPL_SSE3 1
#    endif
#    if defined(__SSSE3__)
#      define VC_IMPL_SSE 1
#      define VC_IMPL_SSSE3 1
#    endif
#    if defined(__SSE2__)
#      define VC_IMPL_SSE 1
#      define VC_IMPL_SSE2 1
#    endif

#    if defined(VC_IMPL_SSE)
       // nothing
#    else
#      define VC_IMPL_Scalar 1
#    endif
#  endif
#  if defined(VC_IMPL_AVX) || defined(VC_IMPL_SSE)
#    ifdef __FMA4__
#      define VC_IMPL_FMA4 1
#    endif
#    ifdef __XOP__
#      define VC_IMPL_XOP 1
#    endif
#    ifdef __F16C__
#      define VC_IMPL_F16C 1
#    endif
#    ifdef __POPCNT__
#      define VC_IMPL_POPCNT 1
#    endif
#    ifdef __SSE4A__
#      define VC_IMPL_SSE4a 1
#    endif
#    ifdef __FMA__
#      define VC_IMPL_FMA 1
#    endif
#  endif

#else // VC_IMPL

#  if (VC_IMPL & IMPL_MASK) == AVX // AVX supersedes SSE
#    define VC_IMPL_AVX 1
#  elif (VC_IMPL & IMPL_MASK) == Scalar
#    define VC_IMPL_Scalar 1
#  elif (VC_IMPL & IMPL_MASK) == SSE4_2
#    define VC_IMPL_SSE4_2 1
#    define VC_IMPL_SSE4_1 1
#    define VC_IMPL_SSSE3 1
#    define VC_IMPL_SSE3 1
#    define VC_IMPL_SSE2 1
#    define VC_IMPL_SSE 1
#  elif (VC_IMPL & IMPL_MASK) == SSE4_1
#    define VC_IMPL_SSE4_1 1
#    define VC_IMPL_SSSE3 1
#    define VC_IMPL_SSE3 1
#    define VC_IMPL_SSE2 1
#    define VC_IMPL_SSE 1
#  elif (VC_IMPL & IMPL_MASK) == SSSE3
#    define VC_IMPL_SSSE3 1
#    define VC_IMPL_SSE3 1
#    define VC_IMPL_SSE2 1
#    define VC_IMPL_SSE 1
#  elif (VC_IMPL & IMPL_MASK) == SSE3
#    define VC_IMPL_SSE3 1
#    define VC_IMPL_SSE2 1
#    define VC_IMPL_SSE 1
#  elif (VC_IMPL & IMPL_MASK) == SSE2
#    define VC_IMPL_SSE2 1
#    define VC_IMPL_SSE 1
#  elif (VC_IMPL & IMPL_MASK) == SSE
#    define VC_IMPL_SSE 1
#    if defined(__SSE4_2__)
#      define VC_IMPL_SSE4_2 1
#    endif
#    if defined(__SSE4_1__)
#      define VC_IMPL_SSE4_1 1
#    endif
#    if defined(__SSE3__)
#      define VC_IMPL_SSE3 1
#    endif
#    if defined(__SSSE3__)
#      define VC_IMPL_SSSE3 1
#    endif
#    if defined(__SSE2__)
#      define VC_IMPL_SSE2 1
#    endif
#  elif (VC_IMPL & IMPL_MASK) == 0 && (VC_IMPL & SSE4a)
     // this is for backward compatibility only where SSE4a was included in the main
     // line of available SIMD instruction sets
#    define VC_IMPL_SSE3 1
#    define VC_IMPL_SSE2 1
#    define VC_IMPL_SSE 1
#  endif
#  if (VC_IMPL & XOP)
#    define VC_IMPL_XOP 1
#  endif
#  if (VC_IMPL & FMA4)
#    define VC_IMPL_FMA4 1
#  endif
#  if (VC_IMPL & F16C)
#    define VC_IMPL_F16C 1
#  endif
#  if (VC_IMPL & POPCNT)
#    define VC_IMPL_POPCNT 1
#  endif
#  if (VC_IMPL & SSE4a)
#    define VC_IMPL_SSE4a 1
#  endif
#  if (VC_IMPL & FMA)
#    define VC_IMPL_FMA 1
#  endif
#  undef VC_IMPL

#endif // VC_IMPL

// If AVX is enabled in the compiler it will use VEX coding for the SIMD instructions.
#ifdef __AVX__
#  define VC_USE_VEX_CODING 1
#endif

#if defined(VC_GCC) && VC_GCC < 0x40300 && !defined(VC_IMPL_Scalar)
#    ifndef VC_DONT_WARN_OLD_GCC
#      warning "GCC < 4.3 does not have full support for SSE2 intrinsics. Using scalar types/operations only. Define VC_DONT_WARN_OLD_GCC to silence this warning."
#    endif
#    undef VC_IMPL_SSE
#    undef VC_IMPL_SSE2
#    undef VC_IMPL_SSE3
#    undef VC_IMPL_SSE4_1
#    undef VC_IMPL_SSE4_2
#    undef VC_IMPL_SSSE3
#    undef VC_IMPL_AVX
#    undef VC_IMPL_FMA4
#    undef VC_IMPL_XOP
#    undef VC_IMPL_F16C
#    undef VC_IMPL_POPCNT
#    undef VC_IMPL_SSE4a
#    undef VC_IMPL_FMA
#    undef VC_USE_VEX_CODING
#    define VC_IMPL_Scalar 1
#endif

# if !defined(VC_IMPL_Scalar) && !defined(VC_IMPL_SSE) && !defined(VC_IMPL_AVX)
#  error "No suitable Vc implementation was selected! Probably VC_IMPL was set to an invalid value."
# elif defined(VC_IMPL_SSE) && !defined(VC_IMPL_SSE2)
#  error "SSE requested but no SSE2 support. Vc needs at least SSE2!"
# endif

#undef Scalar
#undef SSE
#undef SSE2
#undef SSE3
#undef SSSE3
#undef SSE4_1
#undef SSE4_2
#undef AVX

#undef XOP
#undef FMA4
#undef F16C
#undef POPCNT
#undef SSE4a
#undef FMA

#undef IMPL_MASK
#undef EXT_MASK

namespace ROOT {
namespace Vc {
enum AlignedFlag {
    Aligned = 0
};
enum UnalignedFlag {
    Unaligned = 1
};
enum StreamingAndAlignedFlag { // implies Aligned
    Streaming = 2
};
enum StreamingAndUnalignedFlag {
    StreamingAndUnaligned = 3
};
#endif // DOXYGEN

/**
 * \ingroup Utilities
 *
 * Enum that specifies the alignment and padding restrictions to use for memory allocation with
 * Vc::malloc.
 */
enum MallocAlignment {
    /**
     * Align on boundary of vector sizes (e.g. 16 Bytes on SSE platforms) and pad to allow
     * vector access to the end. Thus the allocated memory contains a multiple of
     * VectorAlignment bytes.
     */
    AlignOnVector,
    /**
     * Align on boundary of cache line sizes (e.g. 64 Bytes on x86) and pad to allow
     * full cache line access to the end. Thus the allocated memory contains a multiple of
     * 64 bytes.
     */
    AlignOnCacheline,
    /**
     * Align on boundary of page sizes (e.g. 4096 Bytes on x86) and pad to allow
     * full page access to the end. Thus the allocated memory contains a multiple of
     * 4096 bytes.
     */
    AlignOnPage
};

#if __cplusplus >= 201103 /*C++11*/
#define Vc_CONSTEXPR constexpr
#elif defined(__GNUC__)
#define Vc_CONSTEXPR inline __attribute__((__always_inline__, __const__))
#elif defined(VC_MSVC)
#define Vc_CONSTEXPR inline __forceinline
#else
#define Vc_CONSTEXPR inline
#endif
Vc_CONSTEXPR StreamingAndUnalignedFlag operator|(UnalignedFlag, StreamingAndAlignedFlag) { return StreamingAndUnaligned; }
Vc_CONSTEXPR StreamingAndUnalignedFlag operator|(StreamingAndAlignedFlag, UnalignedFlag) { return StreamingAndUnaligned; }
Vc_CONSTEXPR StreamingAndUnalignedFlag operator&(UnalignedFlag, StreamingAndAlignedFlag) { return StreamingAndUnaligned; }
Vc_CONSTEXPR StreamingAndUnalignedFlag operator&(StreamingAndAlignedFlag, UnalignedFlag) { return StreamingAndUnaligned; }

Vc_CONSTEXPR StreamingAndAlignedFlag operator|(AlignedFlag, StreamingAndAlignedFlag) { return Streaming; }
Vc_CONSTEXPR StreamingAndAlignedFlag operator|(StreamingAndAlignedFlag, AlignedFlag) { return Streaming; }
Vc_CONSTEXPR StreamingAndAlignedFlag operator&(AlignedFlag, StreamingAndAlignedFlag) { return Streaming; }
Vc_CONSTEXPR StreamingAndAlignedFlag operator&(StreamingAndAlignedFlag, AlignedFlag) { return Streaming; }

/**
 * \ingroup Utilities
 *
 * Enum to identify a certain SIMD instruction set.
 *
 * You can use \ref VC_IMPL for the currently active implementation.
 *
 * \see ExtraInstructions
 */
enum Implementation {
    /// uses only fundamental types
    ScalarImpl,
    /// x86 SSE + SSE2
    SSE2Impl,
    /// x86 SSE + SSE2 + SSE3
    SSE3Impl,
    /// x86 SSE + SSE2 + SSE3 + SSSE3
    SSSE3Impl,
    /// x86 SSE + SSE2 + SSE3 + SSSE3 + SSE4.1
    SSE41Impl,
    /// x86 SSE + SSE2 + SSE3 + SSSE3 + SSE4.1 + SSE4.2
    SSE42Impl,
    /// x86 AVX
    AVXImpl,
    /// x86 AVX + AVX2
    AVX2Impl,
    ImplementationMask = 0xfff
};

/**
 * \ingroup Utilities
 *
 * The list of available instructions is not easily described by a linear list of instruction sets.
 * On x86 the following instruction sets always include their predecessors:
 * SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2
 *
 * But there are additional instructions that are not necessarily required by this list. These are
 * covered in this enum.
 */
enum ExtraInstructions {
    //! Support for float16 conversions in hardware
    Float16cInstructions  = 0x01000,
    //! Support for FMA4 instructions
    Fma4Instructions      = 0x02000,
    //! Support for XOP instructions
    XopInstructions       = 0x04000,
    //! Support for the population count instruction
    PopcntInstructions    = 0x08000,
    //! Support for SSE4a instructions
    Sse4aInstructions     = 0x10000,
    //! Support for FMA instructions (3 operand variant)
    FmaInstructions       = 0x20000,
    // PclmulqdqInstructions,
    // AesInstructions,
    // RdrandInstructions
    ExtraInstructionsMask = 0xfffff000u
};

#ifndef DOXYGEN

#ifdef VC_IMPL_Scalar
#define VC_IMPL ::ROOT::Vc::ScalarImpl
#elif defined(VC_IMPL_AVX)
#define VC_IMPL ::ROOT::Vc::AVXImpl
#elif defined(VC_IMPL_SSE4_2)
#define VC_IMPL ::ROOT::Vc::SSE42Impl
#elif defined(VC_IMPL_SSE4_1)
#define VC_IMPL ::ROOT::Vc::SSE41Impl
#elif defined(VC_IMPL_SSSE3)
#define VC_IMPL ::ROOT::Vc::SSSE3Impl
#elif defined(VC_IMPL_SSE3)
#define VC_IMPL ::ROOT::Vc::SSE3Impl
#elif defined(VC_IMPL_SSE2)
#define VC_IMPL ::ROOT::Vc::SSE2Impl
#endif

template<unsigned int Features> struct ImplementationT { enum _Value {
    Value = Features,
    Implementation = Features & Vc::ImplementationMask,
    ExtraInstructions = Features & Vc::ExtraInstructionsMask
}; };

typedef ImplementationT<
#ifdef VC_USE_VEX_CODING
    // everything will use VEX coding, so the system has to support AVX even if VC_IMPL_AVX is not set
    // but AFAIU the OSXSAVE and xgetbv tests do not have to positive (unless, of course, the
    // compiler decides to insert an instruction that uses the full register size - so better be on
    // the safe side)
    AVXImpl
#else
    VC_IMPL
#endif
#ifdef VC_IMPL_SSE4a
    + Vc::Sse4aInstructions
#ifdef VC_IMPL_XOP
    + Vc::XopInstructions
#ifdef VC_IMPL_FMA4
    + Vc::Fma4Instructions
#endif
#endif
#endif
#ifdef VC_IMPL_POPCNT
    + Vc::PopcntInstructions
#endif
#ifdef VC_IMPL_FMA
    + Vc::FmaInstructions
#endif
    > CurrentImplementation;

namespace Internal {
    template<Implementation Impl> struct HelperImpl;
    typedef HelperImpl<VC_IMPL> Helper;

    template<typename A> struct FlagObject;
    template<> struct FlagObject<AlignedFlag> { static Vc_CONSTEXPR AlignedFlag the() { return Aligned; } };
    template<> struct FlagObject<UnalignedFlag> { static Vc_CONSTEXPR UnalignedFlag the() { return Unaligned; } };
    template<> struct FlagObject<StreamingAndAlignedFlag> { static Vc_CONSTEXPR StreamingAndAlignedFlag the() { return Streaming; } };
    template<> struct FlagObject<StreamingAndUnalignedFlag> { static Vc_CONSTEXPR StreamingAndUnalignedFlag the() { return StreamingAndUnaligned; } };
} // namespace Internal

namespace Warnings
{
    void _operator_bracket_warning()
#ifdef VC_HAVE_ATTRIBUTE_WARNING
        __attribute__((warning("\n\tUse of Vc::Vector::operator[] to modify scalar entries is known to miscompile with GCC 4.3.x.\n\tPlease upgrade to a more recent GCC or avoid operator[] altogether.\n\t(This warning adds an unnecessary function call to operator[] which should work around the problem at a little extra cost.)")))
#endif
        ;
} // namespace Warnings

namespace Error
{
    template<typename L, typename R> struct invalid_operands_of_types {};
} // namespace Error

#endif // DOXYGEN
} // namespace Vc
} // namespace ROOT

#undef Vc_CONSTEXPR
#include "version.h"

#endif // VC_GLOBAL_H
