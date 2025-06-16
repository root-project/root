//===-- ElbrusTargetParser - Parser for Elbrus features ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a target parser to recognise Elbrus hardware features.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_ELBRUSTARGETPARSER_H
#define LLVM_SUPPORT_ELBRUSTARGETPARSER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#define XV_MASK( q) (1ULL << (q))

namespace llvm {
class StringRef;

namespace Elbrus {

/**
 * Список порядковых номером опций.
 */
enum XVecOptName {
    XV_MMX          = 0ULL,
    XV_POPCNT       = 1ULL,
    XV_SSE          = 2ULL,
    XV_SSE2         = 3ULL,
    XV_SSE3         = 4ULL,
    XV_SSSE3        = 5ULL,
    XV_SSE4_1       = 6ULL,
    XV_SSE4_2       = 7ULL,
    XV_AVX          = 8ULL,
    XV_AVX2         = 9ULL,
    XV_SSE4A        = 10ULL,
    XV_FMA4         = 11ULL,
    XV_XOP          = 12ULL,
    XV_FMA          = 13ULL,
    XV_BMI          = 14ULL,
    XV_BMI2         = 15ULL,
    XV_AES          = 16ULL,
    XV_PCLMUL       = 17ULL,
    XV_3DNOW        = 18ULL,
    XV_3DNOWA       = 19ULL,
    XV_CLFLUSHOPT   = 20ULL,
    XV_CLWB         = 21ULL,
    XV_CLZERO       = 22ULL,
    XV_F16C         = 23ULL,
    XV_LZCNT        = 24ULL,
    XV_MWAITX       = 25ULL,
    XV_RDRND        = 26ULL,
    XV_RDSEED       = 27ULL,
    XV_SHA          = 28ULL,
    XV_ABM          = 29ULL,
    XV_TBM          = 30ULL,
    XV_AVXVNNI      = 31ULL,
    XV_XVECEMUL     = 32ULL,
    XV_LAST         = 33ULL,
};

/**
 * Элементарные битовые маски для xvec-опций.
 */
enum XVecOptBit {
    XVB_MMX         = XV_MASK( XV_MMX),
    XVB_POPCNT      = XV_MASK( XV_POPCNT),
    XVB_SSE         = XV_MASK( XV_SSE),
    XVB_SSE2        = XV_MASK( XV_SSE2),
    XVB_SSE3        = XV_MASK( XV_SSE3),
    XVB_SSSE3       = XV_MASK( XV_SSSE3),
    XVB_SSE4_1      = XV_MASK( XV_SSE4_1),
    XVB_SSE4_2      = XV_MASK( XV_SSE4_2),
    XVB_AVX         = XV_MASK( XV_AVX),
    XVB_AVX2        = XV_MASK( XV_AVX2),
    XVB_SSE4A       = XV_MASK( XV_SSE4A),
    XVB_FMA4        = XV_MASK( XV_FMA4),
    XVB_XOP         = XV_MASK( XV_XOP),
    XVB_FMA         = XV_MASK( XV_FMA),
    XVB_BMI         = XV_MASK( XV_BMI),
    XVB_BMI2        = XV_MASK( XV_BMI2),
    XVB_AES         = XV_MASK( XV_AES),
    XVB_PCLMUL      = XV_MASK( XV_PCLMUL),
    XVB_3DNOW       = XV_MASK( XV_3DNOW),
    XVB_3DNOWA      = XV_MASK( XV_3DNOWA),
    XVB_CLFLUSHOPT  = XV_MASK( XV_CLFLUSHOPT),
    XVB_CLWB        = XV_MASK( XV_CLWB),
    XVB_CLZERO      = XV_MASK( XV_CLZERO),
    XVB_F16C        = XV_MASK( XV_F16C),
    XVB_LZCNT       = XV_MASK( XV_LZCNT),
    XVB_RDRND       = XV_MASK( XV_RDRND),
    XVB_MWAITX      = XV_MASK( XV_MWAITX),
    XVB_RDSEED      = XV_MASK( XV_RDSEED),
    XVB_SHA         = XV_MASK( XV_SHA),
    XVB_ABM         = XV_MASK( XV_ABM),
    XVB_TBM         = XV_MASK( XV_TBM),
    XVB_AVXVNNI     = XV_MASK( XV_AVXVNNI),
    XVB_XVECEMUL    = XV_MASK( XV_XVECEMUL),
};

/**
 * Информация по отдельной xvec-опции.
 */
struct XVecOptInfo {
    const char *name; // строковое название опции
    XVecOptName xopt; // порядковый номер опции
    uint64_t onopts; // маска дополнительно включаемых опций
    uint64_t offopts; // маска дополнительно выключаемых опций
};

/**
 * Список xvec-опций.
 */
extern XVecOptInfo XVecOpts[];

extern void updateXVecFeatures( uint64_t &fmask, XVecOptInfo *xvoi,
                                llvm::StringMap<bool> &Features, bool Enabled);

} // namespace Elbrus
} // namespace llvm

#endif
