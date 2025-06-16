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

#include "llvm/TargetParser/ElbrusTargetParser.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;
using namespace llvm::Elbrus;

#define XVB_XVECEMUL_ON ((~0ULL) >> (64 - XV_XVECEMUL))
#define XVB_XVECEMUL_OFF XVB_XVECEMUL_ON

/**
 * Список xvec-опций.
 */
llvm::Elbrus::XVecOptInfo llvm::Elbrus::XVecOpts[] = {
    {"mmx",        XV_MMX,        (0),                 (XVB_3DNOW)},
    {"popcnt",     XV_POPCNT,     (0),                 (0)},
    {"sse",        XV_SSE,        (0),                 (XVB_SSE2)},
    {"sse2",       XV_SSE2,       (XVB_SSE),           (XVB_SSE3)},
    {"sse3",       XV_SSE3,       (XVB_SSE2),          (XVB_SSSE3|XVB_SSE4A)},
    {"ssse3",      XV_SSSE3,      (XVB_SSE3),          (XVB_SSE4_1)},
    {"sse4.1",     XV_SSE4_1,     (XVB_SSSE3),         (XVB_SSE4_2)},
    {"sse4.2",     XV_SSE4_2,     (XVB_SSE4_1),        (XVB_AVX)},
    {"avx",        XV_AVX,        (XVB_SSE4_2),        (XVB_FMA|XVB_FMA4|XVB_F16C|XVB_AVX2)},
    {"avx2",       XV_AVX2,       (XVB_AVX),           (0)},
    {"sse4a",      XV_SSE4A,      (XVB_SSE3),          (XVB_FMA)},
    {"fma4",       XV_FMA4,       (XVB_SSE4A|XVB_AVX), (XVB_XOP)},
    {"xop",        XV_XOP,        (XVB_FMA4),          (0)},
    {"fma",        XV_FMA,        (XVB_AVX),           (0)},
    {"bmi",        XV_BMI,        (0),                 (0)},
    {"bmi2",       XV_BMI2,       (0),                 (0)},
    {"aes",        XV_AES,        (XVB_SSE2),          (0)},
    {"pclmul",     XV_PCLMUL,     (XVB_SSE2),          (0)},
    {"3dnow",      XV_3DNOW,      (XVB_MMX),           (XVB_3DNOWA)},
    {"3dnowa",     XV_3DNOWA,     (XVB_3DNOW),         (0)},
    {"clflushopt", XV_CLFLUSHOPT, (0),                 (0)},
    {"clwb",       XV_CLWB,       (0),                 (0)},
    {"clzero",     XV_CLZERO,     (0),                 (0)},
    {"f16c",       XV_F16C,       (XVB_AVX),           (0)},
    {"lzcnt",      XV_LZCNT,      (0),                 (0)},
    {"mwaitx",     XV_MWAITX,     (0),                 (0)},
    {"rdrnd",      XV_RDRND,      (0),                 (0)},
    {"rdseed",     XV_RDSEED,     (0),                 (0)},
    {"sha",        XV_SHA,        (XVB_SSE2),          (0)},
    {"abm",        XV_ABM,        (XVB_POPCNT),        (0)},
    {"tbm",        XV_TBM,        (0),                 (0)},
    {"avxvnni",    XV_AVXVNNI,    (XVB_AVX2),          (0)},
    {"xvec-emul",  XV_XVECEMUL,   (XVB_XVECEMUL_ON),   (XVB_XVECEMUL_OFF)},
    {0,            XV_LAST,       (0),                 (0)},
};

void __attribute__((used))
llvm::Elbrus::updateXVecFeatures( uint64_t &fmask, XVecOptInfo *xvoi,
                                  llvm::StringMap<bool> &Features, bool Enabled)
{
    uint64_t fbit = 1ULL << xvoi->xopt;

    if ( (fmask & fbit) == 0 ) {
        uint64_t extmask = Enabled ? xvoi->onopts : xvoi->offopts;

        Features[xvoi->name] = Enabled;
        fmask = fmask | fbit;
        for ( int i = 0; XVecOpts[i].name; ++i ) {
            if ( extmask & XV_MASK( XVecOpts[i].xopt) ) {
                llvm::Elbrus::updateXVecFeatures( fmask, XVecOpts + i, Features, Enabled);
            }
        }
    }

    return;
} /* llvm::Elbrus::updateXVecFeatures */

