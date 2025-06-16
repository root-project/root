//===- ElbrusMCTargetDesc.h - Elbrus target description -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "ElbrusMCTargetDesc.h"
#include "llvm/MC/MCRegisterInfo.h"

using namespace llvm;
using namespace llvm::Elbrus;

static const llvm::MCSchedModel GenericModel = {
};

enum ElbrusSubTypeName
{
    S_ELBRUS_12C = 0,
    S_ELBRUS_16C,
    S_ELBRUS_1CP,
    S_ELBRUS_2CP,
    S_ELBRUS_2C3,
    S_ELBRUS_48C,
    S_ELBRUS_4C,
    S_ELBRUS_8C,
    S_ELBRUS_8C2,
    S_ELBRUS_8V7,
    S_ELBRUS_V2,
    S_ELBRUS_V3,
    S_ELBRUS_V4,
    S_ELBRUS_V5,
    S_ELBRUS_V6,
    S_ELBRUS_V7,
    S_SUBTYPE_LAST
};

enum ElbrusFeatureMacroMask
{
    FMM_VEC = ~0ULL,
};

/* ElbrusSubTypeKV */
static llvm::SubtargetSubTypeKV ElbrusSubTypeKV[ElbrusSubTypeName::S_SUBTYPE_LAST] = {
    {"elbrus-12c",    {{{FMM_VEC, 0x0ULL, 0x0ULL,}}}, {{{}}}, &MCSchedModel::Default},
    {"elbrus-16c",    {{{FMM_VEC, 0x0ULL, 0x0ULL,}}}, {{{}}}, &MCSchedModel::Default},
    {"elbrus-1c+",    {{{FMM_VEC, 0x0ULL, 0x0ULL,}}}, {{{}}}, &MCSchedModel::Default},
    {"elbrus-2c+",    {{{FMM_VEC, 0x0ULL, 0x0ULL,}}}, {{{}}}, &MCSchedModel::Default},
    {"elbrus-2c3",    {{{FMM_VEC, 0x0ULL, 0x0ULL,}}}, {{{}}}, &MCSchedModel::Default},
    {"elbrus-48c",    {{{FMM_VEC, 0x0ULL, 0x0ULL,}}}, {{{}}}, &MCSchedModel::Default},
    {"elbrus-4c",     {{{FMM_VEC, 0x0ULL, 0x0ULL,}}}, {{{}}}, &MCSchedModel::Default},
    {"elbrus-8c",     {{{FMM_VEC, 0x0ULL, 0x0ULL,}}}, {{{}}}, &MCSchedModel::Default},
    {"elbrus-8c2",    {{{FMM_VEC, 0x0ULL, 0x0ULL,}}}, {{{}}}, &MCSchedModel::Default},
    {"elbrus-8v7",    {{{FMM_VEC, 0x0ULL, 0x0ULL,}}}, {{{}}}, &MCSchedModel::Default},
    {"elbrus-v2",     {{{FMM_VEC, 0x0ULL, 0x0ULL,}}}, {{{}}}, &MCSchedModel::Default},
    {"elbrus-v3",     {{{FMM_VEC, 0x0ULL, 0x0ULL,}}}, {{{}}}, &MCSchedModel::Default},
    {"elbrus-v4",     {{{FMM_VEC, 0x0ULL, 0x0ULL,}}}, {{{}}}, &MCSchedModel::Default},
    {"elbrus-v5",     {{{FMM_VEC, 0x0ULL, 0x0ULL,}}}, {{{}}}, &MCSchedModel::Default},
    {"elbrus-v6",     {{{FMM_VEC, 0x0ULL, 0x0ULL,}}}, {{{}}}, &MCSchedModel::Default},
    {"elbrus-v7",     {{{FMM_VEC, 0x0ULL, 0x0ULL,}}}, {{{}}}, &MCSchedModel::Default},
};

const std::vector<SubtargetFeatureKV> &
ElbrusMCSubtargetInfo::getFeatures() {
    static bool isinit = false;
    static std::vector<SubtargetFeatureKV> farr;
    static pthread_mutex_t sync = PTHREAD_MUTEX_INITIALIZER;

    if ( !isinit ) {
        pthread_mutex_lock( &sync);
        if ( !isinit ) {
            for ( int i = 0; XVecOpts[i].name ; ++i ) {
                //uint64_t fbit = XV_MASK( XVecOpts[i].xopt);
                uint64_t fbits = XVecOpts[i].onopts;
                SubtargetFeatureKV fkv = {0, 0, 0, {{{fbits,0x0,0x0}}}};

                assert( i < llvm::Elbrus::XV_LAST);
                assert( XVecOpts[i].xopt == i);
                fkv.Key = XVecOpts[i].name;
                fkv.Desc = "Enable instsructions";
                fkv.Value = XVecOpts[i].xopt;
                farr.push_back( fkv);
            }
            std::sort( farr.begin(), farr.end());
            assert( farr.size() == llvm::Elbrus::XV_LAST);
            isinit = true;
        }
        pthread_mutex_unlock( &sync);
    }

    return (farr);
}

const std::vector<SubtargetSubTypeKV> &
ElbrusMCSubtargetInfo::getSubtypes() {
    static bool isinit = false;
    static std::vector<SubtargetSubTypeKV> tarr;
    static pthread_mutex_t sync = PTHREAD_MUTEX_INITIALIZER;

    if ( !isinit ) {
        pthread_mutex_lock( &sync);
        if ( !isinit ) {
            for ( int i = 0; i < ElbrusSubTypeName::S_SUBTYPE_LAST; ++i ) {
                tarr.push_back( ElbrusSubTypeKV[i]);
            }
            assert( tarr.size() == ElbrusSubTypeName::S_SUBTYPE_LAST);
            isinit = true;
        }
        pthread_mutex_unlock( &sync);
    }

    return (tarr);
}

static MCAsmInfo *
createElbrusMCAsmInfo( const MCRegisterInfo &MRI, const Triple &TheTriple,
                       const MCTargetOptions &Options)
{
    MCAsmInfo *r = new MCAsmInfo();
    return (r);
}

static MCInstPrinter *
createElbrusMCInstPrinter( const Triple &T,
                           unsigned SyntaxVariant, const MCAsmInfo &MAI,
                           const MCInstrInfo &MII, const MCRegisterInfo &MRI)
{
    MCInstPrinter *r = new ElbrusMCInstPrinter( MAI, MII, MRI);
    return (r);
}

static MCRegisterInfo *
createElbrusMCRegisterInfo(const Triple &TT) {
    MCRegisterInfo *r = new MCRegisterInfo();  

    return r;
}

static MCSubtargetInfo *
createElbrusMCSubtargetInfo( const Triple &TT, StringRef CPU, StringRef FS)
{
    MCSubtargetInfo *r;
    StringRef TuneCPU;
   
    r = new ElbrusMCSubtargetInfo( TT, CPU, TuneCPU, FS, nullptr, nullptr,
                                   nullptr, nullptr, nullptr, nullptr);

    return (r);
}

extern "C" LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeElbrusTargetMC()
{
    for ( Target *T : {&getTheElbrus32Target(), &getTheElbrus64Target(), &getTheElbrus128Target()} )
    {
        // Register the MC asm info.
        RegisterMCAsmInfoFn C( *T, createElbrusMCAsmInfo);

        // Register the MCInstPrinter.
        TargetRegistry::RegisterMCInstPrinter( *T, createElbrusMCInstPrinter);

        // Register the MCSubtargetInfo.
        TargetRegistry::RegisterMCSubtargetInfo( *T, createElbrusMCSubtargetInfo);

        // Register the MC register info.
        TargetRegistry::RegisterMCRegInfo( *T, createElbrusMCRegisterInfo);
    }
}

