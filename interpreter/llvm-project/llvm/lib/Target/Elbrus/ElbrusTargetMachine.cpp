//===-- ElbrusTargetMachine.cpp - Define TargetMachine for Elbrus ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "ElbrusTargetMachine.h"
#include "ElbrusTargetObjectFile.h"
#include "ElbrusTargetTransformInfo.h"
#include "TargetInfo/ElbrusTargetInfo.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

extern "C" LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeElbrusTarget()
{
    RegisterTargetMachine<Elbrus32TargetMachine> A( getTheElbrus32Target());
    RegisterTargetMachine<Elbrus64TargetMachine> B( getTheElbrus64Target());
    RegisterTargetMachine<Elbrus128TargetMachine> C( getTheElbrus128Target());
}

static std::string
computeDataLayout( const Triple T)
{
    std::string r = "e-m:e";

    if ( T.getArch() == Triple::e2k128 ) {
        r += "-p:128:128";
    } else if ( T.getArch() == Triple::e2k64 ) {
        r += "-p:64:64";
    } else if ( T.getArch() == Triple::e2k32 ) {
        r += "-p:32:32";
    } else {
        assert( 0);
    }

    r += "-i64:64:64-f80:128:128-n32:64-S128";

    return (r);
}

static Reloc::Model
getEffectiveRelocModel( std::optional<Reloc::Model> RM)
{
    Reloc::Model r = RM.has_value() ? (*RM) : Reloc::Static;

    return (r);
}

ElbrusTargetMachine::ElbrusTargetMachine( const Target &T, const Triple &TT, StringRef CPU,
                                          StringRef TuneCPU, StringRef FS, const TargetOptions &Options,
                                          std::optional<Reloc::Model> RM, std::optional<CodeModel::Model> CM,
                                          CodeGenOptLevel OL, bool JIT)
    : LLVMTargetMachine( T, computeDataLayout(TT), TT, CPU, FS, Options,
                         getEffectiveRelocModel( RM),
                         getEffectiveCodeModel(CM, CodeModel::Large), OL),
      TLOF( std::make_unique<ElbrusLinuxTargetObjectFile>()),
      Subtarget( TT, CPU, TuneCPU, FS, *this)
{
    initAsmInfo();
}

ElbrusTargetMachine::~ElbrusTargetMachine() {
}

TargetTransformInfo
ElbrusTargetMachine::getTargetTransformInfo( const Function &F) const {
  return TargetTransformInfo( ElbrusTTIImpl( this, F));
}

void
Elbrus32TargetMachine::anchor() {
}

Elbrus32TargetMachine::Elbrus32TargetMachine( const Target &T, const Triple &TT,
                                              StringRef CPU, StringRef FS,
                                              const TargetOptions &Options,
                                              std::optional<Reloc::Model> RM,
                                              std::optional<CodeModel::Model> CM,
                                              CodeGenOptLevel OL, bool JIT)
    : ElbrusTargetMachine( T, TT, CPU, "", FS, Options, RM, CM, OL, JIT)
{
}

void
Elbrus64TargetMachine::anchor() {
}

Elbrus64TargetMachine::Elbrus64TargetMachine( const Target &T, const Triple &TT,
                                              StringRef CPU, StringRef FS,
                                              const TargetOptions &Options,
                                              std::optional<Reloc::Model> RM,
                                              std::optional<CodeModel::Model> CM,
                                              CodeGenOptLevel OL, bool JIT)
    : ElbrusTargetMachine( T, TT, CPU, "", FS, Options, RM, CM, OL, JIT)
{
}

void
Elbrus128TargetMachine::anchor() {
}

Elbrus128TargetMachine::Elbrus128TargetMachine( const Target &T, const Triple &TT,
                                                StringRef CPU, StringRef FS,
                                                const TargetOptions &Options,
                                                std::optional<Reloc::Model> RM,
                                                std::optional<CodeModel::Model> CM,
                                                CodeGenOptLevel OL, bool JIT)
    : ElbrusTargetMachine( T, TT, CPU, "", FS, Options, RM, CM, OL, JIT)
{
}

