//===- ElbrusTargetMachine.h - Define TargetMachine for Elbrus --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Elbrus specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ELBRUS_ELBRUSTARGETMACHINE_H
#define LLVM_LIB_TARGET_ELBRUS_ELBRUSTARGETMACHINE_H

#include "ElbrusSubtarget.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class ElbrusTargetMachine : public LLVMTargetMachine {
private:
    std::unique_ptr<TargetLoweringObjectFile> TLOF;
    ElbrusSubtarget Subtarget;
public:
    ElbrusTargetMachine( const Target &T, const Triple &TT, StringRef CPU,
                         StringRef TuneCPU, StringRef FS, const TargetOptions &Options,
                         std::optional<Reloc::Model> RM, std::optional<CodeModel::Model> CM,
                         CodeGenOptLevel OL, bool JIT);
    ~ElbrusTargetMachine() override;

    TargetTransformInfo getTargetTransformInfo( const Function &F) const override;

    const ElbrusSubtarget *getSubtargetImpl( const Function &F) const override { return &Subtarget; }
    TargetLoweringObjectFile *getObjFileLowering() const override { return TLOF.get(); }
    bool isCodeGenLccrt() const override { return (true); };
};

/// Elbrus 32-bit target machine.
///
class Elbrus32TargetMachine : public ElbrusTargetMachine
{
private:
    virtual void anchor();
public:
    Elbrus32TargetMachine( const Target &T, const Triple &TT, StringRef CPU,
                           StringRef FS, const TargetOptions &Options,
                           std::optional<Reloc::Model> RM, std::optional<CodeModel::Model> CM,
                           CodeGenOptLevel OL, bool JIT);
};

/// Elbrus 64-bit target machine.
///
class Elbrus64TargetMachine : public ElbrusTargetMachine
{
private:
    virtual void anchor();
public:
    Elbrus64TargetMachine( const Target &T, const Triple &TT, StringRef CPU,
                           StringRef FS, const TargetOptions &Options,
                           std::optional<Reloc::Model> RM, std::optional<CodeModel::Model> CM,
                           CodeGenOptLevel OL, bool JIT);
};

/// Elbrus 128-bit target machine.
///
class Elbrus128TargetMachine : public ElbrusTargetMachine
{
private:
    virtual void anchor();
public:
    Elbrus128TargetMachine( const Target &T, const Triple &TT, StringRef CPU,
                            StringRef FS, const TargetOptions &Options,
                            std::optional<Reloc::Model> RM, std::optional<CodeModel::Model> CM,
                            CodeGenOptLevel OL, bool JIT);
};

} // end namespace llvm

#endif

