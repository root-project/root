//===-- ElbrusTargetObjectFile.h - Elbrus Object Info -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ELBRUS_ELBRUSTARGETOBJECTFILE_H
#define LLVM_LIB_TARGET_ELBRUS_ELBRUSTARGETOBJECTFILE_H

#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"

namespace llvm
{

    class MCContext;
    class TargetMachine;

    class ElbrusLinuxTargetObjectFile : public TargetLoweringObjectFileELF
    {
        public:
            ElbrusLinuxTargetObjectFile();
            void Initialize( MCContext &Ctx, const TargetMachine &TM) override;
            const MCExpr *getDebugThreadLocalSymbol( const MCSymbol *Sym) const override;
    };

} // end namespace llvm

#endif

