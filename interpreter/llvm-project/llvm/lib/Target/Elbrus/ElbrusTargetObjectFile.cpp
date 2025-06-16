//===------- ElbrusTargetObjectFile.cpp - Elbtus Object Info Impl ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ElbrusTargetObjectFile.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

ElbrusLinuxTargetObjectFile::ElbrusLinuxTargetObjectFile()
    : TargetLoweringObjectFileELF()
{
}

void
ElbrusLinuxTargetObjectFile::Initialize( MCContext &Ctx, const TargetMachine &TM)
{
    TargetLoweringObjectFileELF::Initialize( Ctx, TM);
    InitializeELF( TM.Options.UseInitArray);
}

const MCExpr *
ElbrusLinuxTargetObjectFile::getDebugThreadLocalSymbol( const MCSymbol *Sym) const
{
    const MCExpr *r = 0;

    return (r);
}

