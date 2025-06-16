//===-- ElbrusSubtarget.cpp - Elbrus Subtarget Information ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Elbrus specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "ElbrusSubtarget.h"
#include "ElbrusTargetMachine.h"

using namespace llvm;

ElbrusSubtarget::ElbrusSubtarget( const Triple &TT, StringRef &CPU,
                                  StringRef &TuneCPU, StringRef &FS,
                                  const ElbrusTargetMachine &TM)
    : TargetSubtargetInfo( TT, ElbrusCPUName::get( CPU), TuneCPU, FS,
                           ElbrusMCSubtargetInfo::getFeatures(),
                           ElbrusMCSubtargetInfo::getSubtypes(),
                           0, 0, 0, 0, 0, 0),
      TargetTriple( TT), TM(TM),
      InstrInfo( initializeSubtargetDependencies( ElbrusCPUName::get( CPU), FS)),
      TLInfo( TM, *this), FrameLowering( *this)
{
}

