//===-- ElbrusISelLowering.cpp - Elbrus DAG Lowering Implementation -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the interfaces that Elbrus uses to lower LLVM code into
// a selection DAG.
//
//===----------------------------------------------------------------------===//

#include "ElbrusRegisterInfo.h"
#include "ElbrusISelLowering.h"
#include "ElbrusTargetMachine.h"

using namespace llvm;

#define GET_REGINFO_MC_DESC
#define GET_REGINFO_TARGET_DESC
#include "ElbrusGenRegisterInfo.inc"

ElbrusTargetLowering::ElbrusTargetLowering( const TargetMachine &TM,
                                            const ElbrusSubtarget &STI)
    : TargetLowering(TM)
{
    // Set up the register classes.
    addRegisterClass( MVT::i32, &Elbrus::GR32RegClass);
    addRegisterClass( MVT::i64, &Elbrus::GR64RegClass);

    computeRegisterProperties( STI.getRegisterInfo());
}

