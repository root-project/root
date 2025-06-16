//===-- ElbrusFrameLowering.cpp - Elbrus Frame Information ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Elbrus implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "ElbrusFrameLowering.h"

using namespace llvm;

ElbrusFrameLowering::ElbrusFrameLowering( const ElbrusSubtarget &STI)
    : TargetFrameLowering( TargetFrameLowering::StackGrowsDown, Align( 16), 0)
{
}

void
ElbrusFrameLowering::emitPrologue( MachineFunction &MF, MachineBasicBlock &MBB) const
{
    llvm_unreachable( "ElbrusFrameLowering::emitPrologue: Unsupported");
}

void
ElbrusFrameLowering::emitEpilogue( MachineFunction &MF, MachineBasicBlock &MBB) const
{
    llvm_unreachable( "ElbrusFrameLowering::emitPrologue: Unsupported");
}

