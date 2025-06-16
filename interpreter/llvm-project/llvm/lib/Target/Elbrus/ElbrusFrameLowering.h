//===- ElbrusFrameLowering.h - Define frame lowering for Elbrus -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ELBRUS_ELBRUSFRAMELOWERING_H
#define LLVM_LIB_TARGET_ELBRUS_ELBRUSFRAMELOWERING_H

#include "llvm/CodeGen/TargetFrameLowering.h"

namespace llvm
{
    class ElbrusSubtarget;

    class ElbrusFrameLowering: public TargetFrameLowering
    {
        public:
            explicit ElbrusFrameLowering( const ElbrusSubtarget &ST);
            void emitPrologue( MachineFunction &MF, MachineBasicBlock &MBB) const override;
            void emitEpilogue( MachineFunction &MF, MachineBasicBlock &MBB) const override;
            bool hasFP(const MachineFunction &MF) const override { return (true); }
    };
} // end llvm namespace

#endif
