//===-- ElbrusRegisterInfo.h - Elbrus Register Information Impl -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Elbrus implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ELBRUS_ELBRUSREGISTERINFO_H
#define LLVM_LIB_TARGET_ELBRUS_ELBRUSREGISTERINFO_H

#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

#define GET_REGINFO_ENUM
#define GET_REGINFO_HEADER
#include "ElbrusGenRegisterInfo.inc"

namespace llvm
{
    class ElbrusRegisterInfo : public TargetRegisterInfo
    {
        private:
            RegClassWeight RCW;  
        public:
            explicit ElbrusRegisterInfo()
                : TargetRegisterInfo( 0, 0, 0, 0, 0, LaneBitmask(0), 0, 0) {}
            const MCPhysReg *getCalleeSavedRegs( const MachineFunction *MF) const override { return (0); }
            ArrayRef<const uint32_t *> getRegMasks() const override { return ArrayRef<const uint32_t *>(); }
            ArrayRef<const char *> getRegMaskNames() const override { return ArrayRef<const char *>(); }
            BitVector getReservedRegs( const MachineFunction &MF) const override { return BitVector(); }
            const RegClassWeight &getRegClassWeight( const TargetRegisterClass *RC) const override { return RCW; }
            unsigned getRegUnitWeight( unsigned RegUnit) const override { return (0); }
            unsigned getNumRegPressureSets() const override { return (0); }
            const char *getRegPressureSetName( unsigned Idx) const override { return (0); }
            unsigned getRegPressureSetLimit( const MachineFunction &MF, unsigned Idx) const override { return (0); }
            const int *getRegClassPressureSets( const TargetRegisterClass *RC) const override { return (0); }
            const int *getRegUnitPressureSets( unsigned RegUnit) const override { return (0); }
            bool eliminateFrameIndex( MachineBasicBlock::iterator MI, int SPAdj,
                                      unsigned FIOperandNum, RegScavenger *RS = nullptr) const override {
                return (0); }
            Register getFrameRegister( const MachineFunction &MF) const override { return (0); }
    };
} // end namespace llvm

#endif

