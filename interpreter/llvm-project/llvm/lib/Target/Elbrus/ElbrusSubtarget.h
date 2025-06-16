//===-- ElbrusSubtarget.h - Define Subtarget for the Elbrus -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Elbrus specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ELBRUS_ELBRUSSUBTARGET_H
#define LLVM_LIB_TARGET_ELBRUS_ELBRUSSUBTARGET_H

#include "ElbrusFrameLowering.h"
#include "ElbrusISelLowering.h"
#include "ElbrusInstrInfo.h"
#include "ElbrusRegisterInfo.h"
#include "MCTargetDesc/ElbrusMCTargetDesc.h"
#include "TargetInfo/ElbrusTargetInfo.h"
#include "llvm/CodeGen/SelectionDAGTargetInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DataLayout.h"
#include <string>

namespace llvm
{
    class ElbrusTargetMachine;

    class ElbrusSubtarget : public TargetSubtargetInfo
    {
        protected:
            /// TargetTriple - What processor and OS we're targeting.
            Triple TargetTriple;

            /// stackAlignment - The minimum alignment known to hold of the stack frame on
            /// entry to the function and which must be maintained by every function.
            unsigned StackAlignment;

            const ElbrusTargetMachine &TM;
            ElbrusInstrInfo InstrInfo;
            ElbrusRegisterInfo RegInfo;
            ElbrusTargetLowering TLInfo;
            ElbrusFrameLowering FrameLowering;
            //TargetSelectionDAGInfo TSInfo;

        public:
            /// This constructor initializes the data members to match that
            /// of the specified triple.
            ///
            ElbrusSubtarget( const Triple &TT, StringRef &CPU, StringRef &TuneCPU, StringRef &FS,
                             const ElbrusTargetMachine &TM);

            /// getStackAlignment - Returns the minimum alignment known to hold of the
            /// stack frame on entry to the function and which must be maintained by every
            /// function for this subtarget.
            unsigned getStackAlignment() const { return StackAlignment; }

            const Triple &getTargetTriple() const { return TargetTriple; }
            const ElbrusTargetMachine &getTargetMachine() const { return TM; }
            const ElbrusFrameLowering *getFrameLowering() const override { return &FrameLowering; }
            const ElbrusInstrInfo *getInstrInfo() const override { return &InstrInfo; }
            const ElbrusTargetLowering *getTargetLowering() const override { return &TLInfo; }
            //const TargetSelectionDAGInfo *getSelectionDAGInfo() const override { return &TSInfo; }
            const ElbrusRegisterInfo *getRegisterInfo() const override { return &RegInfo; }

            /// initializeSubtargetDependencies - Initializes using a CPU and feature string
            /// so that we can use initializer lists for subtarget initialization.
            ElbrusSubtarget &initializeSubtargetDependencies( StringRef CPU, StringRef FS)
            {
                StackAlignment = 16;
                return (*this);
            }
    };
} // end namespace llvm

#endif

