//===-- ElbrusInstrInfo.h - Elbrus Instruction Information ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Elbrus implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ELBRUS_ELBRUSINSTRINFO_H
#define LLVM_LIB_TARGET_ELBRUS_ELBRUSINSTRINFO_H

#include "llvm/CodeGen/TargetInstrInfo.h"

namespace llvm
{
    class ElbrusSubtarget;

    class ElbrusInstrInfo : public TargetInstrInfo
    {
        public:
            explicit ElbrusInstrInfo( ElbrusSubtarget &ST) : TargetInstrInfo() {}
    };

} // end namespace llvm

#endif

