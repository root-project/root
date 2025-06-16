//===-- ElbrusISelLowering.h - Elbrus DAG Lowering Interface ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that Elbrus uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ELBRUS_ELBRUSISELLOWERING_H
#define LLVM_LIB_TARGET_ELBRUS_ELBRUSISELLOWERING_H

#include "llvm/CodeGen/TargetLowering.h"

namespace llvm
{
    class ElbrusSubtarget;

    class ElbrusTargetLowering : public TargetLowering
    {
        public:
            explicit ElbrusTargetLowering( const TargetMachine &TM, const ElbrusSubtarget &STI);
    };

} // end namespace llvm

#endif 

