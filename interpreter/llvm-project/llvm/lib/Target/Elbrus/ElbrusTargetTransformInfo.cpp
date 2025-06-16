//===-- X86TargetTransformInfo.cpp - Elbrus specific TTI pass -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements a TargetTransformInfo analysis pass specific to the
/// Elbrus target machine. It uses the target's detailed information to provide
/// more precise answers to certain TTI queries, while letting the target
/// independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#include "ElbrusTargetTransformInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/CodeGen/CostTable.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/Debug.h"
#include <optional>

using namespace llvm;

InstructionCost ElbrusTTIImpl::getVectorInstrCost( unsigned Opcode, Type *Val,
                                                   TTI::TargetCostKind CostKind,
                                                   unsigned Index, Value *Op0,
                                                   Value *Op1)
{
    assert( Val->isVectorTy() && "This must be a vector type");

    if ( (Opcode == Instruction::ExtractElement)
         || (Opcode == Instruction::InsertElement) )
    {
        return 10;
    }

    return BaseT::getVectorInstrCost( Opcode, Val, CostKind, Index, Op0, Op1);
} /* ElbrusTTIImpl::getVectorInstrCost */
