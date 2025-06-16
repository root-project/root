//===-- ElbrusTargetTransformInfo.h - Elbrus specific TTI -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file a TargetTransformInfo::Concept conforming object specific to the
/// Elbrus target machine. It uses the target's detailed information to
/// provide more precise answers to certain TTI queries, while letting the
/// target independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ELBRUS_X86TARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_ELBRUS_X86TARGETTRANSFORMINFO_H

#include "ElbrusTargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include <optional>

namespace llvm {

class InstCombiner;

class ElbrusTTIImpl : public BasicTTIImplBase<ElbrusTTIImpl> {
  typedef BasicTTIImplBase<ElbrusTTIImpl> BaseT;
  typedef TargetTransformInfo TTI;
  friend BaseT;

  const ElbrusSubtarget *ST;
  const ElbrusTargetLowering *TLI;

  const ElbrusSubtarget *getST() const { return ST; }
  const ElbrusTargetLowering *getTLI() const { return TLI; }

public:
  explicit ElbrusTTIImpl( const ElbrusTargetMachine *TM, const Function &F)
      : BaseT( TM, F.getParent()->getDataLayout()), ST( TM->getSubtargetImpl( F)),
        TLI( ST->getTargetLowering()) {}

  using BaseT::getVectorInstrCost;
  InstructionCost getVectorInstrCost( unsigned Opcode, Type *Val,
                                      TTI::TargetCostKind CostKind,
                                      unsigned Index, Value *Op0, Value *Op1);
};

} // end namespace llvm

#endif
