//===- ElbrusInstPrinter.h - Elbrus target description --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ELBRUS_MCTARGETDESC_ELBRUSINSTPRINTER_H
#define LLVM_LIB_TARGET_ELBRUS_MCTARGETDESC_ELBRUSINSTPRINTER_H

#include "llvm/MC/MCInstPrinter.h"

namespace llvm
{
    class ElbrusInstPrinter : public MCInstPrinter
    {
        public:
            ElbrusInstPrinter( const MCAsmInfo &MAI, const MCInstrInfo &MII,
                               const MCRegisterInfo &MRI)
                : MCInstPrinter( MAI, MII, MRI) {}
            virtual void printInst( const MCInst *MI, raw_ostream &O, StringRef Annot,
                                    const MCSubtargetInfo &STI) {}
    };
} // end namespace llvm

#endif

