//===- ElbrusMCTargetDesc.h - Elbrus target description ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ELBRUS_MCTARGETDESC_ELBRUSMCTARGETDESC_H
#define LLVM_LIB_TARGET_ELBRUS_MCTARGETDESC_ELBRUSMCTARGETDESC_H

#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/TargetParser/ElbrusTargetParser.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "TargetInfo/ElbrusTargetInfo.h" 

#define GET_SUBTARGETINFO_ENUM
#define GET_SUBTARGETINFO_MC_DESC

namespace llvm
{
    class ElbrusMCInstPrinter : public MCInstPrinter
    {
        public:
            ElbrusMCInstPrinter( const MCAsmInfo &mai, const MCInstrInfo &mii, const MCRegisterInfo &mri)
                : MCInstPrinter(mai, mii, mri) {}
            std::pair<const char *, uint64_t> getMnemonic( const MCInst *MI) override {}
            void printInst( const MCInst *MI, uint64_t Address, StringRef Annot,
                            const MCSubtargetInfo &STI, raw_ostream &OS) override {}
    };

    struct ElbrusMCSubtargetInfo : public MCSubtargetInfo
    {
        ElbrusMCSubtargetInfo( const Triple &TT, StringRef CPU, StringRef TuneCPU, StringRef FS,
                               const MCWriteProcResEntry *WPR, const MCWriteLatencyEntry *WL,
                               const MCReadAdvanceEntry *RA, const InstrStage *IS,
                               const unsigned *OC, const unsigned *FP)
            : MCSubtargetInfo( TT, ElbrusCPUName::get( CPU), TuneCPU, FS,
                               getFeatures(), getSubtypes(), WPR, WL, RA, IS, OC, FP) {}
        static const std::vector<SubtargetFeatureKV> &getFeatures();
        static const std::vector<SubtargetSubTypeKV> &getSubtypes();
    };
} // end namespace llvm

#endif

