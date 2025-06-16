//===- ElbrusMCAsmInfo.h - Elbrus asm properties ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the ElbrusMCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ELBRUS_MCTARGETDESC_ELBRUSMCASMINFO_H
#define LLVM_LIB_TARGET_ELBRUS_MCTARGETDESC_ELBRUSMCASMINFO_H

#include "llvm/MC/MCAsmInfoELF.h"

namespace llvm
{
    class Triple;

    class ElbrusMCAsmInfo : public MCAsmInfoELF
    {
        public:
            explicit ElbrusMCAsmInfo( const Triple &T) {}
    };
} // end namespace llvm

#endif

