//===-- ElbrusTargetInfo.h - Elbrus Target Implementation -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ELBRUS_TARGETINFO_ELBRUSTARGETINFO_H
#define LLVM_LIB_TARGET_ELBRUS_TARGETINFO_ELBRUSTARGETINFO_H

#include "llvm/ADT/StringRef.h"

namespace llvm
{
    class Target;

    Target &getTheElbrus32Target();
    Target &getTheElbrus64Target();
    Target &getTheElbrus128Target();

    struct ElbrusCPUName
    {
        StringRef CPUName;
        ElbrusCPUName( const std::string &CPU);
        ElbrusCPUName( const StringRef &CPU);
        static const std::string preprocessCPUName( const std::string &CPU);
        static const StringRef& get( const std::string &CPU);
        static const StringRef& get( const StringRef &CPU);
    };
} // namespace llvm

#endif

