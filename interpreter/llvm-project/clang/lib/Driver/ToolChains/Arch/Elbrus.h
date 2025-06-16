//===--- Elbrus.h - Elbrus-specific Tool Helpers --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_ARCH_ELBRUS_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_ARCH_ELBRUS_H

#include "clang/Driver/Driver.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Option/Option.h"
#include <vector>

namespace clang {
namespace driver {
namespace tools {
namespace elbrus {

std::string getElbrusTargetCPU( const llvm::opt::ArgList &Args);

void getElbrusTargetFeatures( const Driver &D, const llvm::Triple &Triple,
                              const llvm::opt::ArgList &Args,
                              std::vector<llvm::StringRef> &Features);

} // end namespace elbrus
} // end namespace target
} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_ARCH_ELBRUS_H
