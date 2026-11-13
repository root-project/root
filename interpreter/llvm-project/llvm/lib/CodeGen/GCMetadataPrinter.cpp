//===- GCMetadataPrinter.cpp - Garbage collection infrastructure ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the abstract base class GCMetadataPrinter.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GCMetadataPrinter.h"

using namespace llvm;

LLVM_INSTANTIATE_REGISTRY(GCMetadataPrinterRegistry)

GCMetadataPrinter::GCMetadataPrinter() = default;

GCMetadataPrinter::~GCMetadataPrinter() = default;

void GCMetadataPrinter::beginAssembly(Module &M, GCModuleInfo &Info, AsmPrinter &AP) {}

void GCMetadataPrinter::finishAssembly(Module &M, GCModuleInfo &Info, AsmPrinter &AP) {}

bool GCMetadataPrinter::emitStackMaps(StackMaps &SM, AsmPrinter &AP) { return false; }
