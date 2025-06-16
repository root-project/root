//===-- ElbrusTargetInfo.cpp - Elbrus Target Implementation ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TargetInfo/ElbrusTargetInfo.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

Target &
llvm::getTheElbrus32Target() {
    static Target TheElbrus32Target;
    return TheElbrus32Target;
}

Target &
llvm::getTheElbrus64Target() {
    static Target TheElbrus64Target;
    return TheElbrus64Target;
}

Target &
llvm::getTheElbrus128Target() {
    static Target TheElbrus128Target;
    return TheElbrus128Target;
}

extern "C" LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeElbrusTargetInfo()
{
    Target &targ32 = getTheElbrus32Target();
    Target &targ64 = getTheElbrus64Target();
    Target &targ128 = getTheElbrus128Target();

    RegisterTarget<Triple::e2k32,  /*HasJIT=*/true>  X( targ32,  "e2k32",  "Elbrus 32",  "Elbrus");
    RegisterTarget<Triple::e2k64,  /*HasJIT=*/true>  Y( targ64,  "e2k64",  "Elbrus 64",  "Elbrus");
    RegisterTarget<Triple::e2k128, /*HasJIT=*/false> Z( targ128, "e2k128", "Elbrus 128", "Elbrus");
}

ElbrusCPUName::ElbrusCPUName( const std::string &CPU)
    : CPUName( preprocessCPUName( CPU))
{
}

ElbrusCPUName::ElbrusCPUName( const StringRef &CPU)
    : CPUName( preprocessCPUName( CPU.str()))
{
}

const std::string
ElbrusCPUName::preprocessCPUName( const std::string &CPU)
{
    return (CPU == "generic" ? "elbrus-v2" : CPU);
}

const StringRef&
ElbrusCPUName::get( const std::string &CPU)
{
    static StringRef V2     = StringRef( "elbrus-v2");
    static StringRef V2_2CP = StringRef( "elbrus-2c+");
    static StringRef V3     = StringRef( "elbrus-v3");
    static StringRef V3_4C  = StringRef( "elbrus-4c");
    static StringRef V4     = StringRef( "elbrus-v4");
    static StringRef V4_8C  = StringRef( "elbrus-8c");
    static StringRef V4_1CP = StringRef( "elbrus-1c+");
    static StringRef V5     = StringRef( "elbrus-v5");
    static StringRef V5_8C2 = StringRef( "elbrus-8c2");
    static StringRef V6     = StringRef( "elbrus-v6");
    static StringRef V6_16C = StringRef( "elbrus-16c");
    static StringRef V6_2C3 = StringRef( "elbrus-2c3");
    static StringRef V6_12C = StringRef( "elbrus-12c");
    static StringRef V7     = StringRef( "elbrus-v7");
    static StringRef V7_48C = StringRef( "elbrus-48c");
    static StringRef V7_8V  = StringRef( "elbrus-8v7");

    if ( (CPU == "elbrus-v2")      )  { return V2; }
    else if ( (CPU == "elbrus-2c+") ) { return V2_2CP; }
    else if ( (CPU == "elbrus-v3") )  { return V3; }
    else if ( (CPU == "elbrus-4c") )  { return V3_4C; }
    else if ( (CPU == "elbrus-v4") )  { return V4; }
    else if ( (CPU == "elbrus-8c") )  { return V4_8C; }
    else if ( (CPU == "elbrus-1c+") ) { return V4_1CP; }
    else if ( (CPU == "elbrus-v5") )  { return V5; }
    else if ( (CPU == "elbrus-8c") )  { return V5_8C2; }
    else if ( (CPU == "elbrus-v6") )  { return V6; }
    else if ( (CPU == "elbrus-16c") ) { return V6_16C; }
    else if ( (CPU == "elbrus-2c3") ) { return V6_2C3; }
    else if ( (CPU == "elbrus-12c") ) { return V6_12C; }
    else if ( (CPU == "elbrus-v7") )  { return V7; }
    else if ( (CPU == "elbrus-48c") ) { return V7_48C; }
    else if ( (CPU == "elbrus-8v7") ) { return V7_8V; }

    return V2;
}

const StringRef &
ElbrusCPUName::get( const StringRef &CPU)
{
    std::string scpu( CPU.data());

    return ElbrusCPUName::get( scpu);
}

#if 0
extern "C" void LLVMInitializeElbrusTargetInfo();
extern "C" void LLVMInitializeElbrusTarget();
extern "C" void LLVMInitializeElbrusTargetMC();
extern "C" void LLVMInitializeElbrusAsmPrinter();

void __attribute__((constructor)) initElbrus() {
    LLVMInitializeElbrusTargetInfo();
    LLVMInitializeElbrusTarget();
    LLVMInitializeElbrusTargetMC();
    LLVMInitializeElbrusAsmPrinter();
}
#endif

