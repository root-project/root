//===-- llvm/CodeGen/Lccrt.h - Lccrt Framework --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a class to be used as the base class for target specific
// asm writers.  This class primarily handles common functionality used by
// all asm writers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LCCRT_H
#define LLVM_CODEGEN_LCCRT_H

#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

#include <string>

namespace llvm
{
    class Triple;
    class Pass;
    class MCCodeEmitter;
    class MCAsmBackend;
    class LLVMContext;
    class Module;
    class Function;
    class BasicBlock;
    class GlobalVariable;
    class Type;
    class PointerType;
    class StructType;
    class Constant;

    namespace Lccrt
    {
        Pass *createAsmPass( TargetMachine &TM, raw_pwrite_stream &Out, CodeGenFileType type);
        MCCodeEmitter *createMCCodeEmitter( const Triple &);
        MCAsmBackend *createMCAsmBackend( const Triple &);
        std::string getVersion( const Triple &);
        std::string getToolchain( const Triple &, const char *name);
        std::string getToolchainPath( const Triple &, const char *name);
        std::string getLibPath( const Triple &, const char *name);
        std::string getIncludePath( const Triple &, const char *name);
    }
}

#endif
