//=- llvm/CodeGen/Lccrt/LccrtIpa.h - Lccrt-IR IPA-translation -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LccrtModuleIpaEmitter/LccrtFunctionIpaEmitter classes, which
// implement translation IPA-Results from LLVM-IR to LCCRT-IR.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_LCCRT_LCCRTIPA_H
#define LLVM_LIB_CODEGEN_LCCRT_LCCRTIPA_H

#include "llvm/CodeGen/LccrtIpaPass.h"
#include "llvm/Support/Compiler.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCMachObjectWriter.h"

#ifdef LLVM_WITH_LCCRT
#include "lccrt.h"

namespace llvm {

typedef struct
{
    unsigned int length;
    void *items;
} MetaAliases;

class LLVM_LIBRARY_VISIBILITY LccrtModuleIpaEmitter
{
public:
    LccrtModuleIpaEmitter( Pass *parentPass);

    void
    open( lccrt_module_ptr, const Module *);

    void
    close();

    lccrt_eic_t
    getMetadata() const;

    const FunctionIPAResults *
    getFunctionIPAResults(GlobalValue::GUID) const;

    const AliasResult **
    getAliases() const;

    static IPAResults *
    findIPAResults( Pass *);

private:
    Pass *parentPass;
    const IPAResults *ipaResults;
    MetaAliases *aliases;
    lccrt_eic_t ecat_ipa;
};

class LLVM_LIBRARY_VISIBILITY LccrtFunctionIpaEmitter
{
public:
    LccrtFunctionIpaEmitter( LccrtModuleIpaEmitter *,
                             lccrt_function_ptr,
                             const Function *);

    void
    setOperIpaResult( lccrt_oper_ptr, const Instruction &);

private:
    LccrtModuleIpaEmitter *mipa;
    const FunctionIPAResults *functionIPAResults;
    const MetaAliases *aliases;
    lccrt_eic_t ecat_ipa;
};

} /* llvm */
#endif /* LLVM_WITH_LCCRT */

#endif /* LLVM_LIB_CODEGEN_LCCRT_LCCRTIPA_H */
