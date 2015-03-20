//===--- CodeGen/ModuleBuilder.h - Build LLVM from ASTs ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ModuleBuilder interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CODEGEN_MODULEBUILDER_H
#define LLVM_CLANG_CODEGEN_MODULEBUILDER_H

#include "clang/AST/ASTConsumer.h"
#include <string>

namespace llvm {
  class GlobalValue;
  class LLVMContext;
  class Module;
  class raw_ostream;
}

namespace clang {
  class DiagnosticsEngine;
  class CoverageSourceInfo;
  class LangOptions;
  class CodeGenOptions;
  class Decl;

  class CodeGenerator : public ASTConsumer {
    virtual void anchor();
  public:
    virtual llvm::Module* GetModule() = 0;
    virtual llvm::Module* ReleaseModule() = 0;
    virtual llvm::Module* StartModule(const std::string& ModuleName,
                                      llvm::LLVMContext& C,
                                      const CodeGenOptions& CGO) = 0;
    virtual const Decl *GetDeclForMangledName(llvm::StringRef MangledName) = 0;
    virtual void print(llvm::raw_ostream& out) = 0;
    virtual void forgetGlobal(llvm::GlobalValue* GV) = 0;
  };

  /// CreateLLVMCodeGen - Create a CodeGenerator instance.
  /// It is the responsibility of the caller to call delete on
  /// the allocated CodeGenerator instance.
  CodeGenerator *CreateLLVMCodeGen(DiagnosticsEngine &Diags,
                                   const std::string &ModuleName,
                                   const CodeGenOptions &CGO,
                                   llvm::LLVMContext& C,
                                   CoverageSourceInfo *CoverageInfo = nullptr);
}

#endif
