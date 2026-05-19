//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_INCREMENTAL_ACTION_H
#define CLING_INCREMENTAL_ACTION_H

#include "cling/Interpreter/InvocationOptions.h"

#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/MultiplexConsumer.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <vector>

namespace llvm {
  class Module;
}

namespace clang {
  class ASTContext;
  class ASTConsumer;
  class CodeGenerator;
  class CompilerInstance;
} // namespace clang

namespace cling {
  class DeclCollector;
  class Interpreter;
  class PCHGeneratorWrapper;
  class IncrementalAction : public clang::WrapperFrontendAction {
  private:
    bool IsTerminating = false;
    clang::CompilerInstance& CI;
    CompilerOptions COpts;
    cling::DeclCollector *DeclCollectorConsumer = nullptr;
    PCHGeneratorWrapper *PCHGenWrapper = nullptr;
    /// When CodeGen is created the first llvm::Module gets cached in many
    /// places and we must keep it alive.
    std::unique_ptr<llvm::Module> CachedInCodeGenModule;

  public:
    IncrementalAction(clang::CompilerInstance& CI, llvm::LLVMContext& LLVMCtx,
                      CompilerOptions COpts, llvm::Error& Err);

    clang::FrontendAction* getWrapped() const { return WrappedAction.get(); }

    clang::TranslationUnitKind getTranslationUnitKind() override {
      return clang::TU_Incremental;
    }

    DeclCollector* getDeclCollectorConsumer() const {
      assert(DeclCollectorConsumer);
      return DeclCollectorConsumer;
    }

    std::unique_ptr<clang::ASTConsumer>
    CreateMultiplexConsumer(clang::CompilerInstance& CI,
                            llvm::StringRef InFile);

    std::unique_ptr<clang::ASTConsumer>
    CreateASTConsumer(clang::CompilerInstance& CI,
                      llvm::StringRef InFile) override;

    void ExecuteAction() override;

    bool BeginSourceFileAction(clang::CompilerInstance& CI) override;

    void EndSourceFile() override {
      if (IsTerminating && getWrapped())
        clang::WrapperFrontendAction::EndSourceFile();
    }

    void FinalizeAction() {
      assert(!IsTerminating && "Already finalized!");
      IsTerminating = true;
      EndSourceFile();
    }

    void GenPCH(clang::ASTContext &Ctx);

    std::unique_ptr<llvm::Module> GenModule();

    clang::CodeGenerator* getCodeGen();

    void CacheCodeGenModule();

    llvm::Module* getCachedCodeGenModule() const;
  };
} // end namespace cling
#endif // CLING_INCREMENTAL_ACTION_H