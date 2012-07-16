//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Axel Naumann <axel@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_CHAINED_CONSUMER_H
#define CLING_CHAINED_CONSUMER_H

#include "cling/Interpreter/CompilationOptions.h"

#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallVector.h"

#include "clang/Sema/SemaConsumer.h"

namespace clang {
  class DeclGroupRef;
  class ASTContext;
}

namespace cling {

  class ASTNodeEraser;
  class ChainedMutationListener;
  class ChainedDeserializationListener;
  class Transaction;
  class VerifyingSemaConsumer;

  class ChainedConsumer: public clang::SemaConsumer {

  public:
    enum EConsumerIndex {
      kEvaluateTSynthesizer,
      kDeclExtractor,
      kValuePrinterSynthesizer,
      kASTDumper,
      kCodeGenerator,
      kConsumersCount
    };
    ChainedConsumer();
    virtual ~ChainedConsumer();

    // ASTConsumer
    virtual void Initialize(clang::ASTContext& Context);
    virtual bool HandleTopLevelDecl(clang::DeclGroupRef D);
    virtual void HandleInterestingDecl(clang::DeclGroupRef D);
    virtual void HandleTagDeclDefinition(clang::TagDecl* D);
    virtual void HandleVTable(clang::CXXRecordDecl* RD,
                              bool DefinitionRequired);
    virtual void CompleteTentativeDefinition(clang::VarDecl* D);
    virtual void HandleTranslationUnit(clang::ASTContext& Ctx);

    virtual clang::ASTMutationListener* GetASTMutationListener();
    virtual clang::ASTDeserializationListener* GetASTDeserializationListener();
    virtual void PrintStats();

    // SemaConsumer
    virtual void InitializeSema(clang::Sema& S);
    virtual void ForgetSema();

    bool IsInTransaction() { return m_InTransaction; }

    void Add(EConsumerIndex I, clang::ASTConsumer* C);
    void RecoverFromError();
    clang::ASTConsumer** getConsumers() {
      return Consumers;
    }

    bool Exists(EConsumerIndex I) {
      return Consumers[I] != 0;
    }

    clang::ASTConsumer* getConsumer(EConsumerIndex I) {
      return Consumers[I];
    }

    bool IsConsumerEnabled(EConsumerIndex I);

    void Update(VerifyingSemaConsumer* ESSC);
    void pushCompilationOpts(CompilationOptions CO) {
      COStack.push_back(CO);
    }

    void popCompilationOpts() {
      assert(COStack.size() && "Cannot pop elements back.");
      COStack.pop_back();
    }

    CompilationOptions& getCompilationOpts() {
      return COStack.back();
    }

    const CompilationOptions& getCompilationOpts() const{
      return COStack.back();
    }

    Transaction* getTransaction() { return m_CurTransaction; }
    const Transaction* getTransaction() const { return m_CurTransaction; }
    void setTransaction(Transaction* curT) { m_CurTransaction = curT; }
    void setTransaction(const Transaction* curT) { 
      m_CurTransaction = const_cast<Transaction*>(curT); 
    }

  private:
    clang::ASTConsumer* Consumers[kConsumersCount]; // owns them
    llvm::SmallVector<CompilationOptions, 2> COStack;
    llvm::OwningPtr<ASTNodeEraser> NodeEraser;
    llvm::OwningPtr<ChainedMutationListener> MutationListener;
    llvm::OwningPtr<ChainedDeserializationListener> DeserializationListener;
    bool m_InTransaction;
    clang::ASTContext* m_Context;
    clang::Sema* m_Sema;

    Transaction* m_CurTransaction;

    friend class IncrementalParser;
  };
} // namespace cling

#endif // CLING_CHAINED_CONSUMER_H
