//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Axel Naumann <axel@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_CHAINED_CONSUMER_H
#define CLING_CHAINED_CONSUMER_H

#include "cling/Interpreter/CompilationOptions.h"

#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallVector.h"

#include "clang/AST/DeclGroup.h"
#include "clang/Sema/SemaConsumer.h"

namespace clang {
  class ASTContext;
}

namespace cling {

  class ASTNodeEraser;
  class ChainedMutationListener;
  class ChainedDeserializationListener;
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
    virtual void HandleVTable(clang::CXXRecordDecl* RD, bool DefinitionRequired);
    virtual void CompleteTentativeDefinition(clang::VarDecl* D);
    virtual void HandleTranslationUnit(clang::ASTContext& Ctx);

    virtual clang::ASTMutationListener* GetASTMutationListener();
    virtual clang::ASTDeserializationListener* GetASTDeserializationListener();
    virtual void PrintStats();
    
    // SemaConsumer
    virtual void InitializeSema(clang::Sema& S);
    virtual void ForgetSema();

    // Transaction Support
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

    bool EnableConsumer(EConsumerIndex I) {
      assert(Exists(I) && "Cannot disable. Consumer not set!");
      bool prev;
      CompilationOptions CO = getCompilationOpts();
      switch(I) {
      case kEvaluateTSynthesizer : 
        prev = CO.DynamicScoping; CO.DynamicScoping = 1; break;
      case kDeclExtractor :
        prev = CO.DeclarationExtraction; CO.DeclarationExtraction = 1; break;
      case kValuePrinterSynthesizer : 
        prev = CO.ValuePrinting; CO.ValuePrinting = CompilationOptions::Auto; break;
      case kASTDumper : 
        prev = CO.Debug; CO.Debug = 1; break;
      case kCodeGenerator : 
        prev = CO.CodeGeneration; CO.CodeGeneration = 1; break;
      case kConsumersCount : break;
      }

      return prev;
    }

    bool DisableConsumer(EConsumerIndex I) {
      assert(Exists(I) && "Cannot disable. Consumer not set!");
      bool prev = false;
      CompilationOptions CO = getCompilationOpts();
      switch(I) {
      case kEvaluateTSynthesizer : 
        prev = CO.DynamicScoping; CO.DynamicScoping = 0; break;
      case kDeclExtractor :
        prev = CO.DeclarationExtraction; CO.DeclarationExtraction = 0; break;
      case kValuePrinterSynthesizer : 
        prev = CO.ValuePrinting; CO.ValuePrinting = 0; break;
      case kASTDumper : 
        prev = CO.Debug; CO.Debug = 0; break;
      case kCodeGenerator : 
        prev = CO.CodeGeneration; CO.CodeGeneration = 0; break;
      case kConsumersCount : break;
      }

      return prev;
    }

    void RestorePreviousState(EConsumerIndex I, bool Previous) {
      assert(Exists(I) && "Cannot disable. Consumer not set!");

      CompilationOptions CO = getCompilationOpts();
      switch(I) {
      case kEvaluateTSynthesizer : CO.DynamicScoping = Previous; break;
      case kDeclExtractor : CO.DeclarationExtraction = Previous; break;
      case kValuePrinterSynthesizer : CO.ValuePrinting = Previous; break;
      case kASTDumper : CO.Debug = Previous; break;
      case kCodeGenerator : CO.CodeGeneration = Previous; break;
      case kConsumersCount : break;
      }
    }

    bool IsConsumerEnabled(EConsumerIndex I) {
      if (!Exists(I))
        return false;

      const CompilationOptions CO = getCompilationOpts();
      switch(I) {
      case kEvaluateTSynthesizer : return CO.DynamicScoping;
      case kDeclExtractor : return CO.DeclarationExtraction;
      case kValuePrinterSynthesizer : return CO.ValuePrinting == CompilationOptions::Auto;
      case kASTDumper : return CO.Debug;
      case kCodeGenerator : return CO.CodeGeneration;
      case kConsumersCount : return false;
      }

      return false;
    }

    bool IsQueueing() { return m_Queueing; }

    void DumpQueue();
    void Update(VerifyingSemaConsumer* ESSC);
    void pushCompilationOpts(CompilationOptions CO) {
      COStack.push_back(CO);
    }

    void popCompilationOpts() {
      assert(COStack.size() && "Cannot pop elements back.");
      COStack.pop_back();
    }

    const CompilationOptions& getCompilationOpts() {
      return COStack.back();
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
    enum HandlerIndex {
      kTopLevelDecl,
      kInterestingDecl,
      kTagDeclDefinition,
      kVTable,      
      kCompleteTentativeDefinition
    };
    struct DGRInfo {
      clang::DeclGroupRef D;
      HandlerIndex I;
      DGRInfo(clang::DeclGroupRef d, HandlerIndex i) : D(d), I(i){}
    };
    llvm::SmallVector<DGRInfo, 64> DeclsQueue;
    bool m_Queueing;

    friend class IncrementalParser;
  };
} // namespace cling

#endif
