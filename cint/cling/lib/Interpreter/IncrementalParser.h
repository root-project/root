//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Axel Naumann <axel@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_INCREMENTAL_PARSER_H
#define CLING_INCREMENTAL_PARSER_H

#include "ChainedConsumer.h"
#include "Transaction.h"
#include "cling/Interpreter/CompilationOptions.h"

#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Basic/SourceLocation.h"

#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringRef.h"

#include <vector>

namespace llvm {
  struct GenericValue;
  class MemoryBuffer;
}

namespace clang {
  class ASTConsumer;
  class CodeGenerator;
  class CompilerInstance;
  class Decl;
  class FileID;
  class FunctionDecl;
  class Parser;
  class Sema;
  class SourceLocation;
}


namespace cling {
  class ChainedConsumer;
  class CIFactory;
  class ExecutionContext;
  class Interpreter;

  class IncrementalParser {
  public:
    enum EParseResult {
      kSuccess,
      kSuccessWithWarnings,
      kFailed
    };
    IncrementalParser(Interpreter* interp, int argc, const char* const *argv,
                      const char* llvmdir);
    ~IncrementalParser();
    void Initialize();
    clang::CompilerInstance* getCI() const { return m_CI.get(); }
    clang::Parser* getParser() const { return m_Parser.get(); }

    /// \{
    /// \name Transaction Support

    ///\brief Starts a transaction.
    ///
    void beginTransaction();

    ///\brief Finishes a transaction.
    ///
    void endTransaction() const;

    ///\brief Commits a given transaction. I.e pipes it through the consumer
    /// chain including codegen.
    ///
    ///\param[in] T - The transaction to be committed.
    ///
    void commitTransaction(const Transaction* T) const;

    ///\brief Commits the current transaction if it was compete. I.e pipes it 
    /// through the consumer chain, including codegen.
    ///
    void commitCurrentTransaction() const;

    /// \}

    ///\brief Compiles the given input with the given compilation options.
    ///
    EParseResult Compile(llvm::StringRef input, const CompilationOptions& Opts);

    void Parse(llvm::StringRef input,
               llvm::SmallVector<clang::DeclGroupRef, 4>& DGRs);

    llvm::MemoryBuffer* getCurBuffer() const {
      return m_MemoryBuffer.back();
    }
    void enablePrintAST(bool print /*=true*/) {
      m_Consumer->getCompilationOpts().Debug = print;
    }
    void enableDynamicLookup(bool value = true);
    bool isDynamicLookupEnabled() const { return m_DynamicLookupEnabled; }
    bool isSyntaxOnly() const { return m_SyntaxOnly; }
    const Transaction* getLastTransaction() const { 
      return m_Transactions.back(); 
    }

    Transaction* getLastTransaction() { 
      return m_Transactions.back(); 
    }

    clang::CodeGenerator* GetCodeGenerator() const;

  private:
    void CreateSLocOffsetGenerator();
    EParseResult Compile(llvm::StringRef input);
    EParseResult Parse(llvm::StringRef input);

    // our interpreter context
    Interpreter* m_Interpreter;

   // compiler instance.
    llvm::OwningPtr<clang::CompilerInstance> m_CI;

    // parser (incremental)
    llvm::OwningPtr<clang::Parser> m_Parser;

    // enable/disable dynamic scope
    bool m_DynamicLookupEnabled;

    // One buffer for each command line, owner by the source file manager
    std::vector<llvm::MemoryBuffer*> m_MemoryBuffer;

    // file ID of the memory buffer
    clang::FileID m_VirtualFileID;

    // CI owns it
    ChainedConsumer* m_Consumer;

    ///\brief Holds information for the all transactions.
    ///
    llvm::SmallVector<Transaction*, 64> m_Transactions;

    // whether to run codegen; cannot be flipped during lifetime of *this
    bool m_SyntaxOnly;
  };
}
#endif // CLING_INCREMENTAL_PARSER_H
