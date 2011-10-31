//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Axel Naumann <axel@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_INCREMENTAL_PARSER_H
#define CLING_INCREMENTAL_PARSER_H

#include "ChainedConsumer.h"

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
  class PCHGenerator;
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
    void Initialize(const char* startupPCH);
    clang::CompilerInstance* getCI() const { return m_CI.get(); }
    clang::Parser* getParser() const { return m_Parser.get(); }
    EParseResult CompileLineFromPrompt(llvm::StringRef input);
    EParseResult CompileAsIs(llvm::StringRef input);
    void Parse(llvm::StringRef input, 
               llvm::SmallVector<clang::DeclGroupRef, 4>& DGRs);

    llvm::MemoryBuffer* getCurBuffer() {
      return m_MemoryBuffer.back();
    }
    void enablePrintAST(bool print /*=true*/) {
      m_Consumer->RestorePreviousState(ChainedConsumer::kASTDumper, print);
    }
    void enableDynamicLookup(bool value = true);
    bool isDynamicLookupEnabled() { return m_DynamicLookupEnabled; }
    clang::Decl* getFirstTopLevelDecl() const { return m_FirstTopLevelDecl; }
    clang::Decl* getLastTopLevelDecl() const { return m_LastTopLevelDecl; }
    
    void addConsumer(ChainedConsumer::EConsumerIndex I, clang::ASTConsumer* consumer);
    clang::CodeGenerator* GetCodeGenerator();

    bool usingStartupPCH() const { return m_UsingStartupPCH; }
    void writeStartupPCH();
  private:
    void CreateSLocOffsetGenerator();
    EParseResult Compile(llvm::StringRef input);
    EParseResult Parse(llvm::StringRef input);
    void loadStartupPCH(const char* filename);

    Interpreter* m_Interpreter; // our interpreter context
    llvm::OwningPtr<clang::CompilerInstance> m_CI; // compiler instance.
    llvm::OwningPtr<clang::Parser> m_Parser; // parser (incremental)
    bool m_DynamicLookupEnabled; // enable/disable dynamic scope
    std::vector<llvm::MemoryBuffer*> m_MemoryBuffer; // One buffer for each command line, owner by the source file manager
    clang::FileID m_VirtualFileID; // file ID of the memory buffer
    ChainedConsumer* m_Consumer; // CI owns it
    clang::Decl* m_FirstTopLevelDecl; // first top level decl
    clang::Decl* m_LastTopLevelDecl; // last top level decl after most recent call to parse()
    bool m_UsingStartupPCH; // Whether the interpreter is using a PCH file to accelerate its startup
    std::string m_StartupPCHFilename; // Set when writing the PCH
    llvm::OwningPtr<clang::PCHGenerator> m_StartupPCHGenerator; // Startup PCH generator
  };
}
#endif // CLING_INCREMENTAL_PARSER_H
