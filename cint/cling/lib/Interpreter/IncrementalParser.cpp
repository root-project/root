//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Axel Naumann <axel@cern.ch>
//------------------------------------------------------------------------------

#include "IncrementalParser.h"

#include "ASTDumper.h"
#include "ChainedConsumer.h"
#include "DeclExtractor.h"
#include "DynamicLookup.h"
#include "ValuePrinterSynthesizer.h"
#include "cling/Interpreter/CIFactory.h"
#include "cling/Interpreter/Interpreter.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Basic/FileManager.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Parse/Parser.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Serialization/ASTWriter.h"

#include "llvm/LLVMContext.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_os_ostream.h"

#include <ctime>
#include <iostream>
#include <stdio.h>
#include <sstream>

using namespace clang;

namespace cling {
  IncrementalParser::IncrementalParser(Interpreter* interp, 
                                       int argc, const char* const *argv,
                                       const char* llvmdir):
    m_Interpreter(interp),
    m_DynamicLookupEnabled(false),
    m_Consumer(0),
    m_FirstTopLevelDecl(0),
    m_LastTopLevelDecl(0),
    m_SyntaxOnly(false)
  {
    CompilerInstance* CI 
      = CIFactory::createCI(llvm::MemoryBuffer::getMemBuffer("", "CLING"), 
                            argc, argv, llvmdir);
    assert(CI && "CompilerInstance is (null)!");
    m_CI.reset(CI);
    // Ugly hack to avoid the default of ProgramAction which is SyntaxOnly
    // and we can't override it later. This will be fixed once cling is 
    // supports the latest changes in the clang's driver 
    // FIXME: REIMPLEMENT
    m_SyntaxOnly = (CI->getFrontendOpts().ProgramAction == clang::frontend::EmitHTML);

    CreateSLocOffsetGenerator();

    m_Consumer = dyn_cast<ChainedConsumer>(&CI->getASTConsumer());
    assert(m_Consumer && "Expected ChainedConsumer!");
    // Add consumers to the ChainedConsumer, which owns them
    EvaluateTSynthesizer* ES = new EvaluateTSynthesizer(interp);
    ES->Attach(m_Consumer);
    addConsumer(ChainedConsumer::kEvaluateTSynthesizer, ES);

    DeclExtractor* DE = new DeclExtractor();
    DE->Attach(m_Consumer);
    addConsumer(ChainedConsumer::kDeclExtractor, DE);

    ValuePrinterSynthesizer* VPS = new ValuePrinterSynthesizer(interp);
    VPS->Attach(m_Consumer);
    addConsumer(ChainedConsumer::kValuePrinterSynthesizer, VPS);
    addConsumer(ChainedConsumer::kASTDumper, new ASTDumper());
    if (!m_SyntaxOnly) {
      CodeGenerator* CG = CreateLLVMCodeGen(CI->getDiagnostics(), 
                                            "cling input",
                                            CI->getCodeGenOpts(), 
                                  /*Owned by codegen*/ * new llvm::LLVMContext()
                                            );
      assert(CG && "No CodeGen?!");
      addConsumer(ChainedConsumer::kCodeGenerator, CG);
    }
    m_Consumer->Initialize(CI->getASTContext());
    m_Consumer->InitializeSema(CI->getSema());
    // Initialize the parser.
    m_Parser.reset(new Parser(CI->getPreprocessor(), CI->getSema()));
    CI->getPreprocessor().EnterMainSourceFile();
    m_Parser->Initialize();
  }

  // Each input line is contained in separate memory buffer. The SourceManager
  // assigns sort-of invalid FileID for each buffer, i.e there is no FileEntry
  // for the MemoryBuffer's FileID. That in turn is problem because invalid
  // SourceLocations are given to the diagnostics. Thus the diagnostics cannot
  // order the overloads, for example
  //
  // Our work-around is creating a virtual file, which doesn't exist on the disk
  // with enormous size (no allocation is done). That file has valid FileEntry 
  // and so on... We use it for generating valid SourceLocations with valid
  // offsets so that it doesn't cause any troubles to the diagnostics.
  //
  // +---------------------+
  // | Main memory buffer  |
  // +---------------------+
  // |  Virtual file SLoc  |
  // |    address space    |<-----------------+
  // |         ...         |<------------+    |
  // |         ...         |             |    |
  // |         ...         |<----+       |    |
  // |         ...         |     |       |    |
  // +~~~~~~~~~~~~~~~~~~~~~+     |       |    |
  // |     input_line_1    | ....+.......+..--+
  // +---------------------+     |       |   
  // |     input_line_2    | ....+.....--+
  // +---------------------+     |
  // |          ...        |     |
  // +---------------------+     |
  // |     input_line_N    | ..--+  
  // +---------------------+
  //
  void IncrementalParser::CreateSLocOffsetGenerator() {
    SourceManager& SM = getCI()->getSourceManager();
    FileManager& FM = SM.getFileManager();
    const FileEntry* FE 
      = FM.getVirtualFile("Interactrive/InputLineIncluder", 1U << 15U, time(0));
    m_VirtualFileID = SM.createFileID(FE, SourceLocation(), SrcMgr::C_User);

    assert(!m_VirtualFileID.isInvalid() && "No VirtualFileID created?");
  }

  IncrementalParser::~IncrementalParser() {
     GetCodeGenerator()->ReleaseModule();
  }
  
  void IncrementalParser::Initialize() {

    // Init the consumers    

    CompileAsIs(""); // Consume initialization.
  }

  IncrementalParser::EParseResult 
  IncrementalParser::CompileLineFromPrompt(llvm::StringRef input) {
    assert(input.str()[0] != '#' 
           && "Preprocessed line! Call CompilePreprocessed instead");
    
    bool p, q;
    m_Consumer->RestorePreviousState(ChainedConsumer::kEvaluateTSynthesizer,
                                     isDynamicLookupEnabled());

    p = m_Consumer->EnableConsumer(ChainedConsumer::kDeclExtractor);
    q = m_Consumer->EnableConsumer(ChainedConsumer::kValuePrinterSynthesizer);
    EParseResult Result = Compile(input);
    m_Consumer->RestorePreviousState(ChainedConsumer::kDeclExtractor, p);
    m_Consumer->RestorePreviousState(ChainedConsumer::kValuePrinterSynthesizer, q);

    return Result;

  }

  IncrementalParser::EParseResult 
  IncrementalParser::CompileAsIs(llvm::StringRef input) {
    bool p, q;
    m_Consumer->RestorePreviousState(ChainedConsumer::kEvaluateTSynthesizer,
                                     isDynamicLookupEnabled());

    p = m_Consumer->DisableConsumer(ChainedConsumer::kDeclExtractor);
    q = m_Consumer->DisableConsumer(ChainedConsumer::kValuePrinterSynthesizer);
    EParseResult Result = Compile(input);
    m_Consumer->RestorePreviousState(ChainedConsumer::kDeclExtractor, p);
    m_Consumer->RestorePreviousState(ChainedConsumer::kValuePrinterSynthesizer, q);

    return Result;
  }

  void IncrementalParser::Parse(llvm::StringRef input, 
                                llvm::SmallVector<DeclGroupRef, 4>& DGRs){
    m_Consumer->DisableConsumer(ChainedConsumer::kCodeGenerator);

    Parse(input);
    for (llvm::SmallVector<ChainedConsumer::DGRInfo, 64>::iterator 
           I = m_Consumer->DeclsQueue.begin(), E = m_Consumer->DeclsQueue.end(); 
         I != E; ++I) {
      DGRs.push_back((*I).D);
    }

    m_Consumer->EnableConsumer(ChainedConsumer::kCodeGenerator);
  }

  IncrementalParser::EParseResult 
  IncrementalParser::Compile(llvm::StringRef input) {
    // Just in case when Parse is called, we want to complete the transaction
    // coming from parse and then start new one.
    m_Consumer->HandleTranslationUnit(getCI()->getASTContext());

    // Reset the module builder to clean up global initializers, c'tors, d'tors:
    GetCodeGenerator()->Initialize(getCI()->getASTContext());

    EParseResult Result = Parse(input);

    // Check for errors coming from our custom consumers.
    DiagnosticConsumer& DClient = m_CI->getDiagnosticClient();
    DClient.BeginSourceFile(getCI()->getLangOpts(), &getCI()->getPreprocessor());
    m_Consumer->HandleTranslationUnit(getCI()->getASTContext());

    DClient.EndSourceFile();
    m_CI->getDiagnostics().Reset();

    if (!m_SyntaxOnly) {
      m_Interpreter->runStaticInitializersOnce();
    }

    return Result;
  }

  IncrementalParser::EParseResult 
  IncrementalParser::Parse(llvm::StringRef input) {

    // Add src to the memory buffer, parse it, and add it to
    // the AST. Returns the CompilerInstance (and thus the AST).
    // Diagnostics are reset for each call of parse: they are only covering
    // src.

    Preprocessor& PP = m_CI->getPreprocessor();
    DiagnosticConsumer& DClient = m_CI->getDiagnosticClient();
    DClient.BeginSourceFile(m_CI->getLangOpts(), &PP);
    // Reset the transaction information
    getLastTransaction().Reset();

    if (input.size()) {
      std::ostringstream source_name;
      source_name << "input_line_" << (m_MemoryBuffer.size() + 1);
      m_MemoryBuffer.push_back(llvm::MemoryBuffer::getMemBufferCopy(input, source_name.str()));
      SourceManager& SM = getCI()->getSourceManager();

      // Create SourceLocation, which will allow clang to order the overload
      // candidates for example
      SourceLocation NewLoc = SM.getLocForStartOfFile(m_VirtualFileID);
      NewLoc = NewLoc.getLocWithOffset(m_MemoryBuffer.size() + 1);
      
      // Create FileID for the current buffer
      FileID FID = SM.createFileIDForMemBuffer(m_MemoryBuffer.back(),
                                               /*LoadedID*/0,
                                               /*LoadedOffset*/0, NewLoc);
      
      PP.EnterSourceFile(FID, 0, NewLoc);
      
      Token &tok = const_cast<Token&>(m_Parser->getCurToken());
      tok.setKind(tok::semi);
    }

    Parser::DeclGroupPtrTy ADecl;
    
    bool atEOF = false;
    if (m_Parser->getCurToken().is(tok::eof)) {
      atEOF = true;
    }
    else {
      atEOF = m_Parser->ParseTopLevelDecl(ADecl);
    }

    while (!atEOF) {
      // Not end of file.
      // If we got a null return and something *was* parsed, ignore it.  This
      // is due to a top-level semicolon, an action override, or a parse error
      // skipping something.
      if (ADecl) {
        DeclGroupRef DGR = ADecl.getAsVal<DeclGroupRef>();
        for (DeclGroupRef::iterator i=DGR.begin(); i< DGR.end(); ++i) {
         if (!m_FirstTopLevelDecl) 	 
           m_FirstTopLevelDecl = *i;
         if (!m_LastTransaction.FirstDecl) 	 
           m_LastTransaction.FirstDecl = *i;

          m_LastTopLevelDecl = *i;
          m_LastTransaction.LastDecl = *i;
        } 
        m_Consumer->HandleTopLevelDecl(DGR);
      } // ADecl
      if (m_Parser->getCurToken().is(tok::eof)) {
        atEOF = true;
      }
      else {
        atEOF = m_Parser->ParseTopLevelDecl(ADecl);
      }
    };
    
    // Process any TopLevelDecls generated by #pragma weak.
    for (llvm::SmallVector<Decl*,2>::iterator
           I = getCI()->getSema().WeakTopLevelDecls().begin(),
           E = getCI()->getSema().WeakTopLevelDecls().end(); I != E; ++I) {
      m_Consumer->HandleTopLevelDecl(DeclGroupRef(*I));
    }

    getCI()->getSema().PerformPendingInstantiations();

    DClient.EndSourceFile();

    DiagnosticsEngine& Diag = getCI()->getSema().getDiagnostics();
    if (Diag.hasErrorOccurred())
      return IncrementalParser::kFailed;
    else if (Diag.getNumWarnings())
      return IncrementalParser::kSuccessWithWarnings;

    return IncrementalParser::kSuccess;
  }

  void IncrementalParser::enableDynamicLookup(bool value) {
    m_DynamicLookupEnabled = value;
    Sema& S = m_CI->getSema();
    if (isDynamicLookupEnabled()) {
      assert(!S.ExternalSource && "Already set Sema ExternalSource");
      S.ExternalSource = new DynamicIDHandler(&S);
    }
    else {
      delete S.ExternalSource;
      S.ExternalSource = 0;
    }      
  }

  void IncrementalParser::addConsumer(ChainedConsumer::EConsumerIndex I, ASTConsumer* consumer) {
    if (m_Consumer->Exists(I))
      return;

    m_Consumer->Add(I, consumer);
    if (I == ChainedConsumer::kCodeGenerator)
      m_Consumer->EnableConsumer(I);
  }

  CodeGenerator* IncrementalParser::GetCodeGenerator() const { 
    return 
      (CodeGenerator*)m_Consumer->getConsumer(ChainedConsumer::kCodeGenerator); 
  }

} // namespace cling
