//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "IncrementalParser.h"

#include "ASTTransformer.h"
#include "AutoSynthesizer.h"
#include "CheckEmptyTransactionTransformer.h"
#include "ClingPragmas.h"
#include "DeclCollector.h"
#include "DeclExtractor.h"
#include "DefinitionShadower.h"
#include "DeviceKernelInliner.h"
#include "DynamicLookup.h"
#include "IncrementalAction.h"
#include "NullDerefProtectionTransformer.h"
#include "TransactionPool.h"
#include "ValueExtractionSynthesizer.h"
#include "ValuePrinterSynthesizer.h"
#include "cling/Interpreter/CIFactory.h"
#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/InterpreterCallbacks.h"
#include "cling/Interpreter/Transaction.h"
#include "cling/Utils/Diagnostics.h"
#include "cling/Utils/Output.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/FrontendTool/Utils.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Parse/Parser.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/Serialization/ASTWriter.h"
#include "clang/Serialization/ASTReader.h"
#include "llvm/Support/Path.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#include <stdio.h>

using namespace clang;

namespace {

  ///\brief Check the compile-time C++ ABI version vs the run-time ABI version,
  /// a mismatch could cause havoc. Reports if ABI versions differ.
  static bool CheckABICompatibility(cling::Interpreter& Interp) {
#if defined(__GLIBCXX__)
    #define CLING_CXXABI_VERS       std::to_string(__GLIBCXX__)
    const char* CLING_CXXABI_NAME = "__GLIBCXX__";
    static constexpr bool CLING_CXXABI_BACKWARDCOMP = true;
#elif defined(_LIBCPP_VERSION)
    #define CLING_CXXABI_VERS       std::to_string(_LIBCPP_ABI_VERSION)
    const char* CLING_CXXABI_NAME = "_LIBCPP_ABI_VERSION";
    static constexpr bool CLING_CXXABI_BACKWARDCOMP = false;
#elif defined(_CRT_MSVCP_CURRENT)
    #define CLING_CXXABI_VERS        _CRT_MSVCP_CURRENT
    const char* CLING_CXXABI_NAME = "_CRT_MSVCP_CURRENT";
    static constexpr bool CLING_CXXABI_BACKWARDCOMP = false;
#else
    #error "Unknown platform for ABI check";
#endif

    const std::string CurABI = Interp.getMacroValue(CLING_CXXABI_NAME);
    if (CurABI == CLING_CXXABI_VERS)
      return true;
    if (CurABI.empty()) {
    cling::errs() <<
      "Warning in cling::IncrementalParser::CheckABICompatibility():\n"
      "  Failed to extract C++ standard library version.\n";
    }

    if (CLING_CXXABI_BACKWARDCOMP && CurABI < CLING_CXXABI_VERS) {
       // Backward compatible ABIs allow us to interpret old headers
       // against a newer stdlib.so.
       return true;
    }

    cling::errs() <<
      "Warning in cling::IncrementalParser::CheckABICompatibility():\n"
      "  Possible C++ standard library mismatch, compiled with "
      << CLING_CXXABI_NAME << " '" << CLING_CXXABI_VERS << "'\n"
      "  Extraction of runtime standard library version was: '"
      << CurABI << "'\n";

    return false;
  }

  /// \brief Overrides the current DiagnosticConsumer to supress many warnings
  /// issued as a result of incremental compilation (see `HandleDiagnostic()`).
  ///
  /// Diagnostics passing the filter are, by default, forwarded to the previous
  /// DiagnosticConsumer instance.  A different consumer may be specified via
  /// `setTargetConsumer()`.
  /// In that case, given that internal state might be updated as part of
  /// `{Begin,End}SourceFile` (e.g. in TextDiagnosticPrinter), calls to such
  /// functions will be forwarded to both, the user-specified and the original
  /// consumer; however `HandleDiagnostic()` calls shall only be seen by the
  /// former.
  ///
  /// On destruction, the original (i.e. overridden consumer) is restored.
  ///
  class FilteringDiagConsumer : public cling::utils::DiagnosticsOverride {
    std::stack<bool> m_IgnorePromptDiags;
    llvm::PointerIntPair<DiagnosticConsumer*, 1, bool /*Own*/> m_Target{};

    void SyncDiagCountWithTarget() {
      NumWarnings = m_PrevClient.getNumWarnings();
      NumErrors = m_PrevClient.getNumErrors();
    }

    void BeginSourceFile(const LangOptions &LangOpts,
                         const Preprocessor *PP=nullptr) override {
      m_PrevClient.BeginSourceFile(LangOpts, PP);
      if (auto C = m_Target.getPointer())
        C->BeginSourceFile(LangOpts, PP);
    }

    void EndSourceFile() override {
      m_PrevClient.EndSourceFile();
      if (auto C = m_Target.getPointer())
        C->EndSourceFile();
      SyncDiagCountWithTarget();
    }

    void finish() override {
      m_PrevClient.finish();
      if (auto C = m_Target.getPointer())
        C->finish();
      SyncDiagCountWithTarget();
    }

    void clear() override {
      m_PrevClient.clear();
      if (auto C = m_Target.getPointer())
        C->clear();
      SyncDiagCountWithTarget();
    }

    bool IncludeInDiagnosticCounts() const override {
      return m_PrevClient.IncludeInDiagnosticCounts();
    }

    void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                          const Diagnostic &Info) override {
      if (Info.getID() == diag::warn_falloff_nonvoid_function) {
        DiagLevel = DiagnosticsEngine::Error;
      }
      if (Ignoring()) {
        if (Info.getID() == diag::warn_unused_expr
            || Info.getID() == diag::warn_unused_result
            || Info.getID() == diag::warn_unused_call
            || Info.getID() == diag::warn_unused_comparison)
          return; // ignore!
        if (Info.getID() == diag::ext_return_has_expr) {
          // An error that we need to suppress.
          auto Diags = const_cast<DiagnosticsEngine*>(Info.getDiags());
          assert(Diags->hasErrorOccurred() && "Expected ErrorOccurred");
          if (m_PrevClient.getNumErrors() == 0) { // first error
            Diags->Reset(true /*soft - only counts, not mappings*/);
          } // else we had other errors, too.
          return; // ignore!
        }
      }

      // In principle, for simplicity, we preserve the old behavior of
      // delivering diagnostics to just one consumer (that is why we don't emit
      // to both), but we allow the "sink" to be changed.
      // Note, however, that consumers might update their internal state in
      // calls to, e.g. `BeginSourceFile()` or `EndSourceFile()` (actually,
      // `TextDiagnosticPrinter` is an example of this), so in order to be able
      // to restore the original consumer, we need to keep forwarding these
      // calls also to `m_PrevClient` (see above).
      if (auto C = m_Target.getPointer())
        C->HandleDiagnostic(DiagLevel, Info);
      else
        m_PrevClient.HandleDiagnostic(DiagLevel, Info);
      SyncDiagCountWithTarget();
    }

    bool Ignoring() const {
      return !m_IgnorePromptDiags.empty() && m_IgnorePromptDiags.top();
    }

  public:
    FilteringDiagConsumer(DiagnosticsEngine& Diags, bool Own)
      : DiagnosticsOverride(Diags, Own) {}
    ~FilteringDiagConsumer() override { setTargetConsumer(nullptr); }

    /// \brief Sets the DiagnosticConsumer that sees `HandleDiagnostic()` calls.
    /// \param[in] Consumer - The target DiagnosticConsumer, or `nullptr` to
    ///    revert to original client.
    /// \param[in] Own - Whether we own the pointee
    ///
    void setTargetConsumer(DiagnosticConsumer* Consumer, bool Own = false) {
      if (m_Target.getInt())
        if (auto C = m_Target.getPointer())
          delete C;

      m_Target.setPointer(Consumer);
      m_Target.setInt(Own);
    }

    DiagnosticConsumer* getTargetConsumer() const
    { return m_Target.getPointer(); }

    struct RAAI {
      FilteringDiagConsumer& m_Client;
      RAAI(DiagnosticConsumer& F, bool Ignore) :
       m_Client(static_cast<FilteringDiagConsumer&>(F)) {
        m_Client.m_IgnorePromptDiags.push(Ignore);
      }
      ~RAAI() { m_Client.m_IgnorePromptDiags.pop(); }
    };
  };
} // unnamed namespace

namespace cling {

  IncrementalAction::IncrementalAction(CompilerInstance& CI,
                                       llvm::LLVMContext& LLVMCtx,
                                      //  Interpreter& m_Interp,
                                       CompilerOptions COpts, llvm::Error& Err)
      : WrapperFrontendAction([&]() {
          llvm::ErrorAsOutParameter EAO(&Err);
          std::unique_ptr<FrontendAction> Act;
          switch (CI.getFrontendOpts().ProgramAction) {
            default:
              Err = llvm::createStringError(
                  std::errc::state_not_recoverable,
                  "Driver initialization failed. "
                  "Incremental mode for action %d is not supported",
                  CI.getFrontendOpts().ProgramAction);
              return Act;
            case frontend::ASTDump:
            case frontend::ASTPrint:
            case frontend::ParseSyntaxOnly:
              Act = CreateFrontendAction(CI);
              break;
            case frontend::PluginAction:
            case frontend::EmitAssembly:
            case frontend::EmitBC:
            case frontend::EmitObj:
            case frontend::PrintPreprocessedInput:
            case frontend::EmitLLVMOnly:
              Act.reset(new EmitLLVMOnlyAction(&LLVMCtx));
              break;
          }
          return Act;
        }()),
        CI(CI), // IncrParser(IncrParser),
        // m_Interpreter(m_Interp), 
        COpts(COpts) {}

  class PCHGeneratorWrapper : public MultiplexConsumer {
  public:
    using MultiplexConsumer::MultiplexConsumer;

    void HandleTranslationUnit(ASTContext&) override {
      // Delay until Finalize().
    }

    void Finalize(ASTContext& Ctx) {
      MultiplexConsumer::HandleTranslationUnit(Ctx);
    }
  };

  std::unique_ptr<ASTConsumer>
  IncrementalAction::CreateMultiplexConsumer(CompilerInstance& CI,
                                             StringRef InFile) {
    std::vector<std::unique_ptr<ASTConsumer>> Consumers;
    // With C++ modules, we now attach the consumers that will handle the
    // generation of the PCM file itself in case we want to generate
    // a C++ module with the current interpreter instance.
    if (COpts.CxxModules && !COpts.ModuleName.empty()) {
      // Code below from the (private) code in the GenerateModuleAction class.
      llvm::SmallVector<char, 256> Output;
      llvm::sys::path::append(Output, COpts.CachePath,
                              COpts.ModuleName + ".pcm");
      StringRef ModuleOutputFile = StringRef(Output.data(), Output.size());

      std::unique_ptr<raw_pwrite_stream> OS =
          CI.createOutputFile(ModuleOutputFile, /*Binary=*/true,
                              /*RemoveFileOnSignal=*/false,
                              /*useTemporary=*/true,
                              /*CreateMissingDirectories=*/true);
      assert(OS);

      std::string Sysroot;

      auto PCHBuff = std::make_shared<PCHBuffer>();

      Consumers.push_back(std::make_unique<PCHGenerator>(
          CI.getPreprocessor(), CI.getModuleCache(), ModuleOutputFile, Sysroot,
          PCHBuff, CI.getFrontendOpts().ModuleFileExtensions,
          /*AllowASTWithErrors=*/false,
          /*IncludeTimestamps=*/
          +CI.getFrontendOpts().BuildingImplicitModule));
      Consumers.push_back(
          CI.getPCHContainerWriter().CreatePCHContainerGenerator(
              CI, "", ModuleOutputFile.str(), std::move(OS), PCHBuff));
      std::unique_ptr<PCHGeneratorWrapper> PCHGenW =
          std::make_unique<PCHGeneratorWrapper>(std::move(Consumers));
      PCHGenWrapper = PCHGenW.get();
      return PCHGenW;
    }

    return nullptr;
  }

  std::unique_ptr<ASTConsumer>
  IncrementalAction::CreateASTConsumer(CompilerInstance& CI,
                                       StringRef InFile) {
    std::unique_ptr<ASTConsumer> C =
        WrapperFrontendAction::CreateASTConsumer(CI, InFile);
    auto DC = std::make_unique<cling::DeclCollector>();
    DeclCollectorConsumer = DC.get();
    DC->Setup(std::move(C), CI.getPreprocessor());
    std::unique_ptr<ASTConsumer> MC = CreateMultiplexConsumer(CI, InFile);
    if (MC) {
      std::vector<std::unique_ptr<ASTConsumer>> Cs;
      Cs.push_back(std::move(DC));
      Cs.push_back(std::move(MC));
      return std::make_unique<MultiplexConsumer>(std::move(Cs));
    }
    // DC->Setup(std::move(C), CI.getPreprocessor());
    return DC;
  }

  void IncrementalAction::ExecuteAction() {
    CompilerInstance& CI = getCompilerInstance();
    if (!CI.hasPreprocessor())
      return;

    // FIXME: Move the truncation aspect of this into Sema, we delayed this
    // till here so the source manager would be initialized.
    if (hasCodeCompletionSupport() &&
        !CI.getFrontendOpts().CodeCompletionAt.FileName.empty())
      CI.createCodeCompletionConsumer();

    // Use a code completion consumer?
    CodeCompleteConsumer* CompletionConsumer = nullptr;
    if (CI.hasCodeCompletionConsumer())
      CompletionConsumer = &CI.getCodeCompletionConsumer();

    if (!CI.hasSema())
      CI.createSema(getTranslationUnitKind(), CompletionConsumer);
  }

  bool IncrementalAction::BeginSourceFileAction(CompilerInstance& CI) {
    if (COpts.CxxModules)
      CIFactory::collectModule(CI);

    if (COpts.CxxModules && !COpts.ModuleName.empty()) {
      // Set the current module name for clang. With that clang doesn't start
      // to build the current module on demand when we include a header
      // from the current module.
      CI.getLangOpts().CurrentModule = COpts.ModuleName;
      CI.getLangOpts().setCompilingModule(LangOptions::CMK_ModuleMap);

      SourceManager& SM = CI.getSourceManager();
      // Push the current module to the build stack so that clang knows when
      // we have a cyclic dependency.

      SM.pushModuleBuildStack(COpts.ModuleName,
                              FullSourceLoc(SourceLocation(), SM));
    }

    return WrapperFrontendAction::BeginSourceFileAction(CI);
  }

  std::unique_ptr<llvm::Module> IncrementalAction::GenModule() {
    static unsigned ID = 0;
    if (CodeGenerator* CG = getCodeGen()) {
      // Clang's CodeGen is designed to work with a single llvm::Module. In
      // many cases for convenience various CodeGen parts have a reference to
      // the llvm::Module (TheModule or Module) which does not change when a
      // new module is pushed. However, the execution engine wants to take
      // ownership of the module which does not map well to CodeGen's design.
      // To work this around we created an empty module to make CodeGen happy.
      // We should make sure it always stays empty.
      assert(((!CachedInCodeGenModule ||
               !CI.getPreprocessorOpts().Includes.empty() ||
               !CI.getPreprocessorOpts().ImplicitPCHInclude.empty()) ||
              (CachedInCodeGenModule->empty() &&
               CachedInCodeGenModule->global_empty() &&
               CachedInCodeGenModule->alias_empty() &&
               CachedInCodeGenModule->ifunc_empty())) &&
             "CodeGen wrote to a readonly module");
      std::unique_ptr<llvm::Module> M(CG->ReleaseModule());
      CG->StartModule("incr_module_" + std::to_string(ID++), M->getContext());
      return M;
    }
    return nullptr;
  }

  void IncrementalAction::GenPCH(ASTContext &Ctx) {
    static bool PCHGenerated = false;
    assert(!IsTerminating && "Already finalized!");
    assert(!PCHGenerated && "Already generated!");
    if (PCHGenWrapper && !PCHGenerated)
      PCHGenWrapper->Finalize(Ctx);
    PCHGenerated = true;
  }

  CodeGenerator* IncrementalAction::getCodeGen() {
    FrontendAction* WrappedAct = getWrapped();
    return static_cast<CodeGenAction*>(WrappedAct)->getCodeGenerator();
  }

  void IncrementalAction::CacheCodeGenModule() { CachedInCodeGenModule = GenModule(); }

  llvm::Module* IncrementalAction::getCachedCodeGenModule() const {
    return CachedInCodeGenModule.get();
  }

  IncrementalParser::IncrementalParser(Interpreter* interp, CompilerInstance* CI,
                      IncrementalAction* Act)
      : m_Interpreter(interp), m_CI(CI), m_Act(Act) {
    m_Consumer = m_Act->getDeclCollectorConsumer();
    if (!m_Consumer) {
      cling::errs() << "No AST consumer available.\n";
      return;
    }

    if (m_CI->getFrontendOpts().ProgramAction != frontend::ParseSyntaxOnly) {
      if (m_Act->getCodeGen())
        m_CodeGen = m_Act->getCodeGen();

      assert(m_CodeGen);
    }

    // Is the CompilerInstance being used to generate output only?
    if (m_Interpreter->getOptions().CompilerOpts.HasOutput)
      return;

    DiagnosticsEngine& Diag = m_CI->getDiagnostics();
    m_DiagConsumer.reset(new FilteringDiagConsumer(Diag, false));

    initializeVirtualFile();
  }

  bool
  IncrementalParser::Initialize(llvm::SmallVectorImpl<ParseResultTransaction>&
                                result, bool isChildInterpreter) {
    m_TransactionPool.reset(new TransactionPool);

    if (m_CI->getPreprocessor().TUKind != TU_Incremental)
      return true; // This a one-time, non-incremental action.

    Preprocessor& PP = m_CI->getPreprocessor();
    CompilationOptions CO = m_Interpreter->makeDefaultCompilationOpts();
    Transaction* CurT = beginTransaction(CO);

    addClingPragmas(*m_Interpreter);

    // Must happen after attaching the PCH, else PCH elements will end up
    // being lexed.
    PP.EnterMainSourceFile();

    Sema* TheSema = &m_CI->getSema();
    m_Parser.reset(new Parser(PP, *TheSema, false /*skipFuncBodies*/));

    // Initialize the parser after PP has entered the main source file.
    m_Parser->Initialize();

    ExternalASTSource *External = TheSema->getASTContext().getExternalSource();
    if (External)
      External->StartTranslationUnit(m_Consumer);

    // Start parsing the "main file" to warm up lexing (enter caching lex mode
    // for ParseInternal()'s call EnterSourceFile() to make sense.
    while (!m_Parser->ParseTopLevelDecl()) {}

    // If I belong to the parent Interpreter, am using C++, and -noruntime
    // wasn't given on command line, then #include <new> and check ABI
    if (!isChildInterpreter && m_CI->getLangOpts().CPlusPlus &&
        !m_Interpreter->getOptions().NoRuntime) {
      // <new> is needed by the ValuePrinter so it's a good thing to include it.
      // We need to include it to determine the version number of the standard
      // library implementation.
      ParseInternal("#include <new>");
      // That's really C++ ABI compatibility. C has other problems ;-)
      CheckABICompatibility(*m_Interpreter);
    }

    // DO NOT commit the transactions here: static initialization in these
    // transactions requires gCling through local_cxa_atexit(), but that has not
    // been defined yet!
    ParseResultTransaction PRT = endTransaction(CurT);
    result.push_back(PRT);
    return true;
  }

  bool IncrementalParser::isValid(bool initialized) const {
    return m_CI && m_CI->hasFileManager() && m_Consumer
           && !m_VirtualFileID.isInvalid()
           && (!initialized || (m_TransactionPool && m_Parser));
  }

  namespace {
    template <class T>
    struct Reversed {
      const T &m_orig;
      auto begin() -> decltype(m_orig.rbegin()) { return m_orig.rbegin(); }
      auto end() -> decltype (m_orig.rend()) { return m_orig.rend(); }
    };
    template <class T>
    Reversed<T> reverse(const T& orig) { return {orig}; }
  }

  const Transaction* IncrementalParser::getLastWrapperTransaction() const {
    if (auto *T = getCurrentTransaction())
      if (T->getWrapperFD())
        return T;

    for (auto T: reverse(m_Transactions))
      if (T->getWrapperFD())
        return T;
    return nullptr;
  }

  const Transaction* IncrementalParser::getCurrentTransaction() const {
    return m_Consumer->getTransaction();
  }

  void IncrementalParser::setDiagnosticConsumer(DiagnosticConsumer* Consumer,
						bool Own) {
    static_cast<
      FilteringDiagConsumer&>(*m_DiagConsumer).setTargetConsumer(Consumer, Own);
  }

  DiagnosticConsumer* IncrementalParser::getDiagnosticConsumer() const {
    return static_cast<
      FilteringDiagConsumer&>(*m_DiagConsumer).getTargetConsumer();
  }

  SourceLocation IncrementalParser::getNextAvailableUniqueSourceLoc() {
    const SourceManager& SM = getCI()->getSourceManager();
    SourceLocation Result = SM.getLocForStartOfFile(m_VirtualFileID);
    return Result.getLocWithOffset(m_VirtualFileLocOffset++);
  }

  IncrementalParser::~IncrementalParser() {
    Transaction* T = const_cast<Transaction*>(getFirstTransaction());
    while (T) {
      assert((T->getState() == Transaction::kCommitted
              || T->getState() == Transaction::kRolledBackWithErrors
              || T->getState() == Transaction::kNumStates // reset from the pool
              || T->getState() == Transaction::kRolledBack)
             && "Not committed?");
      const Transaction* nextT = T->getNext();
      m_TransactionPool->releaseTransaction(T, false);
      T = const_cast<Transaction*>(nextT);
    }
  }

  void IncrementalParser::addTransaction(Transaction* T) {
    if (!T->isNestedTransaction() && T != getLastTransaction()) {
      if (getLastTransaction())
        m_Transactions.back()->setNext(T);
      m_Transactions.push_back(T);
    }
  }


  Transaction* IncrementalParser::beginTransaction(const CompilationOptions&
                                                   Opts) {
    Transaction* OldCurT = m_Consumer->getTransaction();
    Transaction* NewCurT = m_TransactionPool->takeTransaction(m_CI->getSema());
    NewCurT->setCompilationOpts(Opts);
    // If we are in the middle of transaction and we see another begin
    // transaction - it must be nested transaction.
    if (OldCurT && OldCurT != NewCurT
        && (OldCurT->getState() == Transaction::kCollecting
            || OldCurT->getState() == Transaction::kCompleted)) {
      OldCurT->addNestedTransaction(NewCurT); // takes the ownership
    }

    m_Consumer->setTransaction(NewCurT);
    return NewCurT;
  }

  IncrementalParser::ParseResultTransaction
  IncrementalParser::endTransaction(Transaction* T) {
    assert(T && "Null transaction!?");
    assert(T->getState() == Transaction::kCollecting);

#ifndef NDEBUG
    if (T->hasNestedTransactions()) {
      for(Transaction::const_nested_iterator I = T->nested_begin(),
            E = T->nested_end(); I != E; ++I)
        assert((*I)->isCompleted() && "Nested transaction not completed!?");
    }
#endif

    T->setState(Transaction::kCompleted);

    DiagnosticsEngine& Diag = getCI()->getSema().getDiagnostics();

    //TODO: Make the enum orable.
    EParseResult ParseResult = kSuccess;

    assert((Diag.hasFatalErrorOccurred() ? Diag.hasErrorOccurred() : true)
            && "Diag.hasFatalErrorOccurred without Diag.hasErrorOccurred !");

    if (Diag.hasErrorOccurred() || T->getIssuedDiags() == Transaction::kErrors) {
      T->setIssuedDiags(Transaction::kErrors);
      ParseResult = kFailed;
    } else if (Diag.getNumWarnings() > 0) {
      T->setIssuedDiags(Transaction::kWarnings);
      ParseResult = kSuccessWithWarnings;
    }

    // Empty transaction, send it back to the pool.
    if (T->empty()) {
      assert((!m_Consumer->getTransaction()
              || (m_Consumer->getTransaction() == T))
             && "Cannot release different T");
      // If a nested transaction the active one should be its parent
      // from now on. FIXME: Merge conditional with commitTransaction
      if (T->isNestedTransaction())
        m_Consumer->setTransaction(T->getParent());
      else
        m_Consumer->setTransaction((Transaction*)0);

      m_TransactionPool->releaseTransaction(T);
      return ParseResultTransaction(nullptr, ParseResult);
    }

    addTransaction(T);
    return ParseResultTransaction(T, ParseResult);
  }

  std::string IncrementalParser::makeModuleName() {
    return std::string("cling-module-") + std::to_string(m_ModuleNo++);
  }

  llvm::Module* IncrementalParser::StartModule() {
    return getCodeGenerator()->StartModule(makeModuleName(),
                                           *m_Interpreter->getLLVMContext(),
                                           getCI()->getCodeGenOpts());
  }

  void IncrementalParser::commitTransaction(ParseResultTransaction& PRT,
                                            bool ClearDiagClient) {
    Transaction* T = PRT.getPointer();
    if (!T) {
      if (PRT.getInt() != kSuccess) {
        // Nothing has been emitted to Codegen, reset the Diags.
        DiagnosticsEngine& Diags = getCI()->getSema().getDiagnostics();
        Diags.Reset(/*soft=*/true);
        if (ClearDiagClient)
          Diags.getClient()->clear();
      }
      return;
    }

    assert(T->isCompleted() && "Transaction not ended!?");
    assert(T->getState() != Transaction::kCommitted
           && "Committing an already committed transaction.");
    assert((T->getIssuedDiags() == Transaction::kErrors || !T->empty())
           && "Valid Transactions must not be empty;");

    // If committing a nested transaction the active one should be its parent
    // from now on.
    if (T->isNestedTransaction())
      m_Consumer->setTransaction(T->getParent());

    // Check for errors...
    if (T->getIssuedDiags() == Transaction::kErrors) {
      // Make module visible to TransactionUnloader.
      bool MustStartNewModule = false;
      if (!T->isNestedTransaction() && hasCodeGenerator()) {
        MustStartNewModule = true;
        std::unique_ptr<llvm::Module> M(getCodeGenerator()->ReleaseModule());

        if (M) {
          T->setModule(std::move(M));
        }
      }
      // Module has been released from Codegen, reset the Diags now.
      DiagnosticsEngine& Diags = getCI()->getSema().getDiagnostics();
      Diags.Reset(/*soft=*/true);
      if (ClearDiagClient)
        Diags.getClient()->clear();

      PRT.setPointer(nullptr);
      PRT.setInt(kFailed);
      m_Interpreter->unload(*T);

      // Create a new module if necessary.
      if (MustStartNewModule)
        StartModule();

      return;
    }

    if (T->hasNestedTransactions()) {
      Transaction* TopmostParent = T->getTopmostParent();
      EParseResult PR = kSuccess;
      if (TopmostParent->getIssuedDiags() == Transaction::kErrors)
        PR = kFailed;
      else if (TopmostParent->getIssuedDiags() == Transaction::kWarnings)
        PR = kSuccessWithWarnings;

      for (Transaction::const_nested_iterator I = T->nested_begin(),
            E = T->nested_end(); I != E; ++I)
        if ((*I)->getState() != Transaction::kCommitted) {
          ParseResultTransaction PRT(*I, PR);
          commitTransaction(PRT);
        }
    }

    // If there was an error coming from the transformers.
    if (T->getIssuedDiags() == Transaction::kErrors) {
      m_Interpreter->unload(*T);
      return;
    }

    // Here we expect a template instantiation. We need to open the transaction
    // that we are currently work with.
    {
      Transaction* prevConsumerT = m_Consumer->getTransaction();
      m_Consumer->setTransaction(T);
      Transaction* nestedT = beginTransaction(T->getCompilationOpts());
      // Process used vtables and generate implicit bodies.
      getCI()->getSema().DefineUsedVTables();
      // Pull all template instantiations in that came from the consumers.
      getCI()->getSema().PerformPendingInstantiations();
#ifdef _WIN32
      // Microsoft-specific:
      // Late parsed templates can leave unswallowed "macro"-like tokens.
      // They will seriously confuse the Parser when entering the next
      // source file. So lex until we are EOF.
      Token Tok;
      Tok.setKind(tok::annot_repl_input_end);
      do {
        getCI()->getSema().getPreprocessor().Lex(Tok);
      } while (Tok.isNot(tok::annot_repl_input_end));
#endif

      ParseResultTransaction nestedPRT = endTransaction(nestedT);
      commitTransaction(nestedPRT);
      m_Consumer->setTransaction(prevConsumerT);
    }
    m_Consumer->HandleTranslationUnit(getCI()->getASTContext());


    // The static initializers might run anything and can thus cause more
    // decls that need to end up in a transaction. But this one is done
    // with CodeGen...
    if (T->getCompilationOpts().CodeGeneration && hasCodeGenerator()) {
      Transaction* prevConsumerT = m_Consumer->getTransaction();
      m_Consumer->setTransaction(T);
      codeGenTransaction(T);
      T->setState(Transaction::kCommitted);
      if (!T->getParent()) {
        if (m_Interpreter->executeTransaction(*T)
            >= Interpreter::kExeFirstError) {
          // Roll back on error in initializers.
          // T maybe pointing to freed memory after this call:
          // Interpreter::unload
          //   IncrementalParser::deregisterTransaction
          //     TransactionPool::releaseTransaction
          m_Interpreter->unload(*T);
          return;
        }
      }
      m_Consumer->setTransaction(prevConsumerT);
    }
    T->setState(Transaction::kCommitted);

    {
      Transaction* prevConsumerT = m_Consumer->getTransaction();
      if (InterpreterCallbacks* callbacks = m_Interpreter->getCallbacks())
        callbacks->TransactionCommitted(*T);
      m_Consumer->setTransaction(prevConsumerT);
    }
  }

  void IncrementalParser::emitTransaction(Transaction* T) {
    for (auto DI = T->decls_begin(), DE = T->decls_end(); DI != DE; ++DI)
      m_CI->getSema().getASTConsumer().HandleTopLevelDecl(DI->m_DGR);
  }

  void IncrementalParser::codeGenTransaction(Transaction* T) {
    // codegen the transaction
    assert(T->getCompilationOpts().CodeGeneration && "CodeGen turned off");
    assert(T->getState() == Transaction::kCompleted && "Must be completed");
    assert(hasCodeGenerator() && "No CodeGen");

    // Could trigger derserialization of decls.
    Transaction* deserT = beginTransaction(CompilationOptions());


    // Commit this transaction first - T might need symbols from it, so
    // trigger emission of weak symbols by providing use.
    ParseResultTransaction PRT = endTransaction(deserT);
    commitTransaction(PRT);
    deserT = PRT.getPointer();

    // This llvm::Module is done; finalize it and pass it to the execution
    // engine.
    if (!T->isNestedTransaction() && hasCodeGenerator()) {
      if (InterpreterCallbacks* callbacks = m_Interpreter->getCallbacks())
        callbacks->TransactionCodeGenStarted(*T);

      // Update CodeGen to current optimization level, which might be different
      // from what it had when constructed.
      auto &CGOpts = m_CI->getCodeGenOpts();
      CGOpts.OptimizationLevel = T->getCompilationOpts().OptLevel;
      CGOpts.setInlining((CGOpts.OptimizationLevel == 0)
                         ? CodeGenOptions::OnlyAlwaysInlining
                         : CodeGenOptions::NormalInlining);

      // The initializers are emitted to the symbol "_GLOBAL__sub_I_" + filename.
      // Make that unique!
      deserT = beginTransaction(CompilationOptions());
      // Reset the module builder to clean up global initializers, c'tors, d'tors
      getCodeGenerator()->HandleTranslationUnit(getCI()->getASTContext());
      auto PRT = endTransaction(deserT);
      commitTransaction(PRT);
      deserT = PRT.getPointer();

      std::unique_ptr<llvm::Module> M(getCodeGenerator()->ReleaseModule());

      if (M)
        T->setModule(std::move(M));

      if (T->getIssuedDiags() != Transaction::kNone) {
        // Module has been released from Codegen, reset the Diags now.
        DiagnosticsEngine& Diags = getCI()->getSema().getDiagnostics();
        Diags.Reset(/*soft=*/true);
        Diags.getClient()->clear();
      }

      if (InterpreterCallbacks* callbacks = m_Interpreter->getCallbacks())
        callbacks->TransactionCodeGenFinished(*T);

      // Create a new module.
      StartModule();
    }
  }

  void IncrementalParser::deregisterTransaction(Transaction& T) {
    if (&T == m_Consumer->getTransaction())
      m_Consumer->setTransaction(T.getParent());

    if (Transaction* Parent = T.getParent()) {
      Parent->removeNestedTransaction(&T);
      T.setParent(nullptr);
    } else {
      if (&T == m_Transactions.back()) {
        // Remove from the queue
        m_Transactions.pop_back();
        if (!m_Transactions.empty())
          m_Transactions.back()->setNext(nullptr);
      } else {
        // If T is not the last transaction it must not be a previous
        // transaction either, but a "disconnected" one, i.e. one that
        // was not yet committed.
        assert(std::find(m_Transactions.begin(), m_Transactions.end(), &T)
              == m_Transactions.end() && "Out of order transaction removal");
      }
    }

    m_TransactionPool->releaseTransaction(&T);
  }

  std::vector<const Transaction*> IncrementalParser::getAllTransactions() {
    std::vector<const Transaction*> result(m_Transactions.size());
    const cling::Transaction* T = getFirstTransaction();
    while (T) {
      result.push_back(T);
      T = T->getNext();
    }
    return result;
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
  void IncrementalParser::initializeVirtualFile() {
    SourceManager& SM = m_CI->getSourceManager();
    FileManager& FM = m_CI->getFileManager();
    // Build the virtual file, Give it a name that's likely not to ever
    // be #included (so we won't get a clash in clang's cache).
    const char* Filename = "<<< includer >>>";
    FileEntryRef FE = FM.getVirtualFileRef(Filename, 1U << 15U, time(0));

    // Tell ASTReader to create a FileID even if this file does not exist:
    SM.setFileIsTransient(FE);

    SourceLocation Result = SM.getLocForStartOfFile(SM.getMainFileID());
    m_VirtualFileID = SM.createFileID(FE, Result, SrcMgr::C_User);

    auto Buffer =
        llvm::MemoryBuffer::getMemBufferCopy("/*CLING DEFAULT MEMBUF*/;\n");

    SM.overrideFileContents(FE, std::move(Buffer));

    // SourceManager& SM = getCI()->getSourceManager();
    // m_VirtualFileID = SM.getMainFileID();
    if (m_VirtualFileID.isInvalid())
      cling::errs() << "VirtualFileID could not be created.\n";
  }

  IncrementalParser::ParseResultTransaction
  IncrementalParser::Compile(llvm::StringRef input,
                             const CompilationOptions& Opts) {
    Transaction* CurT = beginTransaction(Opts);
    EParseResult ParseRes = ParseInternal(input);

    if (ParseRes == kSuccessWithWarnings)
      CurT->setIssuedDiags(Transaction::kWarnings);
    else if (ParseRes == kFailed)
      CurT->setIssuedDiags(Transaction::kErrors);

    ParseResultTransaction PRT = endTransaction(CurT);
    commitTransaction(PRT);

    return PRT;
  }

  // Add the input to the memory buffer, parse it, and add it to the AST.
  IncrementalParser::EParseResult
  IncrementalParser::ParseInternal(llvm::StringRef input) {
    if (input.empty())
      return IncrementalParser::kSuccess;
    Sema& S = getCI()->getSema();

    // Recover resources if we crash before exiting this method.
    llvm::CrashRecoveryContextCleanupRegistrar<Sema> CleanupSema(&S);

    Preprocessor& PP = m_CI->getPreprocessor();
    if (!PP.getCurrentLexer()) {
       PP.EnterSourceFile(m_CI->getSourceManager().getMainFileID(),
             nullptr, SourceLocation());
    }
    assert(PP.isIncrementalProcessingEnabled() && "Not in incremental mode!?");

    smallstream source_name;
    // FIXME: Pre-increment to avoid failing tests.
    source_name << "input_line_" << ++InputCount;

    // Create an uninitialized memory buffer, copy code in and append "\n"
    size_t InputSize = input.size(); // don't include trailing 0
    // MemBuffer size should *not* include terminating zero
    std::unique_ptr<llvm::WritableMemoryBuffer>
      MB(llvm::WritableMemoryBuffer::getNewUninitMemBuffer(InputSize + 1,
                                                           source_name.str()));
    char* MBStart = MB->getBufferStart();
    memcpy(MBStart, input.data(), InputSize);
    MBStart[InputSize] = '\n';

    SourceManager& SM = getCI()->getSourceManager();

    // Create SourceLocation, which will allow clang to order the overload
    // candidates for example
    SourceLocation NewLoc = getNextAvailableUniqueSourceLoc();

    // Create FileID for the current buffer.
    FileID FID;
    // Create FileEntry and FileID for the current buffer.
    // Enabling the completion point only works on FileEntries.
    FileEntryRef FE =
        SM.getFileManager().getVirtualFileRef(source_name.str(), InputSize,
                                              0 /* mod time*/);
    SM.overrideFileContents(FE, std::move(MB));

    // Ensure HeaderFileInfo exists before lookup to prevent assertion
    HeaderSearch& HS = PP.getHeaderSearchInfo();
    HS.getFileInfo(FE);

    FID = SM.createFileID(FE, NewLoc, SrcMgr::C_User);

    // NewLoc only used for diags.
    PP.EnterSourceFile(FID, /*DirLookup*/nullptr, NewLoc);
    m_Consumer->getTransaction()->setBufferFID(FID);

    llvm::Error res = ParseOrWrapTopLevelDecl();
    if (res) {
      llvm::consumeError(std::move(res));
      return kFailed;
    }

    if (PP.getLangOpts().DelayedTemplateParsing) {
      // Microsoft-specific:
      // Late parsed templates can leave unswallowed "macro"-like tokens.
      // They will seriously confuse the Parser when entering the next
      // source file. So lex until we are EOF.
      Token Tok;
      do {
        PP.Lex(Tok);
      } while (Tok.isNot(tok::annot_repl_input_end));
    }

    Token AssertTok;
    PP.Lex(AssertTok);
    assert(AssertTok.is(tok::annot_repl_input_end) &&
           "Lexer must be EOF when starting incremental parse!");

    DiagnosticsEngine& Diags = getCI()->getDiagnostics();
    if (m_Consumer->getTransaction()->getIssuedDiags() == Transaction::kErrors)
      return kFailed;
    else if (Diags.getNumWarnings())
      return kSuccessWithWarnings;

    return kSuccess;
  }

  llvm::Error IncrementalParser::ParseOrWrapTopLevelDecl() {
    // Recover resources if we crash before exiting this method.
    Sema& S = getCI()->getSema();
    DiagnosticsEngine& Diags = getCI()->getDiagnostics();

    const CompilationOptions& CO =
        m_Consumer->getTransaction()->getCompilationOpts();
    FilteringDiagConsumer::RAAI RAAITmp(*m_DiagConsumer, CO.IgnorePromptDiags);

    llvm::CrashRecoveryContextCleanupRegistrar<Sema> CleanupSema(&S);
    Sema::GlobalEagerInstantiationScope GlobalInstantiations(S, /*Enabled=*/true);
    Sema::LocalEagerInstantiationScope LocalInstantiations(S);

    // Skip previous eof due to last incremental input.
    if (m_Parser->getCurToken().is(tok::annot_repl_input_end)) {
      m_Parser->ConsumeAnyToken();
    }

    Parser::DeclGroupPtrTy ADecl;
    Sema::ModuleImportState ImportState;
    for (bool AtEOF = m_Parser->ParseFirstTopLevelDecl(ADecl, ImportState);
         !AtEOF; AtEOF = m_Parser->ParseTopLevelDecl(ADecl, ImportState)) {
      if (ADecl && !S.getASTConsumer().HandleTopLevelDecl(ADecl.get())) {
        m_Consumer->getTransaction()->setIssuedDiags(Transaction::kErrors);
        return llvm::make_error<llvm::StringError>(
            "Parsing failed. "
            "The consumer rejected a decl",
            std::error_code());
      }
    }

    // If never entered the while block, there's a chance an error occured
    if (Diags.hasErrorOccurred()) {
      m_Consumer->getTransaction()->setIssuedDiags(Transaction::kErrors);
      // Diags.Reset(/*soft=*/true);
      // Diags.getClient()->clear();
      return llvm::make_error<llvm::StringError>("Parsing failed.",
                                                 std::error_code());
    }

    // Process any TopLevelDecls generated by #pragma weak.
    for (Decl* D : S.WeakTopLevelDecls()) {
      DeclGroupRef DGR(D);
      S.getASTConsumer().HandleTopLevelDecl(DGR);
    }

    LocalInstantiations.perform();
    GlobalInstantiations.perform();

    return llvm::Error::success();
  }

  void IncrementalParser::printTransactionStructure() const {
    for(size_t i = 0, e = m_Transactions.size(); i < e; ++i) {
      m_Transactions[i]->printStructureBrief();
    }
  }

  void IncrementalParser::SetTransformers(bool isChildInterpreter) {
    // Add transformers to the IncrementalParser, which owns them
    Sema* TheSema = &m_CI->getSema();
    // if the interpreter compiles ptx code, some transformers should not be
    // used
    bool isCUDADevice = m_Interpreter->getOptions().CompilerOpts.CUDADevice;
    // Register the AST Transformers
    typedef std::unique_ptr<ASTTransformer> ASTTPtr_t;
    std::vector<ASTTPtr_t> ASTTransformers;
    ASTTransformers.emplace_back(new AutoSynthesizer(TheSema));
    ASTTransformers.emplace_back(new EvaluateTSynthesizer(TheSema));
    if (hasCodeGenerator() && !m_Interpreter->getOptions().NoRuntime) {
      // Don't protect against crashes if we cannot run anything.
      // cling might also be in a PCH-generation mode; don't inject our Sema
      // pointer into the PCH.
#ifndef CLING_WITH_ADAPTIVECPP
      if (!isCUDADevice && m_Interpreter->getOptions().PtrCheck)
        ASTTransformers.emplace_back(
            new NullDerefProtectionTransformer(m_Interpreter));
#endif
      if (isCUDADevice)
        ASTTransformers.emplace_back(
            new DeviceKernelInliner(TheSema));
    }
    ASTTransformers.emplace_back(new DefinitionShadower(*TheSema, *m_Interpreter));

    typedef std::unique_ptr<WrapperTransformer> WTPtr_t;
    std::vector<WTPtr_t> WrapperTransformers;
    if (!m_Interpreter->getOptions().NoRuntime && !isCUDADevice)
      WrapperTransformers.emplace_back(new ValuePrinterSynthesizer(TheSema));
    WrapperTransformers.emplace_back(new DeclExtractor(TheSema));
    if (!m_Interpreter->getOptions().NoRuntime && !isCUDADevice)
      WrapperTransformers.emplace_back(new ValueExtractionSynthesizer(TheSema,
                                                           isChildInterpreter));
    WrapperTransformers.emplace_back(new CheckEmptyTransactionTransformer(TheSema));

    m_Consumer->SetTransformers(std::move(ASTTransformers),
                                std::move(WrapperTransformers));
  }


} // namespace cling
