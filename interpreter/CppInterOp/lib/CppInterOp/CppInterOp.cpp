//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "CppInterOp/CppInterOp.h"

#include "Compatibility.h"
#include "Sins.h" // for access to private members
#include "Tracing.h"

#include "clang/AST/Attrs.inc"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Comment.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclAccessPair.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/QualTypeNames.h"
#include "clang/AST/RawCommentList.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/LangStandard.h"
#include "clang/Basic/Linkage.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Basic/Version.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Interpreter/Interpreter.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/Ownership.h"
#include "clang/Sema/Redeclaration.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/TemplateDeduction.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/CoreContainers.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <sys/types.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <utility>
// Stream redirect.
#ifdef _WIN32
#include <io.h>
#ifndef STDOUT_FILENO
#define STDOUT_FILENO 1
#define STDERR_FILENO 2
// For exec().
#include <stdio.h>
#define popen(x, y) (_popen(x, y))
#define pclose (_pclose)
#endif
#else
#include <dlfcn.h>
#include <unistd.h>
#endif // WIN32

//  Runtime symbols required if the library using JIT (Cpp::Evaluate) does
//  not link to llvm
#if !defined(CPPINTEROP_USE_CLING) && !defined(EMSCRIPTEN)
struct __clang_Interpreter_NewTag {
} __ci_newtag;
#if CLANG_VERSION_MAJOR > 21
extern "C" void* __clang_Interpreter_SetValueWithAlloc(void* This, void* OutVal,
                                                       void* OpaqueType);
#else
void* __clang_Interpreter_SetValueWithAlloc(void* This, void* OutVal,
                                            void* OpaqueType);
#endif

extern "C" void __clang_Interpreter_SetValueNoAlloc(void* This, void* OutVal,
                                                    void* OpaqueType, ...);
#endif // CPPINTEROP_USE_CLING

// LSan ships as part of ASan only on Linux and macOS. MSVC and
// Emscripten set the ASan feature macros but do not provide
// __lsan_ignore_object, so emitting the hook there would fail to
// JIT-link the wrapper.
#if !defined(_WIN32) && !defined(__EMSCRIPTEN__) &&                            \
    (defined(__SANITIZE_ADDRESS__) ||                                          \
     (defined(__has_feature) && __has_feature(address_sanitizer)))
#define CPPINTEROP_ASAN_BUILD 1
#endif

namespace CppImpl {

using namespace clang;
using namespace llvm;

struct InterpreterInfo {
  compat::Interpreter* Interpreter = nullptr;
  bool isOwned = true;
  // Store the list of builtin types.
  llvm::StringMap<QualType> BuiltinMap;
  // Per-interpreter wrapper caches. Keyed on AST nodes that belong to this
  // interpreter, so the caches must be destroyed together with it.
  std::map<const FunctionDecl*, void*> WrapperStore;
  std::map<const Decl*, void*> DtorWrapperStore;

  InterpreterInfo(compat::Interpreter* I, bool Owned)
      : Interpreter(I), isOwned(Owned) {}

  // Enable move constructors.
  InterpreterInfo(InterpreterInfo&& other) noexcept
      : Interpreter(other.Interpreter), isOwned(other.isOwned) {
    other.Interpreter = nullptr;
    other.isOwned = false;
  }
  InterpreterInfo& operator=(InterpreterInfo&& other) noexcept {
    if (this != &other) {
      // Delete current resource if owned
      if (isOwned)
        delete Interpreter;

      Interpreter = other.Interpreter;
      isOwned = other.isOwned;

      other.Interpreter = nullptr;
      other.isOwned = false;
    }
    return *this;
  }

  ~InterpreterInfo() {
    if (isOwned)
      delete Interpreter;
  }

  // Disable copy semantics (to avoid accidental double deletes)
  InterpreterInfo(const InterpreterInfo&) = delete;
  InterpreterInfo& operator=(const InterpreterInfo&) = delete;
};

static void DefaultProcessCrashHandler(void*);
// Function-static storage for interpreters
static std::deque<InterpreterInfo>&
GetInterpreters(bool SetCrashHandler = true) {
  // static int FakeArgc = 1;
  // static const std::string VersionStr = GetVersion();
  // static const char* ArgvBuffer[] = {VersionStr.c_str(), nullptr};
  // static const char** FakeArgv = ArgvBuffer;
  // static llvm::InitLLVM X(FakeArgc, FakeArgv);
  // Cannot be a llvm::ManagedStatic because X will call shutdown which will
  // trigger destruction on llvm::ManagedStatics and the destruction of the
  // InterpreterInfos require to have llvm around.
  // FIXME: Currently we never call llvm::llvm_shutdown and sInterpreters leaks.
  static llvm::ManagedStatic<std::deque<InterpreterInfo>> sInterpreters;
  static std::once_flag ProcessInitialized;
  std::call_once(ProcessInitialized, [SetCrashHandler]() {
    if (SetCrashHandler)
      llvm::sys::PrintStackTraceOnErrorSignal("CppInterOp");

    if (getenv("CPPINTEROP_LOG") != nullptr)
      CppInterOp::Tracing::InitTracing();

    // Initialize all targets (required for device offloading)
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();

    if (SetCrashHandler)
      llvm::sys::AddSignalHandler(DefaultProcessCrashHandler,
                                  /*Cookie=*/nullptr);

    // std::atexit(llvm::llvm_shutdown);
  });

  return *sInterpreters;
}

// Global crash handler for the entire process
static void DefaultProcessCrashHandler(void*) {
  // Access the static deque via the getter
  std::deque<InterpreterInfo>& Interps = GetInterpreters();

  llvm::errs() << "\n**************************************************\n";
  llvm::errs() << "  CppInterOp CRASH DETECTED\n";
  if (CppInterOp::Tracing::TraceInfo::TheTraceInfo) {
    std::string Path =
        CppInterOp::Tracing::TraceInfo::TheTraceInfo->writeToFile();
    if (!Path.empty())
      llvm::errs() << "  Reproducer saved to: " << Path << "\n";
    else
      llvm::errs() << "  Failed to write reproducer file.\n";
  } else {
    llvm::errs() << "  Re-run with CPPINTEROP_LOG=1 for a crash reproducer\n";
  }

  if (!Interps.empty()) {
    llvm::errs() << "  Active Interpreters:\n";
    for (const auto& Info : Interps) {
      if (Info.Interpreter)
        llvm::errs() << "    - " << Info.Interpreter << "\n";
    }
  }

  llvm::errs() << "**************************************************\n";
  llvm::errs().flush();

  // Print backtrace (includes JIT symbols if registered)
  llvm::sys::PrintStackTrace(llvm::errs());

  llvm::errs() << "**************************************************\n";
  llvm::errs().flush();

  // The process must actually terminate for EXPECT_DEATH to pass.
  // We use _exit to avoid calling atexit() handlers which might be corrupted.
  llvm::sys::Process::Exit(/*RetCode=*/1, /*NoCleanup=*/false);
}

static void RegisterInterpreter(compat::Interpreter* I, bool Owned) {
  std::deque<InterpreterInfo>& Interps = GetInterpreters(Owned);
  Interps.emplace_back(I, Owned);
}

static InterpreterInfo& getInterpInfo(compat::Interpreter* I = nullptr) {
  auto& Interps = GetInterpreters();
  assert(!Interps.empty() &&
         "Interpreter instance must be set before calling this!");
  if (I) {
    for (auto& Info : Interps)
      if (Info.Interpreter == I)
        return Info;
  }
  return Interps.back();
}

static compat::Interpreter& getInterp(TInterp_t I = nullptr) {
  if (I)
    return *static_cast<compat::Interpreter*>(I);
  return *getInterpInfo().Interpreter;
}

TInterp_t GetInterpreter() {
  INTEROP_TRACE();
  std::deque<InterpreterInfo>& Interps = GetInterpreters();
  if (Interps.empty())
    return INTEROP_RETURN(nullptr);
  return INTEROP_RETURN(Interps.back().Interpreter);
}

void UseExternalInterpreter(TInterp_t I) {
  INTEROP_TRACE(I);
  assert(GetInterpreters(false).empty() && "sInterpreter already in use!");
  RegisterInterpreter(static_cast<compat::Interpreter*>(I), /*Owned=*/false);
  return INTEROP_VOID_RETURN();
}

bool ActivateInterpreter(TInterp_t I) {
  INTEROP_TRACE(I);
  if (!I)
    return INTEROP_RETURN(false);

  std::deque<InterpreterInfo>& Interps = GetInterpreters();
  auto found =
      std::find_if(Interps.begin(), Interps.end(),
                   [&I](const auto& Info) { return Info.Interpreter == I; });
  if (found == Interps.end())
    return INTEROP_RETURN(false);

  if (std::next(found) != Interps.end()) // if not already last element.
    std::rotate(found, found + 1, Interps.end());

  return INTEROP_RETURN(true); // success
}

bool DeleteInterpreter(TInterp_t I /*=nullptr*/) {
  INTEROP_TRACE(I);
  std::deque<InterpreterInfo>& Interps = GetInterpreters();
  if (Interps.empty())
    return INTEROP_RETURN(false);

  if (!I) {
    Interps.pop_back(); // Triggers ~InterpreterInfo() and potential delete
    return INTEROP_RETURN(true);
  }

  auto found =
      std::find_if(Interps.begin(), Interps.end(),
                   [&I](const auto& Info) { return Info.Interpreter == I; });
  if (found == Interps.end())
    return INTEROP_RETURN(false); // failure

  Interps.erase(found);
  return INTEROP_RETURN(true);
}

static clang::Sema& getSema() { return getInterp().getCI()->getSema(); }
static clang::ASTContext& getASTContext() { return getSema().getASTContext(); }

static void ForceCodeGen(Decl* D, compat::Interpreter& I) {
  // The decl was deferred by CodeGen. Force its emission.
  // FIXME: In ASTContext::DeclMustBeEmitted we should check if the
  // Decl::isUsed is set or we should be able to access CodeGen's
  // addCompilerUsedGlobal.
  ASTContext& C = I.getSema().getASTContext();

  D->addAttr(UsedAttr::CreateImplicit(C));
#ifdef CPPINTEROP_USE_CLING
  cling::Interpreter::PushTransactionRAII RAII(&I);
  I.getCI()->getASTConsumer().HandleTopLevelDecl(DeclGroupRef(D));
#else // CLANG_REPL
  I.getCI()->getASTConsumer().HandleTopLevelDecl(DeclGroupRef(D));
  // Take the newest llvm::Module produced by CodeGen and send it to JIT.
  auto GeneratedPTU = I.Parse("");
  if (!GeneratedPTU)
    llvm::logAllUnhandledErrors(GeneratedPTU.takeError(), llvm::errs(),
                                "[ForceCodeGen] Failed to generate PTU:");

  // From cling's BackendPasses.cpp
  // FIXME: We need to upstream this code in IncrementalExecutor::addModule
  for (auto& GV : GeneratedPTU->TheModule->globals()) {
    llvm::GlobalValue::LinkageTypes LT = GV.getLinkage();
    if (GV.isDeclaration() || !GV.hasName() ||
        GV.getName().starts_with(".str") ||
        !llvm::GlobalVariable::isDiscardableIfUnused(LT) ||
        LT != llvm::GlobalValue::InternalLinkage)
      continue; // nothing to do
    GV.setLinkage(llvm::GlobalValue::WeakAnyLinkage);
  }
  if (auto Err = I.Execute(*GeneratedPTU))
    llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(),
                                "[ForceCodeGen] Failed to execute PTU:");
#endif
}

#define DEBUG_TYPE "jitcall"
bool JitCall::AreArgumentsValid(void* result, ArgList args, void* self,
                                size_t nary) const {
  bool Valid = true;
  if (Cpp::IsConstructor(m_FD)) {
    assert(result && "Must pass the location of the created object!");
    Valid &= (bool)result;
  }
  if (Cpp::GetFunctionRequiredArgs(m_FD) > args.m_ArgSize) {
    assert(0 && "Must pass at least the minimal number of args!");
    Valid = false;
  }
  if (args.m_ArgSize) {
    assert(args.m_Args != nullptr && "Must pass an argument list!");
    Valid &= (bool)args.m_Args;
  }
  if (!Cpp::IsConstructor(m_FD) && !Cpp::IsDestructor(m_FD) &&
      Cpp::IsMethod(m_FD) && !Cpp::IsStaticMethod(m_FD)) {
    assert(self && "Must pass the pointer to object");
    Valid &= (bool)self;
  }
  const auto* FD = cast<FunctionDecl>((const Decl*)m_FD);
  if (!FD->getReturnType()->isVoidType() && !result) {
    assert(0 && "We are discarding the return type of the function!");
    Valid = false;
  }
  if (Cpp::IsConstructor(m_FD) && nary == 0UL) {
    assert(0 && "Number of objects to construct should be atleast 1");
    Valid = false;
  }
  if (Cpp::IsConstructor(m_FD)) {
    const auto* CD = cast<CXXConstructorDecl>((const Decl*)m_FD);
    if (CD->getMinRequiredArguments() != 0 && nary > 1) {
      assert(0 &&
             "Cannot pass initialization parameters to array new construction");
      Valid = false;
    }
  }
  assert(m_Kind != kDestructorCall && "Wrong overload!");
  Valid &= m_Kind != kDestructorCall;
  return Valid;
}

void JitCall::ReportInvokeStart(void* result, ArgList args, void* self) const {
  std::string Name;
  llvm::raw_string_ostream OS(Name);
  auto* FD = (const FunctionDecl*)m_FD;
  FD->getNameForDiagnostic(OS, FD->getASTContext().getPrintingPolicy(),
                           /*Qualified=*/true);
  LLVM_DEBUG(dbgs() << "Run '" << Name << "', compiled at: "
                    << (void*)m_GenericCall << " with result at: " << result
                    << " , args at: " << args.m_Args << " , arg count: "
                    << args.m_ArgSize << " , self at: " << self << "\n";);

  if (auto* TI = CppInterOp::Tracing::TraceInfo::TheTraceInfo) {
    std::string SelfPart = self ? TI->lookupHandle(self) : "";
    std::string Entry =
        llvm::formatv("  // JitCall::Invoke {0}(nargs={1}, self={2})", Name,
                      args.m_ArgSize, SelfPart.empty() ? "nullptr" : SelfPart);
    TI->appendToLog(Entry);
  }
}

void JitCall::ReportInvokeStart(void* object, unsigned long nary,
                                int withFree) const {
  std::string Name;
  llvm::raw_string_ostream OS(Name);
  auto* FD = (const FunctionDecl*)m_FD;
  FD->getNameForDiagnostic(OS, FD->getASTContext().getPrintingPolicy(),
                           /*Qualified=*/true);
  LLVM_DEBUG(dbgs() << "Finish '" << Name
                    << "', compiled at: " << (void*)m_DestructorCall);

  if (auto* TI = CppInterOp::Tracing::TraceInfo::TheTraceInfo) {
    std::string ObjPart = object ? TI->lookupHandle(object) : "nullptr";
    std::string Entry = llvm::formatv(
        "  // JitCall::InvokeDestructor {0}(object={1}, nary={2}, "
        "withFree={3})",
        Name, ObjPart, nary, withFree);
    TI->appendToLog(Entry);
  }
}

#undef DEBUG_TYPE

std::string GetVersion() {
  INTEROP_TRACE();
  const char* const VERSION = CPPINTEROP_VERSION;
  std::string fullVersion = "CppInterOp version";
  fullVersion += VERSION;
  fullVersion += "\n (based on "
#ifdef CPPINTEROP_USE_CLING
                 "cling ";
#else
                 "clang-repl";
#endif // CPPINTEROP_USE_CLING
  return INTEROP_RETURN(fullVersion + "[" + clang::getClangFullVersion() +
                        "])\n");
}

std::string Demangle(const std::string& mangled_name) {
  INTEROP_TRACE(mangled_name);
  // Both itaniumDemangle and microsoftDemangle return a malloc'd buffer
  // that the caller owns; the implicit std::string conversion copies the
  // bytes but never frees the original. See llvm/Demangle/Demangle.h.
#ifdef _WIN32
  char* Raw = microsoftDemangle(mangled_name, nullptr, nullptr);
#else
  char* Raw = llvm::itaniumDemangle(mangled_name);
#endif
  std::string demangle = Raw ? Raw : "";
  std::free(Raw);
  return INTEROP_RETURN(demangle);
}

void EnableDebugOutput(bool value /* =true*/) {
  INTEROP_TRACE(value);
  llvm::DebugFlag = value;
  return INTEROP_VOID_RETURN();
}

bool IsDebugOutputEnabled() {
  INTEROP_TRACE();
  return INTEROP_RETURN(llvm::DebugFlag);
}

static void InstantiateFunctionDefinition(Decl* D) {
  compat::SynthesizingCodeRAII RAII(&getInterp());
  if (auto* FD = llvm::dyn_cast_or_null<FunctionDecl>(D)) {
    getSema().InstantiateFunctionDefinition(SourceLocation(), FD,
                                            /*Recursive=*/true,
                                            /*DefinitionRequired=*/true);
    // FIXME: this can go into a RAII object
    clang::DiagnosticsEngine& Diags = getSema().getDiagnostics();
    if (!FD->isDefined() && Diags.hasErrorOccurred()) {
      // instantiation failed, need to reset DiagnosticsEngine
      Diags.Reset(/*soft=*/true);
      Diags.getClient()->clear();
    }
  }
}

bool IsAggregate(TCppScope_t scope) {
  INTEROP_TRACE(scope);
  Decl* D = static_cast<Decl*>(scope);

  // Aggregates are only arrays or tag decls.
  if (ValueDecl* ValD = dyn_cast<ValueDecl>(D))
    if (ValD->getType()->isArrayType())
      return INTEROP_RETURN(true);

  // struct, class, union
  if (CXXRecordDecl* CXXRD = dyn_cast<CXXRecordDecl>(D))
    return INTEROP_RETURN(CXXRD->isAggregate());

  return INTEROP_RETURN(false);
}

bool IsNamespace(TCppScope_t scope) {
  INTEROP_TRACE(scope);
  Decl* D = static_cast<Decl*>(scope);
  return INTEROP_RETURN(isa<NamespaceDecl>(D));
}

bool IsClass(TCppScope_t scope) {
  INTEROP_TRACE(scope);
  Decl* D = static_cast<Decl*>(scope);
  return INTEROP_RETURN(isa<CXXRecordDecl>(D));
}

bool IsFunction(TCppScope_t scope) {
  INTEROP_TRACE(scope);
  Decl* D = static_cast<Decl*>(scope);
  return INTEROP_RETURN(isa<FunctionDecl>(D));
}

bool IsFunctionPointerType(TCppType_t type) {
  INTEROP_TRACE(type);
  QualType QT = QualType::getFromOpaquePtr(type);
  return INTEROP_RETURN(QT->isFunctionPointerType());
}

bool IsClassPolymorphic(TCppScope_t klass) {
  INTEROP_TRACE(klass);
  Decl* D = static_cast<Decl*>(klass);
  if (auto* CXXRD = llvm::dyn_cast<CXXRecordDecl>(D))
    if (auto* CXXRDD = CXXRD->getDefinition())
      return INTEROP_RETURN(CXXRDD->isPolymorphic());
  return INTEROP_RETURN(false);
}

static SourceLocation GetValidSLoc(Sema& semaRef) {
  auto& SM = semaRef.getSourceManager();
  return SM.getLocForStartOfFile(SM.getMainFileID());
}

// See TClingClassInfo::IsLoaded
bool IsComplete(TCppScope_t scope) {
  INTEROP_TRACE(scope);
  if (!scope)
    return INTEROP_RETURN(false);

  Decl* D = static_cast<Decl*>(scope);

  if (isa<ClassTemplateSpecializationDecl>(D)) {
    QualType QT = QualType::getFromOpaquePtr(GetTypeFromScope(scope));
    clang::Sema& S = getSema();
    SourceLocation fakeLoc = GetValidSLoc(S);
    compat::SynthesizingCodeRAII RAII(&getInterp());
    return INTEROP_RETURN(S.isCompleteType(fakeLoc, QT));
  }

  if (auto* CXXRD = dyn_cast<CXXRecordDecl>(D))
    return INTEROP_RETURN(CXXRD->hasDefinition());
  else if (auto* TD = dyn_cast<TagDecl>(D))
    return INTEROP_RETURN(TD->getDefinition());

  // Everything else is considered complete.
  return INTEROP_RETURN(true);
}

size_t SizeOf(TCppScope_t scope) {
  INTEROP_TRACE(scope);
  assert(scope);
  if (!IsComplete(scope))
    return INTEROP_RETURN(0);

  if (auto* RD = dyn_cast<RecordDecl>(static_cast<Decl*>(scope))) {
    ASTContext& Context = RD->getASTContext();
    const ASTRecordLayout& Layout = Context.getASTRecordLayout(RD);
    return INTEROP_RETURN(Layout.getSize().getQuantity());
  }

  return INTEROP_RETURN(0);
}

bool IsBuiltin(TCppConstType_t type) {
  INTEROP_TRACE(type);
  QualType Ty = QualType::getFromOpaquePtr(type);
  if (Ty->isBuiltinType() || Ty->isAnyComplexType())
    return INTEROP_RETURN(true);
  // Check for std::complex<T> specializations.
  if (const auto* RD = Ty->getAsCXXRecordDecl()) {
    if (const auto* CTSD = dyn_cast<ClassTemplateSpecializationDecl>(RD)) {
      IdentifierInfo* II = CTSD->getSpecializedTemplate()->getIdentifier();
      if (II && II->isStr("complex") &&
          CTSD->getDeclContext()->isStdNamespace())
        return INTEROP_RETURN(true);
    }
  }
  return INTEROP_RETURN(false);
}

bool IsTemplate(TCppScope_t handle) {
  INTEROP_TRACE(handle);
  auto* D = (clang::Decl*)handle;
  return INTEROP_RETURN(llvm::isa_and_nonnull<clang::TemplateDecl>(D));
}

bool IsTemplateSpecialization(TCppScope_t handle) {
  INTEROP_TRACE(handle);
  auto* D = (clang::Decl*)handle;
  return INTEROP_RETURN(
      llvm::isa_and_nonnull<clang::ClassTemplateSpecializationDecl>(D));
}

bool IsTypedefed(TCppScope_t handle) {
  INTEROP_TRACE(handle);
  auto* D = (clang::Decl*)handle;
  return INTEROP_RETURN(llvm::isa_and_nonnull<clang::TypedefNameDecl>(D));
}

bool IsAbstract(TCppType_t klass) {
  INTEROP_TRACE(klass);
  auto* D = (clang::Decl*)klass;
  if (auto* CXXRD = llvm::dyn_cast_or_null<clang::CXXRecordDecl>(D))
    return INTEROP_RETURN(CXXRD->isAbstract());

  return INTEROP_RETURN(false);
}

bool IsEnumScope(TCppScope_t handle) {
  INTEROP_TRACE(handle);
  auto* D = (clang::Decl*)handle;
  return INTEROP_RETURN(llvm::isa_and_nonnull<clang::EnumDecl>(D));
}

bool IsEnumConstant(TCppScope_t handle) {
  INTEROP_TRACE(handle);
  auto* D = (clang::Decl*)handle;
  return INTEROP_RETURN(llvm::isa_and_nonnull<clang::EnumConstantDecl>(D));
}

bool IsEnumType(TCppType_t type) {
  INTEROP_TRACE(type);
  QualType QT = QualType::getFromOpaquePtr(type);
  return INTEROP_RETURN(QT->isEnumeralType());
}

static bool isSmartPointer(const RecordType* RT) {
  auto IsUseCountPresent = [](const RecordDecl* Record) {
    ASTContext& C = Record->getASTContext();
    return !Record->lookup(&C.Idents.get("use_count")).empty();
  };
  auto IsOverloadedOperatorPresent = [](const RecordDecl* Record,
                                        OverloadedOperatorKind Op) {
    ASTContext& C = Record->getASTContext();
    DeclContextLookupResult Result =
        Record->lookup(C.DeclarationNames.getCXXOperatorName(Op));
    return !Result.empty();
  };

  const RecordDecl* Record = RT->getDecl();
  if (IsUseCountPresent(Record))
    return true;

  bool foundStarOperator = IsOverloadedOperatorPresent(Record, OO_Star);
  bool foundArrowOperator = IsOverloadedOperatorPresent(Record, OO_Arrow);
  if (foundStarOperator && foundArrowOperator)
    return true;

  const CXXRecordDecl* CXXRecord = dyn_cast<CXXRecordDecl>(Record);
  if (!CXXRecord)
    return false;

  auto FindOverloadedOperators = [&](const CXXRecordDecl* Base) {
    // If we find use_count, we are done.
    if (IsUseCountPresent(Base))
      return false; // success.
    if (!foundStarOperator)
      foundStarOperator = IsOverloadedOperatorPresent(Base, OO_Star);
    if (!foundArrowOperator)
      foundArrowOperator = IsOverloadedOperatorPresent(Base, OO_Arrow);
    if (foundStarOperator && foundArrowOperator)
      return false; // success.
    return true;
  };

  return !CXXRecord->forallBases(FindOverloadedOperators);
}

bool IsSmartPtrType(TCppType_t type) {
  INTEROP_TRACE(type);
  QualType QT = QualType::getFromOpaquePtr(type);
  if (const RecordType* RT = QT->getAs<RecordType>()) {
    // Add quick checks for the std smart prts to cover most of the cases.
    std::string typeString = GetTypeAsString(type);
    llvm::StringRef tsRef(typeString);
    if (tsRef.starts_with("std::unique_ptr") ||
        tsRef.starts_with("std::shared_ptr") ||
        tsRef.starts_with("std::weak_ptr"))
      return INTEROP_RETURN(true);
    return INTEROP_RETURN(isSmartPointer(RT));
  }
  return INTEROP_RETURN(false);
}

TCppType_t GetIntegerTypeFromEnumScope(TCppScope_t handle) {
  INTEROP_TRACE(handle);
  auto* D = (clang::Decl*)handle;
  if (auto* ED = llvm::dyn_cast_or_null<clang::EnumDecl>(D)) {
    return INTEROP_RETURN(ED->getIntegerType().getAsOpaquePtr());
  }

  return INTEROP_RETURN(nullptr);
}

TCppType_t GetIntegerTypeFromEnumType(TCppType_t enum_type) {
  INTEROP_TRACE(enum_type);
  if (!enum_type)
    return INTEROP_RETURN(nullptr);

  QualType QT = QualType::getFromOpaquePtr(enum_type);
  if (auto* ET = QT->getAs<EnumType>())
    return INTEROP_RETURN(ET->getDecl()->getIntegerType().getAsOpaquePtr());

  return INTEROP_RETURN(nullptr);
}

std::vector<TCppScope_t> GetEnumConstants(TCppScope_t handle) {
  INTEROP_TRACE(handle);
  auto* D = (clang::Decl*)handle;

  if (auto* ED = llvm::dyn_cast_or_null<clang::EnumDecl>(D)) {
    std::vector<TCppScope_t> enum_constants;
    for (auto* ECD : ED->enumerators()) {
      enum_constants.push_back((TCppScope_t)ECD);
    }

    return INTEROP_RETURN(enum_constants);
  }

  return INTEROP_RETURN(std::vector<TCppScope_t>{});
}

TCppType_t GetEnumConstantType(TCppScope_t handle) {
  INTEROP_TRACE(handle);
  if (!handle)
    return INTEROP_RETURN(nullptr);

  auto* D = (clang::Decl*)handle;
  if (auto* ECD = llvm::dyn_cast<clang::EnumConstantDecl>(D))
    return INTEROP_RETURN(ECD->getType().getAsOpaquePtr());

  return INTEROP_RETURN(nullptr);
}

TCppIndex_t GetEnumConstantValue(TCppScope_t handle) {
  INTEROP_TRACE(handle);
  auto* D = (clang::Decl*)handle;
  if (auto* ECD = llvm::dyn_cast_or_null<clang::EnumConstantDecl>(D)) {
    const llvm::APSInt& Val = ECD->getInitVal();
    return INTEROP_RETURN(Val.getExtValue());
  }
  return INTEROP_RETURN(0);
}

size_t GetSizeOfType(TCppType_t type) {
  INTEROP_TRACE(type);
  QualType QT = QualType::getFromOpaquePtr(type);
  if (const TagType* TT = QT->getAs<TagType>())
    return INTEROP_RETURN(SizeOf(TT->getDecl()));

  // FIXME: Can we get the size of a non-tag type?
  auto TI = getSema().getASTContext().getTypeInfo(QT);
  size_t TypeSize = TI.Width;
  return INTEROP_RETURN(TypeSize / 8);
}

bool IsVariable(TCppScope_t scope) {
  INTEROP_TRACE(scope);
  auto* D = (clang::Decl*)scope;
  return INTEROP_RETURN(llvm::isa_and_nonnull<clang::VarDecl>(D));
}

std::string GetName(TCppType_t klass) {
  INTEROP_TRACE(klass);
  auto* D = (clang::NamedDecl*)klass;

  if (llvm::isa_and_nonnull<TranslationUnitDecl>(D)) {
    return INTEROP_RETURN("");
  }

  if (auto* ND = llvm::dyn_cast_or_null<NamedDecl>(D)) {
    return INTEROP_RETURN(ND->getNameAsString());
  }

  return INTEROP_RETURN("<unnamed>");
}

static std::string GetCompleteNameImpl(TCppType_t klass, bool qualified) {
  auto& C = getSema().getASTContext();
  auto* D = (Decl*)klass;

  if (auto* ND = llvm::dyn_cast_or_null<NamedDecl>(D)) {
    PrintingPolicy Policy = C.getPrintingPolicy();
    Policy.SuppressUnwrittenScope = true;
    if (qualified) {
      Policy.FullyQualifiedName = true;
      Policy.Suppress_Elab = true;
    } else {
      Policy.SuppressScope = true;
      Policy.AnonymousTagLocations = false;
      Policy.SuppressTemplateArgsInCXXConstructors = false;
      Policy.SuppressDefaultTemplateArgs = false;
      Policy.AlwaysIncludeTypeForTemplateArgument = true;
    }

    if (auto* TD = llvm::dyn_cast<TagDecl>(ND)) {
      std::string type_name;
      QualType QT = C.Get_Tag_Type(TD);
      QT.getAsStringInternal(type_name, Policy);
      return type_name;
    }
    if (auto* FD = llvm::dyn_cast<FunctionDecl>(ND)) {
      std::string func_name;
      llvm::raw_string_ostream name_stream(func_name);
      FD->getNameForDiagnostic(name_stream, Policy, qualified);
      name_stream.flush();
      return func_name;
    }

    return qualified ? ND->getQualifiedNameAsString() : ND->getNameAsString();
  }

  if (llvm::isa_and_nonnull<TranslationUnitDecl>(D)) {
    return "";
  }

  return "<unnamed>";
}

std::string GetCompleteName(TCppType_t klass) {
  INTEROP_TRACE(klass);
  return INTEROP_RETURN(GetCompleteNameImpl(klass, /*qualified=*/false));
}

std::string GetQualifiedName(TCppType_t klass) {
  INTEROP_TRACE(klass);
  auto* D = (Decl*)klass;
  if (auto* ND = llvm::dyn_cast_or_null<NamedDecl>(D)) {
    return INTEROP_RETURN(ND->getQualifiedNameAsString());
  }

  if (llvm::isa_and_nonnull<TranslationUnitDecl>(D)) {
    return INTEROP_RETURN("");
  }

  return INTEROP_RETURN("<unnamed>");
}

std::string GetQualifiedCompleteName(TCppType_t klass) {
  INTEROP_TRACE(klass);
  return INTEROP_RETURN(GetCompleteNameImpl(klass, /*qualified=*/true));
}

std::string GetDoxygenComment(TCppScope_t scope, bool strip_comment_markers) {
  INTEROP_TRACE(scope, strip_comment_markers);
  auto* D = static_cast<Decl*>(scope);
  if (!D)
    return INTEROP_RETURN("");

  D = D->getCanonicalDecl();
  ASTContext& C = D->getASTContext();

  const RawComment* RC = C.getRawCommentForAnyRedecl(D);
  if (!RC)
    return INTEROP_RETURN("");

  (void)C.getCommentForDecl(D, /*PP=*/nullptr);

  const SourceManager& SM = C.getSourceManager();

  if (!strip_comment_markers)
    return INTEROP_RETURN(RC->getRawText(SM).str());

  return INTEROP_RETURN(RC->getFormattedText(SM, C.getDiagnostics()));
}

std::vector<TCppScope_t> GetUsingNamespaces(TCppScope_t scope) {
  INTEROP_TRACE(scope);
  auto* D = (clang::Decl*)scope;

  if (auto* DC = llvm::dyn_cast_or_null<clang::DeclContext>(D)) {
    std::vector<TCppScope_t> namespaces;
    for (auto UD : DC->using_directives()) {
      namespaces.push_back((TCppScope_t)UD->getNominatedNamespace());
    }
    return INTEROP_RETURN(namespaces);
  }

  return INTEROP_RETURN(std::vector<TCppScope_t>{});
}

TCppScope_t GetGlobalScope() {
  INTEROP_TRACE();
  return INTEROP_RETURN(
      getSema().getASTContext().getTranslationUnitDecl()->getFirstDecl());
}

static Decl* GetScopeFromType(QualType QT) {
  if (auto* Type = QT.getCanonicalType().getTypePtrOrNull()) {
    Type = Type->getPointeeOrArrayElementType();
    Type = Type->getUnqualifiedDesugaredType();
    if (auto* ET = llvm::dyn_cast<EnumType>(Type))
      return ET->getDecl();
    CXXRecordDecl* CXXRD = Type->getAsCXXRecordDecl();
    if (CXXRD)
      return CXXRD->getCanonicalDecl();
  }
  return 0;
}

TCppScope_t GetScopeFromType(TCppType_t type) {
  INTEROP_TRACE(type);
  QualType QT = QualType::getFromOpaquePtr(type);
  return INTEROP_RETURN((TCppScope_t)GetScopeFromType(QT));
}

static clang::Decl* GetUnderlyingScope(clang::Decl* D) {
  if (auto* TND = dyn_cast_or_null<TypedefNameDecl>(D)) {
    if (auto* Scope = GetScopeFromType(TND->getUnderlyingType()))
      D = Scope;
  } else if (auto* USS = dyn_cast_or_null<UsingShadowDecl>(D)) {
    if (auto* Scope = USS->getTargetDecl())
      D = Scope;
  }

  return D->getCanonicalDecl();
}

TCppScope_t GetUnderlyingScope(TCppScope_t scope) {
  INTEROP_TRACE(scope);
  if (!scope)
    return INTEROP_RETURN(nullptr);
  return INTEROP_RETURN(GetUnderlyingScope((clang::Decl*)scope));
}

TCppScope_t GetScope(const std::string& name, TCppScope_t parent) {
  INTEROP_TRACE(name, parent);
  // FIXME: GetScope should be replaced by a general purpose lookup
  // and filter function. The function should be like GetNamed but
  // also take in a filter parameter which determines which results
  // to pass back
  if (name == "")
    return INTEROP_RETURN(GetGlobalScope());

  auto* ND = (NamedDecl*)GetNamed(name, parent);

  if (!ND || ND == (NamedDecl*)-1)
    return INTEROP_RETURN(nullptr);

  if (llvm::isa<NamespaceDecl>(ND) || llvm::isa<RecordDecl>(ND) ||
      llvm::isa<ClassTemplateDecl>(ND) || llvm::isa<TypedefNameDecl>(ND) ||
      llvm::isa<TypeAliasTemplateDecl>(ND) || llvm::isa<TypeAliasDecl>(ND))
    return INTEROP_RETURN((TCppScope_t)(ND->getCanonicalDecl()));

  return INTEROP_RETURN(nullptr);
}

TCppScope_t GetScopeFromCompleteName(const std::string& name) {
  INTEROP_TRACE(name);
  std::string delim = "::";
  size_t start = 0;
  size_t end = name.find(delim);
  TCppScope_t curr_scope = 0;
  while (end != std::string::npos) {
    curr_scope = GetScope(name.substr(start, end - start), curr_scope);
    start = end + delim.length();
    end = name.find(delim, start);
  }
  return INTEROP_RETURN(GetScope(name.substr(start, end), curr_scope));
}

TCppScope_t GetNamed(const std::string& name,
                     TCppScope_t parent /*= nullptr*/) {
  INTEROP_TRACE(name, parent);
  clang::DeclContext* Within = 0;
  if (parent) {
    auto* D = (clang::Decl*)parent;
    D = GetUnderlyingScope(D);
    Within = llvm::dyn_cast<clang::DeclContext>(D);
  }
#ifdef CPPINTEROP_USE_CLING
  if (Within)
    Within->getPrimaryContext()->buildLookup();
#endif
  compat::SynthesizingCodeRAII RAII(&getInterp());
  auto* ND = CppInternal::utils::Lookup::Named(&getSema(), name, Within);
  if (ND && ND != (clang::NamedDecl*)-1) {
    return INTEROP_RETURN((TCppScope_t)(ND->getCanonicalDecl()));
  }

  return INTEROP_RETURN(nullptr);
}

TCppScope_t GetParentScope(TCppScope_t scope) {
  INTEROP_TRACE(scope);
  auto* D = (clang::Decl*)scope;

  if (llvm::isa_and_nonnull<TranslationUnitDecl>(D)) {
    return INTEROP_RETURN(nullptr);
  }
  auto* ParentDC = D->getDeclContext();

  if (!ParentDC)
    return INTEROP_RETURN(nullptr);

  auto* P = clang::Decl::castFromDeclContext(ParentDC)->getCanonicalDecl();

  if (auto* TU = llvm::dyn_cast_or_null<TranslationUnitDecl>(P))
    return INTEROP_RETURN((TCppScope_t)TU->getFirstDecl());

  return INTEROP_RETURN((TCppScope_t)P);
}

TCppIndex_t GetNumBases(TCppScope_t klass) {
  INTEROP_TRACE(klass);
  auto* D = (Decl*)klass;

  if (auto* CTSD = llvm::dyn_cast_or_null<ClassTemplateSpecializationDecl>(D))
    if (!CTSD->hasDefinition())
      compat::InstantiateClassTemplateSpecialization(getInterp(), CTSD);
  if (auto* CXXRD = llvm::dyn_cast_or_null<CXXRecordDecl>(D)) {
    if (CXXRD->hasDefinition())
      return INTEROP_RETURN(CXXRD->getNumBases());
  }

  return INTEROP_RETURN(0);
}

TCppScope_t GetBaseClass(TCppScope_t klass, TCppIndex_t ibase) {
  INTEROP_TRACE(klass, ibase);
  auto* D = (Decl*)klass;
  auto* CXXRD = llvm::dyn_cast_or_null<CXXRecordDecl>(D);
  if (!CXXRD || CXXRD->getNumBases() <= ibase)
    return INTEROP_RETURN(nullptr);

  auto type = (CXXRD->bases_begin() + ibase)->getType();
  if (auto RT = type->getAs<RecordType>())
    return INTEROP_RETURN((TCppScope_t)RT->getDecl()->getCanonicalDecl());

  return INTEROP_RETURN(nullptr);
}

// FIXME: Consider dropping this interface as it seems the same as
// IsTypeDerivedFrom.
bool IsSubclass(TCppScope_t derived, TCppScope_t base) {
  INTEROP_TRACE(derived, base);
  if (derived == base)
    return INTEROP_RETURN(true);

  if (!derived || !base)
    return INTEROP_RETURN(false);

  auto* derived_D = (clang::Decl*)derived;
  auto* base_D = (clang::Decl*)base;

  if (!isa<CXXRecordDecl>(derived_D) || !isa<CXXRecordDecl>(base_D))
    return INTEROP_RETURN(false);

  auto Derived = cast<CXXRecordDecl>(derived_D);
  auto Base = cast<CXXRecordDecl>(base_D);
  return INTEROP_RETURN(
      IsTypeDerivedFrom(GetTypeFromScope(Derived), GetTypeFromScope(Base)));
}

// Copied from VTableBuilder.cpp
// This is an internal helper function for the CppInterOp library (as evident
// by the 'static' declaration), while the similar GetBaseClassOffset()
// function below is exposed to library users.
static unsigned ComputeBaseOffset(const ASTContext& Context,
                                  const CXXRecordDecl* DerivedRD,
                                  const CXXBasePath& Path) {
  CharUnits NonVirtualOffset = CharUnits::Zero();

  unsigned NonVirtualStart = 0;
  const CXXRecordDecl* VirtualBase = nullptr;

  // First, look for the virtual base class.
  for (int I = Path.size(), E = 0; I != E; --I) {
    const CXXBasePathElement& Element = Path[I - 1];

    if (Element.Base->isVirtual()) {
      NonVirtualStart = I;
      QualType VBaseType = Element.Base->getType();
      VirtualBase = VBaseType->getAsCXXRecordDecl();
      break;
    }
  }

  // Now compute the non-virtual offset.
  for (unsigned I = NonVirtualStart, E = Path.size(); I != E; ++I) {
    const CXXBasePathElement& Element = Path[I];

    // Check the base class offset.
    const ASTRecordLayout& Layout = Context.getASTRecordLayout(Element.Class);

    const CXXRecordDecl* Base = Element.Base->getType()->getAsCXXRecordDecl();

    NonVirtualOffset += Layout.getBaseClassOffset(Base);
  }

  // FIXME: This should probably use CharUnits or something. Maybe we should
  // even change the base offsets in ASTRecordLayout to be specified in
  // CharUnits.
  // return BaseOffset(DerivedRD, VirtuaBose, aBlnVirtualOffset);
  if (VirtualBase) {
    const ASTRecordLayout& Layout = Context.getASTRecordLayout(DerivedRD);
    CharUnits VirtualOffset = Layout.getVBaseClassOffset(VirtualBase);
    return (NonVirtualOffset + VirtualOffset).getQuantity();
  }
  return NonVirtualOffset.getQuantity();
}

int64_t GetBaseClassOffset(TCppScope_t derived, TCppScope_t base) {
  INTEROP_TRACE(derived, base);
  if (base == derived)
    return INTEROP_RETURN(0);

  assert(derived || base);

  auto* DD = (Decl*)derived;
  auto* BD = (Decl*)base;
  if (!isa<CXXRecordDecl>(DD) || !isa<CXXRecordDecl>(BD))
    return INTEROP_RETURN(-1);
  CXXRecordDecl* DCXXRD = cast<CXXRecordDecl>(DD);
  CXXRecordDecl* BCXXRD = cast<CXXRecordDecl>(BD);
  // GCC's -Wmaybe-uninitialized false-positives here only under ASan:
  // -fsanitize=address keeps the SmallDenseMap's union storage live across
  // poison/unpoison calls and blocks the SROA pass that normally folds away
  // the LargeRep read on the Small==true branch. The load survives into the
  // IR the uninit pass sees, and it can no longer prove the `Small` guard.
  // Clang's analyzer does not false-positive here; plain-O2 GCC does not
  // either. Narrow the suppression to GCC + ASan.
#if defined(__GNUC__) && !defined(__clang__) && defined(__SANITIZE_ADDRESS__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
  CXXBasePaths Paths(/*FindAmbiguities=*/false, /*RecordPaths=*/true,
                     /*DetectVirtual=*/false);
#if defined(__GNUC__) && !defined(__clang__) && defined(__SANITIZE_ADDRESS__)
#pragma GCC diagnostic pop
#endif
  DCXXRD->isDerivedFrom(BCXXRD, Paths);

  // FIXME: We might want to cache these requests as they seem expensive.
  return INTEROP_RETURN(
      ComputeBaseOffset(getSema().getASTContext(), DCXXRD, Paths.front()));
}

template <typename DeclType>
static void GetClassDecls(TCppScope_t klass,
                          std::vector<TCppFunction_t>& methods) {
  if (!klass)
    return;

  auto* D = (clang::Decl*)klass;

  if (auto* TD = dyn_cast<TypedefNameDecl>(D))
    D = GetScopeFromType(TD->getUnderlyingType());

  if (!D || !isa<CXXRecordDecl>(D))
    return;

  auto* CXXRD = dyn_cast<CXXRecordDecl>(D);
  compat::SynthesizingCodeRAII RAII(&getInterp());
  if (CXXRD->hasDefinition())
    CXXRD = CXXRD->getDefinition();
  getSema().ForceDeclarationOfImplicitMembers(CXXRD);
  for (Decl* DI : CXXRD->decls()) {
    if (auto* MD = dyn_cast<DeclType>(DI))
      methods.push_back(MD);
    else if (auto* USD = dyn_cast<UsingShadowDecl>(DI)) {
      auto* MD = dyn_cast<DeclType>(USD->getTargetDecl());
      if (!MD)
        continue;

      auto* CUSD = dyn_cast<ConstructorUsingShadowDecl>(DI);
      if (!CUSD) {
        methods.push_back(MD);
        continue;
      }

      auto* CXXCD = dyn_cast_or_null<CXXConstructorDecl>(CUSD->getTargetDecl());
      if (!CXXCD) {
        methods.push_back(MD);
        continue;
      }
      if (CXXCD->isDeleted())
        continue;

      // Result is appended to the decls, i.e. CXXRD, iterator
      // non-shadowed decl will be push_back later
      // methods.push_back(Result);
      getSema().findInheritingConstructor(SourceLocation(), CXXCD, CUSD);
    }
  }
}

void GetClassMethods(TCppScope_t klass, std::vector<TCppFunction_t>& methods) {
  INTEROP_TRACE(klass, INTEROP_OUT(methods));
  GetClassDecls<CXXMethodDecl>(klass, methods);
  return INTEROP_VOID_RETURN();
}

void GetFunctionTemplatedDecls(TCppScope_t klass,
                               std::vector<TCppFunction_t>& methods) {
  INTEROP_TRACE(klass, INTEROP_OUT(methods));
  GetClassDecls<FunctionTemplateDecl>(klass, methods);
  return INTEROP_VOID_RETURN();
}

bool HasDefaultConstructor(TCppScope_t scope) {
  INTEROP_TRACE(scope);
  auto* D = (clang::Decl*)scope;

  if (auto* CXXRD = llvm::dyn_cast_or_null<CXXRecordDecl>(D))
    return INTEROP_RETURN(CXXRD->hasDefaultConstructor());

  return INTEROP_RETURN(false);
}

TCppFunction_t GetDefaultConstructor(compat::Interpreter& interp,
                                     TCppScope_t scope) {
  if (!HasDefaultConstructor(scope))
    return nullptr;

  auto* CXXRD = (clang::CXXRecordDecl*)scope;
  compat::SynthesizingCodeRAII RAII(&getInterp());
  return interp.getCI()->getSema().LookupDefaultConstructor(CXXRD);
}

TCppFunction_t GetDefaultConstructor(TCppScope_t scope) {
  INTEROP_TRACE(scope);
  return INTEROP_RETURN(GetDefaultConstructor(getInterp(), scope));
}

TCppFunction_t GetDestructor(TCppScope_t scope) {
  INTEROP_TRACE(scope);
  auto* D = (clang::Decl*)scope;

  if (auto* CXXRD = llvm::dyn_cast_or_null<CXXRecordDecl>(D)) {
    getSema().ForceDeclarationOfImplicitMembers(CXXRD);
    return INTEROP_RETURN(CXXRD->getDestructor());
  }

  return INTEROP_RETURN(nullptr);
}

void DumpScope(TCppScope_t scope) {
  INTEROP_TRACE(scope);
  auto* D = (clang::Decl*)scope;
  D->dump();
  return INTEROP_VOID_RETURN();
}

std::vector<TCppFunction_t> GetFunctionsUsingName(TCppScope_t scope,
                                                  const std::string& name) {
  INTEROP_TRACE(scope, name);
  auto* D = (Decl*)scope;

  if (!scope || name.empty())
    return INTEROP_RETURN(std::vector<TCppFunction_t>{});

  D = GetUnderlyingScope(D);

  std::vector<TCppFunction_t> funcs;
  llvm::StringRef Name(name);
  auto& S = getSema();
  DeclarationName DName = &getASTContext().Idents.get(name);
  clang::LookupResult R(S, DName, SourceLocation(), Sema::LookupOrdinaryName,
                        RedeclarationKind::ForVisibleRedeclaration);

  CppInternal::utils::Lookup::Named(&S, R, Decl::castToDeclContext(D));

  if (R.empty())
    return INTEROP_RETURN(funcs);

  R.resolveKind();

  for (auto* Found : R)
    if (llvm::isa<FunctionDecl>(Found))
      funcs.push_back(Found);

  return INTEROP_RETURN(funcs);
}

TCppType_t GetFunctionReturnType(TCppFunction_t func) {
  INTEROP_TRACE(func);
  auto* D = (clang::Decl*)func;
  if (auto* FD = llvm::dyn_cast_or_null<clang::FunctionDecl>(D)) {
    QualType Type = FD->getReturnType();
    if (Type->isUndeducedAutoType()) {
      bool needInstantiation = false;
      if (IsTemplatedFunction(FD) && !FD->isDefined())
        needInstantiation = true;
      if (auto* MD = llvm::dyn_cast<clang::CXXMethodDecl>(FD)) {
        if (IsTemplateSpecialization(MD->getParent()))
          needInstantiation = true;
      }

      if (needInstantiation) {
        InstantiateFunctionDefinition(FD);
      }
      Type = FD->getReturnType();
    }
    return INTEROP_RETURN(Type.getAsOpaquePtr());
  }

  if (auto* FD = llvm::dyn_cast_or_null<clang::FunctionTemplateDecl>(D))
    return INTEROP_RETURN(
        (FD->getTemplatedDecl())->getReturnType().getAsOpaquePtr());

  return INTEROP_RETURN(nullptr);
}

TCppIndex_t GetFunctionNumArgs(TCppFunction_t func) {
  INTEROP_TRACE(func);
  auto* D = (clang::Decl*)func;
  if (auto* FD = llvm::dyn_cast_or_null<FunctionDecl>(D))
    return INTEROP_RETURN(FD->getNumParams());

  if (auto* FD = llvm::dyn_cast_or_null<clang::FunctionTemplateDecl>(D))
    return INTEROP_RETURN((FD->getTemplatedDecl())->getNumParams());

  return INTEROP_RETURN(0);
}

TCppIndex_t GetFunctionRequiredArgs(TCppConstFunction_t func) {
  INTEROP_TRACE(func);
  const auto* D = static_cast<const clang::Decl*>(func);
  if (auto* FD = llvm::dyn_cast_or_null<FunctionDecl>(D))
    return INTEROP_RETURN(FD->getMinRequiredArguments());

  if (auto* FD = llvm::dyn_cast_or_null<clang::FunctionTemplateDecl>(D))
    return INTEROP_RETURN((FD->getTemplatedDecl())->getMinRequiredArguments());

  return INTEROP_RETURN(0);
}

TCppType_t GetFunctionArgType(TCppFunction_t func, TCppIndex_t iarg) {
  INTEROP_TRACE(func, iarg);
  auto* D = (clang::Decl*)func;

  if (auto* FD = llvm::dyn_cast_or_null<clang::FunctionDecl>(D)) {
    if (iarg < FD->getNumParams()) {
      auto* PVD = FD->getParamDecl(iarg);
      return INTEROP_RETURN(PVD->getOriginalType().getAsOpaquePtr());
    }
  }

  return INTEROP_RETURN(nullptr);
}

std::string GetFunctionSignature(TCppFunction_t func) {
  INTEROP_TRACE(func);
  if (!func)
    return INTEROP_RETURN("<unknown>");

  auto* D = (clang::Decl*)func;
  clang::FunctionDecl* FD;

  if (llvm::dyn_cast<FunctionDecl>(D))
    FD = llvm::dyn_cast<FunctionDecl>(D);
  else if (auto* FTD = llvm::dyn_cast<clang::FunctionTemplateDecl>(D))
    FD = FTD->getTemplatedDecl();
  else
    return INTEROP_RETURN("<unknown>");

  std::string Signature;
  raw_string_ostream SS(Signature);
  PrintingPolicy Policy = getASTContext().getPrintingPolicy();
  // Skip printing the body
  Policy.TerseOutput = true;
  Policy.FullyQualifiedName = true;
  Policy.SuppressDefaultTemplateArgs = false;
  FD->print(SS, Policy);
  SS.flush();
  return INTEROP_RETURN(Signature);
}

// Internal functions that are not needed outside the library are
// encompassed in an anonymous namespace as follows.
namespace {
bool IsTemplatedFunction(Decl* D) {
  return llvm::isa_and_nonnull<FunctionTemplateDecl>(D);
}

bool IsTemplateInstantiationOrSpecialization(Decl* D) {
  if (auto* FD = llvm::dyn_cast_or_null<FunctionDecl>(D)) {
    auto TK = FD->getTemplatedKind();
    return TK ==
               FunctionDecl::TemplatedKind::TK_FunctionTemplateSpecialization ||
           TK == FunctionDecl::TemplatedKind::
                     TK_DependentFunctionTemplateSpecialization ||
           TK == FunctionDecl::TemplatedKind::TK_FunctionTemplate;
  }

  return false;
}
} // namespace

bool IsFunctionDeleted(TCppConstFunction_t function) {
  INTEROP_TRACE(function);
  const auto* FD =
      cast<const FunctionDecl>(static_cast<const clang::Decl*>(function));
  return INTEROP_RETURN(FD->isDeleted());
}

bool IsTemplatedFunction(TCppFunction_t func) {
  INTEROP_TRACE(func);
  auto* D = (Decl*)func;
  return INTEROP_RETURN(IsTemplatedFunction(D) ||
                        IsTemplateInstantiationOrSpecialization(D));
}

// FIXME: This lookup is broken, and should no longer be used in favour of
// `GetClassTemplatedMethods` If the candidate set returned is =1, that means
// the template function exists and >1 means overloads
bool ExistsFunctionTemplate(const std::string& name, TCppScope_t parent) {
  INTEROP_TRACE(name, parent);
  DeclContext* Within = 0;
  if (parent) {
    auto* D = (Decl*)parent;
    Within = llvm::dyn_cast<DeclContext>(D);
  }

  auto* ND = CppInternal::utils::Lookup::Named(&getSema(), name, Within);

  if ((intptr_t)ND == (intptr_t)0)
    return INTEROP_RETURN(false);

  if ((intptr_t)ND != (intptr_t)-1)
    return INTEROP_RETURN(IsTemplatedFunction(ND) ||
                          IsTemplateInstantiationOrSpecialization(ND));

  // FIXME: Cycle through the Decls and check if there is a templated function
  return INTEROP_RETURN(true);
}

// Looks up all constructors in the current DeclContext
void LookupConstructors(const std::string& name, TCppScope_t parent,
                        std::vector<TCppFunction_t>& funcs) {
  INTEROP_TRACE(name, parent, INTEROP_OUT(funcs));
  auto* D = (Decl*)parent;

  if (auto* CXXRD = llvm::dyn_cast_or_null<CXXRecordDecl>(D)) {
    getSema().ForceDeclarationOfImplicitMembers(CXXRD);
    DeclContextLookupResult Result = getSema().LookupConstructors(CXXRD);
    // Obtaining all constructors when we intend to lookup a method under a
    // scope can lead to crashes. We avoid that by accumulating constructors
    // only if the Decl matches the lookup name.
    for (auto* i : Result)
      if (GetName(i) == name)
        funcs.push_back(i);
  }
  return INTEROP_VOID_RETURN();
}

bool GetClassTemplatedMethods(const std::string& name, TCppScope_t parent,
                              std::vector<TCppFunction_t>& funcs) {
  INTEROP_TRACE(name, parent, INTEROP_OUT(funcs));
  auto* D = (Decl*)parent;
  if (!D && name.empty())
    return INTEROP_RETURN(false);

  // Accumulate constructors
  LookupConstructors(name, parent, funcs);
  auto& S = getSema();
  D = GetUnderlyingScope(D);
  llvm::StringRef Name(name);
  DeclarationName DName = &getASTContext().Idents.get(name);
  clang::LookupResult R(S, DName, SourceLocation(), Sema::LookupOrdinaryName,
                        RedeclarationKind::ForVisibleRedeclaration);
  auto* DC = clang::Decl::castToDeclContext(D);
  CppInternal::utils::Lookup::Named(&S, R, DC);

  if (R.getResultKind() == clang_LookupResult_Not_Found && funcs.empty())
    return INTEROP_RETURN(false);

  // Distinct match, single Decl
  else if (R.getResultKind() == clang_LookupResult_Found) {
    if (IsTemplatedFunction(R.getFoundDecl()))
      funcs.push_back(R.getFoundDecl());
  }
  // Loop over overload set
  else if (R.getResultKind() == clang_LookupResult_Found_Overloaded) {
    for (auto* Found : R)
      if (IsTemplatedFunction(Found))
        funcs.push_back(Found);
  }

  // TODO: Handle ambiguously found LookupResult
  // else if (R.getResultKind() == clang::LookupResult::Ambiguous) {
  //  auto kind = R.getAmbiguityKind();
  //  ...
  //  Produce a diagnostic describing the ambiguity that resulted
  //  from name lookup as done in Sema::DiagnoseAmbiguousLookup
  //
  return INTEROP_RETURN(!funcs.empty());
}

// Adapted from inner workings of Sema::BuildCallExpr
TCppFunction_t
BestOverloadFunctionMatch(const std::vector<TCppFunction_t>& candidates,
                          const std::vector<TemplateArgInfo>& explicit_types,
                          const std::vector<TemplateArgInfo>& arg_types) {
  INTEROP_TRACE(candidates, explicit_types, arg_types);
  auto& S = getSema();
  auto& C = S.getASTContext();

  compat::SynthesizingCodeRAII RAII(&getInterp());

  // The overload resolution interfaces in Sema require a list of expressions.
  // However, unlike handwritten C++, we do not always have a expression.
  // Here we synthesize a placeholder expression to be able to use
  // Sema::AddOverloadCandidate. Made up expressions are fine because the
  // interface uses the list size and the expression types.
  struct WrapperExpr : public OpaqueValueExpr {
    WrapperExpr() : OpaqueValueExpr(clang::Stmt::EmptyShell()) {}
  };
  auto* Exprs = new WrapperExpr[arg_types.size()];
  llvm::SmallVector<Expr*> Args;
  Args.reserve(arg_types.size());
  size_t idx = 0;
  for (auto i : arg_types) {
    QualType Type = QualType::getFromOpaquePtr(i.m_Type);
    // XValue is an object that can be "moved" whereas PRValue is temporary
    // value. This enables overloads that require the object to be moved
    ExprValueKind ExprKind = ExprValueKind::VK_XValue;
    if (Type->isLValueReferenceType())
      ExprKind = ExprValueKind::VK_LValue;

    new (&Exprs[idx]) OpaqueValueExpr(SourceLocation::getFromRawEncoding(1),
                                      Type.getNonReferenceType(), ExprKind);
    Args.push_back(&Exprs[idx]);
    ++idx;
  }

  // Create a list of template arguments.
  llvm::SmallVector<TemplateArgument> TemplateArgs;
  TemplateArgs.reserve(explicit_types.size());
  for (auto explicit_type : explicit_types) {
    QualType ArgTy = QualType::getFromOpaquePtr(explicit_type.m_Type);
    if (explicit_type.m_IntegralValue) {
      // We have a non-type template parameter. Create an integral value from
      // the string representation.
      auto Res = llvm::APSInt(explicit_type.m_IntegralValue);
      Res = Res.extOrTrunc(C.getIntWidth(ArgTy));
      TemplateArgs.push_back(TemplateArgument(C, Res, ArgTy));
    } else {
      TemplateArgs.push_back(ArgTy);
    }
  }

  TemplateArgumentListInfo ExplicitTemplateArgs{};
  for (auto TA : TemplateArgs)
    ExplicitTemplateArgs.addArgument(
        S.getTrivialTemplateArgumentLoc(TA, QualType(), SourceLocation()));

  OverloadCandidateSet Overloads(
      SourceLocation(), OverloadCandidateSet::CandidateSetKind::CSK_Normal);

  for (void* i : candidates) {
    Decl* D = static_cast<Decl*>(i);
    if (auto* FD = dyn_cast<FunctionDecl>(D)) {
      S.AddOverloadCandidate(FD, DeclAccessPair::make(FD, FD->getAccess()),
                             Args, Overloads);
    } else if (auto* FTD = dyn_cast<FunctionTemplateDecl>(D)) {
      // AddTemplateOverloadCandidate is causing a memory leak
      // It is a known bug at clang
      // call stack: AddTemplateOverloadCandidate -> MakeDeductionFailureInfo
      // source:
      // https://github.com/llvm/llvm-project/blob/release/19.x/clang/lib/Sema/SemaOverload.cpp#L731-L756
      S.AddTemplateOverloadCandidate(
          FTD, DeclAccessPair::make(FTD, FTD->getAccess()),
          &ExplicitTemplateArgs, Args, Overloads);
    }
  }

  OverloadCandidateSet::iterator Best;
  Overloads.BestViableFunction(S, SourceLocation(), Best);

  FunctionDecl* Result = Best != Overloads.end() ? Best->Function : nullptr;
  delete[] Exprs;
  return INTEROP_RETURN(Result);
}

// Gets the AccessSpecifier of the function and checks if it is equal to
// the provided AccessSpecifier.
bool CheckMethodAccess(TCppFunction_t method, AccessSpecifier AS) {
  auto* D = (Decl*)method;
  if (auto* CXXMD = llvm::dyn_cast_or_null<CXXMethodDecl>(D)) {
    return CXXMD->getAccess() == AS;
  }

  return false;
}

bool IsMethod(TCppConstFunction_t method) {
  INTEROP_TRACE(method);
  return INTEROP_RETURN(
      dyn_cast_or_null<CXXMethodDecl>(static_cast<const clang::Decl*>(method)));
}

bool IsPublicMethod(TCppFunction_t method) {
  INTEROP_TRACE(method);
  return INTEROP_RETURN(CheckMethodAccess(method, AccessSpecifier::AS_public));
}

bool IsProtectedMethod(TCppFunction_t method) {
  INTEROP_TRACE(method);
  return INTEROP_RETURN(
      CheckMethodAccess(method, AccessSpecifier::AS_protected));
}

bool IsPrivateMethod(TCppFunction_t method) {
  INTEROP_TRACE(method);
  return INTEROP_RETURN(CheckMethodAccess(method, AccessSpecifier::AS_private));
}

bool IsConstructor(TCppConstFunction_t method) {
  INTEROP_TRACE(method);
  const auto* D = static_cast<const Decl*>(method);
  if (const auto* FTD = dyn_cast<FunctionTemplateDecl>(D))
    return INTEROP_RETURN(IsConstructor(FTD->getTemplatedDecl()));
  return INTEROP_RETURN(llvm::isa_and_nonnull<CXXConstructorDecl>(D));
}

bool IsDestructor(TCppConstFunction_t method) {
  INTEROP_TRACE(method);
  const auto* D = static_cast<const Decl*>(method);
  return INTEROP_RETURN(llvm::isa_and_nonnull<CXXDestructorDecl>(D));
}

bool IsStaticMethod(TCppConstFunction_t method) {
  INTEROP_TRACE(method);
  const auto* D = static_cast<const Decl*>(method);
  if (auto* CXXMD = llvm::dyn_cast_or_null<CXXMethodDecl>(D)) {
    return INTEROP_RETURN(CXXMD->isStatic());
  }

  return INTEROP_RETURN(false);
}

bool IsExplicit(TCppConstFunction_t method) {
  INTEROP_TRACE(method);
  if (!method)
    return INTEROP_RETURN(false);

  const auto* D = static_cast<const Decl*>(method);

  if (const auto* FTD = llvm::dyn_cast_or_null<FunctionTemplateDecl>(D))
    D = FTD->getTemplatedDecl();

  if (const auto* CD = llvm::dyn_cast_or_null<CXXConstructorDecl>(D))
    return INTEROP_RETURN(CD->isExplicit());

  if (const auto* CD = llvm::dyn_cast_or_null<CXXConversionDecl>(D))
    return INTEROP_RETURN(CD->isExplicit());

  if (const auto* DGD = llvm::dyn_cast_or_null<CXXDeductionGuideDecl>(D))
    return INTEROP_RETURN(DGD->isExplicit());

  return INTEROP_RETURN(false);
}

TCppFuncAddr_t GetFunctionAddress(const char* mangled_name) {
  INTEROP_TRACE(mangled_name);
  auto& I = getInterp();
  auto FDAorErr = compat::getSymbolAddress(I, mangled_name);
  if (llvm::Error Err = FDAorErr.takeError())
    llvm::consumeError(std::move(Err)); // nullptr if missing
  else
    return INTEROP_RETURN(llvm::jitTargetAddressToPointer<void*>(*FDAorErr));

  return INTEROP_RETURN(nullptr);
}

static TCppFuncAddr_t GetFunctionAddress(const FunctionDecl* FD) {
  const auto get_mangled_name = [](const FunctionDecl* FD) {
    auto MangleCtxt = getASTContext().createMangleContext();

    if (!MangleCtxt->shouldMangleDeclName(FD)) {
      return FD->getNameInfo().getName().getAsString();
    }

    std::string mangled_name;
    llvm::raw_string_ostream ostream(mangled_name);

    MangleCtxt->mangleName(FD, ostream);

    ostream.flush();
    delete MangleCtxt;

    return mangled_name;
  };

  // Constructor and Destructors needs to be handled differently
  if (!llvm::isa<CXXConstructorDecl>(FD) && !llvm::isa<CXXDestructorDecl>(FD))
    return GetFunctionAddress(get_mangled_name(FD).c_str());

  return 0;
}

TCppFuncAddr_t GetFunctionAddress(TCppFunction_t method) {
  INTEROP_TRACE(method);
  auto* D = static_cast<Decl*>(method);
  if (auto* FD = llvm::dyn_cast_or_null<FunctionDecl>(D)) {
    if ((IsTemplateInstantiationOrSpecialization(FD) ||
         FD->getTemplatedKind() == FunctionDecl::TK_MemberSpecialization) &&
        !FD->getDefinition())
      InstantiateFunctionDefinition(D);
    ASTContext& C = getASTContext();
    if (isDiscardableGVALinkage(C.GetGVALinkageForFunction(FD)))
      ForceCodeGen(FD, getInterp());
    return INTEROP_RETURN(GetFunctionAddress(FD));
  }
  return INTEROP_RETURN(nullptr);
}

bool IsVirtualMethod(TCppFunction_t method) {
  INTEROP_TRACE(method);
  auto* D = (Decl*)method;
  if (auto* CXXMD = llvm::dyn_cast_or_null<CXXMethodDecl>(D)) {
    return INTEROP_RETURN(CXXMD->isVirtual());
  }

  return INTEROP_RETURN(false);
}

void GetDatamembers(TCppScope_t scope, std::vector<TCppScope_t>& datamembers) {
  INTEROP_TRACE(scope, INTEROP_OUT(datamembers));
  auto* D = (Decl*)scope;

  if (auto* CXXRD = llvm::dyn_cast_or_null<CXXRecordDecl>(D)) {
    getSema().ForceDeclarationOfImplicitMembers(CXXRD);
    if (CXXRD->hasDefinition())
      CXXRD = CXXRD->getDefinition();

    llvm::SmallVector<RecordDecl::decl_iterator, 2> stack_begin;
    llvm::SmallVector<RecordDecl::decl_iterator, 2> stack_end;
    stack_begin.push_back(CXXRD->decls_begin());
    stack_end.push_back(CXXRD->decls_end());
    while (!stack_begin.empty()) {
      if (stack_begin.back() == stack_end.back()) {
        stack_begin.pop_back();
        stack_end.pop_back();
        continue;
      }
      Decl* D = *(stack_begin.back());
      if (auto* FD = llvm::dyn_cast<FieldDecl>(D)) {
        if (FD->isAnonymousStructOrUnion()) {
          if (const auto* RT = FD->getType()->getAs<RecordType>()) {
            if (auto* CXXRD = llvm::dyn_cast<CXXRecordDecl>(RT->getDecl())) {
              stack_begin.back()++;
              stack_begin.push_back(CXXRD->decls_begin());
              stack_end.push_back(CXXRD->decls_end());
              continue;
            }
          }
        }
        datamembers.push_back((TCppScope_t)D);

      } else if (auto* USD = llvm::dyn_cast<UsingShadowDecl>(D)) {
        if (llvm::isa<FieldDecl>(USD->getTargetDecl()))
          datamembers.push_back(USD);
      }
      stack_begin.back()++;
    }
  }
  return INTEROP_VOID_RETURN();
}

void GetStaticDatamembers(TCppScope_t scope,
                          std::vector<TCppScope_t>& datamembers) {
  INTEROP_TRACE(scope, INTEROP_OUT(datamembers));
  GetClassDecls<VarDecl>(scope, datamembers);
  return INTEROP_VOID_RETURN();
}

void GetEnumConstantDatamembers(TCppScope_t scope,
                                std::vector<TCppScope_t>& datamembers,
                                bool include_enum_class) {
  INTEROP_TRACE(scope, INTEROP_OUT(datamembers), include_enum_class);
  std::vector<TCppScope_t> EDs;
  GetClassDecls<EnumDecl>(scope, EDs);
  for (TCppScope_t i : EDs) {
    auto* ED = static_cast<EnumDecl*>(i);

    bool is_class_tagged = ED->isScopedUsingClassTag();
    if (is_class_tagged && !include_enum_class)
      continue;

    std::copy(ED->enumerator_begin(), ED->enumerator_end(),
              std::back_inserter(datamembers));
  }
  return INTEROP_VOID_RETURN();
}

TCppScope_t LookupDatamember(const std::string& name, TCppScope_t parent) {
  INTEROP_TRACE(name, parent);
  clang::DeclContext* Within = 0;
  if (parent) {
    auto* D = (clang::Decl*)parent;
    Within = llvm::dyn_cast<clang::DeclContext>(D);
  }

  auto* ND = CppInternal::utils::Lookup::Named(&getSema(), name, Within);
  if (ND && ND != (clang::NamedDecl*)-1) {
    if (llvm::isa_and_nonnull<clang::FieldDecl>(ND)) {
      return INTEROP_RETURN((TCppScope_t)ND);
    }
  }

  return INTEROP_RETURN(nullptr);
}

bool IsLambdaClass(TCppType_t type) {
  INTEROP_TRACE(type);
  QualType QT = QualType::getFromOpaquePtr(type);
  if (auto* CXXRD = QT->getAsCXXRecordDecl()) {
    return INTEROP_RETURN(CXXRD->isLambda());
  }
  return INTEROP_RETURN(false);
}

TCppType_t GetVariableType(TCppScope_t var) {
  INTEROP_TRACE(var);
  auto* D = static_cast<Decl*>(var);

  if (auto DD = llvm::dyn_cast_or_null<DeclaratorDecl>(D)) {
    QualType QT = DD->getType();

    // Check if the type is a typedef type
    if (QT->isTypedefNameType()) {
      return INTEROP_RETURN(QT.getAsOpaquePtr());
    }

    // Else, return the canonical type
    QT = QT.getCanonicalType();
    return INTEROP_RETURN(QT.getAsOpaquePtr());
  }

  if (auto* ECD = llvm::dyn_cast_or_null<EnumConstantDecl>(D))
    return INTEROP_RETURN(ECD->getType().getAsOpaquePtr());

  return INTEROP_RETURN(nullptr);
}

intptr_t GetVariableOffset(compat::Interpreter& I, Decl* D,
                           CXXRecordDecl* BaseCXXRD) {
  if (!D)
    return 0;

  auto& C = I.getSema().getASTContext();

  if (auto* FD = llvm::dyn_cast<FieldDecl>(D)) {
    clang::RecordDecl* FieldParentRecordDecl = FD->getParent();
    intptr_t offset = C.toCharUnitsFromBits(C.getFieldOffset(FD)).getQuantity();
    while (FieldParentRecordDecl->isAnonymousStructOrUnion()) {
      clang::RecordDecl* anon = FieldParentRecordDecl;
      FieldParentRecordDecl = llvm::dyn_cast<RecordDecl>(anon->getParent());
      for (auto F = FieldParentRecordDecl->field_begin();
           F != FieldParentRecordDecl->field_end(); ++F) {
        const auto* RT = F->getType()->getAs<RecordType>();
        if (!RT)
          continue;
        if (anon == RT->getDecl()) {
          FD = *F;
          break;
        }
      }
      offset += C.toCharUnitsFromBits(C.getFieldOffset(FD)).getQuantity();
    }
    if (BaseCXXRD && BaseCXXRD != FieldParentRecordDecl->getCanonicalDecl()) {
      // FieldDecl FD belongs to some class C, but the base class BaseCXXRD is
      // not C. That means BaseCXXRD derives from C. Offset needs to be
      // calculated for Derived class

      // Depth first Search is performed to the class that declares FD from
      // the base class
      std::vector<CXXRecordDecl*> stack;
      std::map<CXXRecordDecl*, CXXRecordDecl*> direction;
      stack.push_back(BaseCXXRD);
      while (!stack.empty()) {
        CXXRecordDecl* RD = stack.back();
        stack.pop_back();
        size_t num_bases = GetNumBases(RD);
        bool flag = false;
        for (size_t i = 0; i < num_bases; i++) {
          auto* CRD = static_cast<CXXRecordDecl*>(GetBaseClass(RD, i));
          direction[CRD] = RD;
          if (CRD == FieldParentRecordDecl) {
            flag = true;
            break;
          }
          stack.push_back(CRD);
        }
        if (flag)
          break;
      }
      if (auto* RD = llvm::dyn_cast<CXXRecordDecl>(FieldParentRecordDecl)) {
        // add in the offsets for the (multi level) base classes
        RD = RD->getCanonicalDecl();
        while (BaseCXXRD != RD) {
          CXXRecordDecl* Parent = direction.at(RD);
          offset +=
              C.getASTRecordLayout(Parent).getBaseClassOffset(RD).getQuantity();
          RD = Parent;
        }
      } else {
        assert(false && "Unreachable");
      }
    }
    return offset;
  }

  if (auto* VD = llvm::dyn_cast<VarDecl>(D)) {
    auto GD = GlobalDecl(VD);
    std::string mangledName;
    compat::maybeMangleDeclName(GD, mangledName);
    void* address = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(
        mangledName.c_str());

    if (!address)
      address = I.getAddressOfGlobal(GD);
    if (!address) {
      if (!VD->hasInit()) {
        compat::SynthesizingCodeRAII RAII(&getInterp());
        getSema().InstantiateVariableDefinition(SourceLocation(), VD);
        VD = VD->getDefinition();
      }
      if (VD->hasInit() &&
          (VD->isConstexpr() || VD->getType().isConstQualified())) {
        if (const APValue* val = VD->evaluateValue()) {
          if (VD->getType()->isIntegralType(C)) {
            return (intptr_t)val->getInt().getRawData();
          }
        }
      }
    }
    if (!address) {
      auto Linkage = C.GetGVALinkageForVariable(VD);
      if (isDiscardableGVALinkage(Linkage))
        ForceCodeGen(VD, I);
    }
    auto VDAorErr = compat::getSymbolAddress(I, StringRef(mangledName));
    if (!VDAorErr) {
      llvm::logAllUnhandledErrors(VDAorErr.takeError(), llvm::errs(),
                                  "Failed to GetVariableOffset:");
      return 0;
    }
    return (intptr_t)jitTargetAddressToPointer<void*>(VDAorErr.get());
  }

  return 0;
}

intptr_t GetVariableOffset(TCppScope_t var, TCppScope_t parent) {
  INTEROP_TRACE(var, parent);
  auto* D = static_cast<Decl*>(var);
  auto* RD = llvm::dyn_cast_or_null<CXXRecordDecl>(static_cast<Decl*>(parent));
  return INTEROP_RETURN(GetVariableOffset(getInterp(), D, RD));
}

// Check if the Access Specifier of the variable matches the provided value.
bool CheckVariableAccess(TCppScope_t var, AccessSpecifier AS) {
  auto* D = (Decl*)var;
  return D->getAccess() == AS;
}

bool IsPublicVariable(TCppScope_t var) {
  INTEROP_TRACE(var);
  return INTEROP_RETURN(CheckVariableAccess(var, AccessSpecifier::AS_public));
}

bool IsProtectedVariable(TCppScope_t var) {
  INTEROP_TRACE(var);
  return INTEROP_RETURN(
      CheckVariableAccess(var, AccessSpecifier::AS_protected));
}

bool IsPrivateVariable(TCppScope_t var) {
  INTEROP_TRACE(var);
  return INTEROP_RETURN(CheckVariableAccess(var, AccessSpecifier::AS_private));
}

bool IsStaticVariable(TCppScope_t var) {
  INTEROP_TRACE(var);
  auto* D = (Decl*)var;
  if (llvm::isa_and_nonnull<VarDecl>(D)) {
    return INTEROP_RETURN(true);
  }

  return INTEROP_RETURN(false);
}

bool IsConstVariable(TCppScope_t var) {
  INTEROP_TRACE(var);
  auto* D = (clang::Decl*)var;

  if (auto* VD = llvm::dyn_cast_or_null<ValueDecl>(D)) {
    return INTEROP_RETURN(VD->getType().isConstQualified());
  }

  return INTEROP_RETURN(false);
}

bool IsRecordType(TCppType_t type) {
  INTEROP_TRACE(type);
  QualType QT = QualType::getFromOpaquePtr(type);
  return INTEROP_RETURN(QT->isRecordType());
}

bool IsPODType(TCppType_t type) {
  INTEROP_TRACE(type);
  QualType QT = QualType::getFromOpaquePtr(type);

  if (QT.isNull())
    return INTEROP_RETURN(false);

  return INTEROP_RETURN(QT.isPODType(getASTContext()));
}

bool IsIntegerType(TCppType_t type, Signedness* s) {
  INTEROP_TRACE(type, s);
  if (!type)
    return INTEROP_RETURN(false);
  QualType QT = QualType::getFromOpaquePtr(type);
  if (!QT->hasIntegerRepresentation())
    return INTEROP_RETURN(false);
  if (s) {
    *s = QT->hasSignedIntegerRepresentation() ? Signedness::kSigned
                                              : Signedness::kUnsigned;
  }
  return INTEROP_RETURN(true);
}

bool IsFloatingType(TCppType_t type) {
  INTEROP_TRACE(type);
  if (!type)
    return INTEROP_RETURN(false);
  QualType QT = QualType::getFromOpaquePtr(type);
  return INTEROP_RETURN(QT->hasFloatingRepresentation());
}

bool IsSameType(TCppType_t type_a, TCppType_t type_b) {
  INTEROP_TRACE(type_a, type_b);
  if (!type_a || !type_b)
    return INTEROP_RETURN(false);
  QualType QT1 = QualType::getFromOpaquePtr(type_a);
  QualType QT2 = QualType::getFromOpaquePtr(type_b);
  return INTEROP_RETURN(getASTContext().hasSameType(QT1, QT2));
}

bool IsPointerType(TCppType_t type) {
  INTEROP_TRACE(type);
  QualType QT = QualType::getFromOpaquePtr(type);
  return INTEROP_RETURN(QT->isPointerType());
}

bool IsVoidPointerType(TCppType_t type) {
  INTEROP_TRACE(type);
  if (!type)
    return INTEROP_RETURN(false);
  QualType QT = QualType::getFromOpaquePtr(type);
  return INTEROP_RETURN(QT->isVoidPointerType());
}

TCppType_t GetPointeeType(TCppType_t type) {
  INTEROP_TRACE(type);
  if (!IsPointerType(type))
    return INTEROP_RETURN(nullptr);
  QualType QT = QualType::getFromOpaquePtr(type);
  return INTEROP_RETURN(QT->getPointeeType().getAsOpaquePtr());
}

bool IsReferenceType(TCppType_t type) {
  INTEROP_TRACE(type);
  QualType QT = QualType::getFromOpaquePtr(type);
  return INTEROP_RETURN(QT->isReferenceType());
}

ValueKind GetValueKind(TCppType_t type) {
  INTEROP_TRACE(type);
  QualType QT = QualType::getFromOpaquePtr(type);
  if (QT->isRValueReferenceType())
    return INTEROP_RETURN(ValueKind::RValue);
  if (QT->isLValueReferenceType())
    return INTEROP_RETURN(ValueKind::LValue);
  return INTEROP_RETURN(ValueKind::None);
}

TCppType_t GetPointerType(TCppType_t type) {
  INTEROP_TRACE(type);
  QualType QT = QualType::getFromOpaquePtr(type);
  return INTEROP_RETURN(getASTContext().getPointerType(QT).getAsOpaquePtr());
}

TCppType_t GetReferencedType(TCppType_t type, bool rvalue) {
  INTEROP_TRACE(type, rvalue);
  QualType QT = QualType::getFromOpaquePtr(type);
  if (rvalue)
    return INTEROP_RETURN(
        getASTContext().getRValueReferenceType(QT).getAsOpaquePtr());
  return INTEROP_RETURN(
      getASTContext().getLValueReferenceType(QT).getAsOpaquePtr());
}

TCppType_t GetNonReferenceType(TCppType_t type) {
  INTEROP_TRACE(type);
  if (!IsReferenceType(type))
    return INTEROP_RETURN(nullptr);
  QualType QT = QualType::getFromOpaquePtr(type);
  return INTEROP_RETURN(QT.getNonReferenceType().getAsOpaquePtr());
}

TCppType_t GetUnderlyingType(TCppType_t type) {
  INTEROP_TRACE(type);
  if (!type)
    return INTEROP_RETURN(nullptr);
  QualType QT = QualType::getFromOpaquePtr(type);
  QT = QT->getCanonicalTypeUnqualified();

  // Recursively remove array dimensions
  while (QT->isArrayType())
    QT = QualType(QT->getArrayElementTypeNoTypeQual(), 0);

  // Recursively reduce pointer depth till we are left with a pointerless
  // type.
  for (auto PT = QT->getPointeeType(); !PT.isNull();
       PT = QT->getPointeeType()) {
    QT = PT;
  }
  QT = QT->getCanonicalTypeUnqualified();
  return INTEROP_RETURN(QT.getAsOpaquePtr());
}

std::string GetTypeAsString(TCppType_t var) {
  INTEROP_TRACE(var);
  QualType QT = QualType::getFromOpaquePtr(var);
  PrintingPolicy Policy(getASTContext().getPrintingPolicy());
  Policy.Bool = true;               // Print bool instead of _Bool.
  Policy.SuppressTagKeyword = true; // Do not print `class std::string`.
  Policy.Suppress_Elab = true;
  Policy.FullyQualifiedName = true;
  return INTEROP_RETURN(QT.getAsString(Policy));
}

TCppType_t GetCanonicalType(TCppType_t type) {
  INTEROP_TRACE(type);
  if (!type)
    return INTEROP_RETURN(nullptr);
  QualType QT = QualType::getFromOpaquePtr(type);
  return INTEROP_RETURN(QT.getCanonicalType().getAsOpaquePtr());
}

bool HasTypeQualifier(TCppType_t type, QualKind qual) {
  INTEROP_TRACE(type, qual);
  if (!type)
    return INTEROP_RETURN(false);

  QualType QT = QualType::getFromOpaquePtr(type);
  if (qual & QualKind::Const) {
    if (!QT.isConstQualified())
      return INTEROP_RETURN(false);
  }
  if (qual & QualKind::Volatile) {
    if (!QT.isVolatileQualified())
      return INTEROP_RETURN(false);
  }
  if (qual & QualKind::Restrict) {
    if (!QT.isRestrictQualified())
      return INTEROP_RETURN(false);
  }
  return INTEROP_RETURN(true);
}

TCppType_t RemoveTypeQualifier(TCppType_t type, QualKind qual) {
  INTEROP_TRACE(type, qual);
  if (!type)
    return INTEROP_RETURN(type);

  auto QT = QualType(QualType::getFromOpaquePtr(type));
  if (qual & QualKind::Const)
    QT.removeLocalConst();
  if (qual & QualKind::Volatile)
    QT.removeLocalVolatile();
  if (qual & QualKind::Restrict)
    QT.removeLocalRestrict();
  return INTEROP_RETURN(QT.getAsOpaquePtr());
}

TCppType_t AddTypeQualifier(TCppType_t type, QualKind qual) {
  INTEROP_TRACE(type, qual);
  if (!type)
    return INTEROP_RETURN(type);

  auto QT = QualType(QualType::getFromOpaquePtr(type));
  if (qual & QualKind::Const) {
    if (!QT.isConstQualified())
      QT.addConst();
  }
  if (qual & QualKind::Volatile) {
    if (!QT.isVolatileQualified())
      QT.addVolatile();
  }
  if (qual & QualKind::Restrict) {
    if (!QT.isRestrictQualified())
      QT.addRestrict();
  }
  return INTEROP_RETURN(QT.getAsOpaquePtr());
}

// Registers all permutations of a word set
static void RegisterPerms(llvm::StringMap<QualType>& Map, QualType QT,
                          llvm::SmallVectorImpl<llvm::StringRef>& Words) {
  std::sort(Words.begin(), Words.end());
  do {
    std::string Key;
    for (size_t i = 0; i < Words.size(); ++i) {
      if (i > 0)
        Key += ' ';
      Key += Words[i].str();
    }
    Map[Key] = QT;
  } while (std::next_permutation(Words.begin(), Words.end()));
}
ALLOW_ACCESS(ASTContext, Types, llvm::SmallVector<clang::Type*, 0>);
static void PopulateBuiltinMap(ASTContext& Context) {
  const PrintingPolicy Policy(Context.getLangOpts());
  auto& BuiltinMap = GetInterpreters().back().BuiltinMap;
  const auto& Types = ACCESS(Context, Types);

  for (clang::Type* T : Types) {
    auto* BT = llvm::dyn_cast<BuiltinType>(T);
    if (!BT || BT->isPlaceholderType())
      continue;

    QualType QT(BT, 0);
    std::string Name = QT.getAsString(Policy);
    if (Name.empty() || Name[0] == '<')
      continue;

    // Initial entry (e.g., "int", "unsigned long")
    BuiltinMap[Name] = QT;

    llvm::SmallVector<llvm::StringRef, 4> Words;
    llvm::StringRef(Name).split(Words, ' ', -1, false);

    bool hasInt = false;
    bool hasSigned = false;
    bool hasUnsigned = false;
    bool hasChar = false;
    bool isModifiable = false;

    for (auto W : Words) {
      if (W == "int")
        hasInt = true;
      else if (W == "signed")
        hasSigned = true;
      else if (W == "unsigned")
        hasUnsigned = true;
      else if (W == "char")
        hasChar = true;

      if (W == "long" || W == "short" || hasInt)
        isModifiable = true;
    }

    // Skip things like 'float' or 'double' that aren't combined
    if (!isModifiable && !hasUnsigned && !hasSigned)
      continue;

    // Register base permutations (e.g., "long long" or "unsigned int")
    if (Words.size() > 1)
      RegisterPerms(BuiltinMap, QT, Words);

    // Expansion: Add "int" suffix where missing (e.g., "short" -> "short int")
    if (!hasInt && !hasChar) {
      auto WithInt = Words;
      WithInt.push_back("int");
      RegisterPerms(BuiltinMap, QT, WithInt);

      // If we are adding 'int', we should also try adding 'signed'
      // to cover cases like "short" -> "signed short int"
      if (!hasSigned && !hasUnsigned) {
        auto WithBoth = WithInt;
        WithBoth.push_back("signed");
        RegisterPerms(BuiltinMap, QT, WithBoth);
      }
    }

    // Expansion: Add "signed" prefix
    // (e.g., "int" -> "signed int", "long" -> "signed long")
    if (!hasSigned && !hasUnsigned) {
      auto WithSigned = Words;
      WithSigned.push_back("signed");
      RegisterPerms(BuiltinMap, QT, WithSigned);
    }
  }

  // Explicit global synonym
  BuiltinMap["signed"] = Context.IntTy;
  BuiltinMap["unsigned"] = Context.UnsignedIntTy;
}
static QualType findBuiltinType(llvm::StringRef typeName, ASTContext& Context) {
  llvm::StringMap<QualType>& BuiltinMap = GetInterpreters().back().BuiltinMap;
  if (BuiltinMap.empty())
    PopulateBuiltinMap(Context);

  // Fast Lookup
  auto It = BuiltinMap.find(typeName);
  if (It != BuiltinMap.end())
    return It->second;

  return QualType(); // Return null if not a builtin
}
static std::optional<QualType> GetTypeInternal(Decl* D) {
  if (!D)
    return {};
  // Even though typedefs derive from TypeDecl, their getTypeForDecl()
  // returns a nullptr.
  if (const auto* TND = llvm::dyn_cast_or_null<TypedefNameDecl>(D))
    return TND->getUnderlyingType();

  if (auto* VD = dyn_cast<ValueDecl>(D))
    return VD->getType();

  if (const auto* TD = llvm::dyn_cast_or_null<TypeDecl>(D))
#if CLANG_VERSION_MAJOR < 22
    return QualType(TD->getTypeForDecl(), 0);
#else
    return getASTContext().getTypeDeclType(TD);
#endif

  return {};
}

TCppType_t GetType(const std::string& name) {
  INTEROP_TRACE(name);
  QualType builtin = findBuiltinType(name, getASTContext());
  if (!builtin.isNull())
    return INTEROP_RETURN(builtin.getAsOpaquePtr());

  return INTEROP_RETURN(GetTypeFromScope(GetNamed(name, /*parent=*/nullptr)));
}

TCppType_t GetComplexType(TCppType_t type) {
  INTEROP_TRACE(type);
  QualType QT = QualType::getFromOpaquePtr(type);

  return INTEROP_RETURN(getASTContext().getComplexType(QT).getAsOpaquePtr());
}

TCppType_t GetTypeFromScope(TCppScope_t klass) {
  INTEROP_TRACE(klass);
  if (!klass)
    return INTEROP_RETURN(nullptr);

  if (auto QT = GetTypeInternal((Decl*)klass))
    return INTEROP_RETURN(QT->getAsOpaquePtr());

  return INTEROP_RETURN(nullptr);
}

// Internal functions that are not needed outside the library are
// encompassed in an anonymous namespace as follows.
namespace {
static unsigned long long gWrapperSerial = 0LL;

enum EReferenceType { kNotReference, kLValueReference, kRValueReference };

// Start of JitCall Helper Functions

#define DEBUG_TYPE "jitcall"

// FIXME: Use that routine throughout CallFunc's port in places such as
// make_narg_call.
inline void indent(std::ostringstream& buf, int indent_level) {
  static const std::string kIndentString("   ");
  for (int i = 0; i < indent_level; ++i)
    buf << kIndentString;
}

void* compile_wrapper(compat::Interpreter& I, const std::string& wrapper_name,
                      const std::string& wrapper,
                      bool withAccessControl = true) {
  LLVM_DEBUG(dbgs() << "Compiling '" << wrapper_name << "'\n");
  return I.compileFunction(wrapper_name, wrapper, false /*ifUnique*/,
                           withAccessControl);
}

void get_type_as_string(QualType QT, std::string& type_name, ASTContext& C,
                        PrintingPolicy Policy) {
  // TODO: Implement cling desugaring from utils::AST
  //       cling::utils::Transform::GetPartiallyDesugaredType()
  if (!QT->isTypedefNameType() || QT->isBuiltinType())
    QT = QT.getDesugaredType(C);
  Policy.Suppress_Elab = true;
  Policy.SuppressTagKeyword = !QT->isEnumeralType();
  Policy.FullyQualifiedName = true;
  Policy.UsePreferredNames = false;
  QT.getAsStringInternal(type_name, Policy);
}

static void GetDeclName(const clang::Decl* D, ASTContext& Context,
                        std::string& name) {
  // Helper to extract a fully qualified name from a Decl
  PrintingPolicy Policy(Context.getPrintingPolicy());
  Policy.SuppressTagKeyword = true;
  Policy.SuppressUnwrittenScope = true;
  Policy.Print_Canonical_Types = true;
  if (const TypeDecl* TD = dyn_cast<TypeDecl>(D)) {
    // This is a class, struct, or union member.
    QualType QT;
    if (const TypedefDecl* Typedef = dyn_cast<const TypedefDecl>(TD)) {
      // Handle the typedefs to anonymous types.
      QT = Typedef->getTypeSourceInfo()->getType();
    } else
#if CLANG_VERSION_MAJOR < 22
      QT = {TD->getTypeForDecl(), 0};
#else
      QT = TD->getASTContext().getTypeDeclType(TD);
#endif
    get_type_as_string(QT, name, Context, Policy);
  } else if (const NamedDecl* ND = dyn_cast<NamedDecl>(D)) {
    // This is a namespace member.
    raw_string_ostream stream(name);
    ND->getNameForDiagnostic(stream, Policy, /*Qualified=*/true);
    stream.flush();
  }
}

void collect_type_info(const FunctionDecl* FD, QualType& QT,
                       std::ostringstream& typedefbuf,
                       std::ostringstream& callbuf, std::string& type_name,
                       EReferenceType& refType, bool& isPointer,
                       int indent_level, bool forArgument) {
  //
  //  Collect information about the type of a function parameter
  //  needed for building the wrapper function.
  //
  ASTContext& C = FD->getASTContext();
  PrintingPolicy Policy(C.getPrintingPolicy());
  Policy.Suppress_Elab = true;
  refType = kNotReference;
  if (QT->isRecordType()) {
    if (forArgument) {
      get_type_as_string(QT, type_name, C, Policy);
      return;
    }
    if (auto* CXXRD = QT->getAsCXXRecordDecl()) {
      if (CXXRD->isLambda()) {
        std::string fn_name;
        llvm::raw_string_ostream stream(fn_name);
        Policy.FullyQualifiedName = true;
        Policy.SuppressUnwrittenScope = true;
        FD->getNameForDiagnostic(stream, Policy,
                                 /*Qualified=*/false);
        type_name = "__internal_CppInterOp::function<decltype(" + fn_name +
                    ")>::result_type";
        return;
      }
    }
  }
  if (QT->isFunctionPointerType()) {
    std::string fp_typedef_name;
    {
      std::ostringstream nm;
      nm << "FP" << gWrapperSerial++;
      type_name = nm.str();
      raw_string_ostream OS(fp_typedef_name);
      QT.print(OS, Policy, type_name);
      OS.flush();
    }

    indent(typedefbuf, indent_level);

    typedefbuf << "typedef " << fp_typedef_name << ";\n";
    return;
  } else if (QT->isMemberPointerType()) {
    std::string mp_typedef_name;
    {
      std::ostringstream nm;
      nm << "MP" << gWrapperSerial++;
      type_name = nm.str();
      raw_string_ostream OS(mp_typedef_name);
      QT.print(OS, Policy, type_name);
      OS.flush();
    }

    indent(typedefbuf, indent_level);

    typedefbuf << "typedef " << mp_typedef_name << ";\n";
    return;
  } else if (QT->isPointerType()) {
    isPointer = true;
    QT = cast<clang::PointerType>(QT.getCanonicalType())->getPointeeType();
  } else if (QT->isReferenceType()) {
    if (QT->isRValueReferenceType())
      refType = kRValueReference;
    else
      refType = kLValueReference;
    QT = cast<ReferenceType>(QT.getCanonicalType())->getPointeeType();
  }
  // Fall through for the array type to deal with reference/pointer ro array
  // type.
  if (QT->isArrayType()) {
    std::string ar_typedef_name;
    {
      std::ostringstream ar;
      ar << "AR" << gWrapperSerial++;
      type_name = ar.str();
      raw_string_ostream OS(ar_typedef_name);
      QT.print(OS, Policy, type_name);
      OS.flush();
    }
    indent(typedefbuf, indent_level);
    typedefbuf << "typedef " << ar_typedef_name << ";\n";
    return;
  }
  get_type_as_string(QT, type_name, C, Policy);
}

void make_narg_ctor(const FunctionDecl* FD, const unsigned N,
                    std::ostringstream& typedefbuf, std::ostringstream& callbuf,
                    const std::string& class_name, int indent_level,
                    bool array = false) {
  // Make a code string that follows this pattern:
  //
  // ClassName(args...)
  //    OR
  // ClassName[nary] // array of objects
  //

  if (array)
    callbuf << class_name << "[nary]";
  else
    callbuf << class_name;

  // We cannot pass initialization parameters if we call array new
  if (N && !array) {
    callbuf << "(";
    for (unsigned i = 0U; i < N; ++i) {
      const ParmVarDecl* PVD = FD->getParamDecl(i);
      QualType Ty = PVD->getType();
      QualType QT = Ty.getCanonicalType();
      std::string type_name;
      EReferenceType refType = kNotReference;
      bool isPointer = false;
      collect_type_info(FD, QT, typedefbuf, callbuf, type_name, refType,
                        isPointer, indent_level, true);
      if (i) {
        callbuf << ',';
        if (i % 2) {
          callbuf << ' ';
        } else {
          callbuf << "\n";
          indent(callbuf, indent_level);
        }
      }
      if (refType != kNotReference) {
        callbuf << "(" << type_name.c_str()
                << (refType == kLValueReference ? "&" : "&&") << ")*("
                << type_name.c_str() << "*)args[" << i << "]";
      } else if (isPointer) {
        callbuf << "*(" << type_name.c_str() << "**)args[" << i << "]";
      } else {
        callbuf << "*(" << type_name.c_str() << "*)args[" << i << "]";
      }
    }
    callbuf << ")";
  }
  // This can be zero or default-initialized
  else if (const auto* CD = dyn_cast<CXXConstructorDecl>(FD);
           CD && CD->isDefaultConstructor() && !array) {
    callbuf << "()";
  }
}

const DeclContext* get_non_transparent_decl_context(const FunctionDecl* FD) {
  auto* DC = FD->getDeclContext();
  while (DC->isTransparentContext()) {
    DC = DC->getParent();
    assert(DC && "All transparent contexts should have a parent!");
  }
  return DC;
}

void make_narg_call(const FunctionDecl* FD, const std::string& return_type,
                    const unsigned N, std::ostringstream& typedefbuf,
                    std::ostringstream& callbuf, const std::string& class_name,
                    int indent_level) {
  //
  // Make a code string that follows this pattern:
  //
  // ((<class>*)obj)-><method>(*(<arg-i-type>*)args[i], ...)
  //

  // Sometimes it's necessary that we cast the function we want to call
  // first to its explicit function type before calling it. This is supposed
  // to prevent that we accidentally ending up in a function that is not
  // the one we're supposed to call here (e.g. because the C++ function
  // lookup decides to take another function that better fits). This method
  // has some problems, e.g. when we call a function with default arguments
  // and we don't provide all arguments, we would fail with this pattern.
  // Same applies with member methods which seem to cause parse failures
  // even when we supply the object parameter. Therefore we only use it in
  // cases where we know it works and set this variable to true when we do.

  // true if not a overloaded operators or the overloaded operator is call
  // operator
  bool op_flag = !FD->isOverloadedOperator() ||
                 FD->getOverloadedOperator() == clang::OO_Call;

  bool ShouldCastFunction = !isa<CXXMethodDecl>(FD) &&
                            N == FD->getNumParams() && op_flag &&
                            !FD->isTemplateInstantiation();
  if (ShouldCastFunction) {
    callbuf << "(";
    callbuf << "(";
    callbuf << return_type << " (&)";
    {
      callbuf << "(";
      for (unsigned i = 0U; i < N; ++i) {
        if (i) {
          callbuf << ',';
          if (i % 2) {
            callbuf << ' ';
          } else {
            callbuf << "\n";
            indent(callbuf, indent_level + 1);
          }
        }
        const ParmVarDecl* PVD = FD->getParamDecl(i);
        QualType Ty = PVD->getType();
        QualType QT = Ty.getCanonicalType();
        std::string arg_type;
        ASTContext& C = FD->getASTContext();
        get_type_as_string(QT, arg_type, C, C.getPrintingPolicy());
        callbuf << arg_type;
      }
      if (FD->isVariadic())
        callbuf << ", ...";
      callbuf << ")";
    }

    callbuf << ")";
  }

  if (const CXXMethodDecl* MD = dyn_cast<CXXMethodDecl>(FD)) {
    // This is a class, struct, or union member.
    if (MD->isConst())
      callbuf << "((const " << class_name << "*)obj)->";
    else
      callbuf << "((" << class_name << "*)obj)->";

    if (op_flag)
      callbuf << class_name << "::";
  } else if (isa<NamedDecl>(get_non_transparent_decl_context(FD))) {
    // This is a namespace member.
    if (op_flag || N <= 1)
      callbuf << class_name << "::";
  }
  //   callbuf << fMethod->Name() << "(";
  {
    std::string name;
    {
      std::string complete_name;
      llvm::raw_string_ostream stream(complete_name);
      PrintingPolicy PP = FD->getASTContext().getPrintingPolicy();
      PP.FullyQualifiedName = true;
      PP.SuppressUnwrittenScope = true;
      PP.Suppress_Elab = true;
      FD->getNameForDiagnostic(stream, PP,
                               /*Qualified=*/false);
      name = complete_name;

      // If a template has consecutive parameter packs, then it is impossible to
      // use the explicit name in the wrapper, since the type deduction is what
      // determines the split of the packs. Instead, we'll revert to the
      // non-templated function name and hope that the type casts in the wrapper
      // will suffice.
      std::string simple_name = FD->getNameAsString();
      if (FD->isTemplateInstantiation() && FD->getPrimaryTemplate()) {
        const FunctionTemplateDecl* FTDecl =
            llvm::dyn_cast<FunctionTemplateDecl>(FD->getPrimaryTemplate());
        if (FTDecl) {
          auto* templateParms = FTDecl->getTemplateParameters();
          int numPacks = 0;
          for (size_t iParam = 0, nParams = templateParms->size();
               iParam < nParams; ++iParam) {
            if (templateParms->getParam(iParam)->isTemplateParameterPack())
              numPacks += 1;
            else
              numPacks = 0;
          }
          if (numPacks > 1) {
            name = simple_name;
          }
        }
      }
      if (FD->isOverloadedOperator())
        name = simple_name;
    }
    if (op_flag || N <= 1)
      callbuf << name;
  }
  if (ShouldCastFunction)
    callbuf << ")";

  callbuf << "(";
  for (unsigned i = 0U; i < N; ++i) {
    const ParmVarDecl* PVD = FD->getParamDecl(i);
    QualType Ty = PVD->getType();
    QualType QT = Ty.getCanonicalType();
    std::string type_name;
    EReferenceType refType = kNotReference;
    bool isPointer = false;
    collect_type_info(FD, QT, typedefbuf, callbuf, type_name, refType,
                      isPointer, indent_level, true);

    if (i) {
      if (op_flag) {
        callbuf << ", ";
      } else {
        callbuf << ' '
                << clang::getOperatorSpelling(FD->getOverloadedOperator())
                << ' ';
      }
    }

    CXXRecordDecl* rtdecl = QT->getAsCXXRecordDecl();
    if (refType != kNotReference) {
      callbuf << "(" << type_name.c_str()
              << (refType == kLValueReference ? "&" : "&&") << ")*("
              << type_name.c_str() << "*)args[" << i << "]";
    } else if (isPointer) {
      callbuf << "*(" << type_name.c_str() << "**)args[" << i << "]";
    } else if (rtdecl &&
               (rtdecl->hasTrivialCopyConstructor() &&
                !rtdecl->hasSimpleCopyConstructor()) &&
               rtdecl->hasMoveConstructor()) {
      // By-value construction; this may either copy or move, but there is no
      // information here in terms of intent. Thus, simply assume that the
      // intent is to move if there is no viable copy constructor (ie. if the
      // code would otherwise fail to even compile). There does not appear to be
      // a simple way of determining whether a viable copy constructor exists,
      // so check for the most common case: the trivial one, but not uniquely
      // available, while there is a move constructor.

      // include utility header if not already included for std::move
      DeclarationName DMove = &getASTContext().Idents.get("move");
      auto result = getSema().getStdNamespace()->lookup(DMove);
      if (result.empty())
        Cpp::Declare("#include <utility>");

      // move construction as needed for classes (note that this is implicit)
      callbuf << "std::move(*(" << type_name.c_str() << "*)args[" << i << "])";
    } else {
      // pointer falls back to non-pointer case; the argument preserves
      // the "pointerness" (i.e. doesn't reference the value).
      callbuf << "*(" << type_name.c_str() << "*)args[" << i << "]";
    }
  }
  callbuf << ")";
}

void make_narg_ctor_with_return(const FunctionDecl* FD, const unsigned N,
                                const std::string& class_name,
                                std::ostringstream& buf, int indent_level) {
  // Make a code string that follows this pattern:
  //
  // Array new if nary has been passed, and nargs is 0 (must be default ctor)
  // if (nary) {
  //    (*(ClassName**)ret) = (obj) ? new (*(ClassName**)ret) ClassName[nary] :
  //    new ClassName[nary];
  //   }
  // else {
  //    (*(ClassName**)ret) = (obj) ? new (*(ClassName**)ret) ClassName(args...)
  //    : new ClassName(args...);
  //   }
  {
    std::ostringstream typedefbuf;
    std::ostringstream callbuf;
    //
    //  Write the return value assignment part.
    //
    indent(callbuf, indent_level);
    const auto* CD = dyn_cast<CXXConstructorDecl>(FD);

    // Activate this block only if array new is possible
    // if (nary) {
    //    (*(ClassName**)ret) = (obj) ? new (*(ClassName**)ret) ClassName[nary]
    //    : new ClassName[nary];
    //   }
    // else {
    if (CD->isDefaultConstructor()) {
      callbuf << "if (nary > 1) {\n";
      indent(callbuf, indent_level);
      callbuf << "(*(" << class_name << "**)ret) = ";
      callbuf << "(is_arena) ? new (*(" << class_name << "**)ret) ";
      make_narg_ctor(FD, N, typedefbuf, callbuf, class_name, indent_level,
                     true);

      callbuf << ": new ";
      //
      //  Write the actual expression.
      //
      make_narg_ctor(FD, N, typedefbuf, callbuf, class_name, indent_level,
                     true);
      //
      //  End the new expression statement.
      //
      callbuf << ";\n";
      indent(callbuf, indent_level);
      callbuf << "}\n";
      callbuf << "else {\n";
    }

    // Standard branch:
    // (*(ClassName**)ret) = (obj) ? new (*(ClassName**)ret) ClassName(args...)
    // : new ClassName(args...);
    indent(callbuf, indent_level);
    callbuf << "(*(" << class_name << "**)ret) = ";
    callbuf << "(is_arena) ? new (*(" << class_name << "**)ret) ";
    make_narg_ctor(FD, N, typedefbuf, callbuf, class_name, indent_level);

    callbuf << ": new ";
    //
    //  Write the actual expression.
    //
    make_narg_ctor(FD, N, typedefbuf, callbuf, class_name, indent_level);
    //
    //  End the new expression statement.
    //
    callbuf << ";\n";
    indent(callbuf, --indent_level);
    if (CD->isDefaultConstructor())
      callbuf << "}\n";

    //
    //  Output the whole new expression and return statement.
    //
    buf << typedefbuf.str() << callbuf.str();
  }
}

void make_narg_call_with_return(compat::Interpreter& I, const FunctionDecl* FD,
                                const unsigned N, const std::string& class_name,
                                std::ostringstream& buf, int indent_level) {
  // Make a code string that follows this pattern:
  //
  // if (ret) {
  //    new (ret) (return_type) ((class_name*)obj)->func(args...);
  // }
  // else {
  //    (void)(((class_name*)obj)->func(args...));
  // }
  //
  if (const CXXConstructorDecl* CD = dyn_cast<CXXConstructorDecl>(FD)) {
    if (N <= 1 && llvm::isa<UsingShadowDecl>(FD)) {
      auto SpecMemKind = I.getCI()->getSema().getSpecialMember(CD);
      if ((N == 0 && SpecMemKind == CXXSpecialMemberKind::DefaultConstructor) ||
          (N == 1 && (SpecMemKind == CXXSpecialMemberKind::CopyConstructor ||
                      SpecMemKind == CXXSpecialMemberKind::MoveConstructor))) {
        // Using declarations cannot inject special members; do not call
        // them as such. This might happen by using `Base(Base&, int = 12)`,
        // which is fine to be called as `Derived d(someBase, 42)` but not
        // as copy constructor of `Derived`.
        return;
      }
    }
    make_narg_ctor_with_return(FD, N, class_name, buf, indent_level);
    return;
  }
  QualType QT = FD->getReturnType();
  if (QT->isVoidType()) {
    std::ostringstream typedefbuf;
    std::ostringstream callbuf;
    indent(callbuf, indent_level);
    make_narg_call(FD, "void", N, typedefbuf, callbuf, class_name,
                   indent_level);
    callbuf << ";\n";
    indent(callbuf, indent_level);
    callbuf << "return;\n";
    buf << typedefbuf.str() << callbuf.str();
  } else {
    indent(buf, indent_level);

    std::string type_name;
    EReferenceType refType = kNotReference;
    bool isPointer = false;

    std::ostringstream typedefbuf;
    std::ostringstream callbuf;

    collect_type_info(FD, QT, typedefbuf, callbuf, type_name, refType,
                      isPointer, indent_level, false);

    buf << typedefbuf.str();

    buf << "if (ret) {\n";
    ++indent_level;
    {
      //
      //  Write the placement part of the placement new.
      //
      indent(callbuf, indent_level);
      callbuf << "new (ret) ";
      //
      //  Write the type part of the placement new.
      //
      callbuf << "(" << type_name.c_str();
      if (refType != kNotReference) {
        callbuf << "*) (&";
        type_name += "&";
      } else if (isPointer) {
        callbuf << "*) (";
        type_name += "*";
      } else {
        callbuf << ") (";
      }
      //
      //  Write the actual function call.
      //
      make_narg_call(FD, type_name, N, typedefbuf, callbuf, class_name,
                     indent_level);
      //
      //  End the placement new.
      //
      callbuf << ");\n";
      indent(callbuf, indent_level);
      callbuf << "return;\n";
      //
      //  Output the whole placement new expression and return statement.
      //
      buf << typedefbuf.str() << callbuf.str();
    }
    --indent_level;
    indent(buf, indent_level);
    buf << "}\n";
    indent(buf, indent_level);
    buf << "else {\n";
    ++indent_level;
    {
      std::ostringstream typedefbuf;
      std::ostringstream callbuf;
      indent(callbuf, indent_level);
      callbuf << "(void)(";
      make_narg_call(FD, type_name, N, typedefbuf, callbuf, class_name,
                     indent_level);
      callbuf << ");\n";
      indent(callbuf, indent_level);
      callbuf << "return;\n";
      buf << typedefbuf.str() << callbuf.str();
    }
    --indent_level;
    indent(buf, indent_level);
    buf << "}\n";
  }
}

int get_wrapper_code(compat::Interpreter& I, const FunctionDecl* FD,
                     std::string& wrapper_name, std::string& wrapper) {
  assert(FD && "generate_wrapper called without a function decl!");
  ASTContext& Context = FD->getASTContext();
  //
  //  Get the class or namespace name.
  //
  std::string class_name;
  const clang::DeclContext* DC = get_non_transparent_decl_context(FD);
  GetDeclName(cast<Decl>(DC), Context, class_name);
  //
  //  Check to make sure that we can
  //  instantiate and codegen this function.
  //
  bool needInstantiation = false;
  const FunctionDecl* Definition = 0;
  compat::SynthesizingCodeRAII RAII(&getInterp());
  if (!FD->isDefined(Definition)) {
    FunctionDecl::TemplatedKind TK = FD->getTemplatedKind();
    switch (TK) {
    case FunctionDecl::TK_NonTemplate: {
      // Ordinary function, not a template specialization.
      // Note: This might be ok, the body might be defined
      //       in a library, and all we have seen is the
      //       header file.
      // llvm::errs() << "TClingCallFunc::make_wrapper" << ":" <<
      //      "Cannot make wrapper for a function which is "
      //      "declared but not defined!";
      // return 0;
    } break;
    case FunctionDecl::TK_FunctionTemplate: {
      // This decl is actually a function template,
      // not a function at all.
      llvm::errs() << "TClingCallFunc::make_wrapper"
                   << ":"
                   << "Cannot make wrapper for a function template!";
      return 0;
    } break;
    case FunctionDecl::TK_MemberSpecialization: {
      // This function is the result of instantiating an ordinary
      // member function of a class template, or of instantiating
      // an ordinary member function of a class member of a class
      // template, or of specializing a member function template
      // of a class template, or of specializing a member function
      // template of a class member of a class template.
      if (!FD->isTemplateInstantiation()) {
        // We are either TSK_Undeclared or
        // TSK_ExplicitSpecialization.
        // Note: This might be ok, the body might be defined
        //       in a library, and all we have seen is the
        //       header file.
        // llvm::errs() << "TClingCallFunc::make_wrapper" << ":" <<
        //      "Cannot make wrapper for a function template "
        //      "explicit specialization which is declared "
        //      "but not defined!";
        // return 0;
        break;
      }
      const FunctionDecl* Pattern = FD->getTemplateInstantiationPattern();
      if (!Pattern) {
        llvm::errs() << "TClingCallFunc::make_wrapper"
                     << ":"
                     << "Cannot make wrapper for a member function "
                        "instantiation with no pattern!";
        return 0;
      }
      FunctionDecl::TemplatedKind PTK = Pattern->getTemplatedKind();
      TemplateSpecializationKind PTSK =
          Pattern->getTemplateSpecializationKind();
      if (
          // The pattern is an ordinary member function.
          (PTK == FunctionDecl::TK_NonTemplate) ||
          // The pattern is an explicit specialization, and
          // so is not a template.
          ((PTK != FunctionDecl::TK_FunctionTemplate) &&
           ((PTSK == TSK_Undeclared) ||
            (PTSK == TSK_ExplicitSpecialization)))) {
        // Note: This might be ok, the body might be defined
        //       in a library, and all we have seen is the
        //       header file.
        break;
      } else if (!Pattern->hasBody()) {
        llvm::errs() << "TClingCallFunc::make_wrapper"
                     << ":"
                     << "Cannot make wrapper for a member function "
                        "instantiation with no body!";
        return 0;
      }
      if (FD->isImplicitlyInstantiable()) {
        needInstantiation = true;
      }
    } break;
    case FunctionDecl::TK_FunctionTemplateSpecialization: {
      // This function is the result of instantiating a function
      // template or possibly an explicit specialization of a
      // function template.  Could be a namespace scope function or a
      // member function.
      if (!FD->isTemplateInstantiation()) {
        // We are either TSK_Undeclared or
        // TSK_ExplicitSpecialization.
        // Note: This might be ok, the body might be defined
        //       in a library, and all we have seen is the
        //       header file.
        // llvm::errs() << "TClingCallFunc::make_wrapper" << ":" <<
        //      "Cannot make wrapper for a function template "
        //      "explicit specialization which is declared "
        //      "but not defined!";
        // return 0;
        break;
      }
      const FunctionDecl* Pattern = FD->getTemplateInstantiationPattern();
      if (!Pattern) {
        llvm::errs() << "TClingCallFunc::make_wrapper"
                     << ":"
                     << "Cannot make wrapper for a function template"
                        "instantiation with no pattern!";
        return 0;
      }
      FunctionDecl::TemplatedKind PTK = Pattern->getTemplatedKind();
      TemplateSpecializationKind PTSK =
          Pattern->getTemplateSpecializationKind();
      if (
          // The pattern is an ordinary member function.
          (PTK == FunctionDecl::TK_NonTemplate) ||
          // The pattern is an explicit specialization, and
          // so is not a template.
          ((PTK != FunctionDecl::TK_FunctionTemplate) &&
           ((PTSK == TSK_Undeclared) ||
            (PTSK == TSK_ExplicitSpecialization)))) {
        // Note: This might be ok, the body might be defined
        //       in a library, and all we have seen is the
        //       header file.
        break;
      }
      if (!GetFunctionAddress(FD)) {
        if (!Pattern->hasBody()) {
          llvm::errs() << "TClingCallFunc::make_wrapper"
                       << ":"
                       << "Cannot make wrapper for a function template "
                       << "instantiation with no body!";
          return 0;
        }
        if (FD->isImplicitlyInstantiable()) {
          needInstantiation = true;
        }
      }
    } break;
    case FunctionDecl::TK_DependentFunctionTemplateSpecialization: {
      // This function is the result of instantiating or
      // specializing a  member function of a class template,
      // or a member function of a class member of a class template,
      // or a member function template of a class template, or a
      // member function template of a class member of a class
      // template where at least some part of the function is
      // dependent on a template argument.
      if (!FD->isTemplateInstantiation()) {
        // We are either TSK_Undeclared or
        // TSK_ExplicitSpecialization.
        // Note: This might be ok, the body might be defined
        //       in a library, and all we have seen is the
        //       header file.
        // llvm::errs() << "TClingCallFunc::make_wrapper" << ":" <<
        //      "Cannot make wrapper for a dependent function "
        //      "template explicit specialization which is declared "
        //      "but not defined!";
        // return 0;
        break;
      }
      const FunctionDecl* Pattern = FD->getTemplateInstantiationPattern();
      if (!Pattern) {
        llvm::errs() << "TClingCallFunc::make_wrapper"
                     << ":"
                     << "Cannot make wrapper for a dependent function template"
                        "instantiation with no pattern!";
        return 0;
      }
      FunctionDecl::TemplatedKind PTK = Pattern->getTemplatedKind();
      TemplateSpecializationKind PTSK =
          Pattern->getTemplateSpecializationKind();
      if (
          // The pattern is an ordinary member function.
          (PTK == FunctionDecl::TK_NonTemplate) ||
          // The pattern is an explicit specialization, and
          // so is not a template.
          ((PTK != FunctionDecl::TK_FunctionTemplate) &&
           ((PTSK == TSK_Undeclared) ||
            (PTSK == TSK_ExplicitSpecialization)))) {
        // Note: This might be ok, the body might be defined
        //       in a library, and all we have seen is the
        //       header file.
        break;
      }
      if (!Pattern->hasBody()) {
        llvm::errs() << "TClingCallFunc::make_wrapper"
                     << ":"
                     << "Cannot make wrapper for a dependent function template"
                        "instantiation with no body!";
        return 0;
      }
      if (FD->isImplicitlyInstantiable()) {
        needInstantiation = true;
      }
    } break;
    default: {
      // Will only happen if clang implementation changes.
      // Protect ourselves in case that happens.
      llvm::errs() << "TClingCallFunc::make_wrapper"
                   << ":"
                   << "Unhandled template kind!";
      return 0;
    } break;
    }
    // We do not set needInstantiation to true in these cases:
    //
    // isInvalidDecl()
    // TSK_Undeclared
    // TSK_ExplicitInstantiationDefinition
    // TSK_ExplicitSpecialization && !getClassScopeSpecializationPattern()
    // TSK_ExplicitInstantiationDeclaration &&
    //    getTemplateInstantiationPattern() &&
    //    PatternDecl->hasBody() &&
    //    !PatternDecl->isInlined()
    //
    // Set it true in these cases:
    //
    // TSK_ImplicitInstantiation
    // TSK_ExplicitInstantiationDeclaration && (!getPatternDecl() ||
    //    !PatternDecl->hasBody() || PatternDecl->isInlined())
    //
  }
  if (needInstantiation) {
    clang::FunctionDecl* FDmod = const_cast<clang::FunctionDecl*>(FD);
    InstantiateFunctionDefinition(FDmod);

    if (!FD->isDefined(Definition)) {
      llvm::errs() << "TClingCallFunc::make_wrapper"
                   << ":"
                   << "Failed to force template instantiation!";
      return 0;
    }
  }
  if (Definition) {
    FunctionDecl::TemplatedKind TK = Definition->getTemplatedKind();
    switch (TK) {
    case FunctionDecl::TK_NonTemplate: {
      // Ordinary function, not a template specialization.
      if (Definition->isDeleted()) {
        llvm::errs() << "TClingCallFunc::make_wrapper"
                     << ":"
                     << "Cannot make wrapper for a deleted function!";
        return 0;
      } else if (Definition->isLateTemplateParsed()) {
        llvm::errs() << "TClingCallFunc::make_wrapper"
                     << ":"
                     << "Cannot make wrapper for a late template parsed "
                        "function!";
        return 0;
      }
      // else if (Definition->isDefaulted()) {
      //   // Might not have a body, but we can still use it.
      //}
      // else {
      //   // Has a body.
      //}
    } break;
    case FunctionDecl::TK_FunctionTemplate: {
      // This decl is actually a function template,
      // not a function at all.
      llvm::errs() << "TClingCallFunc::make_wrapper"
                   << ":"
                   << "Cannot make wrapper for a function template!";
      return 0;
    } break;
    case FunctionDecl::TK_MemberSpecialization: {
      // This function is the result of instantiating an ordinary
      // member function of a class template or of a member class
      // of a class template.
      if (Definition->isDeleted()) {
        llvm::errs() << "TClingCallFunc::make_wrapper"
                     << ":"
                     << "Cannot make wrapper for a deleted member function "
                        "of a specialization!";
        return 0;
      } else if (Definition->isLateTemplateParsed()) {
        llvm::errs() << "TClingCallFunc::make_wrapper"
                     << ":"
                     << "Cannot make wrapper for a late template parsed "
                        "member function of a specialization!";
        return 0;
      }
      // else if (Definition->isDefaulted()) {
      //   // Might not have a body, but we can still use it.
      //}
      // else {
      //   // Has a body.
      //}
    } break;
    case FunctionDecl::TK_FunctionTemplateSpecialization: {
      // This function is the result of instantiating a function
      // template or possibly an explicit specialization of a
      // function template.  Could be a namespace scope function or a
      // member function.
      if (Definition->isDeleted()) {
        llvm::errs() << "TClingCallFunc::make_wrapper"
                     << ":"
                     << "Cannot make wrapper for a deleted function "
                        "template specialization!";
        return 0;
      } else if (Definition->isLateTemplateParsed()) {
        llvm::errs() << "TClingCallFunc::make_wrapper"
                     << ":"
                     << "Cannot make wrapper for a late template parsed "
                        "function template specialization!";
        return 0;
      }
      // else if (Definition->isDefaulted()) {
      //   // Might not have a body, but we can still use it.
      //}
      // else {
      //   // Has a body.
      //}
    } break;
    case FunctionDecl::TK_DependentFunctionTemplateSpecialization: {
      // This function is the result of instantiating or
      // specializing a  member function of a class template,
      // or a member function of a class member of a class template,
      // or a member function template of a class template, or a
      // member function template of a class member of a class
      // template where at least some part of the function is
      // dependent on a template argument.
      if (Definition->isDeleted()) {
        llvm::errs() << "TClingCallFunc::make_wrapper"
                     << ":"
                     << "Cannot make wrapper for a deleted dependent function "
                        "template specialization!";
        return 0;
      } else if (Definition->isLateTemplateParsed()) {
        llvm::errs() << "TClingCallFunc::make_wrapper"
                     << ":"
                     << "Cannot make wrapper for a late template parsed "
                        "dependent function template specialization!";
        return 0;
      }
      // else if (Definition->isDefaulted()) {
      //   // Might not have a body, but we can still use it.
      //}
      // else {
      //   // Has a body.
      //}
    } break;
    default: {
      // Will only happen if clang implementation changes.
      // Protect ourselves in case that happens.
      llvm::errs() << "TClingCallFunc::make_wrapper"
                   << ":"
                   << "Unhandled template kind!";
      return 0;
    } break;
    }
  }
  unsigned min_args = FD->getMinRequiredArguments();
  unsigned num_params = FD->getNumParams();
  //
  //  Make the wrapper name.
  //
  {
    std::ostringstream buf;
    buf << "__jc";
    // const NamedDecl* ND = dyn_cast<NamedDecl>(FD);
    // std::string mn;
    // fInterp->maybeMangleDeclName(ND, mn);
    // buf << '_' << mn;
    buf << '_' << gWrapperSerial++;
    wrapper_name = buf.str();
  }
  //
  //  Write the wrapper code.
  // FIXME: this should be synthesized into the AST!
  //
  int indent_level = 0;
  std::ostringstream buf;
  buf << "#pragma clang diagnostic push\n"
         "#pragma clang diagnostic ignored \"-Wformat-security\"\n"
         "__attribute__((used)) "
         "__attribute__((annotate(\"__cling__ptrcheck(off)\")))\n"
         "extern \"C\" void ";
  buf << wrapper_name;
  if (Cpp::IsConstructor(FD)) {
    buf << "(void* ret, unsigned long nary, unsigned long nargs, void** args, "
           "void* is_arena)\n"
           "{\n";
  } else
    buf << "(void* obj, unsigned long nargs, void** args, void* ret)\n"
           "{\n";

  ++indent_level;
  if (min_args == num_params) {
    // No parameters with defaults.
    make_narg_call_with_return(I, FD, num_params, class_name, buf,
                               indent_level);
  } else {
    // We need one function call clause compiled for every
    // possible number of arguments per call.
    for (unsigned N = min_args; N <= num_params; ++N) {
      indent(buf, indent_level);
      buf << "if (nargs == " << N << ") {\n";
      ++indent_level;
      make_narg_call_with_return(I, FD, N, class_name, buf, indent_level);
      --indent_level;
      indent(buf, indent_level);
      buf << "}\n";
    }
  }
  --indent_level;
  buf << "}\n"
         "#pragma clang diagnostic pop";
  wrapper = buf.str();
  return 1;
}

JitCall::GenericCall make_wrapper(compat::Interpreter& I,
                                  const FunctionDecl* FD) {
  auto& WrapperStore = getInterpInfo(&I).WrapperStore;

  auto R = WrapperStore.find(FD);
  if (R != WrapperStore.end())
    return (JitCall::GenericCall)R->second;

  std::string wrapper_name;
  std::string wrapper_code;

  if (get_wrapper_code(I, FD, wrapper_name, wrapper_code) == 0)
    return 0;

  // Log the wrapper source for the crash reproducer.
  if (auto* TI = CppInterOp::Tracing::TraceInfo::TheTraceInfo) {
    std::string FuncName;
    llvm::raw_string_ostream FNS(FuncName);
    FD->getNameForDiagnostic(FNS, FD->getASTContext().getPrintingPolicy(),
                             /*Qualified=*/true);
    TI->appendToLog("  // === Wrapper for " + FuncName + " ===");
    // Emit each line of the wrapper source as a comment.
    llvm::StringRef WC(wrapper_code);
    while (!WC.empty()) {
      auto [Line, Rest] = WC.split('\n');
      if (!Line.empty())
        TI->appendToLog(("  // " + Line).str());
      WC = Rest;
    }
    TI->appendToLog("  // === End wrapper ===");
  }

  //
  //   Compile the wrapper code.
  //
  bool withAccessControl = true;
  // We should be able to call private default constructors.
  if (auto Ctor = dyn_cast<CXXConstructorDecl>(FD))
    withAccessControl = !Ctor->isDefaultConstructor();
  void* wrapper =
      compile_wrapper(I, wrapper_name, wrapper_code, withAccessControl);
  if (wrapper) {
    WrapperStore.insert(std::make_pair(FD, wrapper));
  } else {
    llvm::errs() << "TClingCallFunc::make_wrapper"
                 << ":"
                 << "Failed to compile\n"
                 << "==== SOURCE BEGIN ====\n"
                 << wrapper_code << "\n"
                 << "==== SOURCE END ====\n";
  }
  LLVM_DEBUG(dbgs() << "Compiled '" << (wrapper ? "" : "un")
                    << "successfully:\n"
                    << wrapper_code << "'\n");
  return (JitCall::GenericCall)wrapper;
}

// FIXME: Sink in the code duplication from get_wrapper_code.
static std::string PrepareStructorWrapper(const Decl* D,
                                          const char* wrapper_prefix,
                                          std::string& class_name) {
  ASTContext& Context = D->getASTContext();
  GetDeclName(D, Context, class_name);

  //
  //  Make the wrapper name.
  //
  std::string wrapper_name;
  {
    std::ostringstream buf;
    buf << wrapper_prefix;
    // const NamedDecl* ND = dyn_cast<NamedDecl>(FD);
    // string mn;
    // fInterp->maybeMangleDeclName(ND, mn);
    // buf << '_dtor_' << mn;
    buf << '_' << gWrapperSerial++;
    wrapper_name = buf.str();
  }

  return wrapper_name;
}

static JitCall::DestructorCall make_dtor_wrapper(compat::Interpreter& interp,
                                                 const Decl* D) {
  // Make a code string that follows this pattern:
  //
  // void
  // unique_wrapper_ddd(void* obj, unsigned long nary, int withFree)
  // {
  //    if (withFree) {
  //       if (!nary) {
  //          delete (ClassName*) obj;
  //       }
  //       else {
  //          delete[] (ClassName*) obj;
  //       }
  //    }
  //    else {
  //       typedef ClassName DtorName;
  //       if (!nary) {
  //          ((ClassName*)obj)->~DtorName();
  //       }
  //       else {
  //          for (unsigned long i = nary - 1; i > -1; --i) {
  //             (((ClassName*)obj)+i)->~DtorName();
  //          }
  //       }
  //    }
  // }
  //
  //--

  auto& DtorWrapperStore = getInterpInfo(&interp).DtorWrapperStore;

  auto I = DtorWrapperStore.find(D);
  if (I != DtorWrapperStore.end())
    return (JitCall::DestructorCall)I->second;

  //
  //  Make the wrapper name.
  //
  std::string class_name;
  std::string wrapper_name = PrepareStructorWrapper(D, "__dtor", class_name);
  //
  //  Write the wrapper code.
  //
  int indent_level = 0;
  std::ostringstream buf;
#if CPPINTEROP_ASAN_BUILD
  // ASan-only: the ORC JIT's resolution of the delete-expression below
  // does not always route through libasan's operator-delete interposer
  // (observed for classes with an out-of-line destructor), leaving the
  // matching operator-new allocation live in LSan's shadow after the
  // real free. Call __lsan_ignore_object on the object first so LSan
  // treats the allocation as intentional. Real user-side leaks never
  // reach this wrapper and stay fully visible. Exercised by
  // FunctionReflection_GetFunctionCallWrapper in the unit tests;
  // removing this block makes that test report a leak under LSan CI.
  buf << "extern \"C\" void __lsan_ignore_object(const void*);\n";
#endif
  buf << "__attribute__((used)) ";
  buf << "extern \"C\" void ";
  buf << wrapper_name;
  buf << "(void* obj, unsigned long nary, int withFree)\n";
  buf << "{\n";
  //    if (withFree) {
  //       __lsan_ignore_object(obj);          // ASan builds only
  //       if (!nary) {
  //          delete (ClassName*) obj;
  //       }
  //       else {
  //          delete[] (ClassName*) obj;
  //       }
  //    }
  ++indent_level;
  indent(buf, indent_level);
  buf << "if (withFree) {\n";
  ++indent_level;
#if CPPINTEROP_ASAN_BUILD
  indent(buf, indent_level);
  buf << "__lsan_ignore_object(obj);\n";
#endif
  indent(buf, indent_level);
  buf << "if (!nary) {\n";
  ++indent_level;
  indent(buf, indent_level);
  buf << "delete (" << class_name << "*) obj;\n";
  --indent_level;
  indent(buf, indent_level);
  buf << "}\n";
  indent(buf, indent_level);
  buf << "else {\n";
  ++indent_level;
  indent(buf, indent_level);
  buf << "delete[] (" << class_name << "*) obj;\n";
  --indent_level;
  indent(buf, indent_level);
  buf << "}\n";
  --indent_level;
  indent(buf, indent_level);
  buf << "}\n";
  //    else {
  //       typedef ClassName Nm;
  //       if (!nary) {
  //          ((Nm*)obj)->~Nm();
  //       }
  //       else {
  //          for (unsigned long i = nary - 1; i > -1; --i) {
  //             (((Nm*)obj)+i)->~Nm();
  //          }
  //       }
  //    }
  indent(buf, indent_level);
  buf << "else {\n";
  ++indent_level;
  indent(buf, indent_level);
  buf << "typedef " << class_name << " Nm;\n";
  buf << "if (!nary) {\n";
  ++indent_level;
  indent(buf, indent_level);
  buf << "((Nm*)obj)->~Nm();\n";
  --indent_level;
  indent(buf, indent_level);
  buf << "}\n";
  indent(buf, indent_level);
  buf << "else {\n";
  ++indent_level;
  indent(buf, indent_level);
  buf << "do {\n";
  ++indent_level;
  indent(buf, indent_level);
  buf << "(((Nm*)obj)+(--nary))->~Nm();\n";
  --indent_level;
  indent(buf, indent_level);
  buf << "} while (nary);\n";
  --indent_level;
  indent(buf, indent_level);
  buf << "}\n";
  --indent_level;
  indent(buf, indent_level);
  buf << "}\n";
  // End wrapper.
  --indent_level;
  buf << "}\n";
  // Done.
  std::string wrapper(buf.str());
  // fprintf(stderr, "%s\n", wrapper.c_str());
  //
  //   Compile the wrapper code.
  //
  void* F = compile_wrapper(interp, wrapper_name, wrapper,
                            /*withAccessControl=*/false);
  if (F) {
    DtorWrapperStore.insert(std::make_pair(D, F));
  } else {
    llvm::errs() << "make_dtor_wrapper"
                 << "Failed to compile\n"
                 << "==== SOURCE BEGIN ====\n"
                 << wrapper << "\n  ==== SOURCE END ====";
  }
  LLVM_DEBUG(dbgs() << "Compiled '" << (F ? "" : "un") << "successfully:\n"
                    << wrapper << "'\n");
  return (JitCall::DestructorCall)F;
}
#undef DEBUG_TYPE
} // namespace
  // End of JitCall Helper Functions

CPPINTEROP_API JitCall MakeFunctionCallable(TInterp_t I,
                                            TCppConstFunction_t func) {
  INTEROP_TRACE(I, func);
  const auto* D = static_cast<const clang::Decl*>(func);
  if (!D)
    return INTEROP_RETURN(JitCall{});

  auto* interp = static_cast<compat::Interpreter*>(I);

  // FIXME: Unify with make_wrapper.
  if (const auto* Dtor = dyn_cast<CXXDestructorDecl>(D)) {
    if (auto Wrapper = make_dtor_wrapper(*interp, Dtor->getParent()))
      return INTEROP_RETURN(JitCall(JitCall::kDestructorCall, Wrapper, Dtor));
    // FIXME: else error we failed to compile the wrapper.
    return INTEROP_RETURN(JitCall{});
  }

  if (const auto* Ctor = dyn_cast<CXXConstructorDecl>(D)) {
    if (auto Wrapper = make_wrapper(*interp, cast<FunctionDecl>(D)))
      return INTEROP_RETURN(JitCall(JitCall::kConstructorCall, Wrapper, Ctor));
    // FIXME: else error we failed to compile the wrapper.
    return INTEROP_RETURN(JitCall{});
  }

  if (auto Wrapper = make_wrapper(*interp, cast<FunctionDecl>(D))) {
    return INTEROP_RETURN(
        JitCall(JitCall::kGenericCall, Wrapper, cast<FunctionDecl>(D)));
  }
  // FIXME: else error we failed to compile the wrapper.
  return INTEROP_RETURN(JitCall{});
}

CPPINTEROP_API JitCall MakeFunctionCallable(TCppConstFunction_t func) {
  INTEROP_TRACE(func);
  return INTEROP_RETURN(MakeFunctionCallable(&getInterp(), func));
}

namespace {
#if !defined(CPPINTEROP_USE_CLING) && !defined(EMSCRIPTEN)
bool DefineAbsoluteSymbol(compat::Interpreter& I,
                          const char* linker_mangled_name, uint64_t address) {
  using namespace llvm;
  using namespace llvm::orc;

  llvm::orc::LLJIT& Jit = *compat::getExecutionEngine(I);
  llvm::orc::ExecutionSession& ES = Jit.getExecutionSession();
  JITDylib& DyLib = *Jit.getProcessSymbolsJITDylib().get();

  llvm::orc::SymbolMap InjectedSymbols{
      {ES.intern(linker_mangled_name),
       ExecutorSymbolDef(ExecutorAddr(address), JITSymbolFlags::Exported)}};

  if (Error Err = DyLib.define(absoluteSymbols(InjectedSymbols))) {
    logAllUnhandledErrors(std::move(Err), errs(),
                          "DefineAbsoluteSymbol error: ");
    return true;
  }
  return false;
}
#endif

static std::string MakeResourcesPath() {
  StringRef Dir;
#ifdef LLVM_BINARY_DIR
  Dir = LLVM_BINARY_DIR;
#else
  // Dir is bin/ or lib/, depending on where BinaryPath is.
  void* MainAddr = (void*)(intptr_t)GetExecutablePath;
  std::string BinaryPath = GetExecutablePath(/*Argv0=*/nullptr, MainAddr);

  // build/tools/clang/unittests/Interpreter/Executable -> build/
  StringRef Dir = sys::path::parent_path(BinaryPath);

  Dir = sys::path::parent_path(Dir);
  Dir = sys::path::parent_path(Dir);
  Dir = sys::path::parent_path(Dir);
  Dir = sys::path::parent_path(Dir);
  // Dir = sys::path::parent_path(Dir);
#endif // LLVM_BINARY_DIR
  llvm::SmallString<128> P(Dir);
  llvm::sys::path::append(P, CLANG_INSTALL_LIBDIR_BASENAME, "clang",
                          CLANG_VERSION_MAJOR_STRING);
  return std::string(P.str());
}

void AddLibrarySearchPaths(const std::string& ResourceDir,
                           compat::Interpreter* I) {
  // the resource-dir can be of the form
  // /prefix/lib/clang/XX or /prefix/lib/llvm-XX/lib/clang/XX
  // where XX represents version
  // the corresponing path we want to add are
  // /prefix/lib/clang/XX/lib, /prefix/lib/, and
  // /prefix/lib/llvm-XX/lib/clang/XX/lib, /prefix/lib/llvm-XX/lib/,
  // /prefix/lib/
  std::string path1 = ResourceDir + "/lib";
  I->getDynamicLibraryManager()->addSearchPath(path1, false, false);
  size_t pos = ResourceDir.rfind("/llvm-");
  if (pos != std::string::npos) {
    I->getDynamicLibraryManager()->addSearchPath(ResourceDir.substr(0, pos),
                                                 false, false);
  }
  pos = ResourceDir.rfind("/clang");
  if (pos != std::string::npos) {
    I->getDynamicLibraryManager()->addSearchPath(ResourceDir.substr(0, pos),
                                                 false, false);
  }
}
std::string ExtractArgument(const std::vector<const char*>& Args,
                            const std::string& Arg) {
  size_t I = 0;
  for (auto i = Args.begin(); i != Args.end(); i++)
    if ((++I < Args.size()) && (*i == Arg))
      return *(++i);
  return "";
}
} // namespace

TInterp_t CreateInterpreter(const std::vector<const char*>& Args /*={}*/,
                            const std::vector<const char*>& GpuArgs /*={}*/) {
  INTEROP_TRACE(Args, GpuArgs);
  std::string MainExecutableName = sys::fs::getMainExecutable(nullptr, nullptr);
  // In some systems, CppInterOp cannot manually detect the correct resource.
  // Then the -resource-dir passed by the user is assumed to be the correct
  // location. Prioritising it over detecting it within CppInterOp. Extracting
  // the resource-dir from the arguments is required because we set the
  // necessary library search location explicitly below. Because by default,
  // linker flags are ignored in repl (issue #748)
  std::string ResourceDir = ExtractArgument(Args, "-resource-dir");
  if (ResourceDir.empty())
    ResourceDir = MakeResourcesPath();
  llvm::Triple T(llvm::sys::getProcessTriple());
  if ((!sys::fs::is_directory(ResourceDir)) &&
      (T.isOSDarwin() || T.isOSLinux()))
    ResourceDir = DetectResourceDir();

  std::vector<const char*> ClingArgv = {"-resource-dir", ResourceDir.c_str(),
                                        "-std=c++14"};
  ClingArgv.insert(ClingArgv.begin(), MainExecutableName.c_str());
#ifdef _WIN32
  // FIXME : Workaround Sema::PushDeclContext assert on windows
  ClingArgv.push_back("-fno-delayed-template-parsing");
#endif
  ClingArgv.insert(ClingArgv.end(), Args.begin(), Args.end());
  // To keep the Interpreter creation interface between cling and clang-repl
  // to some extent compatible we should put Args and GpuArgs together. On the
  // receiving end we should check for -xcuda to know.
  if (!GpuArgs.empty()) {
    llvm::StringRef Arg0 = GpuArgs[0];
    Arg0 = Arg0.trim().ltrim('-');
    if (Arg0 != "cuda") {
      llvm::errs() << "[CreateInterpreter]: Make sure --cuda is passed as the"
                   << " first argument of the GpuArgs\n";
      return INTEROP_RETURN(nullptr);
    }
  }
  ClingArgv.insert(ClingArgv.end(), GpuArgs.begin(), GpuArgs.end());

  // Process externally passed arguments if present.
  std::vector<std::string> ExtraArgs;
  auto EnvOpt = llvm::sys::Process::GetEnv("CPPINTEROP_EXTRA_INTERPRETER_ARGS");
  if (EnvOpt) {
    StringRef Env(*EnvOpt);
    while (!Env.empty()) {
      StringRef Arg;
      std::tie(Arg, Env) = Env.split(' ');
      ExtraArgs.push_back(Arg.str());
    }
  }
  std::transform(ExtraArgs.begin(), ExtraArgs.end(),
                 std::back_inserter(ClingArgv),
                 [&](const std::string& str) { return str.c_str(); });

  // Force global process initialization.
  (void)GetInterpreters();

#ifdef CPPINTEROP_USE_CLING
  auto I = new compat::Interpreter(ClingArgv.size(), &ClingArgv[0]);
#else
  auto Interp =
      compat::Interpreter::create(static_cast<int>(ClingArgv.size()),
                                  ClingArgv.data(), nullptr, {}, nullptr, true);
  if (!Interp)
    return INTEROP_RETURN(nullptr);
  auto* I = Interp.release();
#endif

  // Honor -mllvm.
  //
  // FIXME: Remove this, one day.
  // This should happen AFTER plugins have been loaded!
  const CompilerInstance* Clang = I->getCI();
  if (!Clang->getFrontendOpts().LLVMArgs.empty()) {
    unsigned NumArgs = Clang->getFrontendOpts().LLVMArgs.size();
    auto Args = std::make_unique<const char*[]>(NumArgs + 2);
    Args[0] = "clang (LLVM option parsing)";
    for (unsigned i = 0; i != NumArgs; ++i)
      Args[i + 1] = Clang->getFrontendOpts().LLVMArgs[i].c_str();
    Args[NumArgs + 1] = nullptr;
    llvm::cl::ParseCommandLineOptions(NumArgs + 1, Args.get());
  }

  if (!T.isWasm())
    AddLibrarySearchPaths(ResourceDir, I);

  if (GetLanguage(I) != InterpreterLanguage::C) {
    I->declare(R"(
    namespace __internal_CppInterOp {
    template <typename Signature>
    struct function;
    template <typename Res, typename... ArgTypes>
    struct function<Res(ArgTypes...)> {
      typedef Res result_type;
    };
    }  // namespace __internal_CppInterOp
  )");
  }

  RegisterInterpreter(I, /*Owned=*/true);

// Define runtime symbols in the JIT dylib for clang-repl
#if !defined(CPPINTEROP_USE_CLING) && !defined(EMSCRIPTEN)
  DefineAbsoluteSymbol(*I, "__ci_newtag",
                       reinterpret_cast<uint64_t>(&__ci_newtag));
// llvm >= 21 has this defined as a C symbol that does not require mangling
#if CLANG_VERSION_MAJOR >= 21
  DefineAbsoluteSymbol(
      *I, "__clang_Interpreter_SetValueWithAlloc",
      reinterpret_cast<uint64_t>(&__clang_Interpreter_SetValueWithAlloc));
#else
  // obtain mangled name
  auto* D = static_cast<clang::Decl*>(
      Cpp::GetNamed("__clang_Interpreter_SetValueWithAlloc"));
  if (auto* FD = llvm::dyn_cast_or_null<FunctionDecl>(D)) {
    auto GD = GlobalDecl(FD);
    std::string mangledName;
    compat::maybeMangleDeclName(GD, mangledName);
    DefineAbsoluteSymbol(
        *I, mangledName.c_str(),
        reinterpret_cast<uint64_t>(&__clang_Interpreter_SetValueWithAlloc));
  }
#endif

  DefineAbsoluteSymbol(
      *I, "__clang_Interpreter_SetValueNoAlloc",
      reinterpret_cast<uint64_t>(&__clang_Interpreter_SetValueNoAlloc));
#endif
  return INTEROP_RETURN(I);
}

InterpreterLanguage GetLanguage(TInterp_t I /*=nullptr*/) {
  INTEROP_TRACE(I);
  compat::Interpreter* interp = &getInterp(I);
  const auto& LO = interp->getCI()->getLangOpts();

  // CUDA and HIP reuse C++ language standards, so LangStd alone reports CXX.
  if (LO.CUDA)
    return INTEROP_RETURN(InterpreterLanguage::CUDA);
  if (LO.HIP)
    return INTEROP_RETURN(InterpreterLanguage::HIP);

  auto standard = clang::LangStandard::getLangStandardForKind(LO.LangStd);
  auto lang = static_cast<InterpreterLanguage>(standard.getLanguage());
  assert(lang != InterpreterLanguage::Unknown && "Unknown language");
  assert(static_cast<unsigned char>(lang) <=
             static_cast<unsigned char>(InterpreterLanguage::HLSL) &&
         "Unhandled Language");
  return INTEROP_RETURN(lang);
}

InterpreterLanguageStandard GetLanguageStandard(TInterp_t I /*=nullptr*/) {
  INTEROP_TRACE(I);
  compat::Interpreter* interp = &getInterp(I);
  const auto& LO = interp->getCI()->getLangOpts();
  auto langStandard = static_cast<InterpreterLanguageStandard>(LO.LangStd);
  assert(langStandard != InterpreterLanguageStandard::lang_unspecified &&
         "Unspecified language standard");
  assert(static_cast<unsigned char>(langStandard) <=
             static_cast<unsigned char>(
                 InterpreterLanguageStandard::lang_unspecified) &&
         "Unhandled language standard.");
  return INTEROP_RETURN(langStandard);
}

void AddSearchPath(const char* dir, bool isUser, bool prepend) {
  INTEROP_TRACE(dir, isUser, prepend);
  getInterp().getDynamicLibraryManager()->addSearchPath(dir, isUser, prepend);
  return INTEROP_VOID_RETURN();
}

const char* GetResourceDir() {
  INTEROP_TRACE();
  return INTEROP_RETURN(
      getInterp().getCI()->getHeaderSearchOpts().ResourceDir.c_str());
}

///\returns 0 on success.
static bool exec(const char* cmd, std::vector<std::string>& outputs) {
#define DEBUG_TYPE "exec"

  std::array<char, 256> buffer;
  struct file_deleter {
    void operator()(FILE* fp) { pclose(fp); }
  };
  std::unique_ptr<FILE, file_deleter> pipe{popen(cmd, "r")};
  LLVM_DEBUG(dbgs() << "Executing command '" << cmd << "'\n");

  if (!pipe) {
    LLVM_DEBUG(dbgs() << "Execute failed!\n");
    perror("exec: ");
    return false;
  }

  LLVM_DEBUG(dbgs() << "Execute returned:\n");
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get())) {
    LLVM_DEBUG(dbgs() << buffer.data());
    llvm::StringRef trimmed = buffer.data();
    outputs.push_back(trimmed.trim().str());
  }

#undef DEBUG_TYPE

  return true;
}

std::string DetectResourceDir(const char* ClangBinaryName /* = clang */) {
  INTEROP_TRACE(ClangBinaryName);
  std::string cmd = std::string(ClangBinaryName) + " -print-resource-dir";
  std::vector<std::string> outs;
  exec(cmd.c_str(), outs);
  if (outs.empty() || outs.size() > 1)
    return INTEROP_RETURN("");

  std::string detected_resource_dir = outs.back();

  std::string version = CLANG_VERSION_MAJOR_STRING;
  // We need to check if the detected resource directory is compatible.
  if (llvm::sys::path::filename(detected_resource_dir) != version)
    return INTEROP_RETURN("");

  return INTEROP_RETURN(detected_resource_dir);
}

void DetectSystemCompilerIncludePaths(std::vector<std::string>& Paths,
                                      const char* CompilerName /*= "c++"*/) {
  INTEROP_TRACE(INTEROP_OUT(Paths), CompilerName);
  std::string cmd = "LC_ALL=C ";
  cmd += CompilerName;
  cmd += " -xc++ -E -v /dev/null 2>&1 | sed -n -e '/^.include/,${' -e '/^ "
         "\\/.*/p' -e '}'";
  std::vector<std::string> outs;
  exec(cmd.c_str(), Paths);
  return INTEROP_VOID_RETURN();
}

void AddIncludePath(const char* dir) {
  INTEROP_TRACE(dir);
  getInterp().AddIncludePath(dir);
  return INTEROP_VOID_RETURN();
}

void GetIncludePaths(std::vector<std::string>& IncludePaths, bool withSystem,
                     bool withFlags) {
  INTEROP_TRACE(INTEROP_OUT(IncludePaths), withSystem, withFlags);
  llvm::SmallVector<std::string> paths(1);
  getInterp().GetIncludePaths(paths, withSystem, withFlags);
  for (auto& i : paths)
    IncludePaths.push_back(i);
  return INTEROP_VOID_RETURN();
}

namespace {

class clangSilent {
public:
  clangSilent(clang::DiagnosticsEngine& diag) : fDiagEngine(diag) {
    fOldDiagValue = fDiagEngine.getSuppressAllDiagnostics();
    fDiagEngine.setSuppressAllDiagnostics(true);
  }

  ~clangSilent() { fDiagEngine.setSuppressAllDiagnostics(fOldDiagValue); }

protected:
  clang::DiagnosticsEngine& fDiagEngine;
  bool fOldDiagValue;
};
} // namespace

int Declare(compat::Interpreter& I, const char* code, bool silent) {
  if (silent) {
    clang::DiagnosticsEngine& Diag = I.getSema().getDiagnostics();
    clangSilent diagSuppr(Diag);
    clang::DiagnosticErrorTrap Trap(Diag);
    auto result = I.declare(code);
    if (Trap.hasErrorOccurred())
      return 1;
    return result;
  }

  return I.declare(code);
}

int Declare(const char* code, bool silent) {
  INTEROP_TRACE(code, silent);
  return INTEROP_RETURN(Declare(getInterp(), code, silent));
}

int Process(const char* code) {
  INTEROP_TRACE(code);
  return INTEROP_RETURN(getInterp().process(code));
}

intptr_t Evaluate(const char* code, bool* HadError /*=nullptr*/) {
  INTEROP_TRACE(code, HadError);
  compat::Value V;

  if (HadError)
    *HadError = false;

  auto res = getInterp().evaluate(code, V);
  if (res != 0) { // 0 is success
    if (HadError)
      *HadError = true;
    // FIXME: Make this return llvm::Expected
    return INTEROP_RETURN(~0UL);
  }

  return INTEROP_RETURN(compat::convertTo<intptr_t>(V));
}

std::string LookupLibrary(const char* lib_name) {
  INTEROP_TRACE(lib_name);
  return INTEROP_RETURN(
      getInterp().getDynamicLibraryManager()->lookupLibrary(lib_name));
}

bool LoadLibrary(const char* lib_stem, bool lookup) {
  INTEROP_TRACE(lib_stem, lookup);
  compat::Interpreter::CompilationResult res =
      getInterp().loadLibrary(lib_stem, lookup);

  return INTEROP_RETURN(res == compat::Interpreter::kSuccess);
}

void UnloadLibrary(const char* lib_stem) {
  INTEROP_TRACE(lib_stem);
  getInterp().getDynamicLibraryManager()->unloadLibrary(lib_stem);
  return INTEROP_VOID_RETURN();
}

std::string SearchLibrariesForSymbol(const char* mangled_name,
                                     bool search_system /*true*/) {
  INTEROP_TRACE(mangled_name, search_system);
  auto* DLM = getInterp().getDynamicLibraryManager();
  return INTEROP_RETURN(
      DLM->searchLibrariesForSymbol(mangled_name, search_system));
}

bool InsertOrReplaceJitSymbol(compat::Interpreter& I,
                              const char* linker_mangled_name,
                              uint64_t address) {
  // FIXME: This approach is problematic since we could replace a symbol
  // whose address was already taken by clients.
  //
  // A safer approach would be to define our symbol replacements early in the
  // bootstrap process like:
  // auto J = LLJITBuilder().create();
  // if (!J)
  //   return Err;
  //
  // if (Jupyter) {
  //   llvm::orc::SymbolMap Overrides;
  //   Overrides[J->mangleAndIntern("printf")] =
  //     { ExecutorAddr::fromPtr(&printf), JITSymbolFlags::Exported };
  //   Overrides[...] =
  //     { ... };
  //   if (auto Err =
  //   J->getProcessSymbolsJITDylib().define(absoluteSymbols(std::move(Overrides)))
  //     return Err;
  // }

  // FIXME: If we still want to do symbol replacement we should use the
  // ReplacementManager which is available in llvm 18.
  using namespace llvm;
  using namespace llvm::orc;

  auto Symbol = compat::getSymbolAddress(I, linker_mangled_name);
  llvm::orc::LLJIT& Jit = *compat::getExecutionEngine(I);
  llvm::orc::ExecutionSession& ES = Jit.getExecutionSession();
  JITDylib& DyLib = *Jit.getProcessSymbolsJITDylib().get();

  if (Error Err = Symbol.takeError()) {
    logAllUnhandledErrors(std::move(Err), errs(),
                          "[InsertOrReplaceJitSymbol] error: ");
#define DEBUG_TYPE "orc"
    LLVM_DEBUG(ES.dump(dbgs()));
#undef DEBUG_TYPE
    return true;
  }

  // Nothing to define, we are redefining the same function.
  if (*Symbol && *Symbol == address) {
    errs() << "[InsertOrReplaceJitSymbol] warning: redefining '"
           << linker_mangled_name << "' with the same address\n";
    return true;
  }

  // Let's inject it.
  llvm::orc::SymbolMap InjectedSymbols;
  auto& DL = compat::getExecutionEngine(I)->getDataLayout();
  char GlobalPrefix = DL.getGlobalPrefix();
  std::string tmp(linker_mangled_name);
  if (GlobalPrefix != '\0') {
    tmp = std::string(1, GlobalPrefix) + tmp;
  }
  auto Name = ES.intern(tmp);
  InjectedSymbols[Name] =
      ExecutorSymbolDef(ExecutorAddr(address), JITSymbolFlags::Exported);

  // We want to replace a symbol with a custom provided one.
  if (Symbol && address)
    // The symbol be in the DyLib or in-process.
    if (auto Err = DyLib.remove({Name})) {
      logAllUnhandledErrors(std::move(Err), errs(),
                            "[InsertOrReplaceJitSymbol] error: ");
      return true;
    }

  if (Error Err = DyLib.define(absoluteSymbols(InjectedSymbols))) {
    logAllUnhandledErrors(std::move(Err), errs(),
                          "[InsertOrReplaceJitSymbol] error: ");
    return true;
  }

  return false;
}

bool InsertOrReplaceJitSymbol(const char* linker_mangled_name,
                              uint64_t address) {
  INTEROP_TRACE(linker_mangled_name, address);
  return INTEROP_RETURN(
      InsertOrReplaceJitSymbol(getInterp(), linker_mangled_name, address));
}

std::string ObjToString(const char* type, void* obj) {
  INTEROP_TRACE(type, obj);
  return INTEROP_RETURN(getInterp().toString(type, obj));
}

static Decl* InstantiateTemplate(TemplateDecl* TemplateD,
                                 TemplateArgumentListInfo& TLI, Sema& S,
                                 bool instantiate_body) {
  // This is not right but we don't have a lot of options to choose from as a
  // template instantiation requires a valid source location.
  SourceLocation fakeLoc = GetValidSLoc(S);
  if (auto* FunctionTemplate = dyn_cast<FunctionTemplateDecl>(TemplateD)) {
    FunctionDecl* Specialization = nullptr;
    clang::sema::TemplateDeductionInfo Info(fakeLoc);
    TemplateDeductionResult Result =
        S.DeduceTemplateArguments(FunctionTemplate, &TLI, Specialization, Info,
                                  /*IsAddressOfFunction*/ true);
    if (Result != TemplateDeductionResult::Success) {
      // FIXME: Diagnose what happened.
      (void)Result;
    }
    if (instantiate_body)
      InstantiateFunctionDefinition(Specialization);
    return Specialization;
  }

  if (auto* VarTemplate = dyn_cast<VarTemplateDecl>(TemplateD)) {
#if CLANG_VERSION_MAJOR < 22
    DeclResult R = S.CheckVarTemplateId(VarTemplate, fakeLoc, fakeLoc, TLI);
#else
    DeclResult R = S.CheckVarTemplateId(VarTemplate, fakeLoc, fakeLoc, TLI,
                                        /*SetWrittenArgs=*/true);
#endif
    if (R.isInvalid()) {
      // FIXME: Diagnose
    }
    return R.get();
  }

  // This will instantiate tape<T> type and return it.
  SourceLocation noLoc;
#if CLANG_VERSION_MAJOR < 22
  QualType TT = S.CheckTemplateIdType(TemplateName(TemplateD), noLoc, TLI);
#else
  QualType TT = S.CheckTemplateIdType(
      ElaboratedTypeKeyword::None, TemplateName(TemplateD), noLoc, TLI,
      /*Scope=*/nullptr, /*ForNestedNameSpecifier=*/false);
#endif
  if (TT.isNull())
    return nullptr;

  // Perhaps we can extract this into a new interface.
  S.RequireCompleteType(fakeLoc, TT, diag::err_tentative_def_incomplete_type);
  return GetScopeFromType(TT);

  // ASTContext &C = S.getASTContext();
  // // Get clad namespace and its identifier clad::.
  // CXXScopeSpec CSS;
  // CSS.Extend(C, GetCladNamespace(), noLoc, noLoc);
  // NestedNameSpecifier* NS = CSS.getScopeRep();

  // // Create elaborated type with namespace specifier,
  // // i.e. class<T> -> clad::class<T>
  // return C.getElaboratedType(ETK_None, NS, TT);
}

Decl* InstantiateTemplate(TemplateDecl* TemplateD,
                          ArrayRef<TemplateArgument> TemplateArgs, Sema& S,
                          bool instantiate_body) {
  // Create a list of template arguments.
  TemplateArgumentListInfo TLI{};
  for (auto TA : TemplateArgs)
    TLI.addArgument(
        S.getTrivialTemplateArgumentLoc(TA, QualType(), SourceLocation()));

  return InstantiateTemplate(TemplateD, TLI, S, instantiate_body);
}

TCppScope_t InstantiateTemplate(compat::Interpreter& I, TCppScope_t tmpl,
                                const TemplateArgInfo* template_args,
                                size_t template_args_size,
                                bool instantiate_body) {
  auto& S = I.getSema();
  auto& C = S.getASTContext();

  llvm::SmallVector<TemplateArgument> TemplateArgs;
  TemplateArgs.reserve(template_args_size);
  for (size_t i = 0; i < template_args_size; ++i) {
    QualType ArgTy = QualType::getFromOpaquePtr(template_args[i].m_Type);
    if (template_args[i].m_IntegralValue) {
      // We have a non-type template parameter. Create an integral value from
      // the string representation.
      auto Res = llvm::APSInt(template_args[i].m_IntegralValue);
      Res = Res.extOrTrunc(C.getIntWidth(ArgTy));
      TemplateArgs.push_back(TemplateArgument(C, Res, ArgTy));
    } else {
      TemplateArgs.push_back(ArgTy);
    }
  }

  TemplateDecl* TmplD = static_cast<TemplateDecl*>(tmpl);
  // We will create a new decl, push a transaction.
  compat::SynthesizingCodeRAII RAII(&getInterp());
  return InstantiateTemplate(TmplD, TemplateArgs, S, instantiate_body);
}

TCppScope_t InstantiateTemplate(TCppScope_t tmpl,
                                const TemplateArgInfo* template_args,
                                size_t template_args_size,
                                bool instantiate_body) {
  INTEROP_TRACE(tmpl, template_args, template_args_size, instantiate_body);
  return INTEROP_RETURN(InstantiateTemplate(
      getInterp(), tmpl, template_args, template_args_size, instantiate_body));
}

void GetClassTemplateInstantiationArgs(TCppScope_t templ_instance,
                                       std::vector<TemplateArgInfo>& args) {
  INTEROP_TRACE(templ_instance, INTEROP_OUT(args));
  auto* CTSD = static_cast<ClassTemplateSpecializationDecl*>(templ_instance);
  for (const auto& TA : CTSD->getTemplateInstantiationArgs().asArray()) {
    switch (TA.getKind()) {
    default:
      assert(0 && "Not yet supported!");
      break;
    case TemplateArgument::Pack:
      for (auto SubTA : TA.pack_elements())
        args.push_back({SubTA.getAsType().getAsOpaquePtr()});
      break;
    case TemplateArgument::Integral:
      // FIXME: Support this case where the problem is where we provide the
      // storage for the m_IntegralValue.
      // llvm::APSInt Val = TA.getAsIntegral();
      // args.push_back({TA.getIntegralType(), TA.getAsIntegral()})
      // break;
    case TemplateArgument::Type:
      args.push_back({TA.getAsType().getAsOpaquePtr()});
    }
  }
  return INTEROP_VOID_RETURN();
}

TCppFunction_t
InstantiateTemplateFunctionFromString(const char* function_template) {
  INTEROP_TRACE(function_template);
  // FIXME: Drop this interface and replace it with the proper overload
  // resolution handling and template instantiation selection.

  // Try to force template instantiation and overload resolution.
  static unsigned long long var_count = 0;
  std::string id = "__Cppyy_GetMethTmpl_" + std::to_string(var_count++);
  std::string instance = "auto " + id + " = " + function_template + ";\n";

  if (!Cpp::Declare(instance.c_str(), /*silent=*/false)) {
    VarDecl* VD = (VarDecl*)Cpp::GetNamed(id, 0);
    DeclRefExpr* DRE = (DeclRefExpr*)VD->getInit()->IgnoreImpCasts();
    return INTEROP_RETURN(DRE->getDecl());
  }
  return INTEROP_RETURN(nullptr);
}

void GetAllCppNames(TCppScope_t scope, std::set<std::string>& names) {
  INTEROP_TRACE(scope, INTEROP_OUT(names));
  auto* D = (clang::Decl*)scope;
  clang::DeclContext* DC;
  clang::DeclContext::decl_iterator decl;

  compat::SynthesizingCodeRAII RAII(&getInterp());

  if (auto* TD = dyn_cast_or_null<TagDecl>(D)) {
    DC = clang::TagDecl::castToDeclContext(TD);
    decl = DC->decls_begin();
    decl++;
  } else if (auto* ND = dyn_cast_or_null<NamespaceDecl>(D)) {
    DC = clang::NamespaceDecl::castToDeclContext(ND);
    decl = DC->decls_begin();
  } else if (auto* TUD = dyn_cast_or_null<TranslationUnitDecl>(D)) {
    DC = clang::TranslationUnitDecl::castToDeclContext(TUD);
    decl = DC->decls_begin();
  } else {
    return INTEROP_VOID_RETURN();
  }

  for (/* decl set above */; decl != DC->decls_end(); decl++) {
    if (auto* ND = llvm::dyn_cast_or_null<NamedDecl>(*decl)) {
      names.insert(ND->getNameAsString());
    }
  }
  return INTEROP_VOID_RETURN();
}

void GetEnums(TCppScope_t scope, std::vector<std::string>& Result) {
  INTEROP_TRACE(scope, INTEROP_OUT(Result));
  auto* D = static_cast<clang::Decl*>(scope);

  if (!llvm::isa_and_nonnull<clang::DeclContext>(D))
    return INTEROP_VOID_RETURN();

  auto* DC = llvm::dyn_cast<clang::DeclContext>(D);

  llvm::SmallVector<clang::DeclContext*, 4> DCs;
  DC->collectAllContexts(DCs);

  // FIXME: We should use a lookup based approach instead of brute force
  for (auto* DC : DCs) {
    for (auto decl = DC->decls_begin(); decl != DC->decls_end(); decl++) {
      if (auto* ND = llvm::dyn_cast_or_null<EnumDecl>(*decl)) {
        Result.push_back(ND->getNameAsString());
      }
    }
  }
  return INTEROP_VOID_RETURN();
}

// FIXME: On the CPyCppyy side the receiver is of type
//        vector<long int> instead of vector<TCppIndex_t>
std::vector<long int> GetDimensions(TCppType_t type) {
  INTEROP_TRACE(type);
  QualType Qual = QualType::getFromOpaquePtr(type);
  if (Qual.isNull())
    return INTEROP_RETURN(std::vector<long int>{});
  Qual = Qual.getCanonicalType();
  std::vector<long int> dims;
  if (Qual->isArrayType()) {
    const clang::ArrayType* ArrayType =
        dyn_cast<clang::ArrayType>(Qual.getTypePtr());
    while (ArrayType) {
      if (const auto* CAT = dyn_cast_or_null<ConstantArrayType>(ArrayType)) {
        llvm::APSInt Size(CAT->getSize());
        long int ArraySize = Size.getLimitedValue();
        dims.push_back(ArraySize);
      } else /* VariableArrayType, DependentSizedArrayType, IncompleteArrayType
              */
      {
        dims.push_back(DimensionValue::UNKNOWN_SIZE);
      }
      ArrayType = ArrayType->getElementType()->getAsArrayTypeUnsafe();
    }
    return INTEROP_RETURN(dims);
  }
  return INTEROP_RETURN(dims);
}

bool IsTypeDerivedFrom(TCppType_t derived, TCppType_t base) {
  INTEROP_TRACE(derived, base);
  auto& S = getSema();
  auto fakeLoc = GetValidSLoc(S);
  auto derivedType = clang::QualType::getFromOpaquePtr(derived);
  auto baseType = clang::QualType::getFromOpaquePtr(base);

  compat::SynthesizingCodeRAII RAII(&getInterp());
  return INTEROP_RETURN(S.IsDerivedFrom(fakeLoc, derivedType, baseType));
}

std::string GetFunctionArgDefault(TCppFunction_t func,
                                  TCppIndex_t param_index) {
  INTEROP_TRACE(func, param_index);
  auto* D = (clang::Decl*)func;
  clang::ParmVarDecl* PI = nullptr;

  if (auto* FD = llvm::dyn_cast_or_null<clang::FunctionDecl>(D))
    PI = FD->getParamDecl(param_index);

  else if (auto* FD = llvm::dyn_cast_or_null<clang::FunctionTemplateDecl>(D))
    PI = (FD->getTemplatedDecl())->getParamDecl(param_index);

  if (PI->hasDefaultArg()) {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    Expr* DefaultArgExpr = nullptr;
    compat::SynthesizingCodeRAII RAII(&getInterp());
    if (PI->hasUninstantiatedDefaultArg())
      DefaultArgExpr = PI->getUninstantiatedDefaultArg();
    else
      DefaultArgExpr = PI->getDefaultArg();
    DefaultArgExpr->printPretty(OS, nullptr, PrintingPolicy(LangOptions()));

    // FIXME: Floats are printed in clang with the precision of their underlying
    // representation and not as written. This is a deficiency in the printing
    // mechanism of clang which we require extra work to mitigate. For example
    // float PI = 3.14 is printed as 3.1400000000000001
    if (PI->getType()->isFloatingType()) {
      if (!Result.empty() && Result.back() == '.')
        return INTEROP_RETURN(Result);
      auto DefaultArgValue = std::stod(Result);
      std::ostringstream oss;
      oss << DefaultArgValue;
      Result = oss.str();
    }
    return INTEROP_RETURN(Result);
  }
  return INTEROP_RETURN("");
}

bool IsConstMethod(TCppFunction_t method) {
  INTEROP_TRACE(method);
  if (!method)
    return INTEROP_RETURN(false);

  auto* D = (clang::Decl*)method;
  if (auto* func = dyn_cast<CXXMethodDecl>(D))
    return INTEROP_RETURN(func->getMethodQualifiers().hasConst());

  return INTEROP_RETURN(false);
}

std::string GetFunctionArgName(TCppFunction_t func, TCppIndex_t param_index) {
  INTEROP_TRACE(func, param_index);
  auto* D = (clang::Decl*)func;
  clang::ParmVarDecl* PI = nullptr;

  if (auto* FD = llvm::dyn_cast_or_null<clang::FunctionDecl>(D))
    PI = FD->getParamDecl(param_index);
  else if (auto* FD = llvm::dyn_cast_or_null<clang::FunctionTemplateDecl>(D))
    PI = (FD->getTemplatedDecl())->getParamDecl(param_index);

  return INTEROP_RETURN(PI->getNameAsString());
}

std::string GetSpellingFromOperator(Operator Operator) {
  INTEROP_TRACE(Operator);
  return INTEROP_RETURN(
      clang::getOperatorSpelling((clang::OverloadedOperatorKind)Operator));
}

Operator GetOperatorFromSpelling(const std::string& op) {
  INTEROP_TRACE(op);
#define OVERLOADED_OPERATOR(Name, Spelling, Token, Unary, Binary, MemberOnly)  \
  if ((Spelling) == op) {                                                      \
    return INTEROP_RETURN((Operator)OO_##Name);                                \
  }
#include "clang/Basic/OperatorKinds.def"
  return INTEROP_RETURN(Operator::OP_None);
}

OperatorArity GetOperatorArity(TCppFunction_t op) {
  INTEROP_TRACE(op);
  Decl* D = static_cast<Decl*>(op);
  if (auto* FD = llvm::dyn_cast<FunctionDecl>(D)) {
    if (FD->isOverloadedOperator()) {
      switch (FD->getOverloadedOperator()) {
#define OVERLOADED_OPERATOR(Name, Spelling, Token, Unary, Binary, MemberOnly)  \
  case OO_##Name:                                                              \
    if ((Unary) && (Binary))                                                   \
      return INTEROP_RETURN(kBoth);                                            \
    if (Unary)                                                                 \
      return INTEROP_RETURN(kUnary);                                           \
    if (Binary)                                                                \
      return INTEROP_RETURN(kBinary);                                          \
    break;
#include "clang/Basic/OperatorKinds.def"
      default:
        break;
      }
    }
  }
  return INTEROP_RETURN((OperatorArity)~0U);
}

void GetOperator(TCppScope_t scope, Operator op,
                 std::vector<TCppFunction_t>& operators, OperatorArity kind) {
  INTEROP_TRACE(scope, op, INTEROP_OUT(operators), kind);
  Decl* D = static_cast<Decl*>(scope);
  compat::SynthesizingCodeRAII RAII(&getInterp());
  if (auto* CXXRD = llvm::dyn_cast_or_null<CXXRecordDecl>(D)) {
    auto fn = [&operators, kind, op](const RecordDecl* RD) {
      ASTContext& C = RD->getASTContext();
      DeclContextLookupResult Result =
          RD->lookup(C.DeclarationNames.getCXXOperatorName(
              (clang::OverloadedOperatorKind)op));
      for (auto* i : Result) {
        if (kind & GetOperatorArity(i))
          operators.push_back(i);
      }
      return true;
    };
    fn(CXXRD);
    CXXRD->forallBases(fn);
  } else if (auto* DC = llvm::dyn_cast_or_null<DeclContext>(D)) {
    ASTContext& C = getSema().getASTContext();
    DeclContextLookupResult Result =
        DC->lookup(C.DeclarationNames.getCXXOperatorName(
            (clang::OverloadedOperatorKind)op));

    for (auto* i : Result) {
      if (kind & GetOperatorArity(i))
        operators.push_back(i);
    }
  }
  return INTEROP_VOID_RETURN();
}

TCppObject_t Allocate(TCppScope_t scope, TCppIndex_t count) {
  INTEROP_TRACE(scope, count);
  return INTEROP_RETURN(
      (TCppObject_t)::operator new(Cpp::SizeOf(scope) * count));
}

void Deallocate(TCppScope_t scope, TCppObject_t address, TCppIndex_t count) {
  INTEROP_TRACE(scope, address, count);
  size_t bytes = Cpp::SizeOf(scope) * count;
  ::operator delete(address, bytes);
  return INTEROP_VOID_RETURN();
}

// FIXME: Add optional arguments to the operator new.
TCppObject_t Construct(compat::Interpreter& interp, TCppScope_t scope,
                       void* arena /*=nullptr*/, TCppIndex_t count /*=1UL*/) {

  if (!Cpp::IsConstructor(scope) && !Cpp::IsClass(scope))
    return nullptr;
  if (Cpp::IsClass(scope) && !HasDefaultConstructor(scope))
    return nullptr;

  TCppFunction_t ctor = nullptr;
  if (Cpp::IsClass(scope))
    ctor = Cpp::GetDefaultConstructor(scope);
  else // a ctor
    ctor = scope;

  if (JitCall JC = MakeFunctionCallable(&interp, ctor)) {
    // invoke the constructor (placement/heap) in one shot
    // flag is non-null for placement new, null for normal new
    void* is_arena = arena ? reinterpret_cast<void*>(1) : nullptr;
    void* result = arena;
    JC.InvokeConstructor(&result, count, /*args=*/{}, is_arena);
    return result;
  }
  return nullptr;
}

TCppObject_t Construct(TCppScope_t scope, void* arena /*=nullptr*/,
                       TCppIndex_t count /*=1UL*/) {
  INTEROP_TRACE(scope, arena, count);
  return INTEROP_RETURN(Construct(getInterp(), scope, arena, count));
}

bool Destruct(compat::Interpreter& interp, TCppObject_t This, const Decl* Class,
              bool withFree, TCppIndex_t nary) {
  if (auto wrapper = make_dtor_wrapper(interp, Class)) {
    (*wrapper)(This, nary, withFree);
    return true;
  }
  return false;
  // FIXME: Enable stronger diagnostics
}

bool Destruct(TCppObject_t This, TCppConstScope_t scope,
              bool withFree /*=true*/, TCppIndex_t count /*=0UL*/) {
  INTEROP_TRACE(This, scope, withFree, count);
  const auto* Class = static_cast<const Decl*>(scope);
  return INTEROP_RETURN(Destruct(getInterp(), This, Class, withFree, count));
}

class StreamCaptureInfo {
  FILE* m_TempFile = nullptr;
  int m_FD = -1;
  int m_DupFD = -1;
  bool m_OwnsFile = true;

public:
#ifdef _MSC_VER
  StreamCaptureInfo(int FD)
      : m_TempFile{[]() {
          FILE* stream = nullptr;
          errno_t err;
          err = tmpfile_s(&stream);
          if (err)
            printf("Cannot create temporary file!\n");
          return stream;
        }()},
        m_FD(FD) {
#else
  StreamCaptureInfo(int FD) : m_FD(FD) {
#if !defined(CPPINTEROP_USE_CLING) && !defined(_WIN32)
    auto& I = getInterp();
    if (I.isOutOfProcess()) {
      // Use interpreter-managed redirection file for out-of-process
      // redirection. Since, we are using custom pipes instead of stdout, sterr,
      // it is kind of necessary to have this complication in StreamCaptureInfo.

      // TODO(issues/733): Refactor the stream redirection
      FILE* redirected = I.getRedirectionFileForOutOfProcess(FD);
      if (redirected) {
        m_TempFile = redirected;
        m_OwnsFile = false;
        if (ftruncate(fileno(m_TempFile), 0) != 0)
          perror("ftruncate");
        if (lseek(fileno(m_TempFile), 0, SEEK_SET) == -1)
          perror("lseek");
      }
    } else {
      m_TempFile = tmpfile();
    }
#else
    m_TempFile = tmpfile();
#endif
#endif

    if (!m_TempFile) {
      perror("StreamCaptureInfo: Unable to create temp file");
      return;
    }

    m_DupFD = dup(FD);

    // Flush now or can drop the buffer when dup2 is called with Fd later.
    // This seems only necessary when piping stdout or stderr, but do it
    // for ttys to avoid over complicated code for minimal benefit.
    ::fflush(FD == STDOUT_FILENO ? stdout : stderr);
    if (dup2(fileno(m_TempFile), FD) < 0)
      perror("StreamCaptureInfo:");
  }
  StreamCaptureInfo(const StreamCaptureInfo&) = delete;
  StreamCaptureInfo& operator=(const StreamCaptureInfo&) = delete;
  StreamCaptureInfo(StreamCaptureInfo&&) = delete;
  StreamCaptureInfo& operator=(StreamCaptureInfo&&) = delete;

  ~StreamCaptureInfo() {
    assert(m_DupFD == -1 && "Captured output not used?");
    // Only close the temp file if we own it
    if (m_OwnsFile && m_TempFile)
      fclose(m_TempFile);
  }

  std::string GetCapturedString() {
    assert(m_DupFD != -1 && "Multiple calls to GetCapturedString");

    fflush(nullptr);
    if (dup2(m_DupFD, m_FD) < 0)
      perror("StreamCaptureInfo:");
    // Go to the end of the file.
    if (fseek(m_TempFile, 0L, SEEK_END) != 0)
      perror("StreamCaptureInfo:");

    // Get the size of the file.
    long bufsize = ftell(m_TempFile);
    if (bufsize == -1) {
      perror("StreamCaptureInfo:");
      close(m_DupFD);
      m_DupFD = -1;
      return "";
    }

    // Allocate our buffer to that size.
    std::unique_ptr<char[]> content(new char[bufsize + 1]);

    // Go back to the start of the file.
    if (fseek(m_TempFile, 0L, SEEK_SET) != 0)
      perror("StreamCaptureInfo:");

    // Read the entire file into memory.
    size_t newLen = fread(content.get(), sizeof(char), bufsize, m_TempFile);
    if (ferror(m_TempFile) != 0)
      fputs("Error reading file", stderr);
    else
      content[newLen++] = '\0'; // Just to be safe.

    std::string result = content.get();
    close(m_DupFD);
    m_DupFD = -1;
#if !defined(_WIN32) && !defined(CPPINTEROP_USE_CLING)
    auto& I = getInterp();
    if (I.isOutOfProcess()) {
      int fd = fileno(m_TempFile);
      if (ftruncate(fd, 0) != 0)
        perror("ftruncate");
      if (lseek(fd, 0, SEEK_SET) == -1)
        perror("lseek");
    }
#endif
    return result;
  }
};

static std::stack<StreamCaptureInfo>& GetRedirectionStack() {
  static std::stack<StreamCaptureInfo> sRedirectionStack;
  return sRedirectionStack;
}

void BeginStdStreamCapture(CaptureStreamKind fd_kind) {
  INTEROP_TRACE(fd_kind);
  GetRedirectionStack().emplace((int)fd_kind);
  return INTEROP_VOID_RETURN();
}

std::string EndStdStreamCapture() {
  INTEROP_TRACE();
  assert(GetRedirectionStack().size());
  StreamCaptureInfo& SCI = GetRedirectionStack().top();
  std::string result = SCI.GetCapturedString();
  GetRedirectionStack().pop();
  return INTEROP_RETURN(result);
}

void CodeComplete(std::vector<std::string>& Results, const char* code,
                  unsigned complete_line /* = 1U */,
                  unsigned complete_column /* = 1U */) {
  INTEROP_TRACE(INTEROP_OUT(Results), code, complete_line, complete_column);
  compat::codeComplete(Results, getInterp(), code, complete_line,
                       complete_column);
  return INTEROP_VOID_RETURN();
}

int Undo(unsigned N) {
  INTEROP_TRACE(N);
  compat::SynthesizingCodeRAII RAII(&getInterp());
#ifdef CPPINTEROP_USE_CLING
  getInterp().unload(N);
  return INTEROP_RETURN(compat::Interpreter::kSuccess);
#else
  return INTEROP_RETURN(getInterp().undo(N));
#endif
}

#ifndef _WIN32
pid_t GetExecutorPID() {
  INTEROP_TRACE();
#ifdef LLVM_BUILT_WITH_OOP_JIT
  auto& I = getInterp();
  return INTEROP_RETURN(I.getOutOfProcessExecutorPID());
#endif
  return INTEROP_RETURN(getpid());
}

#endif

} // namespace CppImpl
