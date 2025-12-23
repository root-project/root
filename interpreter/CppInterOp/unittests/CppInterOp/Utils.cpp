#include "Utils.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/Basic/Version.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Interpreter/PartialTranslationUnit.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"

#include "llvm/TargetParser/Triple.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

using namespace clang;
using namespace llvm;

#if defined(ENABLE_DISPATCH_TESTS)
#define DISPATCH_API(name, type) CppAPIType::name Cpp::name = nullptr;
CPPINTEROP_API_TABLE
#undef DISPATCH_API
namespace {
struct DispatchInitializer {
  DispatchInitializer() {
    if (!Cpp::LoadDispatchAPI(CPPINTEROP_LIB_PATH)) {
      std::abort();
    }
  }
  ~DispatchInitializer() { Cpp::UnloadDispatchAPI(); }
  DispatchInitializer(const DispatchInitializer&) = delete;
  DispatchInitializer& operator=(const DispatchInitializer&) = delete;
  DispatchInitializer(DispatchInitializer&&) noexcept = default;
  DispatchInitializer& operator=(DispatchInitializer&&) noexcept = default;
};
// FIXME: Make this threadsafe by moving it as a function static.
DispatchInitializer g_dispatch_init;
} // namespace
#endif

namespace TestUtils {
TestConfig current_config;
std::vector<const char*> GetInterpreterArgs(
    const std::vector<const char*>& base_args) {
  auto args = base_args;
  if (current_config.use_oop_jit) {
    args.push_back("--use-oop-jit");
  }
  return args;
}
}

void TestUtils::GetAllTopLevelDecls(
    const std::string& code, std::vector<Decl*>& Decls,
    bool filter_implicitGenerated /* = false */,
    const std::vector<const char*>& interpreter_args /* = {} */) {
  Cpp::CreateInterpreter(interpreter_args, {});
#ifdef CPPINTEROP_USE_CLING
  cling::Transaction *T = nullptr;
  Interp->declare(code, &T);

  for (auto DCI = T->decls_begin(), E = T->decls_end(); DCI != E; ++DCI) {
    if (DCI->m_Call != cling::Transaction::kCCIHandleTopLevelDecl)
      continue;
    for (Decl *D : DCI->m_DGR) {
      if (filter_implicitGenerated && D->isImplicit())
        continue;
      Decls.push_back(D);
    }
  }
#else
  PartialTranslationUnit *T = nullptr;
  Interp->process(code, /*Value*/nullptr, &T);
  for (auto *D : T->TUPart->decls()) {
    if (filter_implicitGenerated && D->isImplicit())
      continue;
    Decls.push_back(D);
  }
#endif
}

void TestUtils::GetAllSubDecls(Decl *D, std::vector<Decl*>& SubDecls,
                               bool filter_implicitGenerated /* = false */) {
  if (!isa_and_nonnull<DeclContext>(D))
    return;
  DeclContext *DC = cast<DeclContext>(D);
  for (auto *Di : DC->decls()) {
    if (filter_implicitGenerated && Di->isImplicit())
      continue;
    SubDecls.push_back(Di);
  }
}

bool IsTargetX86() {
#ifndef CPPINTEROP_USE_CLING
  llvm::Triple triple(Interp->getCompilerInstance()->getTargetOpts().Triple);
#else
  llvm::Triple triple(Interp->getCI()->getTargetOpts().Triple);
#endif
  return triple.isX86();
}

const char* get_c_string(CXString string) {
  return static_cast<const char*>(string.data);
}

void dispose_string(CXString string) {
  if (string.private_flags == 1 && string.data)
    free(const_cast<void*>(string.data));
}

CXScope make_scope(const clang::Decl* D, const CXInterpreter I) {
  return {CXCursor_UnexposedDecl, 0, {D, nullptr, I}};
}
