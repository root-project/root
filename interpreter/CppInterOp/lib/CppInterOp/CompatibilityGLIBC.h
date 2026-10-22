//--------------------------------------------------------------------*- C++ -*-
// CppInterOp Compatibility - glibc
//------------------------------------------------------------------------------
#ifndef CPPINTEROP_COMPATIBILITY_GLIBC_H
#define CPPINTEROP_COMPATIBILITY_GLIBC_H

// Declares at_quick_exit, and defines __GLIBC__ before the test below.
#include <cstdlib>

// ===========================================================================
// Resolve glibc's libc_nonshared.a symbols for the in-process JIT.
//
// glibc defines at_quick_exit, atexit, pthread_atfork, and
// __stack_chk_fail_local in libc_nonshared.a rather than libc.so: every ELF
// module links a private, non-exported copy, dlsym sees none of them, and a
// jitted reference fails with "Symbols not found: [ at_quick_exit ]".
//
// __stack_chk_fail_local is pure code and resolves to this library's copy
// (GlibcNonsharedSymbolGenerator). at_quick_exit and pthread_atfork are
// registration functions, and a jitted handler must not outlive the JIT
// that owns it, so they resolve to JIT-side shims (addGlibcNonsharedShims)
// backed by the host-side registry below: quick-exit handlers run once, at
// the earlier of a real quick_exit or the owning interpreter's teardown;
// atfork entries fire on fork while the interpreter lives and are dropped
// at teardown. atexit needs no entry -- LLJIT's platform support defines a
// JIT-aware atexit per JITDylib and runs its handlers on deinitialize.
// Other libc_nonshared.a members vary by glibc build and are omitted.
// ===========================================================================
#ifdef __GLIBC__
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <utility>
#include <vector>

namespace compat {

// The reserved name is the point: this is glibc's own symbol.
// NOLINTNEXTLINE(bugprone-reserved-identifier, readability-identifier-naming)
extern "C" void __stack_chk_fail_local();

inline const llvm::StringMap<void*>& glibcNonsharedSymbols() {
  static const llvm::StringMap<void*> Symbols = {
      {"__stack_chk_fail_local",
       // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
       reinterpret_cast<void*>(&__stack_chk_fail_local)},
  };
  return Symbols;
}

/// Host-side registry for handlers jitted code passes to the shimmed
/// at_quick_exit/pthread_atfork. glibc only ever sees this library's own
/// hooks; entries are tagged with the owning interpreter and flushed before
/// its JIT is torn down, so no jitted pointer outlives the code it targets.
class JitGlibcHandlerRegistry {
public:
  using Handler = void (*)();

  static JitGlibcHandlerRegistry& instance() {
    static JitGlibcHandlerRegistry R;
    return R;
  }

  int addQuickExit(void* Owner, Handler Fn) {
    std::lock_guard<std::mutex> L(M);
    if (!QuickExitHookInstalled) {
      // Host-resident hook: live handlers still run on a real quick_exit.
      if (::at_quick_exit(&runAllQuickExitHandlers) != 0)
        return -1;
      QuickExitHookInstalled = true;
    }
    QuickExit.push_back({Owner, Fn});
    return 0;
  }

  int addAtfork(void* Owner, Handler Prepare, Handler Parent, Handler Child) {
    std::lock_guard<std::mutex> L(M);
    if (!AtforkHookInstalled) {
      if (int Err = ::pthread_atfork(&runAtforkPrepare, &runAtforkParent,
                                     &runAtforkChild))
        return Err;
      AtforkHookInstalled = true;
    }
    Atfork.push_back({Owner, Prepare, Parent, Child});
    return 0;
  }

  /// Runs the owner's quick-exit handlers (LIFO, exactly once) and drops its
  /// atfork entries. Call while the owner's JIT can still execute code.
  void flushOwner(void* Owner) {
    while (Handler Fn = takeLastQuickExit(Owner))
      Fn();
    std::lock_guard<std::mutex> L(M);
    Atfork.erase(
        std::remove_if(Atfork.begin(), Atfork.end(),
                       [&](const AtforkEntry& E) { return E.Owner == Owner; }),
        Atfork.end());
  }

  /// The hook registered with ::at_quick_exit: drains every live handler,
  /// LIFO. Public so tests can drive the real quick_exit path in-process.
  static void runAllQuickExitHandlers() {
    while (Handler Fn = instance().takeLastQuickExit(/*Owner=*/nullptr))
      Fn();
  }

private:
  struct QuickExitEntry {
    void* Owner;
    Handler Fn;
  };
  struct AtforkEntry {
    void* Owner;
    Handler Prepare;
    Handler Parent;
    Handler Child;
  };

  // Handlers run outside the lock; they may legally re-register.
  Handler takeLastQuickExit(void* Owner) {
    std::lock_guard<std::mutex> L(M);
    for (auto It = QuickExit.rbegin(); It != QuickExit.rend(); ++It) {
      if (Owner && It->Owner != Owner)
        continue;
      Handler Fn = It->Fn;
      QuickExit.erase(std::next(It).base());
      return Fn;
    }
    return nullptr;
  }

  // The prepare hook snapshots under the lock; the parent/child hooks run
  // lock-free off the snapshot (another thread could hold the mutex across
  // fork and the child would inherit it locked).
  static void runAtforkPrepare() {
    JitGlibcHandlerRegistry& R = instance();
    {
      std::lock_guard<std::mutex> L(R.M);
      R.ForkSnapshot = R.Atfork;
    }
    for (auto It = R.ForkSnapshot.rbegin(); It != R.ForkSnapshot.rend(); ++It)
      if (It->Prepare)
        It->Prepare();
  }
  static void runAtforkParent() {
    for (const AtforkEntry& E : instance().ForkSnapshot)
      if (E.Parent)
        E.Parent();
  }
  static void runAtforkChild() {
    for (const AtforkEntry& E : instance().ForkSnapshot)
      if (E.Child)
        E.Child();
  }

  std::mutex M;
  std::vector<QuickExitEntry> QuickExit;
  std::vector<AtforkEntry> Atfork;
  std::vector<AtforkEntry> ForkSnapshot;
  bool QuickExitHookInstalled = false;
  bool AtforkHookInstalled = false;
};

/// C-signature entry points the JIT-side shims call; bound by name via
/// absoluteSymbols in installGlibcNonsharedSupport.
inline int jitAtQuickExit(void* Owner, void (*Fn)()) {
  return JitGlibcHandlerRegistry::instance().addQuickExit(Owner, Fn);
}
inline int jitPthreadAtfork(void* Owner, void (*Prepare)(), void (*Parent)(),
                            void (*Child)()) {
  return JitGlibcHandlerRegistry::instance().addAtfork(Owner, Prepare, Parent,
                                                       Child);
}

/// Adds an IR module to the main JITDylib defining at_quick_exit and
/// pthread_atfork shims that forward, with the owning interpreter baked in,
/// to the registry entry points above -- the pattern LLJIT's generic
/// platform uses for atexit. An IR module rather than interpreted source so
/// Undo cannot drop it and no PTU bookkeeping is disturbed.
inline llvm::Error addGlibcNonsharedShims(llvm::orc::LLJIT& J, void* Owner) {
  auto TSCtx = std::make_unique<llvm::LLVMContext>();
  auto M =
      std::make_unique<llvm::Module>("<cppinterop-glibc-nonshared>", *TSCtx);
  M->setDataLayout(J.getDataLayout());
#if LLVM_VERSION_MAJOR < 21
  M->setTargetTriple(J.getTargetTriple().str());
#else
  M->setTargetTriple(J.getTargetTriple());
#endif

  llvm::Type* IntTy = llvm::Type::getInt32Ty(*TSCtx);
  llvm::PointerType* PtrTy = llvm::PointerType::getUnqual(*TSCtx);

  auto AddShim = [&](llvm::StringRef Name, llvm::StringRef Helper,
                     unsigned NumArgs) {
    llvm::SmallVector<llvm::Type*, 4> ShimArgs(NumArgs, PtrTy);
    llvm::Function* F = llvm::Function::Create(
        llvm::FunctionType::get(IntTy, ShimArgs, /*isVarArg=*/false),
        llvm::GlobalValue::ExternalLinkage, Name, M.get());
    llvm::SmallVector<llvm::Type*, 5> HelperArgs(NumArgs + 1, PtrTy);
    llvm::FunctionCallee Callee = M->getOrInsertFunction(
        Helper, llvm::FunctionType::get(IntTy, HelperArgs, false));
    llvm::IRBuilder<> B(llvm::BasicBlock::Create(*TSCtx, "entry", F));
    llvm::Value* OwnerV = B.CreateIntToPtr(
        llvm::ConstantInt::get(
            M->getDataLayout().getIntPtrType(*TSCtx),
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            reinterpret_cast<uintptr_t>(Owner)),
        PtrTy);
    llvm::SmallVector<llvm::Value*, 5> Args{OwnerV};
    for (llvm::Argument& A : F->args())
      Args.push_back(&A);
    B.CreateRet(B.CreateCall(Callee, Args));
  };
  AddShim("at_quick_exit", "__cppinterop_at_quick_exit", 1);
  AddShim("pthread_atfork", "__cppinterop_pthread_atfork", 3);

  return J.addIRModule(
      llvm::orc::ThreadSafeModule(std::move(M), std::move(TSCtx)));
}

} // namespace compat
#endif // __GLIBC__

#endif // CPPINTEROP_COMPATIBILITY_GLIBC_H
