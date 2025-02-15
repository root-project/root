//===- AutoLoadEPC.h -- Executor process control for auto-loading -*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_CLINGEPC_H
#define LLVM_EXECUTIONENGINE_ORC_CLINGEPC_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/Shared/AutoLoadDylibLookup.h"
#include "llvm/ExecutionEngine/Orc/Shared/AutoLoadDylibUtils.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/TargetProcessControlTypes.h"
#include "llvm/ExecutionEngine/Orc/Shared/WrapperFunctionUtils.h"
#include "llvm/ExecutionEngine/Orc/SymbolStringPool.h"
#include "llvm/ExecutionEngine/Orc/TaskDispatch.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/MSVCErrorWorkarounds.h"
#include "llvm/TargetParser/Triple.h"

#include <future>
#include <mutex>
#include <vector>

namespace llvm {
namespace orc {

// Temporary implementation for supporting callbacks in AutoLoadEPC.
class ORCCallbacks {
public:
  virtual bool LibraryLoadingFailed(const std::string &, const std::string &,
                                    bool, bool) { return 0; }
  virtual void LibraryUnloaded(const void *, llvm::StringRef) {}
  virtual void LibraryLoaded(const void *, llvm::StringRef) {}
};

/// A ExecutorProcessControl implementation targeting the current process.
class AutoLoadEPC : public ExecutorProcessControl,
                                    private InProcessMemoryAccess {
public:
  AutoLoadEPC(
      std::shared_ptr<SymbolStringPool> SSP, std::unique_ptr<TaskDispatcher> D,
      Triple TargetTriple, unsigned PageSize,
      std::unique_ptr<jitlink::JITLinkMemoryManager> MemMgr);

  /// Create a AutoLoadEPC with the given symbol string pool
  /// and memory manager. If no symbol string pool is given then one will be
  /// created. If no memory manager is given a jitlink::InProcessMemoryManager
  /// will be created and used by default.
  static Expected<std::unique_ptr<AutoLoadEPC>>
  Create(std::shared_ptr<SymbolStringPool> SSP = nullptr,
         std::unique_ptr<TaskDispatcher> D = nullptr,
         std::unique_ptr<jitlink::JITLinkMemoryManager> MemMgr = nullptr);

  /// Describes the result of loading a library.
  enum LoadLibResult {
    kLoadLibSuccess,       ///< library loaded successfully
    kLoadLibAlreadyLoaded, ///< library was already loaded
    kLoadLibNotFound,      ///< library was not found
    kLoadLibLoadError,     ///< loading the library failed
    kLoadLibNumResults
  };

  Expected<tpctypes::DylibHandle> loadDylib(const char *DylibPath) override;
  Expected<tpctypes::DylibHandle> loadDylib(const char *DylibPath, bool useDLOpen = true);
  LoadLibResult loadDylib(StringRef DylibPath, bool Permanent,
                          bool Resolved = false);
  void unloadDylib(StringRef DylibPath);

  Expected<std::vector<tpctypes::LookupResult>>
  lookupSymbols(ArrayRef<LookupRequest> Request) override;

  void resolveSymbolsAsync(ArrayRef<SymbolLookupSet> Request,
                           ResolveSymbolsCompleteFn F) override;

  Expected<int32_t> runAsMain(ExecutorAddr MainFnAddr,
                              ArrayRef<std::string> Args) override;

  Expected<int32_t> runAsVoidFunction(ExecutorAddr VoidFnAddr) override;

  Expected<int32_t> runAsIntFunction(ExecutorAddr IntFnAddr, int Arg) override;

  void callWrapperAsync(ExecutorAddr WrapperFnAddr,
                        IncomingWFRHandler OnComplete,
                        ArrayRef<char> ArgBuffer) override;

  Error disconnect() override;

  std::string lookupDylib(StringRef Path) { return DylibLookup.lookupLibrary(Path); }
  AutoLoadDynamicLibraryLookup &getDylibLookup() { return DylibLookup; }
  const AutoLoadDynamicLibraryLookup &getDylibLookup() const { return DylibLookup; }

  bool IsDylibLoaded(StringRef DylibPath) const {
    // return LoadedLibraries.find(DylibLookup.normalizePath(DylibPath)) !=
    //        LoadedLibraries.end();
    return DylibLookup.isLibraryLoaded(DylibPath);
  }

  void setCallbacks(ORCCallbacks *C) { Callbacks = C; }
  ORCCallbacks *getCallbacks() { return Callbacks; }

private:
  static shared::CWrapperFunctionResult
  jitDispatchViaWrapperFunctionManager(void *Ctx, const void *FnTag,
                                       const char *Data, size_t Size);

  using DylibHandle = void *;
  using DynamicLibs = llvm::DenseMap<DylibHandle, std::string>;

  std::unique_ptr<jitlink::JITLinkMemoryManager> OwnedMemMgr;
  AutoLoadDynamicLibraryLookup DylibLookup;
  DynamicLibs Dylibs;
  // llvm::StringSet<> LoadedLibraries;
  ORCCallbacks *Callbacks;
  char GlobalManglingPrefix = 0;
};

} // end namespace orc
} // end namespace llvm
#endif // LLVM_EXECUTIONENGINE_ORC_CLINGEPC_H