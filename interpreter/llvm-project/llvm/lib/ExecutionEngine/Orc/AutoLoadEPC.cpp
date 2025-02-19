//===--------------- AutoLoadEPC.cpp - EPC for auto-loading ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/AutoLoadEPC.h"

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Shared/OrcRTBridge.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/RegisterEHFrames.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/TargetExecutionUtils.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Process.h"
#include "llvm/TargetParser/Host.h"

// #if defined(LLVM_ON_UNIX)
// #include <array>
// #include <atomic>
// #include <cxxabi.h>
// #include <dlfcn.h>
// #include <errno.h>
// #include <fcntl.h>
// #include <string>
// #include <sys/mman.h>
// #include <unistd.h>

// // PATH_MAX
// #ifdef __APPLE__
// #include <sys/syslimits.h>
// #else
// #include <limits.h>
// #endif

// namespace platform {

// static void DLErr(std::string *Err) {
//   if (!Err)
//     return;
//   if (const char *DyLibError = ::dlerror())
//     *Err = DyLibError;
// }

// static void *DLOpen(const std::string &Path, std::string *Err) {
//   void *Lib = dlopen(Path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
//   DLErr(Err);
//   return Lib;
// }

// static void DLClose(const void *Lib, std::string *Err) {
//   ::dlclose(const_cast<void *>(Lib));
//   DLErr(Err);
// }

// } // namespace platform
// #endif

#define DEBUG_TYPE "orc"

namespace llvm {
namespace orc {

AutoLoadEPC::AutoLoadEPC(
    std::shared_ptr<SymbolStringPool> SSP, std::unique_ptr<TaskDispatcher> D,
    Triple TargetTriple, unsigned PageSize,
    std::unique_ptr<jitlink::JITLinkMemoryManager> MemMgr)
    : ExecutorProcessControl(std::move(SSP), std::move(D)),
      InProcessMemoryAccess(TargetTriple.isArch64Bit()) {

  OwnedMemMgr = std::move(MemMgr);
  if (!OwnedMemMgr)
    OwnedMemMgr = std::make_unique<jitlink::InProcessMemoryManager>(
        sys::Process::getPageSizeEstimate());

  this->TargetTriple = std::move(TargetTriple);
  this->PageSize = PageSize;
  this->MemMgr = OwnedMemMgr.get();
  this->MemAccess = this;
  this->JDI = {ExecutorAddr::fromPtr(jitDispatchViaWrapperFunctionManager),
               ExecutorAddr::fromPtr(this)};
  if (this->TargetTriple.isOSBinFormatMachO())
    GlobalManglingPrefix = '_';

  this->BootstrapSymbols[rt::RegisterEHFrameSectionWrapperName] =
      ExecutorAddr::fromPtr(&llvm_orc_registerEHFrameSectionWrapper);
  this->BootstrapSymbols[rt::DeregisterEHFrameSectionWrapperName] =
      ExecutorAddr::fromPtr(&llvm_orc_deregisterEHFrameSectionWrapper);
  DylibLookup.initializeDynamicLoader([](llvm::StringRef) { /*ignore*/
                                                            return false;
  });
}

Expected<std::unique_ptr<AutoLoadEPC>>
AutoLoadEPC::Create(std::shared_ptr<SymbolStringPool> SSP,
                       std::unique_ptr<TaskDispatcher> D,
                       std::unique_ptr<jitlink::JITLinkMemoryManager> MemMgr) {

  if (!SSP)
    SSP = std::make_shared<SymbolStringPool>();

  if (!D) {
#if LLVM_ENABLE_THREADS
    D = std::make_unique<DynamicThreadPoolTaskDispatcher>();
#else
    D = std::make_unique<InPlaceTaskDispatcher>();
#endif
  }

  auto PageSize = sys::Process::getPageSize();
  if (!PageSize)
    return PageSize.takeError();

  Triple TT(sys::getProcessTriple());

  return std::make_unique<AutoLoadEPC>(std::move(SSP), std::move(D),
                                          std::move(TT), *PageSize,
                                          std::move(MemMgr));
}

Expected<tpctypes::DylibHandle>
AutoLoadEPC::loadDylib(const char *DylibPath) {
  return loadDylib(DylibPath, false);
}

Expected<tpctypes::DylibHandle>
AutoLoadEPC::loadDylib(const char *DylibPath, bool useDLOpen) {
  std::string ErrMsg;
  DylibHandle DylibH;
  if (useDLOpen) {
    DylibH = DLOpen(DylibPath, &ErrMsg);
    if (!DylibH)
      return make_error<StringError>(std::move(ErrMsg),
                                     inconvertibleErrorCode());
  } else {
    sys::DynamicLibrary Dylib =
        sys::DynamicLibrary::getPermanentLibrary(DylibPath, &ErrMsg);
    if (!Dylib.isValid())
      return make_error<StringError>(std::move(ErrMsg),
                                     inconvertibleErrorCode());
      DylibH = Dylib.getOSSpecificHandle();
  }
  return ExecutorAddr::fromPtr(DylibH);
}

AutoLoadEPC::LoadLibResult
AutoLoadEPC::loadDylib(StringRef DylibPath, bool Permanent, bool resolved) {

  std::string canonicalLoadedLib;
  if (resolved) {
    canonicalLoadedLib = DylibPath.str();
  } else {
    canonicalLoadedLib = DylibLookup.lookupLibrary(DylibPath);
    if (canonicalLoadedLib.empty())
      return kLoadLibNotFound;
  }

  if (DylibLookup.isLibraryLoaded(DylibPath))
    return kLoadLibAlreadyLoaded;

  DylibHandle dyLibHandle;

  // Helper function for handling library loading errors
  auto handleLibraryLoadingError =
      [&](const std::string &ErrMsg) -> AutoLoadEPC::LoadLibResult {
    if (Callbacks && Callbacks->LibraryLoadingFailed(ErrMsg, DylibPath.str(),
                                                     Permanent, resolved)) {
      return kLoadLibSuccess;
    }
    LLVM_DEBUG(
        { dbgs() << "orc::AutoLoadEPC::loadLibrary(): " << ErrMsg << '\n'; });
    return kLoadLibLoadError;
  };

  std::string ErrMsg;
  // if (Permanent) {
  //   auto Dylib = sys::DynamicLibrary::getPermanentLibrary(
  //       DylibPath.str().c_str(), &ErrMsg);
  //   if (!Dylib.isValid()) {
  //     return handleLibraryLoadingError(ErrMsg);
  //   }
  //   dyLibHandle = Dylib.getOSSpecificHandle();
  // } else {
    dyLibHandle = DLOpen(canonicalLoadedLib, &ErrMsg);
    if (!dyLibHandle) {
      return handleLibraryLoadingError(ErrMsg);
    }
  // }

  // Notify callbacks if the library was loaded successfully
  if (Callbacks) {
    Callbacks->LibraryLoaded(dyLibHandle, canonicalLoadedLib);
  }

  // Insert the loaded library into the dynamic libraries map
  auto insertionResult = Dylibs.insert(
      std::pair<DylibHandle, std::string>(dyLibHandle, canonicalLoadedLib));
  if (!insertionResult.second) {
    return kLoadLibAlreadyLoaded;
  }

  // Add the library to the lookup for future references
  DylibLookup.addLoadedLib(canonicalLoadedLib);

  return kLoadLibSuccess;
}

void AutoLoadEPC::unloadDylib(StringRef DylibPath) {
  std::string canonicalLoadedLib = DylibLookup.lookupLibrary(DylibPath);
  if (!DylibLookup.isLibraryLoaded(canonicalLoadedLib))
    return;

  DylibHandle dyLibHandle = nullptr;
  for (DynamicLibs::const_iterator I = Dylibs.begin(), E = Dylibs.end(); I != E;
       ++I) {
    if (I->second == canonicalLoadedLib) {
      dyLibHandle = I->first;
      break;
    }
  }

  // TODO: !permanent case

  //   std::string errMsg;
  //   DLClose(dyLibHandle, &errMsg);
  // sys::DynamicLibrary Dylib(dyLibHandle);
  // sys::DynamicLibrary::closeLibrary(Dylib);

  std::string errMsg;
  DLClose(dyLibHandle, &errMsg);
  if (Callbacks)
    Callbacks->LibraryUnloaded(dyLibHandle, canonicalLoadedLib);

  Dylibs.erase(dyLibHandle);
  DylibLookup.eraseLib(canonicalLoadedLib);
  // LoadedLibraries.erase(canonicalLoadedLib);
  return;
}

Expected<std::vector<tpctypes::LookupResult>>
AutoLoadEPC::lookupSymbols(ArrayRef<LookupRequest> Request) {
  std::vector<tpctypes::LookupResult> R;

  for (auto &Elem : Request) {
    sys::DynamicLibrary Dylib(Elem.Handle.toPtr<void *>());
    R.push_back(std::vector<ExecutorSymbolDef>());
    for (auto &KV : Elem.Symbols) {
      auto &Sym = KV.first;
      std::string Tmp((*Sym).data() + !!GlobalManglingPrefix,
                      (*Sym).size() - !!GlobalManglingPrefix);
      void *Addr = Dylib.getAddressOfSymbol(Tmp.c_str());
      if (!Addr && KV.second == SymbolLookupFlags::RequiredSymbol) {
        // FIXME: Collect all failing symbols before erroring out.
        SymbolNameVector MissingSymbols;
        MissingSymbols.push_back(Sym);
        return make_error<SymbolsNotFound>(SSP, std::move(MissingSymbols));
      }
      // FIXME: determine accurate JITSymbolFlags.
      R.back().push_back(
          {ExecutorAddr::fromPtr(Addr), JITSymbolFlags::Exported});
    }
  }

  return R;
}

void AutoLoadEPC::resolveSymbolsAsync(ArrayRef<SymbolLookupSet> Request,
                                         ResolveSymbolsCompleteFn Complete) {
  std::vector<ResolveResult> R;
  for (auto &Symbols : Request) {
    R.emplace_back();
    BloomFilter Filter;
    tpctypes::LookupResult &Result = R.back().SymbolDef;
    for (auto &KV : Symbols) {
      auto &Sym = KV.first;
      std::string Tmp((*Sym).data() + !!GlobalManglingPrefix,
                      (*Sym).size() - !!GlobalManglingPrefix);
      void *Addr = sys::DynamicLibrary::SearchForAddressOfSymbol(Tmp.c_str());

      if (Addr) {
        Result.push_back(
            {ExecutorAddr::fromPtr(Addr), JITSymbolFlags::Exported});
        continue;
      }

      auto lib = DylibLookup.searchLibrariesForSymbol(*Sym);
      auto canonicalLoadedLib = DylibLookup.lookupLibrary(lib);
      if (canonicalLoadedLib.empty()) {
        if (!Filter.IsInitialized())
          DylibLookup.BuildGlobalBloomFilter(Filter);
        Result.push_back(ExecutorSymbolDef());
      } else {
        auto H = loadDylib(canonicalLoadedLib.c_str(), true);
        if (!H)
          return Complete(H.takeError());

        DylibLookup.addLoadedLib(canonicalLoadedLib);
        // LoadedLibraries.insert(canonicalLoadedLib);
        sys::DynamicLibrary Dylib(H->toPtr<void *>());
        void *Addr = Dylib.getAddressOfSymbol(Tmp.c_str());
        if (!Addr && KV.second == SymbolLookupFlags::RequiredSymbol) {
          // FIXME: Collect all failing symbols before erroring out.
          SymbolNameVector MissingSymbols;
          MissingSymbols.push_back(Sym);
          return Complete(
              make_error<SymbolsNotFound>(SSP, std::move(MissingSymbols)));
        }
        Result.push_back(
            {ExecutorAddr::fromPtr(Addr), JITSymbolFlags::Exported});
      }
    }
    if (Filter.IsInitialized())
      R.back().Filter.emplace(std::move(Filter));
  }
  Complete(std::move(R));
}

Expected<int32_t> AutoLoadEPC::runAsMain(ExecutorAddr MainFnAddr,
                                            ArrayRef<std::string> Args) {
  using MainTy = int (*)(int, char *[]);
  return orc::runAsMain(MainFnAddr.toPtr<MainTy>(), Args);
}

Expected<int32_t> AutoLoadEPC::runAsVoidFunction(ExecutorAddr VoidFnAddr) {
  using VoidTy = int (*)();
  return orc::runAsVoidFunction(VoidFnAddr.toPtr<VoidTy>());
}

Expected<int32_t> AutoLoadEPC::runAsIntFunction(ExecutorAddr IntFnAddr,
                                                   int Arg) {
  using IntTy = int (*)(int);
  return orc::runAsIntFunction(IntFnAddr.toPtr<IntTy>(), Arg);
}

void AutoLoadEPC::callWrapperAsync(ExecutorAddr WrapperFnAddr,
                                      IncomingWFRHandler SendResult,
                                      ArrayRef<char> ArgBuffer) {
  using WrapperFnTy =
      shared::CWrapperFunctionResult (*)(const char *Data, size_t Size);
  auto *WrapperFn = WrapperFnAddr.toPtr<WrapperFnTy>();
  SendResult(WrapperFn(ArgBuffer.data(), ArgBuffer.size()));
}

Error AutoLoadEPC::disconnect() {
  D->shutdown();
  return Error::success();
}

shared::CWrapperFunctionResult
AutoLoadEPC::jitDispatchViaWrapperFunctionManager(void *Ctx,
                                                     const void *FnTag,
                                                     const char *Data,
                                                     size_t Size) {

  LLVM_DEBUG({
    dbgs() << "jit-dispatch call with tag " << FnTag << " and " << Size
           << " byte payload.\n";
  });

  std::promise<shared::WrapperFunctionResult> ResultP;
  auto ResultF = ResultP.get_future();
  static_cast<AutoLoadEPC *>(Ctx)
      ->getExecutionSession()
      .runJITDispatchHandler(
          [ResultP = std::move(ResultP)](
              shared::WrapperFunctionResult Result) mutable {
            ResultP.set_value(std::move(Result));
          },
          ExecutorAddr::fromPtr(FnTag), {Data, Size});

  return ResultF.get().release();
}

} // end namespace orc
} // end namespace llvm