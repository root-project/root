//===----- LibraryResolver.cpp - Library Resolution of Unresolved Symbols
//-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Library resolution impl for unresolved symbols
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Shared/LibraryResolver.h"
#include "llvm/ExecutionEngine/Orc/Shared/LibraryScanner.h"

#include "llvm/ADT/StringSet.h"

#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"

#include <mutex>
#include <thread>

#define DEBUG_TYPE "orc-resolver"

namespace llvm::orc {

LibraryResolver::LibraryResolver(const LibraryResolver::Setup &setup)
    : m_cache(setup.cache ? setup.cache : std::make_shared<LibraryPathCache>()),
      m_PathResolver(setup.resolver ? setup.resolver
                                    : std::make_shared<PathResolver>(m_cache)),
      m_scanH(setup.basePaths, m_cache, m_PathResolver),
      FB(setup.filterBuilder), m_libMgr(),
      m_shouldScan(setup.shouldScan ? setup.shouldScan
                                    : [](StringRef) { return true; }),
      scanBatchSize(setup.scanBatchSize) {

  if (m_scanH.getAllUnits().empty()) {
    LLVM_DEBUG(dbgs() << "Warning: No base paths provided for scanning.\n");
  }
}

std::unique_ptr<LibraryResolutionDriver>
LibraryResolutionDriver::create(const LibraryResolver::Setup &setup) {
  auto loader = std::make_unique<LibraryResolver>(setup);
  return std::unique_ptr<LibraryResolutionDriver>(
      new LibraryResolutionDriver(std::move(loader)));
}

void LibraryResolutionDriver::addScanPath(const std::string &path,
                                          PathType kind) {
  Loader->m_scanH.addBasePath(path, kind);
}

bool LibraryResolutionDriver::markLibraryLoaded(StringRef path) {
  auto lib = Loader->m_libMgr.getLibrary(path);
  if (!lib)
    return false;

  lib->setState(LibraryManager::State::Loaded);

  return true;
}

bool LibraryResolutionDriver::markLibraryUnLoaded(StringRef path) {
  auto lib = Loader->m_libMgr.getLibrary(path);
  if (!lib)
    return false;

  lib->setState(LibraryManager::State::Unloaded);

  return true;
}

void LibraryResolutionDriver::resolveSymbols(
    std::vector<std::string> symbols,
    LibraryResolver::OnSearchComplete OnCompletion,
    const SearchConfig &config) {
  Loader->searchSymbolsInLibraries(symbols, std::move(OnCompletion), config);
}

static bool shouldIgnoreSymbol(const object::SymbolRef &Sym,
                               uint32_t IgnoreFlags) {
  Expected<uint32_t> FlagsOrErr = Sym.getFlags();
  if (!FlagsOrErr) {
    consumeError(FlagsOrErr.takeError());
    return true;
  }

  uint32_t Flags = *FlagsOrErr;

  using Filter = SymbolEnumeratorOptions;
  if ((IgnoreFlags & Filter::IgnoreUndefined) &&
      (Flags & object::SymbolRef::SF_Undefined))
    return true;
  if ((IgnoreFlags & Filter::IgnoreIndirect) &&
      (Flags & object::SymbolRef::SF_Indirect))
    return true;
  if ((IgnoreFlags & Filter::IgnoreWeak) &&
      (Flags & object::SymbolRef::SF_Weak))
    return true;

  return false;
}

bool SymbolEnumerator::enumerateSymbols(StringRef Path, OnEachSymbolFn OnEach,
                                        const SymbolEnumeratorOptions &Opts) {
  if (Path.empty())
    return false;

  ObjectFileLoader ObjLoader(Path);

  auto ObjOrErr = ObjLoader.getObjectFile();
  if (!ObjOrErr) {
    // handleAllErrors(ObjOrErr.takeError(), [&](const ErrorInfoBase &EIB) {
    //   errs() << "Error loading object: " << EIB.message() << "\n";
    // });
    consumeError(ObjOrErr.takeError());
    return false;
  }

  object::ObjectFile *Obj = &ObjOrErr.get();

  auto processSymbolRange =
      [&](object::ObjectFile::symbol_iterator_range Range) -> EnumerateResult {
    for (const auto &Sym : Range) {
      if (shouldIgnoreSymbol(Sym, Opts.FilterFlags))
        continue;

      auto NameOrErr = Sym.getName();
      if (!NameOrErr) {
        consumeError(NameOrErr.takeError());
        continue;
      }

      StringRef Name = *NameOrErr;
      if (Name.empty())
        continue;

      EnumerateResult Res = OnEach(Name);
      if (Res != EnumerateResult::Continue)
        return Res;
    }
    return EnumerateResult::Continue;
  };

  EnumerateResult Res = processSymbolRange(Obj->symbols());
  if (Res != EnumerateResult::Continue)
    return Res == EnumerateResult::Stop;

  if (Obj->isELF()) {
    const auto *ElfObj = cast<object::ELFObjectFileBase>(Obj);
    Res = processSymbolRange(ElfObj->getDynamicSymbolIterators());
    if (Res != EnumerateResult::Continue)
      return Res == EnumerateResult::Stop;
  } else if (Obj->isCOFF()) {
    const auto *CoffObj = cast<object::COFFObjectFile>(Obj);
    for (auto I = CoffObj->export_directory_begin(),
              E = CoffObj->export_directory_end();
         I != E; ++I) {
      StringRef Name;
      if (I->getSymbolName(Name))
        continue;
      if (Name.empty())
        continue;

      if (OnEach(Name) != EnumerateResult::Continue)
        return false;
    }
  } else if (Obj->isMachO()) {
  }

  return true;
}

class SymbolSearchContext {
public:
  SymbolSearchContext(SymbolQuery &Q) : m_query(Q) {}

  bool hasSearched(LibraryInfo *lib) const { return m_searched.count(lib); }

  void markSearched(LibraryInfo *lib) { m_searched.insert(lib); }

  inline bool allResolved() const { return m_query.allResolved(); }

  SymbolQuery &query() { return m_query; }

private:
  SymbolQuery &m_query;
  DenseSet<LibraryInfo *> m_searched;
};

void LibraryResolver::resolveSymbolsInLibrary(
    LibraryInfo &lib, SymbolQuery &unresolvedSymbols,
    const SymbolEnumeratorOptions &Opts) {
  LLVM_DEBUG(dbgs() << "Checking unresolved symbols "
                    << " in library : " << lib.getFileName() << "\n";);
  StringSet<> discoveredSymbols;
  bool hasEnumerated = false;

  auto enumerateSymbolsIfNeeded = [&]() {
    if (hasEnumerated)
      return;

    hasEnumerated = true;

    LLVM_DEBUG(dbgs() << "Enumerating symbols in library: " << lib.getFullPath()
                      << "\n";);
    SymbolEnumerator::enumerateSymbols(
        lib.getFullPath(),
        [&](StringRef sym) {
          discoveredSymbols.insert(sym);
          return EnumerateResult::Continue;
        },
        Opts);
  };

  if (!unresolvedSymbols.hasUnresolved()) {
    LLVM_DEBUG(dbgs() << "Skipping library: " << lib.getFullPath()
                      << " — unresolved symbols exist.\n";);
    return;
  }

  enumerateSymbolsIfNeeded();

  if (discoveredSymbols.empty()) {
    LLVM_DEBUG(dbgs() << "  No symbols and remove library : "
                      << lib.getFullPath() << "\n";);
    m_libMgr.removeLibrary(lib.getFullPath());
    return;
  }

  if (!lib.hasFilter()) {
    LLVM_DEBUG(dbgs() << "Building filter for library: " << lib.getFullPath()
                      << "\n";);

    SmallVector<StringRef> SymbolVec;
    SymbolVec.reserve(discoveredSymbols.size());
    for (const auto &KV : discoveredSymbols)
      SymbolVec.push_back(KV.first());

    lib.ensureFilterBuilt(FB, SymbolVec);
    LLVM_DEBUG({
      dbgs() << "discoveredSymbols : " << discoveredSymbols.size() << "\n";
      for (const auto &KV : discoveredSymbols)
        dbgs() << "discoveredSymbols : " << KV.first() << "\n";
    });
  }

  const auto &unresolved = unresolvedSymbols.getUnresolvedSymbols();
  bool hadAnySym = false;
  LLVM_DEBUG(dbgs() << "Total unresolved symbols : " << unresolved.size()
                    << "\n";);
  for (const auto &symbol : unresolved) {
    if (lib.mayContain(symbol)) {
      LLVM_DEBUG(dbgs() << "Checking symbol '" << symbol
                        << "' in library: " << lib.getFullPath() << "\n";);
      if (discoveredSymbols.count(symbol) > 0) {
        LLVM_DEBUG(dbgs() << "  Resolved symbol: " << symbol
                          << " in library: " << lib.getFullPath() << "\n";);
        unresolvedSymbols.resolve(symbol, lib.getFullPath());
        hadAnySym = true;
      }
    }
  }

  using LibraryState = LibraryManager::State;
  if (hadAnySym && lib.getState() != LibraryState::Loaded)
    lib.setState(LibraryState::Queried);
}

void LibraryResolver::searchSymbolsInLibraries(
    std::vector<std::string> &symbolList, OnSearchComplete onComplete,
    const SearchConfig &config) {
  SymbolQuery query(symbolList);

  using LibraryState = LibraryManager::State;
  using LibraryType = PathType;
  auto tryResolveFrom = [&](LibraryState S, LibraryType K) {
    LLVM_DEBUG(dbgs() << "Trying resolve from state=" << static_cast<int>(S)
                      << " type=" << static_cast<int>(K) << "\n";);

    SymbolSearchContext Ctx(query);
    while (!Ctx.allResolved()) {

      for (auto &lib : m_libMgr.getView(S, K)) {
        if (Ctx.hasSearched(lib.get()))
          continue;

        // can use Async here?
        resolveSymbolsInLibrary(*lib, Ctx.query(), config.options);
        Ctx.markSearched(lib.get());

        if (Ctx.allResolved())
          return;
      }

      if (Ctx.allResolved())
        return;

      if (!scanLibrariesIfNeeded(K))
        break; // no more new libs to scan
    }
  };

  for (const auto &[state, type] : config.policy.plan) {
    tryResolveFrom(state, type);
    if (query.allResolved())
      break;
  }

  // done:
  LLVM_DEBUG({
    dbgs() << "Search complete.\n";
    for (const auto &r : query.getAllResults())
      dbgs() << "Resolved Symbol:" << r->Name << " -> " << r->ResolvedLibPath
             << "\n";
  });

  onComplete(query);
}

bool LibraryResolver::scanLibrariesIfNeeded(PathType PK) {
  LLVM_DEBUG(dbgs() << "LibraryResolver::scanLibrariesIfNeeded: Scanning for "
                    << (PK == PathType::User ? "User" : "System")
                    << " libraries\n";);
  if (!m_scanH.leftToScan(PK))
    return false;

  LibraryScanner Scanner(m_scanH, m_libMgr, m_shouldScan);
  Scanner.scanNext(PK, scanBatchSize);
  return true;
}

bool LibraryResolver::symbolExistsInLibrary(
    const LibraryInfo &lib, StringRef symbolName,
    std::vector<std::string> *allSymbols) {
  SymbolEnumeratorOptions opts;
  return symbolExistsInLibrary(lib, symbolName, allSymbols, opts);
}

bool LibraryResolver::symbolExistsInLibrary(
    const LibraryInfo &lib, StringRef symbolName,
    std::vector<std::string> *allSymbols, const SymbolEnumeratorOptions &opts) {
  bool found = false;

  SymbolEnumerator::enumerateSymbols(
      lib.getFullPath(),
      [&](StringRef sym) {
        if (allSymbols)
          allSymbols->emplace_back(sym.str());

        if (sym == symbolName) {
          found = true;
        }

        return EnumerateResult::Continue;
      },
      opts);

  return found;
}

} // end namespace llvm::orc
