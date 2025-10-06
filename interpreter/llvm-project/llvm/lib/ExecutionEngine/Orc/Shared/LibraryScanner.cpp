//===----- LibraryScanner.cpp - Provide Library Scaning implementation
//-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Shared/LibraryScanner.h"
#include "llvm/ExecutionEngine/Orc/Shared/LibraryResolver.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

#include "llvm/Object/COFF.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#ifdef LLVM_ON_UNIX
#include <sys/stat.h>
#include <unistd.h>
#endif // LLVM_ON_UNIX

#ifdef __APPLE__
#include <sys/stat.h>
#undef LC_LOAD_DYLIB
#undef LC_RPATH
#endif // __APPLE__

#define DEBUG_TYPE "orc-scanner"

namespace llvm::orc {

void handleError(Error Err, StringRef context = "") {
  consumeError(handleErrors(std::move(Err), [&](const ErrorInfoBase &EIB) {
    dbgs() << "LLVM Error";
    if (!context.empty())
      dbgs() << " [" << context << "]";
    dbgs() << ": " << EIB.message() << "\n";
  }));
}

bool ObjectFileLoader::isArchitectureCompatible(const object::ObjectFile &Obj) {
  Triple HostTriple(sys::getDefaultTargetTriple());
  Triple ObjTriple = Obj.makeTriple();

  LLVM_DEBUG({
    dbgs() << "Host triple: " << HostTriple.str()
           << ", Object triple: " << ObjTriple.str() << "\n";
  });

  if (ObjTriple.getArch() != Triple::UnknownArch &&
      HostTriple.getArch() != ObjTriple.getArch())
    return false;

  if (ObjTriple.getOS() != Triple::UnknownOS &&
      HostTriple.getOS() != ObjTriple.getOS())
    return false;

  if (ObjTriple.getEnvironment() != Triple::UnknownEnvironment &&
      HostTriple.getEnvironment() != Triple::UnknownEnvironment &&
      HostTriple.getEnvironment() != ObjTriple.getEnvironment())
    return false;

  return true;
}

Expected<object::OwningBinary<object::ObjectFile>>
ObjectFileLoader::loadObjectFileWithOwnership(StringRef FilePath) {
  LLVM_DEBUG(dbgs() << "ObjectFileLoader: Attempting to open file " << FilePath
                    << "\n";);
  auto BinOrErr = object::createBinary(FilePath);
  if (!BinOrErr) {
    LLVM_DEBUG(dbgs() << "ObjectFileLoader: Failed to open file " << FilePath
                      << "\n";);
    return BinOrErr.takeError();
  }

  LLVM_DEBUG(dbgs() << "ObjectFileLoader: Successfully opened file " << FilePath
                    << "\n";);

  auto OwningBin = BinOrErr->takeBinary();
  object::Binary *Bin = OwningBin.first.get();

  if (Bin->isArchive()) {
    LLVM_DEBUG(dbgs() << "ObjectFileLoader: File is an archive, not supported: "
                      << FilePath << "\n";);
    return createStringError(std::errc::invalid_argument,
                             "Archive files are not supported: %s",
                             FilePath.str().c_str());
  }

#if defined(__APPLE__)
  if (auto *UB = dyn_cast<object::MachOUniversalBinary>(Bin)) {
    LLVM_DEBUG(dbgs() << "ObjectFileLoader: Detected Mach-O universal binary: "
                      << FilePath << "\n";);
    for (auto ObjForArch : UB->objects()) {
      auto ObjOrErr = ObjForArch.getAsObjectFile();
      if (!ObjOrErr) {
        LLVM_DEBUG(
            dbgs()
                << "ObjectFileLoader: Skipping invalid architecture slice\n";);

        consumeError(ObjOrErr.takeError());
        continue;
      }

      std::unique_ptr<object::ObjectFile> Obj = std::move(ObjOrErr.get());
      if (isArchitectureCompatible(*Obj)) {
        LLVM_DEBUG(
            dbgs() << "ObjectFileLoader: Found compatible object slice\n";);

        return object::OwningBinary<object::ObjectFile>(
            std::move(Obj), std::move(OwningBin.second));

      } else {
        LLVM_DEBUG(dbgs() << "ObjectFileLoader: Incompatible architecture "
                             "slice skipped\n";);
      }
    }
    LLVM_DEBUG(dbgs() << "ObjectFileLoader: No compatible slices found in "
                         "universal binary\n";);
    return createStringError(inconvertibleErrorCode(),
                             "No compatible object found in fat binary: %s",
                             FilePath.str().c_str());
  }
#endif

  auto ObjOrErr =
      object::ObjectFile::createObjectFile(Bin->getMemoryBufferRef());
  if (!ObjOrErr) {
    LLVM_DEBUG(dbgs() << "ObjectFileLoader: Failed to create object file\n";);
    return ObjOrErr.takeError();
  }
  LLVM_DEBUG(dbgs() << "ObjectFileLoader: Detected object file\n";);

  std::unique_ptr<object::ObjectFile> Obj = std::move(*ObjOrErr);
  if (!isArchitectureCompatible(*Obj)) {
    LLVM_DEBUG(dbgs() << "ObjectFileLoader: Incompatible architecture: "
                      << FilePath << "\n";);
    return createStringError(inconvertibleErrorCode(),
                             "Incompatible object file: %s",
                             FilePath.str().c_str());
  }

  LLVM_DEBUG(dbgs() << "ObjectFileLoader: Object file is compatible\n";);

  return object::OwningBinary<object::ObjectFile>(std::move(Obj),
                                                  std::move(OwningBin.second));
}

template <class ELFT>
bool isELFSharedLibrary(const object::ELFFile<ELFT> &ELFObj) {
  if (ELFObj.getHeader().e_type != ELF::ET_DYN)
    return false;

  auto PHOrErr = ELFObj.program_headers();
  if (!PHOrErr) {
    consumeError(PHOrErr.takeError());
    return true;
  }

  for (auto Phdr : *PHOrErr) {
    if (Phdr.p_type == ELF::PT_INTERP)
      return false;
  }

  return true;
}

bool isSharedLibraryObject(object::ObjectFile &Obj) {
  if (Obj.isELF()) {
    if (auto *ELF32LE = dyn_cast<object::ELF32LEObjectFile>(&Obj))
      return isELFSharedLibrary(ELF32LE->getELFFile());
    if (auto *ELF64LE = dyn_cast<object::ELF64LEObjectFile>(&Obj))
      return isELFSharedLibrary(ELF64LE->getELFFile());
    if (auto *ELF32BE = dyn_cast<object::ELF32BEObjectFile>(&Obj))
      return isELFSharedLibrary(ELF32BE->getELFFile());
    if (auto *ELF64BE = dyn_cast<object::ELF64BEObjectFile>(&Obj))
      return isELFSharedLibrary(ELF64BE->getELFFile());
  } else if (Obj.isMachO()) {
    const object::MachOObjectFile *MachO =
        dyn_cast<object::MachOObjectFile>(&Obj);
    if (!MachO) {
      LLVM_DEBUG(dbgs() << "Failed to cast to MachOObjectFile.\n";);
      return false;
    }
    LLVM_DEBUG({
      bool Result =
          MachO->getHeader().filetype == MachO::HeaderFileType::MH_DYLIB;
      dbgs() << "Mach-O filetype: " << MachO->getHeader().filetype
             << " (MH_DYLIB == " << MachO::HeaderFileType::MH_DYLIB
             << "), shared: " << Result << "\n";
    });

    return MachO->getHeader().filetype == MachO::HeaderFileType::MH_DYLIB;
  } else if (Obj.isCOFF()) {
    const object::COFFObjectFile *coff = dyn_cast<object::COFFObjectFile>(&Obj);
    if (!coff)
      return false;
    return coff->getCharacteristics() & COFF::IMAGE_FILE_DLL;
  } else {
    LLVM_DEBUG(dbgs() << "Binary is not an ObjectFile.\n";);
  }

  return false;
}

bool DylibPathValidator::isSharedLibrary(StringRef Path) {
  LLVM_DEBUG(dbgs() << "Checking if path is a shared library: " << Path
                    << "\n";);

  auto filetype = sys::fs::get_file_type(Path, /*Follow*/ true);
  if (filetype != sys::fs::file_type::regular_file) {
    LLVM_DEBUG(dbgs() << "File type is not a regular file for path: " << Path
                      << "\n";);
    return false;
  }

  file_magic MagicCode;
  identify_magic(Path, MagicCode);

  // Skip archives.
  if (MagicCode == file_magic::archive)
    return false;

  // Universal binary handling.
#if defined(__APPLE__)
  if (MagicCode == file_magic::macho_universal_binary) {
    ObjectFileLoader ObjLoader(Path);
    auto ObjOrErr = ObjLoader.getObjectFile();
    if (!ObjOrErr) {
      consumeError(ObjOrErr.takeError());
      return false;
    }
    return isSharedLibraryObject(ObjOrErr.get());
  }
#endif

  // Object file inspection for PE/COFF, ELF, and Mach-O
  bool NeedsObjectInspection =
#if defined(_WIN32)
      (MagicCode == file_magic::pecoff_executable);
#elif defined(__APPLE__)
      (MagicCode == file_magic::macho_fixed_virtual_memory_shared_lib ||
       MagicCode == file_magic::macho_dynamically_linked_shared_lib ||
       MagicCode == file_magic::macho_dynamically_linked_shared_lib_stub);
#elif defined(LLVM_ON_UNIX)
#ifdef __CYGWIN__
      (MagicCode == file_magic::pecoff_executable);
#else
      (MagicCode == file_magic::elf_shared_object);
#endif
#else
#error "Unsupported platform."
#endif

  if (NeedsObjectInspection) {
    ObjectFileLoader ObjLoader(Path);
    auto ObjOrErr = ObjLoader.getObjectFile();
    if (!ObjOrErr) {
      consumeError(ObjOrErr.takeError());
      return false;
    }
    return isSharedLibraryObject(ObjOrErr.get());
  }

  LLVM_DEBUG(dbgs() << "Path is not identified as a shared library: " << Path
                    << "\n";);
  return false;
}

void DylibSubstitutor::configure(StringRef loaderPath) {
  SmallString<512> execPath(sys::fs::getMainExecutable(nullptr, nullptr));
  sys::path::remove_filename(execPath);

  SmallString<512> loaderDir;
  if (loaderPath.empty()) {
    loaderDir = execPath;
  } else {
    loaderDir = loaderPath.str();
    if (!sys::fs::is_directory(loaderPath))
      sys::path::remove_filename(loaderDir);
  }

#ifdef __APPLE__
  placeholders["@loader_path"] = std::string(loaderDir);
  placeholders["@executable_path"] = std::string(execPath);
#else
  placeholders["$origin"] = std::string(loaderDir);
#endif
}

std::optional<std::string>
SearchPathResolver::resolve(StringRef stem, const DylibSubstitutor &subst,
                            DylibPathValidator &validator) const {
  for (const auto &searchPath : paths) {
    std::string base = subst.substitute(searchPath);

    SmallString<512> fullPath(base);
    if (!placeholderPrefix.empty() && stem.starts_with(placeholderPrefix))
      fullPath.append(stem.drop_front(placeholderPrefix.size()));
    else
      sys::path::append(fullPath, stem);

    LLVM_DEBUG(dbgs() << "SearchPathResolver::resolve FullPath = " << fullPath
                      << "\n";);

    if (auto valid = validator.validate(fullPath.str()))
      return valid;
  }

  return std::nullopt;
}

std::optional<std::string>
DylibResolverImpl::tryWithExtensions(StringRef libStem) const {
  LLVM_DEBUG(dbgs() << "tryWithExtensions: baseName = " << libStem << "\n";);
  SmallVector<StringRef, 4> candidates;

  // Add extensions by platform
#if defined(__APPLE__)
  candidates.push_back(libStem.str() + ".dylib");
#elif defined(_WIN32)
  candidates.push_back(libStem.str() + ".dll");
#else
  candidates.push_back(libStem.str() + ".so");
#endif

  // Optionally try "lib" prefix if not already there
  StringRef filename = sys::path::filename(libStem);
  StringRef base = sys::path::parent_path(libStem);
  SmallString<256> withPrefix(base);
  if (!filename.starts_with("lib")) {
    withPrefix += "lib";
    withPrefix += filename;
    // Apply extension too
#if defined(__APPLE__)
    withPrefix += ".dylib";
#elif defined(_WIN32)
    withPrefix += ".dll";
#else
    withPrefix += ".so";
#endif
    candidates.push_back(withPrefix.str());
  }

  LLVM_DEBUG({
    dbgs() << "  Candidates to try:\n";
    for (const auto &C : candidates)
      dbgs() << "    " << C << "\n";
  });

  // Try all variants using tryAllPaths
  for (const auto &name : candidates) {

    LLVM_DEBUG(dbgs() << "  Trying candidate: " << name << "\n";);

    for (const auto &resolver : resolvers) {
      if (auto result = resolver.resolve(name, substitutor, validator))
        return result;
    }
  }

  LLVM_DEBUG(dbgs() << "  -> No candidate resolved.\n";);

  return std::nullopt;
}

std::optional<std::string>
DylibResolverImpl::resolve(StringRef libStem, bool variateLibStem) const {
  LLVM_DEBUG(dbgs() << "Resolving library stem: " << libStem << "\n";);

  // If it is an absolute path, don't try iterate over the paths.
  if (sys::path::is_absolute(libStem)) {
    LLVM_DEBUG(dbgs() << "  -> Absolute path detected.\n";);
    return validator.validate(libStem);
  }

  if (!libStem.starts_with("@rpath")) {
    if (auto norm = validator.validate(substitutor.substitute(libStem))) {
      LLVM_DEBUG(dbgs() << "  -> Resolved after substitution: " << *norm
                        << "\n";);

      return norm;
    }
  }

  for (const auto &resolver : resolvers) {
    LLVM_DEBUG(dbgs() << "  -> Resolving via search path ... \n";);
    if (auto result = resolver.resolve(libStem, substitutor, validator)) {
      LLVM_DEBUG(dbgs() << "  -> Resolved via search path: " << *result
                        << "\n";);

      return result;
    }
  }

  // Expand libStem with paths, extensions, etc.
  // std::string foundName;
  if (variateLibStem) {
    LLVM_DEBUG(dbgs() << "  -> Trying with extensions...\n";);

    if (auto norm = tryWithExtensions(libStem)) {
      LLVM_DEBUG(dbgs() << "  -> Resolved via tryWithExtensions: " << *norm
                        << "\n";);

      return norm;
    }
  }

  LLVM_DEBUG(dbgs() << "  -> Could not resolve: " << libStem << "\n";);

  return std::nullopt;
}

#ifndef _WIN32
mode_t PathResolver::lstatCached(StringRef path) {
  // If already cached - retun cached result
  if (auto cache = m_cache->read_lstat(path))
    return *cache;

  // Not cached: perform lstat and store
  struct stat buf{};
  mode_t st_mode = (lstat(path.str().c_str(), &buf) == -1) ? 0 : buf.st_mode;

  m_cache->insert_lstat(path, st_mode);

  return st_mode;
}

std::optional<std::string> PathResolver::readlinkCached(StringRef path) {
  // If already cached - retun cached result
  if (auto cache = m_cache->read_link(path))
    return cache;

  // If result not in cache - call system function and cache result
  char buf[PATH_MAX];
  ssize_t len;
  if ((len = readlink(path.str().c_str(), buf, sizeof(buf))) != -1) {
    buf[len] = '\0';
    std::string s(buf);
    m_cache->insert_link(path, s);
    return s;
  }
  return std::nullopt;
}

void createComponent(StringRef Path, StringRef base_path, bool baseIsResolved,
                     SmallVector<StringRef, 16> &component) {
  StringRef Separator = sys::path::get_separator();
  if (!baseIsResolved) {
    if (Path[0] == '~' &&
        (Path.size() == 1 || sys::path::is_separator(Path[1]))) {
      static SmallString<128> home;
      if (home.str().empty())
        sys::path::home_directory(home);
      StringRef(home).split(component, Separator, /*MaxSplit*/ -1,
                            /*KeepEmpty*/ false);
    } else if (base_path.empty()) {
      static SmallString<256> current_path;
      if (current_path.str().empty())
        sys::fs::current_path(current_path);
      StringRef(current_path)
          .split(component, Separator, /*MaxSplit*/ -1, /*KeepEmpty*/ false);
    } else {
      base_path.split(component, Separator, /*MaxSplit*/ -1,
                      /*KeepEmpty*/ false);
    }
  }

  Path.split(component, Separator, /*MaxSplit*/ -1, /*KeepEmpty*/ false);
}

void normalizePathSegments(SmallVector<StringRef, 16> &pathParts) {
  SmallVector<StringRef, 16> normalizedPath;
  for (auto &part : pathParts) {
    if (part == ".") {
      continue;
    } else if (part == "..") {
      if (!normalizedPath.empty() && normalizedPath.back() != "..") {
        normalizedPath.pop_back();
      } else {
        normalizedPath.push_back("..");
      }
    } else {
      normalizedPath.push_back(part);
    }
  }
  pathParts.swap(normalizedPath);
}
#endif

std::optional<std::string> PathResolver::realpathCached(StringRef path,
                                                        std::error_code &ec,
                                                        StringRef base,
                                                        bool baseIsResolved,
                                                        long symloopLevel) {
  ec.clear();

  if (path.empty()) {
    ec = std::make_error_code(std::errc::no_such_file_or_directory);
    LLVM_DEBUG(dbgs() << "PathResolver::realpathCached: Empty path\n";);

    return std::nullopt;
  }

  if (symloopLevel <= 0) {
    ec = std::make_error_code(std::errc::too_many_symbolic_link_levels);
    LLVM_DEBUG(
        dbgs() << "PathResolver::realpathCached: Too many symlink levels: "
               << path << "\n";);

    return std::nullopt;
  }

  // If already cached - retun cached result
  bool isRelative = sys::path::is_relative(path);
  if (!isRelative) {
    if (auto cached = m_cache->read_realpath(path)) {
      ec = cached->errnoCode;
      if (ec) {
        LLVM_DEBUG(dbgs() << "PathResolver::realpathCached: Cached (error) for "
                          << path << "\n";);
      } else {
        LLVM_DEBUG(
            dbgs() << "PathResolver::realpathCached: Cached (success) for "
                   << path << " => " << cached->canonicalPath << "\n";);
      }
      return cached->canonicalPath.empty()
                 ? std::nullopt
                 : std::make_optional(cached->canonicalPath);
    }
  }

  LLVM_DEBUG(dbgs() << "PathResolver::realpathCached: Resolving path: " << path
                    << "\n";);

  // If result not in cache - call system function and cache result

  StringRef Separator(sys::path::get_separator());
  SmallString<256> resolved(Separator);
#ifndef _WIN32
  SmallVector<StringRef, 16> Components;

  if (isRelative) {
    if (baseIsResolved) {
      resolved.assign(base);
      LLVM_DEBUG(dbgs() << "  Using resolved base: " << base << "\n";);
    }
    createComponent(path, base, baseIsResolved, Components);
  } else {
    path.split(Components, Separator, /*MaxSplit*/ -1, /*KeepEmpty*/ false);
  }

  normalizePathSegments(Components);
  LLVM_DEBUG({
    for (auto &C : Components)
      dbgs() << " " << C << " ";

    dbgs() << "\n";
  });

  // Handle path list items
  for (const auto &component : Components) {
    if (component == ".")
      continue;
    if (component == "..") {
      // collapse "a/b/../c" to "a/c"
      size_t s = resolved.rfind(Separator);
      if (s != llvm::StringRef::npos)
        resolved.resize(s);
      if (resolved.empty())
        resolved = Separator;
      continue;
    }

    size_t oldSize = resolved.size();
    sys::path::append(resolved, component);
    const char *resolvedPath = resolved.c_str();
    LLVM_DEBUG(dbgs() << "  Processing component: " << component << " => "
                      << resolvedPath << "\n";);
    mode_t st_mode = lstatCached(resolvedPath);

    if (S_ISLNK(st_mode)) {
      LLVM_DEBUG(dbgs() << "    Found symlink: " << resolvedPath << "\n";);

      auto symlinkOpt = readlinkCached(resolvedPath);
      if (!symlinkOpt) {
        ec = std::make_error_code(std::errc::no_such_file_or_directory);
        m_cache->insert_realpath(path, LibraryPathCache::PathInfo{"", ec});
        LLVM_DEBUG(dbgs() << "    Failed to read symlink: " << resolvedPath
                          << "\n";);

        return std::nullopt;
      }

      StringRef symlink = *symlinkOpt;
      LLVM_DEBUG(dbgs() << "    Symlink points to: " << symlink << "\n";);

      std::string resolvedBase = "";
      if (sys::path::is_relative(symlink)) {
        resolved.resize(oldSize);
        resolvedBase = resolved.str().str();
      }

      auto realSymlink =
          realpathCached(symlink, ec, resolvedBase,
                         /*baseIsResolved=*/true, symloopLevel - 1);
      if (!realSymlink) {
        m_cache->insert_realpath(path, LibraryPathCache::PathInfo{"", ec});
        LLVM_DEBUG(dbgs() << "    Failed to resolve symlink target: " << symlink
                          << "\n";);

        return std::nullopt;
      }

      resolved.assign(*realSymlink);
      LLVM_DEBUG(dbgs() << "    Symlink resolved to: " << resolved << "\n";);

    } else if (st_mode == 0) {
      ec = std::make_error_code(std::errc::no_such_file_or_directory);
      m_cache->insert_realpath(path, LibraryPathCache::PathInfo{"", ec});
      LLVM_DEBUG(dbgs() << "    Component does not exist: " << resolvedPath
                        << "\n";);

      return std::nullopt;
    }
  }
#else
  sys::fs::real_path(path, resolved); // Windows fallback
#endif

  std::string canonical = resolved.str().str();
  {
    m_cache->insert_realpath(path, LibraryPathCache::PathInfo{
                                       canonical,
                                       std::error_code() // success
                                   });
  }
  LLVM_DEBUG(dbgs() << "PathResolver::realpathCached: Final resolved: " << path
                    << " => " << canonical << "\n";);
  return canonical;
}

void LibraryScanHelper::addBasePath(const std::string &path, PathType kind) {
  std::error_code ec;
  std::string canon = resolveCanonical(path, ec);
  if (ec) {
    LLVM_DEBUG(
        dbgs()
            << "LibraryScanHelper::addBasePath: Failed to canonicalize path: "
            << path << "\n";);
    return;
  }
  std::unique_lock<std::shared_mutex> lock(m_mutex);
  if (m_units.count(canon)) {
    LLVM_DEBUG(dbgs() << "LibraryScanHelper::addBasePath: Already added: "
                      << canon << "\n";);
    return;
  }
  kind = kind == PathType::Unknown ? classifyKind(canon) : kind;
  auto unit = std::make_shared<LibraryUnit>(canon, kind);
  m_units[canon] = unit;

  if (kind == PathType::User) {
    LLVM_DEBUG(dbgs() << "LibraryScanHelper::addBasePath: Added User path: "
                      << canon << "\n";);
    m_unscannedUsr.push_back(StringRef(unit->basePath));
  } else {
    LLVM_DEBUG(dbgs() << "LibraryScanHelper::addBasePath: Added System path: "
                      << canon << "\n";);
    m_unscannedSys.push_back(StringRef(unit->basePath));
  }
}

std::vector<std::shared_ptr<LibraryUnit>>
LibraryScanHelper::getNextBatch(PathType kind, size_t batchSize) {
  std::vector<std::shared_ptr<LibraryUnit>> result;
  auto &queue = (kind == PathType::User) ? m_unscannedUsr : m_unscannedSys;

  std::unique_lock<std::shared_mutex> lock(m_mutex);

  while (!queue.empty() && (batchSize == 0 || result.size() < batchSize)) {
    StringRef base = queue.front();
    auto it = m_units.find(base);
    if (it != m_units.end()) {
      auto &unit = it->second;
      ScanState expected = ScanState::NotScanned;
      if (unit->state.compare_exchange_strong(expected, ScanState::Scanning)) {
        result.push_back(unit);
      }
    }
    queue.pop_front();
  }

  return result;
}

bool LibraryScanHelper::isTrackedBasePath(StringRef path) const {
  std::error_code ec;
  std::string canon = resolveCanonical(path, ec);
  if (ec) {
    return false;
  }
  std::shared_lock<std::shared_mutex> lock(m_mutex);
  return m_units.count(canon) > 0;
}

bool LibraryScanHelper::leftToScan(PathType K) const {
  std::shared_lock<std::shared_mutex> lock(m_mutex);
  for (const auto &unit : m_units)
    if (unit.second->kind == K && unit.second->state == ScanState::NotScanned)
      return true;
  return false;
}

std::vector<std::shared_ptr<LibraryUnit>>
LibraryScanHelper::getAllUnits() const {
  std::shared_lock<std::shared_mutex> lock(m_mutex);
  std::vector<std::shared_ptr<LibraryUnit>> result;
  result.reserve(m_units.size());
  for (const auto &[_, unit] : m_units) {
    result.push_back(unit);
  }
  return result;
}

std::string LibraryScanHelper::resolveCanonical(StringRef path,
                                                std::error_code &ec) const {
  auto canon = m_resolver->resolve(path, ec);
  return ec ? path.str() : *canon;
}

PathType LibraryScanHelper::classifyKind(StringRef path) const {
  // Detect home directory
  const char *home = getenv("HOME");
  if (home && path.find(home) == 0)
    return PathType::User;

  static const std::array<std::string, 5> userPrefixes = {
      "/usr/local",    // often used by users for manual installs
      "/opt/homebrew", // common on macOS
      "/opt/local",    // MacPorts
      "/home",         // Linux home dirs
      "/Users",        // macOS user dirs
  };

  for (const auto &prefix : userPrefixes) {
    if (path.find(prefix) == 0)
      return PathType::User;
  }

  return PathType::System;
}

Expected<LibraryDepsInfo> parseMachODeps(const object::MachOObjectFile &Obj) {
  LibraryDepsInfo libdeps;
  LLVM_DEBUG(dbgs() << "Parsing Mach-O dependencies...\n";);
  for (const auto &Command : Obj.load_commands()) {
    switch (Command.C.cmd) {
    case MachO::LC_LOAD_DYLIB: {
      MachO::dylib_command dylibCmd = Obj.getDylibIDLoadCommand(Command);
      const char *name = Command.Ptr + dylibCmd.dylib.name;
      libdeps.addDep(name);
      LLVM_DEBUG(dbgs() << "  Found LC_LOAD_DYLIB: " << name << "\n";);
    } break;
    case MachO::LC_LOAD_WEAK_DYLIB:
    case MachO::LC_REEXPORT_DYLIB:
    case MachO::LC_LOAD_UPWARD_DYLIB:
    case MachO::LC_LAZY_LOAD_DYLIB:
      break;
    case MachO::LC_RPATH: {
      // Extract RPATH
      MachO::rpath_command rpathCmd = Obj.getRpathCommand(Command);
      const char *rpath = Command.Ptr + rpathCmd.path;
      LLVM_DEBUG(dbgs() << "  Found LC_RPATH: " << rpath << "\n";);

      SmallVector<StringRef, 4> RawPaths;
      SplitString(StringRef(rpath), RawPaths,
                  sys::EnvPathSeparator == ':' ? ":" : ";");

      for (const auto &raw : RawPaths) {
        libdeps.addRPath(raw.str()); // Convert to std::string
        LLVM_DEBUG(dbgs() << "    Parsed RPATH entry: " << raw << "\n";);
      }
      break;
    }
    }
  }
  return libdeps;
}

template <class ELFT>
static Expected<StringRef> getDynamicStrTab(const object::ELFFile<ELFT> &Elf) {
  auto DynamicEntriesOrError = Elf.dynamicEntries();
  if (!DynamicEntriesOrError)
    return DynamicEntriesOrError.takeError();

  for (const typename ELFT::Dyn &Dyn : *DynamicEntriesOrError) {
    if (Dyn.d_tag == ELF::DT_STRTAB) {
      auto MappedAddrOrError = Elf.toMappedAddr(Dyn.getPtr());
      if (!MappedAddrOrError)
        return MappedAddrOrError.takeError();
      return StringRef(reinterpret_cast<const char *>(*MappedAddrOrError));
    }
  }

  // If the dynamic segment is not present, we fall back on the sections.
  auto SectionsOrError = Elf.sections();
  if (!SectionsOrError)
    return SectionsOrError.takeError();

  for (const typename ELFT::Shdr &Sec : *SectionsOrError) {
    if (Sec.sh_type == ELF::SHT_DYNSYM)
      return Elf.getStringTableForSymtab(Sec);
  }

  return make_error<StringError>("dynamic string table not found",
                                 inconvertibleErrorCode());
}

template <typename ELFT>
Expected<LibraryDepsInfo> parseELF(const object::ELFFile<ELFT> &Elf) {
  LibraryDepsInfo Deps;
  Expected<StringRef> StrTabOrErr = getDynamicStrTab(Elf);
  if (!StrTabOrErr)
    return StrTabOrErr.takeError();

  const char *Data = StrTabOrErr->data();

  auto DynamicEntriesOrError = Elf.dynamicEntries();
  if (!DynamicEntriesOrError) {
    return DynamicEntriesOrError.takeError();
  }

  for (const typename ELFT::Dyn &Dyn : *DynamicEntriesOrError) {
    switch (Dyn.d_tag) {
    case ELF::DT_NEEDED:
      Deps.addDep(Data + Dyn.d_un.d_val);
      break;
    case ELF::DT_RPATH: {
      SmallVector<StringRef, 4> RawPaths;
      SplitString(Data + Dyn.d_un.d_val, RawPaths,
                  sys::EnvPathSeparator == ':' ? ":" : ";");
      for (const auto &raw : RawPaths)
        Deps.addRPath(raw.str());
      break;
    }
    case ELF::DT_RUNPATH: {
      SmallVector<StringRef, 4> RawPaths;
      SplitString(Data + Dyn.d_un.d_val, RawPaths,
                  sys::EnvPathSeparator == ':' ? ":" : ";");
      for (const auto &raw : RawPaths)
        Deps.addRunPath(raw.str());
      break;
    }
    case ELF::DT_FLAGS_1:
      // Check if this is not a pie executable.
      if (Dyn.d_un.d_val & ELF::DF_1_PIE)
        Deps.isPIE = true;
      break;
      // (Dyn.d_tag == ELF::DT_NULL) continue;
      // (Dyn.d_tag == ELF::DT_AUXILIARY || Dyn.d_tag == ELF::DT_FILTER)
    default:
      break;
    }
  }
  return Deps;
}

Expected<LibraryDepsInfo> parseELFDeps(const object::ELFObjectFileBase &obj) {
  using namespace object;
  LLVM_DEBUG(dbgs() << "parseELFDeps: Detected ELF object\n";);
  if (const auto *ELF = dyn_cast<ELF32LEObjectFile>(&obj))
    return parseELF(ELF->getELFFile());
  else if (const auto *ELF = dyn_cast<ELF32BEObjectFile>(&obj))
    return parseELF(ELF->getELFFile());
  else if (const auto *ELF = dyn_cast<ELF64LEObjectFile>(&obj))
    return parseELF(ELF->getELFFile());
  else if (const auto *ELF = dyn_cast<ELF64BEObjectFile>(&obj))
    return parseELF(ELF->getELFFile());

  LLVM_DEBUG(dbgs() << "parseELFDeps: Unknown ELF format\n";);
  return createStringError(std::errc::not_supported, "Unknown ELF format");
}

Expected<LibraryDepsInfo> LibraryScanner::extractDeps(StringRef filePath) {
  LLVM_DEBUG(dbgs() << "extractDeps: Attempting to open file " << filePath
                    << "\n";);

  ObjectFileLoader ObjLoader(filePath);
  auto ObjOrErr = ObjLoader.getObjectFile();
  if (!ObjOrErr) {
    LLVM_DEBUG(dbgs() << "extractDeps: Failed to open " << filePath << "\n";);
    return ObjOrErr.takeError();
  }

  object::ObjectFile *Obj = &ObjOrErr.get();

  if (auto *elfObj = dyn_cast<object::ELFObjectFileBase>(Obj)) {
    LLVM_DEBUG(dbgs() << "extractDeps: File " << filePath
                      << " is an ELF object\n";);

    return parseELFDeps(*elfObj);
  }

  if (auto *macho = dyn_cast<object::MachOObjectFile>(Obj)) {
    LLVM_DEBUG(dbgs() << "extractDeps: File " << filePath
                      << " is a Mach-O object\n";);
    return parseMachODeps(*macho);
  }

  if (Obj->isCOFF()) {
    // TODO: COFF support
    return LibraryDepsInfo();
  }

  LLVM_DEBUG(dbgs() << "extractDeps: Unsupported binary format for file "
                    << filePath << "\n";);
  return createStringError(inconvertibleErrorCode(),
                           "Unsupported binary format: %s",
                           filePath.str().c_str());
}

std::optional<std::string> LibraryScanner::shouldScan(StringRef filePath) {
  std::error_code EC;

  LLVM_DEBUG(dbgs() << "[shouldScan] Checking: " << filePath << "\n";);

  // [1] Check file existence early
  if (!sys::fs::exists(filePath)) {
    LLVM_DEBUG(dbgs() << "  -> Skipped: file does not exist.\n";);

    return std::nullopt;
  }

  // [2] Resolve to canonical path
  auto CanonicalPathOpt = m_helper.resolve(filePath, EC);
  if (EC || !CanonicalPathOpt) {
    LLVM_DEBUG(dbgs() << "  -> Skipped: failed to resolve path (EC="
                      << EC.message() << ").\n";);

    return std::nullopt;
  }

  const std::string &CanonicalPath = *CanonicalPathOpt;
  LLVM_DEBUG(dbgs() << "  -> Canonical path: " << CanonicalPath << "\n");

  // [3] Check if it's a directory — skip directories
  if (sys::fs::is_directory(CanonicalPath)) {
    LLVM_DEBUG(dbgs() << "  -> Skipped: path is a directory.\n";);

    return std::nullopt;
  }

  // [4] Skip if it's not a shared library.
  if (!DylibPathValidator::isSharedLibrary(CanonicalPath)) {
    LLVM_DEBUG(dbgs() << "  -> Skipped: not a shared library.\n";);

    return std::nullopt;
  }

  // [5] Skip if we've already seen this path (via cache)
  if (m_helper.hasSeenOrMark(CanonicalPath)) {
    LLVM_DEBUG(dbgs() << "  -> Skipped: already seen.\n";);

    return std::nullopt;
  }

  // [6] Already tracked in LibraryManager?
  if (m_libMgr.hasLibrary(CanonicalPath)) {
    LLVM_DEBUG(dbgs() << "  -> Skipped: already tracked by LibraryManager.\n";);

    return std::nullopt;
  }

  // [7] Run user-defined hook (default: always true)
  if (!shouldScanCall(CanonicalPath)) {
    LLVM_DEBUG(dbgs() << "  -> Skipped: user-defined hook rejected.\n";);

    return std::nullopt;
  }

  LLVM_DEBUG(dbgs() << "  -> Accepted: ready to scan " << CanonicalPath
                    << "\n";);
  return CanonicalPath;
}

void LibraryScanner::handleLibrary(StringRef filePath, PathType K, int level) {
  LLVM_DEBUG(dbgs() << "LibraryScanner::handleLibrary: Scanning: " << filePath
                    << ", level=" << level << "\n";);
  auto CanonPathOpt = shouldScan(filePath);
  if (!CanonPathOpt) {
    LLVM_DEBUG(dbgs() << "  Skipped (shouldScan returned false): " << filePath
                      << "\n";);

    return;
  }
  const std::string CanonicalPath = *CanonPathOpt;

  auto DepsOrErr = extractDeps(CanonicalPath);
  if (!DepsOrErr) {
    LLVM_DEBUG(dbgs() << "  Failed to extract deps for: " << CanonicalPath
                      << "\n";);
    handleError(DepsOrErr.takeError());
    return;
  }

  LibraryDepsInfo &Deps = *DepsOrErr;

  LLVM_DEBUG({
    dbgs() << "    Found deps : \n";
    for (const auto &dep : Deps.deps)
      dbgs() << "        : " << dep << "\n";
    dbgs() << "    Found @rpath : " << Deps.rpath.size() << "\n";
    for (const auto &r : Deps.rpath)
      dbgs() << "     : " << r << "\n";
    dbgs() << "    Found @runpath : \n";
    for (const auto &r : Deps.runPath)
      dbgs() << "     : " << r << "\n";
  });

  if (Deps.isPIE && level == 0) {
    LLVM_DEBUG(dbgs() << "  Skipped PIE executable at top level: "
                      << CanonicalPath << "\n";);

    return;
  }

  bool added = m_libMgr.addLibrary(CanonicalPath, K);
  if (!added) {
    LLVM_DEBUG(dbgs() << "  Already added: " << CanonicalPath << "\n";);
    return;
  }

  // Heuristic 1: No RPATH/RUNPATH, skip deps
  if (Deps.rpath.empty() && Deps.runPath.empty()) {
    LLVM_DEBUG(
        dbgs() << "LibraryScanner::handleLibrary: Skipping deps (Heuristic1): "
               << CanonicalPath << "\n";);
    return;
  }

  // Heuristic 2: All RPATH and RUNPATH already tracked
  auto allTracked = [&](const auto &Paths) {
    LLVM_DEBUG(dbgs() << "   Checking : " << Paths.size() << "\n";);
    return std::all_of(Paths.begin(), Paths.end(), [&](StringRef P) {
      LLVM_DEBUG(dbgs() << "      Checking isTrackedBasePath : " << P << "\n";);
      return m_helper.isTrackedBasePath(
          DylibResolver::resolvelinkerFlag(P, CanonicalPath));
    });
  };

  if (allTracked(Deps.rpath) && allTracked(Deps.runPath)) {
    LLVM_DEBUG(
        dbgs() << "LibraryScanner::handleLibrary: Skipping deps (Heuristic2): "
               << CanonicalPath << "\n";);
    return;
  }

  DylibPathValidator validator(m_helper.getPathResolver());
  DylibResolver m_libResolver(validator);
  m_libResolver.configure(
      CanonicalPath, {{Deps.rpath, SearchPathType::RPath},
                      {m_helper.getSearchPaths(), SearchPathType::UsrOrSys},
                      {Deps.runPath, SearchPathType::RunPath}});
  for (StringRef dep : Deps.deps) {
    LLVM_DEBUG(dbgs() << "  Resolving dep: " << dep << "\n";);
    auto dep_fullopt = m_libResolver.resolve(dep);
    if (!dep_fullopt) {
      LLVM_DEBUG(dbgs() << "    Failed to resolve dep: " << dep << "\n";);

      continue;
    }
    LLVM_DEBUG(dbgs() << "    Resolved dep to: " << *dep_fullopt << "\n";);

    handleLibrary(*dep_fullopt, K, level + 1);
  }
}

void LibraryScanner::scanBaseDir(std::shared_ptr<LibraryUnit> unit) {
  if (!sys::fs::is_directory(unit->basePath) || unit->basePath.empty()) {
    LLVM_DEBUG(
        dbgs() << "LibraryScanner::scanBaseDir: Invalid or empty basePath: "
               << unit->basePath << "\n";);
    return;
  }

  LLVM_DEBUG(dbgs() << "LibraryScanner::scanBaseDir: Scanning directory: "
                    << unit->basePath << "\n";);
  std::error_code ec;

  unit->state.store(ScanState::Scanning);

  for (sys::fs::directory_iterator it(unit->basePath, ec), end;
       it != end && !ec; it.increment(ec)) {
    auto entry = *it;
    if (!entry.status())
      continue;

    auto status = *entry.status();
    if (sys::fs::is_regular_file(status) || sys::fs::is_symlink_file(status)) {
      LLVM_DEBUG(dbgs() << "  Found file: " << entry.path() << "\n";);
      // async support ?
      handleLibrary(entry.path(), unit->kind);
    }
  }

  unit->state.store(ScanState::Scanned);
}

void LibraryScanner::scanNext(PathType K, size_t batchSize) {
  LLVM_DEBUG(dbgs() << "LibraryScanner::scanNext: Scanning next batch of size "
                    << batchSize << " for kind "
                    << (K == PathType::User ? "User" : "System") << "\n";);

  auto Units = m_helper.getNextBatch(K, batchSize);
  for (auto &unit : Units) {
    LLVM_DEBUG(dbgs() << "  Scanning unit with basePath: " << unit->basePath
                      << "\n";);

    scanBaseDir(unit);
  }
}

} // end namespace llvm::orc
