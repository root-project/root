//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "Paths.h"
#include "Compatibility.h"

#include "clang/Basic/FileManager.h"
#include "clang/Lex/HeaderSearchOptions.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#if defined(LLVM_ON_UNIX)
#include <dlfcn.h>
#endif

namespace Cpp {
namespace utils {

namespace platform {
#if defined(LLVM_ON_UNIX)
  const char* const kEnvDelim = ":";
#elif defined(_WIN32)
  const char* const kEnvDelim = ";";
#else
  #error "Unknown platform (environmental delimiter)"
#endif

#if defined(LLVM_ON_UNIX)
  bool Popen(const std::string& Cmd, llvm::SmallVectorImpl<char>& Buf,
             bool RdE) {
    if (FILE* PF = ::popen(RdE ? (Cmd + " 2>&1").c_str() : Cmd.c_str(), "r")) {
      Buf.resize(0);
      const size_t Chunk = Buf.capacity_in_bytes();
      while (true) {
        const size_t Len = Buf.size();
        Buf.resize(Len + Chunk);
        const size_t R = ::fread(&Buf[Len], sizeof(char), Chunk, PF);
        if (R < Chunk) {
          Buf.resize(Len + R);
          break;
        }
      }
      ::pclose(PF);
      return !Buf.empty();
    }
    return false;
  }
#endif

  bool GetSystemLibraryPaths(llvm::SmallVectorImpl<std::string>& Paths) {
#if defined(__APPLE__) || defined(__CYGWIN__)
    Paths.push_back("/usr/local/lib/");
    Paths.push_back("/usr/X11R6/lib/");
    Paths.push_back("/usr/lib/");
    Paths.push_back("/lib/");

#ifndef __APPLE__
    Paths.push_back("/lib/x86_64-linux-gnu/");
    Paths.push_back("/usr/local/lib64/");
    Paths.push_back("/usr/lib64/");
    Paths.push_back("/lib64/");
#endif
#elif defined(LLVM_ON_UNIX)
    llvm::SmallString<1024> Buf;
    platform::Popen("LD_DEBUG=libs LD_PRELOAD=DOESNOTEXIST ls", Buf, true);
    const llvm::StringRef Result = Buf.str();

    const std::size_t NPos = std::string::npos;
    const std::size_t LD = Result.find("(LD_LIBRARY_PATH)");
    std::size_t From = Result.find("search path=", LD == NPos ? 0 : LD);
    if (From != NPos) {
      std::size_t To = Result.find("(system search path)", From);
      if (To != NPos) {
        From += 12;
        while (To > From && isspace(Result[To - 1]))
          --To;
        std::string SysPath = Result.substr(From, To - From).str();
        SysPath.erase(std::remove_if(SysPath.begin(), SysPath.end(), ::isspace),
                      SysPath.end());

        llvm::SmallVector<llvm::StringRef, 10> CurPaths;
        SplitPaths(SysPath, CurPaths);
        for (const auto& Path : CurPaths)
          Paths.push_back(Path.str());
      }
    }
#endif
    return true;
  }

  std::string NormalizePath(const std::string& Path) {

    llvm::SmallString<256> Buffer;
    std::error_code EC = llvm::sys::fs::real_path(Path, Buffer, true);
    if (EC)
      return std::string();
    return std::string(Buffer.str());
  }

#if defined(LLVM_ON_UNIX)
  static void DLErr(std::string* Err) {
    if (!Err)
      return;
    if (const char* DyLibError = ::dlerror())
      *Err = DyLibError;
  }

  void* DLOpen(const std::string& Path, std::string* Err /* = nullptr */) {
    void* Lib = ::dlopen(Path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    DLErr(Err);
    return Lib;
  }

  void DLClose(void* Lib, std::string* Err /* = nullptr*/) {
    ::dlclose(Lib);
    DLErr(Err);
  }
#elif defined(_WIN32)

  void* DLOpen(const std::string& Path, std::string* Err /* = nullptr */) {
    auto lib = llvm::sys::DynamicLibrary::getLibrary(Path.c_str(), Err);
    return lib.getOSSpecificHandle();
  }

  void DLClose(void* Lib, std::string* Err /* = nullptr*/) {
    auto dl = llvm::sys::DynamicLibrary(Lib);
    llvm::sys::DynamicLibrary::closeLibrary(dl);
    if (Err) {
      *Err = std::string();
    }
  }
#endif

} // namespace platform

using namespace llvm;
using namespace clang;

// Adapted from clang/lib/Frontend/CompilerInvocation.cpp

void CopyIncludePaths(const clang::HeaderSearchOptions& Opts,
                      llvm::SmallVectorImpl<std::string>& incpaths,
                      bool withSystem, bool withFlags) {
  if (withFlags && Opts.Sysroot != "/") {
    incpaths.push_back("-isysroot");
    incpaths.push_back(Opts.Sysroot);
  }

  /// User specified include entries.
  for (unsigned i = 0, e = Opts.UserEntries.size(); i != e; ++i) {
    const HeaderSearchOptions::Entry &E = Opts.UserEntries[i];
    if (E.IsFramework && E.Group != frontend::Angled)
      llvm::report_fatal_error("Invalid option set!");
    switch (E.Group) {
    case frontend::After:
      if (withFlags) incpaths.push_back("-idirafter");
      break;

    case frontend::Quoted:
      if (withFlags) incpaths.push_back("-iquote");
      break;

    case frontend::System:
      if (!withSystem) continue;
      if (withFlags) incpaths.push_back("-isystem");
      break;

    case frontend::IndexHeaderMap:
      if (!withSystem) continue;
      if (withFlags) incpaths.push_back("-index-header-map");
      if (withFlags) incpaths.push_back(E.IsFramework? "-F" : "-I");
      break;

    case frontend::CSystem:
      if (!withSystem) continue;
      if (withFlags) incpaths.push_back("-c-isystem");
      break;

    case frontend::ExternCSystem:
      if (!withSystem) continue;
      if (withFlags) incpaths.push_back("-extern-c-isystem");
      break;

    case frontend::CXXSystem:
      if (!withSystem) continue;
      if (withFlags) incpaths.push_back("-cxx-isystem");
      break;

    case frontend::ObjCSystem:
      if (!withSystem) continue;
      if (withFlags) incpaths.push_back("-objc-isystem");
      break;

    case frontend::ObjCXXSystem:
      if (!withSystem) continue;
      if (withFlags) incpaths.push_back("-objcxx-isystem");
      break;

    case frontend::Angled:
      if (withFlags) incpaths.push_back(E.IsFramework ? "-F" : "-I");
      break;
    }
    incpaths.push_back(E.Path);
  }

  if (withSystem && !Opts.ResourceDir.empty()) {
    if (withFlags) incpaths.push_back("-resource-dir");
    incpaths.push_back(Opts.ResourceDir);
  }
  if (withSystem && withFlags && !Opts.ModuleCachePath.empty()) {
    incpaths.push_back("-fmodule-cache-path");
    incpaths.push_back(Opts.ModuleCachePath);
  }
  if (withSystem && withFlags && !Opts.UseStandardSystemIncludes)
    incpaths.push_back("-nostdinc");
  if (withSystem && withFlags && !Opts.UseStandardCXXIncludes)
    incpaths.push_back("-nostdinc++");
  if (withSystem && withFlags && Opts.UseLibcxx)
    incpaths.push_back("-stdlib=libc++");
  if (withSystem && withFlags && Opts.Verbose)
    incpaths.push_back("-v");
}

void LogNonExistantDirectory(llvm::StringRef Path) {
#define DEBUG_TYPE "LogNonExistantDirectory"
  LLVM_DEBUG(dbgs() << "  ignoring nonexistent directory \"" << Path << "\"\n");
#undef  DEBUG_TYPE
}

bool SplitPaths(llvm::StringRef PathStr,
                llvm::SmallVectorImpl<llvm::StringRef>& Paths,
                SplitMode Mode, llvm::StringRef Delim, bool Verbose) {
#define DEBUG_TYPE "SplitPths"

  assert(Delim.size() && "Splitting without a delimiter");

#if defined(_WIN32)
  // Support using a ':' delimiter on Windows.
  const bool WindowsColon = (Delim == ":");
#endif

  bool AllExisted = true;
  for (std::pair<llvm::StringRef, llvm::StringRef> Split = PathStr.split(Delim);
       !Split.second.empty(); Split = PathStr.split(Delim)) {

    if (!Split.first.empty()) {
      bool Exists = llvm::sys::fs::is_directory(Split.first);

#if defined(_WIN32)
    // Because drive letters will have a colon we have to make sure the split
    // occurs at a colon not followed by a path separator.
    if (!Exists && WindowsColon && Split.first.size()==1) {
      // Both clang and cl.exe support '\' and '/' path separators.
      if (Split.second.front() == '\\' || Split.second.front() == '/') {
          const std::pair<llvm::StringRef, llvm::StringRef> Tmp =
              Split.second.split(Delim);
          // Split.first = 'C', but we want 'C:', so Tmp.first.size()+2
          Split.first =
              llvm::StringRef(Split.first.data(), Tmp.first.size() + 2);
          Split.second = Tmp.second;
          Exists = llvm::sys::fs::is_directory(Split.first);
      }
    }
#endif

      AllExisted = AllExisted && Exists;

      if (!Exists) {
        if (Mode == kFailNonExistant) {
          if (Verbose) {
            // Exiting early, but still log all non-existant paths that we have
            LogNonExistantDirectory(Split.first);
            while (!Split.second.empty()) {
              Split = PathStr.split(Delim);
              if (llvm::sys::fs::is_directory(Split.first)) {
                LLVM_DEBUG(dbgs() << "  ignoring directory that exists \""
                                  << Split.first << "\"\n");
              } else
                LogNonExistantDirectory(Split.first);
              Split = Split.second.split(Delim);
            }
            if (!llvm::sys::fs::is_directory(Split.first))
              LogNonExistantDirectory(Split.first);
          }
          return false;
        } else if (Mode == kAllowNonExistant)
          Paths.push_back(Split.first);
        else if (Verbose)
          LogNonExistantDirectory(Split.first);
      } else
        Paths.push_back(Split.first);
    }

    PathStr = Split.second;
  }

  // Trim trailing sep in case of A:B:C:D:
  if (!PathStr.empty() && PathStr.ends_with(Delim))
    PathStr = PathStr.substr(0, PathStr.size()-Delim.size());

  if (!PathStr.empty()) {
    if (!llvm::sys::fs::is_directory(PathStr)) {
      AllExisted = false;
      if (Mode == kAllowNonExistant)
        Paths.push_back(PathStr);
      else if (Verbose)
        LogNonExistantDirectory(PathStr);
    } else
      Paths.push_back(PathStr);
  }

  return AllExisted;

#undef  DEBUG_TYPE
}

void AddIncludePaths(llvm::StringRef PathStr,
                         clang::HeaderSearchOptions& HOpts,
                         const char* Delim /* = Cpp::utils::platform::kEnvDelim */) {
#define DEBUG_TYPE "AddIncludePaths"

  llvm::SmallVector<llvm::StringRef, 10> Paths;
  if (Delim && *Delim)
    SplitPaths(PathStr, Paths, kAllowNonExistant, Delim, HOpts.Verbose);
  else
    Paths.push_back(PathStr);

  // Avoid duplicates
  llvm::SmallVector<llvm::StringRef, 10> PathsChecked;
  for (llvm::StringRef Path : Paths) {
    bool Exists = false;
    for (const clang::HeaderSearchOptions::Entry& E : HOpts.UserEntries) {
      if ((Exists = E.Path == Path))
        break;
    }
    if (!Exists)
      PathsChecked.push_back(Path);
  }

  const bool IsFramework = false;
  const bool IsSysRootRelative = true;
  for (llvm::StringRef Path : PathsChecked)
    HOpts.AddPath(Path, clang::frontend::Angled, IsFramework, IsSysRootRelative);

  if (HOpts.Verbose) {
    LLVM_DEBUG(dbgs() << "Added include paths:\n");
    for (llvm::StringRef Path : PathsChecked)
      LLVM_DEBUG(dbgs() << "  " << Path << "\n");
  }

#undef  DEBUG_TYPE
}

} // namespace utils
} // namespace Cpp
