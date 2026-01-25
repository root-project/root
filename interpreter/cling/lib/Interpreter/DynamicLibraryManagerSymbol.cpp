//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
// author:  Alexander Penev <alexander_penev@yahoo.com>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Interpreter/DynamicLibraryManager.h"
#include "cling/Utils/Paths.h"
#include "cling/Utils/Platform.h"
#include "cling/Utils/Output.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <list>
#include <string>
#include <unordered_set>
#include <vector>

#if defined (__FreeBSD__)
#include <sys/user.h>
#include <sys/types.h>
#include <sys/param.h>
#include <sys/queue.h>

// libprocstat pulls in sys/elf.h which seems to clash with llvm/BinaryFormat/ELF.h
// similar collision happens with ZFS. Defining ZFS disables this include.
# ifndef ZFS
#   define ZFS
#   define defined_ZFS_for_libprocstat
# endif
#include <libprocstat.h>
# ifdef defined_ZFS_for_libprocstat
#   undef ZFS
#   undef defined_ZFS_for_libprocstat
# endif

#include <libutil.h>
#endif

#ifdef LLVM_ON_UNIX
#include <dlfcn.h>
#include <unistd.h>
#include <sys/stat.h>
#endif // LLVM_ON_UNIX

#ifdef __APPLE__
#include <mach-o/dyld.h>
#include <sys/stat.h>
#undef LC_LOAD_DYLIB
#undef LC_RPATH
#endif // __APPLE__

#ifdef _WIN32
#include <windows.h>
#include <libloaderapi.h> // For GetModuleFileNameA
#include <memoryapi.h> // For VirtualQuery
#endif

namespace {

using BasePath = std::string;

#ifndef _WIN32
// Cached version of system function lstat
static inline mode_t cached_lstat(const char *path) {
  static llvm::StringMap<mode_t> lstat_cache;

  // If already cached - retun cached result
  auto it = lstat_cache.find(path);
  if (it != lstat_cache.end())
    return it->second;

  // If result not in cache - call system function and cache result
  struct stat buf;
  mode_t st_mode = (lstat(path, &buf) == -1) ? 0 : buf.st_mode;
  lstat_cache.insert(std::pair<llvm::StringRef, mode_t>(path, st_mode));
  return st_mode;
}

// Cached version of system function readlink
static inline llvm::StringRef cached_readlink(const char* pathname) {
  static llvm::StringMap<std::string> readlink_cache;

  // If already cached - retun cached result
  auto it = readlink_cache.find(pathname);
  if (it != readlink_cache.end())
    return llvm::StringRef(it->second);

  // If result not in cache - call system function and cache result
  char buf[PATH_MAX];
  ssize_t len;
  if ((len = readlink(pathname, buf, sizeof(buf))) != -1) {
    buf[len] = '\0';
    std::string s(buf);
    readlink_cache.insert(std::pair<llvm::StringRef, std::string>(pathname, s));
    return readlink_cache[pathname];
  }
  return "";
}
#endif

// Cached version of system function realpath
std::string cached_realpath(llvm::StringRef path, llvm::StringRef base_path = "",
                            bool is_base_path_real = false,
                            long symlooplevel = 40) {
  if (path.empty()) {
    errno = ENOENT;
    return "";
  }

  if (!symlooplevel) {
    errno = ELOOP;
    return "";
  }

  // If already cached - retun cached result
  static llvm::StringMap<std::pair<std::string,int>> cache;
  bool relative_path = llvm::sys::path::is_relative(path);
  if (!relative_path) {
    auto it = cache.find(path);
    if (it != cache.end()) {
      errno = it->second.second;
      return it->second.first;
    }
  }

  // If result not in cache - call system function and cache result

  llvm::StringRef sep(llvm::sys::path::get_separator());
  llvm::SmallString<256> result(sep);
#ifndef _WIN32
  llvm::SmallVector<llvm::StringRef, 16> p;

  // Relative or absolute path
  if (relative_path) {
    if (is_base_path_real) {
      result.assign(base_path);
    } else {
      if (path[0] == '~' && (path.size() == 1 || llvm::sys::path::is_separator(path[1]))) {
        static llvm::SmallString<128> home;
        if (home.str().empty())
          llvm::sys::path::home_directory(home);
        llvm::StringRef(home).split(p, sep, /*MaxSplit*/ -1, /*KeepEmpty*/ false);
      } else if (base_path.empty()) {
        static llvm::SmallString<256> current_path;
        if (current_path.str().empty())
          llvm::sys::fs::current_path(current_path);
        llvm::StringRef(current_path).split(p, sep, /*MaxSplit*/ -1, /*KeepEmpty*/ false);
      } else {
        base_path.split(p, sep, /*MaxSplit*/ -1, /*KeepEmpty*/ false);
      }
    }
  }
  path.split(p, sep, /*MaxSplit*/ -1, /*KeepEmpty*/ false);

  // Handle path list items
  for (auto item : p) {
    if (item == ".")
      continue; // skip "." element in "abc/./def"
    if (item == "..") {
      // collapse "a/b/../c" to "a/c"
      size_t s = result.rfind(sep);
      if (s != llvm::StringRef::npos)
        result.resize(s);
      if (result.empty())
        result = sep;
      continue;
    }

    size_t old_size = result.size();
    llvm::sys::path::append(result, item);
    mode_t st_mode = cached_lstat(result.c_str());
    if (S_ISLNK(st_mode)) {
      llvm::StringRef symlink = cached_readlink(result.c_str());
      if (llvm::sys::path::is_relative(symlink)) {
        result.resize(old_size);
        result = cached_realpath(symlink, result, true, symlooplevel - 1);
      } else {
        result = cached_realpath(symlink, "", true, symlooplevel - 1);
      }
    } else if (st_mode == 0) {
      cache.insert(std::pair<llvm::StringRef, std::pair<std::string,int>>(
        path,
        std::pair<std::string,int>("",ENOENT))
      );
      errno = ENOENT;
      return "";
    }
  }
#else
  llvm::sys::fs::real_path(path, result);
#endif
  cache.insert(std::pair<llvm::StringRef, std::pair<std::string,int>>(
    path,
    std::pair<std::string,int>(result.str().str(),errno))
  );
  return result.str().str();
}

} // anon namespace

namespace cling {

  DynamicLibraryManager::~DynamicLibraryManager() {
    // static_assert(sizeof(Dyld) > 0, "Incomplete type");
    // delete m_Dyld;
  }

  void DynamicLibraryManager::initializeDyld(
                 std::function<bool(llvm::StringRef)> shouldPermanentlyIgnore) {
    //  assert(!m_Dyld && "Already initialized!");
    if (m_DyldController)
      m_DyldController.reset();

    llvm::orc::LibraryResolver::Setup S =
        llvm::orc::LibraryResolver::Setup::create({});
    S.ShouldScanCall = [&,
                        shouldPermanentlyIgnore](llvm::StringRef lib) -> bool {
      if (shouldPermanentlyIgnore) {
        return !shouldPermanentlyIgnore(lib) || !isLibraryLoaded(lib);
      }
      // fallback behavior if no callback provided
      return !isLibraryLoaded(lib);
    };
    m_DyldController = llvm::orc::LibraryResolutionDriver::create(S);

    for (const auto& info : m_SearchPaths) {
      m_DyldController->addScanPath(info.Path,
                                    info.IsUser ? llvm::orc::PathType::User
                                                : llvm::orc::PathType::System);
    }

    for (const auto& lib : m_LoadedLibraries) {
      m_DyldController->markLibraryLoaded(lib.first());
    }
  }

  std::string
  DynamicLibraryManager::searchLibrariesForSymbol(llvm::StringRef mangledName,
                                           bool searchSystem/* = true*/) const {
    assert(m_DyldController && "Must call initialize dyld before!");
    std::string res = "";
    llvm::orc::SearchConfig config;
    config.Policy = {
        {{llvm::orc::LibState::Queried, llvm::orc::PathType::User},
         {llvm::orc::LibState::Unloaded, llvm::orc::PathType::User},
         {llvm::orc::LibState::Queried, llvm::orc::PathType::System},
         {llvm::orc::LibState::Unloaded, llvm::orc::PathType::System}}};
    config.Options.FilterFlags =
        llvm::orc::SymbolEnumeratorOptions::IgnoreUndefined;
    m_DyldController->resolveSymbols(
        {mangledName},
        [&](llvm::orc::LibraryResolver::SymbolQuery& Q) {
          if (auto s = Q.getResolvedLib(mangledName))
            res = *s;
        },
        config);
    return res;
  }

  void DynamicLibraryManager::searchLibrariesForSymbol(
      llvm::ArrayRef<llvm::StringRef> Symbols,
      llvm::orc::LibraryResolver::OnSearchComplete OnCompletion,
      bool searchSystem) const {
    llvm::orc::SearchConfig config;
    config.Policy = {
        {{llvm::orc::LibState::Queried, llvm::orc::PathType::User},
         {llvm::orc::LibState::Unloaded, llvm::orc::PathType::User},
         {llvm::orc::LibState::Queried, llvm::orc::PathType::System},
         {llvm::orc::LibState::Unloaded, llvm::orc::PathType::System}}};
    config.Options.FilterFlags =
        llvm::orc::SymbolEnumeratorOptions::IgnoreUndefined;
    m_DyldController->resolveSymbols(Symbols, std::move(OnCompletion), config);
    return;
  }

  std::string DynamicLibraryManager::getSymbolLocation(void *func) {
  // auto cached_realpath = [&] (StringRef Lib) -> std::string {
  //   auto ResOrOpt = m_DyldController->getPathResolver().resolve(Lib);
  //   return ResOrOpt ? *ResOrOpt : "";
  // }
#if defined(__CYGWIN__) && defined(__GNUC__)
    return {};
#elif defined(_WIN32)
    MEMORY_BASIC_INFORMATION mbi;
    if (!VirtualQuery (func, &mbi, sizeof (mbi)))
      return {};

    HMODULE hMod = (HMODULE) mbi.AllocationBase;
    char moduleName[MAX_PATH];

    if (!GetModuleFileNameA (hMod, moduleName, sizeof (moduleName)))
      return {};

    return cached_realpath(moduleName);

#else
    // assume we have  defined HAVE_DLFCN_H and HAVE_DLADDR
    Dl_info info;
    if (dladdr((void*)func, &info) == 0) {
      // Not in a known shared library, let's give up
      return {};
    } else {
      std::string result = cached_realpath(info.dli_fname);
      if (!result.empty())
        return result;

      // Else absolute path. For all we know that's a binary.
      // Some people have dictionaries in binaries, this is how we find their
      // path: (see also https://stackoverflow.com/a/1024937/6182509)
# if defined(__APPLE__)
      char buf[PATH_MAX] = { 0 };
      uint32_t bufsize = sizeof(buf);
      if (_NSGetExecutablePath(buf, &bufsize) >= 0)
        return cached_realpath(buf);
      return cached_realpath(info.dli_fname);
# elif defined (__FreeBSD__)
      procstat* ps = procstat_open_sysctl();
      kinfo_proc* kp = kinfo_getproc(getpid());

      char buf[PATH_MAX] = "";
      if (kp!=NULL) {
        procstat_getpathname(ps, kp, buf, sizeof(buf));
      };
      free(kp);
      procstat_close(ps);
      return cached_realpath(buf);
# elif defined(LLVM_ON_UNIX)
      char buf[PATH_MAX] = { 0 };
      // Cross our fingers that /proc/self/exe exists.
      if (readlink("/proc/self/exe", buf, sizeof(buf)) > 0)
        return cached_realpath(buf);
      std::string pipeCmd = std::string("which \"") + info.dli_fname + "\"";
      FILE* pipe = popen(pipeCmd.c_str(), "r");
      if (!pipe)
        return cached_realpath(info.dli_fname);
      while (fgets(buf, sizeof(buf), pipe))
         result += buf;

      pclose(pipe);
      return cached_realpath(result);
# else
#  error "Unsupported platform."
# endif
      return {};
   }
#endif
  }

} // namespace cling
