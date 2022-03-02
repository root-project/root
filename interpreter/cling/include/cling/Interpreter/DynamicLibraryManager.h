//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_DYNAMIC_LIBRARY_MANAGER_H
#define CLING_DYNAMIC_LIBRARY_MANAGER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

#include "llvm/Support/Path.h"

namespace cling {
  class Dyld;
  class InterpreterCallbacks;

  ///\brief A helper class managing dynamic shared objects.
  ///
  class DynamicLibraryManager {
  public:
    ///\brief Describes the result of loading a library.
    ///
    enum LoadLibResult {
      kLoadLibSuccess, ///< library loaded successfully
      kLoadLibAlreadyLoaded,  ///< library was already loaded
      kLoadLibNotFound, ///< library was not found
      kLoadLibLoadError, ///< loading the library failed
      kLoadLibNumResults
    };

    /// Describes the library search paths.
    struct SearchPathInfo {
      /// The search path.
      ///
      std::string Path;

      /// True if the Path is on the LD_LIBRARY_PATH.
      ///
      bool IsUser;

      bool operator==(const SearchPathInfo& Other) const {
        return IsUser == Other.IsUser && Path == Other.Path;
      }
    };
    using SearchPathInfos = llvm::SmallVector<SearchPathInfo, 32>;
  private:
    typedef const void* DyLibHandle;
    typedef llvm::DenseMap<DyLibHandle, std::string> DyLibs;
    ///\brief DynamicLibraries loaded by this Interpreter.
    ///
    DyLibs m_DyLibs;
    llvm::StringSet<> m_LoadedLibraries;

    ///\brief System's include path, get initialized at construction time.
    ///
    SearchPathInfos m_SearchPaths;

    InterpreterCallbacks* m_Callbacks = nullptr;

    Dyld* m_Dyld = nullptr;

    ///\brief Concatenates current include paths and the system include paths
    /// and performs a lookup for the filename.
    /// See more information for RPATH and RUNPATH: https://en.wikipedia.org/wiki/Rpath
    ///\param[in] libStem - The filename being looked up
    ///\param[in] RPath - RPATH as provided by loader library, searching for libStem
    ///\param[in] RunPath - RUNPATH as provided by loader library, searching for libStem
    ///\param[in] libLoader - The library that loads libStem. Use "" for main program.
    ///
    ///\returns the canonical path to the file or empty string if not found
    ///
    std::string lookupLibInPaths(llvm::StringRef libStem,
                                 llvm::SmallVector<llvm::StringRef,2> RPath = {},
                                 llvm::SmallVector<llvm::StringRef,2> RunPath = {},
                                 llvm::StringRef libLoader = "") const;

    ///\brief Concatenates current include paths and the system include paths
    /// and performs a lookup for the filename. If still not found it tries to
    /// add the platform-specific extensions (such as so, dll, dylib) and
    /// retries the lookup (from lookupLibInPaths)
    /// See more information for RPATH and RUNPATH: https://en.wikipedia.org/wiki/Rpath
    ///\param[in] filename - The filename being looked up
    ///\param[in] RPath - RPATH as provided by loader library, searching for libStem
    ///\param[in] RunPath - RUNPATH as provided by loader library, searching for libStem
    ///\param[in] libLoader - The library that loads libStem. Use "" for main program.
    ///
    ///\returns the canonical path to the file or empty string if not found
    ///
    std::string lookupLibMaybeAddExt(llvm::StringRef filename,
                                     llvm::SmallVector<llvm::StringRef,2> RPath = {},
                                     llvm::SmallVector<llvm::StringRef,2> RunPath = {},
                                     llvm::StringRef libLoader = "") const;

    /// On a success returns to full path to a shared object that holds the
    /// symbol pointed by func.
    ///
    static std::string getSymbolLocation(void* func);
  public:
    DynamicLibraryManager();
    ~DynamicLibraryManager();
    DynamicLibraryManager(const DynamicLibraryManager&) = delete;
    DynamicLibraryManager& operator=(const DynamicLibraryManager&) = delete;

    InterpreterCallbacks* getCallbacks() { return m_Callbacks; }
    const InterpreterCallbacks* getCallbacks() const { return m_Callbacks; }
    void setCallbacks(InterpreterCallbacks* C) { m_Callbacks = C; }

    ///\brief Returns the system include paths.
    ///
    ///\returns System include paths.
    ///
    const SearchPathInfos& getSearchPaths() const {
       return m_SearchPaths;
    }

    void addSearchPath(llvm::StringRef dir, bool isUser = true,
                       bool prepend = false) {
       if (!dir.empty()) {
          for (auto & item : m_SearchPaths)
            if (dir.equals(item.Path)) return;
          auto pos = prepend ? m_SearchPaths.begin() : m_SearchPaths.end();
          m_SearchPaths.insert(pos, SearchPathInfo{dir, isUser});
       }
    }

    ///\brief Looks up a library taking into account the current include paths
    /// and the system include paths.
    /// See more information for RPATH and RUNPATH: https://en.wikipedia.org/wiki/Rpath
    ///\param[in] libStem - The filename being looked up
    ///\param[in] RPath - RPATH as provided by loader library, searching for libStem
    ///\param[in] RunPath - RUNPATH as provided by loader library, searching for libStem
    ///\param[in] libLoader - The library that loads libStem. Use "" for main program.
    ///\param[in] variateLibStem - If this param is true, and libStem is "L", then
    ///              we search for "L", "libL", "L.so", "libL.so"", etc.
    ///
    ///\returns the canonical path to the file or empty string if not found
    ///
    std::string lookupLibrary(llvm::StringRef libStem,
                              llvm::SmallVector<llvm::StringRef,2> RPath = {},
                              llvm::SmallVector<llvm::StringRef,2> RunPath = {},
                              llvm::StringRef libLoader = "",
                              bool variateLibStem = true) const;

    ///\brief Loads a shared library.
    ///
    ///\param [in] libStem - The file to load.
    ///\param [in] permanent - If false, the file can be unloaded later.
    ///\param [in] resolved - Whether libStem is an absolute path or resolved
    ///               from a previous call to DynamicLibraryManager::lookupLibrary
    ///
    ///\returns kLoadLibSuccess on success, kLoadLibAlreadyLoaded if the library
    /// was already loaded, kLoadLibError if the library cannot be found or any
    /// other error was encountered.
    ///
    LoadLibResult loadLibrary(llvm::StringRef, bool permanent,
                              bool resolved = false);

    void unloadLibrary(llvm::StringRef libStem);

    ///\brief Returns true if the file was a dynamic library and it was already
    /// loaded.
    ///
    bool isLibraryLoaded(llvm::StringRef fullPath) const;

    /// Initialize the dyld.
    ///
    ///\param [in] shouldPermanentlyIgnore - a callback deciding if a library
    ///            should be ignored from the result set. Useful for ignoring
    ///            dangerous libraries such as the ones overriding malloc.
    ///
    void
    initializeDyld(std::function<bool(llvm::StringRef)> shouldPermanentlyIgnore);

    /// Find the first not-yet-loaded shared object that contains the symbol
    ///
    ///\param[in] mangledName - the mangled name to look for.
    ///\param[in] searchSystem - whether to decend into system libraries.
    ///
    ///\returns the library name if found, and empty string otherwise.
    ///
    std::string searchLibrariesForSymbol(llvm::StringRef mangledName,
                                         bool searchSystem = true) const;

    void dump(llvm::raw_ostream* S = nullptr) const;

    /// On a success returns to full path to a shared object that holds the
    /// symbol pointed by func.
    ///
    template <class T>
    static std::string getSymbolLocation(T func) {
      static_assert(std::is_pointer<T>::value, "Must be a function pointer!");
      return getSymbolLocation(reinterpret_cast<void*>(func));
    }


    ///\brief Explicitly tell the execution engine to use symbols from
    ///       a shared library that would otherwise not be used for symbol
    ///       resolution, e.g. because it was dlopened with RTLD_LOCAL.
    ///\param [in] handle - the system specific shared library handle.
    ///
    static void ExposeHiddenSharedLibrarySymbols(void* handle);

    static std::string normalizePath(llvm::StringRef path);

    /// Returns true if file is a shared library.
    ///
    ///\param[in] libFullPath - the full path to file.
    ///
    ///\param[out] exists - sets if the file exists. Useful to distinguish if it
    ///            is a library but of incompatible file format.
    ///
    static bool isSharedLibrary(llvm::StringRef libFullPath, bool* exists = 0);
  };
} // end namespace cling
#endif // CLING_DYNAMIC_LIBRARY_MANAGER_H
