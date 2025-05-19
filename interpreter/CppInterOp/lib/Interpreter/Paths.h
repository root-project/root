//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CPPINTEROP_UTILS_PATHS_H
#define CPPINTEROP_UTILS_PATHS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>

namespace llvm {
class raw_ostream;
}

namespace clang {
class HeaderSearchOptions;
class FileManager;
} // namespace clang

namespace Cpp {
namespace utils {

namespace platform {
///\brief Platform specific delimiter for splitting environment variables.
/// ':' on Unix, and ';' on Windows
extern const char* const kEnvDelim;

///
bool GetSystemLibraryPaths(llvm::SmallVectorImpl<std::string>& Paths);

///\brief Returns a normalized version of the given Path
///
std::string NormalizePath(const std::string& Path);

///\brief Open a handle to a shared library. On Unix the lib is opened with
/// RTLD_LAZY|RTLD_GLOBAL flags.
///
/// \param [in] Path - Library to open
/// \param [out] Err - Write errors to this string when given
///
/// \returns the library handle
///
void* DLOpen(const std::string& Path, std::string* Err = nullptr);

///\brief Close a handle to a shared library.
///
/// \param [in] Lib - Handle to library from previous call to DLOpen
/// \param [out] Err - Write errors to this string when given
///
/// \returns the library handle
///
void DLClose(void* Lib, std::string* Err = nullptr);
} // namespace platform

enum SplitMode {
  kPruneNonExistent, ///< Don't add non-existent paths into output
  kFailNonExistent,  ///< Fail on any non-existent paths
  kAllowNonExistent  ///< Add all paths whether they exist or not
};

///\brief Collect the constituent paths from a PATH string.
/// /bin:/usr/bin:/usr/local/bin -> {/bin, /usr/bin, /usr/local/bin}
///
/// All paths returned existed at the time of the call
/// \param [in] PathStr - The PATH string to be split
/// \param [out] Paths - All the paths in the string that exist
/// \param [in] Mode - If any path doesn't exist stop and return false
/// \param [in] Delim - The delimiter to use
/// \param [in] Verbose - Whether to print out details as 'clang -v' would
///
/// \return true if all paths existed, otherwise false
///
bool SplitPaths(llvm::StringRef PathStr,
                llvm::SmallVectorImpl<llvm::StringRef>& Paths,
                SplitMode Mode = kPruneNonExistent,
                llvm::StringRef Delim = Cpp::utils::platform::kEnvDelim,
                bool Verbose = false);

///\brief Adds multiple include paths separated by a delimiter into the
/// given HeaderSearchOptions.  This adds the paths but does no further
/// processing. See Interpreter::AddIncludePaths or CIFactory::createCI
/// for examples of what needs to be done once the paths have been added.
///
///\param[in] PathStr - Path(s)
///\param[in] Opts - HeaderSearchOptions to add paths into
///\param[in] Delim - Delimiter to separate paths or NULL if a single path
///
void AddIncludePaths(llvm::StringRef PathStr, clang::HeaderSearchOptions& HOpts,
                     const char* Delim = Cpp::utils::platform::kEnvDelim);

///\brief Write to cling::errs that directory does not exist in a format
/// matching what 'clang -v' would do
///
void LogNonExistentDirectory(llvm::StringRef Path);

///\brief Copies the current include paths into the HeaderSearchOptions.
///
///\param[in] Opts - HeaderSearchOptions to read from
///\param[out] Paths - Vector to output elements into
///\param[in] WithSystem - if true, incpaths will also contain system
///       include paths (framework, STL etc).
///\param[in] WithFlags - if true, each element in incpaths will be prefixed
///       with a "-I" or similar, and some entries of incpaths will signal
///       a new include path region (e.g. "-cxx-isystem"). Also, flags
///       defining header search behavior will be included in incpaths, e.g.
///       "-nostdinc".
///
void CopyIncludePaths(const clang::HeaderSearchOptions& Opts,
                      llvm::SmallVectorImpl<std::string>& Paths,
                      bool WithSystem, bool WithFlags);

} // namespace utils
} // namespace Cpp

#endif // CPPINTEROP_UTILS_PATHS_H
