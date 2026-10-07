// Check that the include paths known to the interpreter at runtime stay slim.
// Every one of them is searched for every header lookup, which is slow on
// network file systems like cvmfs. Therefore, the dictionaries of ROOT's
// libraries must not register directories of ROOT's source or build tree other
// than the few that are needed at runtime (see the -excludePath arguments to
// rootcling). The interpreter drops paths that are registered twice, also if
// the spellings differ by redundant slashes, by the separator style or, on
// Windows, by case, e.g. C:\ROOT\include from TCling and C:/ROOT/include from
// the dictionaries; no such duplicate must slip through.
//
// Directories outside of ROOT's source and build trees are not checked: they
// come from the environment, like ROOT_INCLUDE_PATH or the include directories
// of external dependencies that are installed outside of the system prefixes
// (e.g. with Nix, LCG views or conda), so how many of them there are depends
// on the system ROOT was built on.
//
// This macro is interpreted, so it must not use <filesystem> or <regex>: on
// Windows, parts of their implementation are only in the static part of the
// MSVC runtime library, and the interpreter fails to resolve them.
// This can be revised if https://github.com/root-project/root/issues/23664
// is resolved.

#include "TInterpreter.h"
#include "TROOT.h"
#include "TSystem.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

// Removes redundant and trailing slashes as well as "." and ".." components,
// but keeps symlinks as they are. This is what lexically_normal() does, except
// for trailing slashes, but <filesystem> can't be used here (see above).
std::string lexicalPath(std::string path)
{
   std::replace(path.begin(), path.end(), '\\', '/');
   const bool absolute = !path.empty() && path[0] == '/';
   std::vector<std::string> components;
   for (std::size_t begin = 0; begin <= path.size();) {
      std::size_t end = path.find('/', begin);
      if (end == std::string::npos)
         end = path.size();
      const std::string component = path.substr(begin, end - begin);
      if (component == ".." && !components.empty() && components.back() != "..")
         components.pop_back();
      else if (!component.empty() && component != ".")
         components.push_back(component);
      begin = end + 1;
   }
   std::string result = absolute ? "/" : "";
   for (std::size_t i = 0; i < components.size(); ++i)
      result += (i ? "/" : "") + components[i];
   return result.empty() ? "." : result;
}

// Normalizes `path` so that two spellings of the same directory compare equal,
// like std::filesystem::weakly_canonical() would (see above for why it is not
// used). Beyond lexicalPath(), this requires one platform-dependent step:
//
// - On Linux and macOS, symlinks are resolved with realpath(), so that a
//   directory of ROOT's trees is recognized however it is spelled. The case is
//   left as it is: on macOS, whether paths are case-sensitive depends on the
//   volume, so folding the case could merge distinct directories.
// - On Windows, there is no realpath(). Instead, the path is converted to lower
//   case, because paths are case-insensitive there. This is also what the
//   Cling interpreter does when it drops duplicate include paths.
std::string canonicalPath(const std::string &path)
{
#ifdef _WIN32
   std::string result = lexicalPath(path);
   for (auto &c : result)
      c = std::tolower(static_cast<unsigned char>(c));
   return result;
#else
   char *resolved = realpath(path.c_str(), nullptr);
   if (!resolved)
      return lexicalPath(path);
   std::string result = lexicalPath(resolved);
   free(resolved);
   return result;
#endif
}

bool isInside(const std::string &path, const std::string &dir)
{
   return path == dir || path.rfind(dir + '/', 0) == 0;
}

bool contains(const std::vector<std::string> &paths, const std::string &path)
{
   return std::find(paths.begin(), paths.end(), path) != paths.end();
}

// The directories of the search path flags that TCling::GetIncludePath()
// returns, which have the form `<flag>"<dir>"` or `<flag> "<dir>"`.
std::vector<std::string> getIncludeDirs(const std::string &flags)
{
   std::vector<std::string> dirs;
   std::size_t pos = 0;
   while (true) {
      const std::size_t flagBegin = flags.find_first_not_of(' ', pos);
      if (flagBegin == std::string::npos)
         break;
      const std::size_t open = flags.find('"', flagBegin);
      const std::size_t close = open == std::string::npos ? open : flags.find('"', open + 1);
      if (close == std::string::npos) {
         std::cerr << "Cannot parse include path flags: " << flags.substr(flagBegin) << std::endl;
         break;
      }
      std::string flag = flags.substr(flagBegin, open - flagBegin);
      flag.erase(flag.find_last_not_of(' ') + 1);
      if (flag == "-I" || flag == "-iquote" || flag == "-idirafter")
         dirs.push_back(flags.substr(open + 1, close - open - 1));
      pos = close + 1;
   }
   return dirs;
}

std::vector<std::string> getEnvPaths(const char *name)
{
#ifdef _WIN32
   constexpr char kPathSep = ';';
#else
   constexpr char kPathSep = ':';
#endif
   std::vector<std::string> paths;
   const char *value = std::getenv(name);
   const std::string str = value ? value : "";
   for (std::size_t begin = 0; begin <= str.size();) {
      std::size_t end = str.find(kPathSep, begin);
      if (end == std::string::npos)
         end = str.size();
      if (end > begin)
         paths.push_back(str.substr(begin, end - begin));
      begin = end + 1;
   }
   return paths;
}

} // namespace

int checkRuntimeIncludePaths()
{
   for (auto lib : {"libMathCore", "libunordered_mapDict", "libmapDict", "libHist"})
      gSystem->Load(lib);

   // Set by the CMakeLists.txt.
   const char *rootSourceDir = std::getenv("ROOTTEST_ROOT_SOURCE_DIR");
   const char *rootBinaryDir = std::getenv("ROOTTEST_ROOT_BINARY_DIR");
   if (!rootSourceDir || !*rootSourceDir || !rootBinaryDir || !*rootBinaryDir) {
      std::cerr << "ROOTTEST_ROOT_SOURCE_DIR and ROOTTEST_ROOT_BINARY_DIR must be set." << std::endl;
      return 1;
   }
   const std::vector<std::string> srcAndBuildTrees{canonicalPath(rootSourceDir), canonicalPath(rootBinaryDir)};

   // The directories of ROOT's trees that may be include paths at runtime.
   const std::string etcDir = TROOT::GetEtcDir().Data();
   std::vector<std::string> allowed{
      canonicalPath(TROOT::GetIncludeDir().Data()),
      canonicalPath(std::string(rootBinaryDir) + "/include"), // recorded by rootcling in the dictionaries
      canonicalPath(etcDir),                                  // added by TCling
      canonicalPath(etcDir + "/cling"),
      canonicalPath(etcDir + "/cling/plugins/include"),
      canonicalPath(gSystem->WorkingDirectory()), // added by the roottest driver
   };
   for (auto const &path : getEnvPaths("ROOT_INCLUDE_PATH"))
      allowed.push_back(canonicalPath(path));

   // Only the interpreter's list matters for header lookups at runtime. The one
   // of TSystem holds the flags for ACLiC, and gSystem->GetIncludePath() only
   // appends the interpreter's list to it on some platforms.
   const std::string includePath = gInterpreter->GetIncludePath();

   int nErrors = 0;
   std::vector<std::string> seen;
   for (auto const &dir : getIncludeDirs(includePath)) {
      const std::string canonical = canonicalPath(dir);
      bool inTrees = false;
      for (auto const &tree : srcAndBuildTrees)
         inTrees = inTrees || isInside(canonical, tree);
      if (!inTrees)
         continue;
      if (!contains(allowed, canonical)) {
         std::cerr << "Unexpected include path in ROOT's source or build tree: " << dir << std::endl;
         ++nErrors;
      }
      // Spellings that only differ by symlinks can come from the environment,
      // e.g. if ROOT is set up via a symlinked path, so they are not compared.
      const std::string lexical = lexicalPath(dir);
      if (contains(seen, lexical)) {
         std::cerr << "Duplicate include path: " << dir << std::endl;
         ++nErrors;
      }
      seen.push_back(lexical);
   }

   if (nErrors > 0) {
      std::cerr << "ROOT's source tree: " << srcAndBuildTrees[0] << "\n"
                << "ROOT's build tree: " << srcAndBuildTrees[1] << "\n"
                << "gInterpreter->GetIncludePath(): " << includePath << std::endl;
   }
   return nErrors;
}
