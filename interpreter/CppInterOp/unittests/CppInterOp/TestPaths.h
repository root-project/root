#ifndef CPPINTEROP_UNITTESTS_LIBCPPINTEROP_TESTPATHS_H // NOLINT(llvm-header-guard)
#define CPPINTEROP_UNITTESTS_LIBCPPINTEROP_TESTPATHS_H

#include <cstdlib>
#include <string>

namespace TestUtils {

/// CppInterOp artifacts prefix: the directory that holds lib/ and include/.
/// The CPPINTEROP_BIN_DIR environment variable overrides the compile-time
/// default, which lets a relocated test binary find the library. Empty if
/// neither is set.
inline std::string GetCppInterOpDirPath() {
  if (const char* env = std::getenv("CPPINTEROP_BIN_DIR"))
    return env;
#ifdef CPPINTEROP_BIN_DIR
  return CPPINTEROP_BIN_DIR;
#else
  return {};
#endif
}

/// Full path to libclangCppInterOp under the artifacts prefix; empty if the
/// prefix is unknown.
inline std::string GetCppInterOpLibPath() {
  std::string dir = GetCppInterOpDirPath();
  if (dir.empty())
    return {};
#if defined(_WIN32)
  return dir + "/lib/libclangCppInterOp.dll";
#elif defined(__APPLE__)
  return dir + "/lib/libclangCppInterOp.dylib";
#else
  return dir + "/lib/libclangCppInterOp.so";
#endif
}

} // end namespace TestUtils

#endif // CPPINTEROP_UNITTESTS_LIBCPPINTEROP_TESTPATHS_H
