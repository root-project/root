// Storage definitions and initialization for dispatch function pointers.
// Linked only by DispatchTests — not by the normal test suite.

#include "TestPaths.h"
#include "CppInterOp/Dispatch.h"

#include <cstdlib>
#include <iostream>
#include <string>

// Define storage for all raw dispatch function pointers.
using namespace Cpp;
#define CPPINTEROP_API_FUNC(DN, CN, Ret, DeclArgs, CallArgs, RawTypes)         \
  Ret(*CppInternal::DispatchRaw::DN) RawTypes = nullptr;
#include "CppInterOp/CppInterOpAPI.inc"

namespace {
struct DispatchInitializer {
  DispatchInitializer() {
    std::string libPath = TestUtils::GetCppInterOpLibPath();
    if (libPath.empty()) {
      std::cerr << "DispatchTests: cannot locate libclangCppInterOp: set the "
                   "CPPINTEROP_BIN_DIR environment variable to the CppInterOp "
                   "artifacts prefix (the directory containing lib/ and "
                   "include/).\n";
      std::exit(EXIT_FAILURE);
    }
    if (!Cpp::LoadDispatchAPI(libPath.c_str())) {
      std::cerr << "DispatchTests: failed to load dispatch API from '"
                << libPath << "'.\n";
      std::exit(EXIT_FAILURE);
    }
  }
};
DispatchInitializer& GetDispatchInitializer() {
  static DispatchInitializer instance;
  return instance;
}
const DispatchInitializer& g_dispatch_init = GetDispatchInitializer();
} // namespace
