#ifndef CPPINTEROP_UNITTESTS_LIBCPPINTEROP_UTILS_H
#define CPPINTEROP_UNITTESTS_LIBCPPINTEROP_UTILS_H

#include "../../lib/CppInterOp/Compatibility.h"

#include "CppInterOp/CppInterOp.h"
#define CPPINTEROP_TEST_MODE CppInterOpTest

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "gtest/gtest.h"

using namespace clang;
using namespace llvm;

namespace clang {
class Decl;
}
#define Interp (static_cast<compat::Interpreter*>(Cpp::GetInterpreter()))
namespace TestUtils {

struct TestConfig {
    std::string name;
    bool use_oop_jit;

    TestConfig(bool oop_jit, const std::string& n) 
        : name(std::move(n)), use_oop_jit(oop_jit) {}

    TestConfig() 
        : name("InProcessJIT"), use_oop_jit(false) {}
};

extern TestConfig current_config;

// Helper to get interpreter args with current config
std::vector<const char*>
GetInterpreterArgs(const std::vector<const char*>& base_args = {});

void GetAllTopLevelDecls(const std::string& code,
                         std::vector<clang::Decl*>& Decls,
                         bool filter_implicitGenerated = false,
                         const std::vector<const char*>& interpreter_args = {});
void GetAllSubDecls(clang::Decl* D, std::vector<clang::Decl*>& SubDecls,
                    bool filter_implicitGenerated = false);
} // end namespace TestUtils

bool IsTargetX86();

// OOP-JIT is incompatible with two configurations and is excluded
// from the typed-test matrix wholesale (rather than per-test) when
// either applies:
//   * Any sanitizer (ASan/MSan/TSan): upstream LLVM ORC trips
//     `Resolving symbol with incorrect flags`
//     (`llvm/lib/ExecutionEngine/Orc/Core.cpp`, the JITSymbolFlags
//     compare under `OL_notifyResolved`) because
//     sanitizer-instrumented common symbols carry flags the host
//     process didn't declare; the EPC boundary surfaces the
//     mismatch. In-process JIT is unaffected.
//   * Emscripten: the OOP path requires fork/exec + a separate
//     executor binary, which the wasm runtime doesn't provide.
#if defined(__has_feature)
#  if __has_feature(address_sanitizer) ||                                      \
      __has_feature(memory_sanitizer) ||                                       \
      __has_feature(thread_sanitizer)
#    define CPPINTEROP_OOP_DISABLED 1
#  endif
#endif
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#  define CPPINTEROP_OOP_DISABLED 1
#endif
#if defined(__EMSCRIPTEN__)
#  define CPPINTEROP_OOP_DISABLED 1
#endif

// Define type tags for each configuration
struct InProcessJITConfig {
  static constexpr bool isOutOfProcess = false;
  static constexpr const char* name = "InProcessJIT";
};

#if LLVM_VERSION_MAJOR > 21 && !defined(_WIN32) &&                             \
    !defined(CPPINTEROP_OOP_DISABLED)
struct OutOfProcessJITConfig {
  static constexpr bool isOutOfProcess = true;
  static constexpr const char* name = "OutOfProcessJIT";
};
#endif

// Define typed test fixture
template <typename Config> class CPPINTEROP_TEST_MODE : public ::testing::Test {
protected:
  void SetUp() override {
    TestUtils::current_config =
        TestUtils::TestConfig{Config::isOutOfProcess, Config::name};
  }

public:
  static Cpp::TInterp_t
  CreateInterpreter(const std::vector<const char*>& Args = {},
                    const std::vector<const char*>& GpuArgs = {}) {
    auto mergedArgs = TestUtils::GetInterpreterArgs(Args);
    return Cpp::CreateInterpreter(mergedArgs, GpuArgs);
  }

  bool IsOutOfProcess() {
    return Config::isOutOfProcess;
  }
};

struct JITConfigNameGenerator {
  template <typename T>
  static std::string GetName(int) {
    return T::name;
  }
};

#if LLVM_VERSION_MAJOR > 21 && !defined(_WIN32) &&                             \
    !defined(CPPINTEROP_OOP_DISABLED)
using CppInterOpTestTypes = ::testing::Types<InProcessJITConfig, OutOfProcessJITConfig>;
#else
using CppInterOpTestTypes = ::testing::Types<InProcessJITConfig>;
#endif

TYPED_TEST_SUITE(CPPINTEROP_TEST_MODE, CppInterOpTestTypes,
                 JITConfigNameGenerator);

#endif // CPPINTEROP_UNITTESTS_LIBCPPINTEROP_UTILS_H
