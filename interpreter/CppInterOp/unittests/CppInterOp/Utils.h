#ifndef CPPINTEROP_UNITTESTS_LIBCPPINTEROP_UTILS_H
#define CPPINTEROP_UNITTESTS_LIBCPPINTEROP_UTILS_H

#include "../../lib/CppInterOp/Compatibility.h"

#include "clang-c/CXCppInterOp.h"
#include "clang-c/CXString.h"
#include "CppInterOp/CppInterOp.h"

#include "llvm/Support/Valgrind.h"

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

const char* get_c_string(CXString string);

void dispose_string(CXString string);

CXScope make_scope(const clang::Decl* D, const CXInterpreter I);

bool IsTargetX86();

// Define type tags for each configuration
struct InProcessJITConfig {
  static constexpr bool isOutOfProcess = false;
  static constexpr const char* name = "InProcessJIT";
};

#ifdef LLVM_BUILT_WITH_OOP_JIT
struct OutOfProcessJITConfig {
  static constexpr bool isOutOfProcess = true;
  static constexpr const char* name = "OutOfProcessJIT";
};
#endif

// Define typed test fixture
template <typename Config>
class CppInterOpTest : public ::testing::Test {
protected:
  void SetUp() override {
    TestUtils::current_config =
        TestUtils::TestConfig{Config::isOutOfProcess, Config::name};
  }

public:
  static TInterp_t CreateInterpreter(const std::vector<const char*>& Args = {},
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

#ifdef LLVM_BUILT_WITH_OOP_JIT
using CppInterOpTestTypes = ::testing::Types<InProcessJITConfig, OutOfProcessJITConfig>;
#else
using CppInterOpTestTypes = ::testing::Types<InProcessJITConfig>;
#endif

TYPED_TEST_SUITE(CppInterOpTest, CppInterOpTestTypes, JITConfigNameGenerator);


#endif // CPPINTEROP_UNITTESTS_LIBCPPINTEROP_UTILS_H
