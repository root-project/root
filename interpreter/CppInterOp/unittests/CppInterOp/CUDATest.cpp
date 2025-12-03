#include "Utils.h"

#include "CppInterOp/CppInterOp.h"

#include "clang/Basic/Version.h"

#include "gtest/gtest.h"

using namespace TestUtils;

static bool HasCudaSDK() {
  auto supportsCudaSDK = []() {
#ifdef CPPINTEROP_USE_CLING
    // FIXME: Enable this for cling.
    return false;
#endif
    if (!Cpp::CreateInterpreter({}, {"--cuda"}))
      return false;
    return Cpp::Declare("__global__ void test_func() {}"
                        "test_func<<<1,1>>>();") == 0;
  };
  static bool hasCuda = supportsCudaSDK();
  return hasCuda;
}

static bool HasCudaRuntime() {
  auto supportsCuda = []() {
#ifdef CPPINTEROP_USE_CLING
    // FIXME: Enable this for cling.
    return false;
#endif
    if (!HasCudaSDK())
      return false;

    if (!Cpp::CreateInterpreter({}, {"--cuda"}))
      return false;
    if (Cpp::Declare("__global__ void test_func() {}"
                     "test_func<<<1,1>>>();"))
      return false;
    intptr_t result = Cpp::Evaluate("(bool)cudaGetLastError()");
    return !(bool)result;
  };
  static bool hasCuda = supportsCuda();
  return hasCuda;
}

#ifdef CPPINTEROP_USE_CLING
TEST(DISABLED_CUDATest, Sanity) {
#else
TEST(CUDATest, Sanity) {
#endif
#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif
  if (!HasCudaSDK())
    GTEST_SKIP() << "Skipping CUDA tests as CUDA SDK not found";
  EXPECT_TRUE(Cpp::CreateInterpreter({}, {"--cuda"}));
}

TEST(CUDATest, CUDAH) {
#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif
  if (!HasCudaSDK())
    GTEST_SKIP() << "Skipping CUDA tests as CUDA SDK not found";

  Cpp::CreateInterpreter({}, {"--cuda"});
  bool success = !Cpp::Declare("#include <cuda.h>");
  EXPECT_TRUE(success);
}

TEST(CUDATest, CUDARuntime) {
#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif
  if (!HasCudaRuntime())
    GTEST_SKIP() << "Skipping CUDA tests as CUDA runtime not found";

  EXPECT_TRUE(HasCudaRuntime());
}
