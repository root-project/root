#include "CppInterOp/Dispatch.h"

#include "gtest/gtest.h"

#include <string>

// These tests verify the dispatch mechanism itself (dlGetProcAddress,
// LoadDispatchAPI, UnloadDispatchAPI). They must NOT link against
// libclangCppInterOp to ensure true RTLD_LOCAL isolation.

// NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)

TEST(DispatchTest, DlGetProcAddress_Basic) {
  Cpp::CreateInterpreter({});
  Cpp::Declare("namespace N {} class C{}; int I;");

  using IsClassFn_t = bool (*)(Cpp::TCppScope_t);
  auto* IsClassFn = reinterpret_cast<IsClassFn_t>(dlGetProcAddress("IsClass"));
  EXPECT_NE(IsClassFn, nullptr) << "Failed to obtain API function pointer";

  auto* ns = Cpp::GetNamed("N");
  auto* cls = Cpp::GetNamed("C");
  auto* var = Cpp::GetNamed("I");

  EXPECT_FALSE(IsClassFn(ns));
  EXPECT_TRUE(IsClassFn(cls));
  EXPECT_FALSE(IsClassFn(var));
}

TEST(DispatchTest, DlGetProcAddress_Unimplemented) {
  // dlGetProcAddress finds the library, dlsym's CppGetProcAddress, and calls
  // it with the unknown name. CppGetProcAddress returns nullptr and prints
  // a diagnostic to stderr.
  testing::internal::CaptureStderr();
  void* ptr = dlGetProcAddress("NonExistentFunction");
  std::string err = testing::internal::GetCapturedStderr();
  EXPECT_FALSE(err.empty());
  EXPECT_EQ(ptr, nullptr);
}

TEST(DispatchTest, LoadUnloadCycle) {
  Cpp::UnloadDispatchAPI(); // make sure we're not loaded already...
  EXPECT_FALSE(Cpp::LoadDispatchAPI("some/random/invalid/directory.so"));

  Cpp::UnloadDispatchAPI(); // should reset for next load to be successful
  EXPECT_TRUE(Cpp::LoadDispatchAPI(CPPINTEROP_LIB_PATH));

  Cpp::UnloadDispatchAPI(); // should reset for next load to fail
  EXPECT_FALSE(Cpp::LoadDispatchAPI("some/other/random/invalid/directory.so"));

  // NOTE: minimize side-effects: reload assuming the static set is still
  // and other test may depend on this being loaded
  EXPECT_TRUE(Cpp::LoadDispatchAPI(CPPINTEROP_LIB_PATH));
}

// NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
