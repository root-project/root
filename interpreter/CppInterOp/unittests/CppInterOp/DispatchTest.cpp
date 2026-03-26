#include "Utils.h"

#include "CppInterOp/Dispatch.h"

#include "clang/AST/Decl.h"
#include "clang/AST/GlobalDecl.h"

#include "gtest/gtest.h"

#include <vector>

using namespace TestUtils;
using namespace llvm;
using namespace clang;

// NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)

TEST(CPPINTEROP_TEST_MODE, DispatchAPI_CppGetProcAddress_Basic) {
  auto IsClassFn =
      reinterpret_cast<CppAPIType::IsClass>(CppGetProcAddress("IsClass"));
  EXPECT_NE(IsClassFn, nullptr) << "Failed to obtain API function pointer";
  std::vector<Decl*> Decls;
  GetAllTopLevelDecls("namespace N {} class C{}; int I;", Decls);
  EXPECT_FALSE(IsClassFn(Decls[0]));
  EXPECT_TRUE(IsClassFn(Decls[1]));
  EXPECT_FALSE(IsClassFn(Decls[2]));
}

TEST(CPPINTEROP_TEST_MODE, DispatchAPI_CppGetProcAddress_Unimplemented) {
  testing::internal::CaptureStderr();
  void (*func_ptr)() = CppGetProcAddress("NonExistentFunction");
  std::string output = testing::internal::GetCapturedStderr();
  EXPECT_EQ(output,
            "[CppInterOp Dispatch] Failed to find API: NonExistentFunction May "
            "need to be ported to the symbol-address table in Dispatch.h\n");
  EXPECT_EQ(func_ptr, nullptr);
}

TEST(CPPINTEROP_TEST_MODE, DispatchAPI_CppGetProcAddress_Advanced) {
  std::string code = R"(
    int add(int x, int y) { return x + y; }
    int add(double x, double y) { return x + y; }
  )";

  std::vector<Decl*> Decls;
  GetAllTopLevelDecls(code, Decls);
  EXPECT_EQ(Decls.size(), 2);

  auto Add_int = clang::GlobalDecl(static_cast<clang::NamedDecl*>(Decls[0]));
  auto Add_double = clang::GlobalDecl(static_cast<clang::NamedDecl*>(Decls[1]));

  std::string mangled_add_int;
  std::string mangled_add_double;
  compat::maybeMangleDeclName(Add_int, mangled_add_int);
  compat::maybeMangleDeclName(Add_double, mangled_add_double);

  auto DemangleFn =
      reinterpret_cast<CppAPIType::Demangle>(CppGetProcAddress("Demangle"));
  auto GetQualifiedCompleteNameFn =
      reinterpret_cast<CppAPIType::GetQualifiedCompleteName>(
          CppGetProcAddress("GetQualifiedCompleteName"));

  std::string demangled_add_int = DemangleFn(mangled_add_int);
  std::string demangled_add_double = DemangleFn(mangled_add_double);

  EXPECT_NE(demangled_add_int.find(GetQualifiedCompleteNameFn(Decls[0])),
            std::string::npos);
  EXPECT_NE(demangled_add_double.find(GetQualifiedCompleteNameFn(Decls[1])),
            std::string::npos);
}

TEST(CPPINTEROP_TEST_MODE, DispatchAPI_LoadUnloadCycle) {
  Cpp::UnloadDispatchAPI(); // make sure we're no loaded already...
  EXPECT_FALSE(Cpp::LoadDispatchAPI("some/random/invalid/directory.so"));

  Cpp::UnloadDispatchAPI(); // should reset for next load to be successfull
  EXPECT_TRUE(Cpp::LoadDispatchAPI(CPPINTEROP_LIB_PATH));

  Cpp::UnloadDispatchAPI(); // should reset for next load to fail
  EXPECT_FALSE(Cpp::LoadDispatchAPI("some/other/random/invalid/directory.so"));

  // NOTE: minimize side-effects: reload assuming the static set is still
  // and other test may depend of this being loaded
  EXPECT_TRUE(Cpp::LoadDispatchAPI(CPPINTEROP_LIB_PATH));
}
// NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
