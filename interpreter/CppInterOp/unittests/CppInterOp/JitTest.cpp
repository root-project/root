#include "Utils.h"

#include "CppInterOp/CppInterOp.h"

#include "gtest/gtest.h"

using namespace TestUtils;

static int printf_jit(const char* format, ...) {
  llvm::errs() << "printf_jit called!\n";
  return 0;
}

TYPED_TEST(CppInterOpTest, JitTestInsertOrReplaceJitSymbol) {
#ifdef EMSCRIPTEN
  GTEST_SKIP() << "Test fails for Emscipten builds";
#endif
  if (llvm::sys::RunningOnValgrind())
    GTEST_SKIP() << "XFAIL due to Valgrind report";
#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";
  std::vector<Decl*> Decls;
  std::string code = R"(
    extern "C" int printf(const char*,...);
    )";

  GetAllTopLevelDecls(code, Decls);

  EXPECT_FALSE(Cpp::InsertOrReplaceJitSymbol("printf", (uint64_t)&printf_jit));

  testing::internal::CaptureStderr();
  Cpp::Process("printf(\"Blah\");");
  std::string cerrs = testing::internal::GetCapturedStderr();
  EXPECT_STREQ(cerrs.c_str(), "printf_jit called!\n");

  EXPECT_TRUE(
      Cpp::InsertOrReplaceJitSymbol("non_existent", (uint64_t)&printf_jit));
  EXPECT_TRUE(Cpp::InsertOrReplaceJitSymbol("non_existent", 0));
}

TYPED_TEST(CppInterOpTest, JitTestStreamRedirect) {
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";
  // printf and etc are fine here.
  // NOLINTBEGIN(cppcoreguidelines-pro-type-vararg)
  Cpp::BeginStdStreamCapture(Cpp::kStdOut);
  Cpp::BeginStdStreamCapture(Cpp::kStdErr);
  printf("StdOut is captured\n");
  fprintf(stderr, "StdErr is captured\n");
  std::cout << "Out captured"
            << "\n";
  std::cerr << "Err captured"
            << "\n";
  std::string CapturedStringErr = Cpp::EndStdStreamCapture();
  std::string CapturedStringOut = Cpp::EndStdStreamCapture();
  EXPECT_STREQ(CapturedStringOut.c_str(), "StdOut is captured\nOut captured\n");
  EXPECT_STREQ(CapturedStringErr.c_str(), "StdErr is captured\nErr captured\n");

  testing::internal::CaptureStdout();
  std::cout << "Out"
            << "\n";
  printf("StdOut\n");
  std::string outs = testing::internal::GetCapturedStdout();
  EXPECT_STREQ(outs.c_str(), "Out\nStdOut\n");

  testing::internal::CaptureStderr();
  std::cerr << "Err"
            << "\n";
  fprintf(stderr, "StdErr\n");
  std::string cerrs = testing::internal::GetCapturedStderr();
  EXPECT_STREQ(cerrs.c_str(), "Err\nStdErr\n");
  // NOLINTEND(cppcoreguidelines-pro-type-vararg)
}

TYPED_TEST(CppInterOpTest, JitTestStreamRedirectJIT) {
#ifdef EMSCRIPTEN
  GTEST_SKIP() << "Test fails for Emscipten builds";
#endif
  if (llvm::sys::RunningOnValgrind())
    GTEST_SKIP() << "XFAIL due to Valgrind report";
#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif
#ifdef CPPINTEROP_USE_CLING
  GTEST_SKIP() << "Test fails for cling builds";
#endif
  TestFixture::CreateInterpreter();
  Interp->process(R"(
    #include <stdio.h>
    printf("%s\n", "Hello World");
    fprintf(stderr, "%s\n", "Hello Err");
    fflush(nullptr);
  )");

  Cpp::BeginStdStreamCapture(Cpp::kStdOut);
  Cpp::BeginStdStreamCapture(Cpp::kStdErr);
  Interp->process(R"(
    #include <stdio.h>
    printf("%s\n", "Hello World");
    fprintf(stderr, "%s\n", "Hello Err");
    fflush(nullptr);
    )");
  std::string CapturedStringErr = Cpp::EndStdStreamCapture();
  std::string CapturedStringOut = Cpp::EndStdStreamCapture();

  EXPECT_STREQ(CapturedStringOut.c_str(), "Hello World\n");
  EXPECT_STREQ(CapturedStringErr.c_str(), "Hello Err\n");
}

