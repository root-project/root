#include "Utils.h"

#include "clang/Interpreter/CppInterOp.h"

#include "gtest/gtest.h"

using namespace TestUtils;

static int printf_jit(const char* format, ...) {
  llvm::errs() << "printf_jit called!\n";
  return 0;
}

TEST(JitTest, InsertOrReplaceJitSymbol) {
  if (llvm::sys::RunningOnValgrind())
    GTEST_SKIP() << "XFAIL due to Valgrind report";
#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif
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

TEST(Streams, StreamRedirect) {
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
