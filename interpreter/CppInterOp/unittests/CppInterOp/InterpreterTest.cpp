
#include "Utils.h"

#include "CppInterOp/CppInterOp.h"

#ifdef CPPINTEROP_USE_CLING
#include "cling/Interpreter/Interpreter.h"
#endif // CPPINTEROP_USE_CLING

#ifndef CPPINTEROP_USE_CLING
#include "clang/Interpreter/Interpreter.h"
#endif // CPPINTEROP_USE_REPL

#include "clang/Basic/Version.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/BuryPointer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <gmock/gmock.h>
#include "gtest/gtest.h"

#include <algorithm>
#include <csignal>
#include <cstdlib>

using ::testing::StartsWith;

TYPED_TEST(CPPINTEROP_TEST_MODE, Interpreter_Version) {
  EXPECT_THAT(Cpp::GetVersion(), StartsWith("CppInterOp version"));
}

TYPED_TEST(CPPINTEROP_TEST_MODE, Interpreter_BuildInfo) {
  // Pin the headline labels so a future template edit cannot silently
  // drop a field.
  std::string Info = Cpp::GetBuildInfo();
  EXPECT_THAT(Info, ::testing::HasSubstr("build type:"));
  EXPECT_THAT(Info, ::testing::HasSubstr("compiler:"));
  EXPECT_THAT(Info, ::testing::HasSubstr("llvm:"));
  EXPECT_THAT(Info, ::testing::HasSubstr("target:"));
  EXPECT_THAT(Info, ::testing::HasSubstr("cmake-line:"));
}

#ifndef LLVM_ENABLE_ASSERTIONS
TYPED_TEST(CPPINTEROP_TEST_MODE, DISABLED_Interpreter_DebugFlag) {
#else
TYPED_TEST(CPPINTEROP_TEST_MODE, Interpreter_DebugFlag) {
#endif // LLVM_ENABLE_ASSERTIONS
  TestFixture::CreateInterpreter();
  EXPECT_FALSE(Cpp::IsDebugOutputEnabled());
  std::string cerrs;
  testing::internal::CaptureStderr();
  Cpp::Process("int a = 12;");
  cerrs = testing::internal::GetCapturedStderr();
  EXPECT_STREQ(cerrs.c_str(), "");
  Cpp::EnableDebugOutput(true);
  EXPECT_TRUE(Cpp::IsDebugOutputEnabled());
  testing::internal::CaptureStderr();
  Cpp::Process("int b = 12;");
  cerrs = testing::internal::GetCapturedStderr();
  EXPECT_STRNE(cerrs.c_str(), "");

  Cpp::EnableDebugOutput(false);
  EXPECT_FALSE(Cpp::IsDebugOutputEnabled());
  testing::internal::CaptureStderr();
  Cpp::Process("int c = 12;");
  cerrs = testing::internal::GetCapturedStderr();
  EXPECT_STREQ(cerrs.c_str(), "");
}

TYPED_TEST(CPPINTEROP_TEST_MODE, Interpreter_Evaluate) {
#ifdef EMSCRIPTEN
  GTEST_SKIP() << "Test fails for Emscipten builds";
#endif
#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";
  //  EXPECT_TRUE(Cpp::Evaluate(I, "") == 0);
  //EXPECT_TRUE(Cpp::Evaluate(I, "__cplusplus;") == 201402);
  // Due to a deficiency in the clang-repl implementation to get the value we
  // always must omit the ;
  TestFixture::CreateInterpreter();
  EXPECT_TRUE(Cpp::Evaluate("__cplusplus", nullptr) == 201402);

  bool HadError;
  EXPECT_TRUE(Cpp::Evaluate("#error", &HadError) == (intptr_t)~0UL);
  EXPECT_TRUE(HadError);
  // for llvm < 19 this tests different overloads of
  // __clang_Interpreter_SetValueNoAlloc
  EXPECT_EQ(Cpp::Evaluate("int i = 11; ++i", &HadError), 12);
  EXPECT_FALSE(HadError);
  EXPECT_EQ(Cpp::Evaluate("double a = 12.; a", &HadError), 12.);
  EXPECT_FALSE(HadError);
  EXPECT_EQ(Cpp::Evaluate("float b = 13.; b", &HadError), 13.);
  EXPECT_FALSE(HadError);
  EXPECT_EQ(Cpp::Evaluate("long double c = 14.; c", &HadError), 14.);
  EXPECT_FALSE(HadError);
  EXPECT_EQ(Cpp::Evaluate("long double d = 15.; d", &HadError), 15.);
  EXPECT_FALSE(HadError);
  EXPECT_EQ(Cpp::Evaluate("unsigned long long e = 16; e", &HadError), 16);
  EXPECT_FALSE(HadError);
  EXPECT_NE(Cpp::Evaluate("struct S{} s; s", &HadError), (intptr_t)~0UL);
  EXPECT_FALSE(HadError);
}

// Regression (cppyy test12): no-Value-after-success used to abort in
// convertTo. A pure class decl is no-Value across all clang-repl
// versions; `int x = 5;` is bound on newer ones.
TYPED_TEST(CPPINTEROP_TEST_MODE, Interpreter_Evaluate_NonValueStatement) {
#ifdef EMSCRIPTEN
  GTEST_SKIP() << "Test fails for Emscipten builds";
#endif
#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";
  TestFixture::CreateInterpreter();

  bool HadError = false;
  EXPECT_EQ(Cpp::Evaluate("class EvalRegression_NoValue {};", &HadError),
            (intptr_t)~0UL);
  EXPECT_TRUE(HadError);
}

TYPED_TEST(CPPINTEROP_TEST_MODE, Interpreter_DeleteInterpreter) {
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";
  auto* I1 = TestFixture::CreateInterpreter();
  auto* I2 = TestFixture::CreateInterpreter();
  auto* I3 = TestFixture::CreateInterpreter();
  EXPECT_TRUE(I1 && I2 && I3) << "Failed to create interpreters";

  EXPECT_EQ(I3, Cpp::GetInterpreter()) << "I3 is not active";

  EXPECT_TRUE(Cpp::DeleteInterpreter(/*I=*/nullptr));
  EXPECT_EQ(I2, Cpp::GetInterpreter());

  auto* I4 = reinterpret_cast<void*>(static_cast<std::uintptr_t>(~0U));
  EXPECT_FALSE(Cpp::DeleteInterpreter(I4));

  EXPECT_TRUE(Cpp::DeleteInterpreter(I1));
  EXPECT_EQ(I2, Cpp::GetInterpreter()) << "I2 is not active";
}

TYPED_TEST(CPPINTEROP_TEST_MODE, Interpreter_StaticDtorsRunOnDelete) {
#ifdef EMSCRIPTEN
  GTEST_SKIP() << "Test fails for Emscripten builds";
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";

  auto* I = TestFixture::CreateInterpreter();
  ASSERT_NE(I, nullptr);

  // Host-side flag the JIT writes to; survives the interpreter deletion
  // that frees JIT memory. Reset explicitly for --gtest_repeat.
  static int HostFlag;
  HostFlag = 0;
  std::string Inject = "extern \"C\" void* dtor_sink = (void*)" +
                       std::to_string(reinterpret_cast<uintptr_t>(&HostFlag)) +
                       ";";
  Cpp::Declare(Inject.c_str(), I);
  Cpp::Declare(R"(
    struct DtorNotify { ~DtorNotify() { *static_cast<int*>(dtor_sink) = 1; } };
  )",
               I);
  Cpp::Process("static DtorNotify notify;");

  EXPECT_EQ(HostFlag, 0);
  Cpp::DeleteInterpreter(I);
  EXPECT_EQ(HostFlag, 1) << "JIT static dtor did not run on DeleteInterpreter";
}

// Mirrors clang/unittests/Interpreter/InterpreterTest.cpp's
// ShutdownDoesNotMaterializeAgainstDestroyedGlobals at the CppInterOp
// wrapper level: two interpreters each register a static with a
// non-trivial dtor, then std::exit drives the C++ static-dtor phase.
// On LLVM 23+ our InterpreterShutdown calls llvm_shutdown which fires
// ~InterpreterInfo -> ~Interpreter -> JIT deinit; the deinit lookup
// skips lazy materialization (Platform::lookupResolvedInitSymbols,
// llvm/llvm-project#196874) and the process exits cleanly.
//
// Skipped on older LLVM: InterpreterShutdown's llvm_shutdown call is
// gated out (we leak instead of risking the JIT cleanUp crash), and
// driving the chain manually from the test deadlocks rather than
// crashing cleanly through the compat layer. The crash contract is
// pinned upstream by clang's
// ShutdownDoesNotMaterializeAgainstDestroyedGlobals test.
TYPED_TEST(CPPINTEROP_TEST_MODE,
           Interpreter_ShutdownDoesNotMaterializeAgainstDestroyedGlobals) {
#ifdef EMSCRIPTEN
  GTEST_SKIP() << "Test fails for Emscripten builds";
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";
#if LLVM_VERSION_MAJOR <= 22
  GTEST_SKIP() << "Requires LLVM 23+ (lookupResolvedInitSymbols)";
#else
  EXPECT_EXIT(
      {
        auto* I1 = TestFixture::CreateInterpreter();
        auto* I2 = TestFixture::CreateInterpreter();
        Cpp::ActivateInterpreter(I1);
        Cpp::Process("struct S1 { S1() {} ~S1() {} }; static S1 s1;");
        Cpp::ActivateInterpreter(I2);
        Cpp::Process("struct S2 { S2() {} ~S2() {} }; static S2 s2;");
        std::exit(0);
      },
      ::testing::ExitedWithCode(0), "");
#endif
}

TYPED_TEST(CPPINTEROP_TEST_MODE, Interpreter_ActivateInterpreter) {
#ifdef EMSCRIPTEN_STATIC_LIBRARY
  GTEST_SKIP() << "Test fails for Emscipten static library build";
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";
  EXPECT_FALSE(Cpp::ActivateInterpreter(nullptr));
  auto* Cpp14 = TestFixture::CreateInterpreter({"-std=c++14"});
  auto* Cpp17 = TestFixture::CreateInterpreter({"-std=c++17"});
  auto* Cpp20 = TestFixture::CreateInterpreter({"-std=c++20"});

  EXPECT_TRUE(Cpp14 && Cpp17 && Cpp20);
  EXPECT_TRUE(Cpp::Evaluate("__cplusplus") == 202002L)
      << "Failed to activate C++20";

  auto* UntrackedI = reinterpret_cast<void*>(static_cast<std::uintptr_t>(~0U));
  EXPECT_FALSE(Cpp::ActivateInterpreter(UntrackedI));

  EXPECT_TRUE(Cpp::ActivateInterpreter(Cpp14));
  EXPECT_TRUE(Cpp::Evaluate("__cplusplus") == 201402L);

  Cpp::DeleteInterpreter(Cpp14);
  EXPECT_EQ(Cpp::GetInterpreter(), Cpp20);

  EXPECT_TRUE(Cpp::ActivateInterpreter(Cpp17));
  EXPECT_TRUE(Cpp::Evaluate("__cplusplus") == 201703L);
}

TYPED_TEST(CPPINTEROP_TEST_MODE, Interpreter_Process) {
#ifdef EMSCRIPTEN_STATIC_LIBRARY
  GTEST_SKIP() << "Test fails for Emscipten static library build";
#endif
#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";
  std::vector<const char*> interpreter_args = { "-include", "new", "-Xclang", "-iwithsysroot/include/compat" };
  TestFixture::CreateInterpreter(interpreter_args);
  EXPECT_TRUE(Cpp::Process("") == 0);
  EXPECT_TRUE(Cpp::Process("int a = 12;") == 0);
  EXPECT_FALSE(Cpp::Process("error_here;") == 0);
  // Linker/JIT error.
  EXPECT_FALSE(Cpp::Process("int f(); int res = f();") == 0);
}

TYPED_TEST(CPPINTEROP_TEST_MODE, Interpreter_DeclareSilent) {
#if CLANG_VERSION_MAJOR > 21
  GTEST_SKIP() << "Test crashes gtest for llvm 22 based build";
#endif
  TestFixture::CreateInterpreter();

  // Valid code with silent=true should succeed.
  EXPECT_EQ(0, Cpp::Declare("int x = 42;", /*silent=*/true));

  // Invalid code with silent=true should still report failure.
  EXPECT_NE(0, Cpp::Declare("invalid_syntax!!!;", /*silent=*/true));

  // The interpreter should remain usable after a silent failure.
  EXPECT_EQ(0, Cpp::Declare("int y = 123;", /*silent=*/false));
}

TYPED_TEST(CPPINTEROP_TEST_MODE, Interpreter_DeclareReportsParseErrors) {
#ifdef _WIN32
  // The non-silent parse-error case emits an `error: expected expression`
  // diagnostic to stderr, which MSBuild's check-cppinterop custom-build
  // rule scans and treats as a build failure regardless of the gtest
  // result.
  GTEST_SKIP()
      << "non-silent diagnostic trips MSBuild error scanning on Windows";
#endif
#if CLANG_VERSION_MAJOR > 21
  GTEST_SKIP() << "Test crashes gtest for llvm 22 based build";
#endif
  TestFixture::CreateInterpreter();

  // Valid input still returns 0.
  EXPECT_EQ(0, Cpp::Declare("int z = 7;", /*silent=*/false));

  // Parse error -> rc != 0 (Parse-recovered case the trap catches).
  EXPECT_NE(0, Cpp::Declare("int err = ;", /*silent=*/false));

  // Warning-only diagnostics must not trip the trap.
  EXPECT_EQ(0, Cpp::Declare("int __attribute__((deprecated)) v = 0;",
                            /*silent=*/false));
}

TYPED_TEST(CPPINTEROP_TEST_MODE, Interpreter_EmscriptenExceptionHandling) {
#ifndef EMSCRIPTEN
  GTEST_SKIP() << "This test is intended to check exception handling for Emscripten builds.";
#endif
    std::vector<const char*> Args = {
    "-std=c++20",
    "-v",
    "-fwasm-exceptions",
    "-mllvm","-wasm-enable-sjlj"
  };

  Cpp::CreateInterpreter(Args, {});

  const char* tryCatchCode = R"(
    try {
      throw 1;
    } catch (...) {
      0;
    }
  )";

  EXPECT_TRUE(Cpp::Process(tryCatchCode) == 0);
}

TYPED_TEST(CPPINTEROP_TEST_MODE, Interpreter_CreateInterpreter) {
  auto* I = TestFixture::CreateInterpreter();
  EXPECT_TRUE(I);
  // Check if the default standard is c++14

  Cpp::Declare("#if __cplusplus==201402L\n"
               "int cpp14() { return 2014; }\n"
               "#else\n"
               "void cppUnknown() {}\n"
               "#endif");
  EXPECT_TRUE(Cpp::GetNamed("cpp14"));
  EXPECT_FALSE(Cpp::GetNamed("cppUnknown"));

  I = TestFixture::CreateInterpreter({"-std=c++17"});
  Cpp::Declare("#if __cplusplus==201703L\n"
               "int cpp17() { return 2017; }\n"
               "#else\n"
               "void cppUnknown() {}\n"
               "#endif");
  EXPECT_TRUE(Cpp::GetNamed("cpp17"));
  EXPECT_FALSE(Cpp::GetNamed("cppUnknown"));
}

#ifndef CPPINTEROP_USE_CLING
#endif

#ifdef LLVM_BINARY_DIR
TYPED_TEST(CPPINTEROP_TEST_MODE, Interpreter_DetectResourceDir) {
#ifdef EMSCRIPTEN
  GTEST_SKIP() << "Test fails for Emscipten builds";
#endif
#else
TYPED_TEST(CPPINTEROP_TEST_MODE, Interpreter_DISABLED_DetectResourceDir) {
#endif // LLVM_BINARY_DIR
#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif
  TestFixture::CreateInterpreter();
  EXPECT_STRNE(Cpp::DetectResourceDir().c_str(), Cpp::GetResourceDir());
  llvm::SmallString<256> Clang(LLVM_BINARY_DIR);
  llvm::sys::path::append(Clang, "bin", "clang");

  if (!llvm::sys::fs::exists(llvm::Twine(Clang.str().str())))
    GTEST_SKIP() << "Test not run (Clang binary does not exist)";

  std::string DetectedPath = Cpp::DetectResourceDir(Clang.str().str().c_str());
  llvm::SmallString<256> absPath(Cpp::GetResourceDir());
  EXPECT_TRUE(!llvm::sys::fs::make_absolute(absPath));
  llvm::SmallString<256> realPath;
  EXPECT_TRUE(!llvm::sys::fs::real_path(absPath, realPath));
  EXPECT_STREQ(DetectedPath.c_str(), realPath.str().str().c_str());
}

TYPED_TEST(CPPINTEROP_TEST_MODE, Interpreter_DetectSystemCompilerIncludePaths) {
#ifdef EMSCRIPTEN
  GTEST_SKIP() << "Test fails for Emscipten builds";
#endif
#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif
  std::vector<std::string> includes;
  Cpp::DetectSystemCompilerIncludePaths(includes);
  EXPECT_FALSE(includes.empty());
}

TYPED_TEST(CPPINTEROP_TEST_MODE, Interpreter_IncludePaths) {
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";
  std::vector<std::string> includes;
  Cpp::GetIncludePaths(includes);
  EXPECT_FALSE(includes.empty());

  size_t len = includes.size();
  includes.clear();
  Cpp::GetIncludePaths(includes, true, false);
  EXPECT_FALSE(includes.empty());
  EXPECT_TRUE(includes.size() >= len);

  len = includes.size();
  includes.clear();
  Cpp::GetIncludePaths(includes, true, true);
  EXPECT_FALSE(includes.empty());
  EXPECT_TRUE(includes.size() >= len);

  len = includes.size();
  Cpp::AddIncludePath("/non/existent/");
  Cpp::GetIncludePaths(includes);
  EXPECT_NE(std::find(includes.begin(), includes.end(), "/non/existent/"),
             std::end(includes));
}

TYPED_TEST(CPPINTEROP_TEST_MODE, Interpreter_CodeCompletion) {
#if CLANG_VERSION_MAJOR == 20 && defined(CPPINTEROP_USE_CLING) &&              \
    defined(_WIN32)
  GTEST_SKIP() << "Test fails with Cling on Windows";
#endif
  TestFixture::CreateInterpreter();
  std::vector<std::string> cc;
  Cpp::Declare("int foo = 12;");
  Cpp::CodeComplete(cc, "f", 1, 2);
  // We check only for 'float' and 'foo', because they
  // must be present in the result. Other hints may appear
  // there, depending on the implementation, but these two
  // are required to say that the test is working.
  size_t cnt = 0;
  for (auto& r : cc)
    if (r == "float" || r == "foo")
      cnt++;
  EXPECT_EQ(2U, cnt); // float and foo
}

TYPED_TEST(CPPINTEROP_TEST_MODE, Interpreter_GetLanguageCpp) {
  // Default interpreter (C++14)
  TestFixture::CreateInterpreter();
  EXPECT_EQ(Cpp::GetLanguage(nullptr), Cpp::InterpreterLanguage::CPlusPlus);
}

TYPED_TEST(CPPINTEROP_TEST_MODE, Interpreter_GetLanguageStandardCpp) {
  // Other C++ standards
  TestFixture::CreateInterpreter({"-std=c++14"});
  EXPECT_EQ(Cpp::GetLanguageStandard(nullptr),
            Cpp::InterpreterLanguageStandard::cxx14);

  TestFixture::CreateInterpreter({"-std=c++17"});
  EXPECT_EQ(Cpp::GetLanguageStandard(nullptr),
            Cpp::InterpreterLanguageStandard::cxx17);

  TestFixture::CreateInterpreter({"-std=c++20"});
  EXPECT_EQ(Cpp::GetLanguageStandard(nullptr),
            Cpp::InterpreterLanguageStandard::cxx20);

  TestFixture::CreateInterpreter({"-std=c++23"});
  EXPECT_EQ(Cpp::GetLanguageStandard(nullptr),
            Cpp::InterpreterLanguageStandard::cxx23);
}

TYPED_TEST(CPPINTEROP_TEST_MODE, Interpreter_GetLanguageC) {
  TestFixture::CreateInterpreter({"-xc", "-std=c99"});
  EXPECT_EQ(Cpp::GetLanguage(nullptr), Cpp::InterpreterLanguage::C);
  EXPECT_EQ(Cpp::GetLanguageStandard(nullptr),
            Cpp::InterpreterLanguageStandard::c99);
}
TYPED_TEST(CPPINTEROP_TEST_MODE, Interpreter_GetLanguageStandardC) {
  TestFixture::CreateInterpreter({"-xc", "-std=c89"});
  EXPECT_EQ(Cpp::GetLanguageStandard(nullptr),
            Cpp::InterpreterLanguageStandard::c89);

  TestFixture::CreateInterpreter({"-xc", "-std=c11"});
  EXPECT_EQ(Cpp::GetLanguageStandard(nullptr),
            Cpp::InterpreterLanguageStandard::c11);

  TestFixture::CreateInterpreter({"-xc", "-std=c17"});
  EXPECT_EQ(Cpp::GetLanguageStandard(nullptr),
            Cpp::InterpreterLanguageStandard::c17);

  TestFixture::CreateInterpreter({"-xc", "-std=c23"});
  EXPECT_EQ(Cpp::GetLanguageStandard(nullptr),
            Cpp::InterpreterLanguageStandard::c23);
}

TYPED_TEST(CPPINTEROP_TEST_MODE, Interpreter_GetLanguageStandardGNU) {
  TestFixture::CreateInterpreter({"-std=gnu++14"});
  EXPECT_EQ(Cpp::GetLanguageStandard(nullptr),
            Cpp::InterpreterLanguageStandard::gnucxx14);

  TestFixture::CreateInterpreter({"-std=gnu++17"});
  EXPECT_EQ(Cpp::GetLanguageStandard(nullptr),
            Cpp::InterpreterLanguageStandard::gnucxx17);
}

TYPED_TEST(CPPINTEROP_TEST_MODE, Interpreter_ExternalInterpreter) {


#ifndef CPPINTEROP_USE_CLING
  llvm::ExitOnError ExitOnErr;
  clang::IncrementalCompilerBuilder CB;
  CB.SetCompilerArgs({"-std=c++20"});

  // Create the incremental compiler instance.
  std::unique_ptr<clang::CompilerInstance> CI;
  CI = ExitOnErr(CB.CreateCpp());

  // Create the interpreter instance.
  std::unique_ptr<clang::Interpreter> I =
      ExitOnErr(clang::Interpreter::create(std::move(CI)));
  auto ExtInterp = I.get();
#endif // CPPINTEROP_USE_REPL

#ifdef CPPINTEROP_USE_CLING
    std::string MainExecutableName = sys::fs::getMainExecutable(nullptr, nullptr);
    llvm::SmallString<128> P(LLVM_BINARY_DIR);
    llvm::sys::path::append(P, CLANG_INSTALL_LIBDIR_BASENAME, "clang",
                            CLANG_VERSION_MAJOR_STRING);
    std::string ResourceDir = std::string(P.str());
    std::vector<const char *> ClingArgv = {"-resource-dir", ResourceDir.c_str(),
                                           "-std=c++14"};
    ClingArgv.insert(ClingArgv.begin(), MainExecutableName.c_str());
    auto *ExtInterp = new compat::Interpreter(ClingArgv.size(), &ClingArgv[0]);
#endif

  EXPECT_NE(ExtInterp, nullptr);

#if !defined(NDEBUG) && GTEST_HAS_DEATH_TEST
#ifndef _WIN32 // Windows seems to fail to die...
    EXPECT_DEATH(Cpp::UseExternalInterpreter(ExtInterp), "sInterpreter already in use!");
#endif // _WIN32
#endif
  EXPECT_TRUE(Cpp::GetInterpreter()) << "External Interpreter not set";

#ifndef CPPINTEROP_USE_CLING
  // Skip Interpreter destruction: the teardown-order bug tracked in
  // CppInterOp#887 makes it unsafe. BuryPointer is used instead of
  // release() so LSan sees the graph as reachable rather than leaked.
  llvm::BuryPointer(std::move(I));
#endif

#ifdef CPPINTEROP_USE_CLING
  delete ExtInterp;
#endif
}

// Verify the basic crash banner and Active Interpreter reporting
#ifdef GTEST_HAS_DEATH_TEST
TYPED_TEST(CPPINTEROP_TEST_MODE, SignalHandler_BasicBanner) {
  // Ensure a clean registry for each JIT configuration

  // FIXME: Uncomment after resolving compiler-research/CppInterOp#887

  // while (Cpp::GetInterpreter())
  //   Cpp::DeleteInterpreter(/*I=*/nullptr);

  // EXPECT_FALSE(Cpp::GetInterpreter()) << "Failed to delete all interpreters";

  // // Create an interpreter (this calls RegisterInterpreter internally)
  // TInterp_t I = TestFixture::CreateInterpreter();
  // ASSERT_NE(I, nullptr);

  // We expect the banner to appear in stderr when the process dies
  std::string ExpectedMsg = "CppInterOp CRASH DETECTED";
#ifdef _WIN32
  // FIXME: Windows says 'Actual msg:' without maybe capturing the message.
  ExpectedMsg = "";
#endif //_WIN32

  EXPECT_DEATH(
      {
        // Trigger a synchronous signal
        raise(SIGABRT);
      },
      ExpectedMsg);
}
#endif // GTEST_HAS_DEATH_TEST

// Verify that the handler correctly lists multiple interpreters
#ifdef GTEST_HAS_DEATH_TEST
TYPED_TEST(CPPINTEROP_TEST_MODE, SignalHandler_MultipleInterpreters) {
  ASSERT_NE(TestFixture::CreateInterpreter(), nullptr);
  ASSERT_NE(TestFixture::CreateInterpreter(), nullptr);

  // The handler iterates through the deque and prints the pointers

  // We check for the "Active Interpreters:" header and the list format
  std::string ExpectedMsg = "Active Interpreters:.*- 0x";
#ifdef _WIN32
  // FIXME: Windows says 'Actual msg:' without maybe capturing the message.
  ExpectedMsg = "";
#endif //_WIN32

  EXPECT_DEATH({ raise(SIGSEGV); }, ExpectedMsg);
}
#endif // GTEST_HAS_DEATH_TEST

#ifndef EMSCRIPTEN
TYPED_TEST(CPPINTEROP_TEST_MODE, Interpreter_WrapperCacheIsPerInterpreter) {
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";

  // Create first interpreter: add(a,b) returns a + b.
  auto* I1 = TestFixture::CreateInterpreter();
  ASSERT_NE(I1, nullptr);
  Cpp::ActivateInterpreter(I1);
  Cpp::Declare("int add(int a, int b) { return a + b; }");
  auto* AddDecl1 = Cpp::GetNamed("add");
  ASSERT_NE(AddDecl1, nullptr);

  Cpp::Declare("#include <new>"); // Needed by JitCall
  auto JC1 = Cpp::MakeFunctionCallable(AddDecl1);
  ASSERT_TRUE(JC1.isValid());

  int a1 = 3, b1 = 4, r1 = 0;
  void* args1[] = {&a1, &b1};
  JC1.Invoke(&r1, {args1, 2});
  EXPECT_EQ(r1, 7);

  // Create second interpreter: add(a,b) returns a * b.
  auto* I2 = TestFixture::CreateInterpreter();
  ASSERT_NE(I2, nullptr);
  Cpp::ActivateInterpreter(I2);
  Cpp::Declare("int add(int a, int b) { return a * b; }");
  auto* AddDecl2 = Cpp::GetNamed("add");
  ASSERT_NE(AddDecl2, nullptr);

  Cpp::Declare("#include <new>"); // Needed by JitCall
  auto JC2 = Cpp::MakeFunctionCallable(AddDecl2);
  ASSERT_TRUE(JC2.isValid());

  // Delete the first interpreter.
  EXPECT_TRUE(Cpp::DeleteInterpreter(I1));

  // The second interpreter's wrapper must still work and must use its own
  // implementation (multiply), not a stale cache entry from deleted I1.
  int a2 = 3, b2 = 4, r2 = 0;
  void* args2[] = {&a2, &b2};
  JC2.Invoke(&r2, {args2, 2});
  EXPECT_EQ(r2, 12) << "Wrapper should use I2's implementation (multiply), "
                       "not a stale cache from deleted I1";
}

TYPED_TEST(CPPINTEROP_TEST_MODE, Interpreter_DLMIsPerInterpreter) {
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";
#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows: DLM symbol-table walk crashes "
                  "(RVA 0x0 in export table).";
#endif

#ifdef __APPLE__
  static constexpr const char* kSym = "_ret_zero";
#else
  static constexpr const char* kSym = "ret_zero";
#endif

  std::string BinaryPath =
      llvm::sys::fs::getMainExecutable(/*Argv0=*/nullptr, /*MainAddr=*/nullptr);
  ASSERT_FALSE(BinaryPath.empty());
  llvm::StringRef BinDir = llvm::sys::path::parent_path(BinaryPath);

  auto* I1 = TestFixture::CreateInterpreter();
  ASSERT_NE(I1, nullptr);
  Cpp::ActivateInterpreter(I1);
  Cpp::AddSearchPath(BinDir.str().c_str());

  std::string R1 = Cpp::SearchLibrariesForSymbol(kSym, /*system_search=*/false);
  if (R1.empty())
    GTEST_SKIP() << "TestSharedLib not found — cannot test DLM isolation";

  // With a single function-local-static DLM, the path added on I1
  // would reach I2 and the lookup below would succeed; the per-
  // interpreter DLM means I2 starts with a fresh path set.
  auto* I2 = TestFixture::CreateInterpreter();
  ASSERT_NE(I2, nullptr);
  Cpp::ActivateInterpreter(I2);

  std::string R2 = Cpp::SearchLibrariesForSymbol(kSym, /*system_search=*/false);
  EXPECT_TRUE(R2.empty()) << "I2 must not see search paths added on I1; found: "
                          << R2;
}
#endif // !EMSCRIPTEN
