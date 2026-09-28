#include "Utils.h"
#include "CppInterOp/CppInterOp.h"

#include "../../lib/CppInterOp/Compatibility.h"

#include "clang/Basic/Version.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include "gtest/gtest.h"

#include <memory>

#ifndef _WIN32
#include <dlfcn.h>
#endif

// This function isn't referenced outside its translation unit, but it
// can't use the "static" keyword because its address is used for
// GetMainExecutable (since some platforms don't support taking the
// address of main, and some platforms can't implement GetMainExecutable
// without being given the address of a function in the main executable).
std::string GetExecutablePath(const char* Argv0) {
  // This just needs to be some symbol in the binary; C++ doesn't
  // allow taking the address of ::main however.
  void* MainAddr = (void*)intptr_t(GetExecutablePath);
  return llvm::sys::fs::getMainExecutable(Argv0, MainAddr);
}

// Resolve TestSharedLib by path the way DynamicLibraryManager_SharedWeakGlobals
// does: SearchLibrariesForSymbol may not run twice in one process, so probe
// beside the test binary (cmake) then the working directory (bazel). Empty if
// not found.
static std::string ResolveTestSharedLibPath() {
  std::string BinaryPath = GetExecutablePath(/*Argv0=*/nullptr);
  llvm::StringRef Dir = llvm::sys::path::parent_path(BinaryPath);
  for (const char* Candidate : {"libTestSharedLib.so", "TestSharedLib.so"}) {
    llvm::SmallString<256> P(Dir);
    llvm::sys::path::append(P, Candidate);
    if (llvm::sys::fs::exists(P))
      return P.str().str();
    if (llvm::sys::fs::exists(Candidate))
      return Candidate;
  }
  return {};
}

TYPED_TEST(CPPINTEROP_TEST_MODE, DynamicLibraryManager_Sanity) {
#ifdef EMSCRIPTEN
  GTEST_SKIP() << "Test fails for Emscipten builds";
#endif

#if CLANG_VERSION_MAJOR == 20 && defined(CPPINTEROP_USE_CLING) &&              \
    defined(_WIN32)
  GTEST_SKIP() << "Test fails with Cling on Windows";
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";

  EXPECT_TRUE(TestFixture::CreateInterpreter());
  EXPECT_FALSE(Cpp::GetFunctionAddress("ret_zero"));

  std::string BinaryPath = GetExecutablePath(/*Argv0=*/nullptr);
  llvm::StringRef Dir = llvm::sys::path::parent_path(BinaryPath);
  Cpp::AddSearchPath(Dir.str().c_str());

  // FIXME: dlsym on mach-o takes the C-level name, however, the macho-o format
  // adds an additional underscore (_) prefix to the lowered names. Figure out
  // how to harmonize that API.
#ifdef __APPLE__
  std::string PathToTestSharedLib =
      Cpp::SearchLibrariesForSymbol("_ret_zero", /*system_search=*/false);
#else
  std::string PathToTestSharedLib =
      Cpp::SearchLibrariesForSymbol("ret_zero", /*system_search=*/false);
#endif // __APPLE__

  EXPECT_STRNE("", PathToTestSharedLib.c_str())
      << "Cannot find: '" << PathToTestSharedLib << "' in '" << Dir.str()
      << "'";

  EXPECT_TRUE(Cpp::LoadLibrary(PathToTestSharedLib.c_str()));
  // Force ExecutionEngine to be created.
  Cpp::Process("");
  // FIXME: Conda returns false to run this code on osx.
#ifndef __APPLE__
  EXPECT_TRUE(Cpp::GetFunctionAddress("ret_zero"));
#endif //__APPLE__

  Cpp::UnloadLibrary("TestSharedLib");
  // We have no reliable way to check if it was unloaded because posix does not
  // require the library to be actually unloaded but just the handle to be
  // invalidated...
  // EXPECT_FALSE(Cpp::GetFunctionAddress("ret_zero"));
}

// A failed dlopen must report the dlerror() text on stderr.
TYPED_TEST(CPPINTEROP_TEST_MODE, DynamicLibraryManager_LoadFailureReason) {
#if defined(EMSCRIPTEN) || defined(_WIN32)
  GTEST_SKIP() << "Test checks dlerror-style messages";
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";

  EXPECT_TRUE(TestFixture::CreateInterpreter());

  // A file that exists but is not a shared library makes dlopen fail.
  int FD = -1;
  llvm::SmallString<256> BadLib;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile("bad-lib", "so", FD, BadLib));
  {
    llvm::raw_fd_ostream OS(FD, /*shouldClose=*/true);
    OS << "not a shared library";
  }

  testing::internal::CaptureStderr();
  EXPECT_FALSE(Cpp::LoadLibrary(BadLib.c_str(), /*lookup=*/false));
  std::string Err = testing::internal::GetCapturedStderr();
  // dlerror() names the failing file.
  EXPECT_NE(Err.find(llvm::sys::path::filename(BadLib).str()),
            std::string::npos)
      << "stderr was: '" << Err << "'";

  // With an out-parameter the reason goes there and not to stderr.
  std::string Reason = "stale";
  testing::internal::CaptureStderr();
  EXPECT_FALSE(Cpp::LoadLibrary(BadLib.c_str(), /*lookup=*/false, &Reason));
#ifdef CPPINTEROP_USE_CLING
  testing::internal::GetCapturedStderr(); // cling still prints its own text
#else
  EXPECT_EQ(testing::internal::GetCapturedStderr(), "");
#endif
  EXPECT_NE(Reason.find(llvm::sys::path::filename(BadLib).str()),
            std::string::npos)
      << "reason was: '" << Reason << "'";

  // A failed lookup reports that too.
  EXPECT_FALSE(
      Cpp::LoadLibrary("no-such-cppinterop-lib", /*lookup=*/true, &Reason));
  EXPECT_NE(Reason.find("no-such-cppinterop-lib"), std::string::npos)
      << "reason was: '" << Reason << "'";

  // Success clears a stale reason.
  std::string BinaryPath = GetExecutablePath(/*Argv0=*/nullptr);
  Cpp::AddSearchPath(llvm::sys::path::parent_path(BinaryPath).str().c_str());
  Reason = "stale";
  EXPECT_TRUE(Cpp::LoadLibrary("TestSharedLib", /*lookup=*/true, &Reason))
      << "reason was: '" << Reason << "'";
  EXPECT_EQ(Reason, "");

  EXPECT_FALSE(llvm::sys::fs::remove(BadLib));
}

TYPED_TEST(CPPINTEROP_TEST_MODE, DynamicLibraryManager_BasicSymbolLookup) {
#ifndef EMSCRIPTEN
  GTEST_SKIP() << "This test is only intended for Emscripten builds.";
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";

  ASSERT_TRUE(TestFixture::CreateInterpreter());
  EXPECT_FALSE(Cpp::GetFunctionAddress("ret_zero"));

  // Load the library manually. Use known preload path (MEMFS path)
  const char* wasmLibPath = "libTestSharedLib.so";  // Preloaded path in MEMFS
  EXPECT_TRUE(Cpp::LoadLibrary(wasmLibPath, false));

  Cpp::Process("");

  void* Addr = Cpp::GetFunctionAddress("ret_zero");
  EXPECT_NE(Addr, nullptr) << "Symbol 'ret_zero' not found after dlopen.";

  using RetZeroFn = int (*)();
  auto Fn = reinterpret_cast<RetZeroFn>(Addr);
  EXPECT_EQ(Fn(), 0);
}

// Weak globals a loaded library already defines -- header-defined singletons:
// a function-local static in an inline function and a C++17 inline static
// data member -- must be bound by jitted code rather than materialized as
// duplicates (compat::bindProcessWeakGlobals). Otherwise the library and the
// JIT each hold their own instance of what the program means to be one
// singleton, and teardown destroys it twice.
TYPED_TEST(CPPINTEROP_TEST_MODE, DynamicLibraryManager_SharedWeakGlobals) {
#ifdef EMSCRIPTEN
  GTEST_SKIP() << "Test not intended for Emscripten builds";
#endif
#ifdef CPPINTEROP_USE_CLING
  GTEST_SKIP() << "bindProcessWeakGlobals is wired into the clang-repl "
                  "CppInternal::Interpreter path, not cling's interpreter";
#endif
#if defined(_WIN32) || defined(__APPLE__)
  GTEST_SKIP() << "the dlsym-based weak-global binding targets ELF";
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "the dlsym-based weak-global binding is in-process only";

  ASSERT_TRUE(TestFixture::CreateInterpreter());

  // Resolve the library by path (SearchLibrariesForSymbol cannot run twice in
  // one process, and DynamicLibraryManager_Sanity already used it): beside
  // the test binary (cmake), or in the working directory (bazel).
  std::string BinaryPath = GetExecutablePath(/*Argv0=*/nullptr);
  llvm::StringRef Dir = llvm::sys::path::parent_path(BinaryPath);
  std::string PathToTestSharedLib;
  for (const char* Candidate : {"libTestSharedLib.so", "TestSharedLib.so"}) {
    llvm::SmallString<256> P(Dir);
    llvm::sys::path::append(P, Candidate);
    if (llvm::sys::fs::exists(P)) {
      PathToTestSharedLib = P.str().str();
      break;
    }
    if (llvm::sys::fs::exists(Candidate)) {
      PathToTestSharedLib = Candidate;
      break;
    }
  }
  ASSERT_STRNE("", PathToTestSharedLib.c_str());
  ASSERT_TRUE(Cpp::LoadLibrary(PathToTestSharedLib.c_str()));
  Cpp::Process(""); // force the execution engine into existence

  // Repeat TestSharedLib.h's singleton definitions in the JIT (the mangled
  // names must match the library's) and take each side's addresses.
  ASSERT_FALSE(Cpp::Process(R"(
    struct SingletonFixture {
      static SingletonFixture& get() {
        static SingletonFixture instance;
        return instance;
      }
      static inline int s_inline_member = 0;
      int value = 0;
    };
    extern "C" void* singleton_probe_jit_meyers_addr() {
      return &SingletonFixture::get();
    }
    extern "C" void* singleton_probe_jit_member_addr() {
      return &SingletonFixture::s_inline_member;
    }
    extern "C" void singleton_probe_jit_set(int v) {
      SingletonFixture::get().value = v;
      SingletonFixture::s_inline_member = v + 1;
    }
  )"));

  auto GetFn = [](const char* Name) {
    void* A = Cpp::GetFunctionAddress(Name);
    EXPECT_NE(A, nullptr) << Name;
    return A;
  };
  // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
  auto* JitMeyersAddr =
      reinterpret_cast<void* (*)()>(GetFn("singleton_probe_jit_meyers_addr"));
  auto* JitMemberAddr =
      reinterpret_cast<void* (*)()>(GetFn("singleton_probe_jit_member_addr"));
  auto* JitSet =
      reinterpret_cast<void (*)(int)>(GetFn("singleton_probe_jit_set"));
  auto* AotMeyersAddr =
      reinterpret_cast<void* (*)()>(GetFn("singleton_fixture_meyers_addr"));
  auto* AotMeyersValue =
      reinterpret_cast<int (*)()>(GetFn("singleton_fixture_meyers_value"));
  auto* AotMemberAddr =
      reinterpret_cast<void* (*)()>(GetFn("singleton_fixture_member_addr"));
  auto* AotMemberValue =
      reinterpret_cast<int (*)()>(GetFn("singleton_fixture_member_value"));
  // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)

  // One instance, not two: both flavors' addresses agree across the boundary.
  EXPECT_EQ(JitMeyersAddr(), AotMeyersAddr());
  EXPECT_EQ(JitMemberAddr(), AotMemberAddr());

  // And mutations through the jitted side are visible to the library.
  JitSet(41);
  EXPECT_EQ(AotMeyersValue(), 41);
  EXPECT_EQ(AotMemberValue(), 42);
}

// A dynamically initialized header singleton: its function-local static carries
// a _ZGV guard beside the data. This pins bindProcessWeakGlobals's
// guard-pairing branch -- the guard is demoted together with the data (and only
// because the process exports both), so the jitted copy shares the library's
// storage AND its already-run initialization instead of re-initializing a
// private duplicate.
TYPED_TEST(CPPINTEROP_TEST_MODE,
           DynamicLibraryManager_SharedDynamicInitSingleton) {
#ifdef EMSCRIPTEN
  GTEST_SKIP() << "Test not intended for Emscripten builds";
#endif
#ifdef CPPINTEROP_USE_CLING
  GTEST_SKIP() << "bindProcessWeakGlobals is wired into the clang-repl "
                  "CppInternal::Interpreter path, not cling's interpreter";
#endif
#if defined(_WIN32) || defined(__APPLE__)
  GTEST_SKIP() << "the dlsym-based weak-global binding targets ELF";
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "the dlsym-based weak-global binding is in-process only";

  ASSERT_TRUE(TestFixture::CreateInterpreter());
  std::string PathToTestSharedLib = ResolveTestSharedLibPath();
  ASSERT_STRNE("", PathToTestSharedLib.c_str());
  ASSERT_TRUE(Cpp::LoadLibrary(PathToTestSharedLib.c_str()));
  Cpp::Process(""); // force the execution engine into existence

  auto GetFn = [](const char* Name) {
    void* A = Cpp::GetFunctionAddress(Name);
    EXPECT_NE(A, nullptr) << Name;
    return A;
  };
  // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
  auto* AotAddr =
      reinterpret_cast<void* (*)()>(GetFn("dynamic_singleton_addr"));
  auto* AotValue =
      reinterpret_cast<int (*)()>(GetFn("dynamic_singleton_value"));

  // Initialize the library's copy first; its guard variable is now set.
  EXPECT_EQ(AotValue(), 7);

  // Repeat the dynamically initialized singleton in the JIT (matching mangled
  // names). The out-of-line constructor makes clang emit a guarded init.
  ASSERT_FALSE(Cpp::Process(R"(
    struct DynamicInitSingleton {
      int value;
      DynamicInitSingleton();
      static DynamicInitSingleton& get() {
        static DynamicInitSingleton instance;
        return instance;
      }
    };
    extern "C" void* dyn_jit_addr() { return &DynamicInitSingleton::get(); }
    extern "C" int dyn_jit_value() { return DynamicInitSingleton::get().value; }
    extern "C" void dyn_jit_set(int v) { DynamicInitSingleton::get().value = v; }
  )"));

  auto* JitAddr = reinterpret_cast<void* (*)()>(GetFn("dyn_jit_addr"));
  auto* JitValue = reinterpret_cast<int (*)()>(GetFn("dyn_jit_value"));
  auto* JitSet = reinterpret_cast<void (*)(int)>(GetFn("dyn_jit_set"));
  // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)

  // One instance: the addresses agree across the boundary.
  EXPECT_EQ(JitAddr(), AotAddr());
  // Shared guard: the jitted side observes the library's initialization rather
  // than re-running the constructor on a private copy.
  EXPECT_EQ(JitValue(), 7);
  // Shared data: a jitted mutation is visible through the library accessor.
  JitSet(31);
  EXPECT_EQ(AotValue(), 31);
}

// A weak global the jitted module marks __attribute__((used)) lands in
// @llvm.compiler.used. Demoting it forces bindProcessWeakGlobals to rewrite
// that list (a declaration there fails the IR verifier); a second, non-demoted
// used global keeps the list non-empty, so it is rebuilt rather than erased. A
// clean Execute proves the rewrite ran; the shared address proves the demotion.
TYPED_TEST(CPPINTEROP_TEST_MODE, DynamicLibraryManager_SharedUsedWeakGlobal) {
#ifdef EMSCRIPTEN
  GTEST_SKIP() << "Test not intended for Emscripten builds";
#endif
#ifdef CPPINTEROP_USE_CLING
  GTEST_SKIP() << "bindProcessWeakGlobals is wired into the clang-repl "
                  "CppInternal::Interpreter path, not cling's interpreter";
#endif
#if defined(_WIN32) || defined(__APPLE__)
  GTEST_SKIP() << "the dlsym-based weak-global binding targets ELF";
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "the dlsym-based weak-global binding is in-process only";

  ASSERT_TRUE(TestFixture::CreateInterpreter());
  std::string PathToTestSharedLib = ResolveTestSharedLibPath();
  ASSERT_STRNE("", PathToTestSharedLib.c_str());
  ASSERT_TRUE(Cpp::LoadLibrary(PathToTestSharedLib.c_str()));
  Cpp::Process(""); // force the execution engine into existence

  // g_used_singleton is mutable and process-defined (demoted); g_used_kept is
  // const (excluded) and survives, forcing the used-list to be rebuilt.
  ASSERT_FALSE(Cpp::Process(R"(
    inline int g_used_singleton __attribute__((used)) = 0;
    inline const int g_used_kept __attribute__((used)) = 5;
    extern "C" void* used_jit_addr() { return &g_used_singleton; }
    extern "C" void used_jit_set(int v) { g_used_singleton = v; }
    extern "C" int used_kept_value() { return g_used_kept; }
  )"));

  auto GetFn = [](const char* Name) {
    void* A = Cpp::GetFunctionAddress(Name);
    EXPECT_NE(A, nullptr) << Name;
    return A;
  };
  // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
  auto* JitAddr = reinterpret_cast<void* (*)()>(GetFn("used_jit_addr"));
  auto* JitSet = reinterpret_cast<void (*)(int)>(GetFn("used_jit_set"));
  auto* KeptValue = reinterpret_cast<int (*)()>(GetFn("used_kept_value"));
  auto* AotAddr = reinterpret_cast<void* (*)()>(GetFn("g_used_singleton_addr"));
  auto* AotValue = reinterpret_cast<int (*)()>(GetFn("g_used_singleton_value"));
  // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)

  // The demoted used global binds the library's copy.
  EXPECT_EQ(JitAddr(), AotAddr());
  JitSet(17);
  EXPECT_EQ(AotValue(), 17);
  // The non-demoted (const) used global was kept and still reads correctly.
  EXPECT_EQ(KeptValue(), 5);
}

// A header singleton the process does NOT define: dlsym misses, so the pass
// leaves the definition alone and the JIT keeps its own working private copy.
// Pins the dlsym-miss branch (candidate left untouched).
TYPED_TEST(CPPINTEROP_TEST_MODE, DynamicLibraryManager_JitOnlyWeakGlobalKept) {
#ifdef EMSCRIPTEN
  GTEST_SKIP() << "Test not intended for Emscripten builds";
#endif
#ifdef CPPINTEROP_USE_CLING
  GTEST_SKIP() << "bindProcessWeakGlobals is wired into the clang-repl "
                  "CppInternal::Interpreter path, not cling's interpreter";
#endif
#if defined(_WIN32) || defined(__APPLE__)
  GTEST_SKIP() << "the dlsym-based weak-global binding targets ELF";
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "the dlsym-based weak-global binding is in-process only";

  ASSERT_TRUE(TestFixture::CreateInterpreter());
  Cpp::Process(""); // force the execution engine into existence

  ASSERT_FALSE(Cpp::Process(R"(
    struct JitOnlySingleton {
      static JitOnlySingleton& get() { static JitOnlySingleton i; return i; }
      int value = 0;
    };
    extern "C" void* jitonly_addr() { return &JitOnlySingleton::get(); }
    extern "C" void jitonly_set(int v) { JitOnlySingleton::get().value = v; }
    extern "C" int jitonly_get() { return JitOnlySingleton::get().value; }
  )"));

  auto GetFn = [](const char* Name) {
    void* A = Cpp::GetFunctionAddress(Name);
    EXPECT_NE(A, nullptr) << Name;
    return A;
  };
  // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
  auto* Addr = reinterpret_cast<void* (*)()>(GetFn("jitonly_addr"));
  auto* Set = reinterpret_cast<void (*)(int)>(GetFn("jitonly_set"));
  auto* Get = reinterpret_cast<int (*)()>(GetFn("jitonly_get"));
  // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)

  // The kept copy is real, mutable storage.
  EXPECT_NE(Addr(), nullptr);
  Set(23);
  EXPECT_EQ(Get(), 23);
}

// A weak *constant* is excluded from demotion (duplicated constants are
// ODR-harmless). Pins the isConstant() guard: nothing breaks and the jitted
// copy reads the right value even though the process may hold its own.
TYPED_TEST(CPPINTEROP_TEST_MODE, DynamicLibraryManager_WeakConstantNotDemoted) {
#ifdef EMSCRIPTEN
  GTEST_SKIP() << "Test not intended for Emscripten builds";
#endif
#ifdef CPPINTEROP_USE_CLING
  GTEST_SKIP() << "bindProcessWeakGlobals is wired into the clang-repl "
                  "CppInternal::Interpreter path, not cling's interpreter";
#endif
#if defined(_WIN32) || defined(__APPLE__)
  GTEST_SKIP() << "the dlsym-based weak-global binding targets ELF";
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "the dlsym-based weak-global binding is in-process only";

  ASSERT_TRUE(TestFixture::CreateInterpreter());
  Cpp::Process(""); // force the execution engine into existence

  // Address-taken so clang emits the constant global (rather than folding it).
  ASSERT_FALSE(Cpp::Process(R"(
    inline constexpr int k_shared_const = 123;
    extern "C" int const_jit_value() { return k_shared_const; }
    extern "C" const void* const_jit_addr() { return &k_shared_const; }
  )"));

  // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
  auto* Value =
      reinterpret_cast<int (*)()>(Cpp::GetFunctionAddress("const_jit_value"));
  // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
  ASSERT_NE(Value, nullptr);
  EXPECT_EQ(Value(), 123);
}

#if !defined(_WIN32) && !defined(CPPINTEROP_USE_CLING) &&                      \
    CPPINTEROP_WORKAROUND_BIND_PROCESS_WEAK_GLOBALS

// Three of bindProcessWeakGlobals's decisions about a module are invisible to
// the interpreter tests above, which only see whether a singleton was shared.
// They are driven here on a hand-built module, where the resulting IR shows
// which decision was taken.
// TestSharedLib exports this inline static data member. It is constant
// initialized, so the process has no guard symbol for it.
constexpr const char* kProcessData = "_ZN16SingletonFixture15s_inline_memberE";
constexpr const char* kProcessGuard =
    "_ZGVN16SingletonFixture15s_inline_memberE";

static std::unique_ptr<llvm::Module> MakeElfModule(llvm::LLVMContext& Ctx) {
  auto M = std::make_unique<llvm::Module>("bindProcessWeakGlobals", Ctx);
  llvm::Triple HostTriple(llvm::sys::getProcessTriple());
#if CLANG_VERSION_MAJOR < 21
  M->setTargetTriple(HostTriple.str());
#else
  M->setTargetTriple(HostTriple);
#endif
  return M;
}

// The shape clang gives an inline variable: a mutable weak definition. The
// module takes ownership of every global built here.
static llvm::GlobalVariable* AddWeakGlobal(llvm::Module& M, const char* Name) {
  llvm::Type* Ty = llvm::Type::getInt32Ty(M.getContext());
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  return new llvm::GlobalVariable(M, Ty, /*isConstant=*/false,
                                  llvm::GlobalValue::LinkOnceODRLinkage,
                                  llvm::ConstantInt::get(Ty, 0), Name);
}

static llvm::GlobalVariable* AddStrongGlobal(llvm::Module& M,
                                             const char* Name) {
  llvm::Type* Ty = llvm::Type::getInt32Ty(M.getContext());
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  return new llvm::GlobalVariable(M, Ty, /*isConstant=*/false,
                                  llvm::GlobalValue::ExternalLinkage,
                                  llvm::ConstantInt::get(Ty, 0), Name);
}

static llvm::GlobalVariable*
AddUsedList(llvm::Module& M, llvm::ArrayRef<llvm::GlobalVariable*> Els) {
  llvm::SmallVector<llvm::Constant*, 2> Ops(Els.begin(), Els.end());
  auto* ATy = llvm::ArrayType::get(llvm::PointerType::getUnqual(M.getContext()),
                                   Ops.size());
  auto* Used = new llvm::GlobalVariable(
      M, ATy, /*isConstant=*/false, llvm::GlobalValue::AppendingLinkage,
      llvm::ConstantArray::get(ATy, Ops), "llvm.used");
  Used->setSection("llvm.metadata");
  return Used;
}

class BindProcessWeakGlobals : public ::testing::Test {
protected:
  void SetUp() override {
    if (!llvm::Triple(llvm::sys::getProcessTriple()).isOSBinFormatELF())
      GTEST_SKIP() << "the dlsym-based weak-global binding targets ELF";

    std::string Path = ResolveTestSharedLibPath();
    ASSERT_STRNE("", Path.c_str());
    ASSERT_NE(::dlopen(Path.c_str(), RTLD_NOW | RTLD_GLOBAL), nullptr);

    // The premise of every case below.
    ASSERT_NE(::dlsym(RTLD_DEFAULT, kProcessData), nullptr);
    ASSERT_EQ(::dlsym(RTLD_DEFAULT, kProcessGuard), nullptr);
  }
};

// Sharing the data but not the guard would re-run initialization on the
// process copy, so a module-local guard the process lacks blocks the demotion.
TEST_F(BindProcessWeakGlobals, GuardMissingFromProcessKeepsPair) {
  llvm::LLVMContext Ctx;
  auto M = MakeElfModule(Ctx);
  llvm::GlobalVariable* Data = AddWeakGlobal(*M, kProcessData);
  llvm::GlobalVariable* Guard = AddWeakGlobal(*M, kProcessGuard);

  EXPECT_FALSE(compat::bindProcessWeakGlobals(*M));

  EXPECT_TRUE(Data->hasInitializer());
  EXPECT_TRUE(Guard->hasInitializer());
}

// A used list that names nothing demoted stays in place.
TEST_F(BindProcessWeakGlobals, UsedListWithoutDemotedEntriesIsKept) {
  llvm::LLVMContext Ctx;
  auto M = MakeElfModule(Ctx);
  llvm::GlobalVariable* Data = AddWeakGlobal(*M, kProcessData);
  llvm::GlobalVariable* Keep = AddStrongGlobal(*M, "keep_me");
  llvm::GlobalVariable* Used = AddUsedList(*M, {Keep});

  EXPECT_TRUE(compat::bindProcessWeakGlobals(*M));

  EXPECT_TRUE(Data->isDeclaration());
  // The same object, so the list was never erased and rebuilt.
  EXPECT_EQ(M->getNamedGlobal("llvm.used"), Used);
}

// A used list that names a demoted global is rebuilt around the survivors; a
// declaration left in it fails the IR verifier.
TEST_F(BindProcessWeakGlobals, UsedListIsRebuiltAroundSurvivors) {
  llvm::LLVMContext Ctx;
  auto M = MakeElfModule(Ctx);
  llvm::GlobalVariable* Data = AddWeakGlobal(*M, kProcessData);
  llvm::GlobalVariable* Keep = AddStrongGlobal(*M, "keep_me");
  AddUsedList(*M, {Data, Keep});

  EXPECT_TRUE(compat::bindProcessWeakGlobals(*M));

  EXPECT_TRUE(Data->isDeclaration());
  llvm::GlobalVariable* Rebuilt = M->getNamedGlobal("llvm.used");
  ASSERT_NE(Rebuilt, nullptr);
  EXPECT_EQ(Rebuilt->getSection(), "llvm.metadata");
  auto* Init = llvm::cast<llvm::ConstantArray>(Rebuilt->getInitializer());
  llvm::SmallVector<llvm::StringRef, 2> Survivors;
  for (llvm::Value* Op : Init->operand_values())
    Survivors.push_back(Op->getName());
  ASSERT_EQ(Survivors.size(), 1U);
  EXPECT_EQ(Survivors.front(), "keep_me");
  EXPECT_FALSE(llvm::verifyModule(*M, &llvm::errs()));
}

#endif // bindProcessWeakGlobals is declared
