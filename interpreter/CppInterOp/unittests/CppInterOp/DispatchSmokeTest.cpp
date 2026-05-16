#include "CppInterOp/Dispatch.h"

#include "gtest/gtest.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Smoke tests that verify the dispatch wrappers forward correctly for
// representative API functions across all categories. The full test
// suite runs against the direct API; these confirm the dispatch layer
// doesn't introduce regressions.

// --- Interpreter ---

TEST(DispatchSmokeTest, CreateAndDeleteInterpreter) {
  auto* I = Cpp::CreateInterpreter({});
  EXPECT_NE(I, nullptr);
  EXPECT_TRUE(Cpp::DeleteInterpreter());
}

TEST(DispatchSmokeTest, DeclareAndProcess) {
  Cpp::CreateInterpreter({});
  EXPECT_EQ(0, Cpp::Declare("int dispatch_x = 42;"));
  EXPECT_EQ(0, Cpp::Process("dispatch_x++;"));
}

// --- Scope queries ---

TEST(DispatchSmokeTest, ScopeLookup) {
  Cpp::CreateInterpreter({});
  Cpp::Declare("namespace DispNS { class Foo {}; }");

  auto* global = Cpp::GetGlobalScope();
  EXPECT_NE(global, nullptr);

  auto* ns = Cpp::GetNamed("DispNS");
  EXPECT_NE(ns, nullptr);
  EXPECT_TRUE(Cpp::IsNamespace(ns));

  auto* foo = Cpp::GetScope("Foo", ns);
  EXPECT_NE(foo, nullptr);
  EXPECT_TRUE(Cpp::IsClass(foo));
  EXPECT_FALSE(Cpp::IsNamespace(foo));
}

// --- Type queries ---

TEST(DispatchSmokeTest, TypeReflection) {
  Cpp::CreateInterpreter({});
  Cpp::Declare("class DispType { int x; };");

  auto* scope = Cpp::GetNamed("DispType");
  auto* type = Cpp::GetTypeFromScope(scope);
  EXPECT_NE(type, nullptr);

  auto* scope_back = Cpp::GetScopeFromType(type);
  EXPECT_EQ(scope, scope_back);

  EXPECT_GT(Cpp::SizeOf(scope), 0U);
  EXPECT_FALSE(Cpp::IsPointerType(type));
  EXPECT_FALSE(Cpp::IsEnumType(type));
}

// --- Function reflection ---

TEST(DispatchSmokeTest, FunctionReflection) {
  Cpp::CreateInterpreter({});
  Cpp::Declare("namespace DispFn { int add(int a, int b) { return a+b; } }");

  auto* ns = Cpp::GetNamed("DispFn");
  auto fns = Cpp::GetFunctionsUsingName(ns, "add");
  EXPECT_EQ(fns.size(), 1U);

  EXPECT_EQ(Cpp::GetFunctionNumArgs(fns[0]), 2U);
  EXPECT_EQ(Cpp::GetName(fns[0]), "add");
}

// --- Variable reflection ---

TEST(DispatchSmokeTest, VariableReflection) {
  Cpp::CreateInterpreter({});
  Cpp::Declare("namespace DispVar { int gval = 99; }");

  auto* ns = Cpp::GetNamed("DispVar");
  // GetNamed finds variables inside a namespace scope.
  auto* var = Cpp::GetNamed("gval", ns);
  EXPECT_NE(var, nullptr);
  EXPECT_TRUE(Cpp::IsVariable(var));
}

// --- Templates ---

TEST(DispatchSmokeTest, TemplateInstantiation) {
  Cpp::CreateInterpreter({});
  Cpp::Declare("template<typename T> struct DispTmpl { T val; };");

  auto* tmpl = Cpp::GetNamed("DispTmpl");
  EXPECT_TRUE(Cpp::IsTemplate(tmpl));
}

// --- Construct / Destruct ---

TEST(DispatchSmokeTest, ConstructDestruct) {
  Cpp::CreateInterpreter({"-include", "new"});
  Cpp::Declare("struct DispObj { int x = 7; };");

  auto* scope = Cpp::GetNamed("DispObj");
  EXPECT_TRUE(Cpp::IsComplete(scope));

  auto* obj = Cpp::Construct(scope);
  EXPECT_NE(obj, nullptr);
  Cpp::Destruct(obj, scope, /*withFree=*/true);
}

// --- Enum ---

TEST(DispatchSmokeTest, EnumReflection) {
  Cpp::CreateInterpreter({});
  Cpp::Declare("enum DispEnum { A, B, C };");

  auto* e = Cpp::GetNamed("DispEnum");
  EXPECT_TRUE(Cpp::IsEnumScope(e));

  auto constants = Cpp::GetEnumConstants(e);
  EXPECT_EQ(constants.size(), 3U);
}

// --- Demangle ---

TEST(DispatchSmokeTest, Demangle) {
  Cpp::CreateInterpreter({});
  // _Z3foov is the mangled name for "foo()"
  auto result = Cpp::Demangle("_Z3foov");
  EXPECT_EQ(result, "foo()");
}

// --- Version ---

TEST(DispatchSmokeTest, GetVersion) {
  auto ver = Cpp::GetVersion();
  EXPECT_FALSE(ver.empty());
}

// --- C API through dlsym ---
// The cppinterop_* functions are extern "C" symbols exported from
// libclangCppInterOp. A real C consumer would dlopen the library and
// dlsym them directly — that is what these tests verify. They open a
// separate handle (RTLD_LOCAL) to prove the symbols are accessible
// without linking the library.

// NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)

namespace {
/// RAII wrapper for a dlopen handle used by the C API dispatch tests.
class DlHandle {
  void* H = nullptr;

public:
  DlHandle(const DlHandle&) = delete;
  DlHandle& operator=(const DlHandle&) = delete;
  explicit DlHandle(const char* path) {
#ifndef _WIN32
    H = dlopen(path, RTLD_LOCAL | RTLD_NOW);
#endif
  }
  ~DlHandle() {
#ifndef _WIN32
    if (H)
      dlclose(H);
#endif
  }
  void* sym(const char* name) const {
#ifndef _WIN32
    return H ? dlsym(H, name) : nullptr;
#else
    return nullptr;
#endif
  }
  explicit operator bool() const { return H != nullptr; }
};
} // namespace

TEST(DispatchSmokeTest, CAPI_ScalarFunctions) {
  Cpp::CreateInterpreter({});
  Cpp::Declare("namespace DispCAPI { class Cls {}; int var = 1; }");

  DlHandle lib(CPPINTEROP_LIB_PATH);
  ASSERT_TRUE(lib) << "Failed to dlopen " << CPPINTEROP_LIB_PATH;

  using IsClassFn = bool (*)(void*);
  using GetNameFn = char* (*)(void*);
  using GetNamedFn = void* (*)(const char*, void*);

  auto* isClass = reinterpret_cast<IsClassFn>(lib.sym("cppinterop_IsClass"));
  auto* getName = reinterpret_cast<GetNameFn>(lib.sym("cppinterop_GetName"));
  auto* getNamed = reinterpret_cast<GetNamedFn>(lib.sym("cppinterop_GetNamed"));
  ASSERT_NE(isClass, nullptr);
  ASSERT_NE(getName, nullptr);
  ASSERT_NE(getNamed, nullptr);

  auto* cls = getNamed("DispCAPI", nullptr);
  EXPECT_FALSE(isClass(cls)); // namespace, not class

  auto* inner = getNamed("Cls", cls);
  EXPECT_TRUE(isClass(inner));

  char* name = getName(inner);
  EXPECT_STREQ(name, "Cls");
  free(name); // NOLINT
}

TEST(DispatchSmokeTest, CAPI_CollectionFunctions) {
  Cpp::CreateInterpreter({});
  Cpp::Declare("enum DispCAPIEnum { DA, DB, DC };");

  DlHandle lib(CPPINTEROP_LIB_PATH);
  ASSERT_TRUE(lib) << "Failed to dlopen " << CPPINTEROP_LIB_PATH;

  using GetNamedFn = void* (*)(const char*, void*);
  using GetEnumConstantsFn = Cpp::CppInterOpArray (*)(void*);
  using DisposeArrayFn = void (*)(Cpp::CppInterOpArray);

  auto* getNamed = reinterpret_cast<GetNamedFn>(lib.sym("cppinterop_GetNamed"));
  auto* getEnumConstants = reinterpret_cast<GetEnumConstantsFn>(
      lib.sym("cppinterop_GetEnumConstants"));
  auto* disposeArray =
      reinterpret_cast<DisposeArrayFn>(lib.sym("cppinterop_DisposeArray"));
  ASSERT_NE(getNamed, nullptr);
  ASSERT_NE(getEnumConstants, nullptr);
  ASSERT_NE(disposeArray, nullptr);

  auto* e = getNamed("DispCAPIEnum", nullptr);
  auto arr = getEnumConstants(e);
  EXPECT_EQ(arr.size, 3U);
  EXPECT_NE(arr.data, nullptr);
  disposeArray(arr);
}

// NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
