// Tests for the generated C API wrappers (cppinterop_* functions).
// Each test verifies the C wrapper forwards correctly to Cpp::.

#include "Utils.h"

#include "CppInterOp/CXCppInterOp.h"

#include "gtest/gtest.h"

#include <cstdlib>
#include <cstring>

using namespace TestUtils;

TYPED_TEST(CppInterOpTest, CAPI_DeclareAndProcess) {
  TestFixture::CreateInterpreter();
  EXPECT_EQ(0, cppinterop_Declare("int capi_var = 42;", false));
  EXPECT_EQ(0, cppinterop_Process("capi_var++;"));
}

TYPED_TEST(CppInterOpTest, CAPI_ScopeQueries) {
  TestFixture::CreateInterpreter();
  Cpp::Declare("namespace CAPINs { class Foo {}; }");

  auto global = cppinterop_GetGlobalScope();
  EXPECT_NE(global, nullptr);

  auto ns = cppinterop_GetScope("CAPINs", nullptr);
  EXPECT_NE(ns, nullptr);
  EXPECT_TRUE(cppinterop_IsNamespace(ns));
  EXPECT_FALSE(cppinterop_IsClass(ns));

  auto foo = cppinterop_GetScope("Foo", ns);
  EXPECT_NE(foo, nullptr);
  EXPECT_TRUE(cppinterop_IsClass(foo));
  EXPECT_TRUE(cppinterop_IsComplete(foo));
  EXPECT_GT(cppinterop_SizeOf(foo), 0U);

  auto parent = cppinterop_GetParentScope(foo);
  EXPECT_EQ(parent, ns);

  char* name = cppinterop_GetName(foo);
  EXPECT_STREQ(name, "Foo");
  free(name); // NOLINT
}

TYPED_TEST(CppInterOpTest, CAPI_TypeQueries) {
  TestFixture::CreateInterpreter();
  Cpp::Declare("class CAPIType { int x; };");

  auto scope = cppinterop_GetNamed("CAPIType", nullptr);
  auto type = cppinterop_GetTypeFromScope(scope);
  EXPECT_NE(type, nullptr);

  auto back = cppinterop_GetScopeFromType(type);
  EXPECT_EQ(back, scope);

  EXPECT_FALSE(cppinterop_IsPointerType(type));
  EXPECT_FALSE(cppinterop_IsReferenceType(type));
  EXPECT_FALSE(cppinterop_IsEnumScope(scope));
  EXPECT_FALSE(cppinterop_IsBuiltin(type));
  EXPECT_FALSE(cppinterop_IsAbstract(scope));
  EXPECT_TRUE(cppinterop_IsPODType(type));
  EXPECT_GT(cppinterop_GetSizeOfType(type), 0U);
}

TYPED_TEST(CppInterOpTest, CAPI_FunctionQueries) {
  TestFixture::CreateInterpreter();
  Cpp::Declare("int capi_add(int a, int b) { return a + b; }");

  auto fn = cppinterop_GetNamed("capi_add", nullptr);
  EXPECT_NE(fn, nullptr);
  EXPECT_TRUE(cppinterop_IsFunction(fn));
  // The C-API spelling (CppConstFuncRef) is reachable here via the alias
  // in CXCppInterOp.h. Cross-kind narrowing is explicit via struct-init
  // from .data, same as on the C++ side.
  CppConstFuncRef fnAsFunc{fn.data};
  EXPECT_FALSE(cppinterop_IsMethod(fnAsFunc));
  EXPECT_EQ(cppinterop_GetFunctionNumArgs(fnAsFunc), 2U);
  EXPECT_EQ(cppinterop_GetFunctionRequiredArgs(fnAsFunc), 2U);

  auto ret = cppinterop_GetFunctionReturnType(fnAsFunc);
  EXPECT_NE(ret, nullptr);

  char* sig = cppinterop_GetFunctionSignature(fnAsFunc);
  EXPECT_NE(sig, nullptr);
  EXPECT_TRUE(strstr(sig, "capi_add") != nullptr);
  free(sig); // NOLINT
}

TYPED_TEST(CppInterOpTest, CAPI_VariableQueries) {
  TestFixture::CreateInterpreter();
  Cpp::Declare("namespace CAPIVarNs { int gvar = 99; }");

  auto ns = cppinterop_GetNamed("CAPIVarNs", nullptr);
  auto var = cppinterop_GetNamed("gvar", ns);
  EXPECT_NE(var, nullptr);
  EXPECT_TRUE(cppinterop_IsVariable(var));
}

TYPED_TEST(CppInterOpTest, CAPI_EnumQueries) {
  TestFixture::CreateInterpreter();
  Cpp::Declare("enum CAPIEnum { A, B, C };");

  auto e = cppinterop_GetNamed("CAPIEnum", nullptr);
  EXPECT_TRUE(cppinterop_IsEnumScope(e));
}

TYPED_TEST(CppInterOpTest, CAPI_TemplateQueries) {
  TestFixture::CreateInterpreter();
  Cpp::Declare("template<typename T> struct CAPITmpl { T val; };");

  auto tmpl = cppinterop_GetNamed("CAPITmpl", nullptr);
  EXPECT_TRUE(cppinterop_IsTemplate(tmpl));
  EXPECT_FALSE(cppinterop_IsTemplateSpecialization(tmpl));
}

TYPED_TEST(CppInterOpTest, CAPI_ClassMemberQueries) {
  TestFixture::CreateInterpreter();
  Cpp::Declare(R"(
    class CAPICls {
    public:
      int pub_var;
      void pub_fn() {}
      virtual void virt_fn() {}
      static void static_fn() {}
    };
  )");

  auto cls = cppinterop_GetNamed("CAPICls", nullptr);
  EXPECT_TRUE(cppinterop_HasDefaultConstructor(cls));
  EXPECT_TRUE(cppinterop_IsClassPolymorphic(cls));

  auto dtor = cppinterop_GetDestructor(cls);
  EXPECT_NE(dtor, nullptr);
}

TYPED_TEST(CppInterOpTest, CAPI_Construct) {
#ifdef EMSCRIPTEN
  GTEST_SKIP() << "Test fails for Emscripten builds";
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";
  std::vector<const char*> args = {"-include", "new"};
  TestFixture::CreateInterpreter(args);
  Cpp::Declare("struct CAPIObj { int x = 7; };");

  auto scope = cppinterop_GetNamed("CAPIObj", nullptr);
  auto obj = cppinterop_Construct(scope, nullptr, 1);
  EXPECT_NE(obj, nullptr);
  cppinterop_Destruct(obj, scope, true, 0);
}

TYPED_TEST(CppInterOpTest, CAPI_EnumFunctions) {
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Evaluate not supported in OOP JIT";
  TestFixture::CreateInterpreter();
  Cpp::Declare("const int ci = 42;");

  auto scope = cppinterop_GetNamed("ci", nullptr);
  auto type = cppinterop_GetVariableType(scope);

  // HasTypeQualifier takes QualKind (unsigned in C).
  // QualKind::kConst == 1U
  EXPECT_TRUE(cppinterop_HasTypeQualifier(type, /*kConst=*/1U));
  EXPECT_FALSE(cppinterop_HasTypeQualifier(type, /*kVolatile=*/2U));

  // RemoveTypeQualifier returns the unqualified type.
  auto unqual = cppinterop_RemoveTypeQualifier(type, /*kConst=*/1U);
  EXPECT_FALSE(cppinterop_HasTypeQualifier(unqual, /*kConst=*/1U));

  // GetLanguage returns an InterpreterLanguage enum as unsigned char.
  // CXX == 5 (Unknown=0, Asm=1, CIR=2, LLVM_IR=3, C=4, CXX=5)
  EXPECT_EQ(cppinterop_GetLanguage(nullptr), 5);
}

// Exercises the hand-written cppinterop_Evaluate C bridge. The C++ overload
// returning Cpp::Box has no C wrapper (NoCWrapper); C clients use this form.
TYPED_TEST(CppInterOpTest, CAPI_Evaluate) {
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Evaluate not supported in OOP JIT";
  TestFixture::CreateInterpreter();

  bool had_error = true; // ensure success path clears it
  EXPECT_EQ(cppinterop_Evaluate("42", &had_error), 42);
  EXPECT_FALSE(had_error);

  // Null HadError pointer is documented as supported.
  EXPECT_EQ(cppinterop_Evaluate("7 * 6", nullptr), 42);

  // Error path: function writes true to HadError and returns ~0UL.
  // Use a no-Value-after-success snippet (pure declaration) instead of
  // `#error`; both hit the `!V.hasValue()` arm in CXCppInterOp.cpp's
  // legacy Evaluate, but `#error` emits clang diagnostics that MSBuild's
  // error-regex flags as a compile failure on the Windows CI runner.
  had_error = false;
  EXPECT_EQ(static_cast<unsigned long>(
                cppinterop_Evaluate("class CAPIEvalNoValue {};", &had_error)),
            ~0UL);
  EXPECT_TRUE(had_error);
}

TYPED_TEST(CppInterOpTest, CAPI_CollectionReturn) {
  TestFixture::CreateInterpreter();
  Cpp::Declare("enum CAPIColEnum { X, Y, Z };");

  auto e = cppinterop_GetNamed("CAPIColEnum", nullptr);
  Cpp::CppInterOpArray constants = cppinterop_GetEnumConstants(e);
  EXPECT_EQ(constants.size, 3U);
  EXPECT_NE(constants.data, nullptr);
  cppinterop_DisposeArray(constants);
}

TYPED_TEST(CppInterOpTest, CAPI_CollectionOutParam) {
  TestFixture::CreateInterpreter();
  Cpp::Declare(R"(
    struct CAPIMembers {
      int a;
      double b;
      static int c;
    };
  )");

  auto cls = cppinterop_GetNamed("CAPIMembers", nullptr);
  Cpp::CppInterOpArray members = cppinterop_GetDatamembers(cls);
  EXPECT_EQ(members.size, 2U); // a and b (not static c)
  cppinterop_DisposeArray(members);

  Cpp::CppInterOpArray statics = cppinterop_GetStaticDatamembers(cls);
  EXPECT_EQ(statics.size, 1U); // only c
  cppinterop_DisposeArray(statics);
}

TYPED_TEST(CppInterOpTest, CAPI_StringCollectionOutParam) {
  TestFixture::CreateInterpreter();
  Cpp::Declare("namespace CAPIStrNs { enum E1 { P, Q }; enum E2 { R }; }");

  auto ns = cppinterop_GetNamed("CAPIStrNs", nullptr);
  Cpp::CppInterOpStringArray enums = cppinterop_GetEnums(ns);
  EXPECT_EQ(enums.size, 2U);
  // Verify the enum names are present (order may vary).
  bool foundE1 = false, foundE2 = false;
  for (size_t i = 0; i < enums.size; ++i) {
    if (strcmp(enums.data[i], "E1") == 0)
      foundE1 = true;
    if (strcmp(enums.data[i], "E2") == 0)
      foundE2 = true;
  }
  EXPECT_TRUE(foundE1);
  EXPECT_TRUE(foundE2);
  cppinterop_DisposeStringArray(enums);

  // Disposing an empty array should not crash.
  Cpp::CppInterOpArray empty = {nullptr, 0};
  cppinterop_DisposeArray(empty);
}

TYPED_TEST(CppInterOpTest, CAPI_CreateInterpreterInParam) {
  // Test the vector in-param pattern: CreateInterpreter(ptr, size, ptr, size).
  const char* args[] = {"-std=c++17"};
  auto I = cppinterop_CreateInterpreter(args, 1, nullptr, 0);
  EXPECT_NE(I, nullptr);
}

TYPED_TEST(CppInterOpTest, CAPI_GetClassTemplatedMethods) {
  TestFixture::CreateInterpreter();
  Cpp::Declare(R"(
    struct CAPITmplMethods {
      template<typename T> T get() { return T{}; }
      template<typename T> void set(T) {}
      void plain() {}
    };
  )");

  auto cls = cppinterop_GetNamed("CAPITmplMethods", nullptr);
  ASSERT_NE(cls, nullptr);

  // "get" has templated methods.
  Cpp::CppInterOpArray gets = cppinterop_GetClassTemplatedMethods("get", cls);
  EXPECT_EQ(gets.size, 1U);
  cppinterop_DisposeArray(gets);

  // "set" has templated methods.
  Cpp::CppInterOpArray sets = cppinterop_GetClassTemplatedMethods("set", cls);
  EXPECT_EQ(sets.size, 1U);
  cppinterop_DisposeArray(sets);

  // "plain" is not templated — should return empty.
  Cpp::CppInterOpArray plains =
      cppinterop_GetClassTemplatedMethods("plain", cls);
  EXPECT_EQ(plains.size, 0U);
  cppinterop_DisposeArray(plains);

  // Non-existent method — should return empty.
  Cpp::CppInterOpArray none =
      cppinterop_GetClassTemplatedMethods("nonexistent", cls);
  EXPECT_EQ(none.size, 0U);
  cppinterop_DisposeArray(none);
}

TYPED_TEST(CppInterOpTest, CAPI_StringFunctions) {
  TestFixture::CreateInterpreter();

  char* version = cppinterop_GetVersion();
  EXPECT_NE(version, nullptr);
  EXPECT_NE(version[0], '\0');
  free(version); // NOLINT

#ifndef _WIN32
  // Itanium mangling; Windows uses MSVC mangling.
  char* demangled = cppinterop_Demangle("_Z3foov");
  EXPECT_STREQ(demangled, "foo()");
  free(demangled); // NOLINT
#endif
}

// --- Handle tagging death tests ---
// Verify that passing the wrong kind of handle (e.g. a Type where a
// Scope is expected) triggers an assertion. These only fire in debug
// builds with assertions enabled.

TYPED_TEST(CppInterOpTest, CAPI_OpaqueHandleTypes) {
  TestFixture::CreateInterpreter();
  Cpp::Declare("class HandleTest { int x; };");

  // wrap/unwrap round-trip through the C++ API.
  auto raw = Cpp::GetNamed("HandleTest");
  Cpp::DeclRef decl = Cpp::wrap<Cpp::DeclRef>(raw.data);
  EXPECT_EQ(Cpp::unwrap<void>(decl), raw.data);

  auto rawType = Cpp::GetTypeFromScope(raw);
  Cpp::TypeRef type = Cpp::wrap<Cpp::TypeRef>(rawType.data);
  EXPECT_EQ(Cpp::unwrap<void>(type), rawType.data);

  // DeclRef and TypeRef are distinct types:
  // Cpp::unwrap<void>(type) compiles — correct kind
  // Cpp::unwrap<void>(decl) compiles — correct kind
  // But passing TypeRef where DeclRef is expected would not compile
  // once the API signatures use the typed handles.
}
