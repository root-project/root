#include "Utils.h"

#include "../../lib/CppInterOp/Unwrap.h"

#include "CppInterOp/CppInterOp.h"
#include "CppInterOp/CppInterOpTypes.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/Basic/Version.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Sema.h"

#include "llvm/Support/Error.h"

#include "gtest/gtest.h"

#include <array>
#include <cstddef>
#include <string>
#include <utility>

using namespace TestUtils;
using namespace llvm;
using namespace clang;

TYPED_TEST(CPPINTEROP_TEST_MODE, VariableReflection_GetDatamembers) {
  std::vector<Decl*> Decls;
  std::string code = R"(
    class C {
    public:
      int a;
      static int b;
    private:
      int c;
      static int d;
    protected:
      int e;
      static int f;
    };
    void sum(int,int);

    class D : public C {
    public:
      int x;
      using C::e;
    };
    )";

  std::vector<Cpp::DeclRef> datamembers;
  std::vector<Cpp::DeclRef> datamembers1;
  std::vector<Cpp::DeclRef> datamembers2;
  GetAllTopLevelDecls(code, Decls);
  Cpp::GetDatamembers(Decls[0], datamembers);
  Cpp::GetDatamembers(Decls[1], datamembers1);
  Cpp::GetDatamembers(Decls[2], datamembers2);

  // non static field
  EXPECT_EQ(Cpp::GetQualifiedName(datamembers[0]), "C::a");
  EXPECT_EQ(Cpp::GetQualifiedName(datamembers[1]), "C::c");
  EXPECT_EQ(Cpp::GetQualifiedName(datamembers[2]), "C::e");
  EXPECT_EQ(datamembers.size(), 3);
  EXPECT_EQ(datamembers1.size(), 0);

  // static fields
  datamembers.clear();
  datamembers1.clear();

  Cpp::GetStaticDatamembers(Decls[0], datamembers);
  Cpp::GetStaticDatamembers(Decls[1], datamembers1);

  EXPECT_EQ(Cpp::GetQualifiedName(datamembers[0]), "C::b");
  EXPECT_EQ(Cpp::GetQualifiedName(datamembers[1]), "C::d");
  EXPECT_EQ(Cpp::GetQualifiedName(datamembers[2]), "C::f");
  EXPECT_EQ(datamembers.size(), 3);
  EXPECT_EQ(datamembers1.size(), 0);

  // derived class
  EXPECT_EQ(datamembers2.size(), 2);
  EXPECT_EQ(Cpp::GetQualifiedName(datamembers2[0]), "D::x");
  EXPECT_EQ(Cpp::GetQualifiedName(Cpp::GetUnderlyingScope(datamembers2[1])),
            "C::e");
  EXPECT_TRUE(Cpp::IsPublicVariable(datamembers2[0]));
  EXPECT_TRUE(Cpp::IsPublicVariable(datamembers2[1]));
  EXPECT_TRUE(
      Cpp::IsProtectedVariable(Cpp::GetUnderlyingScope(datamembers2[1])));
}
// If on Windows disable  warning due to unnamed structs/unions in defined CODE.
#ifdef _WIN32
#pragma warning(disable : 4201)
#endif
#define CODE                                                                   \
  struct Klass1 {                                                              \
    Klass1(int i) : num(1), b(i) {}                                            \
    int num;                                                                   \
    union {                                                                    \
      double a;                                                                \
      int b;                                                                   \
    };                                                                         \
  } const k1(5);                                                               \
  struct Klass2 {                                                              \
    Klass2(double d) : num(2), a(d) {}                                         \
    int num;                                                                   \
    struct {                                                                   \
      double a;                                                                \
      int b;                                                                   \
    };                                                                         \
  } const k2(2.5);                                                             \
  struct Klass3 {                                                              \
    Klass3(int i) : num(i) {}                                                  \
    int num;                                                                   \
    struct {                                                                   \
      double a;                                                                \
      union {                                                                  \
        float b;                                                               \
        int c;                                                                 \
      };                                                                       \
    };                                                                         \
    int num2;                                                                  \
  } const k3(5);

CODE

TYPED_TEST(CPPINTEROP_TEST_MODE, VariableReflection_DatamembersWithAnonymousStructOrUnion) {
#if CLANG_VERSION_MAJOR == 20 && defined(CPPINTEROP_USE_CLING) && defined(_WIN32)
  GTEST_SKIP() << "Test fails with Cling on Windows";
#endif

  std::vector<Decl*> Decls;
#define Stringify(s) Stringifyx(s)
#define Stringifyx(...) #__VA_ARGS__
  GetAllTopLevelDecls(Stringify(CODE), Decls);
#undef Stringifyx
#undef Stringify
#undef CODE

  std::vector<Cpp::DeclRef> datamembers_klass1;
  std::vector<Cpp::DeclRef> datamembers_klass2;
  std::vector<Cpp::DeclRef> datamembers_klass3;

  Cpp::GetDatamembers(Decls[0], datamembers_klass1);
  Cpp::GetDatamembers(Decls[2], datamembers_klass2);
  Cpp::GetDatamembers(Decls[4], datamembers_klass3);

  EXPECT_EQ(datamembers_klass1.size(), 3);
  EXPECT_EQ(datamembers_klass2.size(), 3);

  EXPECT_EQ(Cpp::GetVariableOffset(datamembers_klass1[0]), 0);
  // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
  EXPECT_EQ(Cpp::GetVariableOffset(datamembers_klass1[1]),
            ((intptr_t) & (k1.a)) - ((intptr_t) & (k1.num)));
  EXPECT_EQ(Cpp::GetVariableOffset(datamembers_klass1[2]),
            ((intptr_t) & (k1.b)) - ((intptr_t) & (k1.num)));

  EXPECT_EQ(Cpp::GetVariableOffset(datamembers_klass2[0]), 0);
  EXPECT_EQ(Cpp::GetVariableOffset(datamembers_klass2[1]),
            ((intptr_t) & (k2.a)) - ((intptr_t) & (k2.num)));
  EXPECT_EQ(Cpp::GetVariableOffset(datamembers_klass2[2]),
            ((intptr_t) & (k2.b)) - ((intptr_t) & (k2.num)));

  EXPECT_EQ(Cpp::GetVariableOffset(datamembers_klass3[0]), 0);
  EXPECT_EQ(Cpp::GetVariableOffset(datamembers_klass3[1]),
            ((intptr_t) & (k3.a)) - ((intptr_t) & (k3.num)));
  EXPECT_EQ(Cpp::GetVariableOffset(datamembers_klass3[2]),
            ((intptr_t) & (k3.b)) - ((intptr_t) & (k3.num)));
  EXPECT_EQ(Cpp::GetVariableOffset(datamembers_klass3[3]),
            ((intptr_t) & (k3.c)) - ((intptr_t) & (k3.num)));
  EXPECT_EQ(Cpp::GetVariableOffset(datamembers_klass3[4]),
            ((intptr_t) & (k3.num2)) - ((intptr_t) & (k3.num)));
  // NOLINTEND(cppcoreguidelines-pro-type-union-access)
#ifdef _WIN32
#pragma warning(default : 4201)
#endif
}

TYPED_TEST(CPPINTEROP_TEST_MODE, VariableReflection_GetTypeAsString) {

  std::string code = R"(
  namespace my_namespace {

  struct Container {
    int value;
  };

  struct Wrapper {
    Container item;
  };

  }
  )";

  TestFixture::CreateInterpreter();
  EXPECT_EQ(Cpp::Declare(code.c_str()), 0);

  Cpp::DeclRef wrapper = Cpp::GetScopeFromCompleteName("my_namespace::Wrapper");
  EXPECT_TRUE(wrapper);

  std::vector<Cpp::DeclRef> datamembers;
  Cpp::GetDatamembers(wrapper, datamembers);
  EXPECT_EQ(datamembers.size(), 1);

  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetVariableType(datamembers[0])),
            "my_namespace::Container");
}

TYPED_TEST(CPPINTEROP_TEST_MODE, VariableReflection_LookupDatamember) {
  std::vector<Decl*> Decls;
  std::string code = R"(
    class C {
    public:
      int a;
      static int b;
    private:
      int c;
      static int d;
    protected:
      int e;
      static int f;
    };
    )";

  GetAllTopLevelDecls(code, Decls);

  EXPECT_EQ(Cpp::GetQualifiedName(Cpp::LookupDatamember("a", Decls[0])), "C::a");
  EXPECT_EQ(Cpp::GetQualifiedName(Cpp::LookupDatamember("c", Decls[0])), "C::c");
  EXPECT_EQ(Cpp::GetQualifiedName(Cpp::LookupDatamember("e", Decls[0])), "C::e");
  EXPECT_EQ(Cpp::GetQualifiedName(Cpp::LookupDatamember("k", Decls[0])), "<unnamed>");
}

TYPED_TEST(CPPINTEROP_TEST_MODE, VariableReflection_GetVariableType) {
  std::vector<Decl*> Decls;
  std::string code = R"(
    class C {};

    template<typename T>
    class E {};

    int a;
    char b;
    C c;
    C *d;
    E<int> e;
    E<int> *f;
    int g[4];
    auto fn = []() { return 1; };
    extern int uarr[];
    int uarr[7];
    )";

  GetAllTopLevelDecls(code, Decls);

  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetVariableType(Decls[2])), "int");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetVariableType(Decls[3])), "char");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetVariableType(Decls[4])), "C");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetVariableType(Decls[5])), "C *");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetVariableType(Decls[6])), "E<int>");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetVariableType(Decls[7])), "E<int> *");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetVariableType(Decls[8])), "int[4]");

  EXPECT_FALSE(Cpp::IsLambdaClass(Cpp::GetVariableType(Decls[8])));
  EXPECT_TRUE(Cpp::IsLambdaClass(Cpp::GetVariableType(Decls[9])));

  std::vector<Decl*> UArrDecls;
  for (Decl* D : Decls)
    if (Cpp::GetName(D) == "uarr")
      UArrDecls.push_back(D);
  ASSERT_EQ(UArrDecls.size(), 2);
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetVariableType(UArrDecls[0])), "int[7]");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetVariableType(UArrDecls[1])), "int[7]");
}

#define CODE                                                                   \
  int a;                                                                       \
  const int N = 5;                                                             \
  static int S = N + 1;                                                        \
  static const int SN = S + 1;                                                 \
  class C {                                                                    \
  public:                                                                      \
    int a;                                                                     \
    double b;                                                                  \
    int* c;                                                                    \
    int d;                                                                     \
    static int s_a;                                                            \
  } c;                                                                         \
  int C::s_a = 7 + SN;

CODE

TYPED_TEST(CPPINTEROP_TEST_MODE, VariableReflection_GetVariableOffset) {
  std::vector<Decl *> Decls;
#define Stringify(s) Stringifyx(s)
#define Stringifyx(...) #__VA_ARGS__
  GetAllTopLevelDecls(Stringify(CODE), Decls);
#undef Stringifyx
#undef Stringify
#undef CODE

  EXPECT_EQ(7, Decls.size());

  std::vector<Cpp::DeclRef> datamembers;
  Cpp::GetDatamembers(Decls[4], datamembers);

  EXPECT_TRUE((bool)Cpp::GetVariableOffset(Decls[0])); // a
  EXPECT_TRUE((bool)Cpp::GetVariableOffset(Decls[1])); // N
  EXPECT_TRUE((bool)Cpp::GetVariableOffset(Decls[2])); // S
  EXPECT_TRUE((bool)Cpp::GetVariableOffset(Decls[3])); // SN

  EXPECT_EQ(Cpp::GetVariableOffset(datamembers[0]), 0);

  EXPECT_EQ(Cpp::GetVariableOffset(datamembers[1]),
            ((intptr_t) & (c.b)) - ((intptr_t) & (c.a)));
  EXPECT_EQ(Cpp::GetVariableOffset(datamembers[2]),
            ((intptr_t) & (c.c)) - ((intptr_t) & (c.a)));
  EXPECT_EQ(Cpp::GetVariableOffset(datamembers[3]),
            ((intptr_t) & (c.d)) - ((intptr_t) & (c.a)));

  auto VD_C_s_a = Cpp::GetNamed("s_a", Decls[4]); // C::s_a
  EXPECT_TRUE((bool)Cpp::GetVariableOffset(VD_C_s_a));

  struct K {
    int x;
    int y;
    int z;
  };
  Cpp::Declare("struct K;");
  Cpp::DeclRef k = Cpp::GetNamed("K");
  EXPECT_TRUE(k);

  Cpp::Declare("struct K { int x; int y; int z; };");
  datamembers.clear();
  Cpp::GetDatamembers(k, datamembers);
  EXPECT_EQ(datamembers.size(), 3);

  EXPECT_EQ(Cpp::GetVariableOffset(datamembers[0]), offsetof(K, x));
  EXPECT_EQ(Cpp::GetVariableOffset(datamembers[1]), offsetof(K, y));
  EXPECT_EQ(Cpp::GetVariableOffset(datamembers[2]), offsetof(K, z));

  Cpp::Declare(R"(
    template <typename T> struct ClassWithStatic {
      static T const ref_value;
    };
    template <typename T> T constexpr ClassWithStatic<T>::ref_value = 42;
  )");

  Cpp::DeclRef klass = Cpp::GetNamed("ClassWithStatic");
  EXPECT_TRUE(klass);

  ASTContext& C = Interp->getCI()->getASTContext();
  std::vector<Cpp::TemplateArgInfo> template_args = {
      {C.IntTy.getAsOpaquePtr()}};
  Cpp::DeclRef klass_instantiated =
      Cpp::InstantiateTemplate(klass, template_args);
  EXPECT_TRUE(klass_instantiated);

  Cpp::DeclRef var = Cpp::GetNamed("ref_value", klass_instantiated);
  EXPECT_TRUE(var);

  EXPECT_TRUE(Cpp::GetVariableOffset(var));
}

TYPED_TEST(CPPINTEROP_TEST_MODE,
           VariableReflection_GetVariableOffset_NonTemplateStaticNoInit) {
  // A non-template class's static data member declared in-class and defined
  // out-of-line reaches the no-initializer branch with no template
  // instantiation pattern. Sema::InstantiateVariableDefinition requires a
  // pattern and dereferences null without one (std::partial_ordering::less
  // is the in-the-wild shape). Must not crash; must resolve the definition.
  TestFixture::CreateInterpreter();
  Cpp::Declare(R"(
    struct NonTemplateStatic {
      static const NonTemplateStatic less;
      int value;
    };
    inline constexpr NonTemplateStatic NonTemplateStatic::less{-1};
  )");
  Cpp::DeclRef klass = Cpp::GetNamed("NonTemplateStatic");
  EXPECT_TRUE(klass);
  Cpp::DeclRef var = Cpp::GetNamed("less", klass);
  EXPECT_TRUE(var);
  EXPECT_TRUE(Cpp::GetVariableOffset(var));

  // Declared and never defined: still no pattern and now no definition
  // either — the query must fail cleanly, not crash.
  Cpp::Declare(R"(
    struct NeverDefined {
      static const NeverDefined missing;
      int value;
    };
  )");
  Cpp::DeclRef klass2 = Cpp::GetNamed("NeverDefined");
  EXPECT_TRUE(klass2);
  Cpp::DeclRef var2 = Cpp::GetNamed("missing", klass2);
  EXPECT_TRUE(var2);
  EXPECT_FALSE(Cpp::GetVariableOffset(var2));
}

TYPED_TEST(CPPINTEROP_TEST_MODE,
           VariableReflection_GetVariableOffset_NoStaleUsedHandle) {
#ifdef __EMSCRIPTEN__
  // The stale-handle crash this test pins is native-JIT mechanics (ORC
  // freeing materialized IR), and a single failed module load poisons
  // Emscripten's process-global dynamic-linking state for every test that
  // follows (issue #1071) — not worth the fragility for a native-only
  // scenario.
  GTEST_SKIP() << "Stale-used-handle scenario is native-JIT specific";
#endif
  // Emitting a discardable-ODR variable through the UsedAttr route records
  // the global as a weak handle in codegen's llvm.used list. If the global
  // is later replaced/erased (weak-def discard when a later PTU re-emits
  // the same entity), the next PTU's emitUsed dereferences the nulled
  // handle. The offset query must not leave used-list residue.
  TestFixture::CreateInterpreter();
  Cpp::Declare(R"(
    struct UsedHandle {
      inline static int probe = 3;
    };
  )");
  Cpp::DeclRef klass = Cpp::GetNamed("UsedHandle");
  EXPECT_TRUE(klass);
  Cpp::DeclRef var = Cpp::GetNamed("probe", klass);
  EXPECT_TRUE(var);
  EXPECT_TRUE(Cpp::GetVariableOffset(var));
  // ForceCodeGen's UsedAttr is planted permanently on the AST decl; the
  // odr-use route must not.
  EXPECT_FALSE(Cpp::unwrap<Decl>(var)->hasAttr<clang::UsedAttr>());
#ifndef CPPINTEROP_USE_CLING
  // The UsedAttr lives on the AST decl, so every later PTU that re-emits
  // the entity re-adds it to that module's llvm.used — the residue the
  // stale-handle crash grows from. Neither a module that re-emits the
  // variable nor an unrelated one may carry llvm.used. (PTU/TheModule is
  // clang::Interpreter surface; cling covers this path via the UsedAttr
  // assert above.)
  {
    auto PTUOrErr = Interp->Parse("int consume_probe = UsedHandle::probe;");
    ASSERT_TRUE(bool(PTUOrErr));
    EXPECT_EQ(PTUOrErr->TheModule->getNamedGlobal("llvm.used"), nullptr);
    if (auto Err = Interp->Execute(*PTUOrErr))
      llvm::consumeError(std::move(Err));
  }
  {
    auto PTUOrErr = Interp->Parse("int flush_ptu = 0;");
    ASSERT_TRUE(bool(PTUOrErr));
    EXPECT_EQ(PTUOrErr->TheModule->getNamedGlobal("llvm.used"), nullptr);
    if (auto Err = Interp->Execute(*PTUOrErr))
      llvm::consumeError(std::move(Err));
  }
  EXPECT_TRUE(Cpp::GetNamed("flush_ptu"));
#endif // !CPPINTEROP_USE_CLING
  // The crash this guards against was cumulative — many force-emitted
  // statics plus JIT materialization cycles, interleaved with absorbed
  // parse failures. Mimic that sweep shape as a regression net; the
  // per-index sources are assembled at compile time by stringification.
  struct SweepCase {
    const char* decl;
    const char* cls;
    const char* var;
    const char* probe;
    const char* use;
  };
#define SWEEP_CASE(n)                                                          \
  {                                                                            \
    "struct Sweep" #n " { inline static int v" #n " = " #n "; };", "Sweep" #n, \
        "v" #n, "template <> struct Sweep" #n "<int>;",                        \
        "int use" #n " = Sweep" #n "::v" #n ";"                                \
  }
  constexpr std::array<SweepCase, 8> Sweeps = {
      {SWEEP_CASE(0), SWEEP_CASE(1), SWEEP_CASE(2), SWEEP_CASE(3),
       SWEEP_CASE(4), SWEEP_CASE(5), SWEEP_CASE(6), SWEEP_CASE(7)}};
#undef SWEEP_CASE
  for (const SweepCase& S : Sweeps) {
    Cpp::Declare(S.decl);
    Cpp::DeclRef k = Cpp::GetNamed(S.cls);
    ASSERT_TRUE(k);
    Cpp::DeclRef v = Cpp::GetNamed(S.var, k);
    ASSERT_TRUE(v);
    EXPECT_TRUE(Cpp::GetVariableOffset(v));
    Cpp::Declare(S.probe, /*silent=*/true); // expected parse failure
    Cpp::Declare(S.use);
  }
  Cpp::Declare("int sweep_done = 1;");
  EXPECT_TRUE(Cpp::GetNamed("sweep_done"));

  // Template-specialization members take the odr-use route too — the
  // Sema-built address-of needs no source spelling of the specialization,
  // and the caller has already instantiated the definition.
  Cpp::Declare(R"(
    template <typename T> struct TmplStatic {
      inline static int member = 7;
    };
  )");
  Cpp::DeclRef tmpl = Cpp::GetNamed("TmplStatic");
  EXPECT_TRUE(tmpl);
  ASTContext& C = Interp->getCI()->getASTContext();
  std::vector<Cpp::TemplateArgInfo> template_args = {
      {C.IntTy.getAsOpaquePtr()}};
  Cpp::DeclRef inst = Cpp::InstantiateTemplate(tmpl, template_args);
  EXPECT_TRUE(inst);
  Cpp::DeclRef member = Cpp::GetNamed("member", inst);
  EXPECT_TRUE(member);
  EXPECT_TRUE(Cpp::GetVariableOffset(member));
  EXPECT_FALSE(Cpp::unwrap<Decl>(member)->hasAttr<clang::UsedAttr>());

  // Anonymous-namespace members cannot be named from a fresh chunk of
  // source; they bail out of the odr-use route the same way.
  Cpp::Declare(R"(
    namespace {
      struct AnonNsStatic {
        inline static int member = 9;
      };
    }
  )");
  Cpp::DeclRef anon_klass = Cpp::GetNamed("AnonNsStatic");
  ASSERT_TRUE(anon_klass);
  Cpp::DeclRef anon_member = Cpp::GetNamed("member", anon_klass);
  ASSERT_TRUE(anon_member);
  EXPECT_TRUE(Cpp::GetVariableOffset(anon_member));
#ifndef CPPINTEROP_USE_CLING
  EXPECT_TRUE(Cpp::unwrap<Decl>(anon_member)->hasAttr<clang::UsedAttr>());
#endif

  // A linkage-spec block is transparent for naming purposes and stays on
  // the odr-use route.
  Cpp::Declare(R"(
    extern "C++" {
      struct CxxLinkStatic {
        inline static int member = 11;
      };
    }
  )");
  Cpp::DeclRef link_klass = Cpp::GetNamed("CxxLinkStatic");
  ASSERT_TRUE(link_klass);
  Cpp::DeclRef link_member = Cpp::GetNamed("member", link_klass);
  ASSERT_TRUE(link_member);
  EXPECT_TRUE(Cpp::GetVariableOffset(link_member));
  EXPECT_FALSE(Cpp::unwrap<Decl>(link_member)->hasAttr<clang::UsedAttr>());
}

#ifndef CPPINTEROP_USE_CLING
TYPED_TEST(CPPINTEROP_TEST_MODE,
           VariableReflection_GetVariableOffset_UnmaterializableDefinition) {
  // A definition whose dynamic initializer needs an undefined symbol parses
  // fine but cannot be materialized: the odr-use anchor's execution fails
  // and the query must fail cleanly. (The anchor's flush block is
  // clang-repl-only, hence the gate.)
#ifdef __EMSCRIPTEN__
  // The intentional materialization failure leaves the process-global wasm
  // dynamic-linking state broken for every later load (see issue #1071).
  GTEST_SKIP() << "A failed module load poisons wasm dynamic linking";
#endif
#ifdef _WIN32
  // The intentional failure makes ORC print "JIT session error :" on
  // stderr, which MSBuild's canonical-error scraping promotes to a build
  // error (MSB8066) even though every test passes.
  GTEST_SKIP() << "MSBuild's canonical-error scraping treats ORC's "
                  "JIT-session-failure stderr line as a build failure";
#endif
  // The intentional failure makes the remote dlupdate fail, and the
  // out-of-process session then blocks instead of reporting the error.
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "A failed dlupdate wedges the out-of-process session";
  TestFixture::CreateInterpreter();
  Cpp::Declare(R"(
    int undefined_fn();
    struct Unmaterializable {
      inline static int probe = undefined_fn();
    };
  )");
  Cpp::DeclRef klass = Cpp::GetNamed("Unmaterializable");
  ASSERT_TRUE(klass);
  Cpp::DeclRef var = Cpp::GetNamed("probe", klass);
  ASSERT_TRUE(var);
  EXPECT_FALSE(Cpp::GetVariableOffset(var));
}
#endif // !CPPINTEROP_USE_CLING

TYPED_TEST(CPPINTEROP_TEST_MODE,
           VariableReflection_GetVariableOffset_OdrUseAnchorPerInterpreter) {
  // The synthesized odr-use anchors must be named per-interpreter: a fresh
  // interpreter's AST (and anything derived from it, e.g. a crash
  // reproducer) may not depend on how many queries a sibling interpreter
  // has run.
  Cpp::InterpRef I1 = TestFixture::CreateInterpreter();
  ASSERT_TRUE(I1);
  Cpp::Declare(R"(struct PerInterpA { inline static int probe = 1; };)");
  Cpp::DeclRef ka = Cpp::GetNamed("PerInterpA");
  ASSERT_TRUE(ka);
  Cpp::DeclRef va = Cpp::GetNamed("probe", ka);
  ASSERT_TRUE(va);
  EXPECT_TRUE(Cpp::GetVariableOffset(va));
  EXPECT_TRUE(Cpp::GetNamed("__cppinterop_odr_use_v0"));

  Cpp::InterpRef I2 = TestFixture::CreateInterpreter();
  ASSERT_TRUE(I2);
  Cpp::Declare(R"(struct PerInterpB { inline static int probe = 2; };)");
  Cpp::DeclRef kb = Cpp::GetNamed("PerInterpB");
  ASSERT_TRUE(kb);
  Cpp::DeclRef vb = Cpp::GetNamed("probe", kb);
  ASSERT_TRUE(vb);
  EXPECT_TRUE(Cpp::GetVariableOffset(vb));
  // The second interpreter's first anchor is also its v0.
  EXPECT_TRUE(Cpp::GetNamed("__cppinterop_odr_use_v0"));
  EXPECT_FALSE(Cpp::GetNamed("__cppinterop_odr_use_v1"));
  EXPECT_TRUE(Cpp::DeleteInterpreter(I2));
}

#define CODE                                                                   \
  class BaseA {                                                                \
  public:                                                                      \
    virtual ~BaseA() {}                                                        \
    int a;                                                                     \
    BaseA(int a) : a(a) {}                                                     \
  };                                                                           \
                                                                               \
  class BaseB : public BaseA {                                                 \
  public:                                                                      \
    virtual ~BaseB() {}                                                        \
    std::string b;                                                             \
    BaseB(int x, std::string b) : BaseA(x), b(b) {}                            \
  };                                                                           \
                                                                               \
  class Base1 {                                                                \
  public:                                                                      \
    virtual ~Base1() {}                                                        \
    int i;                                                                     \
    std::string s;                                                             \
    Base1(int i, std::string s) : i(i), s(s) {}                                \
  };                                                                           \
                                                                               \
  class MyKlass : public BaseB, public Base1 {                                 \
  public:                                                                      \
    virtual ~MyKlass() {}                                                      \
    int k;                                                                     \
    MyKlass(int k, int i, int x, std::string b, std::string s)                 \
        : BaseB(x, b), Base1(i, s), k(k) {}                                    \
  } my_k(5, 4, 3, "Cpp", "Python");

CODE

TYPED_TEST(CPPINTEROP_TEST_MODE, VariableReflection_VariableOffsetsWithInheritance) {
#ifdef __EMSCRIPTEN__
  GTEST_SKIP() << "This test crashes for Emscripten builds of CppInterOp";
#endif
#if CLANG_VERSION_MAJOR == 20 && defined(CPPINTEROP_USE_CLING) && defined(_WIN32)
  GTEST_SKIP() << "Test fails with Cling on Windows";
#endif

  std::vector<const char*> interpreter_args = {"-include", "new"};
  TestFixture::CreateInterpreter(interpreter_args);

  Cpp::Declare("#include<string>");

#define Stringify(s) Stringifyx(s)
#define Stringifyx(...) #__VA_ARGS__
  Cpp::Declare(Stringify(CODE));
#undef Stringifyx
#undef Stringify
#undef CODE

  Cpp::DeclRef myklass = Cpp::GetNamed("MyKlass");
  EXPECT_TRUE(myklass);

  size_t num_bases = Cpp::GetNumBases(myklass);
  EXPECT_EQ(num_bases, 2);

  std::vector<Cpp::DeclRef> datamembers;
  Cpp::GetDatamembers(myklass, datamembers);
  for (size_t i = 0; i < num_bases; i++) {
    Cpp::DeclRef base = Cpp::GetBaseClass(myklass, i);
    EXPECT_TRUE(base);
    for (size_t i = 0; i < Cpp::GetNumBases(base); i++) {
      Cpp::DeclRef bbase = Cpp::GetBaseClass(base, i);
      EXPECT_TRUE(base);
      Cpp::GetDatamembers(bbase, datamembers);
    }
    Cpp::GetDatamembers(base, datamembers);
  }
  EXPECT_EQ(datamembers.size(), 5);

  EXPECT_EQ(Cpp::GetVariableOffset(datamembers[0], myklass),
            ((intptr_t)&(my_k.k)) - ((intptr_t)&(my_k)));

  EXPECT_EQ(Cpp::GetVariableOffset(datamembers[1], myklass),
            ((intptr_t)&(my_k.a)) - ((intptr_t)&(my_k)));

  EXPECT_EQ(Cpp::GetVariableOffset(datamembers[2], myklass),
            ((intptr_t)&(my_k.b)) - ((intptr_t)&(my_k)));

  EXPECT_EQ(Cpp::GetVariableOffset(datamembers[3], myklass),
            ((intptr_t)&(my_k.i)) - ((intptr_t)&(my_k)));

  EXPECT_EQ(Cpp::GetVariableOffset(datamembers[4], myklass),
            ((intptr_t)&(my_k.s)) - ((intptr_t)&(my_k)));
}

TYPED_TEST(CPPINTEROP_TEST_MODE, VariableReflection_IsPublicVariable) {
  std::vector<Decl *> Decls, SubDecls;
  std::string code = R"(
    class C {
    public:
      int a;
    private:
      int b;
    protected:
      int c;
      int sum(int,int);
    };
    )";

  GetAllTopLevelDecls(code, Decls);
  GetAllSubDecls(Decls[0], SubDecls);

  EXPECT_TRUE(Cpp::IsPublicVariable(SubDecls[2]));
  EXPECT_FALSE(Cpp::IsPublicVariable(SubDecls[4]));
  EXPECT_FALSE(Cpp::IsPublicVariable(SubDecls[6]));
  EXPECT_FALSE(Cpp::IsPublicVariable(SubDecls[7]));
}

TYPED_TEST(CPPINTEROP_TEST_MODE, VariableReflection_IsProtectedVariable) {
  std::vector<Decl *> Decls, SubDecls;
  std::string code = R"(
    class C {
    public:
      int a;
    private:
      int b;
    protected:
      int c;
    };
    )";

  GetAllTopLevelDecls(code, Decls);
  GetAllSubDecls(Decls[0], SubDecls);

  EXPECT_FALSE(Cpp::IsProtectedVariable(SubDecls[2]));
  EXPECT_FALSE(Cpp::IsProtectedVariable(SubDecls[4]));
  EXPECT_TRUE(Cpp::IsProtectedVariable(SubDecls[6]));
}

TYPED_TEST(CPPINTEROP_TEST_MODE, VariableReflection_IsPrivateVariable) {
  std::vector<Decl *> Decls, SubDecls;
  std::string code = R"(
    class C {
    public:
      int a;
    private:
      int b;
    protected:
      int c;
    };
    )";

  GetAllTopLevelDecls(code, Decls);
  GetAllSubDecls(Decls[0], SubDecls);

  EXPECT_FALSE(Cpp::IsPrivateVariable(SubDecls[2]));
  EXPECT_TRUE(Cpp::IsPrivateVariable(SubDecls[4]));
  EXPECT_FALSE(Cpp::IsPrivateVariable(SubDecls[6]));
}

TYPED_TEST(CPPINTEROP_TEST_MODE, VariableReflection_IsStaticVariable) {
  std::vector<Decl *> Decls, SubDecls;
  std::string code =  R"(
    class C {
      int a;
      static int b;
    };
    )";

  GetAllTopLevelDecls(code, Decls);
  GetAllSubDecls(Decls[0], SubDecls);

  EXPECT_FALSE(Cpp::IsStaticVariable(SubDecls[1]));
  EXPECT_TRUE(Cpp::IsStaticVariable(SubDecls[2]));
}

TYPED_TEST(CPPINTEROP_TEST_MODE, VariableReflection_IsConstVariable) {
  std::vector<Decl *> Decls, SubDecls;
  std::string code =  R"(
    class C {
      int a;
      const int b = 2;
    };
    )";

  GetAllTopLevelDecls(code, Decls);
  GetAllSubDecls(Decls[0], SubDecls);

  EXPECT_FALSE(Cpp::IsConstVariable(Decls[0]));
  EXPECT_FALSE(Cpp::IsConstVariable(SubDecls[1]));
  EXPECT_TRUE(Cpp::IsConstVariable(SubDecls[2]));
}

TYPED_TEST(CPPINTEROP_TEST_MODE,
           VariableReflection_DISABLED_GetArrayDimensions) {
  std::vector<Decl *> Decls;
  std::string code =  R"(
    int a;
    int b[1];
    int c[1][2];
    )";

  GetAllTopLevelDecls(code, Decls);

  //FIXME: is_vec_eq is an unused variable until test does something
  //auto is_vec_eq = [](const std::vector<size_t> &arr_dims,
  //                    const std::vector<size_t> &truth_vals) {
  //  if (arr_dims.size() != truth_vals.size())
  //    return false;
  //
  //  return std::equal(arr_dims.begin(), arr_dims.end(), truth_vals.begin());
  //};

  // EXPECT_TRUE(is_vec_eq(Cpp::GetArrayDimensions(Decls[0]), {}));
  // EXPECT_TRUE(is_vec_eq(Cpp::GetArrayDimensions(Decls[1]), {1}));
  // EXPECT_TRUE(is_vec_eq(Cpp::GetArrayDimensions(Decls[2]), {1,2}));
}

TYPED_TEST(CPPINTEROP_TEST_MODE, VariableReflection_StaticConstExprDatamember) {

#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif

  TestFixture::CreateInterpreter();

  Cpp::Declare(R"(
  class MyClass {
  public:
    static constexpr int x = 3;
  };

  template<int i>
  class MyTemplatedClass {
  public:
    static constexpr int x = i;
  };

  template<typename _Tp, _Tp __v>
  struct integral_constant
  {
      static constexpr _Tp value = __v;
  };

  template<typename... Eles>
  struct Elements
  : public integral_constant<int, sizeof...(Eles)> {};
  )");

  Cpp::DeclRef MyClass = Cpp::GetNamed("MyClass");
  EXPECT_TRUE(MyClass);

  std::vector<Cpp::DeclRef> datamembers;
  Cpp::GetStaticDatamembers(MyClass, datamembers);
  EXPECT_EQ(datamembers.size(), 1);

  intptr_t offset = Cpp::GetVariableOffset(datamembers[0]);
  EXPECT_EQ(3, *(size_t*)offset);

  ASTContext& C = Interp->getCI()->getASTContext();
  std::vector<Cpp::TemplateArgInfo> template_args = {
      {C.IntTy.getAsOpaquePtr(), "5"}};

  Cpp::DeclRef MyTemplatedClass = Cpp::InstantiateTemplate(
      Cpp::GetNamed("MyTemplatedClass"), template_args);
  EXPECT_TRUE(MyTemplatedClass);

  datamembers.clear();
  Cpp::GetStaticDatamembers(MyTemplatedClass, datamembers);
  EXPECT_EQ(datamembers.size(), 1);

  offset = Cpp::GetVariableOffset(datamembers[0]);
  EXPECT_EQ(5, *(size_t*)offset);

  std::vector<Cpp::TemplateArgInfo> ele_template_args = {
      {C.IntTy.getAsOpaquePtr()}, {C.FloatTy.getAsOpaquePtr()}};

  Cpp::DeclRef Elements =
      Cpp::InstantiateTemplate(Cpp::GetNamed("Elements"), ele_template_args);
  EXPECT_TRUE(Elements);

  EXPECT_EQ(1, Cpp::GetNumBases(Elements));

  Cpp::DeclRef IC = Cpp::GetBaseClass(Elements, 0);

  datamembers.clear();
  Cpp::GetStaticDatamembers(IC, datamembers);
  EXPECT_EQ(datamembers.size(), 1);

  offset = Cpp::GetVariableOffset(datamembers[0]);
  EXPECT_EQ(2, *(size_t*)offset);
}

TYPED_TEST(CPPINTEROP_TEST_MODE,
           VariableReflection_GetEnumConstantDatamembers) {
  TestFixture::CreateInterpreter();

  Cpp::Declare(R"(
  class MyEnumClass {
    enum { FOUR, FIVE, SIX };
    enum A { ONE, TWO, THREE };
    enum class B { SEVEN, EIGHT, NINE };
  };
  )");

  Cpp::DeclRef MyEnumClass = Cpp::GetNamed("MyEnumClass");
  EXPECT_TRUE(MyEnumClass);

  std::vector<Cpp::DeclRef> datamembers;
  Cpp::GetEnumConstantDatamembers(MyEnumClass, datamembers);
  EXPECT_EQ(datamembers.size(), 9);
  EXPECT_TRUE(Cpp::IsEnumType(Cpp::GetVariableType(datamembers[0])));

  std::vector<Cpp::DeclRef> datamembers2;
  Cpp::GetEnumConstantDatamembers(MyEnumClass, datamembers2, false);
  EXPECT_EQ(datamembers2.size(), 6);
}

TYPED_TEST(CPPINTEROP_TEST_MODE, VariableReflection_Is_Get_Pointer) {
  TestFixture::CreateInterpreter();
  std::vector<Decl*> Decls;
  std::string code = R"(
  class A {};
  int a;
  int *b;
  double c;
  double *d;
  A e;
  A *f;
  )";

  GetAllTopLevelDecls(code, Decls);

  EXPECT_FALSE(Cpp::IsPointerType(Cpp::GetVariableType(Decls[1])));
  EXPECT_TRUE(Cpp::IsPointerType(Cpp::GetVariableType(Decls[2])));
  EXPECT_FALSE(Cpp::IsPointerType(Cpp::GetVariableType(Decls[3])));
  EXPECT_TRUE(Cpp::IsPointerType(Cpp::GetVariableType(Decls[4])));
  EXPECT_FALSE(Cpp::IsPointerType(Cpp::GetVariableType(Decls[5])));
  EXPECT_TRUE(Cpp::IsPointerType(Cpp::GetVariableType(Decls[6])));

  EXPECT_EQ(Cpp::GetPointeeType(Cpp::GetVariableType(Decls[2])),
            Cpp::GetVariableType(Decls[1]));
  EXPECT_EQ(Cpp::GetPointeeType(Cpp::GetVariableType(Decls[4])),
            Cpp::GetVariableType(Decls[3]));
  EXPECT_EQ(Cpp::GetPointeeType(Cpp::GetVariableType(Decls[6])),
            Cpp::GetVariableType(Decls[5]));

  EXPECT_FALSE(Cpp::GetPointeeType(Cpp::GetVariableType(Decls[5])));
}

TYPED_TEST(CPPINTEROP_TEST_MODE, VariableReflection_Is_Get_Reference) {
  TestFixture::CreateInterpreter();
  std::vector<Decl*> Decls;
  std::string code = R"(
  class A {};
  int a;
  int &b = a;
  double c;
  double &d = c;
  A e;
  A &f = e;
  )";

  GetAllTopLevelDecls(code, Decls);

  EXPECT_FALSE(Cpp::IsReferenceType(Cpp::GetVariableType(Decls[1])));
  EXPECT_TRUE(Cpp::IsReferenceType(Cpp::GetVariableType(Decls[2])));
  EXPECT_FALSE(Cpp::IsReferenceType(Cpp::GetVariableType(Decls[3])));
  EXPECT_TRUE(Cpp::IsReferenceType(Cpp::GetVariableType(Decls[4])));
  EXPECT_FALSE(Cpp::IsReferenceType(Cpp::GetVariableType(Decls[5])));
  EXPECT_TRUE(Cpp::IsReferenceType(Cpp::GetVariableType(Decls[6])));

  EXPECT_EQ(Cpp::GetNonReferenceType(Cpp::GetVariableType(Decls[2])),
            Cpp::GetVariableType(Decls[1]));
  EXPECT_EQ(Cpp::GetNonReferenceType(Cpp::GetVariableType(Decls[4])),
            Cpp::GetVariableType(Decls[3]));
  EXPECT_EQ(Cpp::GetNonReferenceType(Cpp::GetVariableType(Decls[6])),
            Cpp::GetVariableType(Decls[5]));

  EXPECT_FALSE(Cpp::GetNonReferenceType(Cpp::GetVariableType(Decls[5])));

  EXPECT_EQ(Cpp::GetValueKind(Cpp::GetVariableType(Decls[2])),
            Cpp::ValueKind::LValue);
  EXPECT_EQ(Cpp::GetReferencedType(Cpp::GetVariableType(Decls[1])),
            Cpp::GetVariableType(Decls[2]));
  EXPECT_EQ(Cpp::GetValueKind(
                Cpp::GetReferencedType(Cpp::GetVariableType(Decls[1]), true)),
            Cpp::ValueKind::RValue);
  EXPECT_EQ(Cpp::GetValueKind(Cpp::GetVariableType(Decls[1])),
            Cpp::ValueKind::None);
}

TYPED_TEST(CPPINTEROP_TEST_MODE, VariableReflection_GetPointerType) {
  TestFixture::CreateInterpreter();
  std::vector<Decl*> Decls;
  std::string code = R"(
  class A {};
  int a;
  int *b = &a;
  double c;
  double *d = &c;
  A e;
  A *f = &e;
  )";

  GetAllTopLevelDecls(code, Decls);

  EXPECT_EQ(Cpp::GetPointerType(Cpp::GetVariableType(Decls[1])),
            Cpp::GetVariableType(Decls[2]));
  EXPECT_EQ(Cpp::GetPointerType(Cpp::GetVariableType(Decls[3])),
            Cpp::GetVariableType(Decls[4]));
  EXPECT_EQ(Cpp::GetPointerType(Cpp::GetVariableType(Decls[5])),
            Cpp::GetVariableType(Decls[6]));
}
