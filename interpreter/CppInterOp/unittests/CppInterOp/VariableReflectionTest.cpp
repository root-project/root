#include "Utils.h"

#include "CppInterOp/CppInterOp.h"

#include "clang/AST/ASTContext.h"
#include "clang/Basic/Version.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Sema.h"

#include "gtest/gtest.h"
#include <string>

#include <cstddef>

using namespace TestUtils;
using namespace llvm;
using namespace clang;

TEST(VariableReflectionTest, GetDatamembers) {
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

  std::vector<Cpp::TCppScope_t> datamembers;
  std::vector<Cpp::TCppScope_t> datamembers1;
  std::vector<Cpp::TCppScope_t> datamembers2;
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

TEST(VariableReflectionTest, DatamembersWithAnonymousStructOrUnion) {
  if (llvm::sys::RunningOnValgrind())
    GTEST_SKIP() << "XFAIL due to Valgrind report";

  std::vector<Decl*> Decls;
#define Stringify(s) Stringifyx(s)
#define Stringifyx(...) #__VA_ARGS__
  GetAllTopLevelDecls(Stringify(CODE), Decls);
#undef Stringifyx
#undef Stringify
#undef CODE

  std::vector<Cpp::TCppScope_t> datamembers_klass1;
  std::vector<Cpp::TCppScope_t> datamembers_klass2;
  std::vector<Cpp::TCppScope_t> datamembers_klass3;

  Cpp::GetDatamembers(Decls[0], datamembers_klass1);
  Cpp::GetDatamembers(Decls[2], datamembers_klass2);
  Cpp::GetDatamembers(Decls[4], datamembers_klass3);

  EXPECT_EQ(datamembers_klass1.size(), 3);
  EXPECT_EQ(datamembers_klass2.size(), 3);

  EXPECT_EQ(Cpp::GetVariableOffset(datamembers_klass1[0]), 0);
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
#ifdef _WIN32
#pragma warning(default : 4201)
#endif
}

TEST(VariableReflectionTest, GetTypeAsString) {
  if (llvm::sys::RunningOnValgrind())
    GTEST_SKIP() << "XFAIL due to Valgrind report";

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

  Cpp::CreateInterpreter();
  EXPECT_EQ(Cpp::Declare(code.c_str()), 0);

  Cpp::TCppScope_t wrapper =
      Cpp::GetScopeFromCompleteName("my_namespace::Wrapper");
  EXPECT_TRUE(wrapper);

  std::vector<Cpp::TCppScope_t> datamembers;
  Cpp::GetDatamembers(wrapper, datamembers);
  EXPECT_EQ(datamembers.size(), 1);

  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetVariableType(datamembers[0])),
            "my_namespace::Container");
}

TEST(VariableReflectionTest, LookupDatamember) {
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

TEST(VariableReflectionTest, GetVariableType) {
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
    )";

  GetAllTopLevelDecls(code, Decls);

  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetVariableType(Decls[2])), "int");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetVariableType(Decls[3])), "char");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetVariableType(Decls[4])), "C");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetVariableType(Decls[5])), "C *");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetVariableType(Decls[6])), "E<int>");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetVariableType(Decls[7])), "E<int> *");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetVariableType(Decls[8])), "int[4]");
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

TEST(VariableReflectionTest, GetVariableOffset) {
#ifdef EMSCRIPTEN
#if CLANG_VERSION_MAJOR < 20
  GTEST_SKIP() << "Test fails for Emscipten builds";
#endif
#endif
  std::vector<Decl *> Decls;
#define Stringify(s) Stringifyx(s)
#define Stringifyx(...) #__VA_ARGS__
  GetAllTopLevelDecls(Stringify(CODE), Decls);
#undef Stringifyx
#undef Stringify
#undef CODE

  EXPECT_EQ(7, Decls.size());

  std::vector<Cpp::TCppScope_t> datamembers;
  Cpp::GetDatamembers(Decls[4], datamembers);

  EXPECT_TRUE((bool) Cpp::GetVariableOffset(Decls[0])); // a
  EXPECT_TRUE((bool) Cpp::GetVariableOffset(Decls[1])); // N
  EXPECT_TRUE((bool)Cpp::GetVariableOffset(Decls[2]));  // S
  EXPECT_TRUE((bool)Cpp::GetVariableOffset(Decls[3]));  // SN

  EXPECT_EQ(Cpp::GetVariableOffset(datamembers[0]), 0);

  EXPECT_EQ(Cpp::GetVariableOffset(datamembers[1]),
          ((intptr_t) &(c.b)) - ((intptr_t) &(c.a)));
  EXPECT_EQ(Cpp::GetVariableOffset(datamembers[2]),
          ((intptr_t) &(c.c)) - ((intptr_t) &(c.a)));
  EXPECT_EQ(Cpp::GetVariableOffset(datamembers[3]),
          ((intptr_t) &(c.d)) - ((intptr_t) &(c.a)));

  auto* VD_C_s_a = Cpp::GetNamed("s_a", Decls[4]); // C::s_a
  EXPECT_TRUE((bool) Cpp::GetVariableOffset(VD_C_s_a));

  struct K {
    int x;
    int y;
    int z;
  };
  Cpp::Declare("struct K;");
  Cpp::TCppScope_t k = Cpp::GetNamed("K");
  EXPECT_TRUE(k);

  Cpp::Declare("struct K { int x; int y; int z; };");

  datamembers.clear();
  Cpp::GetDatamembers(k, datamembers);
  EXPECT_EQ(datamembers.size(), 3);

  EXPECT_EQ(Cpp::GetVariableOffset(datamembers[0]), offsetof(K, x));
  EXPECT_EQ(Cpp::GetVariableOffset(datamembers[1]), offsetof(K, y));
  EXPECT_EQ(Cpp::GetVariableOffset(datamembers[2]), offsetof(K, z));
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

TEST(VariableReflectionTest, VariableOffsetsWithInheritance) {
#if CLANG_VERSION_MAJOR == 18 && defined(CPPINTEROP_USE_CLING) &&              \
    defined(_WIN32) && (defined(_M_ARM) || defined(_M_ARM64))
  GTEST_SKIP() << "Test fails with Cling on Windows on ARM";
#endif
  if (llvm::sys::RunningOnValgrind())
    GTEST_SKIP() << "XFAIL due to Valgrind report";

  std::vector<const char*> interpreter_args = {"-include", "new"};
  Cpp::CreateInterpreter(interpreter_args);

  Cpp::Declare("#include<string>");

#define Stringify(s) Stringifyx(s)
#define Stringifyx(...) #__VA_ARGS__
  Cpp::Declare(Stringify(CODE));
#undef Stringifyx
#undef Stringify
#undef CODE

  Cpp::TCppScope_t myklass = Cpp::GetNamed("MyKlass");
  EXPECT_TRUE(myklass);

  size_t num_bases = Cpp::GetNumBases(myklass);
  EXPECT_EQ(num_bases, 2);

  std::vector<Cpp::TCppScope_t> datamembers;
  Cpp::GetDatamembers(myklass, datamembers);
  for (size_t i = 0; i < num_bases; i++) {
    Cpp::TCppScope_t base = Cpp::GetBaseClass(myklass, i);
    EXPECT_TRUE(base);
    for (size_t i = 0; i < Cpp::GetNumBases(base); i++) {
      Cpp::TCppScope_t bbase = Cpp::GetBaseClass(base, i);
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

TEST(VariableReflectionTest, IsPublicVariable) {
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

TEST(VariableReflectionTest, IsProtectedVariable) {
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

TEST(VariableReflectionTest, IsPrivateVariable) {
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

TEST(VariableReflectionTest, IsStaticVariable) {
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

TEST(VariableReflectionTest, IsConstVariable) {
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

TEST(VariableReflectionTest, DISABLED_GetArrayDimensions) {
  std::vector<Decl *> Decls, SubDecls;
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

TEST(VariableReflectionTest, StaticConstExprDatamember) {
  if (llvm::sys::RunningOnValgrind())
    GTEST_SKIP() << "XFAIL due to Valgrind report";

#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif

  Cpp::CreateInterpreter();

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

  Cpp::TCppScope_t MyClass = Cpp::GetNamed("MyClass");
  EXPECT_TRUE(MyClass);

  std::vector<Cpp::TCppScope_t> datamembers;
  Cpp::GetStaticDatamembers(MyClass, datamembers);
  EXPECT_EQ(datamembers.size(), 1);

  intptr_t offset = Cpp::GetVariableOffset(datamembers[0]);
  EXPECT_EQ(3, *(size_t*)offset);

  ASTContext& C = Interp->getCI()->getASTContext();
  std::vector<Cpp::TemplateArgInfo> template_args = {
      {C.IntTy.getAsOpaquePtr(), "5"}};

  Cpp::TCppFunction_t MyTemplatedClass =
      Cpp::InstantiateTemplate(Cpp::GetNamed("MyTemplatedClass"),
                               template_args.data(), template_args.size());
  EXPECT_TRUE(MyTemplatedClass);

  datamembers.clear();
  Cpp::GetStaticDatamembers(MyTemplatedClass, datamembers);
  EXPECT_EQ(datamembers.size(), 1);

  offset = Cpp::GetVariableOffset(datamembers[0]);
  EXPECT_EQ(5, *(size_t*)offset);

  std::vector<Cpp::TemplateArgInfo> ele_template_args = {
      {C.IntTy.getAsOpaquePtr()}, {C.FloatTy.getAsOpaquePtr()}};

  Cpp::TCppFunction_t Elements = Cpp::InstantiateTemplate(
      Cpp::GetNamed("Elements"), ele_template_args.data(),
      ele_template_args.size());
  EXPECT_TRUE(Elements);

  EXPECT_EQ(1, Cpp::GetNumBases(Elements));

  Cpp::TCppScope_t IC = Cpp::GetBaseClass(Elements, 0);

  datamembers.clear();
  Cpp::GetStaticDatamembers(IC, datamembers);
  EXPECT_EQ(datamembers.size(), 1);

  offset = Cpp::GetVariableOffset(datamembers[0]);
  EXPECT_EQ(2, *(size_t*)offset);
}

TEST(VariableReflectionTest, GetEnumConstantDatamembers) {
  Cpp::CreateInterpreter();

  Cpp::Declare(R"(
  class MyEnumClass {
    enum { FOUR, FIVE, SIX };
    enum A { ONE, TWO, THREE };
    enum class B { SEVEN, EIGHT, NINE };
  };
  )");

  Cpp::TCppScope_t MyEnumClass = Cpp::GetNamed("MyEnumClass");
  EXPECT_TRUE(MyEnumClass);

  std::vector<Cpp::TCppScope_t> datamembers;
  Cpp::GetEnumConstantDatamembers(MyEnumClass, datamembers);
  EXPECT_EQ(datamembers.size(), 9);
  EXPECT_TRUE(Cpp::IsEnumType(Cpp::GetVariableType(datamembers[0])));

  std::vector<Cpp::TCppScope_t> datamembers2;
  Cpp::GetEnumConstantDatamembers(MyEnumClass, datamembers2, false);
  EXPECT_EQ(datamembers2.size(), 6);
}

TEST(VariableReflectionTest, Is_Get_Pointer) {
  Cpp::CreateInterpreter();
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

TEST(VariableReflectionTest, Is_Get_Reference) {
  Cpp::CreateInterpreter();
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

  EXPECT_TRUE(Cpp::IsLValueReferenceType(Cpp::GetVariableType(Decls[2])));
  EXPECT_EQ(Cpp::GetReferencedType(Cpp::GetVariableType(Decls[1])),
            Cpp::GetVariableType(Decls[2]));
  EXPECT_TRUE(Cpp::IsRValueReferenceType(
      Cpp::GetReferencedType(Cpp::GetVariableType(Decls[1]), true)));
}

TEST(VariableReflectionTest, GetPointerType) {
  Cpp::CreateInterpreter();
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
