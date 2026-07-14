#include "Utils.h"

#include "CppInterOp/CppInterOp.h"

#include "clang/AST/ASTContext.h"
#include "clang/Basic/Version.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Sema.h"

#include <CppInterOp/CppInterOpTypes.h>
#include <llvm/ADT/ArrayRef.h>

#include "gtest/gtest.h"

#include <array>
#include <string>
#include <vector>

using namespace TestUtils;
using namespace llvm;
using namespace clang;

// Reusable empty template args vector. In the dispatch mode, passing an empty
// initializer list {} does not work since the compiler cannot deduce the type
// for a function pointer
static const std::vector<Cpp::TemplateArgInfo> empty_templ_args = {};

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_GetClassMethods) {
  std::vector<Decl*> Decls;
  std::string code = R"(
    class A;

    class A {
    public:
      int f1(int a, int b) { return a + b; }
      const A *f2() const { return this; }
    private:
      int f3() { return 0; }
      void f4() {}
    protected:
      int f5(int i) { return i; }
    };

    typedef A shadow_A;
    class B {
    public:
        B(int n) : b{n} {}
        int b;
    };

    class C: public B {
    public:
        using B::B;
    };
    using MyInt = int;
    )";

  GetAllTopLevelDecls(code, Decls);

  auto get_method_name = [](Cpp::FuncRef method) {
    return Cpp::GetFunctionSignature(method);
  };

  std::vector<Cpp::FuncRef> methods0;
  Cpp::GetClassMethods(Decls[0], methods0);

  EXPECT_EQ(methods0.size(), 11);
  EXPECT_EQ(get_method_name(methods0[0]), "int A::f1(int a, int b)");
  EXPECT_EQ(get_method_name(methods0[1]), "const A *A::f2() const");
  EXPECT_EQ(get_method_name(methods0[2]), "int A::f3()");
  EXPECT_EQ(get_method_name(methods0[3]), "void A::f4()");
  EXPECT_EQ(get_method_name(methods0[4]), "int A::f5(int i)");
  EXPECT_EQ(get_method_name(methods0[5]), "inline constexpr A::A()");
  EXPECT_EQ(get_method_name(methods0[6]), "inline constexpr A::A(const A &)");
  EXPECT_EQ(get_method_name(methods0[7]), "inline constexpr A &A::operator=(const A &)");
  EXPECT_EQ(get_method_name(methods0[8]), "inline constexpr A::A(A &&)");
  EXPECT_EQ(get_method_name(methods0[9]), "inline constexpr A &A::operator=(A &&)");
  EXPECT_EQ(get_method_name(methods0[10]), "inline A::~A()");

  std::vector<Cpp::FuncRef> methods1;
  Cpp::GetClassMethods(Decls[2], methods1);
  EXPECT_EQ(methods0.size(), methods1.size());
  EXPECT_EQ(methods0[0], methods1[0]);
  EXPECT_EQ(methods0[1], methods1[1]);
  EXPECT_EQ(methods0[2], methods1[2]);
  EXPECT_EQ(methods0[3], methods1[3]);
  EXPECT_EQ(methods0[4], methods1[4]);

  std::vector<Cpp::FuncRef> methods2;
  Cpp::GetClassMethods(Decls[3], methods2);

  EXPECT_EQ(methods2.size(), 6);
  EXPECT_EQ(get_method_name(methods2[0]), "B::B(int n)");
  EXPECT_EQ(get_method_name(methods2[1]), "inline constexpr B::B(const B &)");
  EXPECT_EQ(get_method_name(methods2[2]), "inline constexpr B::B(B &&)");
  EXPECT_EQ(get_method_name(methods2[3]), "inline B::~B()");
  EXPECT_EQ(get_method_name(methods2[4]), "inline B &B::operator=(const B &)");
  EXPECT_EQ(get_method_name(methods2[5]), "inline B &B::operator=(B &&)");

  std::vector<Cpp::FuncRef> methods3;
  Cpp::GetClassMethods(Decls[4], methods3);

  EXPECT_EQ(methods3.size(), 9);
  EXPECT_EQ(get_method_name(methods3[0]), "inline C::C()");
  EXPECT_EQ(get_method_name(methods3[1]), "inline constexpr C::C(const C &)");
  EXPECT_EQ(get_method_name(methods3[2]), "inline constexpr C::C(C &&)");
  EXPECT_EQ(get_method_name(methods3[3]), "inline C &C::operator=(const C &)");
  EXPECT_EQ(get_method_name(methods3[4]), "inline C &C::operator=(C &&)");
  EXPECT_EQ(get_method_name(methods3[5]), "inline C::~C()");
  EXPECT_EQ(get_method_name(methods3[6]), "inline C::B(int)");
  EXPECT_EQ(get_method_name(methods3[7]), "inline constexpr C::B(const B &)");

  // Should not crash.
  std::vector<Cpp::FuncRef> methods4;
  Cpp::GetClassMethods(Decls[5], methods4);
  EXPECT_EQ(methods4.size(), 0);

  std::vector<Cpp::FuncRef> methods5;
  Cpp::GetClassMethods(nullptr, methods5);
  EXPECT_EQ(methods5.size(), 0);

  Decls.clear();
  code = R"(
    struct T {
    template<typename Tmpl>
    T(Tmpl x) : n(x) {}
    T(const T&) = delete;
    T(T&&) = delete;
    void fn() {}
    int n;
    };

    struct TT: public T {
      using T::T;
      using T::fn;
    };
    )";

  GetAllTopLevelDecls(code, Decls);
  EXPECT_EQ(Decls.size(), 2);

  std::vector<Cpp::FuncRef> templ_methods1;
  Cpp::GetClassMethods(Decls[0], templ_methods1);
  EXPECT_EQ(templ_methods1.size(), 5);
  EXPECT_EQ(get_method_name(templ_methods1[0]), "T::T(const T &) = delete");
  EXPECT_EQ(get_method_name(templ_methods1[1]), "T::T(T &&) = delete");
  EXPECT_EQ(get_method_name(templ_methods1[2]), "void T::fn()");
  EXPECT_EQ(get_method_name(templ_methods1[3]),
            "inline T &T::operator=(const T &)");
  EXPECT_EQ(get_method_name(templ_methods1[4]), "inline T::~T()");

  std::vector<Cpp::FuncRef> templ_methods2;
  Cpp::GetClassMethods(Decls[1], templ_methods2);
  EXPECT_EQ(templ_methods2.size(), 7);
  EXPECT_EQ(get_method_name(templ_methods2[0]), "void T::fn()");
  EXPECT_EQ(get_method_name(templ_methods2[1]), "inline TT::TT()");
  EXPECT_EQ(get_method_name(templ_methods2[2]), "inline TT::TT(const TT &)");
  EXPECT_EQ(get_method_name(templ_methods2[3]), "inline TT::TT(TT &&)");
  EXPECT_EQ(get_method_name(templ_methods2[4]),
            "inline TT &TT::operator=(const TT &)");
  EXPECT_EQ(get_method_name(templ_methods2[5]),
            "inline TT &TT::operator=(TT &&)");
  EXPECT_EQ(get_method_name(templ_methods2[6]), "inline TT::~TT()");
}

TYPED_TEST(CPPINTEROP_TEST_MODE,
           FunctionReflection_ConstructorInGetClassMethods) {
  std::vector<Decl*> Decls;
  std::string code = R"(
    struct S {
      int a;
      int b;
    };
    )";

  GetAllTopLevelDecls(code, Decls);

  auto has_constructor = [](Decl* D) {
    std::vector<Cpp::FuncRef> methods;
    Cpp::GetClassMethods(D, methods);
    for (auto method : methods) {
      if (Cpp::IsConstructor(method))
        return true;
    }
    return false;
  };

  EXPECT_TRUE(has_constructor(Decls[0]));
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_HasDefaultConstructor) {
  std::vector<Decl*> Decls;
  std::string code = R"(
    class A {
      private:
        int n;
    };
    class B {
      private:
        int n;
      public:
      B() : n(1) {}
    };
    class C {
      private:
        int n;
      public:
        C() = delete;
        C(int i) : n(i) {}
    };
    int sum(int a, int b){
      return a+b;
    }

    )";

  GetAllTopLevelDecls(code, Decls);
  EXPECT_TRUE(Cpp::HasDefaultConstructor(Decls[0]));
  EXPECT_TRUE(Cpp::HasDefaultConstructor(Decls[1]));
  EXPECT_FALSE(Cpp::HasDefaultConstructor(Decls[3]));
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_GetDestructor) {
  std::vector<Decl*> Decls;
  std::string code = R"(
    class A {
    };
    class B {
      public:
      ~B() {}
    };
    class C {
      public:
      ~C() = delete;
    };
    int sum(int a, int b) {
      return a+b;
    }
    )";

  GetAllTopLevelDecls(code, Decls);

  EXPECT_TRUE(Cpp::GetDestructor(Decls[0]));
  EXPECT_TRUE(Cpp::GetDestructor(Decls[1]));
  auto DeletedDtor = Cpp::GetDestructor(Decls[2]);
  EXPECT_TRUE(DeletedDtor);
  EXPECT_TRUE(Cpp::IsFunctionDeleted(DeletedDtor));
  EXPECT_FALSE(Cpp::GetDestructor(Decls[3]));
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_GetFunctionsUsingName) {
  std::vector<Decl*> Decls;
  std::string code = R"(
    class A {
    public:
      int f1(int a, int b) { return a + b; }
      int f1(int a) { return f1(a, 10); }
      int f1() { return f1(10); }
    private:
      int f2() { return 0; }
    protected:
      int f3(int i) { return i; }
    };

    namespace N {
      int f4(int a) { return a + 1; }
      int f4() { return 0; }
    }

    typedef A shadow_A;

    class Base {
    public:
      int g(int a) { return a; }
      int g() { return 0; }
    };

    class Derived : public Base {
    public:
      using Base::g;
      int h() { return 0; }
    };
    )";

  GetAllTopLevelDecls(code, Decls);

  // This lambda can take in the scope and the name of the function
  // and returns the size of the vector returned by GetFunctionsUsingName
  auto get_number_of_funcs_using_name = [&](Cpp::DeclRef scope,
                                            const std::string& name) {
    auto Funcs = Cpp::GetFunctionsUsingName(scope, name);

    return Funcs.size();
  };

  EXPECT_EQ(get_number_of_funcs_using_name(Decls[0], "f1"), 3);
  EXPECT_EQ(get_number_of_funcs_using_name(Decls[0], "f2"), 1);
  EXPECT_EQ(get_number_of_funcs_using_name(Decls[0], "f3"), 1);
  EXPECT_EQ(get_number_of_funcs_using_name(Decls[0], "f4"), 0);
  EXPECT_EQ(get_number_of_funcs_using_name(Decls[1], "f4"), 2);
  EXPECT_EQ(get_number_of_funcs_using_name(Decls[2], "f1"), 3);
  EXPECT_EQ(get_number_of_funcs_using_name(Decls[2], "f2"), 1);
  EXPECT_EQ(get_number_of_funcs_using_name(Decls[2], "f3"), 1);
  EXPECT_EQ(get_number_of_funcs_using_name(Decls[2], ""), 0);

  Cpp::DeclRef derived = Cpp::GetScope("Derived");
  EXPECT_TRUE(derived);
  EXPECT_EQ(get_number_of_funcs_using_name(derived, "g"), 2);
  EXPECT_EQ(get_number_of_funcs_using_name(derived, "h"), 1);
}

TYPED_TEST(CPPINTEROP_TEST_MODE,
           FunctionReflection_GetFunctionsUsingNameOperators) {
  std::vector<Decl*> Decls;
  std::string code = R"(
    namespace ComparableSpace {
      class NSComparable {};
      bool operator==(const NSComparable&, const NSComparable&) { return true; }
      bool operator==(const NSComparable&, int) { return false; }
      bool operator!=(const NSComparable&, const NSComparable&) { return false; }
      int operators_count() { return 1; }
    }

    struct WithOps {
      bool operator==(const WithOps&) const { return true; }
      WithOps& operator+=(int) { return *this; }
      int operator[](int) const { return 0; }
      int operator()(int, int) const { return 0; }
    };
    )";

  GetAllTopLevelDecls(code, Decls);

  auto count = [](Cpp::DeclRef scope, const std::string& name) {
    return Cpp::GetFunctionsUsingName(scope, name).size();
  };

  // Namespace-scope operator overloads are now findable by name.
  EXPECT_EQ(count(Decls[0], "operator=="), 2);
  EXPECT_EQ(count(Decls[0], "operator!="), 1);

  // Non-existent operator at this scope.
  EXPECT_EQ(count(Decls[0], "operator+"), 0);

  // "operator" followed by a non-identifier char that is not a valid operator
  // spelling: this passes the early identifier-path check but matches no
  // overloaded operator, so it must fall through to the empty DeclarationName
  // and resolve (to nothing) via the identifier path.
  EXPECT_EQ(count(Decls[0], "operator@"), 0);

  // Identifiers that merely start with "operator" must still resolve via the
  // identifier lookup path, not be misread as an operator.
  EXPECT_EQ(count(Decls[0], "operators_count"), 1);

  // Class-scope operators, including multi-token "()" and "[]".
  EXPECT_EQ(count(Decls[1], "operator=="), 1);
  EXPECT_EQ(count(Decls[1], "operator+="), 1);
  EXPECT_EQ(count(Decls[1], "operator[]"), 1);
  EXPECT_EQ(count(Decls[1], "operator()"), 1);
  EXPECT_EQ(count(Decls[1], "operator!="), 0);

  // Sanity-check that the returned decls are the operator overloads.
  auto eqs = Cpp::GetFunctionsUsingName(Decls[0], "operator==");
  ASSERT_EQ(eqs.size(), 2U);
  for (auto f : eqs)
    EXPECT_EQ(Cpp::GetName(Cpp::DeclRef{f.data}), "operator==");
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_GetClassDecls) {
  std::vector<Decl*> Decls, SubDecls;
  std::string code = R"(
    class MyTemplatedMethodClass {
      template<class A, class B>
      long get_size(A, B, int i = 0) {}

      template<class A = int, class B = char>
      long get_float_size(int i, A a = A(), B b = B()) {}

      template<class A>
      void get_char_size(long k, A, char ch = 'a', double l = 0.0) {}

      void f1() {}
      void f2(int i, double d, long l, char ch) {}

      template<class A>
      void get_size(long k, A, char ch = 'a', double l = 0.0) {}

      void f3(int i, double d, long l = 0, char ch = 'a') {}
      void f4(int i = 0, double d = 0.0, long l = 0, char ch = 'a') {}
    };
    )";

  GetAllTopLevelDecls(code, Decls);
  GetAllSubDecls(Decls[0], SubDecls);

  std::vector<Cpp::FuncRef> methods;
  Cpp::GetClassMethods(Decls[0], methods);

  EXPECT_EQ(methods.size(), 10); // includes structors and operators
  EXPECT_EQ(Cpp::GetName(Cpp::DeclRef{methods[0].data}),
            Cpp::GetName(SubDecls[4]));
  EXPECT_EQ(Cpp::GetName(Cpp::DeclRef{methods[1].data}),
            Cpp::GetName(SubDecls[5]));
  EXPECT_EQ(Cpp::GetName(Cpp::DeclRef{methods[2].data}),
            Cpp::GetName(SubDecls[7]));
  EXPECT_EQ(Cpp::GetName(Cpp::DeclRef{methods[3].data}),
            Cpp::GetName(SubDecls[8]));

  code = "class ForwardOnly;";
  methods.clear();
  Decls.clear();
  GetAllTopLevelDecls(code, Decls);
  Cpp::GetClassMethods(Decls[0], methods);
  EXPECT_EQ(methods.size(), 0);

  code = R"(template<class T>
    class Klass;
    template<>
    class Klass<int>;)";
  methods.clear();
  Decls.clear();
  GetAllTopLevelDecls(code, Decls);
  Cpp::GetClassMethods(Decls[1], methods);
  EXPECT_EQ(methods.size(), 0);

  code = R"(template<typename T>
    class Klass {
      T value;
    };
    using KlassInt = Klass<int>;)";
  methods.clear();
  Decls.clear();
  GetAllTopLevelDecls(code, Decls);
  Cpp::GetClassMethods(Decls[1], methods);
  EXPECT_EQ(methods.size(), 6); // 6 implicit created
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_GetFunctionTemplatedDecls) {
  std::vector<Decl*> Decls, SubDecls;
  std::string code = R"(
    class MyTemplatedMethodClass {
      template<class A, class B>
      long get_size(A, B, int i = 0) {}

      template<class A = int, class B = char>
      long get_float_size(int i, A a = A(), B b = B()) {}
           
      template<class A>
      void get_char_size(long k, A, char ch = 'a', double l = 0.0) {}

      void f1() {}
      void f2(int i, double d, long l, char ch) {}

      template<class A>
      void get_size(long k, A, char ch = 'a', double l = 0.0) {}

      void f3(int i, double d, long l = 0, char ch = 'a') {}
      void f4(int i = 0, double d = 0.0, long l = 0, char ch = 'a') {}
    };
    )";

  GetAllTopLevelDecls(code, Decls);
  GetAllSubDecls(Decls[0], SubDecls);

  std::vector<Cpp::FuncRef> template_methods;
  Cpp::GetFunctionTemplatedDecls(Decls[0], template_methods);

  EXPECT_EQ(template_methods.size(), 4);
  EXPECT_EQ(Cpp::GetName(Cpp::DeclRef{template_methods[0].data}),
            Cpp::GetName(SubDecls[1]));
  EXPECT_EQ(Cpp::GetName(Cpp::DeclRef{template_methods[1].data}),
            Cpp::GetName(SubDecls[2]));
  EXPECT_EQ(Cpp::GetName(Cpp::DeclRef{template_methods[2].data}),
            Cpp::GetName(SubDecls[3]));
  EXPECT_EQ(Cpp::GetName(Cpp::DeclRef{template_methods[3].data}),
            Cpp::GetName(SubDecls[6]));
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_GetFunctionReturnType) {
  std::vector<Decl*> Decls, SubDecls, TemplateSubDecls;
  std::string code = R"(
    namespace N { class C {}; }
    enum Switch { OFF, ON };

    class A {
      A (int i) { i++; }
      int f () { return 0; }
    };

    void f1() {}
    double f2() { return 0.2; }
    Switch f3() { return ON; }
    N::C f4() { return N::C(); }
    N::C *f5() { return new N::C(); }
    const N::C f6() { return N::C(); }
    volatile N::C f7() { return N::C(); }
    const volatile N::C f8() { return N::C(); }
    int n;

    class MyTemplatedMethodClass {
      template<class A>
      char get_string(A) {
          return 'A';
      }

      template<class A>
      void get_size() {}

      template<class A>
      long add_size (int i) {
          return sizeof(A) + i;
      }
    };

    template<class ...T> struct RTTest_TemplatedList {};
    template<class ...T> auto rttest_make_tlist(T ... args) {
      return RTTest_TemplatedList<T...>{};
    }

    template<typename T>
    struct Klass {
      auto func(T t) { return t; }
    };
    )";

  GetAllTopLevelDecls(code, Decls, true);
  GetAllSubDecls(Decls[2], SubDecls);
  GetAllSubDecls(Decls[12], TemplateSubDecls);

  // #include <string>

  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionReturnType(Decls[3])), "void");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionReturnType(Decls[4])),
            "double");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionReturnType(Decls[5])),
            "Switch");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionReturnType(Decls[6])), "N::C");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionReturnType(Decls[7])),
            "N::C *");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionReturnType(Decls[8])),
            "const N::C");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionReturnType(Decls[9])),
            "volatile N::C");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionReturnType(Decls[10])),
            "const volatile N::C");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionReturnType(Decls[11])),
            "NULL TYPE");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionReturnType(SubDecls[1])), "void");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionReturnType(SubDecls[2])), "int");
  EXPECT_EQ(
      Cpp::GetTypeAsString(Cpp::GetFunctionReturnType(TemplateSubDecls[1])),
      "char");
  EXPECT_EQ(
      Cpp::GetTypeAsString(Cpp::GetFunctionReturnType(TemplateSubDecls[2])),
      "void");
  EXPECT_EQ(
      Cpp::GetTypeAsString(Cpp::GetFunctionReturnType(TemplateSubDecls[3])),
      "long");

  ASTContext& C = Interp->getCI()->getASTContext();
  std::vector<Cpp::TemplateArgInfo> args = {C.IntTy.getAsOpaquePtr(),
                                            C.DoubleTy.getAsOpaquePtr()};
  std::vector<Cpp::TemplateArgInfo> explicit_args;
  std::vector<Cpp::FuncRef> candidates = {Decls[14]};
  EXPECT_EQ(
      Cpp::GetTypeAsString(Cpp::GetFunctionReturnType(
          Cpp::BestOverloadFunctionMatch(candidates, explicit_args, args))),
      "RTTest_TemplatedList<int, double>");

  std::vector<Cpp::TemplateArgInfo> args2 = {C.DoubleTy.getAsOpaquePtr()};
  EXPECT_EQ(
      Cpp::GetTypeAsString(Cpp::GetFunctionReturnType(Cpp::FuncRef{
          Cpp::GetNamed("func", Cpp::InstantiateTemplate(Decls[15], args2))
              .data})),
      "double");
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_IsAllocator) {
  std::vector<Decl*> Decls;
  std::string code = R"(
    class Klass{
      int val;
    };
    __attribute__((ownership_returns(malloc)))
    Klass* Allocator(){
      Klass* obj = new Klass;
      return obj;
    }
    Klass* __attribute__((malloc)) Allocator2(){
      Klass* obj = new Klass;
      return obj;
    }
    void foo();
    )";
  GetAllTopLevelDecls(code, Decls, true);
  EXPECT_TRUE(Cpp::IsAllocator(Decls[1]));
  EXPECT_TRUE(Cpp::IsAllocator(Decls[2]));
  EXPECT_FALSE(Cpp::IsAllocator(Decls[3]));
  // Builtin check
  code = R"(
  //There is nothing lstdlib.h is included at args
  )";
  TestFixture::CreateInterpreter({"-include", "stdlib.h"});
  Interp->process(code);
  auto mallocDecl = Cpp::GetNamed("malloc");
  EXPECT_TRUE(Cpp::IsAllocator(Cpp::ConstFuncRef{mallocDecl.data}));
  Cpp::DeleteInterpreter();

  // APINotes check
#if !defined(CPPINTEROP_USE_CLING) && !defined(__EMSCRIPTEN__)
  std::string include_flag =
      "-I" + std::string(CPPINTEROP_DIR) + "unittests/CppInterOp/APINotes";
  std::vector<const char*> interpreter_args = {
      "-fmodules", "-fimplicit-module-maps", "-fapinotes-modules",
      include_flag.c_str()};
  TestFixture::CreateInterpreter(interpreter_args);
  code = R"(
  #include "TestHeader.h"
  )";
  Interp->process(code);
  auto testAllocDecl = Cpp::GetNamed("testAlloc");
  EXPECT_TRUE(Cpp::IsAllocator(Cpp::ConstFuncRef{testAllocDecl.data}));

  auto testNotAllocDecl = Cpp::GetNamed("testNotAlloc");
  EXPECT_FALSE(Cpp::IsAllocator(Cpp::ConstFuncRef{testNotAllocDecl.data}));
  Cpp::DeleteInterpreter();
#endif
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_IsDeallocator) {
  std::vector<Decl*> Decls;
  std::string code = R"(
    class Klass{
      int val;
    };
    __attribute__((ownership_takes(malloc, 1)))
    void Deallocator(Klass* arg){
      delete arg;
    }
    void foo();
    )";
  GetAllTopLevelDecls(code, Decls, true);
  EXPECT_TRUE(Cpp::IsDeallocator(Decls[1]));
  EXPECT_FALSE(Cpp::IsDeallocator(Decls[2]));

  code = R"(
  #include <stdlib.h>
    void test(){
      //Do Nothing
    }
  )";
  TestFixture::CreateInterpreter();
  Interp->process(code);
  auto freeDecl = Cpp::GetNamed("free");
  EXPECT_TRUE(Cpp::IsDeallocator(Cpp::ConstFuncRef{freeDecl.data}));
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_GetFunctionNumArgs) {
  std::vector<Decl*> Decls, TemplateSubDecls;
  std::string code = R"(
    void f1() {}
    void f2(int i, double d, long l, char ch) {}
    void f3(int i, double d, long l = 0, char ch = 'a') {}
    void f4(int i = 0, double d = 0.0, long l = 0, char ch = 'a') {}
    int a;

    class MyTemplatedMethodClass {
      template<class A>
      char get_string(A, int i) {
          return 'A';
      }

      template<class A>
      void get_size() {}

      template<class A, class B>
      long add_size (A, int i, B) {
          return sizeof(A) + i;
      }
    };

    )";

  GetAllTopLevelDecls(code, Decls);
  GetAllSubDecls(Decls[5], TemplateSubDecls);
  EXPECT_EQ(Cpp::GetFunctionNumArgs(Decls[0]), (size_t) 0);
  EXPECT_EQ(Cpp::GetFunctionNumArgs(Decls[1]), (size_t) 4);
  EXPECT_EQ(Cpp::GetFunctionNumArgs(Decls[2]), (size_t) 4);
  EXPECT_EQ(Cpp::GetFunctionNumArgs(Decls[3]), (size_t) 4);
  EXPECT_EQ(Cpp::GetFunctionNumArgs(Decls[4]), 0);

  EXPECT_EQ(Cpp::GetFunctionNumArgs(TemplateSubDecls[1]), 2);
  EXPECT_EQ(Cpp::GetFunctionNumArgs(TemplateSubDecls[2]), 0);
  EXPECT_EQ(Cpp::GetFunctionNumArgs(TemplateSubDecls[3]), 3);
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_GetFunctionRequiredArgs) {
  std::vector<Decl*> Decls, TemplateSubDecls;
  std::string code = R"(
    void f1() {}
    void f2(int i, double d, long l, char ch) {}
    void f3(int i, double d, long l = 0, char ch = 'a') {}
    void f4(int i = 0, double d = 0.0, long l = 0, char ch = 'a') {}
    int a;

    class MyTemplatedMethodClass {
      template<class A, class B>
      long get_size(A, B, int i = 0) {}

      template<class A = int, class B = char>
      long get_size(int i, A a = A(), B b = B()) {}

      template<class A>
      void get_size(long k, A, char ch = 'a', double l = 0.0) {}
    };
    )";

  GetAllTopLevelDecls(code, Decls);
  GetAllSubDecls(Decls[5], TemplateSubDecls);

  EXPECT_EQ(Cpp::GetFunctionRequiredArgs(Decls[0]), (size_t) 0);
  EXPECT_EQ(Cpp::GetFunctionRequiredArgs(Decls[1]), (size_t) 4);
  EXPECT_EQ(Cpp::GetFunctionRequiredArgs(Decls[2]), (size_t) 2);
  EXPECT_EQ(Cpp::GetFunctionRequiredArgs(Decls[3]), (size_t) 0);
  EXPECT_EQ(Cpp::GetFunctionRequiredArgs(Decls[4]), 0);

  EXPECT_EQ(Cpp::GetFunctionRequiredArgs(TemplateSubDecls[1]), 2);
  EXPECT_EQ(Cpp::GetFunctionRequiredArgs(TemplateSubDecls[2]), 1);
  EXPECT_EQ(Cpp::GetFunctionRequiredArgs(TemplateSubDecls[3]), 2);
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_GetFunctionArgType) {
  std::vector<Decl*> Decls;
  std::string code = R"(
    void f1(int i, double d, long l, char ch) {}
    void f2(const int i, double d[], long *l, char ch[4]) {}
    int a;

    template <typename T> void f3(T t, const T& r, int i) {}
    )";

  GetAllTopLevelDecls(code, Decls);
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionArgType(Decls[0], 0)), "int");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionArgType(Decls[0], 1)), "double");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionArgType(Decls[0], 2)), "long");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionArgType(Decls[0], 3)), "char");
  EXPECT_EQ(Cpp::GetFunctionArgType(Decls[0], 4), nullptr);

  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionArgType(Decls[1], 0)), "const int");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionArgType(Decls[1], 1)), "double[]");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionArgType(Decls[1], 2)), "long *");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionArgType(Decls[1], 3)), "char[4]");

  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionArgType(Decls[2], 0)), "NULL TYPE");

  EXPECT_TRUE(Cpp::IsTemplatedFunction(Decls[3]));
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionArgType(Decls[3], 0)), "T");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionArgType(Decls[3], 1)),
            "const T &");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionArgType(Decls[3], 2)), "int");
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_IsMethod) {
  std::vector<Decl*> Decls, SubDecls;
  std::string code = R"(
    void f1() {}

    template <typename T> void f2(T t) {}

    class MyClass {
      void m1() {}

      template <typename T>
      void m2(T t) {}
    };
    )";

  GetAllTopLevelDecls(code, Decls);
  GetAllSubDecls(Decls[2], SubDecls);

  EXPECT_FALSE(Cpp::IsMethod(Decls[0]));
  EXPECT_FALSE(Cpp::IsMethod(Decls[1]));
  EXPECT_FALSE(Cpp::IsMethod(SubDecls[0]));
  EXPECT_TRUE(Cpp::IsMethod(SubDecls[1]));
  EXPECT_TRUE(Cpp::IsTemplatedFunction(SubDecls[2]));
  EXPECT_TRUE(Cpp::IsMethod(SubDecls[2]));
  EXPECT_FALSE(Cpp::IsMethod(nullptr));
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_FunctionTypes) {
  std::vector<Decl*> Decls;
  std::string code = R"(
      typedef int myint;
      int (*f1)(int, double) = nullptr;
      void f2(int (&f)(myint, double)) {}
    )";

  GetAllTopLevelDecls(code, Decls);
  EXPECT_EQ(Decls.size(), 3);

  Cpp::TypeRef typ1 = Cpp::GetVariableType(Decls[1]);
  EXPECT_TRUE(typ1);
  EXPECT_FALSE(Cpp::IsFunctionProtoType(typ1));
  EXPECT_TRUE(Cpp::IsFunctionProtoType(Cpp::GetPointeeType(typ1)));

  typ1 = Cpp::GetPointeeType(typ1);
  EXPECT_TRUE(typ1);

  std::vector<Cpp::TypeRef> sig;
  Cpp::GetFnTypeSignature(typ1, sig);
  EXPECT_EQ(sig.size(), 3);
  EXPECT_EQ(Cpp::GetTypeAsString(sig[0]), "int");
  EXPECT_EQ(Cpp::GetTypeAsString(sig[1]), "int");
  EXPECT_EQ(Cpp::GetTypeAsString(sig[2]), "double");

  Cpp::TypeRef typ2 = Cpp::GetFunctionArgType(Decls[2], 0);
  EXPECT_TRUE(typ1);

  typ2 = Cpp::GetNonReferenceType(typ2);
  EXPECT_TRUE(typ2);

  sig.clear();
  Cpp::GetFnTypeSignature(typ2, sig);
  EXPECT_EQ(sig.size(), 3);
  EXPECT_EQ(Cpp::GetTypeAsString(sig[0]), "int");
  EXPECT_EQ(Cpp::GetTypeAsString(sig[1]), "myint");
  EXPECT_EQ(Cpp::GetTypeAsString(sig[2]), "double");

  EXPECT_TRUE(Cpp::IsSameType(typ1, typ2));
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_GetAllocType) {
  std::string code = R"(
    #include <new>
    #include <stdlib.h>

    int* func0(int n){ return new int(n); }

    void* func1(int n){ return (void*) new int(n); }

    void* func2(int n){ return static_cast<void*>(new int(n)); }

    int* func3(int n){ int* x = new int(n); int* y; y=x; return y; }

    int* func4(int n){int* x = new int(n); int* y = x; return y; }

    void* func5(int n){ return malloc(sizeof(int)); }

    void* func6(int n){ void* x = malloc(sizeof(int)); return x; }

    void** func7(int n){ void** arr = new void*[n]; return arr;}

    void** func8(int n){ return new void*[n]; }

    int* func9(int n){ int* x = new int(n); int* y = x; int* z = y; return z; }

    int* func10(int n){ return static_cast<int*>(malloc(sizeof(int(n)))); }

    int* func11(int n){
      int* x = new int(n);
      int* y = nullptr;
      y = x;
      x = nullptr;
      return y;
    }

    int* func12(int n){ int* x = static_cast<int*>(malloc(sizeof(int))); return x;}

    int* func13(int n){ int* x = new int(n); return (((x))); }

    int func14(int n){ return n; }

    int* func15(int n);

    int* func16(int n) try { return new int(n); } catch(...) { return nullptr; }

    int* func17(int* p){ return p; }

    int* func18_t(int n){ return nullptr; }
    typedef int* (*FnPtr18)(int);
    FnPtr18 func18(int n){ return func18_t; }

    int* func19_helper(int n){ return nullptr; }
    int* func19(int n){ return func19_helper(n); }

    int* func20(int* (*fp)(int), int n){ return fp(n); }

    int* func21(int n){ static char buf[16]; return new (&buf) int(n); }

    int* func22(int n){ []{ return; }(); return new int(n); }

    int* func23;

    int* func24(int n){ return func0(n); }

    void** func25(int n){ void** arr = func7(n); return arr; }

    int* func26(bool b){
      if(b)
        return (int*)malloc(sizeof(int));
      return new int;
    }

    int* func27(int n){
      int* p = new int[n];
      p += 1;
      return p;
    }
    )";
  TestFixture::CreateInterpreter();
  Interp->declare(code);

#define TESTAC(N, EXP)                                                         \
  EXPECT_EQ(                                                                   \
      Cpp::GetAllocType(Cpp::ConstFuncRef { Cpp::GetNamed("func" #N).data }),  \
      Cpp::AllocType::EXP)

  TESTAC(0, New);
  TESTAC(1, New);
  TESTAC(2, New);
  TESTAC(3, New);
  TESTAC(4, New);
  TESTAC(5, Malloc);
  TESTAC(6, Malloc);
  TESTAC(7, NewArr);
  TESTAC(8, NewArr);
  TESTAC(9, New);
  TESTAC(10, Malloc);
  TESTAC(11, New);
  TESTAC(12, Malloc);
  TESTAC(13, New);
  TESTAC(14, None);
  TESTAC(15, Unknown);
  TESTAC(16, Unknown);
  TESTAC(17, None);
  TESTAC(18, None);
  TESTAC(19, None);
  TESTAC(20, Unknown);
  TESTAC(21, None);
  TESTAC(22, New);
  TESTAC(23, None);
  TESTAC(24, New);
  TESTAC(25, NewArr);
  TESTAC(26, Unknown);
  // FIXME: Pointer overwriten by a non-assignment operator
  TESTAC(27, NewArr);

#undef TESTAC

  Cpp::DeleteInterpreter();
}
TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_GetFunctionSignature) {
  std::vector<Decl*> Decls;
  std::string code = R"(
    class C {
      void f(int i, double d, long l = 0, char ch = 'a') {}
      template<typename T>
      void ft(T a) {}
    };

    namespace N
    {
      void f(int i, double d, long l = 0, char ch = 'a') {}
    }

    template<typename T>
    void ft(T a) {}
    void f1() {}
    C f2(int i, double d, long l = 0, char ch = 'a') { return C(); }
    C *f3(int i, double d, long l = 0, char ch = 'a') { return new C(); }
    void f4(int i = 0, double d = 0.0, long l = 0, char ch = 'a') {}
    class ABC {};
    )";

  GetAllTopLevelDecls(code, Decls, true);
  GetAllSubDecls(Decls[0], Decls);
  GetAllSubDecls(Decls[1], Decls);

  EXPECT_EQ(Cpp::GetFunctionSignature(Decls[2]), "void ft(T a)");
  EXPECT_EQ(Cpp::GetFunctionSignature(Decls[3]), "void f1()");
  EXPECT_EQ(Cpp::GetFunctionSignature(Decls[4]),
            "C f2(int i, double d, long l = 0, char ch = 'a')");
  EXPECT_EQ(Cpp::GetFunctionSignature(Decls[5]),
            "C *f3(int i, double d, long l = 0, char ch = 'a')");
  EXPECT_EQ(Cpp::GetFunctionSignature(Decls[6]),
            "void f4(int i = 0, double d = 0., long l = 0, char ch = 'a')");
  EXPECT_EQ(Cpp::GetFunctionSignature(Decls[7]), "<unknown>");
  EXPECT_EQ(Cpp::GetFunctionSignature(Decls[9]),
            "void C::f(int i, double d, long l = 0, char ch = 'a')");
  EXPECT_EQ(Cpp::GetFunctionSignature(Decls[10]), "void C::ft(T a)");
  EXPECT_EQ(Cpp::GetFunctionSignature(Decls[15]),
            "void N::f(int i, double d, long l = 0, char ch = 'a')");
  EXPECT_EQ(Cpp::GetFunctionSignature(nullptr), "<unknown>");
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_IsTemplatedFunction) {
  std::vector<Decl*> Decls;
  std::vector<Decl*> SubDeclsC1;
  std::string code = R"(
    void f1(int a) {}

    template<typename T>
    void f2(T a) {}

    class C1 {
      void f1(int a) {}

      template<typename T>
      void f2(T a) {}
    };

    class ABC {};
    )";

  GetAllTopLevelDecls(code, Decls);
  GetAllSubDecls(Decls[2], SubDeclsC1);

  EXPECT_FALSE(Cpp::IsTemplatedFunction(Decls[0]));
  EXPECT_TRUE(Cpp::IsTemplatedFunction(Decls[1]));
  EXPECT_FALSE(Cpp::IsTemplatedFunction(Decls[3]));
  EXPECT_FALSE(Cpp::IsTemplatedFunction(SubDeclsC1[1]));
  EXPECT_TRUE(Cpp::IsTemplatedFunction(SubDeclsC1[2]));
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_ExistsFunctionTemplate) {
  std::vector<Decl*> Decls;
  std::string code = R"(
    template<typename T>
    void f(T a) {}

    class C {
      template<typename T>
      void f(T a) {}
    };

    void f(char ch) {}
    )";

  GetAllTopLevelDecls(code, Decls);
  EXPECT_TRUE(Cpp::ExistsFunctionTemplate("f", nullptr));
  EXPECT_TRUE(Cpp::ExistsFunctionTemplate("f", Decls[1]));
  EXPECT_FALSE(Cpp::ExistsFunctionTemplate("f", Decls[2]));
}

TYPED_TEST(CPPINTEROP_TEST_MODE,
           FunctionReflection_InstantiateTemplateFunctionFromString) {
  std::vector<const char*> interpreter_args = { "-include", "new" };
  TestFixture::CreateInterpreter(interpreter_args);
  std::string code = R"(#include <memory>)";
  Interp->process(code);
  const char* str = "std::make_unique<int,int>";
  auto Instance1 = Cpp::InstantiateTemplateFunctionFromString(str);
  EXPECT_TRUE(Instance1);
}

TYPED_TEST(CPPINTEROP_TEST_MODE,
           FunctionReflection_InstantiateFunctionTemplate) {
  std::vector<Decl*> Decls;
  std::string code = R"(
template<typename T> T TrivialFnTemplate() { return T(); }
)";

  GetAllTopLevelDecls(code, Decls);
  ASTContext& C = Interp->getCI()->getASTContext();

  std::vector<Cpp::TemplateArgInfo> args1 = {C.IntTy.getAsOpaquePtr()};
  auto Instance1 = Cpp::InstantiateTemplate(Decls[0], args1);
  EXPECT_TRUE(isa<FunctionDecl>(Cpp::unwrap<Decl>(Instance1)));
  FunctionDecl* FD = cast<FunctionDecl>(Cpp::unwrap<Decl>(Instance1));
  FunctionDecl* FnTD1 = FD->getTemplateInstantiationPattern();
  EXPECT_TRUE(FnTD1->isThisDeclarationADefinition());
  TemplateArgument TA1 = FD->getTemplateSpecializationArgs()->get(0);
  EXPECT_TRUE(TA1.getAsType()->isIntegerType());
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_InstantiateTemplateMethod) {
  std::vector<Decl*> Decls;
  std::string code = R"(
    class MyTemplatedMethodClass {
      public:
          template<class A> long get_size(A&);
      };

      template<class A>
      long MyTemplatedMethodClass::get_size(A&) {
          return sizeof(A);
      }
  )";

  GetAllTopLevelDecls(code, Decls);
  ASTContext& C = Interp->getCI()->getASTContext();

  std::vector<Cpp::TemplateArgInfo> args1 = {C.IntTy.getAsOpaquePtr()};
  auto Instance1 = Cpp::InstantiateTemplate(Decls[1], args1);
  EXPECT_TRUE(isa<FunctionDecl>(Cpp::unwrap<Decl>(Instance1)));
  FunctionDecl* FD = cast<FunctionDecl>(Cpp::unwrap<Decl>(Instance1));
  FunctionDecl* FnTD1 = FD->getTemplateInstantiationPattern();
  EXPECT_TRUE(FnTD1->isThisDeclarationADefinition());
  TemplateArgument TA1 = FD->getTemplateSpecializationArgs()->get(0);
  EXPECT_TRUE(TA1.getAsType()->isIntegerType());
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_LookupConstructors) {

  std::vector<Decl*> Decls;
  std::string code = R"(
    class MyClass {
      public:
        MyClass();
        void helperMethod();
        MyClass(const MyClass&);
        static void staticFunc();
        MyClass(MyClass&&);
        template<typename T>
        MyClass(T);
        ~MyClass();
    };

    MyClass::MyClass() {}
    void MyClass::helperMethod() {}
    MyClass::MyClass(const MyClass&) {}
    void MyClass::staticFunc() {}
    MyClass::MyClass(MyClass&&) {}
    template<typename T>
    MyClass::MyClass(T t) {}
  )";

  GetAllTopLevelDecls(code, Decls);
  std::vector<Cpp::FuncRef> ctors;
  Cpp::LookupConstructors("MyClass", Decls[0], ctors);

  EXPECT_EQ(ctors.size(), 4)
      << "Constructor lookup did not retrieve the expected set";
  EXPECT_EQ(Cpp::GetFunctionSignature(ctors[0]), "MyClass::MyClass()");
  EXPECT_EQ(Cpp::GetFunctionSignature(ctors[1]),
            "MyClass::MyClass(const MyClass &)");
  EXPECT_EQ(Cpp::GetFunctionSignature(ctors[2]),
            "MyClass::MyClass(MyClass &&)");
  EXPECT_EQ(Cpp::GetFunctionSignature(ctors[3]), "MyClass::MyClass(T t)");
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_GetClassTemplatedMethods) {

  std::vector<Decl*> Decls;
  std::string code = R"(
    class MyClass {
      public:
        MyClass();
        void helperMethod();
        template<typename T>
        MyClass(T);
        template<typename U>
        void templatedMethod(U param);
        template<typename U, typename V>
        U templatedMethod(U a, V b);
        static void staticFunc();
        template<typename T>
        static void templatedStaticMethod(T param);
        ~MyClass();
    };

    MyClass::MyClass() {}
    void MyClass::helperMethod() {}
    template<typename T>
    MyClass::MyClass(T t) {}
    template<typename U>
    void MyClass::templatedMethod(U param) {}
    template<typename U, typename V>
    U MyClass::templatedMethod(U a, V b) { return a * b; }
    void MyClass::staticFunc() {}
    template<typename T>
    void MyClass::templatedStaticMethod(T param) {}

    class TBase {
    public:
      template<typename T>
      void usedTemplated(T param);
      template<typename T, typename U>
      void usedTemplated(T t, U u);
    };
    class TDerived : public TBase {
    public:
      using TBase::usedTemplated;
    };
  )";

  GetAllTopLevelDecls(code, Decls);
  std::vector<Cpp::FuncRef> templatedMethods;
  Cpp::GetClassTemplatedMethods("MyClass", Decls[0], templatedMethods);
  Cpp::GetClassTemplatedMethods("templatedMethod", Decls[0], templatedMethods);
  Cpp::GetClassTemplatedMethods("templatedStaticMethod", Decls[0],
                                templatedMethods);

  EXPECT_EQ(templatedMethods.size(), 6)
      << "Templated methods lookup did not retrieve the expected set";
  EXPECT_EQ(Cpp::GetFunctionSignature(templatedMethods[0]),
            "MyClass::MyClass()");
  EXPECT_EQ(Cpp::GetFunctionSignature(templatedMethods[1]),
            "MyClass::MyClass(T t)");
  EXPECT_EQ(Cpp::GetFunctionSignature(templatedMethods[2]),
            "inline constexpr MyClass::MyClass(const MyClass &)");
  EXPECT_EQ(Cpp::GetFunctionSignature(templatedMethods[3]),
            "void MyClass::templatedMethod(U param)");
  EXPECT_EQ(Cpp::GetFunctionSignature(templatedMethods[4]),
            "U MyClass::templatedMethod(U a, V b)");
  EXPECT_EQ(Cpp::GetFunctionSignature(templatedMethods[5]),
            "void MyClass::templatedStaticMethod(T param)");

  std::vector<Cpp::FuncRef> usingMethods;
  Cpp::DeclRef derived = Cpp::GetScope("TDerived");
  EXPECT_TRUE(derived);
  Cpp::GetClassTemplatedMethods("usedTemplated", derived, usingMethods);
  EXPECT_EQ(usingMethods.size(), 2);
  EXPECT_EQ(Cpp::GetFunctionSignature(usingMethods[0]),
            "void TBase::usedTemplated(T param)");
  EXPECT_EQ(Cpp::GetFunctionSignature(usingMethods[1]),
            "void TBase::usedTemplated(T t, U u)");
}

TYPED_TEST(CPPINTEROP_TEST_MODE,
           FunctionReflection_GetClassTemplatedMethods_VariadicsAndOthers) {
  std::vector<Decl*> Decls;
  std::string code = R"(
    class MyClass {
      public:
        template<typename... Args>
        void variadicMethod(Args... args);

        template<int N>
        int fixedMethod();

        template<typename T = int>
        T defaultMethod(T param);

        template<typename U, typename... V>
        U variadicMethod(U first, V... rest);

        template<typename T, typename... Args>
        static void staticVariadic(T t, Args... args);
    };

    template<typename... Args>
    void MyClass::variadicMethod(Args... args) {}
    template<int N>
    int MyClass::fixedMethod() { return N; }
    template<typename T>
    T MyClass::defaultMethod(T param) { return param; }
    template<typename U, typename... V>
    U MyClass::variadicMethod(U first, V... rest) { return first; }
    template<typename T, typename... Args>
    void MyClass::staticVariadic(T t, Args... args) {}
  )";

  GetAllTopLevelDecls(code, Decls);
  std::vector<Cpp::FuncRef> templatedMethods;
  Cpp::GetClassTemplatedMethods("fixedMethod", Decls[0], templatedMethods);
  Cpp::GetClassTemplatedMethods("defaultMethod", Decls[0], templatedMethods);
  Cpp::GetClassTemplatedMethods("variadicMethod", Decls[0], templatedMethods);
  Cpp::GetClassTemplatedMethods("staticVariadic", Decls[0], templatedMethods);

  EXPECT_EQ(templatedMethods.size(), 5)
      << "Templated methods lookup did not retrieve the expected set";
  EXPECT_EQ(Cpp::GetFunctionSignature(templatedMethods[0]),
            "int MyClass::fixedMethod()");
  EXPECT_EQ(Cpp::GetFunctionSignature(templatedMethods[1]),
            "T MyClass::defaultMethod(T param)");
  EXPECT_EQ(Cpp::GetFunctionSignature(templatedMethods[2]),
            "void MyClass::variadicMethod(Args ...args)");
  EXPECT_EQ(Cpp::GetFunctionSignature(templatedMethods[3]),
            "U MyClass::variadicMethod(U first, V ...rest)");
  EXPECT_EQ(Cpp::GetFunctionSignature(templatedMethods[4]),
            "void MyClass::staticVariadic(T t, Args ...args)");
}

TYPED_TEST(CPPINTEROP_TEST_MODE,
           FunctionReflection_InstantiateVariadicFunction) {
  std::vector<Decl*> Decls;
  std::string code = R"(
    class MyClass {};

    template<typename... Args>
    void VariadicFn(Args... args) {}

    template<typename... Args>
    void VariadicFnExtended(int fixedParam, Args... args) {}
  )";

  GetAllTopLevelDecls(code, Decls);
  ASTContext& C = Interp->getCI()->getASTContext();

  std::vector<Cpp::TemplateArgInfo> args1 = {C.DoubleTy.getAsOpaquePtr(),
                                             C.IntTy.getAsOpaquePtr()};
  auto Instance1 = Cpp::InstantiateTemplate(Decls[1], args1);
  EXPECT_TRUE(Cpp::IsTemplatedFunction(Cpp::FuncRef{Instance1.data}));
  EXPECT_EQ(Cpp::GetFunctionSignature(Cpp::FuncRef{Instance1.data}),
            "template<> void VariadicFn<<double, int>>(double args, int args)");

  FunctionDecl* FD = cast<FunctionDecl>(Cpp::unwrap<Decl>(Instance1));
  FunctionDecl* FnTD1 = FD->getTemplateInstantiationPattern();
  EXPECT_TRUE(FnTD1->isThisDeclarationADefinition());
  EXPECT_EQ(FD->getNumParams(), 2);

  const TemplateArgumentList* TA1 = FD->getTemplateSpecializationArgs();
  llvm::ArrayRef<TemplateArgument> Args = TA1->get(0).getPackAsArray();
  EXPECT_EQ(Args.size(), 2);
  EXPECT_TRUE(Args[0].getAsType()->isFloatingType());
  EXPECT_TRUE(Args[1].getAsType()->isIntegerType());

  // handle to MyClass type
  auto MyClassType = Cpp::GetTypeFromScope(Decls[0]);
  std::vector<Cpp::TemplateArgInfo> args2 = {MyClassType.data,
                                             C.DoubleTy.getAsOpaquePtr()};

  // instantiate VariadicFnExtended
  auto Instance2 = Cpp::InstantiateTemplate(Decls[2], args2, true);
  EXPECT_TRUE(Cpp::IsTemplatedFunction(Cpp::FuncRef{Instance2.data}));

  FunctionDecl* FD2 = cast<FunctionDecl>(Cpp::unwrap<Decl>(Instance2));
  FunctionDecl* FnTD2 = FD2->getTemplateInstantiationPattern();
  EXPECT_TRUE(FnTD2->isThisDeclarationADefinition());

  // VariadicFnExtended has one fixed param + 2 elements in TemplateArgument
  // pack
  EXPECT_EQ(FD2->getNumParams(), 3);

  const TemplateArgumentList* TA2 = FD2->getTemplateSpecializationArgs();
  llvm::ArrayRef<TemplateArgument> PackArgs2 = TA2->get(0).getPackAsArray();
  EXPECT_EQ(PackArgs2.size(), 2);

  EXPECT_TRUE(PackArgs2[0].getAsType()->isRecordType());   // MyClass
  EXPECT_TRUE(PackArgs2[1].getAsType()->isFloatingType()); // double
  EXPECT_EQ(Cpp::GetFunctionSignature(Cpp::FuncRef{Instance2.data}),
            "template<> void VariadicFnExtended<<MyClass, double>>(int "
            "fixedParam, MyClass args, double args)");
}

TYPED_TEST(CPPINTEROP_TEST_MODE,
           FunctionReflection_BestOverloadFunctionMatch0) {
  // make sure templates are not instantiated multiple times
  std::vector<Decl*> Decls;
  std::string code = R"(
  template <typename T1, typename T2 = int>
  void Tfn(T1& t1, T2& t2) {}
  )";
  GetAllTopLevelDecls(code, Decls);
  EXPECT_EQ(Decls.size(), 1);

  std::vector<Cpp::FuncRef> candidates;
  candidates.reserve(Decls.size());
  for (auto* i : Decls)
    candidates.push_back(i);

  ASTContext& C = Interp->getCI()->getASTContext();

  std::vector<Cpp::TemplateArgInfo> args0 = {
      C.getLValueReferenceType(C.DoubleTy).getAsOpaquePtr(),
      C.getLValueReferenceType(C.IntTy).getAsOpaquePtr(),
  };

  std::vector<Cpp::TemplateArgInfo> explicit_args0;
  std::vector<Cpp::TemplateArgInfo> explicit_args1 = {
      C.DoubleTy.getAsOpaquePtr()};
  std::vector<Cpp::TemplateArgInfo> explicit_args2 = {
      C.DoubleTy.getAsOpaquePtr(),
      C.IntTy.getAsOpaquePtr(),
  };

  Cpp::FuncRef fn0 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args0, args0);
  EXPECT_TRUE(fn0);

  Cpp::FuncRef fn =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args1, args0);
  EXPECT_EQ(fn, fn0);

  fn = Cpp::BestOverloadFunctionMatch(candidates, explicit_args2, args0);
  EXPECT_EQ(fn, fn0);

  fn = Cpp::FuncRef{Cpp::InstantiateTemplate(Decls[0], explicit_args1).data};
  EXPECT_EQ(fn, fn0);

  fn = Cpp::FuncRef{Cpp::InstantiateTemplate(Decls[0], explicit_args2).data};
  EXPECT_EQ(fn, fn0);
}

TYPED_TEST(CPPINTEROP_TEST_MODE,
           FunctionReflection_BestOverloadFunctionMatch1) {
  std::vector<Decl*> Decls;
  std::string code = R"(
    class MyTemplatedMethodClass {
      public:
          template<class A> long get_size(A&);
          template<class A> long get_size();
          template<class A, class B> long get_size(A a, B b);
          template<class A> long get_size(float a);
          template <int N, class T> long get_size(T a);
      };

      template<class A>
      long MyTemplatedMethodClass::get_size(A&) {
          return sizeof(A);
      }

      template<class A>
      long MyTemplatedMethodClass::get_size() {
          return sizeof(A) + 1;
      }

      template<class A>
      long MyTemplatedMethodClass::get_size(float a) {
          return sizeof(A) + long(a);
      }

      template<class A, class B>
      long MyTemplatedMethodClass::get_size(A a, B b) {
          return sizeof(A) + sizeof(B);
      }

      template <int N, class T>
      long MyTemplatedMethodClass::get_size(T a) {
        return N + sizeof(T) + a;
      }
  )";

  GetAllTopLevelDecls(code, Decls);
  std::vector<Cpp::FuncRef> candidates;

  for (auto decl : Decls)
    if (Cpp::IsTemplatedFunction(decl))
      candidates.push_back((Cpp::FuncRef)decl);

  ASTContext& C = Interp->getCI()->getASTContext();

  std::vector<Cpp::TemplateArgInfo> args0;
  std::vector<Cpp::TemplateArgInfo> args1 = {
      C.getLValueReferenceType(C.IntTy).getAsOpaquePtr()};
  std::vector<Cpp::TemplateArgInfo> args2 = {C.CharTy.getAsOpaquePtr(), C.FloatTy.getAsOpaquePtr()};
  std::vector<Cpp::TemplateArgInfo> args3 = {C.FloatTy.getAsOpaquePtr()};

  std::vector<Cpp::TemplateArgInfo> explicit_args0;
  std::vector<Cpp::TemplateArgInfo> explicit_args1 = {C.IntTy.getAsOpaquePtr()};
  std::vector<Cpp::TemplateArgInfo> explicit_args2 = {
      {C.IntTy.getAsOpaquePtr(), "1"}, C.IntTy.getAsOpaquePtr()};

  Cpp::FuncRef func1 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args0, args1);
  Cpp::FuncRef func2 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args1, args0);
  Cpp::FuncRef func3 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args0, args2);
  Cpp::FuncRef func4 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args1, args3);
  Cpp::FuncRef func5 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args2, args3);

  EXPECT_EQ(Cpp::GetFunctionSignature(func1),
            "template<> long MyTemplatedMethodClass::get_size<int>(int &)");
  EXPECT_EQ(Cpp::GetFunctionSignature(func2),
            "template<> long MyTemplatedMethodClass::get_size<int>()");
  EXPECT_EQ(Cpp::GetFunctionSignature(func3),
            "template<> long MyTemplatedMethodClass::get_size<char, float>(char a, float b)");
  EXPECT_EQ(Cpp::GetFunctionSignature(func4),
            "template<> long MyTemplatedMethodClass::get_size<int>(float a)");
  EXPECT_EQ(Cpp::GetFunctionSignature(func5),
            "template<> long MyTemplatedMethodClass::get_size<1, int>(int a)");
}

TYPED_TEST(CPPINTEROP_TEST_MODE,
           FunctionReflection_BestOverloadFunctionMatch2) {
  std::vector<Decl*> Decls;
  std::string code = R"(
    template<typename T>
    struct A { T value; };

    A<int> a;

    template<typename T>
    void somefunc(A<T> arg) {}

    template<typename T>
    void somefunc(T arg) {}

    template<typename T>
    void somefunc(A<T> arg1, A<T> arg2) {}

    template<typename T>
    void somefunc(T arg1, T arg2) {}

    void somefunc(int arg1, double arg2) {}
  )";

  GetAllTopLevelDecls(code, Decls);
  std::vector<Cpp::FuncRef> candidates;

  for (auto decl : Decls)
    if (Cpp::IsFunction(decl) || Cpp::IsTemplatedFunction(decl))
      candidates.push_back((Cpp::FuncRef)decl);

  EXPECT_EQ(candidates.size(), 5);

  ASTContext& C = Interp->getCI()->getASTContext();

  std::vector<Cpp::TemplateArgInfo> args1 = {C.IntTy.getAsOpaquePtr()};
  std::vector<Cpp::TemplateArgInfo> args2 = {
      Cpp::GetVariableType(Cpp::GetNamed("a")).data};
  std::vector<Cpp::TemplateArgInfo> args3 = {C.IntTy.getAsOpaquePtr(),
                                             C.IntTy.getAsOpaquePtr()};
  std::vector<Cpp::TemplateArgInfo> args4 = {
      Cpp::GetVariableType(Cpp::GetNamed("a")).data,
      Cpp::GetVariableType(Cpp::GetNamed("a")).data};
  std::vector<Cpp::TemplateArgInfo> args5 = {C.IntTy.getAsOpaquePtr(),
                                             C.DoubleTy.getAsOpaquePtr()};

  std::vector<Cpp::TemplateArgInfo> explicit_args;

  Cpp::FuncRef func1 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args, args1);
  Cpp::FuncRef func2 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args, args2);
  Cpp::FuncRef func3 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args, args3);
  Cpp::FuncRef func4 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args, args4);
  Cpp::FuncRef func5 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args, args5);

  EXPECT_EQ(Cpp::GetFunctionSignature(func1),
            "template<> void somefunc<int>(int arg)");
  EXPECT_EQ(Cpp::GetFunctionSignature(func2),
            "template<> void somefunc<int>(A<int> arg)");
  EXPECT_EQ(Cpp::GetFunctionSignature(func3),
            "template<> void somefunc<int>(int arg1, int arg2)");
  EXPECT_EQ(Cpp::GetFunctionSignature(func4),
            "template<> void somefunc<int>(A<int> arg1, A<int> arg2)");
  EXPECT_EQ(Cpp::GetFunctionSignature(func5),
            "void somefunc(int arg1, double arg2)");
}

TYPED_TEST(CPPINTEROP_TEST_MODE,
           FunctionReflection_BestOverloadFunctionMatch3) {
  std::vector<Decl*> Decls;
  std::string code = R"(
    template<typename T>
    struct A {
      T value;

      template<typename TT>
      A<TT> operator-(A<TT> rhs) {
        return A<TT>{value - rhs.value};
      }
    };

    A<int> a;

    template<typename T>
    A<T> operator+(A<T> lhs, A<T> rhs) {
      return A<T>{lhs.value + rhs.value};
    }

    template<typename T>
    A<T> operator+(A<T> lhs, int rhs) {
      return A<T>{lhs.value + rhs};
    }
  )";

  GetAllTopLevelDecls(code, Decls);
  std::vector<Cpp::FuncRef> candidates;

  for (auto decl : Decls)
    if (Cpp::IsTemplatedFunction(decl))
      candidates.push_back((Cpp::FuncRef)decl);

  EXPECT_EQ(candidates.size(), 2);

  ASTContext& C = Interp->getCI()->getASTContext();

  std::vector<Cpp::TemplateArgInfo> args1 = {
      Cpp::GetVariableType(Cpp::GetNamed("a")).data,
      Cpp::GetVariableType(Cpp::GetNamed("a")).data};
  std::vector<Cpp::TemplateArgInfo> args2 = {
      Cpp::GetVariableType(Cpp::GetNamed("a")).data, C.IntTy.getAsOpaquePtr()};
  std::vector<Cpp::TemplateArgInfo> args3 = {
      Cpp::GetVariableType(Cpp::GetNamed("a")).data,
      C.DoubleTy.getAsOpaquePtr()};
  std::vector<Cpp::TemplateArgInfo> explicit_args;

  Cpp::FuncRef func1 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args, args1);
  Cpp::FuncRef func2 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args, args2);
  Cpp::FuncRef func3 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args, args3);

  candidates.clear();
  Cpp::GetOperator(
      Cpp::GetScopeFromType(Cpp::GetVariableType(Cpp::GetNamed("a"))),
      Cpp::Operator::OP_Minus, candidates);

  EXPECT_EQ(candidates.size(), 1);

  std::vector<Cpp::TemplateArgInfo> args4 = {
      Cpp::GetVariableType(Cpp::GetNamed("a")).data};

  Cpp::FuncRef func4 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args, args4);

  EXPECT_EQ(Cpp::GetFunctionSignature(func1),
            "template<> A<int> operator+<int>(A<int> lhs, A<int> rhs)");
  EXPECT_EQ(Cpp::GetFunctionSignature(func2),
            "template<> A<int> operator+<int>(A<int> lhs, int rhs)");
  EXPECT_EQ(Cpp::GetFunctionSignature(func3),
            "template<> A<int> operator+<int>(A<int> lhs, int rhs)");

  EXPECT_EQ(Cpp::GetFunctionSignature(func4),
            "template<> A<int> A<int>::operator-<int>(A<int> rhs)");
}

TYPED_TEST(CPPINTEROP_TEST_MODE,
           FunctionReflection_BestOverloadFunctionMatch4) {
  std::vector<Decl*> Decls, SubDecls;
  std::string code = R"(
    template<typename T>
    struct A { T value; };

    class B {
    public:
      void fn() {}
      template <typename T>
      void fn(T x) {}
      template <typename T>
      void fn(A<T> x) {}
      template <typename T1, typename T2>
      void fn(A<T1> x, A<T2> y) {}
    };

    A<int> a;
    A<double> b;
  )";

  GetAllTopLevelDecls(code, Decls);
  GetAllSubDecls(Decls[1], SubDecls);
  std::vector<Cpp::FuncRef> candidates;
  for (auto i : SubDecls) {
    if ((Cpp::IsFunction(i) || Cpp::IsTemplatedFunction(i)) &&
        Cpp::GetName(i) == "fn")
      candidates.push_back(i);
  }

  EXPECT_EQ(candidates.size(), 4);

  ASTContext& C = Interp->getCI()->getASTContext();

  std::vector<Cpp::TemplateArgInfo> args1 = {};
  std::vector<Cpp::TemplateArgInfo> args2 = {C.IntTy.getAsOpaquePtr()};
  std::vector<Cpp::TemplateArgInfo> args3 = {
      Cpp::GetVariableType(Cpp::GetNamed("a")).data};
  std::vector<Cpp::TemplateArgInfo> args4 = {
      Cpp::GetVariableType(Cpp::GetNamed("a")).data,
      Cpp::GetVariableType(Cpp::GetNamed("b")).data};
  std::vector<Cpp::TemplateArgInfo> args5 = {
      Cpp::GetVariableType(Cpp::GetNamed("a")).data,
      Cpp::GetVariableType(Cpp::GetNamed("a")).data};

  std::vector<Cpp::TemplateArgInfo> explicit_args1;
  std::vector<Cpp::TemplateArgInfo> explicit_args2 = {C.IntTy.getAsOpaquePtr(),
                                                      C.IntTy.getAsOpaquePtr()};

  Cpp::FuncRef func1 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args1, args1);
  Cpp::FuncRef func2 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args1, args2);
  Cpp::FuncRef func3 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args1, args3);
  Cpp::FuncRef func4 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args1, args4);
  Cpp::FuncRef func5 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args2, args5);

  EXPECT_EQ(Cpp::GetFunctionSignature(func1), "void B::fn()");
  EXPECT_EQ(Cpp::GetFunctionSignature(func2),
            "template<> void B::fn<int>(int x)");
  EXPECT_EQ(Cpp::GetFunctionSignature(func3),
            "template<> void B::fn<int>(A<int> x)");
  EXPECT_EQ(Cpp::GetFunctionSignature(func4),
            "template<> void B::fn<int, double>(A<int> x, A<double> y)");
  EXPECT_EQ(Cpp::GetFunctionSignature(func5),
            "template<> void B::fn<int, int>(A<int> x, A<int> y)");
}

TYPED_TEST(CPPINTEROP_TEST_MODE,
           FunctionReflection_BestOverloadFunctionMatch5) {
  std::vector<Decl*> Decls;
  std::string code = R"(
    template<typename T1, typename T2>
    void callme(T1 t1, T2 t2) {}

    template <typename F, typename... Args>
    void callback(F callable, Args&&... args) {
      callable(args...);
    }
  )";
  GetAllTopLevelDecls(code, Decls);
  EXPECT_EQ(Decls.size(), 2);

  std::vector<Cpp::FuncRef> candidates;
  candidates.push_back(Decls[1]);

  ASTContext& C = Interp->getCI()->getASTContext();
  std::vector<Cpp::TemplateArgInfo> explicit_params = {
      C.DoubleTy.getAsOpaquePtr(),
      C.IntTy.getAsOpaquePtr(),
  };

  Cpp::DeclRef callme = Cpp::InstantiateTemplate(Decls[0], explicit_params);
  EXPECT_TRUE(callme);

  std::vector<Cpp::TemplateArgInfo> arg_types = {
      Cpp::GetTypeFromScope(callme).data,
      C.getLValueReferenceType(C.DoubleTy).getAsOpaquePtr(),
      C.getLValueReferenceType(C.IntTy).getAsOpaquePtr(),
  };

  Cpp::FuncRef callback =
      Cpp::BestOverloadFunctionMatch(candidates, empty_templ_args, arg_types);
  EXPECT_TRUE(callback);

  EXPECT_EQ(Cpp::GetFunctionSignature(callback),
            "template<> void callback<void (*)(double, int), <double &, int "
            "&>>(void (*callable)(double, int), double &args, int &args)");
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_IsPublicMethod) {
  std::vector<Decl *> Decls, SubDecls;
  std::string code = R"(
    class C {
    public:
      C() {}
      void pub_f() {}
      ~C() {}
    private:
      void pri_f() {}
    protected:
      void pro_f() {}
      int a;
    };
    )";

  GetAllTopLevelDecls(code, Decls);
  GetAllSubDecls(Decls[0], SubDecls);

  EXPECT_TRUE(Cpp::IsPublicMethod(SubDecls[2]));
  EXPECT_TRUE(Cpp::IsPublicMethod(SubDecls[3]));
  EXPECT_TRUE(Cpp::IsPublicMethod(SubDecls[4]));
  EXPECT_FALSE(Cpp::IsPublicMethod(SubDecls[6]));
  EXPECT_FALSE(Cpp::IsPublicMethod(SubDecls[8]));
  EXPECT_FALSE(Cpp::IsPublicMethod(SubDecls[9]));
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_IsProtectedMethod) {
  std::vector<Decl *> Decls, SubDecls;
  std::string code = R"(
    class C {
    public:
      C() {}
      void pub_f() {}
      ~C() {}
    private:
      void pri_f() {}
    protected:
      void pro_f() {}
    };
    )";

  GetAllTopLevelDecls(code, Decls);
  GetAllSubDecls(Decls[0], SubDecls);

  EXPECT_FALSE(Cpp::IsProtectedMethod(SubDecls[2]));
  EXPECT_FALSE(Cpp::IsProtectedMethod(SubDecls[3]));
  EXPECT_FALSE(Cpp::IsProtectedMethod(SubDecls[4]));
  EXPECT_FALSE(Cpp::IsProtectedMethod(SubDecls[6]));
  EXPECT_TRUE(Cpp::IsProtectedMethod(SubDecls[8]));
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_IsPrivateMethod) {
  std::vector<Decl *> Decls, SubDecls;
  std::string code = R"(
    class C {
    public:
      C() {}
      void pub_f() {}
      ~C() {}
    private:
      void pri_f() {}
    protected:
      void pro_f() {}
    };
    )";

  GetAllTopLevelDecls(code, Decls);
  GetAllSubDecls(Decls[0], SubDecls);

  EXPECT_FALSE(Cpp::IsPrivateMethod(SubDecls[2]));
  EXPECT_FALSE(Cpp::IsPrivateMethod(SubDecls[3]));
  EXPECT_FALSE(Cpp::IsPrivateMethod(SubDecls[4]));
  EXPECT_TRUE(Cpp::IsPrivateMethod(SubDecls[6]));
  EXPECT_FALSE(Cpp::IsPrivateMethod(SubDecls[8]));
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_IsConstructor) {
  std::vector<Decl *> Decls, SubDecls;
  std::string code = R"(
    class C {
    public:
      C() {}
      void pub_f() {}
      ~C() {}
    private:
      void pri_f() {}
    protected:
      void pro_f() {}
    };
    )";

  GetAllTopLevelDecls(code, Decls);
  GetAllSubDecls(Decls[0], SubDecls);

  EXPECT_TRUE(Cpp::IsConstructor(SubDecls[2]));
  EXPECT_FALSE(Cpp::IsConstructor(SubDecls[3]));
  EXPECT_FALSE(Cpp::IsConstructor(SubDecls[4]));
  EXPECT_FALSE(Cpp::IsConstructor(SubDecls[6]));
  EXPECT_FALSE(Cpp::IsConstructor(SubDecls[8]));

  // Test for templated constructor
  std::vector<Decl*> templDecls, templSubDecls;
  std::string templCode = R"(
    class T {
    public:
      template<typename U>
      T(U) {}
      void func() {}
      ~T() {}
    };
  )";

  GetAllTopLevelDecls(templCode, templDecls);
  GetAllSubDecls(templDecls[0], templSubDecls);

  int templCtorCount = 0;
  for (auto* decl : templSubDecls) {
    if (Cpp::IsConstructor(decl))
      templCtorCount++;
  }
  EXPECT_EQ(templCtorCount, 1);
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_IsDestructor) {
  std::vector<Decl *> Decls, SubDecls;
  std::string code = R"(
    class C {
    public:
      C() {}
      void pub_f() {}
      ~C() {}
    private:
      void pri_f() {}
    protected:
      void pro_f() {}
    };
    )";

  GetAllTopLevelDecls(code, Decls);
  GetAllSubDecls(Decls[0], SubDecls);

  EXPECT_FALSE(Cpp::IsDestructor(SubDecls[2]));
  EXPECT_FALSE(Cpp::IsDestructor(SubDecls[3]));
  EXPECT_TRUE(Cpp::IsDestructor(SubDecls[4]));
  EXPECT_FALSE(Cpp::IsDestructor(SubDecls[6]));
  EXPECT_FALSE(Cpp::IsDestructor(SubDecls[8]));
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_IsStaticMethod) {
  std::vector<Decl *> Decls, SubDecls;
  std::string code = R"(
    class C {
      void f1() {}
      static void f2() {}
      template <typename T>
      static void f3() {}
      template <typename T>
      void f4() {}
    };
    )";

  GetAllTopLevelDecls(code, Decls);
  GetAllSubDecls(Decls[0], SubDecls);

  EXPECT_FALSE(Cpp::IsStaticMethod(Decls[0]));
  EXPECT_FALSE(Cpp::IsStaticMethod(SubDecls[1]));
  EXPECT_TRUE(Cpp::IsStaticMethod(SubDecls[2]));
  EXPECT_TRUE(Cpp::IsStaticMethod(SubDecls[3]));
  EXPECT_FALSE(Cpp::IsStaticMethod(SubDecls[4]));
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_GetFunctionAddress) {
#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif

  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";

  std::vector<Decl*> Decls;
  std::string code = "int f1(int i) { return i * i; }";
  std::vector<const char*> interpreter_args = {"-include", "new", "-Xclang", "-iwithsysroot/include/compat"};

  GetAllTopLevelDecls(code, Decls, /*filter_implicitGenerated=*/false,
                      interpreter_args);

  testing::internal::CaptureStdout();
  Interp->declare("#include <iostream>");
  Interp->process(
    "void * address = (void *) &f1; \n"
    "std::cout << address; \n"
    );

  std::string output = testing::internal::GetCapturedStdout();
  std::stringstream address;
  address << Cpp::GetFunctionAddress(Decls[0]);
  EXPECT_EQ(address.str(), output);

  EXPECT_FALSE(
      Cpp::GetFunctionAddress(Cpp::FuncRef{Cpp::GetGlobalScope().data}));

  Interp->declare(R"(
    template <typename T>
    T add1(T t) { return t + 1; }
  )");

  std::vector<Cpp::FuncRef> funcs;
  Cpp::GetClassTemplatedMethods("add1", Cpp::GetGlobalScope(), funcs);
  EXPECT_EQ(funcs.size(), 1);

  ASTContext& C = Interp->getCI()->getASTContext();
  std::vector<Cpp::TemplateArgInfo> argument = {C.DoubleTy.getAsOpaquePtr()};
  Cpp::DeclRef add1_double =
      Cpp::InstantiateTemplate(Cpp::DeclRef{funcs[0].data}, argument);
  EXPECT_TRUE(add1_double);

  EXPECT_TRUE(Cpp::GetFunctionAddress(Cpp::FuncRef{add1_double.data}));
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_IsVirtualMethod) {
  std::vector<Decl*> Decls, SubDecls;
  std::string code = R"(
    class A {
      public:
      virtual void x() {}
      void y() {}
    };
    )";

  GetAllTopLevelDecls(code, Decls);
  GetAllSubDecls(Decls[0], SubDecls);

  EXPECT_EQ(Cpp::GetName(SubDecls[2]), "x");
  EXPECT_TRUE(Cpp::IsVirtualMethod(SubDecls[2]));
  EXPECT_EQ(Cpp::GetName(SubDecls[3]), "y");
  EXPECT_FALSE(Cpp::IsVirtualMethod(SubDecls[3])); // y()
  EXPECT_FALSE(Cpp::IsVirtualMethod(Decls[0]));
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_JitCallAdvanced) {
#if CLANG_VERSION_MAJOR == 20 && defined(CPPINTEROP_USE_CLING) && defined(_WIN32)
  GTEST_SKIP() << "Test fails with Cling on Windows";
#endif
#ifdef EMSCRIPTEN
#if CLANG_VERSION_MAJOR > 21
  GTEST_SKIP() << "Test fails for Emscipten builds using LLVM 22";
#endif
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";

  Cpp::JitCall JC = Cpp::MakeFunctionCallable(nullptr);
  EXPECT_TRUE(JC.getKind() == Cpp::JitCall::kUnknown);

  std::vector<Decl*> Decls;
  std::string code = R"(
      typedef struct _name {
        _name() { p[0] = (void*)0x1; p[1] = (void*)0x2; p[2] = (void*)0x3; }
        void* p[3];
      } name;
    )";

  std::vector<const char*> interpreter_args = {"-include", "new"};

  GetAllTopLevelDecls(code, Decls, /*filter_implicitGenerated=*/false,
                      interpreter_args);
  auto CtorD = Cpp::GetDefaultConstructor(Decls[0]);
  auto Ctor = Cpp::MakeFunctionCallable(CtorD);
  EXPECT_TRUE((bool)Ctor) << "Failed to build a wrapper for the ctor";
  void* object = nullptr;
  Ctor.Invoke(&object);
  EXPECT_TRUE(object) << "Failed to call the ctor.";
  // Building a wrapper with a typedef decl must be possible.
  EXPECT_TRUE(Cpp::Destruct(object, Decls[1]));
}

#if !defined(NDEBUG) && GTEST_HAS_DEATH_TEST
#ifndef _WIN32 // Death tests do not work on Windows
TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_JitCallDebug) {

  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";

  std::vector<Decl*> Decls, SubDecls;
  std::string code = R"(
    class C {
      int x;
      C() {
        x = 12345;
      }
    };)";

  std::vector<const char*> interpreter_args = {"-include", "new",
                                               "-debug-only=jitcall"};
  GetAllTopLevelDecls(code, Decls, /*filter_implicitGenerated=*/false,
                      interpreter_args);

  const auto CtorD = Cpp::GetDefaultConstructor(Decls[0]);
  auto JC = Cpp::MakeFunctionCallable(CtorD);

  EXPECT_TRUE(JC.getKind() == Cpp::JitCall::kConstructorCall);
  EXPECT_DEATH(
      { JC.InvokeConstructor(/*result=*/nullptr); },
      "Must pass the location of the created object!");

  void* result = Cpp::Allocate(Decls[0]).data;
  EXPECT_DEATH(
      { JC.InvokeConstructor(&result, 0UL); },
      "Number of objects to construct should be atleast 1");
  // InvokeConstructor below uses is_arena=nullptr and so does its own new[5],
  // overwriting result. Release the throw-away arena first.
  Cpp::Deallocate(Decls[0], result);

  // Succeeds; with is_arena=nullptr and nary>1 the ctor wrapper picks its
  // array-new branch and overwrites result with the new[5] pointer. The
  // matching withFree+nary>1 branch of the dtor wrapper emits delete[].
  JC.InvokeConstructor(&result, 5UL);
  Cpp::Destruct(result, Decls[0], /*withFree=*/true, /*count=*/5);

  Decls.clear();
  code = R"(
    class C {
    public:
      int x;
      C(int a) {
        x = a;
      }
      ~C() {}
    };)";

  GetAllTopLevelDecls(code, Decls, /*filter_implicitGenerated=*/false,
                      interpreter_args);
  GetAllSubDecls(Decls[0], SubDecls);

  EXPECT_TRUE(Cpp::IsConstructor(SubDecls[3]));
  JC = Cpp::MakeFunctionCallable(SubDecls[3]);
  EXPECT_TRUE(JC.getKind() == Cpp::JitCall::kConstructorCall);

  result = Cpp::Allocate(Decls[0], 5).data;
  int i = 42;
  void* args0[1] = {(void*)&i};
  EXPECT_DEATH(
      { JC.InvokeConstructor(&result, 5UL, {args0, 1}); },
      "Cannot pass initialization parameters to array new construction");

  JC.InvokeConstructor(&result, 1UL, {args0, 1}, (void*)~0);

  int* obj = reinterpret_cast<int*>(reinterpret_cast<char*>(result));
  EXPECT_TRUE(*obj == 42);

  // Destructors
  Cpp::DeclRef scope_C = Cpp::GetNamed("C");
  Cpp::ObjectRef object_C = Cpp::Construct(scope_C);

  // Make destructor callable and pass arguments
  JC = Cpp::MakeFunctionCallable(SubDecls[4]);
  EXPECT_DEATH(
      { JC.Invoke(&object_C, {args0, 1}); },
      "Destructor called with arguments");

  // Only slot 0 of the 5-slot arena was placement-constructed above, so
  // destruct a single object and then release the whole arena.
  Cpp::Destruct(result, Decls[0], /*withFree=*/false, /*count=*/0);
  Cpp::Deallocate(Decls[0], result, 5);
  Cpp::Destruct(object_C, scope_C, /*withFree=*/true, /*count=*/0);
}
#endif // _WIN32
#endif

template <typename T> T instantiation_in_host() { return T(0); }
#if defined(_WIN32)
template __declspec(dllexport) int instantiation_in_host<int>();
#elif defined(__GNUC__)
template __attribute__((__visibility__("default"))) int
instantiation_in_host<int>();
#else
template int instantiation_in_host<int>();
#endif

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_GetFunctionCallWrapper) {
#ifdef EMSCRIPTEN
  GTEST_SKIP() << "Test fails for Emscipten builds";
#endif
#if defined(CPPINTEROP_USE_CLING) && defined(_WIN32)
  GTEST_SKIP() << "Disabled, invoking functions containing printf does not work with Cling on Windows";
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";
  std::vector<Decl*> Decls;
  std::string code = R"(
    int f1(int i) { return i * i; }
    )";

  std::vector<const char*> interpreter_args = {"-include", "new"};

  GetAllTopLevelDecls(code, Decls, /*filter_implicitGenerated=*/false,
                      interpreter_args);

  Interp->process(R"(
    #include <string>
    void f2(std::string &s) { printf("%s", s.c_str()); };
  )");

  Interp->process(R"(
    namespace NS {
      int f3() { return 3; }

      extern "C" int f4() { return 4; }

      typedef int(*int_func)(void);
      int_func f5() { return f3; }
    }
  )");

  Cpp::JitCall FCI1 =
      Cpp::MakeFunctionCallable(Decls[0]);
  EXPECT_TRUE(FCI1.getKind() == Cpp::JitCall::kGenericCall);
  Cpp::JitCall FCI2 =
      Cpp::MakeFunctionCallable(Cpp::FuncRef{Cpp::GetNamed("f2").data});
  EXPECT_TRUE(FCI2.getKind() == Cpp::JitCall::kGenericCall);
  Cpp::JitCall FCI3 = Cpp::MakeFunctionCallable(
      Cpp::FuncRef{Cpp::GetNamed("f3", Cpp::GetNamed("NS")).data});
  EXPECT_TRUE(FCI3.getKind() == Cpp::JitCall::kGenericCall);
  Cpp::JitCall FCI4 = Cpp::MakeFunctionCallable(
      Cpp::FuncRef{Cpp::GetNamed("f4", Cpp::GetNamed("NS")).data});
  EXPECT_TRUE(FCI4.getKind() == Cpp::JitCall::kGenericCall);

  int i = 9, ret1, ret3, ret4;
  std::string s("Hello World!\n");
  void *args0[1] = { (void *) &i };
  void *args1[1] = { (void *) &s };

  FCI1.Invoke(&ret1, {args0, /*args_size=*/1});
  EXPECT_EQ(ret1, i * i);

  testing::internal::CaptureStdout();
  FCI2.Invoke({args1, /*args_size=*/1});
  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(output, s);

  FCI3.Invoke(&ret3);
  EXPECT_EQ(ret3, 3);

  FCI4.Invoke(&ret4);
  EXPECT_EQ(ret4, 4);

  Cpp::JitCall FCI5 = Cpp::MakeFunctionCallable(
      Cpp::FuncRef{Cpp::GetNamed("f5", Cpp::GetNamed("NS")).data});
  EXPECT_TRUE(FCI5.getKind() == Cpp::JitCall::kGenericCall);

  typedef int (*int_func)();
  int_func callback = nullptr;
  FCI5.Invoke((void*)&callback);
  EXPECT_TRUE(callback);
  EXPECT_EQ(callback(), 3);

  // FIXME: Do we need to support private ctors?
  Interp->process(R"(
    class C {
    public:
      C() { printf("Default Ctor Called\n"); }
      ~C() { printf("Dtor Called\n"); }
    };
  )");

  auto ClassC = Cpp::GetNamed("C");
  auto CtorD = Cpp::GetDefaultConstructor(ClassC);
  auto FCI_Ctor =
    Cpp::MakeFunctionCallable(CtorD);
  void* object = nullptr;
  testing::internal::CaptureStdout();
  FCI_Ctor.Invoke((void*)&object);
  output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(output, "Default Ctor Called\n");
  EXPECT_TRUE(object);

  auto DtorD = Cpp::GetDestructor(ClassC);
  auto FCI_Dtor =
    Cpp::MakeFunctionCallable(DtorD);
  testing::internal::CaptureStdout();
  FCI_Dtor.Invoke(object);
  output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(output, "Dtor Called\n");

  std::vector<Decl*> Decls1;
  std::string code1 = R"(
        template<typename T>
        struct S {

        static T Add(T a, T b) { return a + b; }

        };
    )";

  GetAllTopLevelDecls(code1, Decls1, /*filter_implicitGenerated=*/false,
                      interpreter_args);
  ASTContext& C = Interp->getCI()->getASTContext();

  std::vector<Cpp::TemplateArgInfo> argument = {C.IntTy.getAsOpaquePtr()};
  auto Instance1 = Cpp::InstantiateTemplate(Decls1[0], argument);
  EXPECT_TRUE(
      isa<ClassTemplateSpecializationDecl>(Cpp::unwrap<Decl>(Instance1)));
  auto* CTSD1 =
      cast<ClassTemplateSpecializationDecl>(Cpp::unwrap<Decl>(Instance1));
  auto Add_D = Cpp::GetNamed("Add", CTSD1);
  Cpp::JitCall FCI_Add = Cpp::MakeFunctionCallable(Cpp::FuncRef{Add_D.data});
  EXPECT_TRUE(FCI_Add.getKind() == Cpp::JitCall::kGenericCall);

  int a = 5, b = 10, result;
  void* args[2] = {(void*)&a, (void*)&b};

  FCI_Add.Invoke(&result, {args, /*args_size=*/2});
  EXPECT_EQ(result, a + b);

  // call with pointers
  Interp->process(R"(
  void set_5(int *out) {
    *out = 5;
  }
  )");

  Cpp::DeclRef set_5 = Cpp::GetNamed("set_5");
  EXPECT_TRUE(set_5);

  Cpp::JitCall set_5_f = Cpp::MakeFunctionCallable(Cpp::FuncRef{set_5.data});
  EXPECT_EQ(set_5_f.getKind(), Cpp::JitCall::kGenericCall);

  int* bp = &b;
  void* set_5_args[1] = {(void*)&bp};
  set_5_f.Invoke(nullptr, {set_5_args, 1});
  EXPECT_EQ(b, 5);

  Interp->process(R"(
  class TypedefToPrivateClass {
  private:
    class PC {
    public:
      int m_val = 4;
    };

  public:
    typedef PC PP;
    static PP f() { return PC(); }
  };
  )");

  Cpp::DeclRef TypedefToPrivateClass = Cpp::GetNamed("TypedefToPrivateClass");
  EXPECT_TRUE(TypedefToPrivateClass);

  Cpp::DeclRef f = Cpp::GetNamed("f", TypedefToPrivateClass);
  EXPECT_TRUE(f);

  Cpp::JitCall FCI_f = Cpp::MakeFunctionCallable(Cpp::FuncRef{f.data});
  EXPECT_EQ(FCI_f.getKind(), Cpp::JitCall::kGenericCall);

  void* res = nullptr;
  FCI_f.Invoke(&res, {nullptr, 0});
  EXPECT_TRUE(res);

  // templated operators
  Interp->process(R"(
    class TOperator{
    public:
      template<typename T>
      bool operator<(T t) { return true; }
    };
  )");
  Cpp::DeclRef TOperator = Cpp::GetNamed("TOperator");

  auto TOperatorCtor = Cpp::GetDefaultConstructor(TOperator);
  auto FCI_TOperatorCtor = Cpp::MakeFunctionCallable(TOperatorCtor);
  void* toperator = nullptr;
  FCI_TOperatorCtor.Invoke((void*)&toperator);

  EXPECT_TRUE(toperator);
  std::vector<Cpp::FuncRef> operators;
  Cpp::GetOperator(TOperator, Cpp::Operator::OP_Less, operators);
  EXPECT_EQ(operators.size(), 1);

  Cpp::DeclRef op_templated{operators[0].data};
  auto TAI = Cpp::TemplateArgInfo(Cpp::GetType("int").data);
  Cpp::DeclRef op = Cpp::InstantiateTemplate(op_templated, {TAI});
  auto FCI_op = Cpp::MakeFunctionCallable(Cpp::FuncRef{op.data});
  bool boolean = false;
  FCI_op.Invoke((void*)&boolean, {args, /*args_size=*/1}, toperator);
  EXPECT_TRUE(boolean);
  Cpp::Destruct(toperator, TOperator, /*withFree=*/true, /*count=*/0);

  Interp->process(R"(
    namespace N1 {

    template <typename... _Tn> struct Klass3 {
      using type = int;
    };
    
    namespace N2 {
    template <typename T1, typename T2> class Klass1 {};

    template <typename T1, typename T2> class Klass2 {};

    template <typename T1, typename T2, typename T3, typename T4>
    constexpr Klass2<
      T1, typename Klass3<T2, Klass1<T3, T4>>::type>
    operator+(const Klass2<T1, T2> &__lhs,
              const Klass1<T3, T4> &__rhs) {
      typedef Klass1<T3, T4> __T1;
      typedef typename Klass3<T2, __T1>::type __ct;
      typedef Klass2<T1, __ct> __T2;
      return {};
    }
    } // namespace N2
    } // namespace N1
  
    N1::N2::Klass2<int, double> K1;
    N1::N2::Klass1<char, float> K2;
  )");

  Cpp::TypeRef K1 = Cpp::GetTypeFromScope(Cpp::GetNamed("K1"));
  Cpp::TypeRef K2 = Cpp::GetTypeFromScope(Cpp::GetNamed("K2"));
  operators.clear();
  Cpp::GetOperator(Cpp::GetScope("N2", Cpp::GetScope("N1")),
                   Cpp::Operator::OP_Plus, operators);
  EXPECT_EQ(operators.size(), 1);
  Cpp::FuncRef kop = Cpp::BestOverloadFunctionMatch(operators, empty_templ_args,
                                                    {K1.data, K2.data});
  auto chrono_op_fn_callable = Cpp::MakeFunctionCallable(kop);
  EXPECT_EQ(chrono_op_fn_callable.getKind(), Cpp::JitCall::kGenericCall);

  Interp->process(R"(
    namespace my_std {
    template <typename T1, typename T2>
    struct pair {
        T1 first;
        T2 second;
        pair(T1 fst, T2 snd) : first(fst), second(snd) {}
    };

    template <typename T1, typename T2>
    pair<T1, T2> make_pair(T1 x, T2 y) {
        return pair<T1, T2>(x, y);
    }

    template <int _Int>
    struct __pair_get;

    template <>
    struct __pair_get<0> {
        template <typename T1, typename T2>
        static constexpr T1 __get(pair<T1, T2> __pair) noexcept {
            return __pair.first;
        }
    };

    template <>
    struct __pair_get<1> {
        template <typename T1, typename T2>
        constexpr T2 __get(pair<T1, T2> __pair) noexcept {
            return __pair.second;
        }
    };

    template <int N, typename T1, typename T2>
    static constexpr auto get(pair<T1, T2> __t) noexcept {
        return __pair_get<N>::__get(__t);
    }
    }  // namespace my_std

    namespace libchemist {
    namespace type {
    template <typename T>
    class tensor {};
    }  // namespace type

    template <typename element_type = double,
              typename tensor_type = type::tensor<element_type>>
    class CanonicalMO {};

    template class CanonicalMO<double, type::tensor<double>>;

    auto produce() { return my_std::make_pair(10., type::tensor<double>{}); }

    }  // namespace libchemist

    namespace property_types {
    namespace type {
    template <typename T>
    using canonical_mos = libchemist::CanonicalMO<T>;
    }

    auto produce() { return my_std::make_pair(5., type::canonical_mos<double>{}); }
    }  // namespace property_types

    auto tmp = property_types::produce();
    auto &p = tmp;
  )");

  std::vector<Cpp::FuncRef> unresolved_candidate_methods;
  Cpp::GetClassTemplatedMethods("get", Cpp::GetScope("my_std"),
                                unresolved_candidate_methods);
  Cpp::TypeRef p = Cpp::GetTypeFromScope(Cpp::GetNamed("p"));
  EXPECT_TRUE(p);

  Cpp::FuncRef fn = Cpp::BestOverloadFunctionMatch(
      unresolved_candidate_methods, {{Cpp::GetType("int").data, "0"}},
      {p.data});
  EXPECT_TRUE(fn);

  auto fn_callable = Cpp::MakeFunctionCallable(fn);
  EXPECT_EQ(fn_callable.getKind(), Cpp::JitCall::kGenericCall);

  Interp->process(R"(
    template <typename T>
    bool call_move(T&& t) {
      return true;
    }
  )");

  unresolved_candidate_methods.clear();
  Cpp::GetClassTemplatedMethods("call_move", Cpp::GetGlobalScope(),
                                unresolved_candidate_methods);
  EXPECT_EQ(unresolved_candidate_methods.size(), 1);

  Cpp::FuncRef call_move = Cpp::BestOverloadFunctionMatch(
      unresolved_candidate_methods, {},
      {Cpp::GetReferencedType(Cpp::GetType("int"), true).data});
  EXPECT_TRUE(call_move);

  auto call_move_callable = Cpp::MakeFunctionCallable(call_move);
  EXPECT_EQ(call_move_callable.getKind(), Cpp::JitCall::kGenericCall);

  // instantiation in host, no template function body
  Interp->process("template<typename T> T instantiation_in_host();");

  unresolved_candidate_methods.clear();
  Cpp::GetClassTemplatedMethods("instantiation_in_host", Cpp::GetGlobalScope(),
                                unresolved_candidate_methods);
  EXPECT_EQ(unresolved_candidate_methods.size(), 1);

  Cpp::FuncRef instantiation_in_host = Cpp::BestOverloadFunctionMatch(
      unresolved_candidate_methods, {Cpp::GetType("int").data}, {});
  EXPECT_TRUE(instantiation_in_host);

  Cpp::JitCall instantiation_in_host_callable =
      Cpp::MakeFunctionCallable(instantiation_in_host);
  EXPECT_EQ(instantiation_in_host_callable.getKind(),
            Cpp::JitCall::kGenericCall);

  instantiation_in_host = Cpp::BestOverloadFunctionMatch(
      unresolved_candidate_methods, {Cpp::GetType("double").data}, {});
  EXPECT_TRUE(instantiation_in_host);

  Cpp::BeginStdStreamCapture(Cpp::CaptureStreamKind::kStdErr);
  instantiation_in_host_callable =
      Cpp::MakeFunctionCallable(instantiation_in_host);
  std::string err_msg = Cpp::EndStdStreamCapture();
  EXPECT_TRUE(err_msg.find("instantiation with no body") != std::string::npos);
  EXPECT_EQ(instantiation_in_host_callable.getKind(),
            Cpp::JitCall::kUnknown); // expect to fail

  Interp->process(R"(
    template<typename ...T>
    struct tuple {};

    template<typename ...Ts, typename ...TTs>
    void tuple_tuple(tuple<Ts...> one, tuple<TTs...> two) { return; }

    tuple<int, double> tuple_one;
    tuple<char, char> tuple_two;
  )");

  unresolved_candidate_methods.clear();
  Cpp::GetClassTemplatedMethods("tuple_tuple", Cpp::GetGlobalScope(),
                                unresolved_candidate_methods);
  EXPECT_EQ(unresolved_candidate_methods.size(), 1);

  Cpp::FuncRef tuple_tuple = Cpp::BestOverloadFunctionMatch(
      unresolved_candidate_methods, {},
      {Cpp::GetVariableType(Cpp::GetNamed("tuple_one")).data,
       Cpp::GetVariableType(Cpp::GetNamed("tuple_two")).data});
  EXPECT_TRUE(tuple_tuple);

  auto tuple_tuple_callable = Cpp::MakeFunctionCallable(tuple_tuple);
  EXPECT_EQ(tuple_tuple_callable.getKind(), Cpp::JitCall::kGenericCall);

  Interp->process(R"(
    namespace EnumFunctionSameName {
    enum foo { FOO = 42 };
    void foo() {}
    unsigned int bar(enum foo f) { return (unsigned int)f; }
    }
  )");

  Cpp::DeclRef bar =
      Cpp::GetNamed("bar", Cpp::GetScope("EnumFunctionSameName"));
  EXPECT_TRUE(bar);

  auto bar_callable = Cpp::MakeFunctionCallable(Cpp::FuncRef{bar.data});
  EXPECT_EQ(bar_callable.getKind(), Cpp::JitCall::kGenericCall);

  Cpp::Declare(R"(
  struct A {
    A() {}
    A(A&& other) {};
  };

  A consumable;

  template<typename T>
  void consume(T t) {}
  )");

  unresolved_candidate_methods.clear();
  Cpp::GetClassTemplatedMethods("consume", Cpp::GetGlobalScope(),
                                unresolved_candidate_methods);
  EXPECT_EQ(unresolved_candidate_methods.size(), 1);

  Cpp::FuncRef consume = Cpp::BestOverloadFunctionMatch(
      unresolved_candidate_methods, {},
      {Cpp::GetVariableType(Cpp::GetNamed("consumable")).data});
  EXPECT_TRUE(consume);

  auto consume_callable = Cpp::MakeFunctionCallable(consume);
  EXPECT_EQ(consume_callable.getKind(), Cpp::JitCall::kGenericCall);

  Cpp::Declare(R"(
  template<typename T1, typename T2>
  struct Product {};

  template<typename T>
  struct KlassProduct {
    template<typename O>
    const Product<T, O>
    operator*(const KlassProduct<O> &other) const { return Product<T, O>(); }

    template<typename TT, typename O>
    const Product<TT, O>
    operator*(const KlassProduct<O> &other) const { return Product<TT, O>(); }
  };
  )");

  Cpp::DeclRef KlassProduct = Cpp::GetNamed("KlassProduct");
  EXPECT_TRUE(KlassProduct);

  Cpp::DeclRef KlassProduct_int = Cpp::InstantiateTemplate(KlassProduct, {TAI});
  EXPECT_TRUE(KlassProduct_int);
  TAI = Cpp::TemplateArgInfo(Cpp::GetType("float").data);
  Cpp::DeclRef KlassProduct_float =
      Cpp::InstantiateTemplate(KlassProduct, {TAI});
  EXPECT_TRUE(KlassProduct_float);

  operators.clear();
  Cpp::GetOperator(KlassProduct_int, Cpp::Operator::OP_Star, operators);
  EXPECT_EQ(operators.size(), 2);

  Cpp::FuncRef op2 = Cpp::BestOverloadFunctionMatch(
      operators, {}, {{Cpp::GetTypeFromScope(KlassProduct_float).data}});
  EXPECT_TRUE(op2);

  auto op_callable = Cpp::MakeFunctionCallable(op2);
  EXPECT_EQ(op_callable.getKind(), Cpp::JitCall::kGenericCall);

  Cpp::Declare(R"(
    enum class MyEnum { A, B, C };
    template <MyEnum E>
    class TemplatedEnum {};

    namespace MyNameSpace {
    enum class MyEnum { A, B, C };
    template <MyEnum E>
    class TemplatedEnum {};
    }
  )");

  Cpp::DeclRef TemplatedEnum = Cpp::GetScope("TemplatedEnum");
  EXPECT_TRUE(TemplatedEnum);

  auto TAI_enum = Cpp::TemplateArgInfo(
      Cpp::GetTypeFromScope(Cpp::GetNamed("MyEnum")).data, "1");
  Cpp::DeclRef TemplatedEnum_instantiated =
      Cpp::InstantiateTemplate(TemplatedEnum, {TAI_enum});
  EXPECT_TRUE(TemplatedEnum_instantiated);

  Cpp::ObjectRef obj = Cpp::Construct(TemplatedEnum_instantiated);
  EXPECT_TRUE(obj);
  Cpp::Destruct(obj, TemplatedEnum_instantiated);
  obj = nullptr;

  Cpp::DeclRef MyNameSpace_TemplatedEnum =
      Cpp::GetScope("TemplatedEnum", Cpp::GetScope("MyNameSpace"));
  EXPECT_TRUE(TemplatedEnum);

  TAI_enum = Cpp::TemplateArgInfo(
      Cpp::GetTypeFromScope(
          Cpp::GetNamed("MyEnum", Cpp::GetScope("MyNameSpace")))
          .data,
      "1");
  Cpp::DeclRef MyNameSpace_TemplatedEnum_instantiated =
      Cpp::InstantiateTemplate(MyNameSpace_TemplatedEnum, {TAI_enum});
  EXPECT_TRUE(TemplatedEnum_instantiated);

  obj = Cpp::Construct(MyNameSpace_TemplatedEnum_instantiated);
  EXPECT_TRUE(obj);
  Cpp::Destruct(obj, MyNameSpace_TemplatedEnum_instantiated);
  obj = nullptr;

  Cpp::Declare(R"(
    auto get_fn(int x) { return [x](int y){ return x + y; }; }
  )");

  Cpp::DeclRef get_fn = Cpp::GetNamed("get_fn");
  EXPECT_TRUE(get_fn);

  auto get_fn_callable = Cpp::MakeFunctionCallable(Cpp::FuncRef{get_fn.data});
  EXPECT_EQ(get_fn_callable.getKind(), Cpp::JitCall::kGenericCall);

  EXPECT_TRUE(Cpp::IsLambdaClass(
      Cpp::GetFunctionReturnType(Cpp::FuncRef{get_fn.data})));
  EXPECT_FALSE(
      Cpp::IsLambdaClass(Cpp::GetFunctionReturnType(Cpp::FuncRef{bar.data})));

  Cpp::Declare(R"(
    template <typename F>
    void callback(F&&) {}

    int (*fn_ptr)(int, int) = nullptr;
  )");

  unresolved_candidate_methods.clear();
  Cpp::GetClassTemplatedMethods("callback", Cpp::GetGlobalScope(),
                                unresolved_candidate_methods);
  EXPECT_EQ(unresolved_candidate_methods.size(), 1);

  Cpp::FuncRef callback_func = Cpp::BestOverloadFunctionMatch(
      unresolved_candidate_methods, {},
      {Cpp::GetVariableType(Cpp::GetNamed("fn_ptr")).data});
  EXPECT_TRUE(callback_func);

  auto callback_callable = Cpp::MakeFunctionCallable(callback_func);
  EXPECT_EQ(callback_callable.getKind(), Cpp::JitCall::kGenericCall);

  Decls.clear();
  GetAllTopLevelDecls(R"(
    bool callme(bool (f)()) { return f(); }
    bool callme_ptr(bool (*f)()) { return f(); }
    bool callme_ref(bool (&f)()) { return f(); }
  )",
                      Decls, false, interpreter_args);

  auto f1 = Cpp::MakeFunctionCallable(Cpp::FuncRef{Decls[0]});
  EXPECT_EQ(f1.getKind(), Cpp::JitCall::Kind::kGenericCall);

  auto f2 = Cpp::MakeFunctionCallable(Cpp::FuncRef{Decls[1]});
  EXPECT_EQ(f2.getKind(), Cpp::JitCall::Kind::kGenericCall);

  auto f3 = Cpp::MakeFunctionCallable(Cpp::FuncRef{Decls[2]});
  EXPECT_EQ(f3.getKind(), Cpp::JitCall::Kind::kGenericCall);
}

TYPED_TEST(CPPINTEROP_TEST_MODE,
           FunctionReflection_WrapAliasTemplateReturnType) {
  // Regression test for cppyy issue
  // https://github.com/compiler-research/cppyy/issues/218 (original reproducer:
  // `std::make_any<...>`, return type `std::enable_if_t<is_constructible_v<...>,
  // std::any>`). Building a wrapper for a function template whose return type is
  // a type-alias-template specialisation used to fail to compile; the snippet
  // below is a stdlib-free distillation. It needs three ingredients: an alias
  // (`enable_if_t`) whose sugar carries a non-type argument that prints as an
  // *expression* (`trait_v<int>`, which FullyQualifiedName does not qualify); a
  // predicate in a namespace, so unqualified `trait_v` does not resolve in the
  // global-scope wrapper; and a class (non-builtin) canonical type, so
  // get_type_as_string keeps the sugar instead of taking its isBuiltinType()
  // branch. See get_type_as_string for how the fix desugars this.
  //
  // Only checks JitCall::getKind(), i.e. that the wrapper *compiles*; it never
  // Invoke()s it, so no out-of-process skip is needed.
#ifdef EMSCRIPTEN
#if CLANG_VERSION_MAJOR > 21
  // The Emscripten JIT (LLVM 22) cannot compile this by-value wrapper -- a
  // separate limitation from the type printing under test, shared with the
  // other placement-new wrapper tests (e.g. FunctionReflection_Construct).
  GTEST_SKIP() << "Test fails for Emscripten builds using LLVM 22";
#endif
#endif
  std::vector<const char*> interpreter_args = {"-std=c++17", "-include", "new"};
  TestFixture::CreateInterpreter(interpreter_args);
  // Single namespace block: a preceding top-level class plus a namespace in one
  // process() call makes the LLVM 20 REPL wrap the namespace into non-global
  // scope.
  Interp->process(R"(
    namespace N {
      struct Ret { int x; };
      template <class T> inline constexpr bool trait_v = sizeof(T) > 0;
      template <bool, class T> struct ei {};
      template <class T> struct ei<true, T> { using type = T; };
      template <bool B, class T> using enable_if_t = typename ei<B, T>::type;
      template <class T> enable_if_t<trait_v<T>, Ret> f() { return {}; }
    }
  )");

  auto spec = Cpp::InstantiateTemplateFunctionFromString("N::f<int>");
  ASSERT_TRUE(spec) << "Sema failed to substitute N::f<int>";

  // Without the fix the wrapper fails to compile (`use of undeclared identifier
  // 'trait_v'`) and MakeFunctionCallable returns a kUnknown JitCall.
  Cpp::JitCall JC = Cpp::MakeFunctionCallable(spec);
  EXPECT_EQ(JC.getKind(), Cpp::JitCall::kGenericCall);
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_IsConstMethod) {
  std::vector<Decl*> Decls, SubDecls;
  std::string code = R"(
    class C {
      void f1() const {}
      void f2() {}
    };
    )";

  GetAllTopLevelDecls(code, Decls);
  GetAllSubDecls(Decls[0], SubDecls);
  Cpp::FuncRef method = nullptr; // Simulate an invalid method pointer

  EXPECT_TRUE(Cpp::IsConstMethod(SubDecls[1]));  // f1
  EXPECT_FALSE(Cpp::IsConstMethod(SubDecls[2])); // f2
  EXPECT_FALSE(Cpp::IsConstMethod(method));
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_GetFunctionArgName) {
  std::vector<Decl*> Decls;
  std::string code = R"(
    void f1(int i, double d, long l, char ch) {}
    void f2(const int i, double d[], long *l, char ch[4]) {}

    template<class A, class B>
    long get_size(A, B, int i = 0) {}

    template<class A = int, class B = char>
    long get_size(int i, A a = A(), B b = B()) {}

    template<class A>
    void get_size(long k, A, char ch = 'a', double l = 0.0) {}
    )";

  GetAllTopLevelDecls(code, Decls);
  EXPECT_EQ(Cpp::GetFunctionArgName(Decls[0], 0), "i");
  EXPECT_EQ(Cpp::GetFunctionArgName(Decls[0], 1), "d");
  EXPECT_EQ(Cpp::GetFunctionArgName(Decls[0], 2), "l");
  EXPECT_EQ(Cpp::GetFunctionArgName(Decls[0], 3), "ch");
  EXPECT_EQ(Cpp::GetFunctionArgName(Decls[1], 0), "i");
  EXPECT_EQ(Cpp::GetFunctionArgName(Decls[1], 1), "d");
  EXPECT_EQ(Cpp::GetFunctionArgName(Decls[1], 2), "l");
  EXPECT_EQ(Cpp::GetFunctionArgName(Decls[1], 3), "ch");

  EXPECT_EQ(Cpp::GetFunctionArgName(Decls[2], 0), "");
  EXPECT_EQ(Cpp::GetFunctionArgName(Decls[2], 1), "");
  EXPECT_EQ(Cpp::GetFunctionArgName(Decls[2], 2), "i");

  EXPECT_EQ(Cpp::GetFunctionArgName(Decls[3], 0), "i");
  EXPECT_EQ(Cpp::GetFunctionArgName(Decls[3], 1), "a");
  EXPECT_EQ(Cpp::GetFunctionArgName(Decls[3], 2), "b");

  EXPECT_EQ(Cpp::GetFunctionArgName(Decls[4], 0), "k");
  EXPECT_EQ(Cpp::GetFunctionArgName(Decls[4], 1), "");
  EXPECT_EQ(Cpp::GetFunctionArgName(Decls[4], 2), "ch");
  EXPECT_EQ(Cpp::GetFunctionArgName(Decls[4], 3), "l");
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_GetFunctionArgDefault) {
  std::vector<Decl*> Decls;
  std::string code = R"(
    void f1(int i, double d = 4.0, const char *s = "default", char ch = 'c') {}
    void f2(float i = 0.0, double d = 3.123, long m = 34126) {}

    template<class A, class B>
    long get_size(A, B, int i = 0) {}

    template<class A = int, class B = char>
    long get_size(int i, A a = A(), B b = B()) {}

    template<class A>
    void get_size(long k, A, char ch = 'a', double l = 0.0) {}

    template<typename T>
    struct Other {};

    template <typename T, typename S = Other<T>>
    struct MyStruct {
      T t;
      S s;
      void fn(T t, S s = S()) {}
    };
    )";

  GetAllTopLevelDecls(code, Decls);

  EXPECT_EQ(Cpp::GetFunctionArgDefault(Decls[0], 0), "");
  EXPECT_EQ(Cpp::GetFunctionArgDefault(Decls[0], 1), "4.");
  EXPECT_EQ(Cpp::GetFunctionArgDefault(Decls[0], 2), "\"default\"");
  EXPECT_EQ(Cpp::GetFunctionArgDefault(Decls[0], 3), "\'c\'");
  EXPECT_EQ(Cpp::GetFunctionArgDefault(Decls[1], 0), "0.");
  EXPECT_EQ(Cpp::GetFunctionArgDefault(Decls[1], 1), "3.123");
  EXPECT_EQ(Cpp::GetFunctionArgDefault(Decls[1], 2), "34126");

  EXPECT_EQ(Cpp::GetFunctionArgDefault(Decls[2], 0), "");
  EXPECT_EQ(Cpp::GetFunctionArgDefault(Decls[2], 1), "");
  EXPECT_EQ(Cpp::GetFunctionArgDefault(Decls[2], 2), "0");

  EXPECT_EQ(Cpp::GetFunctionArgDefault(Decls[3], 0), "");
  EXPECT_EQ(Cpp::GetFunctionArgDefault(Decls[3], 1), "A()");
  EXPECT_EQ(Cpp::GetFunctionArgDefault(Decls[3], 2), "B()");

  EXPECT_EQ(Cpp::GetFunctionArgDefault(Decls[4], 0), "");
  EXPECT_EQ(Cpp::GetFunctionArgDefault(Decls[4], 1), "");
  EXPECT_EQ(Cpp::GetFunctionArgDefault(Decls[4], 2), "\'a\'");
  EXPECT_EQ(Cpp::GetFunctionArgDefault(Decls[4], 3), "0.");

  ASTContext& C = Interp->getCI()->getASTContext();
  std::vector<Cpp::TemplateArgInfo> template_args = {C.IntTy.getAsOpaquePtr()};
  Cpp::DeclRef my_struct = Cpp::InstantiateTemplate(Decls[6], template_args);
  EXPECT_TRUE(my_struct);

  std::vector<Cpp::FuncRef> fns = Cpp::GetFunctionsUsingName(my_struct, "fn");
  EXPECT_EQ(fns.size(), 1);

  Cpp::FuncRef fn = fns[0];
  EXPECT_EQ(Cpp::GetFunctionArgDefault(fn, 0), "");
  EXPECT_EQ(Cpp::GetFunctionArgDefault(fn, 1), "S()");
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_Construct) {
#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif
#ifdef EMSCRIPTEN
#if CLANG_VERSION_MAJOR > 21
  GTEST_SKIP() << "Test fails for Emscipten builds using LLVM 22";
#endif
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";
  std::vector<const char*> interpreter_args = {"-include", "new"};
  std::vector<Decl*> Decls, SubDecls;

  std::string code = R"(
    #include <new>
    extern "C" int printf(const char*,...);
    class C {
    public:
      int x;
      C() {
        x = 12345;
        printf("Constructor Executed");
      }
    };
    void construct() { return; }
    )";

  GetAllTopLevelDecls(code, Decls, false, interpreter_args);
  GetAllSubDecls(Decls[1], SubDecls);
  testing::internal::CaptureStdout();
  Cpp::DeclRef scope = Cpp::GetNamed("C");
  Cpp::ObjectRef object = Cpp::Construct(scope);
  EXPECT_TRUE(object);
  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(output, "Constructor Executed");
  output.clear();
  // Construct(scope, arena=nullptr) new-expressions the object; release it.
  Cpp::Destruct(object, scope, /*withFree=*/true, /*count=*/0);

  // Placement.
  testing::internal::CaptureStdout();
  void* where = Cpp::Allocate(scope).data;
  EXPECT_TRUE(where == Cpp::Construct(scope, where).data);
  // Check for the value of x which should be at the start of the object.
  EXPECT_TRUE(*(int*)where == 12345);
  output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(output, "Constructor Executed");
  output.clear();
  Cpp::Destruct(where, scope, /*withFree=*/false, /*count=*/0);
  Cpp::Deallocate(scope, where);

  // Pass a constructor
  testing::internal::CaptureStdout();
  where = Cpp::Allocate(scope).data;
  EXPECT_TRUE(where == Cpp::Construct(SubDecls[3], where).data);
  EXPECT_TRUE(*(int*)where == 12345);
  output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(output, "Constructor Executed");
  output.clear();
  Cpp::Destruct(where, scope, /*withFree=*/false, /*count=*/0);
  Cpp::Deallocate(scope, where);

  // Pass a non-class decl, this should fail. Capture the failing
  // Construct's nullptr in a separate variable so the arena pointer in
  // `where` stays alive for Deallocate. FIXME: Construct's failure path
  // could own the arena release itself rather than leaking this contract
  // to every caller — see Cpp::Construct in lib/CppInterOp/CppInterOp.cpp.
  where = Cpp::Allocate(scope).data;
  auto construct_fail = Cpp::Construct(Decls[2], where);
  EXPECT_FALSE(construct_fail);
  Cpp::Deallocate(scope, where);
}

// The wrappers behind Construct and by-value returns placement-new into a
// caller-provided buffer. A class-scope operator new with no placement form
// hides the global `operator new(size_t, void*)`, so those wrappers only
// compile if they spell it `::new (buf) C(...)`.
TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_ConstructClassScopeNew) {
#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif
#ifdef EMSCRIPTEN
#if CLANG_VERSION_MAJOR > 21
  GTEST_SKIP() << "Test fails for Emscipten builds using LLVM 22";
#endif
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";
  std::vector<const char*> interpreter_args = {"-include", "new"};
  std::vector<Decl*> Decls;

  std::string code = R"(
    class WithClassNew {
    public:
      int x;
      WithClassNew() : x(42) {}
      static void* operator new(__SIZE_TYPE__ sz) { return ::operator new(sz); }
      static void* operator new[](__SIZE_TYPE__ sz) {
        return ::operator new[](sz);
      }
      static void operator delete(void* p) { ::operator delete(p); }
      static void operator delete[](void* p) { ::operator delete[](p); }
    };
    WithClassNew MakeWithClassNew() { return WithClassNew(); }
    )";

  GetAllTopLevelDecls(code, Decls, false, interpreter_args);
  Cpp::DeclRef scope = Cpp::GetNamed("WithClassNew");
  ASSERT_TRUE(scope);

  // Heap construction; the non-arena branch of the wrapper must keep using
  // the unqualified `new` so it picks up the class-scope operator new.
  Cpp::ObjectRef object = Cpp::Construct(scope);
  ASSERT_TRUE(object);
  EXPECT_EQ(*static_cast<int*>(object.data), 42);
  Cpp::Destruct(object, scope, /*withFree=*/true, /*count=*/0);

  // Placement construction into an arena.
  void* where = Cpp::Allocate(scope).data;
  ASSERT_TRUE(where);
  EXPECT_TRUE(where == Cpp::Construct(scope, where).data);
  EXPECT_EQ(*static_cast<int*>(where), 42);
  Cpp::Destruct(where, scope, /*withFree=*/false, /*count=*/0);
  Cpp::Deallocate(scope, where);

  // Placement construction of an array (the wrapper's `nary > 1` branch).
  constexpr size_t count = 3;
  const size_t size = Cpp::SizeOf(scope);
  where = Cpp::Allocate(scope, count).data;
  ASSERT_TRUE(where);
  EXPECT_TRUE(where == Cpp::Construct(scope, where, count).data);
  for (size_t i = 0; i < count; ++i)
    EXPECT_EQ(*reinterpret_cast<int*>(static_cast<char*>(where) + (i * size)), 42);
  Cpp::Destruct(where, scope, /*withFree=*/false, count);
  Cpp::Deallocate(scope, where, count);

  // A by-value return placement-news the result into `ret`
  // (make_narg_call_with_return); this must also bypass the class-scope
  // operator new.
  Cpp::JitCall JC = Cpp::MakeFunctionCallable(Decls[1]);
  ASSERT_TRUE(JC.getKind() == Cpp::JitCall::kGenericCall);
  int result = 0; // WithClassNew's layout is a single int
  JC.Invoke(&result);
  EXPECT_EQ(result, 42);
}

// Test zero initialization of PODs and default initialization cases
TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_ConstructPOD) {
#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif
#ifdef EMSCRIPTEN
#if CLANG_VERSION_MAJOR > 21
  GTEST_SKIP() << "Test fails for Emscipten builds using LLVM 22";
#endif
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";
  std::vector<const char*> interpreter_args = {"-include", "new"};
  TestFixture::CreateInterpreter(interpreter_args);

  Interp->declare(R"(
    namespace PODS {
      struct SomePOD_B {
          int fInt;
      };
      struct SomePOD_C {
          int fInt;
          double fDouble;
      };
    })");

  auto ns = Cpp::GetNamed("PODS");
  Cpp::DeclRef scope = Cpp::GetNamed("SomePOD_B", ns);
  EXPECT_TRUE(scope);
  Cpp::ObjectRef object = Cpp::Construct(scope);
  EXPECT_TRUE(object);
  int* fInt = reinterpret_cast<int*>(reinterpret_cast<char*>(object.data));
  EXPECT_TRUE(*fInt == 0);
  Cpp::Destruct(object, scope, /*withFree=*/true, /*count=*/0);

  scope = Cpp::GetNamed("SomePOD_C", ns);
  EXPECT_TRUE(scope);
  object = Cpp::Construct(scope);
  EXPECT_TRUE(object);
  auto* fDouble = reinterpret_cast<double*>(
      reinterpret_cast<char*>(object.data) + sizeof(int));
  EXPECT_EQ(*fDouble, 0.0);
  Cpp::Destruct(object, scope, /*withFree=*/true, /*count=*/0);
}

// Test nested constructor calls
TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_ConstructNested) {
#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif
#ifdef EMSCRIPTEN
#if CLANG_VERSION_MAJOR > 21
  GTEST_SKIP() << "Test fails for Emscipten builds using LLVM 22";
#endif
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";

  std::vector<const char*> interpreter_args = {"-include", "new"};
  TestFixture::CreateInterpreter(interpreter_args);

  Interp->declare(R"(
    #include <new>
    extern "C" int printf(const char*,...);
    class A {
    public:
      int a_val;
      A() : a_val(7) {
        printf("A Constructor Called\n");
      }
    };

    class B {
    public:
      A a;
      int b_val;
      B() : b_val(99) {
        printf("B Constructor Called\n");
      }
    };
    )");

  testing::internal::CaptureStdout();
  Cpp::DeclRef scope_A = Cpp::GetNamed("A");
  Cpp::DeclRef scope_B = Cpp::GetNamed("B");
  Cpp::ObjectRef object = Cpp::Construct(scope_B);
  EXPECT_TRUE(object);
  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(output, "A Constructor Called\nB Constructor Called\n");
  output.clear();
  Cpp::Destruct(object, scope_B, /*withFree=*/true, /*count=*/0);

  // In-memory construction
  testing::internal::CaptureStdout();
  void* arena = Cpp::Allocate(scope_B).data;
  EXPECT_TRUE(arena == Cpp::Construct(scope_B, arena).data);

  // Check if both integers a_val and b_val were set.
  EXPECT_EQ(*(int*)arena, 7);
  size_t a_size = Cpp::SizeOf(scope_A);
  int* b_val_ptr =
      reinterpret_cast<int*>(reinterpret_cast<char*>(arena) + a_size);
  EXPECT_EQ(*b_val_ptr, 99);
  Cpp::Deallocate(scope_B, arena);
  output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(output, "A Constructor Called\nB Constructor Called\n");
  output.clear();
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_ConstructArray) {
#if defined(EMSCRIPTEN)
  GTEST_SKIP() << "Test fails for Emscripten builds";
#endif
#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";

  TestFixture::CreateInterpreter();

  Interp->declare(R"(
      #include <new>
      extern "C" int printf(const char*,...);
      class C {
        int x;
        C() {
          x = 42;
          printf("\nConstructor Executed\n");
        }
      };
      )");

  Cpp::DeclRef scope = Cpp::GetNamed("C");
  std::string output;

  size_t a = 5;                          // Construct an array of 5 objects
  void* where = Cpp::Allocate(scope, a).data; // operator new

  testing::internal::CaptureStdout();
  EXPECT_TRUE(where == Cpp::Construct(scope, where, a).data); // placement new
  // Check for the value of x which should be at the start of the object.
  EXPECT_TRUE(*(int*)where == 42);
  // Check for the value of x in the second object
  int* obj = reinterpret_cast<int*>(reinterpret_cast<char*>(where) +
                                    Cpp::SizeOf(scope));
  EXPECT_TRUE(*obj == 42);

  // Check for the value of x in the last object
  obj = reinterpret_cast<int*>(reinterpret_cast<char*>(where) +
                               (Cpp::SizeOf(scope) * 4));
  EXPECT_TRUE(*obj == 42);
  EXPECT_TRUE(Cpp::Destruct(where, scope, /*withFree=*/false, 5));
  Cpp::Deallocate(scope, where, 5);
  output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(output,
            "\nConstructor Executed\n\nConstructor Executed\n\nConstructor "
            "Executed\n\nConstructor Executed\n\nConstructor Executed\n");
  output.clear();
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_Destruct) {
#ifdef EMSCRIPTEN
  GTEST_SKIP() << "Test fails for Emscipten builds";
#endif

#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";

  std::vector<const char*> interpreter_args = {"-include", "new"};
  TestFixture::CreateInterpreter(interpreter_args);

  Interp->declare(R"(
    #include <new>
    extern "C" int printf(const char*,...);
    class C {
      C() {}
      ~C() {
        printf("Destructor Executed");
      }
    };
    )");

  testing::internal::CaptureStdout();
  Cpp::DeclRef scope = Cpp::GetNamed("C");
  Cpp::ObjectRef object = Cpp::Construct(scope);
  EXPECT_TRUE(Cpp::Destruct(object, scope));
  std::string output = testing::internal::GetCapturedStdout();

  EXPECT_EQ(output, "Destructor Executed");

  output.clear();
  testing::internal::CaptureStdout();
  object = Cpp::Construct(scope);
  // Make sure we do not call delete by adding an explicit Deallocate. If we
  // called delete the Deallocate will cause a double deletion error.
  EXPECT_TRUE(Cpp::Destruct(object, scope, /*withFree=*/false));
  Cpp::Deallocate(scope, object);
  output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(output, "Destructor Executed");

  // C API
  testing::internal::CaptureStdout();

  // Failure Test, this wrapper should not compile since we explicitly delete
  // the destructor
  Interp->declare(R"(
  class D {
    public:
      D() {}
      ~D() = delete;
  };
  )");

  scope = Cpp::GetNamed("D");
  object = Cpp::Construct(scope);
  EXPECT_FALSE(Cpp::Destruct(object, scope));
  testing::internal::GetCapturedStdout();
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_DestructArray) {
#ifdef EMSCRIPTEN
  GTEST_SKIP() << "Test fails for Emscipten builds";
#endif

#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";

  std::vector<const char*> interpreter_args = {"-include", "new"};
  TestFixture::CreateInterpreter(interpreter_args);

  Interp->declare(R"(
      #include <new>
      extern "C" int printf(const char*,...);
      class C {
        int x;
        C() {
          printf("\nCtor Executed\n");
          x = 42;
        }
        ~C() {
          printf("\nDestructor Executed\n");
        }
      };
      )");

  Cpp::DeclRef scope = Cpp::GetNamed("C");
  std::string output;

  size_t a = 5;                          // Construct an array of 5 objects
  void* where = Cpp::Allocate(scope, a).data;                 // operator new
  EXPECT_TRUE(where == Cpp::Construct(scope, where, a).data); // placement new

  // verify the array of objects has been constructed
  int* obj = reinterpret_cast<int*>(reinterpret_cast<char*>(where) +
                                    Cpp::SizeOf(scope) * 4);
  EXPECT_TRUE(*obj == 42);

  testing::internal::CaptureStdout();
  // destruct 3 out of 5 objects
  EXPECT_TRUE(Cpp::Destruct(where, scope, false, 3));
  output = testing::internal::GetCapturedStdout();

  EXPECT_EQ(
      output,
      "\nDestructor Executed\n\nDestructor Executed\n\nDestructor Executed\n");
  output.clear();
  testing::internal::CaptureStdout();

  // destruct the rest
  auto *new_head = reinterpret_cast<void*>(reinterpret_cast<char*>(where) +
                                          (Cpp::SizeOf(scope) * 3));
  EXPECT_TRUE(Cpp::Destruct(new_head, scope, false, 2));

  output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(output, "\nDestructor Executed\n\nDestructor Executed\n");
  output.clear();

  // deallocate since we call the destructor withFree = false
  Cpp::Deallocate(scope, where, a);

  // perform the same withFree=true
  where = nullptr;
  where = Cpp::Construct(scope, nullptr, a).data;
  EXPECT_TRUE(where);
  testing::internal::CaptureStdout();
  EXPECT_TRUE(Cpp::Destruct(where, scope, true, a));
  output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(output,
            "\nDestructor Executed\n\nDestructor Executed\n\nDestructor "
            "Executed\n\nDestructor Executed\n\nDestructor Executed\n");
  output.clear();
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_UndoTest) {
#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif
#if CLANG_VERSION_MAJOR == 20 && defined(CPPINTEROP_USE_CLING) &&           \
    defined(__APPLE__)
  GTEST_SKIP() << "Disabled on osx for cling based on llvm 20. Needs fixing.";
#endif
#if defined(CPPINTEROP_USE_CLING) && defined(CPPINTEROP_ASAN_BUILD)
  GTEST_SKIP() << "cling unload walks a module already freed by ORC "
                  "clone-on-emit; skip until the cling-side fix lands.";
#endif
#ifdef EMSCRIPTEN
  GTEST_SKIP() << "Test fails for Emscipten builds";
#else
  TestFixture::CreateInterpreter();
  EXPECT_EQ(Cpp::Process("int a = 5;"), 0);
  EXPECT_EQ(Cpp::Process("int b = 10;"), 0);
  EXPECT_EQ(Cpp::Process("int x = 5;"), 0);
  Cpp::Undo();
  EXPECT_NE(Cpp::Process("int y = x;"), 0);
  EXPECT_EQ(Cpp::Process("int x = 10;"), 0);
  EXPECT_EQ(Cpp::Process("int y = 10;"), 0);
  Cpp::Undo(2);
  EXPECT_EQ(Cpp::Process("int x = 20;"), 0);
  EXPECT_EQ(Cpp::Process("int y = 20;"), 0);
  int ret = Cpp::Undo(100);
#ifdef CPPINTEROP_USE_CLING
  EXPECT_EQ(ret, 0);
#else
  EXPECT_EQ(ret, 1);
#endif
#endif
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_FailingTest1) {
#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif
#ifdef EMSCRIPTEN_SHARED_LIBRARY
  GTEST_SKIP() << "Test fails for Emscipten shared library builds";
#endif
  TestFixture::CreateInterpreter();
  EXPECT_FALSE(Cpp::Declare(R"(
    class WithOutEqualOp1 {};
    class WithOutEqualOp2 {};

    WithOutEqualOp1 o1;
    WithOutEqualOp2 o2;

    template<class C1, class C2>
    bool is_equal(const C1& c1, const C2& c2) { return (bool)(c1 == c2); }
  )"));

  Cpp::TypeRef o1 = Cpp::GetTypeFromScope(Cpp::GetNamed("o1"));
  Cpp::TypeRef o2 = Cpp::GetTypeFromScope(Cpp::GetNamed("o2"));
  std::vector<Cpp::FuncRef> fns;
  Cpp::GetClassTemplatedMethods("is_equal", Cpp::GetGlobalScope(), fns);
  EXPECT_EQ(fns.size(), 1);

  std::vector<Cpp::TemplateArgInfo> args = {{o1.data}, {o2.data}};
  Cpp::DeclRef fn = Cpp::InstantiateTemplate(Cpp::DeclRef{fns[0].data}, args);
  EXPECT_TRUE(fn);

  Cpp::JitCall jit_call = Cpp::MakeFunctionCallable(Cpp::FuncRef{fn.data});
  EXPECT_EQ(jit_call.getKind(), Cpp::JitCall::kUnknown); // expected to fail
  EXPECT_FALSE(Cpp::Declare("int x = 1;"));
  EXPECT_FALSE(Cpp::Declare("int y = x;"));
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_IsExplicit) {
  std::vector<Decl*> Decls;
  std::vector<Decl*> SubDecls;
  std::string code = R"(
    class C {
    public:
      C() {}
      explicit C(int) {}
      C(const C&) {}
      explicit C(C&&) {}
      
      operator int() { return 0; }
      explicit operator bool() { return true; }
      
      void regular_method() {}
    private:
      explicit C(double) {}
    };
    )";

  GetAllTopLevelDecls(code, Decls);
  GetAllSubDecls(Decls[0], SubDecls);

  // constructors
  EXPECT_FALSE(Cpp::IsExplicit(SubDecls[2])); // C()
  EXPECT_TRUE(Cpp::IsExplicit(SubDecls[3]));  // explicit C(int)
  EXPECT_FALSE(Cpp::IsExplicit(SubDecls[4])); // C(const C&) copy ctor
  EXPECT_TRUE(Cpp::IsExplicit(SubDecls[5]));  // explicit C(C&&) move ctor

  // conversion operators
  EXPECT_FALSE(Cpp::IsExplicit(SubDecls[6])); // operator int()
  EXPECT_TRUE(Cpp::IsExplicit(SubDecls[7]));  // explicit operator bool()
  EXPECT_FALSE(Cpp::IsExplicit(SubDecls[8])); // regular_method()
  EXPECT_TRUE(Cpp::IsExplicit(SubDecls[10])); // private explicit C(double)
  EXPECT_FALSE(Cpp::IsExplicit(Decls[0]));
  EXPECT_FALSE(Cpp::IsExplicit(nullptr));
}
TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_IsExplicitTemplated) {
  std::vector<Decl*> Decls;
  std::vector<Decl*> SubDecls;
  std::string code = R"(
    class T {
    public:
      T() = delete;
      
      template<typename U>
      T(U) {}
      
      template<typename U>
      explicit T(U, int) {}
      
      template<typename U>
      operator U() { return U{}; }
      
      template<typename U>
      explicit operator U*() { return nullptr; }
    };
    )";

  GetAllTopLevelDecls(code, Decls);
  GetAllSubDecls(Decls[0], SubDecls);

  int implicitCtorCount = 0;
  int explicitCtorCount = 0;
  int implicitConvCount = 0;
  int explicitConvCount = 0;

  for (auto* decl : SubDecls) {
    // skip deleted constructors
    if (auto* CD = llvm::dyn_cast_or_null<CXXConstructorDecl>(
            static_cast<clang::Decl*>(decl))) {
      if (CD->isDeleted())
        continue;
    }

    if (Cpp::IsConstructor(decl)) {
      if (Cpp::IsExplicit(decl))
        explicitCtorCount++;
      else
        implicitCtorCount++;
    } else {
      // conversion operator
      auto* D = static_cast<clang::Decl*>(decl);
      if (auto* FTD = llvm::dyn_cast_or_null<FunctionTemplateDecl>(D))
        D = FTD->getTemplatedDecl();

      if (llvm::isa<CXXConversionDecl>(D)) {
        if (Cpp::IsExplicit(decl))
          explicitConvCount++;
        else
          implicitConvCount++;
      }
    }
  }

  EXPECT_EQ(implicitCtorCount, 1); // T(U)
  EXPECT_EQ(explicitCtorCount, 1); // explicit T(U, int)
  EXPECT_EQ(implicitConvCount, 1); // operator U()
  EXPECT_EQ(explicitConvCount, 1); // explicit operator U*()
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_IsExplicitDeductionGuide) {
  // Deduction guides are a C++17 feature
  std::vector<const char*> interpreter_args = {"-include", "new", "-std=c++17"};
  Cpp::CreateInterpreter(interpreter_args, {});

  Interp->declare(R"(
    template<typename T>
    struct Wrapper {
      T value;
      Wrapper(T v) : value(v) {}
    };
    template<typename T>
    explicit Wrapper(T*) -> Wrapper<T*>;
  )");

  auto* TUD = Interp->getCI()->getASTContext().getTranslationUnitDecl();
  bool foundExplicitGuide = false;
  Decl* guide = nullptr;
  for (auto* D : TUD->decls()) {
    if (Cpp::IsExplicit(D)) {
      foundExplicitGuide = true;
      guide = D;
      break;
    }
  }

  EXPECT_TRUE(foundExplicitGuide);
  EXPECT_TRUE(guide != nullptr);
  EXPECT_EQ(Cpp::GetFunctionSignature(guide),
            "explicit Wrapper(T *) -> Wrapper<T *>");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionReturnType(guide)),
            "Wrapper<T *>");
}

// C++23 "deducing this" (explicit object parameters, P0847R7): the object
// parameter binds to the receiver, not the argument list.
TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_DeducingThisIntrospection) {
  std::vector<Decl*> Decls;
  std::vector<Decl*> SubDecls;
  std::string code = R"(
    struct Widget {
      int value = 42;
      int get(this Widget& self) { return self.value; }
      int add(this Widget& self, int x, int y = 5) { return self.value + x + y; }
      int plain(int x) { return value + x; }
    };
  )";

  GetAllTopLevelDecls(code, Decls, /*filter_implicitGenerated=*/true,
                      /*interpreter_args=*/{"-std=c++23"});
  GetAllSubDecls(Decls[0], SubDecls, /*filter_implicitGenerated=*/true);

  // SubDecls: [0]=field value, [1]=get, [2]=add, [3]=plain
  Decl* get = SubDecls[1];
  Decl* add = SubDecls[2];
  Decl* plain = SubDecls[3];

  // The explicit object parameter is not counted as a callee argument.
  EXPECT_EQ(Cpp::GetFunctionNumArgs(get), (size_t)0);
  EXPECT_EQ(Cpp::GetFunctionRequiredArgs(get), (size_t)0);

  EXPECT_EQ(Cpp::GetFunctionNumArgs(add), (size_t)2);
  EXPECT_EQ(Cpp::GetFunctionRequiredArgs(add), (size_t)1); // y defaulted

  // The non-object parameters are exposed at 0-based indices that skip `self`.
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionArgType(add, 0)), "int");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionArgType(add, 1)), "int");
  EXPECT_EQ(Cpp::GetFunctionArgName(add, 0), "x");
  EXPECT_EQ(Cpp::GetFunctionArgName(add, 1), "y");
  EXPECT_EQ(Cpp::GetFunctionArgDefault(add, 1), "5");

  // A traditional (implicit-object) method is unaffected.
  EXPECT_EQ(Cpp::GetFunctionNumArgs(plain), (size_t)1);
  EXPECT_EQ(Cpp::GetFunctionArgName(plain, 0), "x");
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_DeducingThisJitCall) {
#ifdef EMSCRIPTEN
#if CLANG_VERSION_MAJOR > 21
  GTEST_SKIP() << "Test fails for Emscripten builds using LLVM 22";
#endif
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";

  std::string code = R"(
    struct Widget {
      int value = 42;
      int get(this Widget& self) { return self.value; }
      int add(this Widget& self, int x, int y) { return self.value + x + y; }
    };
  )";

  std::vector<Decl*> Decls;
  // `-include new` is needed for the constructor wrapper's placement new.
  GetAllTopLevelDecls(code, Decls, /*filter_implicitGenerated=*/false,
                      /*interpreter_args=*/{"-std=c++23", "-include", "new"});

  Cpp::DeclRef Widget = Cpp::GetNamed("Widget");
  ASSERT_TRUE(Widget);

  // Construct a Widget so the in-class initializer (value = 42) runs.
  auto Ctor = Cpp::MakeFunctionCallable(Cpp::GetDefaultConstructor(Widget));
  void* object = nullptr;
  Ctor.Invoke((void*)&object, {}, /*self=*/nullptr);
  ASSERT_TRUE(object);

  // get(): explicit object parameter bound to the receiver, no call args.
  Cpp::JitCall GetCall = Cpp::MakeFunctionCallable(
      Cpp::FuncRef{Cpp::GetNamed("get", Widget).data});
  EXPECT_EQ(GetCall.getKind(), Cpp::JitCall::kGenericCall);
  int result = 0;
  GetCall.Invoke(&result, {}, object);
  EXPECT_EQ(result, 42);

  // add(): explicit object parameter plus regular arguments.
  Cpp::JitCall AddCall = Cpp::MakeFunctionCallable(
      Cpp::FuncRef{Cpp::GetNamed("add", Widget).data});
  int x = 20;
  int y = 3;
  std::array<void*, 2> args = {(void*)&x, (void*)&y};
  result = 0;
  AddCall.Invoke(&result, {args.data(), /*args_size=*/2}, object);
  EXPECT_EQ(result, 42 + 20 + 3);

  Cpp::Destruct(object, Widget);
}

TYPED_TEST(CPPINTEROP_TEST_MODE,
           FunctionReflection_DeducingThisRValueRefJitCall) {
#ifdef EMSCRIPTEN
#if CLANG_VERSION_MAJOR > 21
  GTEST_SKIP() << "Test fails for Emscripten builds using LLVM 22";
#endif
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";

  // The wrapper must bind the receiver as an rvalue for both the C++23
  // explicit-object form (`this W&&`) and the traditional `&&` qualifier.
  std::string code = R"(
    struct RWidget {
      int value = 13;
      int consume(this RWidget&& self) { return self.value; }
    };
    struct QWidget {
      int value = 17;
      int consume() && { return value; }
    };
  )";

  std::vector<Decl*> Decls;
  GetAllTopLevelDecls(code, Decls, /*filter_implicitGenerated=*/false,
                      /*interpreter_args=*/{"-std=c++23", "-include", "new"});

  for (const char* name : {"RWidget", "QWidget"}) {
    Cpp::DeclRef W = Cpp::GetNamed(name);
    ASSERT_TRUE(W) << name;
    auto Ctor = Cpp::MakeFunctionCallable(Cpp::GetDefaultConstructor(W));
    void* object = nullptr;
    Ctor.Invoke((void*)&object);
    ASSERT_TRUE(object) << name;

    Cpp::JitCall Call = Cpp::MakeFunctionCallable(
        Cpp::FuncRef{Cpp::GetNamed("consume", W).data});
    EXPECT_EQ(Call.getKind(), Cpp::JitCall::kGenericCall) << name;
    int result = 0;
    Call.Invoke(&result, {}, object);
    EXPECT_EQ(result, std::string(name) == "RWidget" ? 13 : 17) << name;

    Cpp::Destruct(object, W);
  }
}

TYPED_TEST(CPPINTEROP_TEST_MODE,
           FunctionReflection_DeducingThisTemplateOverloadMatch) {
  std::vector<Decl*> Decls;
  std::vector<Decl*> SubDecls;
  std::string code = R"(
    struct TWidget {
      int value = 9;
      template <class Self> int via(this Self&& self) { return self.value; }
    };
  )";

  GetAllTopLevelDecls(code, Decls, /*filter_implicitGenerated=*/true,
                      /*interpreter_args=*/{"-std=c++23"});
  GetAllSubDecls(Decls[0], SubDecls, /*filter_implicitGenerated=*/true);

  std::vector<Cpp::FuncRef> candidates;
  for (auto* decl : SubDecls)
    if (Cpp::IsTemplatedFunction(decl))
      candidates.push_back((Cpp::FuncRef)decl);
  ASSERT_EQ(candidates.size(), (size_t)1);

  // No explicit template/call args: `Self` deduces from the synthesized
  // receiver (only the AddMethodTemplateCandidate path can do this).
  std::vector<Cpp::TemplateArgInfo> no_explicit_args;
  std::vector<Cpp::TemplateArgInfo> no_args;
  Cpp::FuncRef matched =
      Cpp::BestOverloadFunctionMatch(candidates, no_explicit_args, no_args);
  ASSERT_TRUE(matched);
  EXPECT_NE(Cpp::GetFunctionSignature(matched).find("via"), std::string::npos);
}

TYPED_TEST(CPPINTEROP_TEST_MODE,
           FunctionReflection_DeducingThisAbbreviatedAuto) {
  // `this auto&&` is sugar for an invented template parameter: a method
  // template with a distinct AST shape from an explicit `template<class S>`.
  // It must still deduce the object parameter from the synthesized receiver.
  std::vector<Decl*> Decls;
  std::vector<Decl*> SubDecls;
  std::string code = R"(
    struct AAWidget {
      int value = 23;
      int get(this auto&& self) { return self.value; }
    };
  )";

  GetAllTopLevelDecls(code, Decls, /*filter_implicitGenerated=*/true,
                      /*interpreter_args=*/{"-std=c++23"});
  GetAllSubDecls(Decls[0], SubDecls, /*filter_implicitGenerated=*/true);

  std::vector<Cpp::FuncRef> candidates;
  for (auto* decl : SubDecls)
    if (Cpp::IsTemplatedFunction(decl))
      candidates.push_back((Cpp::FuncRef)decl);
  ASSERT_EQ(candidates.size(), (size_t)1);

  std::vector<Cpp::TemplateArgInfo> no_explicit_args;
  std::vector<Cpp::TemplateArgInfo> no_args;
  Cpp::FuncRef matched =
      Cpp::BestOverloadFunctionMatch(candidates, no_explicit_args, no_args);
  ASSERT_TRUE(matched);
  EXPECT_NE(Cpp::GetFunctionSignature(matched).find("get"), std::string::npos);
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_DeducingThisByValueCopy) {
#ifdef EMSCRIPTEN
#if CLANG_VERSION_MAJOR > 21
  GTEST_SKIP() << "Test fails for Emscripten builds using LLVM 22";
#endif
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";

  // A by-value explicit object parameter (`this W self`) operates on an
  // independent copy: mutating it must not affect the original receiver.
  std::string code = R"(
    struct CopyWidget {
      int value = 1;
      int bump(this CopyWidget self) { self.value += 100; return self.value; }
      int read(this CopyWidget& self) { return self.value; }
    };
  )";

  std::vector<Decl*> Decls;
  GetAllTopLevelDecls(code, Decls, /*filter_implicitGenerated=*/false,
                      /*interpreter_args=*/{"-std=c++23", "-include", "new"});

  Cpp::DeclRef W = Cpp::GetNamed("CopyWidget");
  ASSERT_TRUE(W);
  auto Ctor = Cpp::MakeFunctionCallable(Cpp::GetDefaultConstructor(W));
  void* object = nullptr;
  Ctor.Invoke((void*)&object, {}, /*self=*/nullptr);
  ASSERT_TRUE(object);

  Cpp::JitCall Bump =
      Cpp::MakeFunctionCallable(Cpp::FuncRef{Cpp::GetNamed("bump", W).data});
  int result = 0;
  Bump.Invoke(&result, {}, object);
  EXPECT_EQ(result, 101); // the copy was mutated

  Cpp::JitCall Read =
      Cpp::MakeFunctionCallable(Cpp::FuncRef{Cpp::GetNamed("read", W).data});
  result = 0;
  Read.Invoke(&result, {}, object);
  EXPECT_EQ(result, 1); // ... the original is untouched

  Cpp::Destruct(object, W);
}

TYPED_TEST(CPPINTEROP_TEST_MODE, FunctionReflection_DeducingThisInheritance) {
#ifdef EMSCRIPTEN
#if CLANG_VERSION_MAJOR > 21
  GTEST_SKIP() << "Test fails for Emscripten builds using LLVM 22";
#endif
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";

  // A base-class explicit object method invoked on a derived object: the
  // wrapper binds a Derived* receiver to a Base& object parameter.
  std::string code = R"(
    struct Base {
      int value = 8;
      int get(this Base& self) { return self.value; }
    };
    struct Derived : Base { };
  )";

  std::vector<Decl*> Decls;
  GetAllTopLevelDecls(code, Decls, /*filter_implicitGenerated=*/false,
                      /*interpreter_args=*/{"-std=c++23", "-include", "new"});

  Cpp::DeclRef Derived = Cpp::GetNamed("Derived");
  Cpp::DeclRef Base = Cpp::GetNamed("Base");
  ASSERT_TRUE(Derived);
  ASSERT_TRUE(Base);
  auto Ctor = Cpp::MakeFunctionCallable(Cpp::GetDefaultConstructor(Derived));
  void* object = nullptr;
  Ctor.Invoke((void*)&object, {}, /*self=*/nullptr);
  ASSERT_TRUE(object);

  // get() is declared on Base, invoked with the Derived object.
  Cpp::JitCall Get =
      Cpp::MakeFunctionCallable(Cpp::FuncRef{Cpp::GetNamed("get", Base).data});
  EXPECT_EQ(Get.getKind(), Cpp::JitCall::kGenericCall);
  int result = 0;
  Get.Invoke(&result, {}, object);
  EXPECT_EQ(result, 8);

  Cpp::Destruct(object, Derived);
}

// Use-cases from https://devblogs.microsoft.com/cppblog/cpp23-deducing-this/
// Cases that live inside a function body are driven via a JIT-called helper.

// JIT-call a nullary `int ns::fn()`. (GetNamed takes a name + scope, so the
// namespace is resolved first.)
static int JitCallIntNullary(const char* ns, const char* fn) {
  Cpp::DeclRef Scope = Cpp::GetNamed(ns);
  EXPECT_TRUE(Scope) << ns;
  Cpp::DeclRef Fn = Cpp::GetNamed(fn, Scope);
  EXPECT_TRUE(Fn) << fn;
  Cpp::JitCall JC = Cpp::MakeFunctionCallable(Cpp::FuncRef{Fn.data});
  EXPECT_EQ(JC.getKind(), Cpp::JitCall::kGenericCall) << fn;
  int result = 0;
  JC.Invoke(&result, {});
  return result;
}

TYPED_TEST(CPPINTEROP_TEST_MODE,
           FunctionReflection_DeducingThisBlogDeduplication) {
#ifdef EMSCRIPTEN
#if CLANG_VERSION_MAJOR > 21
  GTEST_SKIP() << "Test fails for Emscripten builds using LLVM 22";
#endif
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";

  // Blog use-case 1: code de-duplication. One forwarding accessor
  // `value(this Self&&)` replaces the cv/ref overloads; the driver writes
  // through it on an lvalue and reads the mutation back.
  std::string code = R"(
    #include <utility>
    namespace BlogDedup {
      struct Optional {
        int m_value = 5;
        template <class Self> auto&& value(this Self&& self) {
          return std::forward<Self>(self).m_value;
        }
      };
      int drive() { Optional o; o.value() = 17; return o.value(); }
    }
  )";

  std::vector<Decl*> Decls;
  GetAllTopLevelDecls(code, Decls, /*filter_implicitGenerated=*/false,
                      /*interpreter_args=*/{"-std=c++23", "-include", "new"});
  EXPECT_EQ(JitCallIntNullary("BlogDedup", "drive"), 17);
}

TYPED_TEST(CPPINTEROP_TEST_MODE,
           FunctionReflection_DeducingThisBlogCRTPPostfix) {
#ifdef EMSCRIPTEN
#if CLANG_VERSION_MAJOR > 21
  GTEST_SKIP() << "Test fails for Emscripten builds using LLVM 22";
#endif
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";

  // Blog use-case 2: CRTP without templating the base. `add_postfix_increment`
  // supplies `operator++(this Self&&, int)` once; the derived prefix operator
  // hides it, so a using-declaration re-exposes it (standard name hiding).
  std::string code = R"(
    namespace BlogCRTP {
      struct add_postfix_increment {
        template <typename Self>
        auto operator++(this Self&& self, int) { auto tmp = self; ++self; return tmp; }
      };
      struct some_type : add_postfix_increment {
        using add_postfix_increment::operator++;
        int v = 0;
        some_type& operator++() { ++v; return *this; }
      };
      int drive() { some_type c; auto old = c++; return old.v * 100 + c.v; }
    }
  )";

  std::vector<Decl*> Decls;
  GetAllTopLevelDecls(code, Decls, /*filter_implicitGenerated=*/false,
                      /*interpreter_args=*/{"-std=c++23", "-include", "new"});
  EXPECT_EQ(JitCallIntNullary("BlogCRTP", "drive"), 1); // old.v=0, c.v=1
}

TYPED_TEST(CPPINTEROP_TEST_MODE,
           FunctionReflection_DeducingThisBlogRecursiveLambda) {
#ifdef EMSCRIPTEN
#if CLANG_VERSION_MAJOR > 21
  GTEST_SKIP() << "Test fails for Emscripten builds using LLVM 22";
#endif
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";

  // Blog use-case 4: recursive lambdas via the explicit object parameter.
  std::string code = R"(
    namespace BlogRecLambda {
      int fib(int n) {
        auto f = [](this auto const& self, int n) -> int {
          return n < 2 ? n : self(n - 1) + self(n - 2);
        };
        return f(n);
      }
      int drive() { return fib(10); }
    }
  )";

  std::vector<Decl*> Decls;
  GetAllTopLevelDecls(code, Decls, /*filter_implicitGenerated=*/false,
                      /*interpreter_args=*/{"-std=c++23", "-include", "new"});
  EXPECT_EQ(JitCallIntNullary("BlogRecLambda", "drive"), 55);
}

TYPED_TEST(CPPINTEROP_TEST_MODE,
           FunctionReflection_DeducingThisBlogLambdaForwarding) {
#ifdef EMSCRIPTEN
#if CLANG_VERSION_MAJOR > 21
  GTEST_SKIP() << "Test fails for Emscripten builds using LLVM 22";
#endif
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";

  // Blog use-case 3: a closure with an explicit object parameter forwards based
  // on its own value category. (std::forward_like is not in the host libstdc++;
  // the deducing-this closure mechanism is what is exercised.)
  std::string code = R"(
    #include <utility>
    namespace BlogLambdaFwd {
      struct Scheduler { int submit(int m) { return m; } };
      int drive() {
        Scheduler scheduler;
        int message = 42;
        auto callback = [message, &scheduler](this auto&& self) -> int {
          return scheduler.submit(message);
        };
        return callback();
      }
    }
  )";

  std::vector<Decl*> Decls;
  GetAllTopLevelDecls(code, Decls, /*filter_implicitGenerated=*/false,
                      /*interpreter_args=*/{"-std=c++23", "-include", "new"});
  EXPECT_EQ(JitCallIntNullary("BlogLambdaFwd", "drive"), 42);
}

TYPED_TEST(CPPINTEROP_TEST_MODE,
           FunctionReflection_DeducingThisBlogPassByValue) {
#ifdef EMSCRIPTEN
#if CLANG_VERSION_MAJOR > 21
  GTEST_SKIP() << "Test fails for Emscripten builds using LLVM 22";
#endif
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";

  // Blog use-case 5: pass the object by value (good codegen for small types).
  // Invoke the explicit-object method directly on a constructed object.
  std::string code = R"(
    namespace BlogByValue {
      struct just_a_little_guy {
        int how_smol = 21;
        int uwu(this just_a_little_guy self) { return self.how_smol * 2; }
      };
    }
  )";

  std::vector<Decl*> Decls;
  GetAllTopLevelDecls(code, Decls, /*filter_implicitGenerated=*/false,
                      /*interpreter_args=*/{"-std=c++23", "-include", "new"});

  Cpp::DeclRef W =
      Cpp::GetNamed("just_a_little_guy", Cpp::GetNamed("BlogByValue"));
  ASSERT_TRUE(W);
  auto Ctor = Cpp::MakeFunctionCallable(Cpp::GetDefaultConstructor(W));
  void* object = nullptr;
  Ctor.Invoke((void*)&object, {}, /*self=*/nullptr);
  ASSERT_TRUE(object);

  Cpp::JitCall Uwu =
      Cpp::MakeFunctionCallable(Cpp::FuncRef{Cpp::GetNamed("uwu", W).data});
  int result = 0;
  Uwu.Invoke(&result, {}, object);
  EXPECT_EQ(result, 42);

  Cpp::Destruct(object, W);
}

TYPED_TEST(CPPINTEROP_TEST_MODE,
           FunctionReflection_DeducingThisBlogSfinaeTransform) {
#ifdef EMSCRIPTEN
#if CLANG_VERSION_MAJOR > 21
  GTEST_SKIP() << "Test fails for Emscripten builds using LLVM 22";
#endif
#endif
  if (TypeParam::isOutOfProcess)
    GTEST_SKIP() << "Test fails for OOP JIT builds";

  // Blog use-case 6: SFINAE-friendly callables (optional::transform). The
  // cv/ref category flows through Self; a callable is applied to the contained
  // value.
  std::string code = R"(
    namespace BlogTransform {
      template <class T>
      struct Optional6 {
        T m_value;
        template <class Self, class F>
        auto transform(this Self&& self, F&& f) { return f(self.m_value); }
      };
      int triple(int x) { return x * 3; }
      int drive() { Optional6<int> o{14}; return o.transform(triple); }
    }
  )";

  std::vector<Decl*> Decls;
  GetAllTopLevelDecls(code, Decls, /*filter_implicitGenerated=*/false,
                      /*interpreter_args=*/{"-std=c++23", "-include", "new"});
  EXPECT_EQ(JitCallIntNullary("BlogTransform", "drive"), 42);
}
