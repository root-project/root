#include "Utils.h"

#include "CppInterOp/CppInterOp.h"

#include "clang/AST/ASTContext.h"
#include "clang/Basic/Version.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Sema.h"

#include <llvm/ADT/ArrayRef.h>

#include "clang-c/CXCppInterOp.h"

#include "gtest/gtest.h"

#include <string>

using namespace TestUtils;
using namespace llvm;
using namespace clang;

TEST(FunctionReflectionTest, GetClassMethods) {
  std::vector<Decl*> Decls;
  std::string code = R"(
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

  auto get_method_name = [](Cpp::TCppFunction_t method) {
    return Cpp::GetFunctionSignature(method);
  };

  std::vector<Cpp::TCppFunction_t> methods0;
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

  std::vector<Cpp::TCppFunction_t> methods1;
  Cpp::GetClassMethods(Decls[1], methods1);
  EXPECT_EQ(methods0.size(), methods1.size());
  EXPECT_EQ(methods0[0], methods1[0]);
  EXPECT_EQ(methods0[1], methods1[1]);
  EXPECT_EQ(methods0[2], methods1[2]);
  EXPECT_EQ(methods0[3], methods1[3]);
  EXPECT_EQ(methods0[4], methods1[4]);

  std::vector<Cpp::TCppFunction_t> methods2;
  Cpp::GetClassMethods(Decls[2], methods2);

  EXPECT_EQ(methods2.size(), 6);
  EXPECT_EQ(get_method_name(methods2[0]), "B::B(int n)");
  EXPECT_EQ(get_method_name(methods2[1]), "inline constexpr B::B(const B &)");
  EXPECT_EQ(get_method_name(methods2[2]), "inline constexpr B::B(B &&)");
  EXPECT_EQ(get_method_name(methods2[3]), "inline B::~B()");
  EXPECT_EQ(get_method_name(methods2[4]), "inline B &B::operator=(const B &)");
  EXPECT_EQ(get_method_name(methods2[5]), "inline B &B::operator=(B &&)");

  std::vector<Cpp::TCppFunction_t> methods3;
  Cpp::GetClassMethods(Decls[3], methods3);

  EXPECT_EQ(methods3.size(), 9);
  EXPECT_EQ(get_method_name(methods3[0]), "B::B(int n)");
  EXPECT_EQ(get_method_name(methods3[1]), "inline constexpr B::B(const B &)");
  EXPECT_EQ(get_method_name(methods3[3]), "inline C::C()");
  EXPECT_EQ(get_method_name(methods3[4]), "inline constexpr C::C(const C &)");
  EXPECT_EQ(get_method_name(methods3[5]), "inline constexpr C::C(C &&)");
  EXPECT_EQ(get_method_name(methods3[6]), "inline C &C::operator=(const C &)");
  EXPECT_EQ(get_method_name(methods3[7]), "inline C &C::operator=(C &&)");
  EXPECT_EQ(get_method_name(methods3[8]), "inline C::~C()");

  // Should not crash.
  std::vector<Cpp::TCppFunction_t> methods4;
  Cpp::GetClassMethods(Decls[4], methods4);
  EXPECT_EQ(methods4.size(), 0);

  std::vector<Cpp::TCppFunction_t> methods5;
  Cpp::GetClassMethods(nullptr, methods5);
  EXPECT_EQ(methods5.size(), 0);

  // C API
  auto* I = clang_createInterpreterFromRawPtr(Cpp::GetInterpreter());
  auto C_API_SHIM = [&](Cpp::TCppFunction_t method) {
    auto Str = clang_getFunctionSignature(
        make_scope(static_cast<clang::Decl*>(method), I));
    auto Res = std::string(get_c_string(Str));
    dispose_string(Str);
    return Res;
  };
  EXPECT_EQ(C_API_SHIM(methods0[0]), "int A::f1(int a, int b)");
  // Clean up resources
  clang_Interpreter_takeInterpreterAsPtr(I);
  clang_Interpreter_dispose(I);
}

TEST(FunctionReflectionTest, ConstructorInGetClassMethods) {
  std::vector<Decl*> Decls;
  std::string code = R"(
    struct S {
      int a;
      int b;
    };
    )";

  GetAllTopLevelDecls(code, Decls);

  auto has_constructor = [](Decl* D) {
    std::vector<Cpp::TCppFunction_t> methods;
    Cpp::GetClassMethods(D, methods);
    for (auto method : methods) {
      if (Cpp::IsConstructor(method))
        return true;
    }
    return false;
  };

  EXPECT_TRUE(has_constructor(Decls[0]));
}

TEST(FunctionReflectionTest, HasDefaultConstructor) {
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

  // C API
  auto* I = clang_createInterpreterFromRawPtr(Cpp::GetInterpreter());
  EXPECT_TRUE(clang_hasDefaultConstructor(make_scope(Decls[0], I)));
  EXPECT_TRUE(clang_hasDefaultConstructor(make_scope(Decls[1], I)));
  EXPECT_FALSE(clang_hasDefaultConstructor(make_scope(Decls[3], I)));
  // Clean up resources
  clang_Interpreter_takeInterpreterAsPtr(I);
  clang_Interpreter_dispose(I);
}

TEST(FunctionReflectionTest, GetDestructor) {
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

  // C API
  auto* I = clang_createInterpreterFromRawPtr(Cpp::GetInterpreter());
  EXPECT_TRUE(clang_getDestructor(make_scope(Decls[0], I)).data[0]);
  EXPECT_TRUE(clang_getDestructor(make_scope(Decls[1], I)).data[0]);
  // Clean up resources
  clang_Interpreter_takeInterpreterAsPtr(I);
  clang_Interpreter_dispose(I);
}

TEST(FunctionReflectionTest, GetFunctionsUsingName) {
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
    )";

  GetAllTopLevelDecls(code, Decls);

  // This lambda can take in the scope and the name of the function
  // and returns the size of the vector returned by GetFunctionsUsingName
  auto get_number_of_funcs_using_name = [&](Cpp::TCppScope_t scope,
          const std::string &name) {
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
}

TEST(FunctionReflectionTest, GetClassDecls) {
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

  std::vector<Cpp::TCppFunction_t> methods;
  Cpp::GetClassMethods(Decls[0], methods);

  EXPECT_EQ(methods.size(), 10); // includes structors and operators
  EXPECT_EQ(Cpp::GetName(methods[0]), Cpp::GetName(SubDecls[4]));
  EXPECT_EQ(Cpp::GetName(methods[1]), Cpp::GetName(SubDecls[5]));
  EXPECT_EQ(Cpp::GetName(methods[2]), Cpp::GetName(SubDecls[7]));
  EXPECT_EQ(Cpp::GetName(methods[3]), Cpp::GetName(SubDecls[8]));
}

TEST(FunctionReflectionTest, GetFunctionTemplatedDecls) {
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

  std::vector<Cpp::TCppFunction_t> template_methods;
  Cpp::GetFunctionTemplatedDecls(Decls[0], template_methods);

  EXPECT_EQ(template_methods.size(), 4);
  EXPECT_EQ(Cpp::GetName(template_methods[0]), Cpp::GetName(SubDecls[1]));
  EXPECT_EQ(Cpp::GetName(template_methods[1]), Cpp::GetName(SubDecls[2]));
  EXPECT_EQ(Cpp::GetName(template_methods[2]), Cpp::GetName(SubDecls[3]));
  EXPECT_EQ(Cpp::GetName(template_methods[3]), Cpp::GetName(SubDecls[6]));
}

TEST(FunctionReflectionTest, GetFunctionReturnType) {
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
  std::vector<Cpp::TCppFunction_t> candidates = {Decls[14]};
  EXPECT_EQ(
      Cpp::GetTypeAsString(Cpp::GetFunctionReturnType(
          Cpp::BestOverloadFunctionMatch(candidates, explicit_args, args))),
      "RTTest_TemplatedList<int, double>");

  std::vector<Cpp::TemplateArgInfo> args2 = {C.DoubleTy.getAsOpaquePtr()};
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionReturnType(Cpp::GetNamed(
                "func", Cpp::InstantiateTemplate(Decls[15], args2.data(), 1)))),
            "double");
}

TEST(FunctionReflectionTest, GetFunctionNumArgs) {
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

TEST(FunctionReflectionTest, GetFunctionRequiredArgs) {
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

TEST(FunctionReflectionTest, GetFunctionArgType) {
  std::vector<Decl*> Decls, SubDecls;
  std::string code = R"(
    void f1(int i, double d, long l, char ch) {}
    void f2(const int i, double d[], long *l, char ch[4]) {}
    int a;
    )";

  GetAllTopLevelDecls(code, Decls);
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionArgType(Decls[0], 0)), "int");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionArgType(Decls[0], 1)), "double");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionArgType(Decls[0], 2)), "long");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionArgType(Decls[0], 3)), "char");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionArgType(Decls[1], 0)), "const int");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionArgType(Decls[1], 1)), "double[]");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionArgType(Decls[1], 2)), "long *");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionArgType(Decls[1], 3)), "char[4]");
  EXPECT_EQ(Cpp::GetTypeAsString(Cpp::GetFunctionArgType(Decls[2], 0)), "NULL TYPE");
}

TEST(FunctionReflectionTest, GetFunctionSignature) {
  std::vector<Decl*> Decls, SubDecls;
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

TEST(FunctionReflectionTest, IsTemplatedFunction) {
  std::vector<Decl*> Decls, SubDeclsC1, SubDeclsC2;
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

  // C API
  auto* I = clang_createInterpreterFromRawPtr(Cpp::GetInterpreter());
  EXPECT_FALSE(clang_isTemplatedFunction(make_scope(Decls[0], I)));
  EXPECT_TRUE(clang_isTemplatedFunction(make_scope(Decls[1], I)));
  EXPECT_FALSE(clang_isTemplatedFunction(make_scope(Decls[3], I)));
  EXPECT_FALSE(clang_isTemplatedFunction(make_scope(SubDeclsC1[1], I)));
  EXPECT_TRUE(clang_isTemplatedFunction(make_scope(SubDeclsC1[2], I)));
  // Clean up resources
  clang_Interpreter_takeInterpreterAsPtr(I);
  clang_Interpreter_dispose(I);
}

TEST(FunctionReflectionTest, ExistsFunctionTemplate) {
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
  EXPECT_TRUE(Cpp::ExistsFunctionTemplate("f", 0));
  EXPECT_TRUE(Cpp::ExistsFunctionTemplate("f", Decls[1]));
  EXPECT_FALSE(Cpp::ExistsFunctionTemplate("f", Decls[2]));

  // C API
  auto* I = clang_createInterpreterFromRawPtr(Cpp::GetInterpreter());
  EXPECT_TRUE(clang_existsFunctionTemplate("f", make_scope(Decls[1], I)));
  EXPECT_FALSE(clang_existsFunctionTemplate("f", make_scope(Decls[2], I)));
  // Clean up resources
  clang_Interpreter_takeInterpreterAsPtr(I);
  clang_Interpreter_dispose(I);
}

TEST(FunctionReflectionTest, InstantiateTemplateFunctionFromString) {
#if CLANG_VERSION_MAJOR == 18 && defined(CPPINTEROP_USE_CLING) &&              \
    defined(_WIN32) && (defined(_M_ARM) || defined(_M_ARM64))
  GTEST_SKIP() << "Test fails with Cling on Windows on ARM";
#endif
  if (llvm::sys::RunningOnValgrind())
    GTEST_SKIP() << "XFAIL due to Valgrind report";
  std::vector<const char*> interpreter_args = { "-include", "new" };
  Cpp::CreateInterpreter(interpreter_args);
  std::string code = R"(#include <memory>)";
  Interp->process(code);
  const char* str = "std::make_unique<int,int>";
  auto* Instance1 = (Decl*)Cpp::InstantiateTemplateFunctionFromString(str);
  EXPECT_TRUE(Instance1);
}

TEST(FunctionReflectionTest, InstantiateFunctionTemplate) {
  std::vector<Decl*> Decls;
  std::string code = R"(
template<typename T> T TrivialFnTemplate() { return T(); }
)";

  GetAllTopLevelDecls(code, Decls);
  ASTContext& C = Interp->getCI()->getASTContext();

  std::vector<Cpp::TemplateArgInfo> args1 = {C.IntTy.getAsOpaquePtr()};
  auto Instance1 = Cpp::InstantiateTemplate(Decls[0], args1.data(),
                                            /*type_size*/ args1.size());
  EXPECT_TRUE(isa<FunctionDecl>((Decl*)Instance1));
  FunctionDecl* FD = cast<FunctionDecl>((Decl*)Instance1);
  FunctionDecl* FnTD1 = FD->getTemplateInstantiationPattern();
  EXPECT_TRUE(FnTD1->isThisDeclarationADefinition());
  TemplateArgument TA1 = FD->getTemplateSpecializationArgs()->get(0);
  EXPECT_TRUE(TA1.getAsType()->isIntegerType());
}

TEST(FunctionReflectionTest, InstantiateTemplateMethod) {
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
  auto Instance1 = Cpp::InstantiateTemplate(Decls[1], args1.data(),
                                            /*type_size*/ args1.size());
  EXPECT_TRUE(isa<FunctionDecl>((Decl*)Instance1));
  FunctionDecl* FD = cast<FunctionDecl>((Decl*)Instance1);
  FunctionDecl* FnTD1 = FD->getTemplateInstantiationPattern();
  EXPECT_TRUE(FnTD1->isThisDeclarationADefinition());
  TemplateArgument TA1 = FD->getTemplateSpecializationArgs()->get(0);
  EXPECT_TRUE(TA1.getAsType()->isIntegerType());
}

TEST(FunctionReflectionTest, LookupConstructors) {
  if (llvm::sys::RunningOnValgrind())
    GTEST_SKIP() << "XFAIL due to Valgrind report";

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
  std::vector<Cpp::TCppFunction_t> ctors;
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

TEST(FunctionReflectionTest, GetClassTemplatedMethods) {
  if (llvm::sys::RunningOnValgrind())
    GTEST_SKIP() << "XFAIL due to Valgrind report";

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
  )";

  GetAllTopLevelDecls(code, Decls);
  std::vector<Cpp::TCppFunction_t> templatedMethods;
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
}

TEST(FunctionReflectionTest, GetClassTemplatedMethods_VariadicsAndOthers) {
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
  std::vector<Cpp::TCppFunction_t> templatedMethods;
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

TEST(FunctionReflectionTest, InstantiateVariadicFunction) {
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
  auto Instance1 = Cpp::InstantiateTemplate(Decls[1], args1.data(),
                                            /*type_size*/ args1.size());
  EXPECT_TRUE(Cpp::IsTemplatedFunction(Instance1));
  EXPECT_EQ(Cpp::GetFunctionSignature(Instance1),
            "template<> void VariadicFn<<double, int>>(double args, int args)");

  FunctionDecl* FD = cast<FunctionDecl>((Decl*)Instance1);
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
  std::vector<Cpp::TemplateArgInfo> args2 = {MyClassType,
                                             C.DoubleTy.getAsOpaquePtr()};

  // instantiate VariadicFnExtended
  auto Instance2 =
      Cpp::InstantiateTemplate(Decls[2], args2.data(), args2.size(), true);
  EXPECT_TRUE(Cpp::IsTemplatedFunction(Instance2));

  FunctionDecl* FD2 = cast<FunctionDecl>((Decl*)Instance2);
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
  EXPECT_EQ(Cpp::GetFunctionSignature(Instance2),
            "template<> void VariadicFnExtended<<MyClass, double>>(int "
            "fixedParam, MyClass args, double args)");
}

TEST(FunctionReflectionTest, BestOverloadFunctionMatch1) {
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
  std::vector<Cpp::TCppFunction_t> candidates;

  for (auto decl : Decls)
    if (Cpp::IsTemplatedFunction(decl)) candidates.push_back((Cpp::TCppFunction_t)decl);

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

  Cpp::TCppFunction_t func1 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args0, args1);
  Cpp::TCppFunction_t func2 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args1, args0);
  Cpp::TCppFunction_t func3 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args0, args2);
  Cpp::TCppFunction_t func4 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args1, args3);
  Cpp::TCppFunction_t func5 =
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

TEST(FunctionReflectionTest, BestOverloadFunctionMatch2) {
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
  std::vector<Cpp::TCppFunction_t> candidates;

  for (auto decl : Decls)
    if (Cpp::IsFunction(decl) || Cpp::IsTemplatedFunction(decl))
      candidates.push_back((Cpp::TCppFunction_t)decl);

  EXPECT_EQ(candidates.size(), 5);

  ASTContext& C = Interp->getCI()->getASTContext();

  std::vector<Cpp::TemplateArgInfo> args1 = {C.IntTy.getAsOpaquePtr()};
  std::vector<Cpp::TemplateArgInfo> args2 = {
      Cpp::GetVariableType(Cpp::GetNamed("a"))};
  std::vector<Cpp::TemplateArgInfo> args3 = {C.IntTy.getAsOpaquePtr(),
                                             C.IntTy.getAsOpaquePtr()};
  std::vector<Cpp::TemplateArgInfo> args4 = {
      Cpp::GetVariableType(Cpp::GetNamed("a")),
      Cpp::GetVariableType(Cpp::GetNamed("a"))};
  std::vector<Cpp::TemplateArgInfo> args5 = {C.IntTy.getAsOpaquePtr(),
                                             C.DoubleTy.getAsOpaquePtr()};

  std::vector<Cpp::TemplateArgInfo> explicit_args;

  Cpp::TCppFunction_t func1 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args, args1);
  Cpp::TCppFunction_t func2 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args, args2);
  Cpp::TCppFunction_t func3 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args, args3);
  Cpp::TCppFunction_t func4 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args, args4);
  Cpp::TCppFunction_t func5 =
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

TEST(FunctionReflectionTest, BestOverloadFunctionMatch3) {
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
  std::vector<Cpp::TCppFunction_t> candidates;

  for (auto decl : Decls)
    if (Cpp::IsTemplatedFunction(decl))
      candidates.push_back((Cpp::TCppFunction_t)decl);

  EXPECT_EQ(candidates.size(), 2);

  ASTContext& C = Interp->getCI()->getASTContext();

  std::vector<Cpp::TemplateArgInfo> args1 = {
      Cpp::GetVariableType(Cpp::GetNamed("a")),
      Cpp::GetVariableType(Cpp::GetNamed("a"))};
  std::vector<Cpp::TemplateArgInfo> args2 = {
      Cpp::GetVariableType(Cpp::GetNamed("a")), C.IntTy.getAsOpaquePtr()};
  std::vector<Cpp::TemplateArgInfo> args3 = {
      Cpp::GetVariableType(Cpp::GetNamed("a")), C.DoubleTy.getAsOpaquePtr()};

  std::vector<Cpp::TemplateArgInfo> explicit_args;

  Cpp::TCppFunction_t func1 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args, args1);
  Cpp::TCppFunction_t func2 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args, args2);
  Cpp::TCppFunction_t func3 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args, args3);

  candidates.clear();
  Cpp::GetOperator(
      Cpp::GetScopeFromType(Cpp::GetVariableType(Cpp::GetNamed("a"))),
      Cpp::Operator::OP_Minus, candidates);

  EXPECT_EQ(candidates.size(), 1);

  std::vector<Cpp::TemplateArgInfo> args4 = {
      Cpp::GetVariableType(Cpp::GetNamed("a"))};

  Cpp::TCppFunction_t func4 =
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

TEST(FunctionReflectionTest, BestOverloadFunctionMatch4) {
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
  std::vector<Cpp::TCppFunction_t> candidates;
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
      Cpp::GetVariableType(Cpp::GetNamed("a"))};
  std::vector<Cpp::TemplateArgInfo> args4 = {
      Cpp::GetVariableType(Cpp::GetNamed("a")),
      Cpp::GetVariableType(Cpp::GetNamed("b"))};
  std::vector<Cpp::TemplateArgInfo> args5 = {
      Cpp::GetVariableType(Cpp::GetNamed("a")),
      Cpp::GetVariableType(Cpp::GetNamed("a"))};

  std::vector<Cpp::TemplateArgInfo> explicit_args1;
  std::vector<Cpp::TemplateArgInfo> explicit_args2 = {C.IntTy.getAsOpaquePtr(),
                                                      C.IntTy.getAsOpaquePtr()};

  Cpp::TCppFunction_t func1 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args1, args1);
  Cpp::TCppFunction_t func2 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args1, args2);
  Cpp::TCppFunction_t func3 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args1, args3);
  Cpp::TCppFunction_t func4 =
      Cpp::BestOverloadFunctionMatch(candidates, explicit_args1, args4);
  Cpp::TCppFunction_t func5 =
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

TEST(FunctionReflectionTest, IsPublicMethod) {
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

TEST(FunctionReflectionTest, IsProtectedMethod) {
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

TEST(FunctionReflectionTest, IsPrivateMethod) {
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

TEST(FunctionReflectionTest, IsConstructor) {
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

TEST(FunctionReflectionTest, IsDestructor) {
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

TEST(FunctionReflectionTest, IsStaticMethod) {
  std::vector<Decl *> Decls, SubDecls;
  std::string code = R"(
    class C {
      void f1() {}
      static void f2() {}
    };
    )";

  GetAllTopLevelDecls(code, Decls);
  GetAllSubDecls(Decls[0], SubDecls);

  EXPECT_FALSE(Cpp::IsStaticMethod(Decls[0]));
  EXPECT_FALSE(Cpp::IsStaticMethod(SubDecls[1]));
  EXPECT_TRUE(Cpp::IsStaticMethod(SubDecls[2]));
}

TEST(FunctionReflectionTest, GetFunctionAddress) {
#ifdef EMSCRIPTEN
#if CLANG_VERSION_MAJOR < 20
  GTEST_SKIP() << "Test fails for Emscipten builds";
#endif
#endif
  if (llvm::sys::RunningOnValgrind())
    GTEST_SKIP() << "XFAIL due to Valgrind report";
#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif
  std::vector<Decl*> Decls, SubDecls;
  std::string code = "int f1(int i) { return i * i; }";
  std::vector<const char*> interpreter_args = {"-include", "new"};

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

  EXPECT_FALSE(Cpp::GetFunctionAddress(Cpp::GetGlobalScope()));
}

TEST(FunctionReflectionTest, IsVirtualMethod) {
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

TEST(FunctionReflectionTest, JitCallAdvanced) {
#ifdef EMSCRIPTEN
#if CLANG_VERSION_MAJOR < 20
  GTEST_SKIP() << "Test fails for Emscipten builds";
#endif
#endif
  if (llvm::sys::RunningOnValgrind())
    GTEST_SKIP() << "XFAIL due to Valgrind report";

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
  auto *CtorD
    = (clang::CXXConstructorDecl*)Cpp::GetDefaultConstructor(Decls[0]);
  auto Ctor = Cpp::MakeFunctionCallable(CtorD);
  EXPECT_TRUE((bool)Ctor) << "Failed to build a wrapper for the ctor";
  void* object = nullptr;
  Ctor.Invoke(&object);
  EXPECT_TRUE(object) << "Failed to call the ctor.";
  // Building a wrapper with a typedef decl must be possible.
  EXPECT_TRUE(Cpp::Destruct(object, Decls[1]));

  // C API
  auto* I = clang_createInterpreterFromRawPtr(Cpp::GetInterpreter());
  auto S = clang_getDefaultConstructor(make_scope(Decls[0], I));
  void* object_c = nullptr;
  clang_invoke(S, &object_c, nullptr, 0, nullptr);
  EXPECT_TRUE(object_c) << "Failed to call the ctor.";
  clang_destruct(object_c, make_scope(Decls[1], I), true);
  // Clean up resources
  clang_Interpreter_takeInterpreterAsPtr(I);
  clang_Interpreter_dispose(I);
}

#if !defined(NDEBUG) && GTEST_HAS_DEATH_TEST
#ifndef _WIN32 // Death tests do not work on Windows
TEST(FunctionReflectionTest, JitCallDebug) {
#ifdef EMSCRIPTEN
#if CLANG_VERSION_MAJOR < 20
  GTEST_SKIP() << "Test fails for Emscipten builds";
#endif
#endif
  if (llvm::sys::RunningOnValgrind())
    GTEST_SKIP() << "XFAIL due to Valgrind report";

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

  const auto* CtorD = Cpp::GetDefaultConstructor(Decls[0]);
  auto JC = Cpp::MakeFunctionCallable(CtorD);

  EXPECT_TRUE(JC.getKind() == Cpp::JitCall::kConstructorCall);
  EXPECT_DEATH(
      { JC.InvokeConstructor(/*result=*/nullptr); },
      "Must pass the location of the created object!");

  void* result = Cpp::Allocate(Decls[0]);
  EXPECT_DEATH(
      { JC.InvokeConstructor(&result, 0UL); },
      "Number of objects to construct should be atleast 1");

  // Succeeds
  JC.InvokeConstructor(&result, 5UL);

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

  result = Cpp::Allocate(Decls[0], 5);
  int i = 42;
  void* args0[1] = {(void*)&i};
  EXPECT_DEATH(
      { JC.InvokeConstructor(&result, 5UL, {args0, 1}); },
      "Cannot pass initialization parameters to array new construction");

  JC.InvokeConstructor(&result, 1UL, {args0, 1}, (void*)~0);

  int* obj = reinterpret_cast<int*>(reinterpret_cast<char*>(result));
  EXPECT_TRUE(*obj == 42);

  // Destructors
  Cpp::TCppScope_t scope_C = Cpp::GetNamed("C");
  Cpp::TCppObject_t object_C = Cpp::Construct(scope_C);

  // Make destructor callable and pass arguments
  JC = Cpp::MakeFunctionCallable(SubDecls[4]);
  EXPECT_DEATH(
      { JC.Invoke(&object_C, {args0, 1}); },
      "Destructor called with arguments");
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

TEST(FunctionReflectionTest, GetFunctionCallWrapper) {
#ifdef EMSCRIPTEN
  GTEST_SKIP() << "Test fails for Emscipten builds";
#endif
  if (llvm::sys::RunningOnValgrind())
    GTEST_SKIP() << "XFAIL due to Valgrind report";
#if defined(CPPINTEROP_USE_CLING) && defined(_WIN32)
  GTEST_SKIP() << "Disabled, invoking functions containing printf does not work with Cling on Windows";
#endif
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
      Cpp::MakeFunctionCallable(Cpp::GetNamed("f2"));
  EXPECT_TRUE(FCI2.getKind() == Cpp::JitCall::kGenericCall);
  Cpp::JitCall FCI3 =
    Cpp::MakeFunctionCallable(Cpp::GetNamed("f3", Cpp::GetNamed("NS")));
  EXPECT_TRUE(FCI3.getKind() == Cpp::JitCall::kGenericCall);
  Cpp::JitCall FCI4 =
      Cpp::MakeFunctionCallable(Cpp::GetNamed("f4", Cpp::GetNamed("NS")));
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

#if CLANG_VERSION_MAJOR > 16
  Cpp::JitCall FCI5 =
      Cpp::MakeFunctionCallable(Cpp::GetNamed("f5", Cpp::GetNamed("NS")));
  EXPECT_TRUE(FCI5.getKind() == Cpp::JitCall::kGenericCall);

  typedef int (*int_func)();
  int_func callback = nullptr;
  FCI5.Invoke((void*)&callback);
  EXPECT_TRUE(callback);
  EXPECT_EQ(callback(), 3);
#endif

  // FIXME: Do we need to support private ctors?
  Interp->process(R"(
    class C {
    public:
      C() { printf("Default Ctor Called\n"); }
      ~C() { printf("Dtor Called\n"); }
    };
  )");

  clang::NamedDecl *ClassC = (clang::NamedDecl*)Cpp::GetNamed("C");
  auto *CtorD = (clang::CXXConstructorDecl*)Cpp::GetDefaultConstructor(ClassC);
  auto FCI_Ctor =
    Cpp::MakeFunctionCallable(CtorD);
  void* object = nullptr;
  testing::internal::CaptureStdout();
  FCI_Ctor.Invoke((void*)&object);
  output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(output, "Default Ctor Called\n");
  EXPECT_TRUE(object != nullptr);

  auto *DtorD = (clang::CXXDestructorDecl*)Cpp::GetDestructor(ClassC);
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
  auto Instance1 = Cpp::InstantiateTemplate(Decls1[0], argument.data(),
                                            /*type_size*/ argument.size());
  EXPECT_TRUE(isa<ClassTemplateSpecializationDecl>((Decl*)Instance1));
  auto* CTSD1 = static_cast<ClassTemplateSpecializationDecl*>(Instance1);
  auto* Add_D = Cpp::GetNamed("Add",CTSD1);
  Cpp::JitCall FCI_Add = Cpp::MakeFunctionCallable(Add_D);
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

  Cpp::TCppScope_t set_5 = Cpp::GetNamed("set_5");
  EXPECT_TRUE(set_5);

  Cpp::JitCall set_5_f = Cpp::MakeFunctionCallable(set_5);
  EXPECT_EQ(set_5_f.getKind(), Cpp::JitCall::kGenericCall);

  int* bp = &b;
  void* set_5_args[1] = {(void*)&bp};
  set_5_f.Invoke(nullptr, {set_5_args, 1});
  EXPECT_EQ(b, 5);

#if CLANG_VERSION_MAJOR > 16
  // typedef resolution testing
  // supported for clang version >16 only
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

  Cpp::TCppScope_t TypedefToPrivateClass =
      Cpp::GetNamed("TypedefToPrivateClass");
  EXPECT_TRUE(TypedefToPrivateClass);

  Cpp::TCppScope_t f = Cpp::GetNamed("f", TypedefToPrivateClass);
  EXPECT_TRUE(f);

  Cpp::JitCall FCI_f = Cpp::MakeFunctionCallable(f);
  EXPECT_EQ(FCI_f.getKind(), Cpp::JitCall::kGenericCall);

  void* res = nullptr;
  FCI_f.Invoke(&res, {nullptr, 0});
  EXPECT_TRUE(res);
#endif

  // templated operators
  Interp->process(R"(
    class TOperator{
    public:
      template<typename T>
      bool operator<(T t) { return true; }
    };
  )");
  Cpp::TCppScope_t TOperator = Cpp::GetNamed("TOperator");

  auto* TOperatorCtor = Cpp::GetDefaultConstructor(TOperator);
  auto FCI_TOperatorCtor = Cpp::MakeFunctionCallable(TOperatorCtor);
  void* toperator = nullptr;
  FCI_TOperatorCtor.Invoke((void*)&toperator);

  EXPECT_TRUE(toperator);
  std::vector<Cpp::TCppScope_t> operators;
  Cpp::GetOperator(TOperator, Cpp::OP_Less, operators);
  EXPECT_EQ(operators.size(), 1);

  Cpp::TCppScope_t op_templated = operators[0];
  auto TAI = Cpp::TemplateArgInfo(Cpp::GetType("int"));
  Cpp::TCppScope_t op = Cpp::InstantiateTemplate(op_templated, &TAI, 1);
  auto FCI_op = Cpp::MakeFunctionCallable(op);
  bool boolean = false;
  FCI_op.Invoke((void*)&boolean, {args, /*args_size=*/1}, toperator);
  EXPECT_TRUE(boolean);

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

  Cpp::TCppType_t K1 = Cpp::GetTypeFromScope(Cpp::GetNamed("K1"));
  Cpp::TCppType_t K2 = Cpp::GetTypeFromScope(Cpp::GetNamed("K2"));
  operators.clear();
  Cpp::GetOperator(Cpp::GetScope("N2", Cpp::GetScope("N1")), Cpp::OP_Plus,
                   operators);
  EXPECT_EQ(operators.size(), 1);
  Cpp::TCppFunction_t kop =
      Cpp::BestOverloadFunctionMatch(operators, {}, {K1, K2});
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

  std::vector<Cpp::TCppFunction_t> unresolved_candidate_methods;
  Cpp::GetClassTemplatedMethods("get", Cpp::GetScope("my_std"),
                                unresolved_candidate_methods);
  Cpp::TCppType_t p = Cpp::GetTypeFromScope(Cpp::GetNamed("p"));
  EXPECT_TRUE(p);

  Cpp::TCppScope_t fn = Cpp::BestOverloadFunctionMatch(
      unresolved_candidate_methods, {{Cpp::GetType("int"), "0"}}, {p});
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

  Cpp::TCppScope_t call_move = Cpp::BestOverloadFunctionMatch(
      unresolved_candidate_methods, {},
      {Cpp::GetReferencedType(Cpp::GetType("int"), true)});
  EXPECT_TRUE(call_move);

  auto call_move_callable = Cpp::MakeFunctionCallable(call_move);
  EXPECT_EQ(call_move_callable.getKind(), Cpp::JitCall::kGenericCall);

  // instantiation in host, no template function body
  Interp->process("template<typename T> T instantiation_in_host();");

  unresolved_candidate_methods.clear();
  Cpp::GetClassTemplatedMethods("instantiation_in_host", Cpp::GetGlobalScope(),
                                unresolved_candidate_methods);
  EXPECT_EQ(unresolved_candidate_methods.size(), 1);

  Cpp::TCppScope_t instantiation_in_host = Cpp::BestOverloadFunctionMatch(
      unresolved_candidate_methods, {Cpp::GetType("int")}, {});
  EXPECT_TRUE(instantiation_in_host);

  Cpp::JitCall instantiation_in_host_callable =
      Cpp::MakeFunctionCallable(instantiation_in_host);
  EXPECT_EQ(instantiation_in_host_callable.getKind(),
            Cpp::JitCall::kGenericCall);

  instantiation_in_host = Cpp::BestOverloadFunctionMatch(
      unresolved_candidate_methods, {Cpp::GetType("double")}, {});
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

  Cpp::TCppScope_t tuple_tuple = Cpp::BestOverloadFunctionMatch(
      unresolved_candidate_methods, {},
      {Cpp::GetVariableType(Cpp::GetNamed("tuple_one")),
       Cpp::GetVariableType(Cpp::GetNamed("tuple_two"))});
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

  Cpp::TCppScope_t bar =
      Cpp::GetNamed("bar", Cpp::GetScope("EnumFunctionSameName"));
  EXPECT_TRUE(bar);

  auto bar_callable = Cpp::MakeFunctionCallable(bar);
  EXPECT_EQ(bar_callable.getKind(), Cpp::JitCall::kGenericCall);
}

TEST(FunctionReflectionTest, IsConstMethod) {
  std::vector<Decl*> Decls, SubDecls;
  std::string code = R"(
    class C {
      void f1() const {}
      void f2() {}
    };
    )";

  GetAllTopLevelDecls(code, Decls);
  GetAllSubDecls(Decls[0], SubDecls);
  Cpp::TCppFunction_t method = nullptr; // Simulate an invalid method pointer

  EXPECT_TRUE(Cpp::IsConstMethod(SubDecls[1]));  // f1
  EXPECT_FALSE(Cpp::IsConstMethod(SubDecls[2])); // f2
  EXPECT_FALSE(Cpp::IsConstMethod(method));
}

TEST(FunctionReflectionTest, GetFunctionArgName) {
  std::vector<Decl*> Decls, SubDecls;
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

TEST(FunctionReflectionTest, GetFunctionArgDefault) {
  std::vector<Decl*> Decls, SubDecls;
  std::string code = R"(
    void f1(int i, double d = 4.0, const char *s = "default", char ch = 'c') {}
    void f2(float i = 0.0, double d = 3.123, long m = 34126) {}

    template<class A, class B>
    long get_size(A, B, int i = 0) {}

    template<class A = int, class B = char>
    long get_size(int i, A a = A(), B b = B()) {}

    template<class A>
    void get_size(long k, A, char ch = 'a', double l = 0.0) {}

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
}

TEST(FunctionReflectionTest, Construct) {
#ifdef EMSCRIPTEN
#if CLANG_VERSION_MAJOR < 20
  GTEST_SKIP() << "Test fails for Emscipten builds";
#endif
#endif
  if (llvm::sys::RunningOnValgrind())
    GTEST_SKIP() << "XFAIL due to Valgrind report";
#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif
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
  Cpp::TCppScope_t scope = Cpp::GetNamed("C");
  Cpp::TCppObject_t object = Cpp::Construct(scope);
  EXPECT_TRUE(object != nullptr);
  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(output, "Constructor Executed");
  output.clear();

  // Placement.
  testing::internal::CaptureStdout();
  void* where = Cpp::Allocate(scope);
  EXPECT_TRUE(where == Cpp::Construct(scope, where));
  // Check for the value of x which should be at the start of the object.
  EXPECT_TRUE(*(int *)where == 12345);
  Cpp::Deallocate(scope, where);
  output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(output, "Constructor Executed");
  output.clear();

  // Pass a constructor
  testing::internal::CaptureStdout();
  where = Cpp::Allocate(scope);
  EXPECT_TRUE(where == Cpp::Construct(SubDecls[3], where));
  EXPECT_TRUE(*(int*)where == 12345);
  Cpp::Deallocate(scope, where);
  output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(output, "Constructor Executed");
  output.clear();

  // Pass a non-class decl, this should fail
  where = Cpp::Allocate(scope);
  where = Cpp::Construct(Decls[2], where);
  EXPECT_TRUE(where == nullptr);
  // C API
  testing::internal::CaptureStdout();
  auto* I = clang_createInterpreterFromRawPtr(Cpp::GetInterpreter());
  auto scope_c = make_scope(static_cast<clang::Decl*>(scope), I);
  auto object_c = clang_construct(scope_c, nullptr, 1UL);
  EXPECT_TRUE(object_c != nullptr);
  output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(output, "Constructor Executed");
  output.clear();
  auto* dummy = clang_allocate(8);
  EXPECT_TRUE(dummy);
  clang_deallocate(dummy);
  // Clean up resources
  clang_Interpreter_takeInterpreterAsPtr(I);
  clang_Interpreter_dispose(I);
}

// Test zero initialization of PODs and default initialization cases
TEST(FunctionReflectionTest, ConstructPOD) {
#ifdef EMSCRIPTEN
#if CLANG_VERSION_MAJOR < 20
  GTEST_SKIP() << "Test fails for Emscipten builds";
#endif
#endif
  if (llvm::sys::RunningOnValgrind())
    GTEST_SKIP() << "XFAIL due to Valgrind report";
#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif
  std::vector<const char*> interpreter_args = {"-include", "new"};
  Cpp::CreateInterpreter(interpreter_args);

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

  auto *ns = Cpp::GetNamed("PODS");
  Cpp::TCppScope_t scope = Cpp::GetNamed("SomePOD_B", ns);
  EXPECT_TRUE(scope);
  Cpp::TCppObject_t object = Cpp::Construct(scope);
  EXPECT_TRUE(object != nullptr);
  int* fInt = reinterpret_cast<int*>(reinterpret_cast<char*>(object));
  EXPECT_TRUE(*fInt == 0);

  scope = Cpp::GetNamed("SomePOD_C", ns);
  EXPECT_TRUE(scope);
  object = Cpp::Construct(scope);
  EXPECT_TRUE(object);
  auto* fDouble =
      reinterpret_cast<double*>(reinterpret_cast<char*>(object) + sizeof(int));
  EXPECT_EQ(*fDouble, 0.0);
}

// Test nested constructor calls
TEST(FunctionReflectionTest, ConstructNested) {
#ifdef EMSCRIPTEN
#if CLANG_VERSION_MAJOR < 20
  GTEST_SKIP() << "Test fails for Emscipten builds";
#endif
#endif
  if (llvm::sys::RunningOnValgrind())
    GTEST_SKIP() << "XFAIL due to Valgrind report";
#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif

  std::vector<const char*> interpreter_args = {"-include", "new"};
  Cpp::CreateInterpreter(interpreter_args);

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
  Cpp::TCppScope_t scope_A = Cpp::GetNamed("A");
  Cpp::TCppScope_t scope_B = Cpp::GetNamed("B");
  Cpp::TCppObject_t object = Cpp::Construct(scope_B);
  EXPECT_TRUE(object != nullptr);
  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(output, "A Constructor Called\nB Constructor Called\n");
  output.clear();

  // In-memory construction
  testing::internal::CaptureStdout();
  void* arena = Cpp::Allocate(scope_B);
  EXPECT_TRUE(arena == Cpp::Construct(scope_B, arena));

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

TEST(FunctionReflectionTest, ConstructArray) {
#if defined(EMSCRIPTEN)
  GTEST_SKIP() << "Test fails for Emscripten builds";
#endif
  if (llvm::sys::RunningOnValgrind())
    GTEST_SKIP() << "XFAIL due to Valgrind report";
#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif
#if defined(__APPLE__) && (CLANG_VERSION_MAJOR == 16)
  GTEST_SKIP() << "Test fails on Clang16 OS X";
#endif

  Cpp::CreateInterpreter();

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

  Cpp::TCppScope_t scope = Cpp::GetNamed("C");
  std::string output;

  size_t a = 5;                          // Construct an array of 5 objects
  void* where = Cpp::Allocate(scope, a); // operator new

  testing::internal::CaptureStdout();
  EXPECT_TRUE(where == Cpp::Construct(scope, where, a)); // placement new
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

TEST(FunctionReflectionTest, Destruct) {
#ifdef EMSCRIPTEN
  GTEST_SKIP() << "Test fails for Emscipten builds";
#endif
  if (llvm::sys::RunningOnValgrind())
    GTEST_SKIP() << "XFAIL due to Valgrind report";

#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif

  std::vector<const char*> interpreter_args = {"-include", "new"};
  Cpp::CreateInterpreter(interpreter_args);

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
  Cpp::TCppScope_t scope = Cpp::GetNamed("C");
  Cpp::TCppObject_t object = Cpp::Construct(scope);
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
  auto* I = clang_createInterpreterFromRawPtr(Cpp::GetInterpreter());
  auto scope_c = make_scope(static_cast<clang::Decl*>(scope), I);
  auto object_c = clang_construct(scope_c, nullptr, 1UL);
  clang_destruct(object_c, scope_c, true);
  output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(output, "Destructor Executed");
  output.clear();
  // Clean up resources
  clang_Interpreter_takeInterpreterAsPtr(I);
  clang_Interpreter_dispose(I);

  // Failure test, this wrapper should not compile since we explicitly delete
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
}

TEST(FunctionReflectionTest, DestructArray) {
#ifdef EMSCRIPTEN
  GTEST_SKIP() << "Test fails for Emscipten builds";
#endif
  if (llvm::sys::RunningOnValgrind())
    GTEST_SKIP() << "XFAIL due to Valgrind report";

#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif
#if defined(__APPLE__) && (CLANG_VERSION_MAJOR == 16)
  GTEST_SKIP() << "Test fails on Clang16 OS X";
#endif

  std::vector<const char*> interpreter_args = {"-include", "new"};
  Cpp::CreateInterpreter(interpreter_args);

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

  Cpp::TCppScope_t scope = Cpp::GetNamed("C");
  std::string output;

  size_t a = 5;                          // Construct an array of 5 objects
  void* where = Cpp::Allocate(scope, a); // operator new
  EXPECT_TRUE(where == Cpp::Construct(scope, where, a)); // placement new

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
  Cpp::Deallocate(scope, where, 5);

  // perform the same withFree=true
  where = Cpp::Allocate(scope, a);
  EXPECT_TRUE(where == Cpp::Construct(scope, where, a));
  testing::internal::CaptureStdout();
  // FIXME : This should work with the array of objects as well
  // Cpp::Destruct(where, scope, true, 5);
  EXPECT_TRUE(Cpp::Destruct(where, scope, true));
  output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(output, "\nDestructor Executed\n");
  output.clear();
}

TEST(FunctionReflectionTest, UndoTest) {
#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif
#ifdef EMSCRIPTEN
  GTEST_SKIP() << "Test fails for Emscipten builds";
#else
  Cpp::CreateInterpreter();
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

TEST(FunctionReflectionTest, FailingTest1) {
#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif
#ifdef EMSCRIPTEN_SHARED_LIBRARY
  GTEST_SKIP() << "Test fails for Emscipten shared library builds";
#endif
  Cpp::CreateInterpreter();
  EXPECT_FALSE(Cpp::Declare(R"(
    class WithOutEqualOp1 {};
    class WithOutEqualOp2 {};

    WithOutEqualOp1 o1;
    WithOutEqualOp2 o2;

    template<class C1, class C2>
    bool is_equal(const C1& c1, const C2& c2) { return (bool)(c1 == c2); }
  )"));

  Cpp::TCppType_t o1 = Cpp::GetTypeFromScope(Cpp::GetNamed("o1"));
  Cpp::TCppType_t o2 = Cpp::GetTypeFromScope(Cpp::GetNamed("o2"));
  std::vector<Cpp::TCppFunction_t> fns;
  Cpp::GetClassTemplatedMethods("is_equal", Cpp::GetGlobalScope(), fns);
  EXPECT_EQ(fns.size(), 1);

  Cpp::TemplateArgInfo args[2] = {{o1}, {o2}};
  Cpp::TCppScope_t fn = Cpp::InstantiateTemplate(fns[0], args, 2);
  EXPECT_TRUE(fn);

  Cpp::JitCall jit_call = Cpp::MakeFunctionCallable(fn);
  EXPECT_EQ(jit_call.getKind(), Cpp::JitCall::kUnknown); // expected to fail
  EXPECT_FALSE(Cpp::Declare("int x = 1;"));
  EXPECT_FALSE(Cpp::Declare("int y = x;"));
}
