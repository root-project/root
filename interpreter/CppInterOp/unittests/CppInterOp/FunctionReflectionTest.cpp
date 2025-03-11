#include "Utils.h"

#include "clang/AST/ASTContext.h"
#include "clang/Basic/Version.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Interpreter/CppInterOp.h"
#include "clang/Sema/Sema.h"

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
  std::vector<Cpp::TCppFunction_t> template_methods;

  Cpp::GetClassMethods(Decls[0], methods);
  Cpp::GetFunctionTemplatedDecls(Decls[0], template_methods);

  EXPECT_EQ(Cpp::GetName(template_methods[0]), Cpp::GetName(SubDecls[1]));
  EXPECT_EQ(Cpp::GetName(template_methods[1]), Cpp::GetName(SubDecls[2]));
  EXPECT_EQ(Cpp::GetName(template_methods[2]), Cpp::GetName(SubDecls[3]));
  EXPECT_EQ(Cpp::GetName(methods[0])         , Cpp::GetName(SubDecls[4]));
  EXPECT_EQ(Cpp::GetName(methods[1])         , Cpp::GetName(SubDecls[5]));
  EXPECT_EQ(Cpp::GetName(template_methods[3]), Cpp::GetName(SubDecls[6]));
  EXPECT_EQ(Cpp::GetName(methods[2])         , Cpp::GetName(SubDecls[7]));
  EXPECT_EQ(Cpp::GetName(methods[3])         , Cpp::GetName(SubDecls[8]));
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
    };

    namespace N
    {
      void f(int i, double d, long l = 0, char ch = 'a') {}
    }

    void f1() {}
    C f2(int i, double d, long l = 0, char ch = 'a') { return C(); }
    C *f3(int i, double d, long l = 0, char ch = 'a') { return new C(); }
    void f4(int i = 0, double d = 0.0, long l = 0, char ch = 'a') {}
    class ABC {};
    )";

  GetAllTopLevelDecls(code, Decls, true);
  GetAllSubDecls(Decls[0], Decls);
  GetAllSubDecls(Decls[1], Decls);

  EXPECT_EQ(Cpp::GetFunctionSignature(Decls[2]), "void f1()");
  EXPECT_EQ(Cpp::GetFunctionSignature(Decls[3]),
            "C f2(int i, double d, long l = 0, char ch = 'a')");
  EXPECT_EQ(Cpp::GetFunctionSignature(Decls[4]),
            "C *f3(int i, double d, long l = 0, char ch = 'a')");
  EXPECT_EQ(Cpp::GetFunctionSignature(Decls[5]),
            "void f4(int i = 0, double d = 0., long l = 0, char ch = 'a')");
  EXPECT_EQ(Cpp::GetFunctionSignature(Decls[6]),
            "<unknown>");
  EXPECT_EQ(Cpp::GetFunctionSignature(Decls[8]),
            "void C::f(int i, double d, long l = 0, char ch = 'a')");
  EXPECT_EQ(Cpp::GetFunctionSignature(Decls[13]),
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
}

TEST(FunctionReflectionTest, InstantiateTemplateFunctionFromString) {
  if (llvm::sys::RunningOnValgrind())
    GTEST_SKIP() << "XFAIL due to Valgrind report";
  Cpp::CreateInterpreter();
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
  if (llvm::sys::RunningOnValgrind())
    GTEST_SKIP() << "XFAIL due to Valgrind report";
#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif
  std::vector<Decl*> Decls, SubDecls;
  std::string code = "int f1(int i) { return i * i; }";

  GetAllTopLevelDecls(code, Decls);

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
  if (llvm::sys::RunningOnValgrind())
    GTEST_SKIP() << "XFAIL due to Valgrind report";
  std::vector<Decl*> Decls;
  std::string code = R"(
      typedef struct _name {
        _name() { p[0] = (void*)0x1; p[1] = (void*)0x2; p[2] = (void*)0x3; }
        void* p[3];
      } name;
    )";

  GetAllTopLevelDecls(code, Decls);
  auto *CtorD
    = (clang::CXXConstructorDecl*)Cpp::GetDefaultConstructor(Decls[0]);
  auto Ctor = Cpp::MakeFunctionCallable(CtorD);
  EXPECT_TRUE((bool)Ctor) << "Failed to build a wrapper for the ctor";
  void* object = nullptr;
  Ctor.Invoke(&object);
  EXPECT_TRUE(object) << "Failed to call the ctor.";
  // Building a wrapper with a typedef decl must be possible.
  Cpp::Destruct(object, Decls[1]);
}


TEST(FunctionReflectionTest, GetFunctionCallWrapper) {
  if (llvm::sys::RunningOnValgrind())
    GTEST_SKIP() << "XFAIL due to Valgrind report";
#if defined(CPPINTEROP_USE_CLING) && defined(_WIN32)
  GTEST_SKIP() << "Disabled, invoking functions containing printf does not work with Cling on Windows";
#endif
  std::vector<Decl*> Decls;
  std::string code = R"(
    int f1(int i) { return i * i; }
    )";

  GetAllTopLevelDecls(code, Decls);

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

  GetAllTopLevelDecls(code1, Decls1);
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
  FCI_op.Invoke((void*)&boolean, {args, /*args_size=*/1}, object);
  EXPECT_TRUE(boolean);
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
  if (llvm::sys::RunningOnValgrind())
    GTEST_SKIP() << "XFAIL due to Valgrind report";
#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif

  Cpp::CreateInterpreter();

  Interp->declare(R"(
    #include <new>
    extern "C" int printf(const char*,...);
    class C {
      int x;
      C() {
        x = 12345;
        printf("Constructor Executed");
      }
    };
    )");

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
}

TEST(FunctionReflectionTest, Destruct) {
  if (llvm::sys::RunningOnValgrind())
    GTEST_SKIP() << "XFAIL due to Valgrind report";

#ifdef _WIN32
  GTEST_SKIP() << "Disabled on Windows. Needs fixing.";
#endif

  Cpp::CreateInterpreter();

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
  Cpp::Destruct(object, scope);
  std::string output = testing::internal::GetCapturedStdout();

  EXPECT_EQ(output, "Destructor Executed");

  output.clear();
  testing::internal::CaptureStdout();
  object = Cpp::Construct(scope);
  // Make sure we do not call delete by adding an explicit Deallocate. If we
  // called delete the Deallocate will cause a double deletion error.
  Cpp::Destruct(object, scope, /*withFree=*/false);
  Cpp::Deallocate(scope, object);
  output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(output, "Destructor Executed");
}
