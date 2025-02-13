#include "Utils.h"

#include "clang/Interpreter/CppInterOp.h"
#include "clang-c/CXCppInterOp.h"

#include "clang/AST/ASTContext.h"
#include "clang/Interpreter/CppInterOp.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Sema.h"

#include "clang/AST/ASTDumper.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/GlobalDecl.h"

#include "llvm/Support/Valgrind.h"

#include "gtest/gtest.h"

#include <string>

using namespace TestUtils;
using namespace llvm;
using namespace clang;

TEST(ScopeReflectionTest, Demangle) {
  if (llvm::sys::RunningOnValgrind())
    GTEST_SKIP() << "XFAIL due to Valgrind report";

  std::string code = R"(
    int add(int x, int y) { return x + y; }
    int add(double x, double y) { return x + y; }
  )";

  std::vector<Decl*> Decls;
  GetAllTopLevelDecls(code, Decls);
  EXPECT_EQ(Decls.size(), 2);

  auto Add_int = clang::GlobalDecl(static_cast<clang::NamedDecl*>(Decls[0]));
  auto Add_double = clang::GlobalDecl(static_cast<clang::NamedDecl*>(Decls[1]));

  std::string mangled_add_int;
  std::string mangled_add_double;
  compat::maybeMangleDeclName(Add_int, mangled_add_int);
  compat::maybeMangleDeclName(Add_double, mangled_add_double);

  std::string demangled_add_int = Cpp::Demangle(mangled_add_int);
  std::string demangled_add_double = Cpp::Demangle(mangled_add_double);

  EXPECT_NE(demangled_add_int.find(Cpp::GetQualifiedCompleteName(Decls[0])),
            std::string::npos);
  EXPECT_NE(demangled_add_double.find(Cpp::GetQualifiedCompleteName(Decls[1])),
            std::string::npos);
}

TEST(ScopeReflectionTest, IsAggregate) {
  std::vector<Decl *> Decls;
  std::string code = R"(
    char cv[4] = {};
    int x[] = {};
    union u { int a; const char* b; };
    struct S {
      int x;
      struct Foo {
        int i;
        int j;
        int a[3];
      } b;
   };
   int y = 10; // Not an aggregate type
  )";

  GetAllTopLevelDecls(code, Decls);
  EXPECT_TRUE(Cpp::IsAggregate(Decls[0]));
  EXPECT_TRUE(Cpp::IsAggregate(Decls[1]));
  EXPECT_TRUE(Cpp::IsAggregate(Decls[2]));
  EXPECT_TRUE(Cpp::IsAggregate(Decls[3]));
  EXPECT_FALSE(Cpp::IsAggregate(Decls[4]));
}

// Check that the CharInfo table has been constructed reasonably.
TEST(ScopeReflectionTest, IsNamespace) {
  std::vector<Decl*> Decls;
  GetAllTopLevelDecls("namespace N {} class C{}; int I;", Decls);
  EXPECT_TRUE(Cpp::IsNamespace(Decls[0]));
  EXPECT_FALSE(Cpp::IsNamespace(Decls[1]));
  EXPECT_FALSE(Cpp::IsNamespace(Decls[2]));
}

TEST(ScopeReflectionTest, IsClass) {
  std::vector<Decl*> Decls;
  GetAllTopLevelDecls("namespace N {} class C{}; int I;", Decls);
  EXPECT_FALSE(Cpp::IsClass(Decls[0]));
  EXPECT_TRUE(Cpp::IsClass(Decls[1]));
  EXPECT_FALSE(Cpp::IsClass(Decls[2]));
}

TEST(ScopeReflectionTest, IsClassPolymorphic) {
  std::vector<Decl*> Decls;
  GetAllTopLevelDecls(R"(
    namespace N {}
    
    class C{};
    
    class C2 {
    public:
      virtual ~C2() {}
    };
    
    int I;
  )",
                      Decls);

  EXPECT_FALSE(Cpp::IsClassPolymorphic(Decls[0]));
  EXPECT_FALSE(Cpp::IsClassPolymorphic(Decls[1]));
  EXPECT_TRUE(Cpp::IsClassPolymorphic(Decls[2]));
  EXPECT_FALSE(Cpp::IsClassPolymorphic(Decls[3]));
}

TEST(ScopeReflectionTest, IsComplete) {
  std::vector<Decl*> Decls;
  std::string code = R"(
    namespace N {}
    class C{};
    int I;
    struct S;
    enum E : int;
    union U{};
    template<typename T> struct TemplatedS{ T x; };
    TemplatedS<int> templateF() { return {}; };
  )";
  GetAllTopLevelDecls(code,
                      Decls);
  EXPECT_TRUE(Cpp::IsComplete(Decls[0]));
  EXPECT_TRUE(Cpp::IsComplete(Decls[1]));
  EXPECT_TRUE(Cpp::IsComplete(Decls[2]));
  EXPECT_FALSE(Cpp::IsComplete(Decls[3]));
  EXPECT_FALSE(Cpp::IsComplete(Decls[4]));
  EXPECT_TRUE(Cpp::IsComplete(Decls[5]));
  Cpp::TCppType_t retTy = Cpp::GetFunctionReturnType(Decls[7]);
  EXPECT_TRUE(Cpp::IsComplete(Cpp::GetScopeFromType(retTy)));
  EXPECT_FALSE(Cpp::IsComplete(nullptr));
}

TEST(ScopeReflectionTest, SizeOf) {
  std::vector<Decl*> Decls;
  std::string code = R"(namespace N {} class C{}; int I; struct S;
                        enum E : int; union U{}; class Size4{int i;};
                        struct Size16 {short a; double b;};
                       )";
  GetAllTopLevelDecls(code, Decls);
  EXPECT_EQ(Cpp::SizeOf(Decls[0]), (size_t)0);
  EXPECT_EQ(Cpp::SizeOf(Decls[1]), (size_t)1);
  EXPECT_EQ(Cpp::SizeOf(Decls[2]), (size_t)0);
  EXPECT_EQ(Cpp::SizeOf(Decls[3]), (size_t)0);
  EXPECT_EQ(Cpp::SizeOf(Decls[4]), (size_t)0);
  EXPECT_EQ(Cpp::SizeOf(Decls[5]), (size_t)1);
  EXPECT_EQ(Cpp::SizeOf(Decls[6]), (size_t)4);
  EXPECT_EQ(Cpp::SizeOf(Decls[7]), (size_t)16);
}


TEST(ScopeReflectionTest, IsBuiltin) {
  // static std::set<std::string> g_builtins =
  // {"bool", "char", "signed char", "unsigned char", "wchar_t", "short", "unsigned short",
  //  "int", "unsigned int", "long", "unsigned long", "long long", "unsigned long long",
  //  "float", "double", "long double", "void"}

  Cpp::CreateInterpreter();
  ASTContext &C = Interp->getCI()->getASTContext();
  EXPECT_TRUE(Cpp::IsBuiltin(C.BoolTy.getAsOpaquePtr()));
  EXPECT_TRUE(Cpp::IsBuiltin(C.CharTy.getAsOpaquePtr()));
  EXPECT_TRUE(Cpp::IsBuiltin(C.SignedCharTy.getAsOpaquePtr()));
  EXPECT_TRUE(Cpp::IsBuiltin(C.VoidTy.getAsOpaquePtr()));
  // ...

  // complex
  EXPECT_TRUE(Cpp::IsBuiltin(C.getComplexType(C.FloatTy).getAsOpaquePtr()));
  EXPECT_TRUE(Cpp::IsBuiltin(C.getComplexType(C.DoubleTy).getAsOpaquePtr()));
  EXPECT_TRUE(Cpp::IsBuiltin(C.getComplexType(C.LongDoubleTy).getAsOpaquePtr()));
  EXPECT_TRUE(Cpp::IsBuiltin(C.getComplexType(C.Float128Ty).getAsOpaquePtr()));

  // std::complex
  std::vector<Decl*> Decls;
  Interp->declare("#include <complex>");
  Sema &S = Interp->getCI()->getSema();
  auto lookup = S.getStdNamespace()->lookup(&C.Idents.get("complex"));
  auto *CTD = cast<ClassTemplateDecl>(lookup.front());
  for (ClassTemplateSpecializationDecl *CTSD : CTD->specializations())
    EXPECT_TRUE(Cpp::IsBuiltin(C.getTypeDeclType(CTSD).getAsOpaquePtr()));
}

TEST(ScopeReflectionTest, IsTemplate) {
  std::vector<Decl *> Decls;
  std::string code = R"(template<typename T>
                        class A{};

                        class C{
                          template<typename T>
                          int func(T t) {
                            return 0;
                          }
                        };

                        template<typename T>
                        T f(T t) {
                          return t;
                        }

                        void g() {} )";

  GetAllTopLevelDecls(code, Decls);
  EXPECT_TRUE(Cpp::IsTemplate(Decls[0]));
  EXPECT_FALSE(Cpp::IsTemplate(Decls[1]));
  EXPECT_TRUE(Cpp::IsTemplate(Decls[2]));
  EXPECT_FALSE(Cpp::IsTemplate(Decls[3]));
}

TEST(ScopeReflectionTest, IsTemplateSpecialization) {
  std::vector<Decl *> Decls;
  std::string code = R"(
    template<typename T>
    class A{};

    A<int> a;
    )";

  GetAllTopLevelDecls(code, Decls);
  EXPECT_FALSE(Cpp::IsTemplateSpecialization(Decls[0]));
  EXPECT_FALSE(Cpp::IsTemplateSpecialization(Decls[1]));
  EXPECT_TRUE(Cpp::IsTemplateSpecialization(
          Cpp::GetScopeFromType(Cpp::GetVariableType(Decls[1]))));
}

TEST(ScopeReflectionTest, IsTypedefed) {
  std::vector<Decl *> Decls;
  std::string code = R"(
    typedef int I;
    using D = double;
    class C {};
    )";

  GetAllTopLevelDecls(code, Decls);
  EXPECT_TRUE(Cpp::IsTypedefed(Decls[0]));
  EXPECT_TRUE(Cpp::IsTypedefed(Decls[1]));
  EXPECT_FALSE(Cpp::IsTypedefed(Decls[2]));
}

TEST(ScopeReflectionTest, IsAbstract) {
  std::vector<Decl *> Decls;
  std::string code = R"(
    class A {};

    class B {
      virtual int f() = 0;
    };

    int sum(int a, int b) {
      return a+b;
    }
  )";

  GetAllTopLevelDecls(code, Decls);
  EXPECT_FALSE(Cpp::IsAbstract(Decls[0]));
  EXPECT_TRUE(Cpp::IsAbstract(Decls[1]));
  EXPECT_FALSE(Cpp::IsAbstract(Decls[2]));
}

TEST(ScopeReflectionTest, IsVariable) {
  std::vector<Decl *> Decls;
  std::string code = R"(
    int i;

    class C {
    public:
      int a;
      static int b;
    };
  )";

  GetAllTopLevelDecls(code, Decls);
  EXPECT_TRUE(Cpp::IsVariable(Decls[0]));
  EXPECT_FALSE(Cpp::IsVariable(Decls[1]));

  std::vector<Decl *> SubDecls;
  GetAllSubDecls(Decls[1], SubDecls);
  EXPECT_FALSE(Cpp::IsVariable(SubDecls[0]));
  EXPECT_FALSE(Cpp::IsVariable(SubDecls[1]));
  EXPECT_FALSE(Cpp::IsVariable(SubDecls[2]));
  EXPECT_TRUE(Cpp::IsVariable(SubDecls[3]));
}

TEST(ScopeReflectionTest, GetName) {
  std::vector<Decl*> Decls;
  std::string code = R"(namespace N {} class C{}; int I; struct S;
                        enum E : int; union U{}; class Size4{int i;};
                        struct Size16 {short a; double b;};
                       )";
  GetAllTopLevelDecls(code, Decls);
  EXPECT_EQ(Cpp::GetName(Decls[0]), "N");
  EXPECT_EQ(Cpp::GetName(Decls[1]), "C");
  EXPECT_EQ(Cpp::GetName(Decls[2]), "I");
  EXPECT_EQ(Cpp::GetName(Decls[3]), "S");
  EXPECT_EQ(Cpp::GetName(Decls[4]), "E");
  EXPECT_EQ(Cpp::GetName(Decls[5]), "U");
  EXPECT_EQ(Cpp::GetName(Decls[6]), "Size4");
  EXPECT_EQ(Cpp::GetName(Decls[7]), "Size16");
  EXPECT_EQ(Cpp::GetName(nullptr), "<unnamed>");
}

TEST(ScopeReflectionTest, GetCompleteName) {
  std::vector<Decl *> Decls;
  std::string code = R"(namespace N {}
                        class C{};
                        int I;
                        struct S;
                        enum E : int;
                        union U{};
                        class Size4{int i;};
                        struct Size16 {short a; double b;};

                        template<typename T>
                        class A {};
                        A<int> a;

                        enum { enum1 };
                       )";
  GetAllTopLevelDecls(code, Decls);

  EXPECT_EQ(Cpp::GetCompleteName(Decls[0]), "N");
  EXPECT_EQ(Cpp::GetCompleteName(Decls[1]), "C");
  EXPECT_EQ(Cpp::GetCompleteName(Decls[2]), "I");
  EXPECT_EQ(Cpp::GetCompleteName(Decls[3]), "S");
  EXPECT_EQ(Cpp::GetCompleteName(Decls[4]), "E");
  EXPECT_EQ(Cpp::GetCompleteName(Decls[5]), "U");
  EXPECT_EQ(Cpp::GetCompleteName(Decls[6]), "Size4");
  EXPECT_EQ(Cpp::GetCompleteName(Decls[7]), "Size16");
  EXPECT_EQ(Cpp::GetCompleteName(Decls[8]), "A");
  EXPECT_EQ(Cpp::GetCompleteName(Cpp::GetScopeFromType(
                                             Cpp::GetVariableType(
                                                     Decls[9]))), "A<int>");
  EXPECT_EQ(Cpp::GetCompleteName(Decls[10]), "(unnamed)");
  EXPECT_EQ(Cpp::GetCompleteName(nullptr), "<unnamed>");
}

TEST(ScopeReflectionTest, GetQualifiedName) {
  std::vector<Decl*> Decls;
  std::string code = R"(namespace N {
                        class C {
                          int i;
                          enum E { A, B };
                        };
                        }
                       )";
  GetAllTopLevelDecls(code, Decls);
  GetAllSubDecls(Decls[0], Decls);
  GetAllSubDecls(Decls[1], Decls);

  EXPECT_EQ(Cpp::GetQualifiedName(0), "<unnamed>");
  EXPECT_EQ(Cpp::GetQualifiedName(Decls[0]), "N");
  EXPECT_EQ(Cpp::GetQualifiedName(Decls[1]), "N::C");
  EXPECT_EQ(Cpp::GetQualifiedName(Decls[3]), "N::C::i");
  EXPECT_EQ(Cpp::GetQualifiedName(Decls[4]), "N::C::E");
}

TEST(ScopeReflectionTest, GetQualifiedCompleteName) {
  std::vector<Decl*> Decls;
  std::string code = R"(namespace N {
                        class C {
                          int i;
                          enum E { A, B };
                        };
                        template<typename T>
                        class A {};
                        A<int> a;
                        }
                       )";
  GetAllTopLevelDecls(code, Decls);
  GetAllSubDecls(Decls[0], Decls);
  GetAllSubDecls(Decls[1], Decls);

  EXPECT_EQ(Cpp::GetQualifiedCompleteName(0), "<unnamed>");
  EXPECT_EQ(Cpp::GetQualifiedCompleteName(Decls[0]), "N");
  EXPECT_EQ(Cpp::GetQualifiedCompleteName(Decls[1]), "N::C");
  EXPECT_EQ(Cpp::GetQualifiedCompleteName(Decls[2]), "N::A");
  EXPECT_EQ(Cpp::GetQualifiedCompleteName(Cpp::GetScopeFromType(Cpp::GetVariableType(Decls[3]))), "N::A<int>");
  EXPECT_EQ(Cpp::GetQualifiedCompleteName(Decls[5]), "N::C::i");
  EXPECT_EQ(Cpp::GetQualifiedCompleteName(Decls[6]), "N::C::E");
}

TEST(ScopeReflectionTest, GetUsingNamespaces) {
  std::vector<Decl *> Decls, Decls1;
  std::string code = R"(
    namespace abc {

    class C {};

    }
    using namespace std;
    using namespace abc;

    using I = int;
  )";

  GetAllTopLevelDecls(code, Decls);
  std::vector<void *> usingNamespaces;
  usingNamespaces = Cpp::GetUsingNamespaces(
          Decls[0]->getASTContext().getTranslationUnitDecl());

  //EXPECT_EQ(Cpp::GetName(usingNamespaces[0]), "runtime");
  EXPECT_EQ(Cpp::GetName(usingNamespaces[usingNamespaces.size()-2]), "std");
  EXPECT_EQ(Cpp::GetName(usingNamespaces[usingNamespaces.size()-1]), "abc");

  std::string code1 = R"(
    int x;
  )";

  GetAllTopLevelDecls(code1, Decls1);
  std::vector<void*> usingNamespaces1;
  usingNamespaces1 = Cpp::GetUsingNamespaces(Decls1[0]);
  EXPECT_EQ(usingNamespaces1.size(), 0);
}

TEST(ScopeReflectionTest, GetGlobalScope) {
  EXPECT_EQ(Cpp::GetQualifiedName(Cpp::GetGlobalScope()), "");
  EXPECT_EQ(Cpp::GetName(Cpp::GetGlobalScope()), "");
}

TEST(ScopeReflectionTest, GetUnderlyingScope) {
  std::vector<Decl *> Decls;
  std::string code = R"(
    namespace N {
      class C {};
    }

    typedef N::C NC;
    typedef int INT;
  )";

  GetAllTopLevelDecls(code, Decls);

  EXPECT_EQ(Cpp::GetQualifiedName(Cpp::GetUnderlyingScope(Decls[0])), "N");
  EXPECT_EQ(Cpp::GetQualifiedName(Cpp::GetUnderlyingScope(Decls[1])), "N::C");
  EXPECT_EQ(Cpp::GetQualifiedName(Cpp::GetUnderlyingScope(Decls[2])), "INT");
  EXPECT_EQ(Cpp::GetQualifiedName(Cpp::GetUnderlyingScope(nullptr)), "<unnamed>");
}

TEST(ScopeReflectionTest, GetScope) {
  std::string code = R"(namespace N {
                        class C {
                          int i;
                          enum E { A, B };
                        };
                        }

                        typedef N::C T;
                       )";

  Cpp::CreateInterpreter();
  Interp->declare(code);
  Cpp::TCppScope_t tu = Cpp::GetScope("", 0);
  Cpp::TCppScope_t ns_N = Cpp::GetScope("N", 0);
  Cpp::TCppScope_t cl_C = Cpp::GetScope("C", ns_N);
  Cpp::TCppScope_t td_T = Cpp::GetScope("T", 0);
  Cpp::TCppScope_t non_existent = Cpp::GetScope("sum", 0);

  EXPECT_EQ(Cpp::GetQualifiedName(tu), "");
  EXPECT_EQ(Cpp::GetQualifiedName(ns_N), "N");
  EXPECT_EQ(Cpp::GetQualifiedName(cl_C), "N::C");
  EXPECT_EQ(Cpp::GetQualifiedName(td_T), "T");
  EXPECT_EQ(Cpp::GetQualifiedName(non_existent), "<unnamed>");
}

TEST(ScopeReflectionTest, GetScopefromCompleteName) {
  std::string code = R"(namespace N1 {
                        namespace N2 {
                          class C {
                            struct S {};
                          };
                        }
                        }
                       )";

  Cpp::CreateInterpreter();

  Interp->declare(code);
  EXPECT_EQ(Cpp::GetQualifiedName(Cpp::GetScopeFromCompleteName("N1")), "N1");
  EXPECT_EQ(Cpp::GetQualifiedName(Cpp::GetScopeFromCompleteName("N1::N2")), "N1::N2");
  EXPECT_EQ(Cpp::GetQualifiedName(Cpp::GetScopeFromCompleteName("N1::N2::C")), "N1::N2::C");
  EXPECT_EQ(Cpp::GetQualifiedName(Cpp::GetScopeFromCompleteName("N1::N2::C::S")), "N1::N2::C::S");
}

TEST(ScopeReflectionTest, GetNamed) {
  std::string code = R"(namespace N1 {
                        namespace N2 {
                          class C {
                            int i;
                            enum E { A, B };
                            struct S {};
                          };
                        }
                        }
                       )";
  Cpp::CreateInterpreter();

  Interp->declare(code);
  Cpp::TCppScope_t ns_N1 = Cpp::GetNamed("N1", nullptr);
  Cpp::TCppScope_t ns_N2 = Cpp::GetNamed("N2", ns_N1);
  Cpp::TCppScope_t cl_C = Cpp::GetNamed("C", ns_N2);
  Cpp::TCppScope_t en_E = Cpp::GetNamed("E", cl_C);
  EXPECT_EQ(Cpp::GetQualifiedName(ns_N1), "N1");
  EXPECT_EQ(Cpp::GetQualifiedName(ns_N2), "N1::N2");
  EXPECT_EQ(Cpp::GetQualifiedName(cl_C), "N1::N2::C");
  EXPECT_EQ(Cpp::GetQualifiedName(Cpp::GetNamed("i", cl_C)), "N1::N2::C::i");
  EXPECT_EQ(Cpp::GetQualifiedName(en_E), "N1::N2::C::E");
  EXPECT_EQ(Cpp::GetQualifiedName(Cpp::GetNamed("A", en_E)), "N1::N2::C::A");
  EXPECT_EQ(Cpp::GetQualifiedName(Cpp::GetNamed("B", en_E)), "N1::N2::C::B");
  EXPECT_EQ(Cpp::GetQualifiedName(Cpp::GetNamed("A", cl_C)), "N1::N2::C::A");
  EXPECT_EQ(Cpp::GetQualifiedName(Cpp::GetNamed("B", cl_C)), "N1::N2::C::B");
  EXPECT_EQ(Cpp::GetQualifiedName(Cpp::GetNamed("S", cl_C)), "N1::N2::C::S");

  Interp->process("#include <string>");
  Cpp::TCppScope_t std_ns = Cpp::GetNamed("std", nullptr);
  Cpp::TCppScope_t std_string_class = Cpp::GetNamed("string", std_ns);
  Cpp::TCppScope_t std_string_npos_var = Cpp::GetNamed("npos", std_string_class);
  EXPECT_EQ(Cpp::GetQualifiedName(std_ns), "std");
  EXPECT_EQ(Cpp::GetQualifiedName(std_string_class), "std::string");
  EXPECT_EQ(Cpp::GetQualifiedName(std_string_npos_var), "std::basic_string<char>::npos");
}

TEST(ScopeReflectionTest, GetParentScope) {
  std::string code = R"(namespace N1 {
                        namespace N2 {
                          class C {
                            int i;
                            enum E { A, B };
                            struct S {};
                          };
                        }
                        }
                       )";

  Cpp::CreateInterpreter();

  Interp->declare(code);
  Cpp::TCppScope_t ns_N1 = Cpp::GetNamed("N1");
  Cpp::TCppScope_t ns_N2 = Cpp::GetNamed("N2", ns_N1);
  Cpp::TCppScope_t cl_C = Cpp::GetNamed("C", ns_N2);
  Cpp::TCppScope_t int_i = Cpp::GetNamed("i", cl_C);
  Cpp::TCppScope_t en_E = Cpp::GetNamed("E", cl_C);
  Cpp::TCppScope_t en_A = Cpp::GetNamed("A", cl_C);
  Cpp::TCppScope_t en_B = Cpp::GetNamed("B", cl_C);

  EXPECT_EQ(Cpp::GetQualifiedName(Cpp::GetParentScope(ns_N1)), "");
  EXPECT_EQ(Cpp::GetQualifiedName(Cpp::GetParentScope(ns_N2)), "N1");
  EXPECT_EQ(Cpp::GetQualifiedName(Cpp::GetParentScope(cl_C)), "N1::N2");
  EXPECT_EQ(Cpp::GetQualifiedName(Cpp::GetParentScope(int_i)), "N1::N2::C");
  EXPECT_EQ(Cpp::GetQualifiedName(Cpp::GetParentScope(en_E)), "N1::N2::C");
  EXPECT_EQ(Cpp::GetQualifiedName(Cpp::GetParentScope(en_A)), "N1::N2::C::E");
  EXPECT_EQ(Cpp::GetQualifiedName(Cpp::GetParentScope(en_B)), "N1::N2::C::E");
}

TEST(ScopeReflectionTest, GetScopeFromType) {
  std::vector<Decl *> Decls;
  std::string code = R"(
    namespace N {
    class C {};
    struct S {};
    typedef C T;
    enum E {};
    }

    N::C c;

    N::S s;

    int i;
    
    N::T t;

    N::E e;

    N::C myFunc();
  )";

  GetAllTopLevelDecls(code, Decls);
  QualType QT1 = llvm::dyn_cast<VarDecl>(Decls[1])->getType();
  QualType QT2 = llvm::dyn_cast<VarDecl>(Decls[2])->getType();
  QualType QT3 = llvm::dyn_cast<VarDecl>(Decls[3])->getType();
  QualType QT4 = llvm::dyn_cast<VarDecl>(Decls[4])->getType();
  QualType QT5 = llvm::dyn_cast<VarDecl>(Decls[5])->getType();
  QualType QT6 = llvm::dyn_cast<FunctionDecl>(Decls[6])->getReturnType();
  EXPECT_EQ(Cpp::GetQualifiedName(Cpp::GetScopeFromType(QT1.getAsOpaquePtr())),
          "N::C");
  EXPECT_EQ(Cpp::GetQualifiedName(Cpp::GetScopeFromType(QT2.getAsOpaquePtr())),
          "N::S");
  EXPECT_EQ(Cpp::GetScopeFromType(QT3.getAsOpaquePtr()),
          (Cpp::TCppScope_t) 0);
  EXPECT_EQ(Cpp::GetQualifiedName(Cpp::GetScopeFromType(QT4.getAsOpaquePtr())),
          "N::C");
  EXPECT_EQ(Cpp::GetQualifiedName(Cpp::GetScopeFromType(QT5.getAsOpaquePtr())),
            "N::E");
  EXPECT_EQ(Cpp::GetQualifiedName(Cpp::GetScopeFromType(QT6.getAsOpaquePtr())),
            "N::C");
}

TEST(ScopeReflectionTest, GetNumBases) {
  std::vector<Decl *> Decls;
  std::string code = R"(
    class A {};
    class B : virtual public A {};
    class C : virtual public A {};
    class D : public B, public C {};
    class E : public D {};
    class NoDef;
  )";

  GetAllTopLevelDecls(code, Decls);

  EXPECT_EQ(Cpp::GetNumBases(Decls[0]), 0);
  EXPECT_EQ(Cpp::GetNumBases(Decls[1]), 1);
  EXPECT_EQ(Cpp::GetNumBases(Decls[2]), 1);
  EXPECT_EQ(Cpp::GetNumBases(Decls[3]), 2);
  EXPECT_EQ(Cpp::GetNumBases(Decls[4]), 1);
  // FIXME: Perhaps we should have a special number or error out as this
  // operation is not well defined if a class has no definition.
  EXPECT_EQ(Cpp::GetNumBases(Decls[5]), 0);
}

TEST(ScopeReflectionTest, GetBaseClass) {
  std::vector<Decl *> Decls;
  std::string code = R"(
    class A {};
    class B : virtual public A {};
    class C : virtual public A {};
    class D : public B, public C {};
    class E : public D {};

    template <typename T>
    class TC1 {};

    template <typename T>
    class TC2 : TC1 <T> {};

    template <typename T>
    class TC3 :public A {};

    TC2<A> var;
    TC3<A> var1;
    int a;
  )";

  GetAllTopLevelDecls(code, Decls);

  auto get_base_class_name = [](Decl *D, int i) {
      return Cpp::GetQualifiedName(Cpp::GetBaseClass(D, i));
  };

  EXPECT_EQ(get_base_class_name(Decls[1], 0), "A");
  EXPECT_EQ(get_base_class_name(Decls[2], 0), "A");
  EXPECT_EQ(get_base_class_name(Decls[3], 0), "B");
  EXPECT_EQ(get_base_class_name(Decls[3], 1), "C");
  EXPECT_EQ(get_base_class_name(Decls[4], 0), "D");
  EXPECT_EQ(get_base_class_name(Decls[10], 0), "<unnamed>");

  auto *VD = Cpp::GetNamed("var");
  auto *VT = Cpp::GetVariableType(VD);
  auto *TC2_A_Decl = Cpp::GetScopeFromType(VT);
  auto *TC1_A_Decl = Cpp::GetBaseClass(TC2_A_Decl, 0);
  EXPECT_EQ(Cpp::GetCompleteName(TC1_A_Decl), "TC1<A>");

  auto* VD1 = Cpp::GetNamed("var1");
  auto* VT1 = Cpp::GetVariableType(VD1);
  auto* TC3_A_Decl = Cpp::GetScopeFromType(VT1);
  auto* A_class = Cpp::GetBaseClass(TC3_A_Decl, 0);
  EXPECT_EQ(Cpp::GetCompleteName(A_class), "A");
}

TEST(ScopeReflectionTest, IsSubclass) {
  std::vector<Decl *> Decls;
  std::string code = R"(
    class A {};
    class B : virtual public A {};
    class C : virtual public A {};
    class D : public B, public C {};
    class E : public D {};
    void check();
  )";

  GetAllTopLevelDecls(code, Decls);

  EXPECT_TRUE(Cpp::IsSubclass(Decls[0], Decls[0]));
  EXPECT_TRUE(Cpp::IsSubclass(Decls[1], Decls[0]));
  EXPECT_TRUE(Cpp::IsSubclass(Decls[2], Decls[0]));
  EXPECT_TRUE(Cpp::IsSubclass(Decls[3], Decls[0]));
  EXPECT_TRUE(Cpp::IsSubclass(Decls[4], Decls[0]));
  EXPECT_FALSE(Cpp::IsSubclass(Decls[0], Decls[1]));
  EXPECT_TRUE(Cpp::IsSubclass(Decls[1], Decls[1]));
  EXPECT_FALSE(Cpp::IsSubclass(Decls[2], Decls[1]));
  EXPECT_TRUE(Cpp::IsSubclass(Decls[3], Decls[1]));
  EXPECT_TRUE(Cpp::IsSubclass(Decls[4], Decls[1]));
  EXPECT_FALSE(Cpp::IsSubclass(Decls[0], Decls[2]));
  EXPECT_FALSE(Cpp::IsSubclass(Decls[1], Decls[2]));
  EXPECT_TRUE(Cpp::IsSubclass(Decls[2], Decls[2]));
  EXPECT_TRUE(Cpp::IsSubclass(Decls[3], Decls[2]));
  EXPECT_TRUE(Cpp::IsSubclass(Decls[4], Decls[2]));
  EXPECT_FALSE(Cpp::IsSubclass(Decls[0], Decls[3]));
  EXPECT_FALSE(Cpp::IsSubclass(Decls[1], Decls[3]));
  EXPECT_FALSE(Cpp::IsSubclass(Decls[2], Decls[3]));
  EXPECT_TRUE(Cpp::IsSubclass(Decls[3], Decls[3]));
  EXPECT_TRUE(Cpp::IsSubclass(Decls[4], Decls[3]));
  EXPECT_FALSE(Cpp::IsSubclass(Decls[0], Decls[4]));
  EXPECT_FALSE(Cpp::IsSubclass(Decls[1], Decls[4]));
  EXPECT_FALSE(Cpp::IsSubclass(Decls[2], Decls[4]));
  EXPECT_FALSE(Cpp::IsSubclass(Decls[3], Decls[4]));
  EXPECT_TRUE(Cpp::IsSubclass(Decls[4], Decls[4]));
  EXPECT_FALSE(Cpp::IsSubclass(Decls[4], Decls[5]));
  EXPECT_FALSE(Cpp::IsSubclass(Decls[4], nullptr));
}

TEST(ScopeReflectionTest, GetBaseClassOffset) {
  std::vector<Decl *> Decls;
#define Stringify(s) Stringifyx(s)
#define Stringifyx(...) #__VA_ARGS__
#define CODE                                                    \
  struct A { int m_a; };                                         \
  struct B { int m_b; };                                         \
  struct C : virtual A, virtual B { int m_c; };                  \
  struct D : virtual A, virtual B, public C { int m_d; };        \
  struct E : public A, public B { int m_e; };                    \
  struct F : public A { int m_f; };                              \
  struct G : public F { int m_g; };

CODE;

  GetAllTopLevelDecls(Stringify(CODE), Decls);
#undef Stringifyx
#undef Stringify
#undef CODE

  auto *c = new C();
  EXPECT_EQ(Cpp::GetBaseClassOffset(Decls[2], Decls[0]), (char *)(A*)c - (char *)c);
  EXPECT_EQ(Cpp::GetBaseClassOffset(Decls[2], Decls[1]), (char *)(B*)c - (char *)c);

  auto *d = new D();
  EXPECT_EQ(Cpp::GetBaseClassOffset(Decls[3], Decls[0]), (char *)(A*)d - (char *)d);
  EXPECT_EQ(Cpp::GetBaseClassOffset(Decls[3], Decls[1]), (char *)(B*)d - (char *)d);
  EXPECT_EQ(Cpp::GetBaseClassOffset(Decls[3], Decls[2]), (char *)(C*)d - (char *)d);

  auto *e = new E();
  EXPECT_EQ(Cpp::GetBaseClassOffset(Decls[4], Decls[0]), (char *)(A*)e - (char *)e);
  EXPECT_EQ(Cpp::GetBaseClassOffset(Decls[4], Decls[1]), (char *)(B*)e - (char *)e);

  auto *g = new G();
  EXPECT_EQ(Cpp::GetBaseClassOffset(Decls[6], Decls[0]), (char *)(A*)g - (char *)g);
}

TEST(ScopeReflectionTest, GetAllCppNames) {
  std::vector<Decl *> Decls;
  std::string code = R"(
    class A { int a; };
    class B { int b; };
    class C : public A, public B { int c; };
    class D : public A, public B, public C { int d; };
    namespace N {
      class A { int a; };
      class B { int b; };
      class C : public A, public B { int c; };
      class D : public A, public B, public C { int d; };
    }
    void myfunc() {
      int a;
    }
  )";

  GetAllTopLevelDecls(code, Decls);

  auto test_get_all_cpp_names =
      [](Decl* D, const std::vector<std::string>& truth_names) {
        std::set<std::string> names;
        Cpp::GetAllCppNames(D, names);
        EXPECT_EQ(names.size(), truth_names.size());

        for (unsigned i = 0; i < truth_names.size() && i < names.size(); i++) {
          EXPECT_TRUE(names.find(truth_names[i]) != names.end());
        }
      };

  test_get_all_cpp_names(Decls[0], {"a"});
  test_get_all_cpp_names(Decls[1], {"b"});
  test_get_all_cpp_names(Decls[2], {"c"});
  test_get_all_cpp_names(Decls[3], {"d"});
  test_get_all_cpp_names(Decls[4], {"A", "B", "C", "D"});
  test_get_all_cpp_names(Decls[5], {});
}

TEST(ScopeReflectionTest, InstantiateNNTPClassTemplate) {
  std::vector<Decl *> Decls;
  std::string code = R"(
    template <int N>
    struct Factorial {
      enum { value = N * Factorial<N - 1>::value };
    };

    template <>
    struct Factorial<0> {
      enum { value = 1 };
    };)";

  GetAllTopLevelDecls(code, Decls);

  ASTContext &C = Interp->getCI()->getASTContext();
  Cpp::TCppType_t IntTy = C.IntTy.getAsOpaquePtr();
  std::vector<Cpp::TemplateArgInfo> args1 = {{IntTy, "5"}};
  EXPECT_TRUE(Cpp::InstantiateTemplate(Decls[0], args1.data(),
                                       /*type_size*/ args1.size()));

  // C API
  auto I = clang_createInterpreterFromRawPtr(Cpp::GetInterpreter());
  CXTemplateArgInfo Args1[] = {{IntTy, "5"}};
  auto C_API_SHIM = [&](auto Decl) {
    return clang_instantiateTemplate(make_scope(Decl, I), Args1, 1).data[0];
  };
  EXPECT_NE(C_API_SHIM(Decls[0]), nullptr);
  // Clean up resources
  clang_Interpreter_takeInterpreterAsPtr(I);
  clang_Interpreter_dispose(I);
}

TEST(ScopeReflectionTest, InstantiateVarTemplate) {
  std::vector<Decl*> Decls;
  std::string code = R"(
template<class T> constexpr T pi = T(3.1415926535897932385L);
)";

  GetAllTopLevelDecls(code, Decls);
  ASTContext& C = Interp->getCI()->getASTContext();

  std::vector<Cpp::TemplateArgInfo> args1 = {C.IntTy.getAsOpaquePtr()};
  auto Instance1 = Cpp::InstantiateTemplate(Decls[0], args1.data(),
                                            /*type_size*/ args1.size());
  EXPECT_TRUE(isa<VarDecl>((Decl*)Instance1));
  auto* VD = cast<VarTemplateSpecializationDecl>((Decl*)Instance1);
  VarTemplateDecl* VDTD1 = VD->getSpecializedTemplate();
  EXPECT_TRUE(VDTD1->isThisDeclarationADefinition());
#if CLANG_VERSION_MAJOR > 13
#if CLANG_VERSION_MAJOR <= 18
  TemplateArgument TA1 = (*VD->getTemplateArgsInfo())[0].getArgument();
#else
  TemplateArgument TA1 = (*VD->getTemplateArgsAsWritten())[0].getArgument();
#endif // CLANG_VERSION_MAJOR
#else
  TemplateArgument TA1 = VD->getTemplateArgsInfo()[0].getArgument();
#endif // CLANG_VERSION_MAJOR
  EXPECT_TRUE(TA1.getAsType()->isIntegerType());
}

TEST(ScopeReflectionTest, InstantiateFunctionTemplate) {
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

TEST(ScopeReflectionTest, InstantiateTemplateFunctionFromString) {
  if (llvm::sys::RunningOnValgrind())
    GTEST_SKIP() << "XFAIL due to Valgrind report";
  Cpp::CreateInterpreter();
  std::string code = R"(#include <memory>)";
  Interp->process(code);
  const char* str = "std::make_unique<int,int>";
  auto* Instance1 = (Decl*)Cpp::InstantiateTemplateFunctionFromString(str);
  EXPECT_TRUE(Instance1);
}

TEST(ScopeReflectionTest, InstantiateTemplate) {
  std::vector<Decl *> Decls;
  std::string code = R"(
    template<typename T>
    class AllDefault {
    public:
      AllDefault(int val) : m_t(val) {}
      template<int aap=1, int noot=2>
      int do_stuff() { return m_t+aap+noot; }

    public:
      T m_t;
    };

    template<typename T = int>
    class C0 {
    public:
      C0(T val) : m_t(val) {}
      template<int aap=1, int noot=2>
      T do_stuff() { return m_t+aap+noot; }

    public:
      T m_t;
    };

    template<typename T, typename R = C0<T>>
    class C1 {
    public:
      C1(const R & val = R()) : m_t(val.m_t) {}
      template<int aap=1, int noot=2>
      T do_stuff() { return m_t+aap+noot; }

    public:
      T m_t;
    };

    template<typename T, unsigned N>
    class C2{};
  )";

  GetAllTopLevelDecls(code, Decls);
  ASTContext &C = Interp->getCI()->getASTContext();

  std::vector<Cpp::TemplateArgInfo> args1 = {C.IntTy.getAsOpaquePtr()};
  auto Instance1 = Cpp::InstantiateTemplate(Decls[0], args1.data(),
                                            /*type_size*/ args1.size());
  EXPECT_TRUE(isa<ClassTemplateSpecializationDecl>((Decl*)Instance1));
  auto *CTSD1 = static_cast<ClassTemplateSpecializationDecl*>(Instance1);
  EXPECT_TRUE(CTSD1->hasDefinition());
  TemplateArgument TA1 = CTSD1->getTemplateArgs().get(0);
  EXPECT_TRUE(TA1.getAsType()->isIntegerType());
  EXPECT_TRUE(CTSD1->hasDefinition());

  auto Instance2 = Cpp::InstantiateTemplate(Decls[1], nullptr,
                                            /*type_size*/ 0);
  EXPECT_TRUE(isa<ClassTemplateSpecializationDecl>((Decl*)Instance2));
  auto *CTSD2 = static_cast<ClassTemplateSpecializationDecl*>(Instance2);
  EXPECT_TRUE(CTSD2->hasDefinition());
  TemplateArgument TA2 = CTSD2->getTemplateArgs().get(0);
  EXPECT_TRUE(TA2.getAsType()->isIntegerType());

  std::vector<Cpp::TemplateArgInfo> args3 = {C.IntTy.getAsOpaquePtr()};
  auto Instance3 = Cpp::InstantiateTemplate(Decls[2], args3.data(),
                                            /*type_size*/ args3.size());
  EXPECT_TRUE(isa<ClassTemplateSpecializationDecl>((Decl*)Instance3));
  auto *CTSD3 = static_cast<ClassTemplateSpecializationDecl*>(Instance3);
  EXPECT_TRUE(CTSD3->hasDefinition());
  TemplateArgument TA3_0 = CTSD3->getTemplateArgs().get(0);
  TemplateArgument TA3_1 = CTSD3->getTemplateArgs().get(1);
  EXPECT_TRUE(TA3_0.getAsType()->isIntegerType());
  EXPECT_TRUE(Cpp::IsRecordType(TA3_1.getAsType().getAsOpaquePtr()));

  std::vector<Cpp::TCppFunction_t> Inst3_methods;
  Cpp::GetClassMethods(Instance3, Inst3_methods);
  EXPECT_EQ(Cpp::GetFunctionSignature(Inst3_methods[0]),
            "C1<int>::C1(const C0<int> &val)");

  std::vector<Cpp::TemplateArgInfo> args4 = {C.IntTy.getAsOpaquePtr(),
                                               {C.IntTy.getAsOpaquePtr(), "3"}};
  auto Instance4 = Cpp::InstantiateTemplate(Decls[3], args4.data(),
                                            /*type_size*/ args4.size());

  EXPECT_TRUE(isa<ClassTemplateSpecializationDecl>((Decl*)Instance4));
  auto *CTSD4 = static_cast<ClassTemplateSpecializationDecl*>(Instance4);
  EXPECT_TRUE(CTSD4->hasDefinition());
  TemplateArgument TA4_0 = CTSD4->getTemplateArgs().get(0);
  TemplateArgument TA4_1 = CTSD4->getTemplateArgs().get(1);
  EXPECT_TRUE(TA4_0.getAsType()->isIntegerType());
  EXPECT_TRUE(TA4_1.getAsIntegral() == 3);
}

TEST(ScopeReflectionTest, GetClassTemplateInstantiationArgs) {
  std::vector<Decl *> Decls;
  std::string code = R"(
    template<typename ...T> struct __Cppyy_AppendTypesSlow {};
    __Cppyy_AppendTypesSlow<int, float, double> v1;
    __Cppyy_AppendTypesSlow<int> v2;
    __Cppyy_AppendTypesSlow<> v3;
  )";

  GetAllTopLevelDecls(code, Decls);

  auto *v1 = Cpp::GetNamed("v1");
  auto *v2 = Cpp::GetNamed("v2");
  auto *v3 = Cpp::GetNamed("v3");
  EXPECT_TRUE(v1 && v2 && v3);

  auto *v1_class = Cpp::GetScopeFromType(Cpp::GetVariableType(v1));
  auto *v2_class = Cpp::GetScopeFromType(Cpp::GetVariableType(v2));
  auto *v3_class = Cpp::GetScopeFromType(Cpp::GetVariableType(v3));
  EXPECT_TRUE(v1_class && v2_class && v3_class);

  std::vector<Cpp::TemplateArgInfo> instance_types;

  Cpp::GetClassTemplateInstantiationArgs(v1_class, instance_types);
  EXPECT_TRUE(instance_types.size() == 3);

  instance_types.clear();

  Cpp::GetClassTemplateInstantiationArgs(v2_class, instance_types);
  EXPECT_TRUE(instance_types.size() == 1);

  instance_types.clear();

  Cpp::GetClassTemplateInstantiationArgs(v3_class, instance_types);
  EXPECT_TRUE(instance_types.size() == 0);
}


TEST(ScopeReflectionTest, IncludeVector) {
  if (llvm::sys::RunningOnValgrind())
      GTEST_SKIP() << "XFAIL due to Valgrind report";
  std::string code = R"(
    #include <vector>
    #include <iostream>
  )";
  Interp->declare(code);
}

TEST(ScopeReflectionTest, GetOperator) {
  if (llvm::sys::RunningOnValgrind())
    GTEST_SKIP() << "XFAIL due to Valgrind report";

  Cpp::CreateInterpreter();

  std::string code = R"(
    class MyClass {
    public:
        int x;
        MyClass(int x) : x(x) {}
    };

    MyClass operator-(MyClass lhs, MyClass rhs) {
        return MyClass(lhs.x - rhs.x);
    }

    MyClass operator+(MyClass lhs, MyClass rhs) {
        return MyClass(lhs.x + rhs.x);
    }

    namespace extra_ops {

    MyClass operator+(MyClass lhs, int rhs) {
        return MyClass(lhs.x + rhs);
    }

    MyClass operator+(int lhs, MyClass rhs) {
        return MyClass(lhs + rhs.x);
    }

    MyClass operator~(MyClass self) {
        return MyClass(-self.x);
    }

    }
  )";

  Cpp::Declare(code.c_str());

  std::vector<Cpp::TCppFunction_t> ops;

  Cpp::GetOperator(Cpp::GetGlobalScope(), Cpp::Operator::OP_Plus, ops);
  EXPECT_EQ(ops.size(), 1);
  ops.clear();

  Cpp::GetOperator(Cpp::GetGlobalScope(), Cpp::Operator::OP_Minus, ops);
  EXPECT_EQ(ops.size(), 1);
  ops.clear();

  Cpp::GetOperator(Cpp::GetGlobalScope(), Cpp::Operator::OP_Star, ops);
  EXPECT_EQ(ops.size(), 0);
  ops.clear();

  // operators defined within a namespace
  Cpp::GetOperator(Cpp::GetScope("extra_ops"), Cpp::Operator::OP_Plus, ops);
  EXPECT_EQ(ops.size(), 2);
  ops.clear();

  // unary operator
  Cpp::GetOperator(Cpp::GetScope("extra_ops"), Cpp::Operator::OP_Tilde, ops);
  EXPECT_EQ(ops.size(), 1);
  ops.clear();

  Cpp::GetOperator(Cpp::GetScope("extra_ops"), Cpp::Operator::OP_Tilde, ops,
                   Cpp::OperatorArity::kUnary);
  EXPECT_EQ(ops.size(), 1);
  ops.clear();

  Cpp::GetOperator(Cpp::GetScope("extra_ops"), Cpp::Operator::OP_Tilde, ops,
                   Cpp::OperatorArity::kBinary);
  EXPECT_EQ(ops.size(), 0);
}
