// RUN: cat %s | %cling 2>&1 | FileCheck %s
// Test Interpreter::lookupFunctionProto()
#include "cling/Interpreter/Interpreter.h"
#include "clang/AST/Decl.h"

#include <cstdio>
using namespace std;


//
//  We need to fetch the global scope declaration,
//  otherwise known as the translation unit decl.
//
const clang::Decl* G = gCling->lookupScope("");
printf("G: 0x%lx\n", (unsigned long) G);
//CHECK: G: 0x{{[1-9a-f][0-9a-f]*$}}



//
//  Test finding a global function taking no args.
//

.rawInput 1
void f() { int x = 1; }
.rawInput 0

const clang::FunctionDecl* F = gCling->lookupFunctionProto(G, "f", "");
printf("F: 0x%lx\n", (unsigned long) F);
//CHECK-NEXT: F: 0x{{[1-9a-f][0-9a-f]*$}}
F->print(llvm::outs());
//CHECK-NEXT: void f() {
//CHECK-NEXT:     int x = 1;
//CHECK-NEXT: }



//
//  Test finding a global function taking a single int argument.
//

.rawInput 1
void a(int v) { int x = v; }
.rawInput 0

const clang::FunctionDecl* A = gCling->lookupFunctionProto(G, "a", "int");
printf("A: 0x%lx\n", (unsigned long) A);
//CHECK: A: 0x{{[1-9a-f][0-9a-f]*$}}
A->print(llvm::outs());
//CHECK-NEXT: void a(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }



//
//  Test finding a global function taking an int and a double argument.
//

.rawInput 1
void b(int vi, double vd) { int x = vi; double y = vd; }
.rawInput 0

const clang::FunctionDecl* B = gCling->lookupFunctionProto(G, "b", "int,double");
printf("B: 0x%lx\n", (unsigned long) B);
//CHECK: B: 0x{{[1-9a-f][0-9a-f]*$}}
B->print(llvm::outs());
//CHECK-NEXT: void b(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }



//
//  Test finding a global overloaded function.
//

.rawInput 1
void c(int vi, int vj) { int x = vi; int y = vj; }
void c(int vi, double vd) { int x = vi; double y = vd; }
.rawInput 0

const clang::FunctionDecl* C1 = gCling->lookupFunctionProto(G, "c", "int,int");
printf("C1: 0x%lx\n", (unsigned long) C1);
//CHECK: C1: 0x{{[1-9a-f][0-9a-f]*$}}
C1->print(llvm::outs());
//CHECK-NEXT: void c(int vi, int vj) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     int y = vj;
//CHECK-NEXT: }

const clang::FunctionDecl* C2 = gCling->lookupFunctionProto(G, "c", "int,double");
printf("C2: 0x%lx\n", (unsigned long) C2);
//CHECK: C2: 0x{{[1-9a-f][0-9a-f]*$}}
C2->print(llvm::outs());
//CHECK-NEXT: void c(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }



//
//  Test finding simple global template instantiations.
//

.rawInput 1
template <class T> void d(T v) { T x = v; }
// Note: In CINT, looking up a class template specialization causes
//       instantiation, but looking up a function template specialization
//       does not.
template void d(int);
template void d(double);
.rawInput 0

const clang::FunctionDecl* D1 = gCling->lookupFunctionProto(G, "d<int>", "int");
printf("D1: 0x%lx\n", (unsigned long) D1);
//CHECK: D1: 0x{{[1-9a-f][0-9a-f]*$}}
D1->print(llvm::outs());
//CHECK-NEXT: void d(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }

const clang::FunctionDecl* D2 = gCling->lookupFunctionProto(G, "d<double>", "double");
printf("D2: 0x%lx\n", (unsigned long) D2);
//CHECK: D2: 0x{{[1-9a-f][0-9a-f]*$}}
D2->print(llvm::outs());
//CHECK-NEXT: void d(double v) {
//CHECK-NEXT:     double x = v;
//CHECK-NEXT: }



//
//  Test finding a member function taking no args.
//

.rawInput 1
class A {
   void A_f() { int x = 1; }
};
.rawInput 0

const clang::Decl* class_A = gCling->lookupScope("A");
printf("class_A: 0x%lx\n", (unsigned long) class_A);
//CHECK: class_A: 0x{{[1-9a-f][0-9a-f]*$}}
const clang::FunctionDecl* class_A_F = gCling->lookupFunctionProto(class_A, "A_f", "");
printf("class_A_F: 0x%lx\n", (unsigned long) class_A_F);
//CHECK-NEXT: class_A_F: 0x{{[1-9a-f][0-9a-f]*$}}
class_A_F->print(llvm::outs());
//CHECK-NEXT: void A_f() {
//CHECK-NEXT:     int x = 1;
//CHECK-NEXT: }



//
//  Test finding a member function taking an int arg.
//

.rawInput 1
class B {
   void B_f(int v) { int x = v; }
};
.rawInput 0

const clang::Decl* class_B = gCling->lookupScope("B");
printf("class_B: 0x%lx\n", (unsigned long) class_B);
//CHECK: class_B: 0x{{[1-9a-f][0-9a-f]*$}}
const clang::FunctionDecl* class_B_F = gCling->lookupFunctionProto(class_B, "B_f", "int");
printf("class_B_F: 0x%lx\n", (unsigned long) class_B_F);
//CHECK-NEXT: class_B_F: 0x{{[1-9a-f][0-9a-f]*$}}
class_B_F->print(llvm::outs());
//CHECK-NEXT: void B_f(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }



//
//  Test finding a member function taking no args in a base class.
//

.rawInput 1
class C {
   void C_f() { int x = 1; }
};
class D : public C {
};
.rawInput 0

const clang::Decl* class_D = gCling->lookupScope("D");
printf("class_D: 0x%lx\n", (unsigned long) class_D);
//CHECK: class_D: 0x{{[1-9a-f][0-9a-f]*$}}
const clang::FunctionDecl* class_D_F = gCling->lookupFunctionProto(class_D, "C_f", "");
printf("class_D_F: 0x%lx\n", (unsigned long) class_D_F);
//CHECK-NEXT: class_D_F: 0x{{[1-9a-f][0-9a-f]*$}}
class_D_F->print(llvm::outs());
//CHECK-NEXT: void C_f() {
//CHECK-NEXT:     int x = 1;
//CHECK-NEXT: }



//
//  Test finding a member function taking an int arg in a base class.
//

.rawInput 1
class E {
   void E_f(int v) { int x = v; }
};
class F : public E {
};
.rawInput 0

const clang::Decl* class_F = gCling->lookupScope("F");
printf("class_F: 0x%lx\n", (unsigned long) class_F);
//CHECK: class_F: 0x{{[1-9a-f][0-9a-f]*$}}
const clang::FunctionDecl* class_F_F = gCling->lookupFunctionProto(class_F, "E_f", "int");
printf("class_F_F: 0x%lx\n", (unsigned long) class_F_F);
//CHECK-NEXT: class_F_F: 0x{{[1-9a-f][0-9a-f]*$}}
class_F_F->print(llvm::outs());
//CHECK-NEXT: void E_f(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }



//
//  One final check to make sure we are at the right line in the output.
//

"abc"
//CHECK: (const char [4]) @0x{{[0-9a-f]+}}
