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
void a(int v) { int x = v; }
void b(int vi, double vd) { int x = vi; double y = vd; }
void c(int vi, int vj) { int x = vi; int y = vj; }
void c(int vi, double vd) { int x = vi; double y = vd; }
template <class T> void d(T v) { T x = v; }
// Note: In CINT, looking up a class template specialization causes
//       instantiation, but looking up a function template specialization
//       does not, so we explicitly request the instantiations we are
//       going to lookup so they will be there to find.
template void d(int);
template void d(double);
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



.rawInput 1
class B {
   void B_f() { int x = 1; }
   void B_g(int v) { int x = v; }
   void B_h(int vi, double vd) { int x = vi; double y = vd; }
   void B_j(int vi, int vj) { int x = vi; int y = vj; }
   void B_j(int vi, double vd) { int x = vi; double y = vd; }
   template <class T> void B_k(T v) { T x = v; }
};
class A : public B {
   void A_f() { int x = 1; }
   void A_g(int v) { int x = v; }
   void A_h(int vi, double vd) { int x = vi; double y = vd; }
   void A_j(int vi, int vj) { int x = vi; int y = vj; }
   void A_j(int vi, double vd) { int x = vi; double y = vd; }
   template <class T> void A_k(T v) { T x = v; }
};
// Note: In CINT, looking up a class template specialization causes
//       instantiation, but looking up a function template specialization
//       does not, so we explicitly request the instantiations we are
//       going to lookup so they will be there to find.
template void A::A_k(int);
template void A::A_k(double);
template void A::B_k(int);
template void A::B_k(double);
.rawInput 0

const clang::Decl* class_A = gCling->lookupScope("A");
printf("class_A: 0x%lx\n", (unsigned long) class_A);
//CHECK: class_A: 0x{{[1-9a-f][0-9a-f]*$}}

const clang::Decl* class_B = gCling->lookupScope("B");
printf("class_B: 0x%lx\n", (unsigned long) class_B);
//CHECK-NEXT: class_B: 0x{{[1-9a-f][0-9a-f]*$}}



//
//  Test finding a member function taking no args.
//

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

const clang::FunctionDecl* func_A_g = gCling->lookupFunctionProto(class_A, "A_g", "int");
printf("func_A_g: 0x%lx\n", (unsigned long) func_A_g);
//CHECK: func_A_g: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_g->print(llvm::outs());
//CHECK-NEXT: void A_g(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }



//
//  Test finding a member function taking an int and a double argument.
//

const clang::FunctionDecl* func_A_h = gCling->lookupFunctionProto(class_A, "A_h", "int,double");
printf("func_A_h: 0x%lx\n", (unsigned long) func_A_h);
//CHECK: func_A_h: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_h->print(llvm::outs());
//CHECK-NEXT: void A_h(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }



//
//  Test finding an overloaded member function.
//

const clang::FunctionDecl* func_A_j1 = gCling->lookupFunctionProto(class_A, "A_j", "int,int");
printf("func_A_j1: 0x%lx\n", (unsigned long) func_A_j1);
//CHECK: func_A_j1: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_j1->print(llvm::outs());
//CHECK-NEXT: void A_j(int vi, int vj) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     int y = vj;
//CHECK-NEXT: }

const clang::FunctionDecl* func_A_j2 = gCling->lookupFunctionProto(class_A, "A_j", "int,double");
printf("func_A_j2: 0x%lx\n", (unsigned long) func_A_j2);
//CHECK: func_A_j2: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_j2->print(llvm::outs());
//CHECK-NEXT: void A_j(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }



//
//  Test finding simple member function template instantiations.
//

const clang::FunctionDecl* func_A_k1 = gCling->lookupFunctionProto(class_A, "A_k<int>", "int");
printf("func_A_k1: 0x%lx\n", (unsigned long) func_A_k1);
//CHECK: func_A_k1: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_k1->print(llvm::outs());
//CHECK-NEXT: void A_k(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }

const clang::FunctionDecl* func_A_k2 = gCling->lookupFunctionProto(class_A, "A_k<double>", "double");
printf("func_A_k2: 0x%lx\n", (unsigned long) func_A_k2);
//CHECK: func_A_k2: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_k2->print(llvm::outs());
//CHECK-NEXT: void A_k(double v) {
//CHECK-NEXT:     double x = v;
//CHECK-NEXT: }



//
//  Test finding a member function taking no args in a base class.
//

const clang::FunctionDecl* func_B_F = gCling->lookupFunctionProto(class_A, "B_f", "");
printf("func_B_F: 0x%lx\n", (unsigned long) func_B_F);
//CHECK: func_B_F: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_F->print(llvm::outs());
//CHECK-NEXT: void B_f() {
//CHECK-NEXT:     int x = 1;
//CHECK-NEXT: }



//
//  Test finding a member function taking an int arg in a base class.
//

const clang::FunctionDecl* func_B_G = gCling->lookupFunctionProto(class_A, "B_g", "int");
printf("func_B_G: 0x%lx\n", (unsigned long) func_B_G);
//CHECK: func_B_G: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_G->print(llvm::outs());
//CHECK-NEXT: void B_g(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }



//
//  Test finding a member function taking an int and a double argument
//  in a base class.
//

const clang::FunctionDecl* func_B_h = gCling->lookupFunctionProto(class_A, "B_h", "int,double");
printf("func_B_h: 0x%lx\n", (unsigned long) func_B_h);
//CHECK: func_B_h: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_h->print(llvm::outs());
//CHECK-NEXT: void B_h(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }



//
//  Test finding an overloaded member function in a base class.
//

const clang::FunctionDecl* func_B_j1 = gCling->lookupFunctionProto(class_A, "B_j", "int,int");
printf("func_B_j1: 0x%lx\n", (unsigned long) func_B_j1);
//CHECK: func_B_j1: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_j1->print(llvm::outs());
//CHECK-NEXT: void B_j(int vi, int vj) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     int y = vj;
//CHECK-NEXT: }

const clang::FunctionDecl* func_B_j2 = gCling->lookupFunctionProto(class_A, "B_j", "int,double");
printf("func_B_j2: 0x%lx\n", (unsigned long) func_B_j2);
//CHECK: func_B_j2: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_j2->print(llvm::outs());
//CHECK-NEXT: void B_j(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }



//
//  Test finding simple member function template instantiations in a base class.
//

const clang::FunctionDecl* func_B_k1 = gCling->lookupFunctionProto(class_A, "B_k<int>", "int");
printf("func_B_k1: 0x%lx\n", (unsigned long) func_B_k1);
//CHECK: func_B_k1: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_k1->print(llvm::outs());
//CHECK-NEXT: void B_k(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }

const clang::FunctionDecl* func_B_k2 = gCling->lookupFunctionProto(class_A, "B_k<double>", "double");
printf("func_B_k2: 0x%lx\n", (unsigned long) func_B_k2);
//CHECK: func_B_k2: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_k2->print(llvm::outs());
//CHECK-NEXT: void B_k(double v) {
//CHECK-NEXT:     double x = v;
//CHECK-NEXT: }



//
//  One final check to make sure we are at the right line in the output.
//

"abc"
//CHECK: (const char [4]) @0x{{[0-9a-f]+}}
