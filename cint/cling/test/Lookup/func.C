// RUN: cat %s | %cling 2>&1 | FileCheck %s
// Test Interpreter::lookupFunctionArgs()
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



//
//  Test finding a global function taking no args.
//

const clang::FunctionDecl* F_args = gCling->lookupFunctionArgs(G, "f", "");
const clang::FunctionDecl* F_proto = gCling->lookupFunctionProto(G, "f", "");

printf("F_args: 0x%lx\n", (unsigned long) F_args);
//CHECK-NEXT: F_args: 0x{{[1-9a-f][0-9a-f]*$}}
F_args->print(llvm::outs());
//CHECK-NEXT: void f() {
//CHECK-NEXT:     int x = 1;
//CHECK-NEXT: }

printf("F_proto: 0x%lx\n", (unsigned long) F_proto);
//CHECK: F_proto: 0x{{[1-9a-f][0-9a-f]*$}}
F_proto->print(llvm::outs());
//CHECK-NEXT: void f() {
//CHECK-NEXT:     int x = 1;
//CHECK-NEXT: }



//
//  Test finding a global function taking a single int argument.
//

const clang::FunctionDecl* A_args = gCling->lookupFunctionArgs(G, "a", "0");
const clang::FunctionDecl* A_proto = gCling->lookupFunctionProto(G, "a", "int");

printf("A_args: 0x%lx\n", (unsigned long) A_args);
//CHECK: A_args: 0x{{[1-9a-f][0-9a-f]*$}}
A_args->print(llvm::outs());
//CHECK-NEXT: void a(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }

printf("A_proto: 0x%lx\n", (unsigned long) A_proto);
//CHECK: A_proto: 0x{{[1-9a-f][0-9a-f]*$}}
A_proto->print(llvm::outs());
//CHECK-NEXT: void a(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }



//
//  Test finding a global function taking an int and a double argument.
//

const clang::FunctionDecl* B_args = gCling->lookupFunctionArgs(G, "b", "0,0.0");
const clang::FunctionDecl* B_proto = gCling->lookupFunctionProto(G, "b", "int,double");

printf("B_args: 0x%lx\n", (unsigned long) B_args);
//CHECK: B_args: 0x{{[1-9a-f][0-9a-f]*$}}
B_args->print(llvm::outs());
//CHECK-NEXT: void b(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }

printf("B_proto: 0x%lx\n", (unsigned long) B_proto);
//CHECK: B_proto: 0x{{[1-9a-f][0-9a-f]*$}}
B_proto->print(llvm::outs());
//CHECK-NEXT: void b(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }



//
//  Test finding a global overloaded function.
//

const clang::FunctionDecl* C1_args = gCling->lookupFunctionArgs(G, "c", "0,0");
const clang::FunctionDecl* C1_proto = gCling->lookupFunctionProto(G, "c", "int,int");

printf("C1_args: 0x%lx\n", (unsigned long) C1_args);
//CHECK: C1_args: 0x{{[1-9a-f][0-9a-f]*$}}
C1_args->print(llvm::outs());
//CHECK-NEXT: void c(int vi, int vj) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     int y = vj;
//CHECK-NEXT: }

printf("C1_proto: 0x%lx\n", (unsigned long) C1_proto);
//CHECK: C1_proto: 0x{{[1-9a-f][0-9a-f]*$}}
C1_proto->print(llvm::outs());
//CHECK-NEXT: void c(int vi, int vj) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     int y = vj;
//CHECK-NEXT: }

const clang::FunctionDecl* C2_args = gCling->lookupFunctionArgs(G, "c", "0,0.0");
const clang::FunctionDecl* C2_proto = gCling->lookupFunctionProto(G, "c", "int,double");

printf("C2_args: 0x%lx\n", (unsigned long) C2_args);
//CHECK: C2_args: 0x{{[1-9a-f][0-9a-f]*$}}
C2_args->print(llvm::outs());
//CHECK-NEXT: void c(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }

printf("C2_proto: 0x%lx\n", (unsigned long) C2_proto);
//CHECK: C2_proto: 0x{{[1-9a-f][0-9a-f]*$}}
C2_proto->print(llvm::outs());
//CHECK-NEXT: void c(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }



//
//  Test finding simple global template instantiations.
//

const clang::FunctionDecl* D1_args = gCling->lookupFunctionArgs(G, "d<int>", "0");
const clang::FunctionDecl* D1_proto = gCling->lookupFunctionProto(G, "d<int>", "int");

printf("D1_args: 0x%lx\n", (unsigned long) D1_args);
//CHECK: D1_args: 0x{{[1-9a-f][0-9a-f]*$}}
D1_args->print(llvm::outs());
//CHECK-NEXT: void d(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }

printf("D1_proto: 0x%lx\n", (unsigned long) D1_proto);
//CHECK: D1_proto: 0x{{[1-9a-f][0-9a-f]*$}}
D1_proto->print(llvm::outs());
//CHECK-NEXT: void d(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }

const clang::FunctionDecl* D2_args = gCling->lookupFunctionArgs(G, "d<double>", "0.0");
const clang::FunctionDecl* D2_proto = gCling->lookupFunctionProto(G, "d<double>", "double");

printf("D2_args: 0x%lx\n", (unsigned long) D2_args);
//CHECK: D2_args: 0x{{[1-9a-f][0-9a-f]*$}}
D2_args->print(llvm::outs());
//CHECK-NEXT: void d(double v) {
//CHECK-NEXT:     double x = v;
//CHECK-NEXT: }

printf("D2_proto: 0x%lx\n", (unsigned long) D2_proto);
//CHECK: D2_proto: 0x{{[1-9a-f][0-9a-f]*$}}
D2_proto->print(llvm::outs());
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

const clang::FunctionDecl* func_A_f_args = gCling->lookupFunctionArgs(class_A, "A_f", "");
const clang::FunctionDecl* func_A_f_proto = gCling->lookupFunctionProto(class_A, "A_f", "");

printf("func_A_f_args: 0x%lx\n", (unsigned long) func_A_f_args);
//CHECK-NEXT: func_A_f_args: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_f_args->print(llvm::outs());
//CHECK-NEXT: void A_f() {
//CHECK-NEXT:     int x = 1;
//CHECK-NEXT: }

printf("func_A_f_proto: 0x%lx\n", (unsigned long) func_A_f_proto);
//CHECK: func_A_f_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_f_proto->print(llvm::outs());
//CHECK-NEXT: void A_f() {
//CHECK-NEXT:     int x = 1;
//CHECK-NEXT: }



//
//  Test finding a member function taking an int arg.
//

const clang::FunctionDecl* func_A_g_args = gCling->lookupFunctionArgs(class_A, "A_g", "0");
const clang::FunctionDecl* func_A_g_proto = gCling->lookupFunctionProto(class_A, "A_g", "int");

printf("func_A_g_args: 0x%lx\n", (unsigned long) func_A_g_args);
//CHECK: func_A_g_args: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_g_args->print(llvm::outs());
//CHECK-NEXT: void A_g(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }

printf("func_A_g_proto: 0x%lx\n", (unsigned long) func_A_g_proto);
//CHECK: func_A_g_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_g_proto->print(llvm::outs());
//CHECK-NEXT: void A_g(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }



//
//  Test finding a member function taking an int and a double argument.
//

const clang::FunctionDecl* func_A_h_args = gCling->lookupFunctionArgs(class_A, "A_h", "0,0.0");
const clang::FunctionDecl* func_A_h_proto = gCling->lookupFunctionProto(class_A, "A_h", "int,double");

printf("func_A_h_args: 0x%lx\n", (unsigned long) func_A_h_args);
//CHECK: func_A_h_args: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_h_args->print(llvm::outs());
//CHECK-NEXT: void A_h(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }

printf("func_A_h_proto: 0x%lx\n", (unsigned long) func_A_h_proto);
//CHECK: func_A_h_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_h_proto->print(llvm::outs());
//CHECK-NEXT: void A_h(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }



//
//  Test finding an overloaded member function.
//

const clang::FunctionDecl* func_A_j1_args = gCling->lookupFunctionArgs(class_A, "A_j", "0,0");
const clang::FunctionDecl* func_A_j1_proto = gCling->lookupFunctionProto(class_A, "A_j", "int,int");

printf("func_A_j1_args: 0x%lx\n", (unsigned long) func_A_j1_args);
//CHECK: func_A_j1_args: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_j1_args->print(llvm::outs());
//CHECK-NEXT: void A_j(int vi, int vj) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     int y = vj;
//CHECK-NEXT: }

printf("func_A_j1_proto: 0x%lx\n", (unsigned long) func_A_j1_proto);
//CHECK: func_A_j1_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_j1_proto->print(llvm::outs());
//CHECK-NEXT: void A_j(int vi, int vj) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     int y = vj;
//CHECK-NEXT: }

const clang::FunctionDecl* func_A_j2_args = gCling->lookupFunctionArgs(class_A, "A_j", "0,0.0");
const clang::FunctionDecl* func_A_j2_proto = gCling->lookupFunctionProto(class_A, "A_j", "int,double");

printf("func_A_j2_args: 0x%lx\n", (unsigned long) func_A_j2_args);
//CHECK: func_A_j2_args: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_j2_args->print(llvm::outs());
//CHECK-NEXT: void A_j(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }

printf("func_A_j2_proto: 0x%lx\n", (unsigned long) func_A_j2_proto);
//CHECK: func_A_j2_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_j2_proto->print(llvm::outs());
//CHECK-NEXT: void A_j(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }



//
//  Test finding simple member function template instantiations.
//

const clang::FunctionDecl* func_A_k1_args = gCling->lookupFunctionArgs(class_A, "A_k<int>", "0");
const clang::FunctionDecl* func_A_k1_proto = gCling->lookupFunctionProto(class_A, "A_k<int>", "int");

printf("func_A_k1_args: 0x%lx\n", (unsigned long) func_A_k1_args);
//CHECK: func_A_k1_args: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_k1_args->print(llvm::outs());
//CHECK-NEXT: void A_k(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }

printf("func_A_k1_proto: 0x%lx\n", (unsigned long) func_A_k1_proto);
//CHECK: func_A_k1_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_k1_proto->print(llvm::outs());
//CHECK-NEXT: void A_k(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }

const clang::FunctionDecl* func_A_k2_args = gCling->lookupFunctionArgs(class_A, "A_k<double>", "0.0");
const clang::FunctionDecl* func_A_k2_proto = gCling->lookupFunctionProto(class_A, "A_k<double>", "double");

printf("func_A_k2_args: 0x%lx\n", (unsigned long) func_A_k2_args);
//CHECK: func_A_k2_args: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_k2_args->print(llvm::outs());
//CHECK-NEXT: void A_k(double v) {
//CHECK-NEXT:     double x = v;
//CHECK-NEXT: }

printf("func_A_k2_proto: 0x%lx\n", (unsigned long) func_A_k2_proto);
//CHECK: func_A_k2_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_A_k2_proto->print(llvm::outs());
//CHECK-NEXT: void A_k(double v) {
//CHECK-NEXT:     double x = v;
//CHECK-NEXT: }



//
//  Test finding a member function taking no args in a base class.
//

const clang::FunctionDecl* func_B_F_args = gCling->lookupFunctionArgs(class_A, "B_f", "");
const clang::FunctionDecl* func_B_F_proto = gCling->lookupFunctionProto(class_A, "B_f", "");

printf("func_B_F_args: 0x%lx\n", (unsigned long) func_B_F_args);
//CHECK: func_B_F_args: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_F_args->print(llvm::outs());
//CHECK-NEXT: void B_f() {
//CHECK-NEXT:     int x = 1;
//CHECK-NEXT: }

printf("func_B_F_proto: 0x%lx\n", (unsigned long) func_B_F_proto);
//CHECK: func_B_F_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_F_proto->print(llvm::outs());
//CHECK-NEXT: void B_f() {
//CHECK-NEXT:     int x = 1;
//CHECK-NEXT: }



//
//  Test finding a member function taking an int arg in a base class.
//

const clang::FunctionDecl* func_B_G_args = gCling->lookupFunctionArgs(class_A, "B_g", "0");
const clang::FunctionDecl* func_B_G_proto = gCling->lookupFunctionProto(class_A, "B_g", "int");

printf("func_B_G_args: 0x%lx\n", (unsigned long) func_B_G_args);
//CHECK: func_B_G_args: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_G_args->print(llvm::outs());
//CHECK-NEXT: void B_g(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }

printf("func_B_G_proto: 0x%lx\n", (unsigned long) func_B_G_proto);
//CHECK: func_B_G_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_G_proto->print(llvm::outs());
//CHECK-NEXT: void B_g(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }



//
//  Test finding a member function taking an int and a double argument
//  in a base class.
//

const clang::FunctionDecl* func_B_h_args = gCling->lookupFunctionArgs(class_A, "B_h", "0,0.0");
const clang::FunctionDecl* func_B_h_proto = gCling->lookupFunctionProto(class_A, "B_h", "int,double");

printf("func_B_h_args: 0x%lx\n", (unsigned long) func_B_h_args);
//CHECK: func_B_h_args: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_h_args->print(llvm::outs());
//CHECK-NEXT: void B_h(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }

printf("func_B_h_proto: 0x%lx\n", (unsigned long) func_B_h_proto);
//CHECK: func_B_h_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_h_proto->print(llvm::outs());
//CHECK-NEXT: void B_h(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }



//
//  Test finding an overloaded member function in a base class.
//

const clang::FunctionDecl* func_B_j1_args = gCling->lookupFunctionArgs(class_A, "B_j", "0,0");
const clang::FunctionDecl* func_B_j1_proto = gCling->lookupFunctionProto(class_A, "B_j", "int,int");

printf("func_B_j1_args: 0x%lx\n", (unsigned long) func_B_j1_args);
//CHECK: func_B_j1_args: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_j1_args->print(llvm::outs());
//CHECK-NEXT: void B_j(int vi, int vj) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     int y = vj;
//CHECK-NEXT: }

printf("func_B_j1_proto: 0x%lx\n", (unsigned long) func_B_j1_proto);
//CHECK: func_B_j1_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_j1_proto->print(llvm::outs());
//CHECK-NEXT: void B_j(int vi, int vj) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     int y = vj;
//CHECK-NEXT: }

const clang::FunctionDecl* func_B_j2_args = gCling->lookupFunctionArgs(class_A, "B_j", "0,0.0");
const clang::FunctionDecl* func_B_j2_proto = gCling->lookupFunctionProto(class_A, "B_j", "int,double");

printf("func_B_j2_args: 0x%lx\n", (unsigned long) func_B_j2_args);
//CHECK: func_B_j2_args: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_j2_args->print(llvm::outs());
//CHECK-NEXT: void B_j(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }

printf("func_B_j2_proto: 0x%lx\n", (unsigned long) func_B_j2_proto);
//CHECK: func_B_j2_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_j2_proto->print(llvm::outs());
//CHECK-NEXT: void B_j(int vi, double vd) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     double y = vd;
//CHECK-NEXT: }



//
//  Test finding simple member function template instantiations in a base class.
//

const clang::FunctionDecl* func_B_k1_args = gCling->lookupFunctionArgs(class_A, "B_k<int>", "0");
const clang::FunctionDecl* func_B_k1_proto = gCling->lookupFunctionProto(class_A, "B_k<int>", "int");

printf("func_B_k1_args: 0x%lx\n", (unsigned long) func_B_k1_args);
//CHECK: func_B_k1_args: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_k1_args->print(llvm::outs());
//CHECK-NEXT: void B_k(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }

printf("func_B_k1_proto: 0x%lx\n", (unsigned long) func_B_k1_proto);
//CHECK: func_B_k1_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_k1_proto->print(llvm::outs());
//CHECK-NEXT: void B_k(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }

const clang::FunctionDecl* func_B_k2_args = gCling->lookupFunctionArgs(class_A, "B_k<double>", "0.0");
const clang::FunctionDecl* func_B_k2_proto = gCling->lookupFunctionProto(class_A, "B_k<double>", "double");

printf("func_B_k2_args: 0x%lx\n", (unsigned long) func_B_k2_args);
//CHECK: func_B_k2_args: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_k2_args->print(llvm::outs());
//CHECK-NEXT: void B_k(double v) {
//CHECK-NEXT:     double x = v;
//CHECK-NEXT: }

printf("func_B_k2_proto: 0x%lx\n", (unsigned long) func_B_k2_proto);
//CHECK: func_B_k2_proto: 0x{{[1-9a-f][0-9a-f]*$}}
func_B_k2_proto->print(llvm::outs());
//CHECK-NEXT: void B_k(double v) {
//CHECK-NEXT:     double x = v;
//CHECK-NEXT: }



//
//  One final check to make sure we are at the right line in the output.
//

"abc"
//CHECK: (const char [4]) @0x{{[0-9a-f]+}}

