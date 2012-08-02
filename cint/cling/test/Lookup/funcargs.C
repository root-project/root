// RUN: cat %s | %cling 2>&1 | FileCheck %s
// Test Interpreter::lookupFunctionArgs()
#include "cling/Interpreter/Interpreter.h"
#include "clang/AST/Decl.h"

using namespace std;


//
//  We need to fetch the global scope declaration,
//  otherwise known as the translation unit decl.
//
const clang::Decl* G = gCling->lookupScope("");
G
//CHECK: (const clang::Decl *) {{[1-9][0-9]*$}}



//
//  Test finding a global function taking no args.
//

.rawInput 1
void f() { int x = 1; }
.rawInput 0

const clang::FunctionDecl* F = gCling->lookupFunctionArgs(G, "f", "");
F
//CHECK-NEXT: (const clang::FunctionDecl *) {{-?[1-9][0-9]*$}}
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

const clang::FunctionDecl* A = gCling->lookupFunctionArgs(G, "a", "0");
//CHECK: 0
A
//CHECK-NEXT: (const clang::FunctionDecl *) {{-?[1-9][0-9]*$}}
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

const clang::FunctionDecl* B = gCling->lookupFunctionArgs(G, "b", "0,0.0");
//CHECK: 0
//CHECK-NEXT: 0,0
B
//CHECK-NEXT: (const clang::FunctionDecl *) {{-?[1-9][0-9]*$}}
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

const clang::FunctionDecl* C1 = gCling->lookupFunctionArgs(G, "c", "0,0");
//CHECK: 0
//CHECK-NEXT: 0,0
C1
//CHECK-NEXT: (const clang::FunctionDecl *) {{-?[1-9][0-9]*$}}
C1->print(llvm::outs());
//CHECK-NEXT: void c(int vi, int vj) {
//CHECK-NEXT:     int x = vi;
//CHECK-NEXT:     int y = vj;
//CHECK-NEXT: }

const clang::FunctionDecl* C2 = gCling->lookupFunctionArgs(G, "c", "0,0.0");
//CHECK: 0
//CHECK-NEXT: 0,0
C2
//CHECK-NEXT: (const clang::FunctionDecl *) {{-?[1-9][0-9]*$}}
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
template void d(int);
template void d(double);
.rawInput 0

const clang::FunctionDecl* D1 = gCling->lookupFunctionArgs(G, "d<int>", "0");
//CHECK: 0
D1
//CHECK-NEXT: (const clang::FunctionDecl *) {{-?[1-9][0-9]*$}}
D1->print(llvm::outs());
//CHECK-NEXT: void d(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }

const clang::FunctionDecl* D2 = gCling->lookupFunctionArgs(G, "d<double>", "0.0");
//CHECK: 0
D2
//CHECK-NEXT: (const clang::FunctionDecl *) {{-?[1-9][0-9]*$}}
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
class_A
//CHECK: (const clang::Decl *) {{[1-9][0-9]*$}}
const clang::FunctionDecl* class_A_F = gCling->lookupFunctionArgs(class_A, "A_f", "");
class_A_F
//CHECK-NEXT: (const clang::FunctionDecl *) {{-?[1-9][0-9]*$}}
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
class_B
//CHECK: (const clang::Decl *) {{[1-9][0-9]*$}}
const clang::FunctionDecl* class_B_F = gCling->lookupFunctionArgs(class_B, "B_f", "0");
//CHECK: 0
class_B_F
//CHECK-NEXT: (const clang::FunctionDecl *) {{-?[1-9][0-9]*$}}
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
class_D
//CHECK: (const clang::Decl *) {{[1-9][0-9]*$}}
const clang::FunctionDecl* class_D_F = gCling->lookupFunctionArgs(class_D, "C_f", "");
class_D_F
//CHECK-NEXT: (const clang::FunctionDecl *) {{-?[1-9][0-9]*$}}
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
class_F
//CHECK: (const clang::Decl *) {{[1-9][0-9]*$}}
const clang::FunctionDecl* class_F_F = gCling->lookupFunctionArgs(class_F, "E_f", "0");
//CHECK: 0
class_F_F
//CHECK-NEXT: (const clang::FunctionDecl *) {{-?[1-9][0-9]*$}}
class_F_F->print(llvm::outs());
//CHECK-NEXT: void E_f(int v) {
//CHECK-NEXT:     int x = v;
//CHECK-NEXT: }



//
//  One final check to make sure we are at the right line in the output.
//

"abc"
//CHECK: (const char [4]) @0x{{[0-9a-f]+}}

