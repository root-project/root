// RUN: cat %s | %cling -Xclang -verify -I%p | FileCheck %s

// The test verifies the expected behavior in cling::utils::Transform class,
// which is supposed to provide different transformation of AST nodes and types.

#include "cling/Interpreter/Interpreter.h"
#include "cling/Utils/AST.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/SmallSet.h"
#include "clang/Frontend/CompilerInstance.h"

.rawInput 1

typedef double Double32_t;
typedef int Int_t;
typedef long Long_t;
typedef Int_t* IntPtr_t;

template <typename T> class A {};
template <typename T, typename U> class B {};
template <typename T, typename U> class C {};
typedef C<A<B<Double32_t, Int_t> >, Double32_t > CTD;
typedef C<A<B<const Double32_t, const Int_t> >, Double32_t > CTDConst;

.rawInput 0

const clang::ASTContext& Ctx = gCling->getCI()->getASTContext();
llvm::SmallSet<const clang::Type*, 4> skip;
skip.insert(gCling->lookupType("Double32_t").getTypePtr());
const clang::Type* t = 0;
clang::QualType QT;
using namespace cling::utils;

// Test desugaring pointers types:
QT = gCling->lookupType("Int_t*");
Transform::GetPartiallyDesugaredType(Ctx, QT, skip).getAsString().c_str()
// CHECK:(const char * const) "int *"

QT = gCling->lookupType("const IntPtr_t*");
Transform::GetPartiallyDesugaredType(Ctx, QT, skip).getAsString().c_str()
// CHECK:(const char * const) "int *const *"


// Test desugaring reference (both r- or l- value) types:
QT = gCling->lookupType("const IntPtr_t&");
Transform::GetPartiallyDesugaredType(Ctx, QT, skip).getAsString().c_str()
// CHECK:(const char * const) "int *const &"

//TODO: QT = gCling->lookupType("IntPtr_t[32]");


//Desugar template parameters:
gCling->lookupScope("A<B<Double32_t, Int_t*> >", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, skip).getAsString().c_str()
// CHECK:(const char * const) "A<B<Double32_t, int *> >"

gCling->lookupScope("CTD", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, skip).getAsString().c_str()
// CHECK: (const char * const) "C<A<B<Double32_t, int> >, Double32_t>"

gCling->lookupScope("CTDConst", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, skip).getAsString().c_str()
// CHECK: (const char * const) "C<A<B<const Double32_t, const int> >, Double32_t>"
