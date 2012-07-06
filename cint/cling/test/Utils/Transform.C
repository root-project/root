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

gCling->lookupScope("B<Long_t, Int_t*>", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, skip).getAsString().c_str()
// CHECK:(const char * const) "B<long, Int_t *>"

gCling->lookupScope("A<B<Double32_t, Int_t*> >", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, skip).getAsString().c_str()
// getDesugaredType - Return the specified type with any "sugar" removed from the type. 
// This takes off typedefs, typeof's etc. If the outer level of the type is 
// already concrete, it returns it unmodified. This is similar to getting the 
// canonical type, but it doesn't remove *all* typedefs. For example, it 
// returns "T*" as "T*", (not as "int*"), because the pointer is concrete.

// CHECK:(const char * const) "A<B<Double32_t, Int_t *> >"

gCling->lookupScope("CTD", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, skip).getAsString().c_str()
// CHECK: (const char * const) "C<A<B<Double32_t, int> >, Double32_t>"

gCling->lookupScope("CTDConst", &t);
QT = clang::QualType(t, 0);
Transform::GetPartiallyDesugaredType(Ctx, QT, skip).getAsString().c_str()
// CHECK: (const char * const) "C<A<B<const Double32_t, const int> >, Double32_t>"
