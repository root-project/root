// RUN: cat %s | %cling | FileCheck %s
// Test Interpreter::lookupScope()
#include "cling/Interpreter/Interpreter.h"
#include "clang/AST/Type.h"

typedef int Int_t;
.rawInput 1
template <typename T> struct W { T member; };
.rawInput 0
const clang::Type* resType = 0;
gCling->lookupScope("W<Int_t>", &resType);
//resType->dump(); 
clang::QualType(resType,0).getAsString().c_str()
//CHECK: (const char * const) "W<Int_t>"
