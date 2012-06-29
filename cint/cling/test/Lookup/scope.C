// RUN: cat %s | %cling | FileCheck %s
// Test Interpreter::lookupScope()
#include "cling/Interpreter/Interpreter.h"
#include "clang/AST/Type.h"

typedef int Int_t;
.rawInput
template <typename T> struct W { T member; };
.rawInput
const Type* resType = 0;
gCling->lookup("W<Int_t>");
resType->dump(); // CHECK: W<Int_t>
