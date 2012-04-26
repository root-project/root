// RUN: cat %s | %cling | FileCheck %s
// XFAIL: *
//   due to last test:
//   cling: /home/axel/build/cling/trunk/lib/ExecutionEngine/JIT/JIT.cpp:400: virtual llvm::GenericValue llvm::JIT::runFunction(llvm::Function *, const std::vector<GenericValue> &): Assertion `(FTy->getNumParams() == ArgValues.size() || (FTy->isVarArg() && FTy->getNumParams() <= ArgValues.size())) && "Wrong number of arguments passed into function!"' failed.

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/Value.h"

cling::Value V;
V // CHECK: (cling::Value) <<<invalid>>> @0x{{.*}}

gCling->evaluate("1;", &V);
V // CHECK: (cling::Value) boxes [(int) 1]

long LongV = 17;
gCling->evaluate("LongV;", &V);
V // CHECK: (cling::Value) boxes [(long) 17]

int* IntP = (int*)0x12;
gCling->evaluate("IntP;", &V);
V // CHECK: (cling::Value) boxes [(int *) 0x12]

gCling->evaluate("V", &V);
V // CHECK: (cling::Value) boxes [(cling::Value) ???]

