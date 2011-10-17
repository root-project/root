// RUN: cat %s | %cling
// RUN: cat %s | %cling | FileCheck %s

// Checks for infinite recursion when we combine nested calls of process line 
// with global initializers.

#include "cling/Interpreter/Interpreter.h"

class MyClass { public:  MyClass(){ gCling->processLine("gCling->getVersion()");} };

MyClass *My = new MyClass(); // CHECK: {{.*Interpreter.*}}

.q
