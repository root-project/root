// RUN: cat %s | %cling
// RUN: cat %s | %cling 2>&1 | FileCheck %s
// Test handling and recovery from calling an unresolved symbol.

// due to segv on int a = 12: XFAIL: *
.rawInput
int foo(); //CHECK: Error: Symbol '
.rawInput
foo()
int a = 12 // SEGVs
foo()
