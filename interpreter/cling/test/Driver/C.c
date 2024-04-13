//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -x c -Xclang -verify 2>&1 | FileCheck %s
// RUN: cat %s | %cling -x c -fsyntax-only -Xclang -verify 2>&1

// Validate cling C mode.

int printf(const char*,...);
printf("CHECK 123 %p\n", gCling); // CHECK: CHECK 123

int i = 1 // CHECK: (int) 1
sizeof(int) // CHECK: (unsigned long) 4
int x = sizeof(int);
printf("CHECK %d\n", x); // CHECK: CHECK 4

// expected-no-diagnostics
.q
