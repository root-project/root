//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify 2>&1 | FileCheck %s
// Test diagnostics masking via `CompilationOptions::IgnoreDiagsMask`
#include "cling/Interpreter/Interpreter.h"

int (ii) = 0 // expected-warning {{redundant parentheses surrounding declarator}}
// CHECK: (int) 0

// [-Wredundant-parens] diagnostic should not be emitted in this case
gCling->declare("int (jj);");

.q
