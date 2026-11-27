//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I%p 2>&1 | FileCheck %s
// Test that unloading a file with instantiated generic lambdas does not crash.

.L genericlambda.h
genericLambda() + genericLambdaRetType()
// CHECK: (int) 3
makeGenericLambda()(4)
// CHECK: (int) 4
.U genericlambda.h

.L genericlambda.h
genericLambda() + genericLambdaRetType()
// CHECK: (int) 3
makeGenericLambda()(5)
// CHECK: (int) 5
.U genericlambda.h

.L genericlambda.h
genericLambda() + genericLambdaRetType()
// CHECK: (int) 3

.q
