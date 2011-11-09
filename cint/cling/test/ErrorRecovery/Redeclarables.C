// RUN: cat %s | %cling -I%p | FileCheck %s

// Test the removal of decls from the redeclaration chain, which are marked as
// redeclarables.

#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/VerifyDiagnosticConsumer.h"

#include "cling/Interpreter/Interpreter.h"

clang::DiagnosticsEngine& Diags = gCling->getCI()->getDiagnostics();
clang::DiagnosticConsumer* Client = new clang::VerifyDiagnosticConsumer(Diags);
Diags.setClient(Client);

extern int my_int;
.rawInput 1
int my_funct();
.rawInput 0

#include "Redeclarables.h"

.rawInput 1
int my_funct() { 
  return 20;
}
.rawInput 0

int my_int = 20;

my_int
// CHECK: (int) 20

my_funct()
// CHECK: (int const) 20

.q
