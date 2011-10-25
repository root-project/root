// RUN: cat %s | %cling
// Actually test clang::DeclContext::removeDecl(). This function in clang is 
// the main method that is used for the error recovery. This means when there 
// is an error in cling's input we need to revert all the declarations that came
// in from the same transaction. Even when we have anonymous declarations we 
// need to be able to remove them from the declaration context. In a compiler's
// point of view there is no way that one can call removeDecl() and pass in anon
// decl, because the method is used when shadowing decls, which must have names.
// The issue is (and we patched it) is that removeDecl is trying to remove the
// anon decl (which doesn't have name) from the symbol (lookup) tables, which 
// doesn't make sense.
// The current test checks if that codepath in removeDecl still exists because
// it is important for the stable error recovery in cling

#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/VerifyDiagnosticConsumer.h"

#include "cling/Interpreter/Interpreter.h"

clang::DiagnosticsEngine& Diags = gCling->getCI()->getDiagnostics();
clang::DiagnosticConsumer* Client = new clang::VerifyDiagnosticConsumer(Diags);
Diags.setClient(Client);

.rawInput

class MyClass {
  struct {
    int a;
    error_here; // expected-error {{C++ requires a type specifier for all declarations}}
  };
};

.rawInput

.q
