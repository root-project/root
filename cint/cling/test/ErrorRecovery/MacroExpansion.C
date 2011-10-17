// RUN: cat %s | %cling -I%p

#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/VerifyDiagnosticConsumer.h"

#include "cling/Interpreter/Interpreter.h"

clang::DiagnosticsEngine& Diags = gCling->getCI()->getDiagnostics();
clang::DiagnosticConsumer* Client = new clang::VerifyDiagnosticConsumer(Diags);
Diags.setClient(Client);

#define BEGIN_NAMESPACE namespace test_namespace {
#define END_NAMESPACE }

.rawInput 1

BEGIN_NAMESPACE int j; END_NAMESPACE
BEGIN_NAMESPACE int j; END_NAMESPACE // expected-error {{redefinition of 'j'}} expected-note {{previous definition is here}} 

.rawInput 0
