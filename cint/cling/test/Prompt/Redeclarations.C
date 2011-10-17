// RUN: cat %s | %cling

#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/VerifyDiagnosticConsumer.h"

#include "cling/Interpreter/Interpreter.h"

clang::DiagnosticsEngine& Diags = gCling->getCI()->getDiagnostics();
clang::DiagnosticConsumer* Client = new clang::VerifyDiagnosticConsumer(Diags);
Diags.setClient(Client);

#include <string>
std::string s;
std::string s; // expected-error {{redefinition of 's'}} expected-note {{previous definition is here}}

const char* a = "test";
const char* a = ""; // expected-error {{redefinition of 'a'}} expected-note {{previous definition is here}}

.q
