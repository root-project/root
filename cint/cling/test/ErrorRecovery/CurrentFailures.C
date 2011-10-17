// RUN: cat %s | %cling -I%p | FileCheck %s
// XFAIL: *

// The issue is that when we don't have semicolon in the end of the input
// in the case of cling where we have incremental compilation the parser 
// encounters an error and tries to recover by looking for so called safe tokens
// or anchors (e.g ';' or EOF). The input doesn't have semicolon and we play 
// with Lexer's EOF tokens. My assumption is that while trying to recover
// and continue parsing the Lexer enters sort-of error state and it breaks with // assertion

#define BEGIN_NAMESPACE namespace test_namespace {
#define END_NAMESPACE }

#include "MacroIncluder.h"

extern "C" int printf // expected-error {{expected ';' after top level declarator}}

a;
