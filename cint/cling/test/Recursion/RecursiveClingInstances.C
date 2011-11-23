// RUN: cat %s | %cling -I%p

// Tests the ability of cling to host itself. We can have cling instances in
// cling's runtime. This is important for people, who use cling embedded in
// their frameworks.

#include "cling/Interpreter/Interpreter.h"

gCling->processLine("const char * const argV = \"cling\";");
gCling->processLine("cling::Interpreter *DefaultInterp;");

gCling->processLine("DefaultInterp = new cling::Interpreter(1, &argV);");
gCling->processLine("DefaultInterp->processLine(\"#include \\\"cling/Interpreter/Interpreter.h\\\"\");");
gCling->processLine("DefaultInterp->processLine(\"gCling->createUniqueName()\");");
