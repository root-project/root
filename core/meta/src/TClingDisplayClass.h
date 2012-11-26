#ifndef ROOT_ClingDisplayClass
#define ROOT_ClingDisplayClass

#include <string>
#include <cstdio>

namespace cling {

class Interpreter;

}

namespace TClingDisplayClass {

void DisplayAllClasses(FILE *out, const class cling::Interpreter *interpreter, bool verbose);
void DisplayClass(FILE *out, const cling::Interpreter *interpreter, const char *className, bool verbose);

void DisplayGlobals(FILE *out, const cling::Interpreter *interpreter);
void DisplayGlobal(FILE *out, const cling::Interpreter *interpreter, const std::string &name);

}

#endif
