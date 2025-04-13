#include "TSystem.h"
#include "TInterpreter.h"
#ifndef _MSC_VER
#include <dlfcn.h>
#endif

// Reproducer for ROOT-8850

int main() {
   gSystem->Load("libImt");
   reinterpret_cast<void(*)(UInt_t)>(gInterpreter->FindSym("ROOT_TImplicitMT_EnableImplicitMT"))(0);
   return 0;
}
