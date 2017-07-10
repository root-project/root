#include "TSystem.h"
#include "TInterpreter.h"
#include <dlfcn.h>

// Reproducer for ROOT-8850

int main() {
   gSystem->Load("libImt");
   reinterpret_cast<void(*)(UInt_t)>(gInterpreter->FindSym("ROOT_TImplicitMT_EnableImplicitMT"))(0);
   return 0;
}
