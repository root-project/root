#include "ROOT10426.h"
#include "TROOT.h"

// Tests ROOT-10426:
// The library containing this source file gets autoloaded
// through symbol resolution during jitting. This library then
// does jitting *while* the outer jitting stackframes are active
// that triggered autoloading.
// Make sure that the nested jitted code can be run (its code
// memory is finalized) and that the triggering jitted memory
// can still be relocated (because it's not yet finalized).
auto sTriggerStaticInit = gROOT->ProcessLine("printf(\"Recursing!\\n\"); 17");

int AutoloadMe::foo() { return 42; }
int triggerSymbol = AutoloadMe{}.foo();
