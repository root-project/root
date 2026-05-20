#include "TError.h"
#include "TROOT.h"

int func() {
   Info("RecursiveMessage", "We are before the start of main, gROOT is not yet initialized");
   return 0;
}

// This will trigger the initialization of gROOT and
// trigger a nested Warning due to the duplicate rootmap file
// This used to cause a dead lock (recursiverly taking the
// non recursive lock GetErrorMutex() ).
auto i = func();

int main(int, char**)
{
  return 0;
}
