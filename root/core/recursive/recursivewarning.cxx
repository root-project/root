#include "TError.h"
#include "TROOT.h"

int main(int, char**)
{
  // gROOT;
  // This will trigger the initialization of gROOT and
  // trigger a nested Warning due to the duplicate rootmap file
  // This used to cause a dead lock (recursiverly taking the
  // non recursive lock GetErrorMutex() ).
  Info("RecursiveMessage", "We are at the start of main, gROOT is not yet initialized");

  return 0;
}
