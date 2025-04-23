#include "ROOT9112.C++"

int runROOT9112() {
  if (!gClassTable->GetDict("OUTER")) {
    std::cerr << "ERROR: cannot find dictionary for class `OUTER`!\n";
    exit(1);
  }
  return 0;
}
