#include "TNamed.h"
#include <iostream>

// ROOT-7830
int reread() {
  gDirectory->Add(new TNamed("This", "That!"));
  long first = gROOT->ProcessLine("This");
  long second = gROOT->ProcessLine("This");
  if (first != second) {
    std::cerr << "FAILURE: Second use of special var rereads the object!\n";
    return 1;
  }
  return 0;
}

