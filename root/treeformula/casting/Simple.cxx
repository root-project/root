//
// Simple.cxx
//
#include <iostream.h>
#include "Simple.h"

ClassImp(Simple)
ClassImp(Simple)
ClassImp(Simple)

Simple::~Simple() {
// Destructor
  if (fShape) {
    delete fShape;
    fShape =0;
  }
}

void Simple::Print(Option_t *option) const {
  // Print the contents
  cout << "fID= " << fID << endl;
  fShape -> Print();

}
