//
// Simple.cxx
//
#include <iostream>
#include "Simple.h"

namespace std {} using namespace std;


Simple::~Simple() {
// Destructor
  if (fShape) {
    delete fShape;
    fShape =0;
  }
}

void Simple::Print(Option_t *) const {
  // Print the contents
  cout << "fID= " << fID << endl;
  fShape -> Print();

}

