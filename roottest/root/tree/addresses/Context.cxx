#include <iostream>
using namespace std;

#include "Context.h"

ClassImp(Context)

//_____________________________________________________________________________
void Context::Print(Option_t *) const
{
   // Print this object

  cout << "Context::Print " << endl;
  cout << "Detector " << fDetector << " SimFlag " << fSimFlag 
       << " Time " << fTimeStamp << endl;
}

