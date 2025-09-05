#include "Context.h"

#include <iostream>

ClassImp(Context)

//_____________________________________________________________________________
void Context::Print(Option_t *) const
{
   // Print this object

  std::cout << "Context::Print " << std::endl;
  std::cout << "Detector " << fDetector << " SimFlag " << fSimFlag
       << " Time " << fTimeStamp << std::endl;
}

