//
// $Id$
//

#include "cltestClass.h"


#include <Riostream.h>
#include <TString.h>
#include <TObjString.h>


ClassImp(TestClass)
ClassImpT(TestClass::IE_t, T)


//________________________________________________________________________
TestClass::TestClass() 
   : TObject(),
     fBMass(4.95)
{
   // Default constructor.
}


//________________________________________________________________________
TestClass::~TestClass()
{
   // Destructor.
}
