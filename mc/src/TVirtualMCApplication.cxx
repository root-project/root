// @(#)root/mc:$Name:  $:$Id: TVirtualMCApplication.cxx,v 1.19 2002/04/08 15:06:08 brun Exp $
// Author: Ivana Hrivnacova, 27/03/2002

#include "TVirtualMCApplication.h"
#include "TError.h"

ClassImp(TVirtualMCApplication)

TVirtualMCApplication* TVirtualMCApplication::fgInstance = 0;

//_____________________________________________________________________________
TVirtualMCApplication::TVirtualMCApplication(const char *name, 
                                             const char *title) 
  : TNamed(name,title)
{
//
// Standard constructor
//

  if (fgInstance) {
    Fatal("TVirtualMCApplication", 
          "Attempt to create two instances of singleton.");
  }
      
  fgInstance = this;
}

//_____________________________________________________________________________
TVirtualMCApplication::TVirtualMCApplication()
  : TNamed()
{    
  //
  // Default constructor
  //
}

//_____________________________________________________________________________
TVirtualMCApplication::~TVirtualMCApplication() 
{
  //
  // Destructor  
  //
  
  fgInstance = 0;
}
