// @(#)root/vmc:$Name:  $:$Id: TVirtualMCApplication.cxx,v 1.2 2003/09/23 14:03:15 brun Exp $
// Author: Ivana Hrivnacova, 27/03/2002

#include "TVirtualMCApplication.h"
#include "TError.h"

//______________________________________________________________________________
//
// Interface to a user Monte Carlo application.
//______________________________________________________________________________

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
  fgInstance = this;
}

//_____________________________________________________________________________
TVirtualMCApplication::~TVirtualMCApplication() 
{
  //
  // Destructor  
  //
  
  fgInstance = 0;
}
