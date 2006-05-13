// @(#)root/vmc:$Name:  $:$Id: TVirtualMCApplication.cxx,v 1.3 2004/02/10 13:46:37 brun Exp $
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
