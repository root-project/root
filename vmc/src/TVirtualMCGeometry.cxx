// @(#)root/vmc:$Name:  $:$Id: TVirtualMCGeometry.cxx,v 1.1 2003/07/15 09:56:58 brun Exp $
// Authors: ... 25/06/2002

#include "TVirtualMCGeometry.h"

//______________________________________________________________________________
//   Virtual MCGeometry provides a virtual interface to Monte Carlo
//   geometry construction. 
//______________________________________________________________________________

ClassImp(TVirtualMCGeometry)

TVirtualMCGeometry* TVirtualMCGeometry::fgInstance=0;

//_____________________________________________________________________________
TVirtualMCGeometry::TVirtualMCGeometry(const char *name, const char *title) 
  : TNamed(name,title)
{
   //
   // Standard constructor
   //
   if (fgInstance) 
      Warning("TVirtualMCGeometry","TVirtualMCGeometry instance already exists");
   else
      fgInstance=this;
}

//_____________________________________________________________________________
TVirtualMCGeometry::TVirtualMCGeometry()
  : TNamed()
{    
   //
   // Default constructor
   //
}

//_____________________________________________________________________________
TVirtualMCGeometry::~TVirtualMCGeometry() 
{
   //
   // Destructor
   //
   fgInstance=0;
}
