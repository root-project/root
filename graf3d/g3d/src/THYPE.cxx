// @(#)root/g3d:$Id$
// Author: Rene Brun   08/12/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "THYPE.h"

ClassImp(THYPE)


//______________________________________________________________________________
// Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/hype.gif"> </P> End_Html
// HYPE is an hyperboloid (not implemented. It has 4 parameters:
//
//     - name       name of the shape
//     - title      shape's title
//     - material  (see TMaterial)
//     - rmin       inner radius of the tube
//     - rmax       outer radius of the tube
//     - dz         half-length of the box along the z-axis
//     - phi        stereo angle


//______________________________________________________________________________
THYPE::THYPE()
{
   // HYPE shape default constructor

   fPhi = 0.;
}


//______________________________________________________________________________
THYPE::THYPE(const char *name, const char *title, const char *material, Float_t rmin,
             Float_t rmax, Float_t dz, Float_t phi)
      : TTUBE(name,title,material,rmin,rmax,dz)
{
   // HYPE shape normal constructor

   fPhi = phi;
}


//______________________________________________________________________________
THYPE::~THYPE()
{
   // HYPE shape default destructor
}
