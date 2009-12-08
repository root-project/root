// @(#)root/g3d:$Id$
// Author: Nenad Buncic   17/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TTRD1.h"
#include "TNode.h"

ClassImp(TTRD1)

//______________________________________________________________________________
// Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/trd1.gif"> </P> End_Html
// TRD1 is a trapezoid with the x dimension varying along z.
// It has 7 parameters:
//
//     - name       name of the shape
//     - title      shape's title
//     - material  (see TMaterial)
//     - dx1        half-length along x at the z surface positioned at -DZ
//     - dx2        half-length along x at the z surface positioned at +DZ
//     - dy         half-length along the y-axis
//     - dz         half-length along the z-axis


//______________________________________________________________________________
TTRD1::TTRD1()
{
   // TRD1 shape default constructor

   fDx2 = 0.;
}


//______________________________________________________________________________
TTRD1::TTRD1(const char *name, const char *title, const char *material, Float_t dx1, Float_t dx2, Float_t dy, Float_t dz)
      : TBRIK(name, title,material,dx1,dy,dz)
{
   // TRD1 shape normal constructor

   fDx2 = dx2;
}


//______________________________________________________________________________
TTRD1::~TTRD1()
{
   // TRD1 shape default destructor
}


//______________________________________________________________________________
void TTRD1::SetPoints(Double_t *points) const
{
   // Create TRD1 points

   Float_t dx1, dx2, dy, dz;

   dx1 = TBRIK::fDx;
   dx2 = fDx2;
   dy  = TBRIK::fDy;
   dz  = TBRIK::fDz;

   if (points) {
      points[ 0] = -dx1 ; points[ 1] = -dy ; points[ 2] = -dz;
      points[ 3] = -dx1 ; points[ 4] =  dy ; points[ 5] = -dz;
      points[ 6] =  dx1 ; points[ 7] =  dy ; points[ 8] = -dz;
      points[ 9] =  dx1 ; points[10] = -dy ; points[11] = -dz;
      points[12] = -dx2 ; points[13] = -dy ; points[14] =  dz;
      points[15] = -dx2 ; points[16] =  dy ; points[17] =  dz;
      points[18] =  dx2 ; points[19] =  dy ; points[20] =  dz;
      points[21] =  dx2 ; points[22] = -dy ; points[23] =  dz;
   }
}
