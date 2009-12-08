// @(#)root/g3d:$Id$
// Author: Nenad Buncic   18/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TCONS.h"
#include "TNode.h"


ClassImp(TCONS)



//______________________________________________________________________________
// Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/cons.gif"> </P> End_Html
// CONS is a segment of a conical tube. It has 10 parameters:
//
//     - name       name of the shape
//     - title      shape's title
//     - material  (see TMaterial)
//     - dz         half-length in z
//     - rmin1      inside radius at -DZ in z
//     - rmax1      outside radius at -DZ in z
//     - rmin2      inside radius at +DZ in z
//     - rmax2      outside radius at +DZ in z
//     - phi1       starting angle of the segment
//     - phi2       ending angle of the segment
//
//
// NOTE: phi1 should be smaller than phi2. If this is not the case,
//       the system adds 360 degrees to phi2.





//______________________________________________________________________________
TCONS::TCONS()
{
   // CONS shape default constructor

   fRmin2 = 0.;
   fRmax2 = 0.;
}


//______________________________________________________________________________
TCONS::TCONS(const char *name, const char *title, const char *material, Float_t dz, Float_t rmin1, Float_t rmax1, Float_t rmin2, Float_t rmax2,
             Float_t phi1, Float_t phi2)
      : TTUBS(name,title,material,rmin1,rmax1,dz,phi1,phi2)
{
   // CONS shape normal constructor

   fRmin2 = rmin2;
   fRmax2 = rmax2;
}


//______________________________________________________________________________
TCONS::TCONS(const char *name, const char *title, const char *material,  Float_t rmax1, Float_t dz
                          , Float_t phi1, Float_t phi2, Float_t rmax2)
             : TTUBS(name,title,material,rmax1,dz,phi1,phi2)
{
   // CONS shape normal constructor

   fRmin2 = 0;
   fRmax2 = rmax2;
}

//______________________________________________________________________________
TCONS::~TCONS()
{
   // CONS shape default destructor
}

//______________________________________________________________________________
void TCONS::SetPoints(Double_t *points) const
{
   // Create CONS points

   Int_t j, n;
   Float_t rmin1, rmax1, dz;

   n = GetNumberOfDivisions()+1;

   rmin1 = TTUBE::fRmin;
   rmax1 = TTUBE::fRmax;
   dz    = TTUBE::fDz;

   Int_t indx = 0;

   if (!fCoTab) MakeTableOfCoSin();

   if (points) {
      for (j = 0; j < n; j++) {
         points[indx++] = rmin1 * fCoTab[j];
         points[indx++] = rmin1 * fSiTab[j];
         points[indx++] = -dz;
      }
      for (j = 0; j < n; j++) {
         points[indx++] = rmax1 * fCoTab[j];
         points[indx++] = rmax1 * fSiTab[j];
         points[indx++] = -dz;
      }
      for (j = 0; j < n; j++) {
         points[indx++] = fRmin2 * fCoTab[j];
         points[indx++] = fRmin2 * fSiTab[j];
         points[indx++] = dz;
      }
      for (j = 0; j < n; j++) {
         points[indx++] = fRmax2 * fCoTab[j];
         points[indx++] = fRmax2 * fSiTab[j];
         points[indx++] = dz;
      }
   }
}
