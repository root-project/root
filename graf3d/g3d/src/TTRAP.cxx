// @(#)root/g3d:$Id$
// Author: Nenad Buncic   19/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TTRAP.h"
#include "TNode.h"
#include "TMath.h"

ClassImp(TTRAP);

/** \class TTRAP
\ingroup g3d
A general trapezoid.

\image html g3d_trap.png

The faces perpendicular to z are trapezia and
their centres are not necessarily on a line parallel to the z axis.
This shape has 14 parameters.

  - name:       name of the shape
  - title:      shape's title
  - material:   (see TMaterial)
  - dz:         half-length along the z axis
  - theta:      polar angle of the line joining the centre of the face
                at -DZ to the centre of the one at +DZ
  - phi:        azimuthal angle of the line joining the centre of the face
                at -DZ to the centre of the one at +DZ
  - h1:         half-length along y of the face at -DZ
  - bl1:        half-length along x of the side at -H1 in y of the face
                at -DZ in z
  - tl1:        half-length along x of the side at +H1 in y of the face
                at -DZ in z
  - alpha1:     angle with respect to the y axis from the centre of the
                side at -H1 in y to the centre of the side at +H1 in y
                of the face at -DZ in z
  - h2:         half-length along y of the face at +DZ
  - bl2:        half-length along x of the side at -H2 in y of the
               face at +DZ in z
  - tl2:        half-length along x of the side at +H2 in y of the face
                at +DZ in z
  - alpha2:     angle with respect to the y axis from the centre of the side
                at -H2 in y to the centre of the side at +H2 in y of the
               face at +DZ in z
*/

////////////////////////////////////////////////////////////////////////////////
/// TRAP shape default constructor

TTRAP::TTRAP()
{
   fH1     = 0.;
   fBl1    = 0.;
   fTl1    = 0.;
   fAlpha1 = 0.;
   fH2     = 0.;
   fBl2    = 0.;
   fTl2    = 0.;
   fAlpha2 = 0.;
}

////////////////////////////////////////////////////////////////////////////////
/// TRAP shape normal constructor

TTRAP::TTRAP(const char *name, const char *title, const char *material, Float_t dz, Float_t theta, Float_t phi,
             Float_t h1, Float_t bl1, Float_t tl1, Float_t alpha1, Float_t h2,
             Float_t bl2, Float_t tl2, Float_t alpha2) : TBRIK(name, title,material,theta,phi,dz)
{
   fH1     = h1;
   fBl1    = bl1;
   fTl1    = tl1;
   fAlpha1 = alpha1;
   fH2     = h2;
   fBl2    = bl2;
   fTl2    = tl2;
   fAlpha2 = alpha2;
}

////////////////////////////////////////////////////////////////////////////////
/// TRAP shape default destructor

TTRAP::~TTRAP()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Create TRAP points

void TTRAP::SetPoints(Double_t *points) const
{
   const Float_t pi = Float_t (TMath::Pi());
   Float_t alpha1 = fAlpha1    * pi/180.0;
   Float_t alpha2 = fAlpha2    * pi/180.0;
   Float_t theta  = TBRIK::fDx * pi/180.0;
   Float_t phi    = TBRIK::fDy * pi/180.0;
   Float_t tth    = TMath::Tan(theta);
   Float_t tx     = tth*TMath::Cos(phi);
   Float_t ty     = tth*TMath::Sin(phi);
   Float_t tth1   = TMath::Tan(alpha1);
   Float_t tth2   = TMath::Tan(alpha2);

   if (points) {
      points[ 0] = -fDz*tx-tth1*fH1-fBl1 ; points[ 1] = -fH1-fDz*ty ; points[ 2] = -fDz;
      points[ 3] = -fDz*tx+tth1*fH1-fTl1 ; points[ 4] =  fH1-fDz*ty ; points[ 5] = -fDz;
      points[ 6] = -fDz*tx+tth1*fH1+fTl1 ; points[ 7] =  fH1-fDz*ty ; points[ 8] = -fDz;
      points[ 9] = -fDz*tx-tth1*fH1+fBl1 ; points[10] = -fH1-fDz*ty ; points[11] = -fDz;
      points[12] =  fDz*tx-tth2*fH2-fBl2 ; points[13] = -fH2+fDz*ty ; points[14] = fDz;
      points[15] =  fDz*tx+tth2*fH2-fTl2 ; points[16] =  fH2+fDz*ty ; points[17] = fDz;
      points[18] =  fDz*tx+tth2*fH2+fTl2 ; points[19] =  fH2+fDz*ty ; points[20] = fDz;
      points[21] =  fDz*tx-tth2*fH2+fBl2 ; points[22] = -fH2+fDz*ty ; points[23] = fDz;
   }

}
