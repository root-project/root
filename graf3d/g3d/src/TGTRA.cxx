// @(#)root/g3d:$Id$
// Author: Nenad Buncic   19/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGTRA.h"
#include "TNode.h"
#include "TMath.h"

ClassImp(TGTRA);

/** \class TGTRA
\ingroup g3d
A general twisted trapezoid.

\image html g3d_gtra.png

The faces perpendicular to z are trapezia
and their centres are not necessarily on a line parallel to the z axis as the
TRAP; additionally, the faces may be twisted so that none of their edges are
parallel. It is a TRAP shape, except that it is twisted in the x-y plane as a
function of z. The parallel sides perpendicular to the z axis are rotated with
respect to the x axis by an angle TWIST, which is one of the parameters. The
shape is defined by the eight corners and is assumed to be constructed of
straight lines joining points on the boundary of the trapezoidal face at z=-DZ
to the corresponding points on the face at z=DZ. Divisions are not allowed.
It has 15 parameters:

  - name:       name of the shape
  - title:      shape's title
  - material:   (see TMaterial)
  - dZ:         half-length along the z axis
  - theta:      polar angle of the line joining the centre of the face
                at -DZ to the centre of the one at +DZ
  - phi:        azimuthal angle of the line joining the centre of
                the face at -DZ to the centre of the one at +DZ
  - twist:      twist angle of the faces parallel to the x-y plane
                at z = +/- DZ around an axis parallel to z passing
                through their centre
  - h1:         half-length along y of the face at -DZ
  - bl1:        half-length along x of the side at -H1 in y of
                the face at -DZ in z
  - tl1:        half-length along x of the side at +H1 in y of the face
                at -DZ in z
  - alpha1:     angle with respect to the y axis from the centre of
                the side at -H1 in y to the centre of the side at
                +H1 in y of the face at -DZ in z
  - h2:         half-length along y of the face at +DZ
  - bL2:        half-length along x of the side at -H2 in y of the face at
               +DZ in z
  - tl2:        half-length along x of the side at +H2 in y of the face
                at +DZ in z
  - alpha2:     angle with respect to the y axis from the centre of the side
                at -H2 in y to the centre of the side at +H2 in y of the
                face at +DZ in z
*/

////////////////////////////////////////////////////////////////////////////////
/// GTRA shape default constructor.

TGTRA::TGTRA ()
{
   fTwist  = 0.;
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
/// GTRA shape normal constructor

TGTRA::TGTRA (const char *name, const char *title, const char *material, Float_t dz, Float_t theta,
              Float_t phi, Float_t twist, Float_t h1, Float_t bl1, Float_t tl1, Float_t alpha1,
              Float_t h2, Float_t bl2, Float_t tl2, Float_t alpha2)
      : TBRIK (name, title, material, theta, phi, dz)
{
   fTwist  = twist;
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
/// GTRA shape default destructor

TGTRA::~TGTRA ()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Create GTRA points

void TGTRA::SetPoints (Double_t *points) const
{
   Double_t x, y, dx, dy, dx1, dx2, dz, theta, phi, alpha1, alpha2, twist;
   const Float_t pi = Float_t (TMath::Pi());

   alpha1 = fAlpha1    * pi/180.0;
   alpha2 = fAlpha2    * pi/180.0;
   theta  = TBRIK::fDx * pi/180.0;
   phi    = TBRIK::fDy * pi/180.0;
   twist  = fTwist     * pi/180.0;

   dx  = 2*fDz*TMath::Sin(theta)*TMath::Cos(phi);
   dy  = 2*fDz*TMath::Sin(theta)*TMath::Sin(phi);
   dz  = TBRIK::fDz;

   dx1 = 2*fH1*TMath::Tan(alpha1);
   dx2 = 2*fH2*TMath::Tan(alpha2);

   if (points) {
      points[ 0] = -fBl1;        points[ 1] = -fH1;    points[ 2] = -dz;
      points[ 9] =  fBl1;        points[10] = -fH1;    points[11] = -dz;
      points[ 6] =  fTl1+dx1;    points[ 7] =  fH1;    points[ 8] = -dz;
      points[ 3] = -fTl1+dx1;    points[4]  =  fH1;    points[5] = -dz;
      points[12] = -fBl2+dx;     points[13] = -fH2+dy; points[14] = dz;
      points[21] =  fBl2+dx;     points[22] = -fH2+dy; points[23] = dz;
      points[18] =  fTl2+dx+dx2; points[19] =  fH2+dy; points[20] = dz;
      points[15] = -fTl2+dx+dx2; points[16] =  fH2+dy; points[17] = dz;
      for (Int_t i = 12; i < 24; i+=3) {
         x = points[i];
         y = points[i+1];
         points[i]     = x*TMath::Cos(twist) + y*TMath::Sin(twist);
         points[i+1]  = -x*TMath::Sin(twist) + y*TMath::Cos(twist);
      }
   }
}
