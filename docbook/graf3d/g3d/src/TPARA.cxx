// @(#)root/g3d:$Id$
// Author: Nenad Buncic   19/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TPARA.h"
#include "TNode.h"
#include "TMath.h"

ClassImp(TPARA)


//______________________________________________________________________________
// Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/para.gif"> </P> End_Html
// PARA is a parallelepiped. It has 9 parameters:
//
//     - name       name of the shape
//     - title      shape's title
//     - material  (see TMaterial)
//     - dx         half-length in x
//     - dy         half-length in y
//     - dz         half-length in z
//     - alpha      angle formed by the y axis and by the plane joining the
//                  centre of the faces parallel to the z-x plane at -DY and +DY
//     - theta      polar angle of the line joining the centres of the faces
//                  at -DZ and +DZ in z
//     - phi        azimuthal angle of the line joining the centres of the
//                  faces at -DZ and +DZ in z


//______________________________________________________________________________
TPARA::TPARA()
{
   // PARA shape default constructor

   fAlpha = 0.;
   fTheta = 0.;
   fPhi   = 0.;
}


//______________________________________________________________________________
TPARA::TPARA(const char *name, const char *title, const char *material, Float_t dx, Float_t dy, Float_t dz,
             Float_t alpha, Float_t theta, Float_t phi) : TBRIK(name, title,material, dx, dy, dz)
{
   // PARA shape normal constructor

   fAlpha = alpha;
   fTheta = theta;
   fPhi   = phi;
}


//______________________________________________________________________________
TPARA::~TPARA()
{
   // PARA shape default destructor
}


//______________________________________________________________________________
void TPARA::SetPoints(Double_t *points) const
{
   // Create PARA points

   if (!points) return;
   Float_t dx, dy, dz, theta, phi, alpha;
   const Float_t pi = Float_t (TMath::Pi());

   alpha = fAlpha * pi/180.0;
   theta = fTheta * pi/180.0;
   phi   = fPhi   * pi/180.0;

   dx = TBRIK::fDx;
   dy = TBRIK::fDy;
   dz = TBRIK::fDz;

   // Parallelepiped change angles to tangents (by Pavel Nevski 12/04/99)
   Double_t txy = TMath::Tan(alpha);
   Double_t tth = TMath::Tan(theta);
   Double_t txz = tth*TMath::Cos(phi);
   Double_t tyz = tth*TMath::Sin(phi);

   *points++ = -dz*txz-txy*dy-dx ; *points++ = -dy-dz*tyz ; *points++ = -dz;
   *points++ = -dz*txz+txy*dy-dx ; *points++ = +dy-dz*tyz ; *points++ = -dz; //3
   *points++ = -dz*txz+txy*dy+dx ; *points++ = +dy-dz*tyz ; *points++ = -dz;
   *points++ = -dz*txz-txy*dy+dx ; *points++ = -dy-dz*tyz ; *points++ = -dz;//1
   *points++ = +dz*txz-txy*dy-dx ; *points++ = -dy+dz*tyz ; *points++ = +dz;
   *points++ = +dz*txz+txy*dy-dx ; *points++ = +dy+dz*tyz ; *points++ = +dz;//7
   *points++ = +dz*txz+txy*dy+dx ; *points++ = +dy+dz*tyz ; *points++ = +dz;
   *points++ = +dz*txz-txy*dy+dx ; *points++ = -dy+dz*tyz ; *points++ = +dz;//5
}
