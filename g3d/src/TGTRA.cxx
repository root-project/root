// @(#)root/g3d:$Name$:$Id$
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

ClassImp(TGTRA)


//______________________________________________________________________________
// Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/gtra.gif"> </P> End_Html
// GTRA is a general twisted trapezoid. The faces perpendicular to z are trapezia
// and their centres are not necessarily on a line parallel to the z axis as the
// TRAP; additionally, the faces may be twisted so that none of their edges are
// parallel. It is a TRAP shape, except that it is twisted in the x-y plane as a
// function of z. The parallel sides perpendicular to the z axis are rotated with
// respect to the x axis by an angle TWIST, which is one of the parameters. The
// shape is defined by the eight corners and is assumed to be constructed of
// straight lines joining points on the boundary of the trapezoidal face at z=-DZ
// to the corresponding points on the face at z=DZ. Divisions are not allowed.
// It has 15 parameters:
//
//     - name       name of the shape
//     - title      shape's title
//     - material  (see TMaterial)
//     - dZ         half-length along the z axis
//     - theta      polar angle of the line joining the centre of the face
//                  at -DZ to the centre of the one at +DZ
//     - phi        azimuthal angle of the line joining the centre of
//                  the face at -DZ to the centre of the one at +DZ
//     - twist      twist angle of the faces parallel to the x-y plane
//                  at z = +/- DZ around an axis parallel to z passing
//                  through their centre
//     - h1         half-length along y of the face at -DZ
//     - bl1        half-length along x of the side at -H1 in y of
//                  the face at -DZ in z
//     - tl1        half-length along x of the side at +H1 in y of the face
//                  at -DZ in z
//     - alpha1     angle with respect to the y axis from the centre of
//                  the side at -H1 in y to the centre of the side at
//                  +H1 in y of the face at -DZ in z
//     - h2         half-length along y of the face at +DZ
//     - bL2        half-length along x of the side at -H2 in y of the face at
//                  +DZ in z
//
//     - tl2        half-length along x of the side at +H2 in y of the face
//                  at +DZ in z
//
//     - alpha2     angle with respect to the y axis from the centre of the side
//                  at -H2 in y to the centre of the side at +H2 in y of the
//                  face at +DZ in z





//______________________________________________________________________________
TGTRA::TGTRA ()
{
//*-*-*-*-*-*-*-*-*-*-*-*GTRA shape default constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ==============================


}


//______________________________________________________________________________
TGTRA::TGTRA (const char *name, const char *title, const char *material, Float_t dz, Float_t theta,
              Float_t phi, Float_t twist, Float_t h1, Float_t bl1, Float_t tl1, Float_t alpha1,
              Float_t h2, Float_t bl2, Float_t tl2, Float_t alpha2)
      : TBRIK (name, title, material, theta, phi, dz)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*GTRA shape normal constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =============================

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


//______________________________________________________________________________
TGTRA::~TGTRA ()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*GTRA shape default destructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =============================


}

//______________________________________________________________________________
void TGTRA::SetPoints (Float_t *buff)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*Create GTRA points*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                            ==================

    Float_t x, y, dx, dy, dx1, dx2, dz, theta, phi, alpha1, alpha2, twist;
    const Float_t PI = Float_t (TMath::Pi());

    alpha1 = fAlpha1    * PI/180.0;
    alpha2 = fAlpha2    * PI/180.0;
    theta  = TBRIK::fDx * PI/180.0;
    phi    = TBRIK::fDy * PI/180.0;
    twist  = fTwist     * PI/180.0;

    dx  = 2*fDz*TMath::Sin(theta)*TMath::Cos(phi);
    dy  = 2*fDz*TMath::Sin(theta)*TMath::Sin(phi);
    dz  = TBRIK::fDz;

    dx1 = 2*fH1*TMath::Tan(alpha1);
    dx2 = 2*fH2*TMath::Tan(alpha2);

    if (buff) {
        buff[ 0] = -fBl1;        buff[ 1] = -fH1;    buff[ 2] = -dz;
        buff[ 3] =  fBl1;        buff[ 4] = -fH1;    buff[ 5] = -dz;
        buff[ 6] =  fTl1+dx1;    buff[ 7] =  fH1;    buff[ 8] = -dz;
        buff[ 9] = -fTl1+dx1;    buff[10] =  fH1;    buff[11] = -dz;
        buff[12] = -fBl2+dx;     buff[13] = -fH2+dy; buff[14] = dz;
        buff[15] =  fBl2+dx;     buff[16] = -fH2+dy; buff[17] = dz;
        buff[18] =  fTl2+dx+dx2; buff[19] =  fH2+dy; buff[20] = dz;
        buff[21] = -fTl2+dx+dx2; buff[22] =  fH2+dy; buff[23] = dz;
        for (Int_t i = 12; i < 24; i+=3) {
            x = buff[i];
            y = buff[i+1];
            buff[i]     = x*TMath::Cos(twist) + y*TMath::Sin(twist);
            buff[i+1]  = -x*TMath::Sin(twist) + y*TMath::Cos(twist);
        }
    }
}
