// @(#)root/g3d:$Name$:$Id$
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

ClassImp(TTRAP)


//______________________________________________________________________________
// Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/trap.gif"> </P> End_Html
// TRAP is a general trapezoid. The faces perpendicular to z are trapezia and
// their centres are not necessarily on a line parallel to the z axis.
// This shape has 14 parameters.
//
//     - name       name of the shape
//     - title      shape's title
//     - material  (see TMaterial)
//     - dz         half-length along the z axis
//     - theta      polar angle of the line joining the centre of the face
//                  at -DZ to the centre of the one at +DZ
//     - phi        azimuthal angle of the line joining the centre of the face
//                  at -DZ to the centre of the one at +DZ
//     - h1         half-length along y of the face at -DZ
//     - bl1        half-length along x of the side at -H1 in y of the face
//                  at -DZ in z
//     - tl1        half-length along x of the side at +H1 in y of the face
//                  at -DZ in z
//     - alpha1     angle with respect to the y axis from the centre of the
//                  side at -H1 in y to the centre of the side at +H1 in y
//                  of the face at -DZ in z
//     - h2         half-length along y of the face at +DZ
//     - bl2        half-length along x of the side at -H2 in y of the
//                  face at +DZ in z
//     - tl2        half-length along x of the side at +H2 in y of the face
//                  at +DZ in z
//     - alpha2     angle with respect to the y axis from the centre of the side
//                  at -H2 in y to the centre of the side at +H2 in y of the
//                  face at +DZ in z





//______________________________________________________________________________
TTRAP::TTRAP()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*TRAP shape default constructor*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ==============================

}


//______________________________________________________________________________
TTRAP::TTRAP(const char *name, const char *title, const char *material, Float_t dz, Float_t theta, Float_t phi,
             Float_t h1, Float_t bl1, Float_t tl1, Float_t alpha1, Float_t h2,
             Float_t bl2, Float_t tl2, Float_t alpha2) : TBRIK(name, title,material,theta,phi,dz)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*TRAP shape normal constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =============================

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
TTRAP::~TTRAP()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*TRAP shape default destructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =============================

}

//______________________________________________________________________________
void TTRAP::SetPoints(Float_t *buff)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*Create TRAP points*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                            ==================

    const Float_t PI = Float_t (TMath::Pi());
    Float_t alpha1 = fAlpha1    * PI/180.0;
    Float_t alpha2 = fAlpha2    * PI/180.0;
    Float_t theta  = TBRIK::fDx * PI/180.0;
    Float_t phi    = TBRIK::fDy * PI/180.0;
    Float_t tth    = TMath::Tan(theta);
    Float_t tx     = tth*TMath::Cos(phi);
    Float_t ty     = tth*TMath::Sin(phi);
    Float_t tth1   = TMath::Tan(alpha1);
    Float_t tth2   = TMath::Tan(alpha2);

    if (buff) {
       buff[ 0] = -fDz*tx+tth1*fH1+fTl1; buff[ 1] =  fH1-fDz*ty; buff[ 2] = -fDz;
       buff[ 3] = -fDz*tx+tth1*fH1-fTl1; buff[ 4] =  fH1-fDz*ty; buff[ 5] = -fDz;
       buff[ 6] = -fDz*tx-tth1*fH1-fBl1; buff[ 7] = -fH1-fDz*ty; buff[ 8] = -fDz;
       buff[ 9] = -fDz*tx-tth1*fH1+fBl1; buff[10] = -fH1-fDz*ty; buff[11] = -fDz;
       buff[12] =  fDz*tx+tth2*fH2+fTl2; buff[13] =  fH2+fDz*ty; buff[14] = fDz;
       buff[15] =  fDz*tx+tth2*fH2-fTl2; buff[16] =  fH2+fDz*ty; buff[17] = fDz;
       buff[18] =  fDz*tx-tth2*fH2-fBl2; buff[19] = -fH2+fDz*ty; buff[20] = fDz;
       buff[21] =  fDz*tx-tth2*fH2+fBl2; buff[22] = -fH2+fDz*ty; buff[23] = fDz;
    }
}
