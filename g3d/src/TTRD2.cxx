// @(#)root/g3d:$Name$:$Id$
// Author: Nenad Buncic   13/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TTRD2.h"
#include "TNode.h"

ClassImp(TTRD2)

//______________________________________________________________________________
// Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/trd2.gif"> </P> End_Html
// TRD2 is a trapezoid with both x and y dimensions varying along z.
// It has 8 parameters:
//
//     - name       name of the shape
//     - title      shape's title
//     - material  (see TMaterial)
//     - dx1        half-length along x at the z surface positioned at -DZ
//     - dx2        half-length along x at the z surface positioned at +DZ
//     - dy1        half-length along y at the z surface positioned at -DZ
//     - dy2        half-length along y at the z surface positioned at +DZ
//     - dz         half-length along the z-axis




//______________________________________________________________________________
TTRD2::TTRD2()
{
//*-*-*-*-*-*-*-*-*-*-*-*TRD2 shape default constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ==============================


}


//______________________________________________________________________________
TTRD2::TTRD2(const char *name, const char *title, const char *material, Float_t dx1, Float_t dx2, Float_t dy1,
       Float_t dy2, Float_t dz) : TBRIK(name, title,material,dx1,dy1,dz)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*TRD2 shape normal constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =============================

    fDx2 = dx2;
    fDy2 = dy2;
}


//______________________________________________________________________________
TTRD2::~TTRD2()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*TRD2 shape default destructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =============================


}

//______________________________________________________________________________
void TTRD2::SetPoints(Float_t *buff)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*Create TRD2 points*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                          ==================

    Float_t dx1, dx2, dy1, dy2, dz;

    dx1 = TBRIK::fDx;
    dx2 = fDx2;
    dy1 = TBRIK::fDy;
    dy2 = fDy2;
    dz  = TBRIK::fDz;

    if (buff) {
        buff[ 0] = -dx1;  buff[ 1] = -dy1;  buff[ 2] = -dz;
        buff[ 3] =  dx1;  buff[ 4] = -dy1;  buff[ 5] = -dz;
        buff[ 6] =  dx1;  buff[ 7] =  dy1;  buff[ 8] = -dz;
        buff[ 9] = -dx1;  buff[10] =  dy1;  buff[11] = -dz;
        buff[12] = -dx2;  buff[13] = -dy2;  buff[14] =  dz;
        buff[15] =  dx2;  buff[16] = -dy2;  buff[17] =  dz;
        buff[18] =  dx2;  buff[19] =  dy2;  buff[20] =  dz;
        buff[21] = -dx2;  buff[22] =  dy2;  buff[23] =  dz;
    }
}
