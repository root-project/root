// @(#)root/g3d:$Name$:$Id$
// Author: Nenad Buncic   18/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TView.h"
#include "TCONE.h"
#include "TNode.h"

ClassImp(TCONE)

//______________________________________________________________________________
// Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/cone.gif"> </P> End_Html
// CONE is a conical tube. It has 8 parameters:
//
//     - name       name of the shape
//     - title      shape's title
//     - material  (see TMaterial)
//     - dz         half-length in z
//     - rmin1      inside radius at -DZ in z
//     - rmax1      outside radius at -DZ in z
//     - rmin2      inside radius at +DZ in z
//     - rmax2      outside radius at +DZ in z






//______________________________________________________________________________
TCONE::TCONE()
{
//*-*-*-*-*-*-*-*-*-*-*-*CONE shape default constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ==============================


}


//______________________________________________________________________________
TCONE::TCONE(const char *name, const char *title, const char *material, Float_t dz,
             Float_t rmin1, Float_t rmax1,
             Float_t rmin2, Float_t rmax2)
      : TTUBE(name, title,material,rmin1,rmax1,dz)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*CONE shape normal constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =============================

    fRmin2 = rmin2;
    fRmax2 = rmax2;
}


//______________________________________________________________________________
TCONE::TCONE(const char *name, const char *title, const char *material, Float_t dz, Float_t rmax1
            , Float_t rmax2) : TTUBE(name, title,material,0,rmax1,dz)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*CONE shape "simplified" constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ==================================

    fRmin2 = 0;
    fRmax2 = rmax2;
}
//______________________________________________________________________________
TCONE::~TCONE()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*CONE shape default destructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =============================

}

//______________________________________________________________________________
void TCONE::SetPoints(Float_t *buff)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*Create CONE points*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                            ==================

    Float_t rmin1, rmax1, dz;
    Int_t j, n;

    n = GetNumberOfDivisions();

    rmin1 = TTUBE::fRmin;
    rmax1 = TTUBE::fRmax;
    dz    = TTUBE::fDz;

    Int_t indx = 0;

//*-* We've to checxk whether the table does exist and create it
//*-* since fCoTab/fSiTab are not saved with any TShape::Streamer function
    if (!fCoTab)   MakeTableOfCoSin();

    if (buff) {
        for (j = 0; j < n; j++) {
            buff[indx++] = rmin1 * fCoTab[j];
            buff[indx++] = rmin1 * fSiTab[j];
            buff[indx++] = -dz;
        }
        for (j = 0; j < n; j++) {

            buff[indx++] = rmax1 * fCoTab[j];
            buff[indx++] = rmax1 * fSiTab[j];
            buff[indx++] = -dz;
        }

        for (j = 0; j < n; j++) {
            buff[indx++] = fRmin2 * fCoTab[j];
            buff[indx++] = fRmin2 * fSiTab[j];
            buff[indx++] = dz;
        }

        for (j = 0; j < n; j++) {
            buff[indx++] = fRmax2 * fCoTab[j];
            buff[indx++] = fRmax2 * fSiTab[j];
            buff[indx++] = dz;
        }
    }
}
