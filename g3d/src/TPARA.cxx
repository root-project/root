// @(#)root/g3d:$Name$:$Id$
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
//*-*-*-*-*-*-*-*-*-*-*-*-*PARA shape default constructor*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ==============================

}


//______________________________________________________________________________
TPARA::TPARA(const char *name, const char *title, const char *material, Float_t dx, Float_t dy, Float_t dz,
             Float_t alpha, Float_t theta, Float_t phi) : TBRIK(name, title,material, dx, dy, dz)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*PARA shape normal constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =============================

    fAlpha = alpha;
    fTheta = theta;
    fPhi   = phi;
}


//______________________________________________________________________________
TPARA::~TPARA()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*PARA shape default destructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =============================

}

//______________________________________________________________________________
void TPARA::SetPoints(Float_t *buff)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*Create PARA points*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                            ==================

    if (!buff) return;
    Float_t dx, dy, dz, theta, phi, alpha;
    const Float_t PI = Float_t (TMath::Pi());

    alpha = fAlpha * PI/180.0;
    theta = fTheta * PI/180.0;
    phi   = fPhi   * PI/180.0;

    dx = TBRIK::fDx;
    dy = TBRIK::fDy;
    dz = TBRIK::fDz;
//    Parallelepiped change angles to tangents (by Pavel Nevski 12/04/99)
      Float_t TXY,TTH,TXZ,TYZ;

      TXY = TMath::Tan(alpha);
      TTH = TMath::Tan(theta);
      TXZ = TTH*TMath::Cos(phi);
      TYZ = TTH*TMath::Sin(phi);

      *buff++ = -dz*TXZ-TXY*dy-dx; *buff++ = -dy-dz*TYZ; *buff++ = -dz;
      *buff++ = -dz*TXZ-TXY*dy+dx; *buff++ = -dy-dz*TYZ; *buff++ = -dz;
      *buff++ = -dz*TXZ+TXY*dy+dx; *buff++ = +dy-dz*TYZ; *buff++ = -dz;
      *buff++ = -dz*TXZ+TXY*dy-dx; *buff++ = +dy-dz*TYZ; *buff++ = -dz;
      *buff++ = +dz*TXZ-TXY*dy-dx; *buff++ = -dy+dz*TYZ; *buff++ = +dz;
      *buff++ = +dz*TXZ-TXY*dy+dx; *buff++ = -dy+dz*TYZ; *buff++ = +dz;
      *buff++ = +dz*TXZ+TXY*dy+dx; *buff++ = +dy+dz*TYZ; *buff++ = +dz;
      *buff++ = +dz*TXZ+TXY*dy-dx; *buff++ = +dy+dz*TYZ; *buff++ = +dz;
}
