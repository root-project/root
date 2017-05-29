// @(#)root/g3d:$Id$
// Author: Nenad Buncic   19/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPARA
#define ROOT_TPARA


////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TPARA                                                                  //
//                                                                        //
// PARA is parallelepiped. It has 6 parameters, the half length in x,     //
// the half length in y, the half length in z, the angle w.r.t. the y     //
// axis from the centre of the low y edge to the centre of the high y     //
// edge, and the theta phi polar angles from the centre of the low        //
// z face to the centre of the high z face.                               //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#include "TBRIK.h"

class TPARA : public TBRIK {
protected:
   Float_t fAlpha;  // angle w.r.t. the y axis from the centre of the low y to the high y
   Float_t fTheta;  // polar angle from the centre of the low z to the high z
   Float_t fPhi;    // polar angle from the centre of the low z to the high z

   virtual void    SetPoints(Double_t *points) const;

public:
   TPARA();
   TPARA(const char *name, const char *title, const char *material, Float_t dx, Float_t dy, Float_t dz,
         Float_t alpha, Float_t theta, Float_t phi);
   virtual ~TPARA();

   virtual Float_t  GetAlpha() const  {return fAlpha;}
   virtual Float_t  GetTheta() const  {return fTheta;}
   virtual Float_t  GetPhi() const    {return fPhi;}

   ClassDef(TPARA,1)  //PARA shape
};

#endif
