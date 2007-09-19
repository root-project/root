// @(#)root/g3d:$Id$
// Author: Nenad Buncic   17/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBRIK
#define ROOT_TBRIK


////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TBRIK                                                                  //
//                                                                        //
// BRIK is a box. It has 3 parameters, the half length in x, y, and z     //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TShape
#include "TShape.h"
#endif

class TBRIK : public TShape {

protected:
   Float_t fDx;        // half length in x
   Float_t fDy;        // half length in y
   Float_t fDz;        // half length in z

   virtual void    SetPoints(Double_t * points) const;
public:
   TBRIK();
   TBRIK(const char *name, const char *title, const char *material, Float_t dx, Float_t dy, Float_t dz);
   virtual ~TBRIK();

   virtual Int_t   DistancetoPrimitive(Int_t px, Int_t py);
   virtual const TBuffer3D &GetBuffer3D(Int_t reqSections) const;
   Float_t         GetDx() const {return fDx;}
   Float_t         GetDy() const {return fDy;}
   Float_t         GetDz() const {return fDz;}
   virtual void    Sizeof3D() const;

   ClassDef(TBRIK,1)  //TBRIK shape
};

#endif
