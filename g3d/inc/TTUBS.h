// @(#)root/g3d:$Id$
// Author: Nenad Buncic   18/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTUBS
#define ROOT_TTUBS


////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TTUBS                                                                  //
//                                                                        //
// TUBS is a phi segment of a tube. It has 5 parameters, the same 3 as    //
// TUBE plus the phi limits. The segment start at first limit and         //
// includes increasing phi value up to the second limit or that plus      //
// 360 degrees.                                                           //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TTUBE
#include "TTUBE.h"
#endif

class TTUBS : public TTUBE {
protected:
   Float_t fPhi1;        // first phi limit
   Float_t fPhi2;        // second phi limit
   virtual void    MakeTableOfCoSin() const;  // Create the table of the fSiTab; fCoTab

   virtual void    SetPoints(Double_t *points) const;

public:
   TTUBS();
   TTUBS(const char *name, const char *title, const char *material, Float_t rmin, Float_t rmax, Float_t dz,
         Float_t phi1, Float_t phi2);
   TTUBS(const char *name, const char *title, const char *material, Float_t rmax, Float_t dz,
         Float_t phi1, Float_t phi2);
   virtual ~TTUBS();

   virtual Int_t   DistancetoPrimitive(Int_t px, Int_t py);
   virtual const TBuffer3D &GetBuffer3D(Int_t reqSections) const;
   virtual Float_t GetPhi1() const {return fPhi1;}
   virtual Float_t GetPhi2() const {return fPhi2;}
   virtual void    Sizeof3D() const;

   ClassDef(TTUBS,1)  //TUBS shape
};

#endif
