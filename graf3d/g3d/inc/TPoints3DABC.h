// @(#)root/g3d:$Id$
// Author: Valery Fine(fine@mail.cern.ch)   24/04/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TPoints3DABC
#define ROOT_TPoints3DABC


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPoints3DABC                                                         //
//                                                                      //
// Abstract class to define Arrays of 3D points                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TObject.h"

class TPoints3DABC : public TObject {

public:
   TPoints3DABC(){;}
   virtual ~TPoints3DABC(){;}

   static  Int_t     DistancetoLine(Int_t px, Int_t py, Float_t x1, Float_t y1, Float_t x2, Float_t y2, Int_t lineWidth = 1 );

   virtual Int_t     Add(Float_t x, Float_t y, Float_t z);
   virtual Int_t     AddLast(Float_t x, Float_t y, Float_t z);
   virtual Int_t     DistancetoPrimitive(Int_t px, Int_t py)=0;
   virtual Int_t     GetLastPosition()const    =0;
   // GetN()  returns the number of allocated cells if any.
   //         GetN() > 0 shows how many cells
   //         can be available via GetP() method.
   //         GetN() == 0 then GetP() must return 0 as well
   virtual Int_t     GetN() const;
   virtual Float_t  *GetP() const;
   virtual Float_t   GetX(Int_t idx)  const    =0;
   virtual Float_t   GetY(Int_t idx)  const    =0;
   virtual Float_t   GetZ(Int_t idx)  const    =0;
   virtual Float_t  *GetXYZ(Float_t *xyz,Int_t idx,Int_t num=1)  const;
   virtual const Float_t  *GetXYZ(Int_t idx)   =0;
   virtual Option_t *GetOption()      const    =0;
   virtual void      PaintPoints(Int_t n, Float_t *p,Option_t *option="")  =0;
   virtual Int_t     SetLastPosition(Int_t idx)=0;
   virtual Int_t     SetNextPoint(Float_t x, Float_t y, Float_t z);
   virtual void      SetOption(Option_t *option="")=0;
   virtual Int_t     SetPoint(Int_t point, Float_t x, Float_t y, Float_t z)=0;
   virtual Int_t     SetPoints(Int_t n, Float_t *p=0, Option_t *option="") =0;
   virtual Int_t     Size() const              =0;

   ClassDef(TPoints3DABC,0)  //A 3-D Points
};

#endif
