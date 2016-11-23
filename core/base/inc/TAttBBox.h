// @(#)root/base:$Id$
// Author: Matevz Tadel  7/4/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAttBBox
#define ROOT_TAttBBox

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

class TAttBBox
{
protected:
   Float_t*  fBBox;   //! Dynamic Float_t[6] X(min,max), Y(min,max), Z(min,max)

   void BBoxInit(Float_t infinity=1e6);
   void BBoxZero(Float_t epsilon=0, Float_t x=0, Float_t y=0, Float_t z=0);
   void BBoxClear();

   void BBoxCheckPoint(Float_t x, Float_t y, Float_t z);
   void BBoxCheckPoint(const Float_t* p);

   void AssertBBoxExtents(Float_t epsilon=0.005);

   TAttBBox(const TAttBBox& tab) : fBBox(0) {
      BBoxInit(); if(tab.fBBox) for(Int_t i=0; i<6; i++) fBBox[i]=tab.fBBox[i];
   }

public:
   TAttBBox(): fBBox(0) { }
   virtual ~TAttBBox() { BBoxClear(); }

   TAttBBox& operator=(const TAttBBox& tab)
     {if(this!=&tab) {BBoxInit(); if(tab.fBBox) for(Int_t i=0; i<6; i++) fBBox[i]=tab.fBBox[i];}
     return *this;}

   Bool_t   GetBBoxOK() const { return fBBox != 0; }
   Float_t* GetBBox()         { return fBBox; }
   Float_t* AssertBBox()      { if(fBBox == 0) ComputeBBox(); return fBBox; }
   void     ResetBBox()       { if(fBBox != 0) BBoxClear(); }

   virtual void ComputeBBox() = 0;

   ClassDef(TAttBBox,1); // Helper for management of bounding-box information
};


// Inline methods:

inline void TAttBBox::BBoxCheckPoint(Float_t x, Float_t y, Float_t z)
{
   if(x < fBBox[0]) fBBox[0] = x;
   if(x > fBBox[1]) fBBox[1] = x;
   if(y < fBBox[2]) fBBox[2] = y;
   if(y > fBBox[3]) fBBox[3] = y;
   if(z < fBBox[4]) fBBox[4] = z;
   if(z > fBBox[5]) fBBox[5] = z;
}

inline void TAttBBox::BBoxCheckPoint(const Float_t* p)
{
   BBoxCheckPoint(p[0], p[1], p[2]);
}

#endif
