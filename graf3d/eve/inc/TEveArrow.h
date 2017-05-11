// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveArrow
#define ROOT_TEveArrow

#include "TEveElement.h"
#include "TEveVector.h"
#include "TNamed.h"
#include "TAtt3D.h"
#include "TAttBBox.h"

class TEveArrow : public TEveElement,
                  public TNamed,
                  public TAtt3D,
                  public TAttBBox
{
   friend class TEveArrowGL;
   friend class TEveArrowEditor;

private:
   TEveArrow(const TEveArrow&);            // Not implemented
   TEveArrow& operator=(const TEveArrow&); // Not implemented

protected:
   Color_t     fColor;

   Float_t     fTubeR;
   Float_t     fConeR;
   Float_t     fConeL;

   TEveVector  fOrigin;
   TEveVector  fVector;

   Int_t       fDrawQuality; // Number of segments of circles.

public:
   TEveArrow(Float_t xVec=0, Float_t yVec=0, Float_t zVec=1,
             Float_t xOrg=0, Float_t yOrg=0, Float_t zOrg=0);
   virtual ~TEveArrow() {}

   virtual TObject* GetObject(const TEveException& ) const
   { const TObject* obj = this; return const_cast<TObject*>(obj); }

   void    StampGeom() { ResetBBox(); AddStamp(kCBTransBBox | kCBObjProps); }

   Float_t GetTubeR() const { return fTubeR; }
   Float_t GetConeR() const { return fConeR; }
   Float_t GetConeL() const { return fConeL; }

   void SetTubeR(Float_t x) { fTubeR = x; StampGeom(); }
   void SetConeR(Float_t x) { fConeR = x; StampGeom(); }
   void SetConeL(Float_t x) { fConeL = x; StampGeom(); }

   TEveVector  GetOrigin() { return fOrigin; }
   TEveVector& RefOrigin() { return fOrigin; }
   TEveVector  GetVector() { return fVector; }
   TEveVector& RefVector() { return fVector; }

   void SetOrigin(const TEveVector& o)             { fOrigin = o;          StampGeom(); }
   void SetOrigin(Float_t x, Float_t y, Float_t z) { fOrigin.Set(x, y, z); StampGeom(); }
   void SetVector(const TEveVector& v)             { fVector = v;          StampGeom(); }
   void SetVector(Float_t x, Float_t y, Float_t z) { fVector.Set(x, y, z); StampGeom(); }

   Int_t GetDrawQuality() const  { return fDrawQuality; }
   void  SetDrawQuality(Int_t q) { fDrawQuality = q;    }

   virtual void ComputeBBox();
   virtual void Paint(Option_t* option="");

   ClassDef(TEveArrow, 0); // Class for gl visualisation of arrow.
};

#endif
