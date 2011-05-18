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

   void SetTubeR(Float_t x) { fTubeR = x; }
   void SetConeR(Float_t x) { fConeR = x; }
   void SetConeL(Float_t x) { fConeL = x; }

   Float_t GetTubeR() const { return fTubeR; }
   Float_t GetConeR() const { return fConeR; }
   Float_t GetConeL() const { return fConeL; }

   TEveVector GetVector() { return fVector; }
   TEveVector GetOrigin() { return fOrigin; }

   Int_t GetDrawQuality() const  { return fDrawQuality; }
   void  SetDrawQuality(Int_t q) { fDrawQuality = q; }

   virtual void ComputeBBox();
   virtual void Paint(Option_t* option="");

   ClassDef(TEveArrow, 0); // Class for gl visualisation of arrow.
};

#endif
