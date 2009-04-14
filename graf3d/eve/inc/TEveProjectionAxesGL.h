// @(#)root/eve:$Id$
// Author: Alja Mrak-Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveProjectionAxesGL
#define ROOT_TEveProjectionAxesGL

#include "TGLObject.h"
#include "TGLAxisPainter.h"

class TEveProjectionAxes;
class TEveProjection;

class TEveProjectionAxesGL : public TGLObject
{
private:
   TEveProjectionAxesGL(const TEveProjectionAxesGL&);            // Not implemented
   TEveProjectionAxesGL& operator=(const TEveProjectionAxesGL&); // Not implemented

   void                 GetRange(Int_t ax, Float_t frustMin, Float_t frustMax, Float_t& start, Float_t& en) const;
   void                 SplitInterval(Float_t x1, Float_t x2, Int_t axis) const;
   void                 SplitIntervalByPos(Float_t min, Float_t max, Int_t axis)const;
   void                 SplitIntervalByVal(Float_t min, Float_t max, Int_t axis)const;
   void                 FilterOverlappingLabels(Int_t idx, Float_t ref) const;
protected:
   TEveProjectionAxes     *fM;          // Model object.
   mutable TEveProjection *fProjection; // Cached model projection
   mutable TGLAxisPainter  fAxisPainter;

public:
   TEveProjectionAxesGL();
   virtual ~TEveProjectionAxesGL() {}

   virtual Bool_t  SetModel(TObject* obj, const Option_t* opt = 0);
   virtual void    SetBBox();
   virtual void    Draw(TGLRnrCtx& rnrCtx) const;
   virtual void    DirectDraw(TGLRnrCtx & rnrCtx) const;

   Bool_t IgnoreSizeForOfInterest() const { return kTRUE; }

   ClassDef(TEveProjectionAxesGL, 0); // GL renderer class for TEveProjectionAxes.
};

#endif
