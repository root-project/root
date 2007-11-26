// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveProjectionManagerGL
#define ROOT_TEveProjectionManagerGL

#include "TGLObject.h"
#include "TEveElement.h"

class TGLViewer;
class TGLScene;
class TGLText;

class TEveProjectionManager;

class TEveProjectionManagerGL : public TGLObject
{
public:
   typedef std::list<Float_t> TMList_t;

private:
   TEveProjectionManagerGL(const TEveProjectionManagerGL&);            // Not implemented
   TEveProjectionManagerGL& operator=(const TEveProjectionManagerGL&); // Not implemented

   mutable TMList_t   fPos;
   mutable TMList_t   fVals;

   mutable Float_t    fRange;
   Float_t            fLabelSize;
   Float_t            fLabelOff;
   Float_t            fTMSize;

   void               DrawTickMarks(Float_t tms) const;
   void               DrawHInfo() const;
   void               DrawVInfo() const;
   const char*        GetText(Float_t) const;

   void               SplitInterval(Int_t axis) const;
   void               SplitIntervalByPos(Float_t min, Float_t max, Int_t axis, Int_t level)const;
   void               SplitIntervalByVal(Float_t min, Float_t max, Int_t axis, Int_t level)const;

   void               SetRange(Float_t val, Int_t axis) const;

protected:
   TEveProjectionManager    *fM;    // Model object.
   TGLText                  *fText; // Text renderer for axis labels.

   virtual void DirectDraw(TGLRnrCtx & rnrCtx) const;

public:
   TEveProjectionManagerGL();
   virtual ~TEveProjectionManagerGL();

   virtual Bool_t SetModel(TObject* obj, const Option_t* opt=0);
   virtual void   SetBBox();
   Bool_t IgnoreSizeForOfInterest() const { return kTRUE;}

   ClassDef(TEveProjectionManagerGL, 0); // GL-renderer for TEveProjectionManager.
};

#endif
