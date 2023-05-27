// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveDigitSetGL
#define ROOT_TEveDigitSetGL

#include "TGLObject.h"
#include "TEveDigitSet.h"

#include <set>

class TGLViewer;
class TGLScene;

class TEveDigitSet;

class TEveDigitSetGL : public TGLObject
{
private:
   TEveDigitSetGL(const TEveDigitSetGL&);            // Not implemented
   TEveDigitSetGL& operator=(const TEveDigitSetGL&); // Not implemented

protected:
   mutable const std::set<Int_t> *fHighlightSet;

   Bool_t SetupColor(const TEveDigitSet::DigitBase_t& q) const;
   void   DrawFrameIfNeeded(TGLRnrCtx& rnrCtx) const;

public:
   TEveDigitSetGL();
   ~TEveDigitSetGL() override {}

   void   SetBBox() override;

   void   DrawHighlight(TGLRnrCtx& rnrCtx, const TGLPhysicalShape* pshp, Int_t lvl=-1) const override;

   Bool_t SupportsSecondarySelect() const override { return kTRUE; }
   Bool_t AlwaysSecondarySelect()   const override { return ((TEveDigitSet*)fExternalObj)->GetAlwaysSecSelect(); }
   void   ProcessSelection(TGLRnrCtx& rnrCtx, TGLSelectRecord& rec) override;

   ClassDefOverride(TEveDigitSetGL, 0); // GL renderer class for TEveDigitSet.
};

#endif
