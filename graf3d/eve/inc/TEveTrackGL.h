// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveTrackGL
#define ROOT_TEveTrackGL

#include "TEveLineGL.h"

class TGLViewer;
class TGLScene;

class TEveTrack;

class TEveTrackGL : public TEveLineGL
{
private:
   TEveTrackGL(const TEveTrackGL&);            // Not implemented
   TEveTrackGL& operator=(const TEveTrackGL&); // Not implemented

protected:
   TEveTrack* fTrack; // Model object.

   void RenderPathMarksAndFirstVertex(TGLRnrCtx& rnrCtx) const;

public:
   TEveTrackGL();
   ~TEveTrackGL() override {}

   Bool_t SetModel(TObject* obj, const Option_t *opt = nullptr) override;
   void   DirectDraw(TGLRnrCtx & rnrCtx) const override;

   Bool_t SupportsSecondarySelect() const override { return kTRUE; }
   void   ProcessSelection(TGLRnrCtx& rnrCtx, TGLSelectRecord& rec) override;

   ClassDefOverride(TEveTrackGL, 0); // GL-renderer for TEveTrack class.
};

#endif
