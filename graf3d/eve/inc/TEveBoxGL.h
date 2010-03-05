// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveBoxGL
#define ROOT_TEveBoxGL

#include "TGLObject.h"

class TGLViewer;
class TGLScene;

class TEveBox;

class TEveBoxGL : public TGLObject
{
private:
   TEveBoxGL(const TEveBoxGL&);            // Not implemented
   TEveBoxGL& operator=(const TEveBoxGL&); // Not implemented

protected:
   TEveBox             *fM;  // Model object.

   void RenderOutline    (const Float_t p[8][3]) const;
   void RenderBoxStdNorm (const Float_t p[8][3]) const;
   void RenderBoxAutoNorm(const Float_t p[8][3]) const;

public:
   TEveBoxGL();
   virtual ~TEveBoxGL() {}

   virtual Bool_t SetModel(TObject* obj, const Option_t* opt=0);
   virtual void   SetBBox();

   virtual void Draw(TGLRnrCtx& rnrCtx) const;
   virtual void DirectDraw(TGLRnrCtx& rnrCtx) const;

   // To support two-level selection
   // virtual Bool_t SupportsSecondarySelect() const { return kTRUE; }
   // virtual void ProcessSelection(TGLRnrCtx & rnrCtx, TGLSelectRecord & rec);

   ClassDef(TEveBoxGL, 0); // GL renderer class for TEveBox.
};

#endif
