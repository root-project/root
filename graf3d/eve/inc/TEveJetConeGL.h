// @(#)root/eve:$Id$
// Author: Matevz Tadel, Jochen Thaeder 2009

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveJetConeGL
#define ROOT_TEveJetConeGL

#include "TGLObject.h"

class TGLViewer;
class TGLScene;

class TEveJetCone;

class TEveJetConeGL : public TGLObject
{
private:
   TEveJetConeGL(const TEveJetConeGL&);            // Not implemented
   TEveJetConeGL& operator=(const TEveJetConeGL&); // Not implemented

protected:
   TEveJetCone             *fM;  // Model object.

public:
   TEveJetConeGL();
   virtual ~TEveJetConeGL() {}

   virtual Bool_t SetModel(TObject* obj, const Option_t* opt=0);
   virtual void   SetBBox();

   virtual void DirectDraw(TGLRnrCtx & rnrCtx) const;

   // To support two-level selection
   // virtual Bool_t SupportsSecondarySelect() const { return kTRUE; }
   // virtual void ProcessSelection(TGLRnrCtx & rnrCtx, TGLSelectRecord & rec);

   ClassDef(TEveJetConeGL, 0); // GL renderer class for TEveJetCone.
};

#endif
