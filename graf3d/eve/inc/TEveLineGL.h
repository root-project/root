// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveLineGL
#define ROOT_TEveLineGL

#include "TGLObject.h"
#include "TPointSet3DGL.h"

class TGLViewer;
class TGLScene;

class TEveLine;

class TEveLineGL : public TPointSet3DGL
{
private:
   TEveLineGL(const TEveLineGL&);            // Not implemented
   TEveLineGL& operator=(const TEveLineGL&); // Not implemented

protected:
   TEveLine* fM; // fModel dynamic-casted to TEveLineGL

public:
   TEveLineGL();
   virtual ~TEveLineGL() {}

   virtual Bool_t SetModel(TObject* obj, const Option_t* opt=0);
   virtual void   DirectDraw(TGLRnrCtx & rnrCtx) const;

   // To support two-level selection
   // virtual Bool_t SupportsSecondarySelect() const { return kTRUE; }
   // virtual void ProcessSelection(UInt_t* ptr, TGLViewer*, TGLScene*);

   ClassDef(TEveLineGL, 0); // GL-renderer for TEveLine class.
};

#endif
