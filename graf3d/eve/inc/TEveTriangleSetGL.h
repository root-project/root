// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveTriangleSetGL
#define ROOT_TEveTriangleSetGL

#include "TGLObject.h"

class TGLRnrCtx;

class TEveTriangleSet;

class TEveTriangleSetGL : public TGLObject
{
private:
   TEveTriangleSetGL(const TEveTriangleSetGL&);            // Not implemented
   TEveTriangleSetGL& operator=(const TEveTriangleSetGL&); // Not implemented

protected:
   TEveTriangleSet* fM; // Model object.

public:
   TEveTriangleSetGL();
   virtual ~TEveTriangleSetGL();

   virtual Bool_t SetModel(TObject* obj, const Option_t *opt = nullptr);
   virtual void   SetBBox();
   virtual void   DirectDraw(TGLRnrCtx & rnrCtx) const;

   // To support two-level selection
   // virtual Bool_t SupportsSecondarySelect() const { return kTRUE; }
   // virtual void ProcessSelection(UInt_t* ptr, TGLViewer*, TGLScene*);

   ClassDef(TEveTriangleSetGL, 0); // GL-renderer for TEveTriangleSet class.
};

#endif
