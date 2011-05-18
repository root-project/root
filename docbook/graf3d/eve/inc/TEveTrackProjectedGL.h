// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveTrackProjectedGL
#define ROOT_TEveTrackProjectedGL

#include "TEveTrackGL.h"

class TGLViewer;
class TGLScene;

class TEveTrackProjected;

class TEveTrackProjectedGL : public TEveTrackGL
{
private:
   TEveTrackProjectedGL(const TEveTrackProjectedGL&);            // Not implemented
   TEveTrackProjectedGL& operator=(const TEveTrackProjectedGL&); // Not implemented

protected:
   TEveTrackProjected* fM; // Model object.

public:
   TEveTrackProjectedGL();
   virtual ~TEveTrackProjectedGL() {}

   virtual Bool_t SetModel(TObject* obj, const Option_t* opt=0);
   virtual void   DirectDraw(TGLRnrCtx & rnrCtx) const;

   ClassDef(TEveTrackProjectedGL, 0); // GL-renderer for TEveTrackProjected class.
};

#endif
