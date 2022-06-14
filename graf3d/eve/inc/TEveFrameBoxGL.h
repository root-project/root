// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveFrameBoxGL
#define ROOT_TEveFrameBoxGL

#include "TEveUtil.h"

class TEveFrameBox;

class TEveFrameBoxGL
{
private:
   TEveFrameBoxGL();                             // Not implemented
   TEveFrameBoxGL(const TEveFrameBoxGL&);            // Not implemented
   TEveFrameBoxGL& operator=(const TEveFrameBoxGL&); // Not implemented

   static void RenderFrame(const TEveFrameBox& b, Bool_t fillp);

public:
   virtual ~TEveFrameBoxGL() {}

   static void Render(const TEveFrameBox* box);

   ClassDef(TEveFrameBoxGL, 0); // GL-renderer for TEveFrameBox class.
};

#endif
