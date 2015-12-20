// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveLineGL.h"
#include "TEveLine.h"

#include "TGLRnrCtx.h"
#include "TGLIncludes.h"

/** \class TEveLineGL
\ingroup TEve
GL-renderer for TEveLine class.
*/

ClassImp(TEveLineGL);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEveLineGL::TEveLineGL() : TPointSet3DGL(), fM(0)
{
   // fDLCache = false; // Disable display list.
   fMultiColor = kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Set model object.

Bool_t TEveLineGL::SetModel(TObject* obj, const Option_t* /*opt*/)
{
   fM = SetModelDynCast<TEveLine>(obj);
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Direct GL rendering for TEveLine.

void TEveLineGL::DirectDraw(TGLRnrCtx & rnrCtx) const
{
   // printf("TEveLineGL::DirectDraw Style %d, LOD %d\n", rnrCtx.Style(), rnrCtx.LOD());

   TEveLine& q = *fM;
   if (q.Size() <= 0) return;

   TGLUtil::LockColor(); // Keep color from TGLPhysicalShape.
   if (q.fRnrLine) {
      TGLCapabilityEnabler sw_smooth(GL_LINE_SMOOTH, q.fSmooth);
      TGLCapabilityEnabler sw_blend(GL_BLEND, q.fSmooth);
      TGLUtil::RenderPolyLine(q, q.GetMainTransparency(), q.GetP(), q.Size());
   }
   TGLUtil::UnlockColor();
   if (q.fRnrPoints) {
      TGLUtil::RenderPolyMarkers(q, 0,q.GetP(), q.Size(),
                                 rnrCtx.GetPickRadius(),
                                 rnrCtx.Selection());
   }
}
