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
#include "TEveGLUtil.h"
#include "TGLIncludes.h"

//______________________________________________________________________________
// TEveLineGL
//
// GL-renderer for TEveLine class.

ClassImp(TEveLineGL)

//______________________________________________________________________________
TEveLineGL::TEveLineGL() : TPointSet3DGL(), fM(0)
{
   // Constructor.

   // fDLCache = false; // Disable display list.
}

//______________________________________________________________________________
TEveLineGL::~TEveLineGL()
{}

/******************************************************************************/

//______________________________________________________________________________
Bool_t TEveLineGL::SetModel(TObject* obj, const Option_t* /*opt*/)
{
   // TPointSet3DGL::SetModel(obj);
   if(SetModelCheckClass(obj, TEveLine::Class())) {
      fM = dynamic_cast<TEveLine*>(obj);
      return kTRUE;
   }
   return kFALSE;
}

/******************************************************************************/

//______________________________________________________________________________
void TEveLineGL::DirectDraw(TGLRnrCtx & rnrCtx) const
{
   // Direct GL rendering for TEveLine.

   // printf("TEveLineGL::DirectDraw Style %d, LOD %d\n", rnrCtx.Style(), rnrCtx.LOD());

   if (rnrCtx.DrawPass() == TGLRnrCtx::kPassOutlineLine)
      return;

   TEveLine& q = *fM;
   if (q.Size() <= 0) return;

   if (q.fRnrPoints)
      TEveGLUtil::RenderPolyMarkers(q, q.GetP(), q.Size());

   if (q.fRnrLine)
      TEveGLUtil::RenderLine(q, q.GetP(), q.Size());
}
