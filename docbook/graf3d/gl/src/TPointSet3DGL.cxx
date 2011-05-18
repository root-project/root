// @(#)root/gl:$Id$
// Author: Matevz Tadel  7/4/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifdef WIN32
#include "Windows4root.h"
#endif

#include "TPointSet3DGL.h"
#include "TPointSet3D.h"

#include <TGLRnrCtx.h>
#include <TGLSelectRecord.h>
#include <TGLIncludes.h>

//______________________________________________________________________
//
// Direct OpenGL renderer for TPointSet3D.

ClassImp(TPointSet3DGL);

//______________________________________________________________________________
Bool_t TPointSet3DGL::SetModel(TObject* obj, const Option_t*)
{
   // Set model.

   return SetModelCheckClass(obj, TPointSet3D::Class());
}

//______________________________________________________________________________
void TPointSet3DGL::SetBBox()
{
   // Set bounding-box.

   SetAxisAlignedBBox(((TPointSet3D*)fExternalObj)->AssertBBox());
}

//______________________________________________________________________________
Bool_t TPointSet3DGL::ShouldDLCache(const TGLRnrCtx& rnrCtx) const
{
   // Override from TGLLogicalShape.
   // To account for large point-sizes we modify the projection matrix
   // during selection and thus we need a direct draw.

   if (rnrCtx.Selection())
      return kFALSE;
   return TGLObject::ShouldDLCache(rnrCtx);
}

//______________________________________________________________________________
void TPointSet3DGL::Draw(TGLRnrCtx& rnrCtx) const
{
   // Draw function for TPointSet3D. Skips line-pass of outline mode.

   if (rnrCtx.IsDrawPassOutlineLine())
      return;

   TGLObject::Draw(rnrCtx);
}

//______________________________________________________________________________
void TPointSet3DGL::DirectDraw(TGLRnrCtx& rnrCtx) const
{
   // Direct GL rendering for TPointSet3D.

   //printf("TPointSet3DGL::DirectDraw Style %d, LOD %d\n", rnrCtx.Style(), rnrCtx.LOD());
   //printf("  sel=%d, secsel=%d\n", rnrCtx.Selection(), rnrCtx.SecSelection());

   TPointSet3D& q = * (TPointSet3D*) fExternalObj;

   TGLUtil::LockColor(); // Keep color from TGLPhysicalShape.
   TGLUtil::RenderPolyMarkers(q, 0, q.GetP(), q.Size(),
                              rnrCtx.GetPickRadius(),
                              rnrCtx.Selection(),
                              rnrCtx.SecSelection());
   TGLUtil::UnlockColor();
}

//______________________________________________________________________________
void TPointSet3DGL::ProcessSelection(TGLRnrCtx& /*rnrCtx*/, TGLSelectRecord& rec)
{
   // Processes secondary selection from TGLViewer.
   // Calls TPointSet3D::PointSelected(Int_t) with index of selected
   // point as an argument.

   if (rec.GetN() < 2) return;
   TPointSet3D& q = * (TPointSet3D*) fExternalObj;
   q.PointSelected(rec.GetItem(1));
}
