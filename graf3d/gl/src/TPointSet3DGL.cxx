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

/** \class TPointSet3DGL
\ingroup opengl
Direct OpenGL renderer for TPointSet3D.
*/

ClassImp(TPointSet3DGL);

////////////////////////////////////////////////////////////////////////////////
/// Set model.

Bool_t TPointSet3DGL::SetModel(TObject* obj, const Option_t*)
{
   return SetModelCheckClass(obj, TPointSet3D::Class());
}

////////////////////////////////////////////////////////////////////////////////
/// Set bounding-box.

void TPointSet3DGL::SetBBox()
{
   SetAxisAlignedBBox(((TPointSet3D*)fExternalObj)->AssertBBox());
}

////////////////////////////////////////////////////////////////////////////////
/// Override from TGLLogicalShape.
/// To account for large point-sizes we modify the projection matrix
/// during selection and thus we need a direct draw.

Bool_t TPointSet3DGL::ShouldDLCache(const TGLRnrCtx& rnrCtx) const
{
   if (rnrCtx.Selection())
      return kFALSE;
   return TGLObject::ShouldDLCache(rnrCtx);
}

////////////////////////////////////////////////////////////////////////////////
/// Draw function for TPointSet3D. Skips line-pass of outline mode.

void TPointSet3DGL::Draw(TGLRnrCtx& rnrCtx) const
{
   if (rnrCtx.IsDrawPassOutlineLine())
      return;

   TGLObject::Draw(rnrCtx);
}

////////////////////////////////////////////////////////////////////////////////
/// Direct GL rendering for TPointSet3D.

void TPointSet3DGL::DirectDraw(TGLRnrCtx& rnrCtx) const
{
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

////////////////////////////////////////////////////////////////////////////////
/// Processes secondary selection from TGLViewer.
/// Calls TPointSet3D::PointSelected(Int_t) with index of selected
/// point as an argument.

void TPointSet3DGL::ProcessSelection(TGLRnrCtx& /*rnrCtx*/, TGLSelectRecord& rec)
{
   if (rec.GetN() < 2) return;
   TPointSet3D& q = * (TPointSet3D*) fExternalObj;
   q.PointSelected(rec.GetItem(1));
}
