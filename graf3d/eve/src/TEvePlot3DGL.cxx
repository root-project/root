// @(#)root/eve:$Id$
// Author: Matevz Tadel, 2009

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEvePlot3DGL.h"
#include "TEvePlot3D.h"
#include "TGLPlot3D.h"

#include "TGLRnrCtx.h"
#include "TGLIncludes.h"

/** \class TEvePlot3DGL
\ingroup TEve
OpenGL renderer class for TEvePlot3D.
*/

ClassImp(TEvePlot3DGL);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEvePlot3DGL::TEvePlot3DGL() :
   TGLObject(), fM(0), fPlotLogical(0)
{
   fDLCache = kFALSE; // Disable display list.
}

////////////////////////////////////////////////////////////////////////////////
/// Set model object.

Bool_t TEvePlot3DGL::SetModel(TObject* obj, const Option_t* /*opt*/)
{
   fM = SetModelDynCast<TEvePlot3D>(obj);
   fPlotLogical = TGLPlot3D::CreatePlot(fM->fPlot, fM->fPlotOption, fM->fLogX, fM->fLogY, fM->fLogZ);
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Set bounding box.

void TEvePlot3DGL::SetBBox()
{
   // !! This ok if master sub-classed from TAttBBox
   //SetAxisAlignedBBox(((TEvePlot3D*)fExternalObj)->AssertBBox());
   fBoundingBox = fPlotLogical->BoundingBox();
}

////////////////////////////////////////////////////////////////////////////////
/// Render with OpenGL.

void TEvePlot3DGL::DirectDraw(TGLRnrCtx & rnrCtx) const
{
   // printf("TEvePlot3DGL::DirectDraw LOD %d\n", rnrCtx.CombiLOD());
   if (fPlotLogical)
   {
      fPlotLogical->DirectDraw(rnrCtx);
   }
}
