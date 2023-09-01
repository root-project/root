// @(#)root/gl:$Id$
// Author:  Timur Pocheptsov  03/08/2004
// NOTE: This code moved from obsoleted TGLSceneObject.h / .cxx - see these
// attic files for previous CVS history

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#include "TGLSphere.h"
#include "TGLRnrCtx.h"
#include "TGLQuadric.h"
#include "TGLIncludes.h"

#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"

// For debug tracing
#include "TClass.h"
#include "TError.h"

/** \class TGLSphere
\ingroup opengl
Implements a native ROOT-GL sphere that can be rendered at
different levels of detail.
*/

ClassImp(TGLSphere);

////////////////////////////////////////////////////////////////////////////////
/// Default ctor

TGLSphere::TGLSphere(const TBuffer3DSphere &buffer) :
   TGLLogicalShape(buffer)
{
   fDLSize = 14;

   fRadius = buffer.fRadiusOuter;

   // TODO:
   // Support hollow & cut spheres
   // buffer.fRadiusInner;
   // buffer.fThetaMin;
   // buffer.fThetaMax;
   // buffer.fPhiMin;
   // buffer.fPhiMax;
}

////////////////////////////////////////////////////////////////////////////////
/// Return display-list offset for given LOD.
/// Calculation based on what is done in virtual QuantizeShapeLOD below.

UInt_t TGLSphere::DLOffset(Short_t lod) const
{
   UInt_t  off = 0;
   if      (lod >= 100) off = 0;
   else if (lod <  10)  off = lod / 2;
   else                 off = lod / 10 + 4;
   return off;
}

////////////////////////////////////////////////////////////////////////////////
/// Factor in scene/viewer LOD and quantize.

Short_t TGLSphere::QuantizeShapeLOD(Short_t shapeLOD, Short_t combiLOD) const
{
   Int_t lod = ((Int_t)shapeLOD * (Int_t)combiLOD) / 100;

   if (lod >= 100)
   {
      lod = 100;
   }
   else if (lod > 10)
   {  // Round LOD above 10 to nearest 10
      Double_t quant = 0.1 * ((static_cast<Double_t>(lod)) + 0.5);
      lod            = 10  *   static_cast<Int_t>(quant);
   }
   else
   {  // Round LOD below 10 to nearest 2
      Double_t quant = 0.5 * ((static_cast<Double_t>(lod)) + 0.5);
      lod            = 2   *   static_cast<Int_t>(quant);
   }
   return static_cast<Short_t>(lod);
}

////////////////////////////////////////////////////////////////////////////////
/// Debug tracing

void TGLSphere::DirectDraw(TGLRnrCtx & rnrCtx) const
{
   if (gDebug > 4) {
      Info("TGLSphere::DirectDraw", "this %zd (class %s) LOD %d", (size_t)this, IsA()->GetName(), rnrCtx.ShapeLOD());
   }

   // 4 stack/slice min for gluSphere to work
   UInt_t divisions = rnrCtx.ShapeLOD();
   if (divisions < 4) {
      divisions = 4;
   }
   gluSphere(rnrCtx.GetGluQuadric(), fRadius, divisions, divisions);
}
