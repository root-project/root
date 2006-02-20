// @(#)root/gl:$Name:  $:$Id: TGLSphere.cxx $
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
#include "TGLDrawFlags.h"
#include "TGLIncludes.h"

#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"

// For debug tracing
#include "TClass.h" 
#include "TError.h"

ClassImp(TGLSphere)

//______________________________________________________________________________
TGLSphere::TGLSphere(const TBuffer3DSphere &buffer) :
   TGLLogicalShape(buffer)
{
   // Default ctor
   fRadius = buffer.fRadiusOuter;

   // TODO:
   // Support hollow & cut spheres
   // buffer.fRadiusInner;
   // buffer.fThetaMin;
   // buffer.fThetaMax;
   // buffer.fPhiMin;
   // buffer.fPhiMax;
}

//______________________________________________________________________________
void TGLSphere::DirectDraw(const TGLDrawFlags & flags) const
{
   // Debug tracing
   if (gDebug > 4) {
      Info("TGLSphere::DirectDraw", "this %d (class %s) LOD %d", this, IsA()->GetName(), flags.LOD());
   }

   // 4 stack/slice min for gluSphere to work
   UInt_t divisions = flags.LOD();
   if (divisions < 4) {
      divisions = 4;
   }
   gluSphere(fgQuad.Get(),fRadius, divisions, divisions);
}
