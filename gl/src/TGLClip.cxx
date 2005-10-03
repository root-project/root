// @(#)root/gl:$Name:  $:$Id: TGLClip.cxx
// Author:  Richard Maunder  16/09/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLClip.h"
#include "TGLIncludes.h"

ClassImp(TGLClip)

//______________________________________________________________________________
TGLClip::TGLClip() : fMode(kInside)
{
}

//______________________________________________________________________________
TGLClip::~TGLClip()
{
}

ClassImp(TGLClipPlane)

//______________________________________________________________________________
TGLClipPlane::TGLClipPlane(const TGLPlane &  plane) : fPlane(plane)
{
}

//______________________________________________________________________________
TGLClipPlane::~TGLClipPlane()
{
}

//______________________________________________________________________________
void TGLClipPlane::Set(const TGLPlane & plane)
{
   fPlane = plane;
}

//______________________________________________________________________________
void TGLClipPlane::Draw(UInt_t /*LOD*/) const
{
   // Not drawn at present
}

//______________________________________________________________________________
void TGLClipPlane::PlaneSet(TGLPlaneSet_t & set) const
{
   set.push_back(fPlane);
}

ClassImp(TGLClipShape)

float TGLClipShape::fgColor[4] = { 0.0, 0.5, 0.5, 0.3 };

//______________________________________________________________________________
TGLClipShape::TGLClipShape(const TGLLogicalShape & logicalShape, const TGLMatrix & transform) :
   TGLPhysicalShape(0, logicalShape, transform, kTRUE, fgColor)
{
   logicalShape.StrongRef(kTRUE);
}

//______________________________________________________________________________
TGLClipShape::~TGLClipShape()
{
}

//______________________________________________________________________________
void TGLClipShape::Draw(UInt_t LOD) const
{
   glDepthMask(GL_FALSE);
   glEnable(GL_BLEND);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   TGLPhysicalShape::Draw(LOD);
   glDepthMask(GL_TRUE);
   glDisable(GL_BLEND);
}
