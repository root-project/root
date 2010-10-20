// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveArrowGL.h"
#include "TEveArrow.h"

#include "TGLRnrCtx.h"
#include "TGLIncludes.h"
#include "TGLUtil.h"
#include "TGLQuadric.h"

//______________________________________________________________________________
// OpenGL renderer class for TEveArrow.
//

ClassImp(TEveArrowGL);

//______________________________________________________________________________
TEveArrowGL::TEveArrowGL() :
   TGLObject(), fM(0)
{
   // Constructor.
}

/******************************************************************************/

//______________________________________________________________________________
Bool_t TEveArrowGL::SetModel(TObject* obj, const Option_t* /*opt*/)
{
   // Set model object.

   fM = SetModelDynCast<TEveArrow>(obj);
   return kTRUE;
}

//______________________________________________________________________________
void TEveArrowGL::SetBBox()
{
   // Set bounding box.

   // !! This ok if master sub-classed from TAttBBox
   SetAxisAlignedBBox(((TEveArrow*)fExternalObj)->AssertBBox());
}

/******************************************************************************/

//______________________________________________________________________________
void TEveArrowGL::DirectDraw(TGLRnrCtx& /*rnrCtx*/) const
{
   // Render with OpenGL.

   static TGLQuadric quad;

   glPushMatrix();

   TGLVertex3 uo(fM->fOrigin.fX, fM->fOrigin.fY, fM->fOrigin.fZ);
   TGLVector3 uv(fM->fVector.fX, fM->fVector.fY, fM->fVector.fZ);
   TGLMatrix local(uo, uv);
   glMultMatrixd(local.CArr());
   Float_t size = fM->fVector.Mag();

   // Line (tube) component
   Float_t r = size*fM->fTubeR;
   Float_t h = size*fM->fConeL;
   gluCylinder(quad.Get(), r, r, size - h, fM->fDrawQuality, 1);
   gluQuadricOrientation(quad.Get(), (GLenum)GLU_INSIDE);
   gluDisk(quad.Get(), 0.0, r, fM->fDrawQuality, 1);

   // Arrow cone
   r = size*fM->fConeR;
   glTranslated(0.0, 0.0, size -h );
   gluDisk(quad.Get(), 0.0, r, fM->fDrawQuality, 1);
   gluQuadricOrientation(quad.Get(), (GLenum)GLU_OUTSIDE);
   gluCylinder(quad.Get(), r, 0., h , fM->fDrawQuality, 1);

   glPopMatrix();
}
