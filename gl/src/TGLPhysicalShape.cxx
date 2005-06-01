// @(#)root/gl:$Name:  $:$Id: TGLPhysicalShape.cxx,v 1.3 2005/05/26 12:29:50 rdm Exp $
// Author:  Richard Maunder  25/05/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// TODO: Credits to Timur for parts of this from TGLSceneObject

// TODO: Function descriptions
// TODO: Class def - same as header

#include "TGLPhysicalShape.h"
#include "TGLLogicalShape.h"
#include "TGLIncludes.h"

ClassImp(TObject)

//______________________________________________________________________________
TGLPhysicalShape::TGLPhysicalShape(ULong_t ID, const TGLLogicalShape & logicalShape,
                                   const TGLMatrix & transform, Bool_t invertedWind) :
   TGLDrawable(ID, kFALSE), // Physical shapes not DL cached by default
   fLogicalShape(logicalShape),
   fTransform(transform),
   fSelected(kFALSE),
   fInvertedWind(invertedWind)
{
   fLogicalShape.AddRef();
   fBoundingBox.Set(fLogicalShape.BoundingBox());
   fBoundingBox.Transform(fTransform);
}

//______________________________________________________________________________
TGLPhysicalShape::TGLPhysicalShape(ULong_t ID, const TGLLogicalShape & logicalShape,
                                   const Double_t * transform, Bool_t invertedWind) :
   TGLDrawable(ID, kFALSE), // Physical shapes not DL cached by default
   fLogicalShape(logicalShape),
   fTransform(transform),
   fSelected(kFALSE),
   fInvertedWind(invertedWind)
{
   fLogicalShape.AddRef();
   // Temporary hack - invert the rotation part of martix as TGeo sends this
   // in opp layout to shear/translation parts. Speak to Andrei about best place
   // to fix - probably when filling TBuffer3D - should always be OGL convention?
	fTransform.InvRot();
   UpdateBoundingBox();
}

//______________________________________________________________________________
TGLPhysicalShape::~TGLPhysicalShape()
{
   fLogicalShape.SubRef();
}

//______________________________________________________________________________
void TGLPhysicalShape::UpdateBoundingBox() 
{
   fBoundingBox.Set(fLogicalShape.BoundingBox());
   fBoundingBox.Transform(fTransform);
}

//______________________________________________________________________________
void TGLPhysicalShape::SetColor(const Float_t rgba[4])
{
   // TODO: Make a color class
   fColor[0] = rgba[0];
   fColor[1] = rgba[1];
   fColor[2] = rgba[2];
   fColor[3] = rgba[3];

   //ambient
   fColor[4] = fColor[5] = fColor[6] = 0.0f;
   //specular
   fColor[8] = fColor[9] = fColor[10] = 0.7f;
   //emission
   fColor[12] = fColor[13] = fColor[14] = 0.f;
   //alpha
   fColor[7] = fColor[11] = fColor[15] = 1.f;
   //shininess
   fColor[16] = 60.f;
}

//______________________________________________________________________________
void TGLPhysicalShape::Draw(UInt_t LOD) const
{
   // TODO: Can be moved to a one off switch when transparent draw sorting
   // back in
   if (IsTransparent()) {
      glEnable(GL_BLEND);
      glDepthMask(GL_FALSE);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

   //TODO: Sorting - Min. state swap for attributes
   glMaterialfv(GL_FRONT, GL_DIFFUSE, fColor);
   glMaterialfv(GL_FRONT, GL_AMBIENT, fColor + 4);
   glMaterialfv(GL_FRONT, GL_SPECULAR, fColor + 8);
   glMaterialfv(GL_FRONT, GL_EMISSION, fColor + 12);
   glMaterialf(GL_FRONT, GL_SHININESS, fColor[16]);

   // Do base work with potential DL caching
   TGLDrawable::Draw(LOD);

   if (IsTransparent()) {
      glDepthMask(GL_TRUE);
      glDisable(GL_BLEND);
   }

   // Selection state drawing is never cached - so outside of DirectDraw()
   if (fSelected) {
      // Selection indicated by bounding box at present
      // NOTE: This is outside of the glMultMatrixd() for the
      // physical translation as the bounding box is translated when created
      // Done at end of scene draw at present - blend/depth problem...?
      //glColor3d(1.0,1.0,1.0);
      //glDisable(GL_DEPTH_TEST);
      //fBoundingBox.Draw();
      //glEnable(GL_DEPTH_TEST);
   }
}

//______________________________________________________________________________
void TGLPhysicalShape::DirectDraw(UInt_t LOD) const
{
   glPushMatrix();
   glLoadName(ID());
   glMultMatrixd(fTransform.CArr());
   if (fInvertedWind) {
      glFrontFace(GL_CW);
   }
   fLogicalShape.Draw(LOD);
   if (fInvertedWind) {
      glFrontFace(GL_CCW);
   }
   glPopMatrix();
}

//______________________________________________________________________________
void TGLPhysicalShape::InvokeContextMenu(TContextMenu & menu, UInt_t x, UInt_t y) const
{
   // Just defer to our logical at present
   fLogicalShape.InvokeContextMenu(menu, x, y);
}

