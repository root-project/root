// @(#)root/gl:$Name:  $:$Id: TGLPhysicalShape.cxx,v 1.9 2005/06/23 15:08:45 brun Exp $
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

// For debug tracing
#include "TClass.h" 
#include "TError.h"

ClassImp(TGLPhysicalShape)

//______________________________________________________________________________
TGLPhysicalShape::TGLPhysicalShape(ULong_t ID, const TGLLogicalShape & logicalShape,
                                   const TGLMatrix & transform, Bool_t invertedWind,
                                   const Float_t rgba[4]) :
   TGLDrawable(ID, kFALSE), // Physical shapes not DL cached by default
   fLogicalShape(logicalShape),
   fTransform(transform),
   fSelected(kFALSE),
   fInvertedWind(invertedWind),
   fModified(kFALSE)
{
   fLogicalShape.AddRef();
   UpdateBoundingBox();

   // Set color and reset modified flag as we are just initialising
   SetColor(rgba);
   fModified = kFALSE;
}

//______________________________________________________________________________
TGLPhysicalShape::TGLPhysicalShape(ULong_t ID, const TGLLogicalShape & logicalShape,
                                   const Double_t * transform, Bool_t invertedWind,
                                   const Float_t rgba[4]) :
   TGLDrawable(ID, kFALSE), // Physical shapes not DL cached by default
   fLogicalShape(logicalShape),
   fTransform(transform),
   fSelected(kFALSE),
   fInvertedWind(invertedWind),
   fModified(kFALSE)
{
   fLogicalShape.AddRef();
   // Temporary hack - invert the 3x3 part of martix as TGeo sends this
   // in opp layout to shear/translation parts. Speak to Andrei about best place
   // to fix - probably when filling TBuffer3D - should always be OGL convention?
   fTransform.Transpose3x3();
   UpdateBoundingBox();

   // Set color and reset modified flag as we are just initialising
   SetColor(rgba);
   fModified = kFALSE;
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

   fModified = kTRUE;
}

//______________________________________________________________________________
void TGLPhysicalShape::Draw(UInt_t LOD) const
{
   // Debug tracing
   if (gDebug > 4) {
      Info("TGLPhysicalShape::Draw", "this %d (class %s) LOD %d", this, IsA()->GetName(), LOD);
   }

   //TODO: Sorting - Min. state swap for attributes
   glMaterialfv(GL_FRONT, GL_DIFFUSE, fColor);
   glMaterialfv(GL_FRONT, GL_AMBIENT, fColor + 4);
   glMaterialfv(GL_FRONT, GL_SPECULAR, fColor + 8);
   glMaterialfv(GL_FRONT, GL_EMISSION, fColor + 12);
   glMaterialf(GL_FRONT, GL_SHININESS, fColor[16]);

   // Do base work with potential DL caching
   TGLDrawable::Draw(LOD);
}

//______________________________________________________________________________
void TGLPhysicalShape::DirectDraw(UInt_t LOD) const
{
   // Debug tracing
   if (gDebug > 4) {
      Info("TGLPhysicalShape::DirectDraw", "this %d (class %s) LOD %d", this, IsA()->GetName(), LOD);
   }

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
void TGLPhysicalShape::DrawWireFrame(UInt_t lod) const
{
   glPushMatrix();
   glLoadName(ID());
   glMultMatrixd(fTransform.CArr());
   
   glColor4fv(fColor);

   if (fInvertedWind) glFrontFace(GL_CW);
   fLogicalShape.DrawWireFrame(lod);
   if (fInvertedWind) glFrontFace(GL_CCW);
   
   glPopMatrix();
}

//______________________________________________________________________________
void TGLPhysicalShape::DrawOutline(UInt_t LOD) const
{
   // Debug tracing
   if (gDebug > 4) {
      Info("TGLPhysicalShape::Draw", "this %d (class %s) LOD %d", this, IsA()->GetName(), LOD);
   }

   glPushMatrix();
   glLoadName(ID());
   glMultMatrixd(fTransform.CArr());
   
   //TODO: Sorting - Min. state swap for attributes
   glMaterialfv(GL_FRONT, GL_DIFFUSE, fColor);
   glMaterialfv(GL_FRONT, GL_AMBIENT, fColor + 4);
   glMaterialfv(GL_FRONT, GL_SPECULAR, fColor + 8);
   glMaterialfv(GL_FRONT, GL_EMISSION, fColor + 12);
   glMaterialf(GL_FRONT, GL_SHININESS, fColor[16]);

   if (fInvertedWind) glFrontFace(GL_CW);
   fLogicalShape.DrawOutline(LOD);
   if (fInvertedWind) glFrontFace(GL_CCW);
   
   glPopMatrix();
}

//______________________________________________________________________________
void TGLPhysicalShape::InvokeContextMenu(TContextMenu & menu, UInt_t x, UInt_t y) const
{
   // Just defer to our logical at present
   fLogicalShape.InvokeContextMenu(menu, x, y);
}

