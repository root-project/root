// @(#)root/gl:$Name:  $:$Id: TGLPhysicalShape.cxx,v 1.15 2005/11/22 18:05:46 brun Exp $
// Author:  Richard Maunder  25/05/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLPhysicalShape.h"
#include "TGLLogicalShape.h"
#include "TGLIncludes.h"

// For debug tracing
#include "TClass.h" 
#include "TError.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLPhysicalShape                                                     //
//                                                                      //
// Concrete physical shape - a GL drawable. Physical shapes are the     //
// objects the user can actually see, select, move in the viewer. It is //
// a placement of the associated local frame TGLLogicaShape into the    //
// world frame. The draw process is:                                    //
//                                                                      // 
// Load attributes - material colors etc                                // 
// Load translation matrix - placement                                  //
// Load gl name (for selection)                                         //
// Call our associated logical shape Draw() to draw placed shape        //
//                                                                      //
// The physical shape supports translation, scaling and rotation,       //
// selection, color changes, and permitted modification flags etc.      //
// A physical shape cannot modify or be bound to another (or no)        //
// logical shape - hence const & handle. It can perform mutable         //
// reference counting on the logical to enable purging.                 //
//                                                                      //
// See base/src/TVirtualViewer3D for description of common external 3D  //
// viewer architecture and how external viewer clients use it.          //
//////////////////////////////////////////////////////////////////////////

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
   fModified(kFALSE), 
   fManip(kManipAll)
{
   // Construct a physical shape using arguments:
   //    ID             - unique drawable id.
   //    logicalShape   - bound logical shape
   //    transform      - transform for placement of logical drawing 
   //    invertedWind   - use inverted face polygon winding?
   //    rgba           - basic four component (RGBA) diffuse color 
   fLogicalShape.AddRef();
   UpdateBoundingBox();

   // Initialise color
   InitColor(rgba);
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
   fModified(kFALSE),
   fManip(kManipAll)
{
   // Construct a physical shape using arguments:
   //    ID             - unique drawable id.
   //    logicalShape   - bound logical shape
   //    transform      - 16 Double_t component transform for placement of logical drawing 
   //    invertedWind   - use inverted face polygon winding?
   //    rgba           - basic four component (RGBA) diffuse color 
   
   fLogicalShape.AddRef();

   // Temporary hack - invert the 3x3 part of martix as TGeo sends this
   // in opp layout to shear/translation parts. Speak to Andrei about best place
   // to fix - probably when filling TBuffer3D - should always be OGL convention?
   fTransform.Transpose3x3();
   UpdateBoundingBox();

   // Initialise color
   InitColor(rgba);
}

//______________________________________________________________________________
TGLPhysicalShape::~TGLPhysicalShape()
{
   fLogicalShape.SubRef();
}

//______________________________________________________________________________
void TGLPhysicalShape::UpdateBoundingBox() 
{
   // Update our internal bounding box (in global frame)
   fBoundingBox.Set(fLogicalShape.BoundingBox());
   fBoundingBox.Transform(fTransform);
}

//______________________________________________________________________________
void TGLPhysicalShape::InitColor(const Float_t rgba[4])
{
   // Initialise the colors, using basic RGBA diffuse material color supplied
   
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

   // Modified flag not set - just initialising
}

//______________________________________________________________________________
void TGLPhysicalShape::SetColor(const Float_t color[17])
{
   // Set full color attributes - see OpenGL material documentation
   // for full description
   //
   // 0...3  - diffuse
   // 4...7  - ambient
   // 8...11 - specular
   // 12..15 - emission
   // 16     - shininess
   
   // TODO: Make a color class
   for (UInt_t i = 0; i < 17; i++) {
      fColor[i] = color[i];
   }
   fModified = kTRUE;
}

//______________________________________________________________________________
void TGLPhysicalShape::Draw(UInt_t LOD) const
{
   // Draw physical shape, using LOD flags, potential from display list cache
   
   // Debug tracing
   if (gDebug > 4) {
      Info("TGLPhysicalShape::Draw", "this %d (class %s) LOD %d", this, IsA()->GetName(), LOD);
   }

   // Setup current colors
   // TODO: Can these be moved to DirectDraw - does DL capture draw attributes
   // to investigate
   // TODO: Sorting - Min. state swap for attributes
   glColor4fv(fColor);
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
   // Draw physical shape, using LOD flags - can be captured into display list cache
   //
   // Load gl name (for selection)
   // Load translation matrix - placement                                  
   // Call associated fLogicalShape Draw() method

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
   // If LOD is pixel or less can draw pixel(point) directly, skipping
   // any logical call, caching etc
   if (LOD == kLODPixel) {
      glBegin(GL_POINTS);
      glVertex3d(0.0, 0.0, 0.0);
      glEnd();
   } else {
      fLogicalShape.Draw(LOD);
   }
   if (fInvertedWind) {
      glFrontFace(GL_CCW);
   }
   glPopMatrix();
}

//______________________________________________________________________________
void TGLPhysicalShape::DrawWireFrame(UInt_t lod) const
{
   // Draw physical shape in wireframe style
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
   // Draw physical shape in outline style
   
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
   // Request creation of context menu on shape, attached to 'menu' at screen position
   // 'x' 'y'
   
   // Just defer to our logical at present
   fLogicalShape.InvokeContextMenu(menu, x, y);
}

