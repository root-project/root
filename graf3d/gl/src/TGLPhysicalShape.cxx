// @(#)root/gl:$Id$
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
#include "TGLPShapeRef.h"
#include "TGLCamera.h"
#include "TGLRnrCtx.h"
#include "TGLIncludes.h"

#include "TGLScene.h"

#include "TColor.h"
#include "TROOT.h"

#include <cmath>

// For debug tracing
#include "TClass.h"
#include "TError.h"

/******************************************************************************/
// TGLPhysicalShape
/******************************************************************************/

//______________________________________________________________________________
//
// Concrete physical shape - a GL drawable. Physical shapes are the
// objects the user can actually see, select, move in the viewer. It is
// a placement of the associated local frame TGLLogicaShape into the
// world frame. The draw process is:
//
// Load attributes - material colors etc
// Load translation matrix - placement
// Load gl name (for selection)
// Call our associated logical shape Draw() to draw placed shape
//
// The physical shape supports translation, scaling and rotation,
// selection, color changes, and permitted modification flags etc.
// A physical shape cannot modify or be bound to another (or no)
// logical shape - hence const & handle. It can perform mutable
// reference counting on the logical to enable purging.
//
// Physical shape also maintains a list of references to it and
// provides notifications of change and destruction.
// See class TGLPShapeRef which needs to be sub-classes for real use.
//
// See base/src/TVirtualViewer3D for description of common external 3D
// viewer architecture and how external viewer clients use it.
//

ClassImp(TGLPhysicalShape)

//______________________________________________________________________________
TGLPhysicalShape::TGLPhysicalShape(UInt_t id, const TGLLogicalShape & logicalShape,
                                   const TGLMatrix & transform, Bool_t invertedWind,
                                   const Float_t rgba[4]) :
   fLogicalShape (&logicalShape),
   fNextPhysical (0),
   fFirstPSRef   (0),
   fID           (id),
   fTransform    (transform),
   fManip        (kManipAll),
   fSelected     (0),
   fInvertedWind (invertedWind),
   fModified     (kFALSE),
   fIsScaleForRnr(kFALSE)
{
   // Construct a physical shape using arguments:
   //    ID             - unique drawable id.
   //    logicalShape   - bound logical shape
   //    transform      - transform for placement of logical drawing
   //    invertedWind   - use inverted face polygon winding?
   //    rgba           - basic four component (RGBA) diffuse color

   fLogicalShape->AddRef(this);
   UpdateBoundingBox();

   // Initialise color
   InitColor(rgba);
}

//______________________________________________________________________________
TGLPhysicalShape::TGLPhysicalShape(UInt_t id, const TGLLogicalShape & logicalShape,
                                   const Double_t * transform, Bool_t invertedWind,
                                   const Float_t rgba[4]) :
   fLogicalShape (&logicalShape),
   fNextPhysical (0),
   fFirstPSRef   (0),
   fID           (id),
   fTransform    (transform),
   fManip        (kManipAll),
   fSelected     (0),
   fInvertedWind (invertedWind),
   fModified     (kFALSE),
   fIsScaleForRnr(kFALSE)
{
   // Construct a physical shape using arguments:
   //    id             - unique drawable id.
   //    logicalShape   - bound logical shape
   //    transform      - 16 Double_t component transform for placement of logical drawing
   //    invertedWind   - use inverted face polygon winding?
   //    rgba           - basic four component (RGBA) diffuse color

   fLogicalShape->AddRef(this);

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
   // Destroy the physical shape.

   // If destroyed from the logical shape itself the pointer has already
   // been cleared.
   if (fLogicalShape) fLogicalShape->SubRef(this);

   // Remove all references.
   while (fFirstPSRef) {
      fFirstPSRef->SetPShape(0);
   }
}

//______________________________________________________________________________
void TGLPhysicalShape::AddReference(TGLPShapeRef* ref)
{
   // Add reference ref.

   assert(ref != 0);

   ref->fNextPSRef = fFirstPSRef;
   fFirstPSRef = ref;
}

//______________________________________________________________________________
void TGLPhysicalShape::RemoveReference(TGLPShapeRef* ref)
{
   // Remove reference ref.

   assert(ref != 0);

   Bool_t found = kFALSE;
   if (fFirstPSRef == ref) {
      fFirstPSRef = ref->fNextPSRef;
      found = kTRUE;
   } else {
      TGLPShapeRef *shp1 = fFirstPSRef, *shp2;
      while ((shp2 = shp1->fNextPSRef) != 0) {
         if (shp2 == ref) {
            shp1->fNextPSRef = shp2->fNextPSRef;
            found = kTRUE;
            break;
         }
         shp1 = shp2;
      }
   }
   if (found) {
      ref->fNextPSRef = 0;
   } else {
      Error("TGLPhysicalShape::RemoveReference", "Attempt to un-ref an unregistered shape-ref.");
   }
}

//______________________________________________________________________________
void TGLPhysicalShape::Modified()
{
   // Call this after modifying the physical so that the information
   // can be propagated to the object referencing it.

   fModified = kTRUE;
   TGLPShapeRef * ref = fFirstPSRef;
   while (ref) {
      ref->PShapeModified();
      ref = ref->fNextPSRef;
   }
}

//______________________________________________________________________________
void TGLPhysicalShape::UpdateBoundingBox()
{
   // Update our internal bounding box (in global frame).

   fBoundingBox.Set(fLogicalShape->BoundingBox());
   fBoundingBox.Transform(fTransform);

   fIsScaleForRnr = fTransform.IsScalingForRender();

   if (fLogicalShape->GetScene())
      fLogicalShape->GetScene()->InvalidateBoundingBox();
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

   fColor[4]  = fColor[5]  = fColor[6]  = 0.0f; //ambient
   fColor[8]  = fColor[9]  = fColor[10] = 0.7f; //specular
   fColor[12] = fColor[13] = fColor[14] = 0.0f; //emission
   fColor[7]  = fColor[11] = fColor[15] = 1.0f; //alpha
   fColor[16] = 60.0f;                          //shininess
}

//______________________________________________________________________________
void TGLPhysicalShape::SetColor(const Float_t color[17])
{
   // Set full color attributes - see OpenGL material documentation
   // for full description.
   // 0->3 diffuse, 4->7 ambient, 8->11 specular, 12->15 emission, 16 shininess

   // TODO: Make a color class
   for (UInt_t i = 0; i < 17; i++) {
      fColor[i] = color[i];
   }

   Modified();
}

//______________________________________________________________________________
void TGLPhysicalShape::SetColorOnFamily(const Float_t color[17])
{
   // Set full color attributes to all physicals sharing the same
   // logical with this object.

   TGLPhysicalShape* pshp = const_cast<TGLPhysicalShape*>(fLogicalShape->GetFirstPhysical());
   while (pshp)
   {
      pshp->SetColor(color);
      pshp = pshp->fNextPhysical;
   }
}

//______________________________________________________________________________
void TGLPhysicalShape::SetDiffuseColor(const Float_t rgba[4])
{
   // Set color from ROOT color index and transparency [0,100].

   for (Int_t i=0; i<4; ++i)
      fColor[i] = rgba[i];
   Modified();
}

//______________________________________________________________________________
void TGLPhysicalShape::SetDiffuseColor(const UChar_t rgba[4])
{
   // Set color from RGBA quadruplet.

   for (Int_t i=0; i<4; ++i)
      fColor[i] = rgba[i]/255.0f;
   Modified();
}

//______________________________________________________________________________
void TGLPhysicalShape::SetDiffuseColor(Color_t ci, UChar_t transparency)
{
   // Set color from standard ROOT representation, that is color index
   // + transparency in range [0, 100].

   if (ci < 0) ci = 1;
   TColor* c = gROOT->GetColor(ci);
   if (c) {
      fColor[0] = c->GetRed();
      fColor[1] = c->GetGreen();
      fColor[2] = c->GetBlue();
      fColor[3] = 1.0f - 0.01*transparency;
   }
   Modified();
}

//______________________________________________________________________________
void TGLPhysicalShape::SetupGLColors(TGLRnrCtx & rnrCtx, const Float_t* color) const
{
   // Setup colors - avoid setting things not required
   // for current draw flags.

   if (color == 0) color = fColor;

   switch (rnrCtx.DrawPass()) {
      case TGLRnrCtx::kPassWireFrame:
      {
         // Wireframe needs basic color only
         glColor4fv(color);
         break;
      }
      case TGLRnrCtx::kPassFill:
      case TGLRnrCtx::kPassOutlineFill:
      {
         // Both need material colors

         // Set back diffuse only for clipping where inner (back) faces
         // are shown. Don't set shinneness or specular as we want
         // back face to appear as 'flat' as possible as crude visual
         // approximation to proper capped clipped solid
         glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color);
         glMaterialfv(GL_FRONT, GL_AMBIENT,  color + 4);
         glMaterialfv(GL_FRONT, GL_SPECULAR, color + 8);
         glMaterialfv(GL_FRONT, GL_EMISSION, color + 12);
         glMaterialf(GL_FRONT, GL_SHININESS, color[16]);
         // Some objects use point/line graphics. Material mode disabled.
         glColor4fv(color);
         break;
      }
      case TGLRnrCtx::kPassOutlineLine:
      {
         // Outline also needs grey wireframe but respecting
         // transparency of main diffuse color.
         TGLUtil::ColorAlpha(rnrCtx.ColorSet().Outline(), 0.5f*color[3]);
         break;
      }
      default:
      {
         assert(kFALSE);
      }
   }
}

//______________________________________________________________________________
void TGLPhysicalShape::Draw(TGLRnrCtx & rnrCtx) const
{
   // Draw physical shape, using LOD flags, potential from display list cache

   // Debug tracing
   if (gDebug > 4) {
      Info("TGLPhysicalShape::Draw", "this %ld (class %s) LOD %d",
           (Long_t)this, IsA()->GetName(), rnrCtx.ShapeLOD());
   }

   // If LOD is pixel or less can draw pixel(point) directly, skipping
   // any logical call, caching etc.
   if (rnrCtx.ShapeLOD() == TGLRnrCtx::kLODPixel)
   {
      if (!rnrCtx.IsDrawPassOutlineLine())
      {
         glColor4fv(fColor);
         glBegin(GL_POINTS);
         glVertex3dv(&fTransform.CArr()[12]);
         glEnd();
      }
      return;
   }

   if (gDebug > 4) {
      Info("TGLPhysicalShape::Draw", "this %ld (class %s) LOD %d",
           (Long_t)this, IsA()->GetName(), rnrCtx.ShapeLOD());
   }

   glPushMatrix();
   glMultMatrixd(fTransform.CArr());
   if (fIsScaleForRnr) glEnable(GL_NORMALIZE);
   if (fInvertedWind)  glFrontFace(GL_CW);
   if (rnrCtx.Highlight())
   {
      glPushAttrib(GL_LIGHTING_BIT | GL_DEPTH_BUFFER_BIT);

      glDisable(GL_LIGHTING);
      glDisable(GL_DEPTH_TEST);

      if (rnrCtx.HighlightOutline())
      {
         static const Int_t offsets[20][2] =
           { {-1,-1}, { 1,-1}, { 1, 1}, {-1, 1},
             { 1, 0}, { 0, 1}, {-1, 0}, { 0,-1},
             { 0,-2}, { 2, 0}, { 0, 2}, {-2, 0},
             {-2,-2}, { 2,-2}, { 2, 2}, {-2, 2},
             { 0,-3}, { 3, 0}, { 0, 3}, {-3, 0} };
         static const Int_t max_off =
           TGLUtil::GetScreenScalingFactor() > 1.5 ? 20 : 12;

         const TGLRect& vp = rnrCtx.RefCamera().RefViewport();

         for (int i = 0; i < max_off; ++i)
         {
            glViewport(vp.X() + offsets[i][0], vp.Y() + offsets[i][1], vp.Width(), vp.Height());
            fLogicalShape->DrawHighlight(rnrCtx, this);
         }

         glViewport(vp.X(), vp.Y(), vp.Width(), vp.Height());
      }
      else
      {
         fLogicalShape->DrawHighlight(rnrCtx, this);
      }

      glPopAttrib();
   }
   else
   {
      SetupGLColors(rnrCtx);
      if (rnrCtx.IsDrawPassOutlineLine())
         TGLUtil::LockColor();
      fLogicalShape->Draw(rnrCtx);
      if (rnrCtx.IsDrawPassOutlineLine())
         TGLUtil::UnlockColor();
   }
   if (fInvertedWind)  glFrontFace(GL_CCW);
   if (fIsScaleForRnr) glDisable(GL_NORMALIZE);
   glPopMatrix();
}

//______________________________________________________________________________
void TGLPhysicalShape::CalculateShapeLOD(TGLRnrCtx& rnrCtx, Float_t& pixSize, Short_t& shapeLOD) const
{
   // Calculate shape-lod, suitible for use under
   // projection defined by 'rnrCtx', taking account of which local
   // axes of the shape support LOD adjustment, and the global
   // 'sceneFlags' passed.
   //
   // Returned shapeLOD component is from 0 (kLODPixel - lowest
   // quality) to 100 (kLODHigh - highest quality).
   //
   // Scene flags are not used. LOD quantization is not done.  RnrCtx
   // is not modified as this is called via lodification stage of
   // rendering.

   TGLLogicalShape::ELODAxes lodAxes = fLogicalShape->SupportedLODAxes();

   if (lodAxes == TGLLogicalShape::kLODAxesNone)
   {  // Shape doesn't support LOD along any axes return special
      // unsupported LOD draw/cache flag.
      // TODO: Still ... could check for kLODPixel when very small,
      //    by using diagonal from bounding-box and some special camera foo.
      pixSize  = 100; // Make up something / irrelevant.
      shapeLOD = TGLRnrCtx::kLODHigh;
      return;
   }

   std::vector <Double_t> boxViewportDiags;
   const TGLBoundingBox & box        = BoundingBox();
   const TGLCamera      & camera     = rnrCtx.RefCamera();

   if (lodAxes == TGLLogicalShape::kLODAxesAll) {
      // Shape supports LOD along all axes - basis LOD hint on diagonal of viewport
      // projection rect round whole bounding box
      boxViewportDiags.push_back(camera.ViewportRect(box).Diagonal());
   } else if (lodAxes == (TGLLogicalShape::kLODAxesY | TGLLogicalShape::kLODAxesZ)) {
      // Shape supports LOD along Y/Z axes (not X). LOD hint based on longest
      // diagonal (largest rect) of either of the X axis end faces
      boxViewportDiags.push_back(camera.ViewportRect(box, TGLBoundingBox::kFaceLowX).Diagonal());
      boxViewportDiags.push_back(camera.ViewportRect(box, TGLBoundingBox::kFaceHighX).Diagonal());
   } else if (lodAxes == (TGLLogicalShape::kLODAxesX | TGLLogicalShape::kLODAxesZ)) {
      // Shape supports LOD along X/Z axes (not Y). See above for Y/Z
      boxViewportDiags.push_back(camera.ViewportRect(box, TGLBoundingBox::kFaceLowY).Diagonal());
      boxViewportDiags.push_back(camera.ViewportRect(box, TGLBoundingBox::kFaceHighY).Diagonal());
   } else if (lodAxes == (TGLLogicalShape::kLODAxesX | TGLLogicalShape::kLODAxesY)) {
      // Shape supports LOD along X/Y axes (not Z). See above for Y/Z
      boxViewportDiags.push_back(camera.ViewportRect(box, TGLBoundingBox::kFaceLowZ).Diagonal());
      boxViewportDiags.push_back(camera.ViewportRect(box, TGLBoundingBox::kFaceHighZ).Diagonal());
   } else {
      // Don't bother to implement LOD calc for shapes supporting LOD along single
      // axis only. Not needed at present + unlikely case - but could be done based
      // on longest of projection of 4 edges of BBox along LOD axis. However this would
      // probably be more costly than just using whole BB projection (as for all axes)
      Error("TGLPhysicalShape::CalcPhysicalLOD", "LOD calculation for single axis not implemented presently");
      shapeLOD = TGLRnrCtx::kLODMed;
      return;
   }

   // Find largest of the projected diagonals
   Double_t largestDiagonal = 0.0;
   for (UInt_t i = 0; i < boxViewportDiags.size(); i++) {
      if (boxViewportDiags[i] > largestDiagonal) {
         largestDiagonal = boxViewportDiags[i];
      }
   }
   pixSize = largestDiagonal;

   if (largestDiagonal <= 1.0) {
      shapeLOD = TGLRnrCtx::kLODPixel;
   } else {
      // TODO: Get real screen size - assuming 2000 pixel screen at present
      // Calculate a non-linear sizing hint for this shape based on diagonal.
      // Needs more experimenting with...
      UInt_t lodApp = static_cast<UInt_t>(std::pow(largestDiagonal,0.4) * 100.0 / std::pow(2000.0,0.4));
      if (lodApp > 1000) lodApp = 1000;
      shapeLOD = (Short_t) lodApp;
   }
}

//______________________________________________________________________________
void TGLPhysicalShape::QuantizeShapeLOD(Short_t shapeLOD, Short_t combiLOD, Short_t& quantLOD) const
{
   // Factor in scene/vierer LOD and Quantize ... forward to
   // logical shape.

   quantLOD = fLogicalShape->QuantizeShapeLOD(shapeLOD, combiLOD);
}

//______________________________________________________________________________
void TGLPhysicalShape::InvokeContextMenu(TContextMenu & menu, UInt_t x, UInt_t y) const
{
   // Request creation of context menu on shape, attached to 'menu' at screen position
   // 'x' 'y'

   // Just defer to our logical at present
   fLogicalShape->InvokeContextMenu(menu, x, y);
}
