// @(#)root/gl:$Name:  $:$Id: TGLPhysicalShape.cxx,v 1.23 2006/04/07 09:20:43 rdm Exp $
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
#include "TGLCamera.h"
#include "TGLDrawFlags.h"
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
   TGLDrawable(ID, kFALSE), // Physical shapes NOT DL cached - see Draw()
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
   TGLDrawable(ID, kFALSE), // Physical shapes NOT DL cached - see Draw()
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
   // Destroy the physical shape
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

   // DL cache is NOT affected by this (hence no Purge()) as colors set in Draw() -
   // not DirectDraw(). See Draw() for reasons

   fModified = kTRUE;
}

//______________________________________________________________________________
void TGLPhysicalShape::Draw(const TGLDrawFlags & flags) const
{
   // Draw physical shape, using LOD flags, potential from display list cache

   // Debug tracing
   if (gDebug > 4) {
      Info("TGLPhysicalShape::Draw", "this %d (class %s) LOD %d", this, IsA()->GetName(), flags.LOD());
   }

   // IMPORTANT: Per drawable DL cache purging does not work currently.
   // This means that DL caching cannot be enabled at TGLPhysicalShape level as
   // modifications (scale/rotate etc) will not result in the cache being
   // purged.

   // Setup current colors
   // NOTE: These *could* be done in DirectDraw(), and hence DL cached if/when enabled
   // at this level. This *might* be faster....
   // However this would mean the DL cache would need to create entries based on draw
   // style as well as LOD from 'flags' - resulting in DLs with different colors, but
   // identical geometry.
   //
   // TODO: Better solution - sorting of physicals - Min. state swap for attributes
   // Or maybe set color in logical then override if modified in physical????
   // as in normal cases physicals sharing same logical have same color.

   // Setup colors - avoid setting things not required
   // for current draw flags
   switch (flags.Style()) {
      case TGLDrawFlags::kWireFrame: {
         // Wireframe needs basic color only
         glColor4fv(fColor);
         break;
      }
      case TGLDrawFlags::kFill:
      case TGLDrawFlags::kOutline: {
         // Both need material colors

         // Set back diffuse only for clipping where inner (back) faces
         // are shown. Don't set shinneness or specular as we want
         // back face to appear as 'flat' as possible as crude visual
         // approximation to proper capped clipped solid
         glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, fColor);
         glMaterialfv(GL_FRONT, GL_AMBIENT, fColor + 4);
         glMaterialfv(GL_FRONT, GL_SPECULAR, fColor + 8);
         glMaterialfv(GL_FRONT, GL_EMISSION, fColor + 12);
         glMaterialf(GL_FRONT, GL_SHININESS, fColor[16]);

         // Outline also needs grey wireframe but respecting
         // transparency of main diffuse color
         if (flags.Style() == TGLDrawFlags::kOutline) {
            glColor4f(0.1, 0.1, 0.1, fColor[3]/2.0);
         } else {
            // Some objects use point/line graphics. Material mode disabled.
            glColor4fv(fColor);
         }

         // TODO: Scene draws outline style in two passes - for second
         // wireframe overlay one we don't need to set materials
         // But don't know the pass here.....
         // Also we only need to set back materials when clipping
         // - this might cause extra uneeded cost
         break;
      }
   }

   TGLDrawable::Draw(flags);
}

//______________________________________________________________________________
void TGLPhysicalShape::DirectDraw(const TGLDrawFlags & flags) const
{
   // Draw physical shape, using LOD flags - can be captured into display list cache
   //
   // Load gl name (for selection)
   // Load translation matrix - placement
   // Call associated fLogicalShape Draw() method

   // Debug tracing
   if (gDebug > 4) {
      Info("TGLPhysicalShape::DirectDraw", "this %d (class %s) LOD %d", this, IsA()->GetName(), flags.LOD());
   }

   glLoadName(ID());
   glPushMatrix();
   glMultMatrixd(fTransform.CArr());
   if (fInvertedWind) {
      glFrontFace(GL_CW);
   }
   // If LOD is pixel or less can draw pixel(point) directly, skipping
   // any logical call, caching etc
   if (flags.LOD() == TGLDrawFlags::kLODPixel) {
      glBegin(GL_POINTS);
      glVertex3d(0.0, 0.0, 0.0);
      glEnd();
   } else {
      fLogicalShape.Draw(flags);
   }
   if (fInvertedWind) {
      glFrontFace(GL_CCW);
   }
   glPopMatrix();
}

//______________________________________________________________________________
TGLDrawFlags TGLPhysicalShape::CalcDrawFlags(const TGLCamera & camera, const TGLDrawFlags & sceneFlags) const
{
   // Return draw flags for shape, suitible for use under projection defined by 'camera',
   // taking account of which local axes of the shape support LOD adjustment, and the
   // global 'sceneFlags' passed.
   //
   // sceneFlags.Style() - copied to shape draw style
   // sceneFlags.LOD() - factored into projection LOD
   //
   // Returned LOD() component is UInt 0 (kLODPixel - lowest quality) to
   // 100 (kLODHigh - highest quality) or special case kLODUnsupported if shape
   // does not support LOD at all

   std::vector <Double_t> boxViewportDiags;
   TGLDrawable::ELODAxes lodAxes = SupportedLODAxes();
   const TGLBoundingBox & box = BoundingBox();

   if (lodAxes == TGLDrawable::kLODAxesNone) {
      // Shape doesn't support LOD along any axes return special unsupported LOD draw/cache flag
      return TGLDrawFlags(sceneFlags.Style(), TGLDrawFlags::kLODUnsupported,
                          sceneFlags.Selection(), sceneFlags.SecSelection());
   }

   if (lodAxes == TGLDrawable::kLODAxesAll) {
      // Shape supports LOD along all axes - basis LOD hint on diagonal of viewport
      // projection rect round whole bounding box
      boxViewportDiags.push_back(camera.ViewportRect(box).Diagonal());
   } else if (lodAxes == (TGLDrawable::kLODAxesY | TGLDrawable::kLODAxesZ)) {
      // Shape supports LOD along Y/Z axes (not X). LOD hint based on longest
      // diagonal (largest rect) of either of the X axis end faces
      boxViewportDiags.push_back(camera.ViewportRect(box, TGLBoundingBox::kFaceLowX).Diagonal());
      boxViewportDiags.push_back(camera.ViewportRect(box, TGLBoundingBox::kFaceHighX).Diagonal());
   } else if (lodAxes == (TGLDrawable::kLODAxesX | TGLDrawable::kLODAxesZ)) {
      // Shape supports LOD along X/Z axes (not Y). See above for Y/Z
      boxViewportDiags.push_back(camera.ViewportRect(box, TGLBoundingBox::kFaceLowY).Diagonal());
      boxViewportDiags.push_back(camera.ViewportRect(box, TGLBoundingBox::kFaceHighY).Diagonal());
   } else if (lodAxes == (TGLDrawable::kLODAxesX | TGLDrawable::kLODAxesY)) {
      // Shape supports LOD along X/Y axes (not Z). See above for Y/Z
      boxViewportDiags.push_back(camera.ViewportRect(box, TGLBoundingBox::kFaceLowZ).Diagonal());
      boxViewportDiags.push_back(camera.ViewportRect(box, TGLBoundingBox::kFaceHighZ).Diagonal());
   }
   else {
      // Don't bother to implement LOD calc for shapes supporting LOD along single
      // axis only. Not needed at present + unlikely case - but could be done based
      // on longest of projection of 4 edges of BBox along LOD axis. However this would
      // probably be more costly than just using whole BB projection (as for all axes)
      Error("TGLScene::CalcPhysicalLOD", "LOD calculation for single axis not implemented presently");
      return TGLDrawFlags(sceneFlags.Style(), TGLDrawFlags::kLODMed,
                          sceneFlags.Selection(), sceneFlags.SecSelection()); 
   }

   // Find largest of the projected diagonals
   Double_t largestDiagonal = 0.0;
   for (UInt_t i = 0; i < boxViewportDiags.size(); i++) {
      if (boxViewportDiags[i] > largestDiagonal) {
         largestDiagonal = boxViewportDiags[i];
      }
   }

   // Pixel or less?
   if (largestDiagonal <= 1.0) {
      return TGLDrawFlags(sceneFlags.Style(), TGLDrawFlags::kLODPixel,
                          sceneFlags.Selection(), sceneFlags.SecSelection());
   }

   // TODO: Get real screen size - assuming 2000 pixel screen at present
   // Calculate a non-linear sizing hint for this shape based on diagonal.
   // Needs more experimenting with...
   UInt_t sizeLOD = static_cast<UInt_t>(pow(largestDiagonal,0.4) * 100.0 / pow(2000.0,0.4));

   // Factor in scene quality
   UInt_t shapeLOD = (sceneFlags.LOD() * sizeLOD) / 100;

   // Round LOD above 10 to nearest 10
   if (shapeLOD > 10) {
      Double_t quant = ((static_cast<Double_t>(shapeLOD)) + 0.5) / 10;
      shapeLOD = static_cast<UInt_t>(quant)*10;
   }
   // Round LOD below 10 to nearest 2
   else {
      Double_t quant = ((static_cast<Double_t>(shapeLOD)) + 0.5) / 2;
      shapeLOD = static_cast<UInt_t>(quant)*3;
   }

   if (shapeLOD > 100) {
      shapeLOD = 100;
   }

   return TGLDrawFlags(sceneFlags.Style(), shapeLOD,
                       sceneFlags.Selection(), sceneFlags.SecSelection());
}

//______________________________________________________________________________
void TGLPhysicalShape::InvokeContextMenu(TContextMenu & menu, UInt_t x, UInt_t y) const
{
   // Request creation of context menu on shape, attached to 'menu' at screen position
   // 'x' 'y'

   // Just defer to our logical at present
   fLogicalShape.InvokeContextMenu(menu, x, y);
}

