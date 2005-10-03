// @(#)root/gl:$Name:  $:$Id: TGLCamera.cxx,v 1.16 2005/08/30 10:29:52 brun Exp $
// Author:  Richard Maunder  25/05/2005
// Parts taken from original by Timur Pocheptsov

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// TODO: Function descriptions
// TODO: Class def - same as header

#include "TGLCamera.h"
#include "TGLIncludes.h"
#include "TGLBoundingBox.h"
#include "TError.h"

ClassImp(TGLCamera)

const Double_t TGLCamera::fgInterestBoxExpansion = 1.3;

//______________________________________________________________________________
TGLCamera::TGLCamera() :
   fViewport(0,0,100,100),
   fProjM(),  fModVM(), fClipM(),
   fCacheDirty(kTRUE),
   fLargestInterest(0.0)
{
   for (UInt_t i = 0; i < kPlanesPerFrustum; i++ ) {
      fFrustumPlanes[i].Set(1.0, 0.0, 0.0, 0.0);
   }
}

//______________________________________________________________________________
TGLCamera::~TGLCamera()
{}

//______________________________________________________________________________
void TGLCamera::SetViewport(const TGLRect & viewport)
{
   fViewport = viewport;
   fCacheDirty = true;
}

//______________________________________________________________________________
void TGLCamera::UpdateCache()
{
   // Update internal cached values
   assert(fCacheDirty);

   glGetDoublev(GL_PROJECTION_MATRIX, fProjM.Arr());
   glGetDoublev(GL_MODELVIEW_MATRIX, fModVM.Arr());

   // Multiply projection by modelview to get the clip matrix
   // TODO: Move this into TGLMatrix or shift all over to ROOT ones
   fClipM[ 0] = fModVM[ 0] * fProjM[ 0] + fModVM[ 1] * fProjM[ 4] + fModVM[ 2] * fProjM[ 8] + fModVM[ 3] * fProjM[12];
   fClipM[ 1] = fModVM[ 0] * fProjM[ 1] + fModVM[ 1] * fProjM[ 5] + fModVM[ 2] * fProjM[ 9] + fModVM[ 3] * fProjM[13];
   fClipM[ 2] = fModVM[ 0] * fProjM[ 2] + fModVM[ 1] * fProjM[ 6] + fModVM[ 2] * fProjM[10] + fModVM[ 3] * fProjM[14];
   fClipM[ 3] = fModVM[ 0] * fProjM[ 3] + fModVM[ 1] * fProjM[ 7] + fModVM[ 2] * fProjM[11] + fModVM[ 3] * fProjM[15];

   fClipM[ 4] = fModVM[ 4] * fProjM[ 0] + fModVM[ 5] * fProjM[ 4] + fModVM[ 6] * fProjM[ 8] + fModVM[ 7] * fProjM[12];
   fClipM[ 5] = fModVM[ 4] * fProjM[ 1] + fModVM[ 5] * fProjM[ 5] + fModVM[ 6] * fProjM[ 9] + fModVM[ 7] * fProjM[13];
   fClipM[ 6] = fModVM[ 4] * fProjM[ 2] + fModVM[ 5] * fProjM[ 6] + fModVM[ 6] * fProjM[10] + fModVM[ 7] * fProjM[14];
   fClipM[ 7] = fModVM[ 4] * fProjM[ 3] + fModVM[ 5] * fProjM[ 7] + fModVM[ 6] * fProjM[11] + fModVM[ 7] * fProjM[15];

   fClipM[ 8] = fModVM[ 8] * fProjM[ 0] + fModVM[ 9] * fProjM[ 4] + fModVM[10] * fProjM[ 8] + fModVM[11] * fProjM[12];
   fClipM[ 9] = fModVM[ 8] * fProjM[ 1] + fModVM[ 9] * fProjM[ 5] + fModVM[10] * fProjM[ 9] + fModVM[11] * fProjM[13];
   fClipM[10] = fModVM[ 8] * fProjM[ 2] + fModVM[ 9] * fProjM[ 6] + fModVM[10] * fProjM[10] + fModVM[11] * fProjM[14];
   fClipM[11] = fModVM[ 8] * fProjM[ 3] + fModVM[ 9] * fProjM[ 7] + fModVM[10] * fProjM[11] + fModVM[11] * fProjM[15];

   fClipM[12] = fModVM[12] * fProjM[ 0] + fModVM[13] * fProjM[ 4] + fModVM[14] * fProjM[ 8] + fModVM[15] * fProjM[12];
   fClipM[13] = fModVM[12] * fProjM[ 1] + fModVM[13] * fProjM[ 5] + fModVM[14] * fProjM[ 9] + fModVM[15] * fProjM[13];
   fClipM[14] = fModVM[12] * fProjM[ 2] + fModVM[13] * fProjM[ 6] + fModVM[14] * fProjM[10] + fModVM[15] * fProjM[14];
   fClipM[15] = fModVM[12] * fProjM[ 3] + fModVM[13] * fProjM[ 7] + fModVM[14] * fProjM[11] + fModVM[15] * fProjM[15];

   // RIGHT clipping plane
   fFrustumPlanes[kRIGHT].Set(fClipM[ 3] - fClipM[ 0],
                              fClipM[ 7] - fClipM[ 4],
                              fClipM[11] - fClipM[ 8],
                              fClipM[15] - fClipM[12]);

   // LEFT clipping plane
   fFrustumPlanes[kLEFT].Set(fClipM[ 3] + fClipM[ 0],
                             fClipM[ 7] + fClipM[ 4],
                             fClipM[11] + fClipM[ 8],
                             fClipM[15] + fClipM[12]);

   // BOTTOM clipping plane
   fFrustumPlanes[kBOTTOM].Set(fClipM[ 3] + fClipM[ 1],
                               fClipM[ 7] + fClipM[ 5],
                               fClipM[11] + fClipM[ 9],
                               fClipM[15] + fClipM[13]);


   // TOP clipping plane
   fFrustumPlanes[kTOP].Set(fClipM[ 3] - fClipM[ 1],
                            fClipM[ 7] - fClipM[ 5],
                            fClipM[11] - fClipM[ 9],
                            fClipM[15] - fClipM[13]);

   // FAR clipping plane
   fFrustumPlanes[kFAR].Set(fClipM[ 3] - fClipM[ 2],
                            fClipM[ 7] - fClipM[ 6],
                            fClipM[11] - fClipM[10],
                            fClipM[15] - fClipM[14]);

   // NEAR clipping plane
   fFrustumPlanes[kNEAR].Set(fClipM[ 3] + fClipM[ 2],
                             fClipM[ 7] + fClipM[ 6],
                             fClipM[11] + fClipM[10],
                             fClipM[15] + fClipM[14]);

   fCacheDirty = kFALSE;
}

//______________________________________________________________________________
TGLBoundingBox TGLCamera::Frustum(Bool_t asBox) const
{
   // Return the the current camera frustum. If asBox == FALSE return
   // a true frustum (truncated square based pyramid). If asBox == TRUE
   // return a true box, using the far clipping plane intersection projected
   // back to the near plane

   // TODO: BoundingBox return name is misleading - and not always valid
   // Need a generic bounding volume object
   if (fCacheDirty) {
      Error("TGLCamera::FrustumBox()", "cache dirty");
   }


   TGLVertex3 vertex[8];

   //    7-------6
   //   /|      /|
   //  3-------2 |
   //  | 4-----|-5
   //  |/      |/
   //  0-------1

   // Get four vertices of frustum on the far clipping plane
   vertex[4] = Intersection(fFrustumPlanes[kFAR], fFrustumPlanes[kBOTTOM], fFrustumPlanes[kLEFT]);
   vertex[5] = Intersection(fFrustumPlanes[kFAR], fFrustumPlanes[kBOTTOM], fFrustumPlanes[kRIGHT]);
   vertex[6] = Intersection(fFrustumPlanes[kFAR], fFrustumPlanes[kTOP],    fFrustumPlanes[kRIGHT]);
   vertex[7] = Intersection(fFrustumPlanes[kFAR], fFrustumPlanes[kTOP],    fFrustumPlanes[kLEFT]);

   if (asBox) {
      // Now find the matching four verticies for above, projected onto near clip plane
      // As near and far clip planes are parallel this forms a orientated box encompassing the frustum
      vertex[0] = fFrustumPlanes[kNEAR].NearestOn(vertex[4]);
      vertex[1] = fFrustumPlanes[kNEAR].NearestOn(vertex[5]);
      vertex[2] = fFrustumPlanes[kNEAR].NearestOn(vertex[6]);
      vertex[3] = fFrustumPlanes[kNEAR].NearestOn(vertex[7]);
   } else {
      // returing true frustum - find verticies at near clipping plane
      vertex[0] = Intersection(fFrustumPlanes[kNEAR], fFrustumPlanes[kBOTTOM], fFrustumPlanes[kLEFT]);
      vertex[1] = Intersection(fFrustumPlanes[kNEAR], fFrustumPlanes[kBOTTOM], fFrustumPlanes[kRIGHT]);
      vertex[2] = Intersection(fFrustumPlanes[kNEAR], fFrustumPlanes[kTOP],    fFrustumPlanes[kRIGHT]);
      vertex[3] = Intersection(fFrustumPlanes[kNEAR], fFrustumPlanes[kTOP],    fFrustumPlanes[kLEFT]);
   }

   return TGLBoundingBox(vertex);
}

//______________________________________________________________________________
TGLVertex3 TGLCamera::EyePoint() const
{
  // Extract the camera eye point using the current frustum planes
  if (fCacheDirty) {
      Error("TGLCamera::FrustumBox()", "cache dirty");
   }
   return Intersection(fFrustumPlanes[kRIGHT], fFrustumPlanes[kLEFT], fFrustumPlanes[kTOP]);
}

//______________________________________________________________________________
TGLVector3 TGLCamera::EyeDirection() const
{
   // Extract the camera eye direction using the current frustum planes
   if (fCacheDirty) {
      Error("TGLCamera::FrustumBox()", "cache dirty");
   }
   // Direction is just normal of near clipping plane
   return fFrustumPlanes[kNEAR].Norm();
}

//______________________________________________________________________________
EOverlap TGLCamera::FrustumOverlap(const TGLBoundingBox & box) const
{
   if (fCacheDirty) {
      Error("TGLCamera::FrustumOverlap()", "cache dirty");
   }

   // Test shape against each plane in frustum - returning overlap result
   // This method can result in kFALSE positives, where shape lies outside
   // frustum, but not outside a single plane of it. In this case the shape
   // will be regarded incorrectly as intersecting (kPartial)
   // TODO: Improve this - have a reliable test (seperating axes).

   Int_t planesInside = 0; // Assume outside to start
   for (Int_t planeIndex = 0; planeIndex < kPlanesPerFrustum; ++planeIndex) {
      EOverlap planeOverlap = box.Overlap(fFrustumPlanes[planeIndex]);

	  // Special case - any object which comes through the near clipping
     // plane is completely removed - disabled at present
     // TODO: In future may want to fade object (opacity) as they approach
      // near clip - how will this be returned? template pair?
      /*if (planeIndex == kNEAR && planeOverlap == kPartial) {
         return kOutside;
      }*/
      // Once we find a single plane which shape is outside, we are outside the frustum
      if ( planeOverlap == kOutside ) {
         return kOutside;
      } else if ( planeOverlap == kInside ) {
         planesInside++;
      }
   }
   // Completely inside frustum
   if ( planesInside == kPlanesPerFrustum ) {
      return kInside;
   } else {
      return kPartial;
   }
}

//______________________________________________________________________________
EOverlap TGLCamera::ViewportOverlap(const TGLBoundingBox & box) const
{
   // No cached values need here
   return ViewportSize(box).Overlap(fViewport);
}

//______________________________________________________________________________
TGLRect TGLCamera::ViewportSize(const TGLBoundingBox & box) const
{
   if (fCacheDirty) {
      Error("TGLCamera::ViewportSize()", "cache dirty");
   }

   // May often result in a rect bigger then the viewport
   // as gluProject does not clip.
   Double_t winX, winY, winZ;
   TGLRect  screenRect;

   // Find the projection of the 8 vertexs of the bounding box onto screen
   // and the enclosing rect round these.
   for (Int_t i = 0; i < 8; i++)
   {
      const TGLVertex3 & vertex = box[i];

      //TODO: Convert TGLRect so this not required
      Int_t viewport[4] = { fViewport.X(), fViewport.Y(), fViewport.Width(), fViewport.Height() };

      gluProject(vertex.X(), vertex.Y(), vertex.Z(), fModVM.CArr(), fProjM.CArr(), viewport, &winX, &winY, &winZ);

      if (i == 0) {
         screenRect.SetCorner(static_cast<Int_t>(winX),static_cast<Int_t>(winY));
      }
      else {
         screenRect.Expand(static_cast<Int_t>(winX), static_cast<Int_t>(winY));
      }
   }

   return screenRect;
}

//______________________________________________________________________________
TGLVector3 TGLCamera::ProjectedShift(const TGLVertex3 & vertex, Int_t xDelta, Int_t yDelta) const
{
   if (fCacheDirty) {
      Error("TGLCamera::ProjectedShift()", "cache dirty");
   }
   //TODO: Convert TGLRect so this not required
   Int_t viewport[4] = { fViewport.X(), fViewport.Y(), fViewport.Width(), fViewport.Height() };
   TGLVertex3 winVertex;
   gluProject(vertex[0], vertex[1], vertex[2], fModVM.CArr(), fProjM.CArr(), viewport, &winVertex[0], &winVertex[1], &winVertex[2]);
   winVertex.Shift(xDelta, yDelta, 0.0);
   TGLVertex3 newVertex;
   gluUnProject(winVertex[0], winVertex[1], winVertex[2], fModVM.CArr(), fProjM.CArr(), viewport, &newVertex[0], &newVertex[1], &newVertex[2]);
   return (newVertex - vertex);
}

//______________________________________________________________________________
Bool_t TGLCamera::OfInterest(const TGLBoundingBox & box) const
{
   Bool_t interest = kFALSE;

   // If interest box is empty we take everything with volume larger than
   // 1% of largest seen so far
   if (fInterestBox.IsEmpty()) {
      if (box.Volume() >= fLargestInterest * 0.01) {
         if (box.Volume() > fLargestInterest) {
            fLargestInterest = box.Volume();
         }
         interest = kTRUE;
      }
   } else {
      // We have a valid interest box

      // Objects are of interest if the have sufficient length or volume ratio c.f.
      // the current interest box, and they at least partially overlap it
      Double_t lengthRatio = box.Extents().Mag() / fInterestBox.Extents().Mag();
      
      // Some objects have zero volume BBs - e.g. single points - skip the volume ratio
      // test for these - no way to threshold on 0
      Double_t volumeRatio = 1.0;
      if (!box.IsEmpty()) {
         volumeRatio = box.Volume() / fInterestBox.Volume();
      }

      if ((lengthRatio > 0.001) || (volumeRatio > 0.0001)) {
         interest = fInterestBox.Overlap(box) != kOutside;
      }
   }

   return interest;
}

//______________________________________________________________________________
Bool_t TGLCamera::UpdateInterest(Bool_t force)
{
   Bool_t exposedUpdate = kFALSE;

   // Construct a new interest box using the current frustum box as a basis
   TGLBoundingBox frustumBox = Frustum(kTRUE);
   TGLBoundingBox newInterestBox(frustumBox);

   // The Z(2) axis of frustum (near->far plane) can be quite shallow c.f. X(0)/Y(1)
   // For interest box we want to expand to ensure it is at least size
   // of smaller X/Y to avoid excessive interest box recalculations
   TGLVector3 frustumExtents = frustumBox.Extents();
   Double_t minBoxLength = frustumExtents.Mag() * fgInterestBoxExpansion;
   newInterestBox.Scale(minBoxLength/frustumExtents[0], minBoxLength/frustumExtents[1], minBoxLength/frustumExtents[2]);

   // Expand the interest box
   //newInterestBox.Scale(fgInterestBoxExpansion);

   // Calculate volume ratio of new to old
   Double_t volRatio = 0.0;
   if (!fInterestBox.IsEmpty()) {
      volRatio = newInterestBox.Volume() / fInterestBox.Volume();
   }

   // Update the existing interest box with new one if:
   // i) Volume ratio old/new interest has changed significantly
   // ii) The current frustum is not inside existing interest
   // iii) Force case (debugging)
   if (volRatio > 8.0 || volRatio < 0.125 || fInterestBox.IsEmpty() || 
       fInterestBox.Overlap(frustumBox) != kInside || force) {
      fPreviousInterestBox = fInterestBox;
      fInterestBox = newInterestBox;

      // Frustum should be fully contained now
      if (fInterestBox.Overlap(frustumBox) != kInside) {
         Error("TGLCamera::UpdateInterest", "update interest box does not contain frustum");
      }
      
      exposedUpdate = kTRUE;

      // Keep the real frustum (real and box versions) as debuging aid
      fInterestFrustum = Frustum(kFALSE);
      fInterestFrustumAsBox = frustumBox;
      
      if (gDebug>2 || force) {
         Info("TGLCamera::UpdateInterest", "changed - volume ratio %f", volRatio );
      }
   }

   return exposedUpdate;
}

//______________________________________________________________________________
void TGLCamera::ResetInterest()
{
   fInterestBox.SetEmpty();
   fLargestInterest = 0.0;
}

//______________________________________________________________________________
Bool_t TGLCamera::AdjustAndClampVal(Double_t & val, Double_t min, Double_t max,
                                    Int_t screenShift, Int_t screenShiftRange, 
                                    Bool_t mod1, Bool_t mod2) const
{  
   if (screenShift == 0) {
      return kFALSE;
   }

   // Calculate a sensitivity based on passed modifiers
   Double_t sens = 1.0;
   
   if (mod1) {
      sens *= 0.1;
      if (mod2) {
         sens *= 0.1;
      }
   } else {
      if (mod2) {
         sens *= 10.0;
      }
   }

   Double_t oldVal = val;
   Double_t shift = static_cast<Double_t>(screenShift) * (val-min) * sens / static_cast<Double_t>(screenShiftRange);
   val -= shift;

   if (val < min) {
      val = min;
   }
   else if (val > max) {
      val = max;
   }

   if (val != oldVal)
   {
      return kTRUE;
   }
   else
   {
      return kFALSE;
   }
}

//______________________________________________________________________________
void TGLCamera::DrawDebugAids() const
{
   // Draw out some debugging aids for the camera:
   //
   // i) The frustum used to create the current interest box (RED)
   // ii) The same frustum as a squared off box (ORANGE)
   // iii) The axis aligned version of the frustum used as interest box basis (YELLOW) 
   // iv) The current interest box (BLUE)

   // Interest box frustum base (RED)
   glColor3d(1.0,0.0,0.0);
   fInterestFrustum.Draw();

   // Interest box frustum as box (ORANGE)
   glColor3d(1.0,0.65,0.15);
   fInterestFrustumAsBox.Draw();

   // Current Interest box (BLUE)
   glColor3d(0.0,0.0,1.0);
   fInterestBox.Draw();

   // Previous interest (GREY)
   glColor3d(.8,.7,.6);
   fPreviousInterestBox.Draw();

   // Also draw line from current eye point out in eye direction - should not
   // appear if calculated correctly
   TGLVertex3 start = EyePoint();
   TGLVertex3 end = start + EyeDirection();
   glColor3d(1.0,1.0,1.0);
   glBegin(GL_LINES);
   glVertex3dv(start.CArr());
   glVertex3dv(end.CArr());
   glEnd();
}
