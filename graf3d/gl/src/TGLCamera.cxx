// @(#)root/gl:$Id$
// Author:  Richard Maunder  25/05/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLCamera.h"
#include "TGLIncludes.h"
#include "TGLBoundingBox.h"
#include "TError.h"
#include "TMath.h"

/** \class TGLCamera
\ingroup opengl
Abstract base camera class - concrete classes for orthographic and
perspective cameras derive from it. This class maintains values for
the current:
  1. Viewport
  2. Projection, modelview and clip matrices - extracted from GL
  3. The 6 frustum planes
  4. Expanded frustum interest box

It provides methods for various projection, overlap and intersection
tests for viewport and world locations, against the true frustum and
expanded interest box, and for extracting eye position and direction.

It also defines the pure virtual manipulation interface methods the
concrete ortho and perspective classes must implement.
*/

ClassImp(TGLCamera);

const Double_t TGLCamera::fgInterestBoxExpansion = 1.3;
UInt_t         TGLCamera::fgDollyDeltaSens       = 500;

////////////////////////////////////////////////////////////////////////////////
/// Default base camera constructor

TGLCamera::TGLCamera() :
   fExternalCenter(kFALSE),
   fFixDefCenter(kFALSE),
   fWasArcBalled(kFALSE),
   fCenter(&fDefCenter),
   fNearClip(0), fFarClip(0),
   fDollyDefault(1.0), fDollyDistance(1.0),
   fVAxisMinAngle(0.01f),
   fCacheDirty(kTRUE),
   fTimeStamp (1),
   fProjM(), fModVM(), fClipM(),
   fViewport(0,0,100,100),
   fLargestSeen(0.0)
{
   for (UInt_t i = 0; i < kPlanesPerFrustum; i++ ) {
      fFrustumPlanes[i].Set(1.0, 0.0, 0.0, 0.0);
   }
   TGLVertex3 origin;
   fCamBase.Set(origin, TGLVector3(1, 0, 0), TGLVector3(0, 0, 1));
}

////////////////////////////////////////////////////////////////////////////////
/// Default base camera constructor

TGLCamera::TGLCamera(const TGLVector3 & hAxis, const TGLVector3 & vAxis) :
   fExternalCenter(kFALSE),
   fFixDefCenter(kFALSE),
   fWasArcBalled(kFALSE),
   fCenter(&fDefCenter),
   fNearClip(0), fFarClip(0),
   fDollyDefault(1.0), fDollyDistance(1.0),
   fVAxisMinAngle(0.01f),
   fCacheDirty(kTRUE),
   fTimeStamp (1),
   fProjM(), fModVM(), fClipM(),
   fViewport(0,0,100,100),
   fLargestSeen(0.0)
{
   for (UInt_t i = 0; i < kPlanesPerFrustum; i++ ) {
      fFrustumPlanes[i].Set(1.0, 0.0, 0.0, 0.0);
   }
   TGLVertex3 origin;
   fCamBase.Set(origin, vAxis, hAxis);
}

////////////////////////////////////////////////////////////////////////////////
/// Base camera destructor.

TGLCamera::~TGLCamera()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Set viewport extents from passed 'viewport' rect.

void TGLCamera::SetViewport(const TGLRect & viewport)
{
   fViewport = viewport;
   IncTimeStamp();
}

////////////////////////////////////////////////////////////////////////////////
/// Update internally cached frustum values

void TGLCamera::UpdateCache() const
{
   assert(fCacheDirty);

   glGetDoublev(GL_PROJECTION_MATRIX, fProjM.Arr());
   glGetDoublev(GL_MODELVIEW_MATRIX, fModVM.Arr());

   // Multiply projection by modelview to get the clip matrix
   // TODO: Move this into TGLMatrix or shift all over to ROOT ones
   fClipM  = fProjM;
   fClipM *= fModVM;

   // RIGHT clipping plane
   fFrustumPlanes[kRight].Set(fClipM[ 3] - fClipM[ 0],
                              fClipM[ 7] - fClipM[ 4],
                              fClipM[11] - fClipM[ 8],
                              fClipM[15] - fClipM[12]);

   // LEFT clipping plane
   fFrustumPlanes[kLeft].Set(fClipM[ 3] + fClipM[ 0],
                             fClipM[ 7] + fClipM[ 4],
                             fClipM[11] + fClipM[ 8],
                             fClipM[15] + fClipM[12]);

   // BOTTOM clipping plane
   fFrustumPlanes[kBottom].Set(fClipM[ 3] + fClipM[ 1],
                               fClipM[ 7] + fClipM[ 5],
                               fClipM[11] + fClipM[ 9],
                               fClipM[15] + fClipM[13]);


   // TOP clipping plane
   fFrustumPlanes[kTop].Set(fClipM[ 3] - fClipM[ 1],
                            fClipM[ 7] - fClipM[ 5],
                            fClipM[11] - fClipM[ 9],
                            fClipM[15] - fClipM[13]);

   // FAR clipping plane
   fFrustumPlanes[kFar].Set(fClipM[ 3] - fClipM[ 2],
                            fClipM[ 7] - fClipM[ 6],
                            fClipM[11] - fClipM[10],
                            fClipM[15] - fClipM[14]);

   // NEAR clipping plane
   fFrustumPlanes[kNear].Set(fClipM[ 3] + fClipM[ 2],
                             fClipM[ 7] + fClipM[ 6],
                             fClipM[11] + fClipM[10],
                             fClipM[15] + fClipM[14]);

   fCacheDirty = kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the current camera frustum. If asBox == kFALSE return
/// a true frustum (truncated square based pyramid). If asBox == kTRUE
/// return a true box, using the far clipping plane intersection projected
/// back to the near plane.
///
/// Camera must have valid frustum cache - call Apply() after last modification, before using
///
/// Note: TGLBoundingBox is not really valid when filled with truncated pyramid
/// - this is used as a visual debug aid only so ok.

TGLBoundingBox TGLCamera::Frustum(Bool_t asBox) const
{
   // TODO: BoundingBox object is not always valid
   // Need a generic bounding volume object
   if (fCacheDirty) {
      Error("TGLCamera::FrustumBox()", "cache dirty - must call Apply()");
   }


   TGLVertex3 vertex[8];

   //    7-------6
   //   /|      /|
   //  3-------2 |
   //  | 4-----|-5
   //  |/      |/
   //  0-------1

   // Get four vertices of frustum on the far clipping plane
   // We assume they always intersect
   vertex[4] = Intersection(fFrustumPlanes[kFar], fFrustumPlanes[kBottom], fFrustumPlanes[kLeft]).second;
   vertex[5] = Intersection(fFrustumPlanes[kFar], fFrustumPlanes[kBottom], fFrustumPlanes[kRight]).second;
   vertex[6] = Intersection(fFrustumPlanes[kFar], fFrustumPlanes[kTop],    fFrustumPlanes[kRight]).second;
   vertex[7] = Intersection(fFrustumPlanes[kFar], fFrustumPlanes[kTop],    fFrustumPlanes[kLeft]).second;

   if (asBox) {
      // Now find the matching four vertices for above, projected onto near clip plane
      // As near and far clip planes are parallel this forms a orientated box encompassing the frustum
      vertex[0] = fFrustumPlanes[kNear].NearestOn(vertex[4]);
      vertex[1] = fFrustumPlanes[kNear].NearestOn(vertex[5]);
      vertex[2] = fFrustumPlanes[kNear].NearestOn(vertex[6]);
      vertex[3] = fFrustumPlanes[kNear].NearestOn(vertex[7]);
   } else {
      // Returning true frustum - find vertices at near clipping plane
      // We assume they always intersect
      vertex[0] = Intersection(fFrustumPlanes[kNear], fFrustumPlanes[kBottom], fFrustumPlanes[kLeft]).second;
      vertex[1] = Intersection(fFrustumPlanes[kNear], fFrustumPlanes[kBottom], fFrustumPlanes[kRight]).second;
      vertex[2] = Intersection(fFrustumPlanes[kNear], fFrustumPlanes[kTop],    fFrustumPlanes[kRight]).second;
      vertex[3] = Intersection(fFrustumPlanes[kNear], fFrustumPlanes[kTop],    fFrustumPlanes[kLeft]).second;
   }

   return TGLBoundingBox(vertex);
}

////////////////////////////////////////////////////////////////////////////////
/// Return the camera eye point (vertex) in world space
/// Camera must have valid frustum cache - call Apply() after last modification, before using

TGLVertex3 TGLCamera::EyePoint() const
{
   if (fCacheDirty) {
      Error("TGLPerspectiveCamera::FrustumBox()", "cache dirty - must call Apply()");
   }

   // Use intersection of right/left/top frustum planes - can be done in
   // other ways from camera values but this is easiest.
   // Note for an ortho camera this will result in an infinite z distance
   // which is theoretically correct although of limited use
   return Intersection(fFrustumPlanes[kRight], fFrustumPlanes[kLeft], fFrustumPlanes[kTop]).second;
}

////////////////////////////////////////////////////////////////////////////////
/// Extract the camera eye direction (vector), running from EyePoint()
/// Camera must have valid frustum cache - call Apply() after last modification, before using

TGLVector3 TGLCamera::EyeDirection() const
{
   if (fCacheDirty) {
      Error("TGLCamera::FrustumBox()", "cache dirty - must call Apply()");
   }
   // Direction is just normal of near clipping plane
   return fFrustumPlanes[kNear].Norm();
}

////////////////////////////////////////////////////////////////////////////////
/// Find the center of the camera frustum from intersection of planes
/// This method will work even with parallel left/right & top/bottom and
/// infinite eye point of ortho cameras
/// Camera must have valid frustum cache - call Apply() after last modification, before using

TGLVertex3 TGLCamera::FrustumCenter() const
{
   if (fCacheDirty) {
      Error("TGLCamera::FrustumCenter()", "cache dirty - must call Apply()");
   }
   std::pair<Bool_t, TGLVertex3> nearBottomLeft = Intersection(fFrustumPlanes[kNear],
                                                               fFrustumPlanes[kBottom],
                                                               fFrustumPlanes[kLeft]);
   std::pair<Bool_t, TGLVertex3> farTopRight    = Intersection(fFrustumPlanes[kFar],
                                                               fFrustumPlanes[kTop],
                                                               fFrustumPlanes[kRight]);
   // Planes should intersect
   if (!nearBottomLeft.first || !farTopRight.first) {
      Error("TGLCamera::FrustumCenter()", "frustum planes invalid");
      return TGLVertex3(0.0, 0.0, 0.0);
   }
   return nearBottomLeft.second + (farTopRight.second - nearBottomLeft.second)/2.0;
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate overlap (kInside, kOutside, kPartial) of box with camera
/// frustum
/// Camera must have valid frustum cache - call Apply() after last modification, before using

Rgl::EOverlap TGLCamera::FrustumOverlap(const TGLBoundingBox & box) const
{
   if (fCacheDirty) {
      Error("TGLCamera::FrustumOverlap()", "cache dirty - must call Apply()");
   }

   // Test shape against each plane in frustum - returning overlap result
   // This method can result in kFALSE positives, where shape lies outside
   // frustum, but not outside a single plane of it. In this case the shape
   // will be regarded incorrectly as intersecting (kPartial)
   // TODO: Improve this - have a reliable test (separating axes).

   Int_t planesInside = 0; // Assume outside to start
   for (Int_t planeIndex = 0; planeIndex < kPlanesPerFrustum; ++planeIndex)
   {
      Rgl::EOverlap planeOverlap = box.Overlap(fFrustumPlanes[planeIndex]);

      // Special case - any object which comes through the near clipping
      // plane is completely removed - disabled at present
      // TODO: In future may want to fade object (opacity) as they approach
      // near clip - how will this be returned? template pair?
      /*if (planeIndex == kNear && planeOverlap == kPartial) {
         return kOutside;
      }*/
      // Once we find a single plane which shape is outside, we are outside the frustum
      if (planeOverlap == Rgl::kOutside) {
         return Rgl::kOutside;
      } else if (planeOverlap == Rgl::kInside) {
         planesInside++;
      }
   }
   // Completely inside frustum
   if (planesInside == kPlanesPerFrustum) {
      return Rgl::kInside;
   } else {
      return Rgl::kPartial;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate overlap (kInside, kOutside, kPartial) of box projection onto viewport
/// (as rect) against the viewport rect.
/// Camera must have valid frustum cache - call Apply() after last modification, before using.

Rgl::EOverlap TGLCamera::ViewportOverlap(const TGLBoundingBox & box) const
{
   return ViewportRect(box).Overlap(fViewport);
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate viewport rectangle which just contains projection of single 'face'
/// of world frame bounding box 'box' onto the viewport. Note use other version
/// of ViewportRect() if you want whole 'box' contained

TGLRect TGLCamera::ViewportRect(const TGLBoundingBox & box,
                                const TGLBoundingBox::EFace face) const
{
   return ViewportRect(box, &face);
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate viewport rectangle which just contains projection of
/// world frame bounding box 'box' onto the viewport. If face is
/// null the rect contains the whole bounding box (8 vertices/6
/// faces). If face is non-null it indicates a box face, and the
/// rect contains the single face (4 vertices). Note use other
/// version of ViewportRect() if you wish to just pass a static
/// EFace enum member (e.g. kFaceLowX)
///
/// Note:
///     1.   Rectangle is NOT clipped by viewport limits - so can result
///         in rect with corners outside viewport - negative etc
///     2.  TGLRect provides int (pixel based) values - not subpixel accurate
///     3. Camera must have valid frustum cache - call Apply() after last
///         modification, before calling

TGLRect TGLCamera::ViewportRect(const TGLBoundingBox & box,
                                const TGLBoundingBox::EFace * face) const
{
   if (fCacheDirty) {
      Error("TGLCamera::ViewportSize()", "cache dirty - must call Apply()");
   }

   // TODO: Maybe TGLRect should be converted to Double_t so subpixel accurate
   // Would give better LOD calculations at small sizes

   // May often result in a rect bigger then the viewport
   // as gluProject does not clip.
   Double_t winX, winY, winZ;
   TGLRect  screenRect;

   // TGLBoundingBox::Vertices() & TGLBoundingBox::FaceVertices() return
   // const & vectors so this *should* all be efficient...
   UInt_t vertexCount;
   if (face) {
      vertexCount = box.FaceVertices(*face).size();
   } else {
      vertexCount = box.NumVertices();
   }

   for (UInt_t i = 0; i < vertexCount; i++)
   {
      const TGLVertex3 & vertex = face ? box.Vertex(box.FaceVertices(*face).at(i)) :
                                         box.Vertex(i);

      gluProject(vertex.X(), vertex.Y(), vertex.Z(),
                 fModVM.CArr(), fProjM.CArr(), fViewport.CArr(),
                 &winX, &winY, &winZ);

      if (i == 0) {
         screenRect.SetCorner(static_cast<Int_t>(winX),static_cast<Int_t>(winY));
      } else {
         screenRect.Expand(static_cast<Int_t>(winX), static_cast<Int_t>(winY));
      }
   }

   return screenRect;
}

////////////////////////////////////////////////////////////////////////////////
/// Convert a 3D world vertex to '3D' viewport (screen) one. The X()/Y()
/// components of the viewport vertex are the horizontal/vertical pixel
/// positions. The Z() component is the viewport depth value - for a
/// default depth range this is 0.0 (at near clip plane) to 1.0 (at far
/// clip plane). See OpenGL gluProject & glDepth documentation
///
/// Camera must have valid frustum cache - call Apply() after last modification, before using

TGLVertex3 TGLCamera::WorldToViewport(const TGLVertex3 & worldVertex,
                                      TGLMatrix* modviewMat) const
{
   if (fCacheDirty) {
      Error("TGLCamera::WorldToViewport()", "cache dirty - must call Apply()");
   }
   TGLVertex3 viewportVertex;
   gluProject(worldVertex[0], worldVertex[1], worldVertex[2],
              modviewMat ? modviewMat->CArr() : fModVM.CArr(),
              fProjM.CArr(), fViewport.CArr(),
              &viewportVertex[0], &viewportVertex[1], &viewportVertex[2]);
   return viewportVertex;
}

////////////////////////////////////////////////////////////////////////////////
/// Convert a 3D vector worldDelta (shift) about vertex worldRef to a viewport
/// (screen) '3D' vector. The X()/Y() components of the vector are the horizontal /
/// vertical pixel deltas. The Z() component is the viewport depth delta - for a
/// default depth range between 0.0 (at near clip plane) to 1.0 (at far clip plane)
/// See OpenGL gluProject & glDepth documentation
///
/// Camera must have valid frustum cache - call Apply()

TGLVector3 TGLCamera::WorldDeltaToViewport(const TGLVertex3 & worldRef,
                                           const TGLVector3 & worldDelta) const
{
   if (fCacheDirty) {
      Error("TGLCamera::WorldToViewport()", "cache dirty - must call Apply()");
   }
   TGLVertex3 other = worldRef + worldDelta;
   TGLVertex3 v1 = WorldToViewport(worldRef);
   TGLVertex3 v2 = WorldToViewport(other);
   return v2 - v1;
}

////////////////////////////////////////////////////////////////////////////////
/// Convert a '3D' viewport vertex to 3D world one. The X()/Y() components
/// of viewportVertex are the horizontal/vertical pixel position.

TGLVertex3 TGLCamera::ViewportToWorld(const TGLVertex3 & viewportVertex,
                                      TGLMatrix* modviewMat) const
{
   // The Z() component is the viewport depth value - for a default
   // depth range this is 0.0 (at near clip plane) to 1.0 (at far clip
   // plane). Without Z() the viewport position corresponds to a line
   // in 3D world space - see:
   // TGLLine3 TGLCamera::ViewportToWorld(Double_t viewportX, Double_t viewportY) const
   //
   // See also OpenGL gluUnProject & glDepth documentation.
   //
   // Camera must have valid frustum cache - call Apply() after last
   // modification, before using.

   if (fCacheDirty) {
      Error("TGLCamera::ViewportToWorld()", "cache dirty - must call Apply()");
   }
   TGLVertex3 worldVertex;
   gluUnProject(viewportVertex[0], viewportVertex[1], viewportVertex[2],
                modviewMat ? modviewMat->CArr() : fModVM.CArr(),
                fProjM.CArr(), fViewport.CArr(),
                &worldVertex[0], &worldVertex[1], &worldVertex[2]);
   return worldVertex;
}

////////////////////////////////////////////////////////////////////////////////
/// Convert a 2D viewport position to 3D world line - the projection of the
/// viewport point into 3D space. Line runs from near to far camera clip planes
/// (the minimum and maximum visible depth). See also
///    TGLVertex3 TGLCamera::ViewportToWorld(const TGLVertex3 & viewportVertex) const
/// for 3D viewport -> 3D world vertex conversions.
/// See also OpenGL gluUnProject & glDepth documentation
///
/// Camera must have valid frustum cache - call Apply() after last modification, before using

TGLLine3 TGLCamera::ViewportToWorld(Double_t viewportX, Double_t viewportY) const
{
   if (fCacheDirty) {
      Error("TGLCamera::Viewport2DToWorldLine()", "cache dirty - must call Apply()");
   }
   // Find world vertices at near and far clip planes, and return line through them
   TGLVertex3 nearClipWorld = ViewportToWorld(TGLVertex3(viewportX, viewportY, 0.0));
   TGLVertex3 farClipWorld = ViewportToWorld(TGLVertex3(viewportX, viewportY, 1.0));
   return TGLLine3(nearClipWorld, farClipWorld - nearClipWorld);
}

////////////////////////////////////////////////////////////////////////////////
/// Convert a 2D viewport position to 3D world line - the projection of the
/// viewport point into 3D space. Line runs from near to far camera clip planes
/// (the minimum and maximum visible depth). See also
///    TGLVertex3 TGLCamera::ViewportToWorld(const TGLVertex3 & viewportVertex) const
/// for 3D viewport -> 3D world vertex conversions.
/// See also OpenGL gluUnProject & glDepth documentation
///
/// Camera must have valid frustum cache - call Apply() after last modification, before using

TGLLine3 TGLCamera::ViewportToWorld(const TPoint & viewport) const
{
   return ViewportToWorld(viewport.GetX(), viewport.GetY());
}

////////////////////////////////////////////////////////////////////////////////
/// Find the intersection of projection of supplied viewport point (a 3D world
/// line - see ViewportToWorld) with supplied world plane. Returns std::pair
/// of Bool_t and TGLVertex3. If line intersects std::pair.first (Bool_t) is
/// kTRUE, and std::pair.second (TGLVertex) contains the intersection vertex.
/// If line does not intersect (line and plane parallel) std::pair.first
/// (Bool_t) if kFALSE, and std::pair.second (TGLVertex) is invalid.
///
/// NOTE: The projection lines is extended for the plane intersection test
/// hence the intersection vertex can lie outside the near/far clip regions
/// (not visible)
///
/// Camera must have valid frustum cache - call Apply() after last modification, before using

std::pair<Bool_t, TGLVertex3> TGLCamera::ViewportPlaneIntersection(Double_t viewportX, Double_t viewportY,
                                                                   const TGLPlane & worldPlane) const
{
   TGLLine3 worldLine = ViewportToWorld(viewportX, viewportY);

   // Find intersection of line with plane
   return Intersection(worldPlane, worldLine, kTRUE /* extended */ );
}

////////////////////////////////////////////////////////////////////////////////
/// Find the intersection of projection of supplied viewport TPoint (a 3D world
/// line - see ViewportToWorld) with supplied world plane. Returns std::pair
/// of bool and vertex. If line intersects
///
/// Camera must have valid frustum cache - call Apply() after last modification, before using

std::pair<Bool_t, TGLVertex3> TGLCamera::ViewportPlaneIntersection(const TPoint & viewport,
                                                                   const TGLPlane & worldPlane) const
{
   return ViewportPlaneIntersection(viewport.GetX(), viewport.GetY(), worldPlane);
}

////////////////////////////////////////////////////////////////////////////////
/// Apply a 2D viewport delta (shift) to the projection of worldRef onto viewport,
/// returning the resultant world vector which equates to it. Useful for making
/// 3D world objects track mouse moves.
///
/// Camera must have valid frustum cache - call Apply()

TGLVector3 TGLCamera::ViewportDeltaToWorld(const TGLVertex3 & worldRef, Double_t viewportXDelta,
                                           Double_t viewportYDelta, TGLMatrix* modviewMat) const
{
   if (fCacheDirty) {
      Error("TGLCamera::ViewportDeltaToWorld()", "cache dirty - must call Apply()");
   }
   TGLVertex3 winVertex = WorldToViewport(worldRef, modviewMat);
   winVertex.Shift(viewportXDelta, viewportYDelta, 0.0);
   return (ViewportToWorld(winVertex, modviewMat) - worldRef);
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate if the an object defined by world frame bounding box
/// is 'of interest' to the camera. This is defined as box:
///
///  1. intersecting completely or partially (kInside/kPartial) with
/// cameras interest box (fInterestBox)
///  2. having significant length OR volume ratio compared to this
/// interest box
///
/// If a box is 'of interest' returns kTRUE, kFALSE otherwise.  See
/// TGLCamera::UpdateInterest() for more details of camera interest
/// box.
///
/// Note: Length/volume ratios NOT dependent on the projected size
/// of box at current camera configuration as we do not want
/// continual changes.  This is used when (re) populating the scene
/// with objects from external client.
///
/// TODO: Might be more logical to move this test out to client -
/// and have accessor for fInterestBox instead?

Bool_t TGLCamera::OfInterest(const TGLBoundingBox & box, Bool_t ignoreSize) const
{
   Bool_t interest = kFALSE;

   // *********** IMPORTANT - Bootstrapping the camera with empty scene
   //
   // Initially the camera can't be Setup() (limits etc) until the
   // scene is populated and it has a valid bounding box to pass to
   // the camera.  However the scene can't be populated without
   // knowing if objects sent are 'of interest' - which needs a camera
   // interest box, made from a properly setup camera frustum - catch
   // 22.
   //
   // To overcome this we track the largest box diagonal seen so far and
   // regard anything over 0.001 of this as 'of interest'. This enables
   // us to get a roughly populated scene with largest objects, setup
   // the camera, and do first draw.  We then do a
   // TGLCamera::UpdateInterest() - which always return kTRUE, and
   // thus fires an internal rebuild to fill scene properly and
   // finally setup camera properly.

   if (fInterestBox.IsEmpty())
   {
      if (box.Diagonal() >= fLargestSeen * 0.001)
      {
         if (box.Diagonal() > fLargestSeen) {
            fLargestSeen = box.Diagonal();
         }
         interest = kTRUE;
      }
   }
   else
   {
      // Objects are of interest if the have length ratio c.f. the
      // current interest box, and they at least partially overlap it.
      // Some objects have zero volume BBs - e.g. single points - skip
      // the test for these as there is no way to threshold on 0.
      if (box.IsEmpty())
      {
         interest = kTRUE;
      }
      else
      {
         if (ignoreSize || box.Diagonal() / fInterestBox.Diagonal() > 0.0001)
            interest = fInterestBox.Overlap(box) != Rgl::kOutside;
      }
   }

   return interest;
}

////////////////////////////////////////////////////////////////////////////////
/// Update the internal interest box (fInterestBox) of the camera.
/// The interest box is an orientated bounding box, calculated as
/// an expanded container round the frustum. It is used to test if
/// if object bounding boxes are of interest (should be accepted
/// into viewer scene) for a camera - see TGLCamera::OfInterest()
///
/// The interest box is updated if the frustum is no longer contained
/// in the existing one, or a new one calculated on the current frustum
/// differs significantly in volume (camera has been zoomed/dollyed
/// sizable amount).
///
/// If the interest box is updated we return kTRUE - kFALSE otherwise.

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

   // Calculate volume ratio of new to old
   Double_t volRatio = 0.0;

   // If the interest box is empty the interest is ALWAYS updated
   // See TGLCamera::OfInterest() comment on bootstrapping
   if (!fInterestBox.IsEmpty()) {
      volRatio = newInterestBox.Volume() / fInterestBox.Volume();
   }

   // Update the existing interest box with new one if:
   //  1. Volume ratio old/new interest has changed significantly
   //  2. The current frustum is not inside existing interest
   //  3. Force case (debugging)
   if (volRatio > 8.0 || volRatio < 0.125 || fInterestBox.IsEmpty() ||
       fInterestBox.Overlap(frustumBox) != Rgl::kInside || force)
   {
      fPreviousInterestBox = fInterestBox;
      fInterestBox = newInterestBox;

      // Frustum should be fully contained now
      if (fInterestBox.Overlap(frustumBox) != Rgl::kInside) {
         Error("TGLCamera::UpdateInterest", "update interest box does not contain frustum");
      }

      exposedUpdate = kTRUE;

      // Keep the real frustum (true and box versions) as debugging aid
      fInterestFrustum = Frustum(kFALSE);
      fInterestFrustumAsBox = frustumBox;

      if (gDebug>2 || force) {
         Info("TGLCamera::UpdateInterest", "changed - volume ratio %f", volRatio );
      }
   }

   return exposedUpdate;
}

////////////////////////////////////////////////////////////////////////////////
/// Clear out the existing interest box

void TGLCamera::ResetInterest()
{
   fInterestBox.SetEmpty();

   // We also reset the bootstrapping variable - see
   // TGLCamera::OfInterest comments.
   fLargestSeen = 0.0;
}

////////////////////////////////////////////////////////////////////////////////
/// Adjust a passed REFERENCE value 'val', based on screenShift delta.
/// Two modifier flags ('mod1' / 'mod2' ) for sensitivity:
///
///  - mod1 = kFALSE, mod2 = kFALSE : normal sensitivity (screenShift/screenShiftRange)
///  - mod1 = kTRUE,  mod2 = kFALSE : 0.1x sensitivity
///  - mod1 = kTRUE,  mod2 = kTRUE  : 0.01x sensitivity
///  - mod1 = kFALSE, mod2 = kTRUE  : 10.0x sensitivity
///
/// 'val' is modified and clamped to 'min' / 'max' range.
/// Return bool kTRUE if val actually changed.
///
/// Used as common interaction function for adjusting zoom/dolly etc

Bool_t TGLCamera::AdjustAndClampVal(Double_t & val, Double_t min, Double_t max,
                                    Int_t screenShift, Int_t screenShiftRange,
                                    Bool_t mod1, Bool_t mod2) const
{
   if (screenShift == 0) {
      return kFALSE;
   }

   // Calculate a sensitivity based on passed modifiers
   Double_t sens = val * static_cast<Double_t>(screenShift);

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
   Double_t shift  = sens / static_cast<Double_t>(screenShiftRange);
   val -= shift;

   if (val < min) {
      val = min;
   }
   else if (val > max) {
      val = max;
   }

   return val != oldVal;
}

////////////////////////////////////////////////////////////////////////////////
/// Adjust a passed screen value and apply modifiers.
/// See AdjustAndClampVal() for details.

Double_t TGLCamera::AdjustDelta(Double_t screenShift, Double_t deltaFactor,
                                Bool_t mod1, Bool_t mod2) const
{
   if (screenShift == 0)
      return 0;

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

   return sens * deltaFactor * screenShift;
}

////////////////////////////////////////////////////////////////////////////////
/// Draw out some debugging aids for the camera:
///
///  1. The frustum used to create the current interest box (RED)
///  2. The same frustum as a squared off box (ORANGE)
///  3. The axis aligned version of the frustum used as interest box basis (YELLOW)
///  4. The current interest box (BLUE)

void TGLCamera::DrawDebugAids() const
{
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

////////////////////////////////////////////////////////////////////////////////
/// Set camera center diffrent than scene center, if enable is kTRUE.

void TGLCamera::SetExternalCenter(Bool_t enable)
{
   if (fExternalCenter == enable)
      return;

   fExternalCenter = enable;
   if (fExternalCenter)
      fCenter = &fExtCenter;
   else
      fCenter = &fDefCenter;

   TGLMatrix bt = fCamBase * fCamTrans;
   fCamBase.SetBaseVec(4, *fCenter);
   TGLMatrix binv = fCamBase; binv.Invert();
   fCamTrans = binv * bt;

   IncTimeStamp();
}

////////////////////////////////////////////////////////////////////////////////
/// Set camera center vector.

void TGLCamera::SetCenterVec(Double_t x, Double_t y, Double_t z)
{
   if (fExternalCenter)
      fExtCenter.Set(x, y, z);
   else
      fDefCenter.Set(x, y, z);

   TGLMatrix bt = fCamBase * fCamTrans;
   fCamBase.SetBaseVec(4, *fCenter);
   TGLMatrix binv = fCamBase; binv.Invert();
   fCamTrans = binv * bt;

   IncTimeStamp();
}

////////////////////////////////////////////////////////////////////////////////
/// Set camera center vector and do not keep the same combined
/// camera transformation matrix.
/// It appears as if the camera warped to the new center.

void TGLCamera::SetCenterVecWarp(Double_t x, Double_t y, Double_t z)
{
   if (fExternalCenter)
      fExtCenter.Set(x, y, z);
   else
      fDefCenter.Set(x, y, z);

   fCamBase.SetBaseVec(4, *fCenter);

   IncTimeStamp();
}

////////////////////////////////////////////////////////////////////////////////
/// Get angle between camera up axis.

Double_t TGLCamera::GetTheta() const
{
   TGLVector3 fwd  = fCamTrans.GetBaseVec(1);
   TGLVector3 zdir = fCamBase.GetBaseVec(3);
   fCamBase.RotateIP(fwd);
   return TMath::ACos(fwd*zdir);
}

////////////////////////////////////////////////////////////////////////////////
/// Truck the camera - 'move camera parallel to film plane'.
/// Returns kTRUE is redraw required (camera change), kFALSE otherwise.

Bool_t TGLCamera::Truck(Double_t xDelta, Double_t yDelta)
{
   if (xDelta != 0 || yDelta != 0)
   {
      fCamTrans.MoveLF(2, xDelta);
      fCamTrans.MoveLF(3, yDelta);

      IncTimeStamp();
      return kTRUE;
   }
   else
   {
      return kFALSE;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Rotate the camera round view volume center established in Setup().
/// Arguments are:
///  - xDelta - horizontal delta (pixels)
///  - YDelta - vertical delta (pixels)

Bool_t TGLCamera::Rotate(Int_t xDelta, Int_t yDelta, Bool_t mod1, Bool_t mod2)
{
   Double_t vRotate = AdjustDelta(xDelta, TMath::TwoPi() / fViewport.Width(), mod1, mod2);
   Double_t hRotate = AdjustDelta(yDelta, TMath::Pi()   / fViewport.Height(), mod1, mod2);

   return RotateRad(hRotate, vRotate);
}

////////////////////////////////////////////////////////////////////////////////
/// Rotate camera around center.

Bool_t TGLCamera::RotateRad(Double_t hRotate, Double_t vRotate)
{
   using namespace TMath;

   if (fWasArcBalled)
   {
      Double_t *M = fCamTrans.Arr();
      Double_t  d = M[2];
      if      (d >  1) d =  1;
      else if (d < -1) d = -1; // Fix numerical errors

      Double_t theta = ASin(d);
      Double_t phi   = Abs(Cos(theta)) > 8.7e-6 ? ATan2(M[1], M[0]) : ATan2(-M[4], M[5]);

      M[0] = M[5] = M[10] = 1;
      M[1] = M[2] = M[4] = M[6] = M[8] = M[9] = 0;
      fCamTrans.RotateLF(1, 2, phi);
      fCamTrans.RotateLF(1, 3, theta);
   }

   if (hRotate != 0.0 || fWasArcBalled)
   {
      TGLVector3 fwd  = fCamTrans.GetBaseVec(1);
      TGLVector3 lft  = fCamTrans.GetBaseVec(2);
      TGLVector3 up   = fCamTrans.GetBaseVec(3);
      TGLVector3 pos  = fCamTrans.GetTranslation();

      Double_t deltaF = pos * fwd;
      Double_t deltaU = pos * up;

      // up vector lock
      TGLVector3 zdir = fCamBase.GetBaseVec(3);

      fCamBase.RotateIP(fwd);
      Double_t theta = ACos(fwd*zdir);
      if (theta + hRotate < fVAxisMinAngle)
         hRotate = fVAxisMinAngle - theta;
      else if (theta + hRotate > Pi() - fVAxisMinAngle)
         hRotate = Pi() - fVAxisMinAngle - theta;

      fCamTrans.MoveLF(1, -deltaF);
      fCamTrans.MoveLF(3, -deltaU);
      fCamTrans.RotateLF(3, 1, hRotate);
      fCamTrans.MoveLF(3,  deltaU);
      fCamTrans.MoveLF(1,  deltaF);

      fWasArcBalled = kFALSE;
   }
   if (vRotate != 0.0)
   {
      fCamTrans.RotatePF(1, 2, -vRotate);
   }

   IncTimeStamp();
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Rotate the camera round view volume center established in Setup().
/// Arguments are:
///  - xDelta - horizontal delta (pixels)
///  - YDelta - vertical delta (pixels)

Bool_t TGLCamera::RotateArcBall(Int_t xDelta, Int_t yDelta, Bool_t mod1, Bool_t mod2)
{
   Double_t vRotate = AdjustDelta(xDelta, TMath::TwoPi() / fViewport.Width(), mod1, mod2);
   Double_t hRotate = AdjustDelta(yDelta, TMath::Pi()   / fViewport.Height(), mod1, mod2);

   return RotateArcBallRad(hRotate, vRotate);
}

////////////////////////////////////////////////////////////////////////////////
/// Rotate camera around center.

Bool_t TGLCamera::RotateArcBallRad(Double_t hRotate, Double_t vRotate)
{
   using namespace TMath;

   TGLVector3 fwd  = fCamTrans.GetBaseVec(1);
   TGLVector3 lft  = fCamTrans.GetBaseVec(2);
   TGLVector3 up   = fCamTrans.GetBaseVec(3);
   TGLVector3 pos  = fCamTrans.GetTranslation();

   Double_t deltaF = pos * fwd;
   Double_t deltaL = pos * lft;
   Double_t deltaU = pos * up;

   fCamTrans.MoveLF(1, -deltaF);
   fCamTrans.MoveLF(2, -deltaL);
   fCamTrans.MoveLF(3, -deltaU);

   if (hRotate != 0.0)
   {
      fCamTrans.RotateLF(3, 1, hRotate);
   }
   if (vRotate != 0.0)
   {
      fCamTrans.RotateLF(1, 2, -vRotate);
   }

   fCamTrans.MoveLF(3, deltaU);
   fCamTrans.MoveLF(2, deltaL);
   fCamTrans.MoveLF(1, deltaF);

   fWasArcBalled = kTRUE;

   IncTimeStamp();
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Dolly the camera - 'move camera along eye line, retaining lens focal length'.
/// Arguments are:
///
///  - 'delta' - mouse viewport delta (pixels) - +ive dolly in, -ive dolly out
///  - 'mod1' / 'mod2' - sensitivity modifiers - see TGLCamera::AdjustAndClampVal()
///
/// Returns kTRUE is redraw required (camera change), kFALSE otherwise.

Bool_t TGLCamera::Dolly(Int_t delta, Bool_t mod1, Bool_t mod2)
{
   Double_t step = AdjustDelta(delta, fDollyDistance, mod1, mod2);
   if (step == 0)
      return kFALSE;

   fCamTrans.MoveLF(1, -step);

   IncTimeStamp();
   return kTRUE;
}
