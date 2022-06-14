// @(#)root/gl:$Id$
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
#include "TGLRnrCtx.h"
#include "TGLManipSet.h"

#include "TGLFaceSet.h"
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"

namespace
{

class TGLClipPlaneLogical : public TGLLogicalShape
{
protected:
   virtual void DirectDraw(TGLRnrCtx & rnrCtx) const
   {
      glBegin(rnrCtx.IsDrawPassFilled() ? GL_QUADS : GL_LINE_LOOP);
      glNormal3d (0.0, 0.0, 1.0);
      glVertex3dv(fBoundingBox[4].CArr());
      glVertex3dv(fBoundingBox[7].CArr());
      glVertex3dv(fBoundingBox[6].CArr());
      glVertex3dv(fBoundingBox[5].CArr());
      glEnd();
   }

public:
   TGLClipPlaneLogical() : TGLLogicalShape() { fDLCache = kFALSE; }
   virtual ~TGLClipPlaneLogical() {}

   void Resize(Double_t ext)
   {
      fBoundingBox.SetAligned(TGLVertex3(-ext, -ext, 0),
                              TGLVertex3( ext,  ext, 0));
      UpdateBoundingBoxesOfPhysicals();
   }

};


class TGLClipBoxLogical : public TGLLogicalShape
{
protected:
   virtual void DirectDraw(TGLRnrCtx & rnrCtx) const
   {
      glEnable(GL_NORMALIZE);
      fBoundingBox.Draw(rnrCtx.IsDrawPassFilled());
      glDisable(GL_NORMALIZE);
   }

public:
   TGLClipBoxLogical() : TGLLogicalShape() { fDLCache = kFALSE; }
   virtual ~TGLClipBoxLogical() {}

   void Resize(const TGLVertex3 & lowVertex, const TGLVertex3 & highVertex)
   {
      fBoundingBox.SetAligned(lowVertex, highVertex);
      UpdateBoundingBoxesOfPhysicals();
   }
};

}


/** \class TGLClip
\ingroup opengl
Abstract clipping shape - derives from TGLPhysicalShape
Adds clip mode (inside/outside) and pure virtual method to
approximate shape as set of planes. This plane set is used to perform
interactive clipping using OpenGL clip planes.
*/

ClassImp(TGLClip);

////////////////////////////////////////////////////////////////////////////////
/// Construct a stand-alone physical clipping object.

TGLClip::TGLClip(const TGLLogicalShape & logical, const TGLMatrix & transform, const float color[4]) :
   TGLPhysicalShape(0, logical, transform, kTRUE, color),
   fMode      (kInside),
   fTimeStamp (1),
   fValid     (kFALSE)
{
   logical.StrongRef(kTRUE);
}

////////////////////////////////////////////////////////////////////////////////
/// Destroy clip object.

TGLClip::~TGLClip()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Setup the clipping object with two vectors.
/// The interpretation of the two is different for plane and box
/// clipping objects.

void TGLClip::Setup(const TGLVector3&, const TGLVector3&)
{
   Warning("TGLClip::Setup", "Called on base-class -- should be re-implemented in derived class.");

}

////////////////////////////////////////////////////////////////////////////////
/// Draw out clipping object with blending and back + front filling.
/// Some clip objects are single face which we want to see both sides of.

void TGLClip::Draw(TGLRnrCtx & rnrCtx) const
{
   glDepthMask(GL_FALSE);
   glEnable(GL_BLEND);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   glDisable(GL_CULL_FACE);
   glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

   TGLPhysicalShape::Draw(rnrCtx);

   glPolygonMode(GL_FRONT, GL_FILL);
   glEnable(GL_CULL_FACE);
   glDisable(GL_BLEND);
   glDepthMask(GL_TRUE);
}

/** \class TGLClipPlane
\ingroup opengl
Concrete clip plane object. This can be translated in all directions
rotated about the Y/Z local axes (the in-plane axes). It cannot be
scaled.
*/

ClassImp(TGLClipPlane);

const float TGLClipPlane::fgColor[4] = { 1.0, 0.6, 0.2, 0.5 };

////////////////////////////////////////////////////////////////////////////////
/// Construct a clip plane object, based on supplied 'plane', with
/// initial manipulation pivot at 'center', with drawn extents (in
/// local x/y axes) of 'extents'
///
/// Plane can have center pivot translated in all directions, and
/// rotated round center in X/Y axes , the in-plane axes. It cannot
/// be scaled
///
/// Note theoretically a plane is of course infinite - however we
/// want to draw the object in viewer - so we fake it with a single
/// GL face (polygon) - extents defines the width/depth of this -
/// should be several times scene extents - see Setup().

TGLClipPlane::TGLClipPlane() :
   TGLClip(* new TGLClipPlaneLogical, TGLMatrix(), fgColor)
{
   SetManip(EManip(kTranslateAll | kRotateX | kRotateY));

   TGLPlane plane(0.0, -1.0, 0.0, 0.0);
   Set(plane);
   fValid = kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Destroy clip plane object

TGLClipPlane::~TGLClipPlane()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Setup the clip object for scene encompassed by bbox.

void TGLClipPlane::Setup(const TGLBoundingBox & bbox)
{
   Double_t extents = bbox.Extents().Mag();
   TGLClipPlaneLogical* cpl = (TGLClipPlaneLogical*) GetLogical();
   cpl->Resize(extents);
   if (!fValid) {
      SetTransform(TGLMatrix(bbox.Center(), BoundingBox().GetNearPlane().Norm()));
   }
   IncTimeStamp();
   fValid = kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Setup the clipping plane by point and normal.
/// Length of the normal determines the size of the plane drawn in
/// GL viewer. The normal points into the direction of visible half-plane.
///
/// This only makes sense if you disable auto-update of the
/// clip-object:
///
///     gl_viewer->SetClipAutoUpdate(kFALSE).
///
/// After calling this also call gl_viewer->RefreshPadEditor(gl_viewer)
/// and gl_viewer->RequestDraw().

void TGLClipPlane::Setup(const TGLVector3& point, const TGLVector3& normal)
{
   TGLVector3 n(normal);
   Double_t extents = n.Mag();
   if (extents > 0)
   {
      n /= extents;
      TGLClipPlaneLogical* cpl = (TGLClipPlaneLogical*) GetLogical();
      cpl->Resize(extents);
      SetTransform(TGLMatrix(point, n));

      IncTimeStamp();
      fValid = kTRUE;
   }
   else
   {
      Warning("TGLClipPlane::Setup", "Normal with zero length passed.");
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Update clip plane object to follow passed 'plane' equation. Center pivot
/// is shifted to nearest point on new plane.

void TGLClipPlane::Set(const TGLPlane& plane)
{
   TGLVertex3 oldCenter = BoundingBox().Center();
   TGLVertex3 newCenter = plane.NearestOn(oldCenter);
   SetTransform(TGLMatrix(newCenter, plane.Norm()));
   IncTimeStamp();
   fValid = kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Return set of planes (actually a single one) describing this clip plane.

void TGLClipPlane::PlaneSet(TGLPlaneSet_t& set) const
{
   set.resize(1);
   set[0] = BoundingBox().GetNearPlane();
   set[0].Negate();
}

/** \class TGLClipBox
\ingroup opengl
Concrete clip box object. Can be translated, rotated and scaled in
all (xyz) axes. By default inside of the box is clipped away.
*/

ClassImp(TGLClipBox);

const float TGLClipBox::fgColor[4] = { 1.0, 0.6, 0.2, 0.3 };

////////////////////////////////////////////////////////////////////////////////
/// Construct an (initially) axis aligned clip pbox object, extents
/// 'halfLengths', centered on 'center' vertex.
/// Box can be translated, rotated and scaled in all (xyz) local axes.

TGLClipBox::TGLClipBox() :
   TGLClip(* new TGLClipBoxLogical, TGLMatrix(), fgColor)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Destroy clip box object.

TGLClipBox::~TGLClipBox()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Setup the clip object for scene encompassed by bbox.

void TGLClipBox::Setup(const TGLBoundingBox& bbox)
{
   TGLVector3 halfLengths = bbox.Extents() * 0.2501;
   TGLVertex3 center      = bbox.Center() + halfLengths;

   TGLClipBoxLogical* cbl = (TGLClipBoxLogical*) GetLogical();
   cbl->Resize(center - halfLengths, center + halfLengths);

   IncTimeStamp();
   fValid = kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Setup the clip box with min/max points directly.
///
/// This only makes sense if you disable auto-update of the
/// clip-object:
///
///     gl_viewer->SetClipAutoUpdate(kFALSE).
///
/// After calling this also call gl_viewer->RefreshPadEditor(gl_viewer)
/// and gl_viewer->RequestDraw().

void TGLClipBox::Setup(const TGLVector3& min_point, const TGLVector3& max_point)
{
   TGLClipBoxLogical* cbl = (TGLClipBoxLogical*) GetLogical();
   cbl->Resize(min_point, max_point);

   IncTimeStamp();
   fValid = kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Return set of 6 planes describing faces of the box but invert them
/// so that they point inside of box.

void TGLClipBox::PlaneSet(TGLPlaneSet_t& set) const
{
   BoundingBox().PlaneSet(set);
   TGLPlaneSet_i i = set.begin();
   while (i != set.end()) {
      i->Negate();
      ++i;
   }
}


/** \class TGLClipSet
\ingroup opengl
A collection of concrete TGLClip objects to be selected from.
*/

ClassImp(TGLClipSet);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TGLClipSet::TGLClipSet() :
   TGLOverlayElement(kViewer),
   fClipPlane   (new TGLClipPlane),
   fClipBox     (new TGLClipBox),
   fCurrentClip (0),
   fAutoUpdate  (kTRUE),
   fShowClip    (kFALSE),
   fShowManip   (kFALSE),
   fManip       (new TGLManipSet)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TGLClipSet::~TGLClipSet()
{
   delete fClipPlane;
   delete fClipBox;
   delete fManip;
}

////////////////////////////////////////////////////////////////////////////////
/// Mouse has entered this element.
/// Forward to ManipSet.

Bool_t TGLClipSet::MouseEnter(TGLOvlSelectRecord& selRec)
{
   return fManip->MouseEnter(selRec);
}

Bool_t TGLClipSet::MouseStillInside(TGLOvlSelectRecord& selRec)
{
   // A new overlay hit is about to be processed.
   // Forward to ManipSet.

   return fManip->MouseStillInside(selRec);
}

////////////////////////////////////////////////////////////////////////////////
/// Handle overlay event.
/// Forward to ManipSet.

Bool_t TGLClipSet::Handle(TGLRnrCtx& rnrCtx, TGLOvlSelectRecord& selRec,
                          Event_t* event)
{
   return fManip->Handle(rnrCtx, selRec, event);
}

////////////////////////////////////////////////////////////////////////////////
/// Mouse has left the element.
/// Forward to ManipSet.

void TGLClipSet::MouseLeave()
{
   return fManip->MouseLeave();
}

////////////////////////////////////////////////////////////////////////////////
/// Render clip-shape and manipulator.

void TGLClipSet::Render(TGLRnrCtx& rnrCtx)
{
   if (!fCurrentClip) return;

   rnrCtx.SetShapeLOD(TGLRnrCtx::kLODHigh);
   rnrCtx.SetDrawPass(TGLRnrCtx::kPassFill);
   if (fShowClip && ! rnrCtx.Selection())
   {
      fCurrentClip->Draw(rnrCtx);
   }
   if (fShowManip)
   {
      fManip->Render(rnrCtx);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Forward request to fill the plane-set to the current clip.

void TGLClipSet::FillPlaneSet(TGLPlaneSet_t& set) const
{
   if (fCurrentClip)
      fCurrentClip->PlaneSet(set);
}

////////////////////////////////////////////////////////////////////////////////
/// Setup clipping objects for given scene bounding box.

void TGLClipSet::SetupClips(const TGLBoundingBox& sceneBBox)
{
   fLastBBox = sceneBBox;
   fClipPlane->Setup(sceneBBox);
   fClipBox  ->Setup(sceneBBox);
}

////////////////////////////////////////////////////////////////////////////////
/// Setup current clipping object for given scene bounding box.

void TGLClipSet::SetupCurrentClip(const TGLBoundingBox& sceneBBox)
{
   fLastBBox = sceneBBox;
   if (fCurrentClip)
      fCurrentClip->Setup(sceneBBox);
}

////////////////////////////////////////////////////////////////////////////////
/// Setup current clipping object for given scene bounding box.

void TGLClipSet::SetupCurrentClipIfInvalid(const TGLBoundingBox& sceneBBox)
{
   fLastBBox = sceneBBox;
   if (fCurrentClip && ! fCurrentClip->IsValid())
      fCurrentClip->Setup(sceneBBox);
}

////////////////////////////////////////////////////////////////////////////////
/// Invalidate clip objects.

void TGLClipSet::InvalidateClips()
{
   fClipPlane->Invalidate();
   fClipBox  ->Invalidate();
}

////////////////////////////////////////////////////////////////////////////////
/// Invalidate current clip object.

void TGLClipSet::InvalidateCurrentClip()
{
   if (fCurrentClip)
      fCurrentClip->Invalidate();
}

////////////////////////////////////////////////////////////////////////////////
/// Get state of clip object 'type' into data vector:
///
/// 'type' requested - 'data' contents returned
/// kClipPlane   4 components - A,B,C,D - of plane eq : Ax+By+CZ+D = 0
/// kBoxPlane    6 components - Box Center X/Y/Z - Box Extents X/Y/Z

void TGLClipSet::GetClipState(TGLClip::EType type, Double_t data[6]) const
{
   switch (type)
   {
      case TGLClip::kClipNone:
         break;

      case TGLClip::kClipPlane:
      {
         if (!fClipPlane->IsValid())
            fClipPlane->Setup(fLastBBox);
         TGLPlaneSet_t planes;
         fClipPlane->PlaneSet(planes);
         data[0] = planes[0].A();
         data[1] = planes[0].B();
         data[2] = planes[0].C();
         data[3] = planes[0].D();
         break;
      }
      case TGLClip::kClipBox:
      {
         if (!fClipBox->IsValid())
            fClipBox->Setup(fLastBBox);
         const TGLBoundingBox & box = fClipBox->BoundingBox();
         TGLVector3 ext = box.Extents();
         data[0] = box.Center().X();
         data[1] = box.Center().Y();
         data[2] = box.Center().Z();
         data[3] = box.Extents().X();
         data[4] = box.Extents().Y();
         data[5] = box.Extents().Z();
         break;
      }
      default:
         Error("TGLClipSet::GetClipState", "invalid clip type '%d'.", type);
         break;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set state of clip object 'type' into data vector:
///
/// 'type' specified        'data' contents interpretation
/// kClipNone               ignored
/// kClipPlane              4 components - A,B,C,D - of plane eq : Ax+By+CZ+D = 0
/// kBoxPlane               6 components - Box Center X/Y/Z - Box Extents X/Y/Z

void TGLClipSet::SetClipState(TGLClip::EType type, const Double_t data[6])
{
   switch (type) {
      case TGLClip::kClipNone: {
         break;
      }
      case TGLClip::kClipPlane: {
         TGLPlane newPlane(-data[0], -data[1], -data[2], -data[3]);
         fClipPlane->Set(newPlane);
         break;
      }
      case TGLClip::kClipBox: {
         //TODO: Pull these inside TGLPhysicalShape
         // Update clip box center
         const TGLBoundingBox & currentBox = fClipBox->BoundingBox();
         TGLVector3 shift(data[0] - currentBox.Center().X(),
                          data[1] - currentBox.Center().Y(),
                          data[2] - currentBox.Center().Z());
         fClipBox->Translate(shift);
         // Update clip box extents

         TGLVector3 currentScale = fClipBox->GetScale();
         TGLVector3 newScale(data[3] / currentBox.Extents().X() * currentScale.X(),
                             data[4] / currentBox.Extents().Y() * currentScale.Y(),
                             data[5] / currentBox.Extents().Z() * currentScale.Z());

         fClipBox->Scale(newScale);
         break;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Get current type active in viewer - returns one of kClipNone
/// kClipPlane or kClipBox.

TGLClip::EType TGLClipSet::GetClipType() const
{
  TGLClip::EType type;
   if (fCurrentClip == 0) {
      type = TGLClip::kClipNone;
   } else if (fCurrentClip == fClipPlane) {
      type = TGLClip::kClipPlane;
   } else if (fCurrentClip == fClipBox) {
      type = TGLClip::kClipBox;
   } else {
      Error("TGLClipSet::GetClipType" , "Unknown clip type");
      type = TGLClip::kClipNone;
   }
   return type;
}

////////////////////////////////////////////////////////////////////////////////
/// Set current clip active in viewer - 'type' is one of kClipNone
/// kClipPlane or kClipBox.

void TGLClipSet::SetClipType(TGLClip::EType type)
{
   switch (type) {
      case TGLClip::kClipNone: {
         fCurrentClip = 0;
         break;
      }
      case TGLClip::kClipPlane: {
         fCurrentClip = fClipPlane;
         break;
      }
      case TGLClip::kClipBox: {
         fCurrentClip = fClipBox;
         break;
      }
      default: {
         Error("TGLClipSet::SetClipType" , "Unknown clip type");
         break;
      }
   }
   fManip->SetPShape(fCurrentClip);
}
