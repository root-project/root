// @(#)root/g3d:$Id$
// Author: Matevz Tadel  7/4/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include "TPointSet3D.h"
#include "TClass.h"

//______________________________________________________________________
//
// TPolyMarker3D using TPointSet3DGL for direct OpenGL rendering.
// Supports only elementary marker types:
// 4, 20, 24 : round points, size in pixels;
//   2, 3, 5 : crosses, size in scene units;
//        28 : as above, line width 2 pixels;
// all other : square points, size in pixels.
//
// Marker-size (from TAttMarker) is multiplied by 5!
//
// An identification of type TObject* can be assigned to each point
// via SetPointId() method. Set the fOwnIds flag if the ids are owned
// by the point-set and should be deleted when pointset is cleared or
// destructed.
//
// Copy-constructor and assignment operator COPIES the ids if the are
// not owned and CLONES them if they are owned.
//
// The ids are not streamed.

ClassImp(TPointSet3D);

//______________________________________________________________________________
TPointSet3D::TPointSet3D(const TPointSet3D &t) :
   TPolyMarker3D(t), TAttBBox(t), fOwnIds(kFALSE), fIds()
{
   // Copy constructor.

   CopyIds(t);
}

//______________________________________________________________________________
TPointSet3D::~TPointSet3D()
{
   // Destructor.

   ClearIds();
}

//______________________________________________________________________________
void TPointSet3D::CopyIds(const TPointSet3D& t)
{
   // Copy id objects from point-set 't'.

   fOwnIds = t.fOwnIds;
   fIds.Expand(t.fIds.GetSize());
   if (fOwnIds) {
      for (Int_t i=0; i<t.fIds.GetSize(); ++i)
         fIds.AddAt(t.fIds.At(i)->Clone(), i);
   } else {
      for (Int_t i=0; i<t.fIds.GetSize(); ++i)
         fIds.AddAt(t.fIds.At(i), i);
   }
}

//______________________________________________________________________________
TPointSet3D& TPointSet3D::operator=(const TPointSet3D& t)
{
   // Assignement operator.

   if (this != &t) {
      ClearIds();
      TPolyMarker3D::operator=(t);
      CopyIds(t);
   }
   return *this;
}

//______________________________________________________________________________
void TPointSet3D::ComputeBBox()
{
   // Compute the bounding box of this points set.

   if (Size() > 0) {
      BBoxInit();
      Int_t    n = Size();
      Float_t* p = fP;
      for (Int_t i = 0; i < n; ++i, p += 3) {
         BBoxCheckPoint(p);
      }
   } else {
      BBoxZero();
   }
}
//______________________________________________________________________________
void TPointSet3D::SetPointId(TObject* id)
{
   // Set id of last point.
   // Use this method if you also use TPolyMarker3D::SetNextPoint().

   SetPointId(fLastPoint, id);
}

//______________________________________________________________________________
void TPointSet3D::SetPointId(Int_t n, TObject* id)
{
   // Set id of point n.

   if (n >= fN) return;
   if (fN > fIds.GetSize())
      fIds.Expand(fN);
   fIds.AddAt(id, n);
}

//______________________________________________________________________________
void TPointSet3D::ClearIds()
{
   // Clears the id-array. If ids are owned the TObjects are deleted.

   if (fOwnIds) {
      for (Int_t i=0; i<fIds.GetSize(); ++i)
         delete GetPointId(i);
   }
   fIds.Expand(0);
}

//______________________________________________________________________________
void TPointSet3D::PointSelected(Int_t n)
{
   // This virtual method is called from TPointSet3DGL when a point is
   // selected.
   // At this point it just prints out n and id of the point (if it exists).
   // To make something useful out of this do:
   //  a) subclass and re-implement this method;
   //  b) extend this class to include TExec or some other kind of callback.

   TObject* id = GetPointId(n);
   printf("TPointSet3D::PointSelected n=%d, id=(%s*)0x%lx\n",
          n, id ? id->IsA()->GetName() : "void", (ULong_t)id);
   if (id)
      id->Print();
}

//______________________________________________________________________________
void TPointSet3D::Streamer(TBuffer &R__b)
{
   // Stream an object of class TPointSet3D.

   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(TPointSet3D::Class(), this);
      if (fOwnIds) {
         Int_t n;
         R__b >> n;
         for (Int_t i=0; i<n; ++i) {
            TObject* o = (TObject*) R__b.ReadObjectAny(TObject::Class());
            if (gDebug > 0) printf("Read[%2d]: ", i); o->Print();
         }
      }
   } else {
      R__b.WriteClassBuffer(TPointSet3D::Class(), this);
      if (fOwnIds) {
         R__b << fIds.GetEntries();
         TObject* o;
         TIter next(&fIds);
         while ((o = next())) {
            if (gDebug > 0) printf("Writing: "); o->Print();
            R__b.WriteObjectAny(o, TObject::Class());
         }
      }
   }
}
