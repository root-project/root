// @(#)root/g3d:$Name:  $:$Id: TPointSet3D.cxx,v 1.4 2006/05/09 19:08:44 brun Exp $
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
// TPointSet3D
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
// An identification of type TObject* can be assigned to each point via
// SetPointId() method. Copy-constructor and assignment operator do
// not copy these ids. They are not streamed either. Set the fOwnIds flag
// if the ids are owned by the point-set and should be deleted when pointset
// is cleared/destructed. Set fFakeIds if ids are of some other type
// casted to TObject*.

ClassImp(TPointSet3D)

//______________________________________________________________________________
TPointSet3D& TPointSet3D::operator=(const TPointSet3D& tp3)
{
   // Assignement operator; clears id array,

   if(this!=&tp3) {
      ClearIds();
      TPolyMarker3D::operator=(tp3);
   }
   return *this;
}

//______________________________________________________________________________
TPointSet3D::~TPointSet3D()
{
   // Destructor.

   ClearIds();
}

//______________________________________________________________________________
void TPointSet3D::ComputeBBox()
{
   // Compute the bounding box of this points set.
   if (Size() > 0) {
      Int_t    n = Size();
      Float_t* p = fP;
      BBoxInit();
      while (n--) {
         BBoxCheckPoint(p);
         p += 3;
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
   if (fN > fNIds) {
      TObject** idarr = new TObject* [fN];
      if (fIds && fNIds) {
         memcpy(idarr, fIds, fNIds*sizeof(TObject*));
         memset(idarr+fNIds, 0, (fN-fNIds)*sizeof(TObject*));
         delete [] fIds;
      }
      fIds  = idarr;
      fNIds = fN;
   }
   fIds[n] = id;
}

//______________________________________________________________________________
TObject* TPointSet3D::GetPointId(Int_t n) const
{
   // Get id of point n.
   // If n is out of range 0 is returned.

   if (n < 0 || n >= fNIds) return 0;
   return fIds[n];
}

//______________________________________________________________________________
void TPointSet3D::ClearIds()
{
   // Clears the id-array. If ids are owned the TObjects are deleted.

   if (fNIds <= 0) return;
   if (fOwnIds) {
      for (Int_t i=0; i<fNIds; ++i)
         if (fIds[i]) delete fIds[i];
   }
   delete [] fIds;
   fNIds = 0;
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
   Bool_t idok = (id != 0 && fFakeIds == kFALSE);
   printf("TPointSet3D::PointSelected n=%d, id=(%s*)0x%lx\n",
          n, idok ? id->IsA()->GetName() : "void", (ULong_t)id);
   if (idok)
      id->Print();
}
