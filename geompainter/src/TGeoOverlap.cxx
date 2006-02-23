// @(#)root/geom:$Name:  $:$Id: TGeoOverlap.cxx,v 1.8 2005/11/18 16:07:59 brun Exp $
// Author: Andrei Gheata   09-02-03

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TNamed.h"
#include "TBrowser.h"
#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoNode.h"
#include "TPolyMarker3D.h"
#include "TVirtualGeoPainter.h"

#include "TGeoOverlap.h"

ClassImp(TGeoOverlap)
/*************************************************************************
 * TGeoOverlap - base class describing geometry overlaps. Overlaps apply
 *   to the nodes contained inside a volume. These should not overlap to
 *   each other nor extrude the shape of their mother volume.
 *
 *************************************************************************/

//______________________________________________________________________________
TGeoOverlap::TGeoOverlap()
{
// Default ctor.
   fOverlap = 0;
   fVolume1 = 0;
   fVolume2 = 0;
   fMatrix1 = 0;
   fMatrix2 = 0;
   fMarker  = 0;
}

//______________________________________________________________________________
TGeoOverlap::TGeoOverlap(const char *name, TGeoVolume *vol1, TGeoVolume *vol2,
                         const TGeoMatrix *matrix1, const TGeoMatrix *matrix2,
                         Bool_t isovlp, Double_t ovlp)
            :TNamed("",name)
{
// Creates a named overlap belonging to volume VOL and having the size OVLP.
   fOverlap = ovlp;
   fVolume1  = vol1;
   fVolume2  = vol2;
   fMatrix1 = new TGeoHMatrix();
   *fMatrix1 = matrix1;
   fMatrix2 = new TGeoHMatrix();
   *fMatrix2 = matrix2;
   fMarker  = new TPolyMarker3D();
   fMarker->SetMarkerColor(2);
   SetIsOverlap(isovlp);
   fMarker->SetMarkerStyle(6);
//   fMarker->SetMarkerSize(0.5);
}

//______________________________________________________________________________
TGeoOverlap::~TGeoOverlap()
{
// Destructor.
   if (fMarker) delete fMarker;
   if (fMatrix1) delete fMatrix1;
   if (fMatrix2) delete fMatrix2;
}

//______________________________________________________________________________
void TGeoOverlap::Browse(TBrowser *b)
{
// Define double-click action
   if (!b) return;
   Draw();
}

//______________________________________________________________________________
Int_t TGeoOverlap::Compare(const TObject *obj) const
{
// Method to compare this overlap with another. Returns :
//   -1 - this is smaller than OBJ
//    0 - equal
//    1 - greater 
   TGeoOverlap *other = 0;
   other = (TGeoOverlap*)obj;
   if (!other) {
      Error("Compare", "other object is not TGeoOverlap");
      return 0;
   }
   if (IsExtrusion()) {
      if (other->IsExtrusion()) return (fOverlap<=other->GetOverlap())?1:-1;
      return -1;
   } else {   
      if (other->IsExtrusion()) return 1;
      return (fOverlap<=other->GetOverlap())?1:-1;
   }
}

//______________________________________________________________________________
Int_t TGeoOverlap::DistancetoPrimitive(Int_t px, Int_t py)
{
// Distance to primitive for an overlap.
   return fVolume1->GetGeoManager()->GetGeomPainter()->DistanceToPrimitiveVol(fVolume1, px, py);
}

//______________________________________________________________________________
void TGeoOverlap::Draw(Option_t *option)
{
// Draw the overlap. One daughter will be blue, the other green,
// extruding points red.
   fVolume1->GetGeoManager()->GetGeomPainter()->DrawOverlap(this, option);
   PrintInfo();
}

//______________________________________________________________________________
void TGeoOverlap::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
// Event interception.
   fVolume1->GetGeoManager()->GetGeomPainter()->ExecuteVolumeEvent(fVolume1, event, px, py);
}

//______________________________________________________________________________
void TGeoOverlap::Paint(Option_t *option)
{
// Paint the overlap.
   fVolume1->GetGeoManager()->GetGeomPainter()->PaintOverlap(this, option);
}

//______________________________________________________________________________
void TGeoOverlap::PrintInfo() const
{
// Print some info.
   printf(" = Overlap %s: %s\n", GetName(), GetTitle());
}

//______________________________________________________________________________
void TGeoOverlap::SetNextPoint(Double_t x, Double_t y, Double_t z)
{
// Set next overlapping point.
   fMarker->SetNextPoint(x,y,z);
}

//______________________________________________________________________________
void TGeoOverlap::Sizeof3D() const
{
// Get 3D size of this.
   fVolume1->GetShape()->Sizeof3D();
   fVolume2->GetShape()->Sizeof3D();
}
