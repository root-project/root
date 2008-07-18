// @(#)root/geom:$Id$
// Author: Andrei Gheata   09-02-03

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TVirtualPad.h"
#include "TMath.h"
#include "TNamed.h"
#include "TBrowser.h"
#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoNode.h"
#include "TGeoBBox.h"
#include "TRandom3.h"
#include "TPolyMarker3D.h"
#include "TVirtualGeoPainter.h"

#include "TGeoOverlap.h"

ClassImp(TGeoOverlap)

//______________________________________________________________________________
// TGeoOverlap - base class describing geometry overlaps. Overlaps apply
//   to the nodes contained inside a volume. These should not overlap to
//   each other nor extrude the shape of their mother volume.
//______________________________________________________________________________

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
void TGeoOverlap::Print(Option_t *) const
{
// Print detailed info.
   PrintInfo();
   printf(" - first volume: %s at position:\n", fVolume1->GetName());
   fMatrix1->Print();   
   fVolume1->InspectShape();   
   printf(" - second volume: %s at position:\n", fVolume2->GetName());
   fMatrix2->Print();   
   fVolume2->InspectShape();   
}

//______________________________________________________________________________
void TGeoOverlap::PrintInfo() const
{
// Print some info.
   printf(" = Overlap %s: %s ovlp=%g\n", GetName(), GetTitle(),fOverlap);
}

//______________________________________________________________________________
void TGeoOverlap::SetNextPoint(Double_t x, Double_t y, Double_t z)
{
// Set next overlapping point.
   fMarker->SetNextPoint(x,y,z);
}

//______________________________________________________________________________
void TGeoOverlap::SampleOverlap(Int_t npoints)
{
// Draw overlap and sample with random points the overlapping region.
   Draw();
   // Select bounding box of the second volume (may extrude first)
   TPolyMarker3D *marker = 0;
   TGeoBBox *box = (TGeoBBox*)fVolume2->GetShape();
   Double_t dx = box->GetDX();
   Double_t dy = box->GetDY();
   Double_t dz = box->GetDZ();
   Double_t pt[3];
   Double_t master[3];
   const Double_t *orig = box->GetOrigin();
   Int_t ipoint = 0;
   Int_t itry = 0;
   Int_t iovlp = 0;
   while (ipoint < npoints) {
   // Shoot randomly in the bounding box.
      pt[0] = orig[0] - dx + 2.*dx*gRandom->Rndm();
      pt[1] = orig[1] - dy + 2.*dy*gRandom->Rndm();
      pt[2] = orig[2] - dz + 2.*dz*gRandom->Rndm();
      if (!fVolume2->Contains(pt)) {
         itry++;
         if (itry>10000 && !ipoint) {
            Error("SampleOverlap", "No point inside volume!!! - aborting");
            break;
         }
         continue;
      }  
      ipoint++;          
      // Check if the point is inside the first volume
      fMatrix2->LocalToMaster(pt, master);
      fMatrix1->MasterToLocal(master, pt);
      Bool_t in = fVolume1->Contains(pt);
      if (IsOverlap() && !in) continue;
      if (!IsOverlap() && in) continue;
      // The point is in the overlapping region.
      iovlp++;
      if (!marker) {
         marker = new TPolyMarker3D();
         marker->SetMarkerColor(kRed);
      }   
      marker->SetNextPoint(master[0], master[1], master[2]);
   }
   if (!iovlp) return;
   marker->Draw("SAME");
   gPad->Modified();
   gPad->Update();
   Double_t capacity = fVolume1->GetShape()->Capacity();
   capacity *= Double_t(iovlp)/Double_t(npoints);
   Double_t err = 1./TMath::Sqrt(Double_t(iovlp));
   Info("SampleOverlap", "#Overlap %s has %g +/- %g [cm3]",
         GetName(), capacity, err*capacity);
}        

//______________________________________________________________________________
void TGeoOverlap::Sizeof3D() const
{
// Get 3D size of this.
   fVolume1->GetShape()->Sizeof3D();
   fVolume2->GetShape()->Sizeof3D();
}

//______________________________________________________________________________
void TGeoOverlap::Validate() const
{
// Validate this overlap.
   Double_t point[3];
   Double_t local[3];
   Double_t safe1,safe2;
   Int_t npoints = fMarker->GetN();
   for (Int_t i=0; i<npoints; i++) {
      fMarker->GetPoint(i, point[0], point[1], point[2]);
      if (IsExtrusion()) {
         fMatrix1->MasterToLocal(point,local);
         safe1 = fVolume1->GetShape()->Safety(local, kFALSE);
         printf("point %d: safe1=%f\n", i, safe1);
      } else {
         fMatrix1->MasterToLocal(point,local);  
         safe1 = fVolume1->GetShape()->Safety(local, kTRUE);
         fMatrix2->MasterToLocal(point,local);  
         safe2 = fVolume2->GetShape()->Safety(local, kTRUE);
         printf("point %d: safe1=%f safe2=%f\n", i, safe1,safe2);
      }
   }
}
         
         
