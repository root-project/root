// @(#)root/geom:$Name:  $:$Id: TGeoOverlap.cxx,v 1.1 2003/02/10 17:23:14 brun Exp $
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
   fVolume  = 0;
   fMarker  = 0;
}

//______________________________________________________________________________
TGeoOverlap::TGeoOverlap(const char *name, TGeoVolume *vol, Double_t ovlp)
            :TNamed("",name)
{
// Creates a named overlap belonging to volume VOL and having the size OVLP.
   fOverlap = ovlp;
   fVolume  = vol;
   if (!fVolume) {
      Error("Ctor", "volume is NULL");
      return;
   }   
   fMarker  = new TPolyMarker3D();
   fMarker->SetMarkerColor(2);
   fMarker->SetMarkerStyle(8);
   fMarker->SetMarkerSize(0.5);
}

//______________________________________________________________________________
TGeoOverlap::~TGeoOverlap()
{
   if (fMarker) delete fMarker;
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
   if (TestBit(kGeoExtrusion)) {
      if (other->TestBit(kGeoExtrusion)) return (fOverlap<=other->GetOverlap())?1:-1;
      return -1;
   } else if (TestBit(kGeoNodeOverlap)) {   
      if (other->TestBit(kGeoExtrusion)) return 1;
      return (fOverlap<=other->GetOverlap())?1:-1;
   }
   return 0;   
}

//______________________________________________________________________________
void TGeoOverlap::SetNextPoint(Double_t x, Double_t y, Double_t z)
{
// Set next overlapping point.
   fMarker->SetNextPoint(x,y,z);
}

ClassImp(TGeoExtrusion)
/*************************************************************************
 *   TGeoExtrusion - class representing the extrusion of a positioned volume
 *      with respect to its mother.
 ************************************************************************/

//______________________________________________________________________________
TGeoExtrusion::TGeoExtrusion()   
{
// Default ctor.
   fNode = 0;
   TObject::SetBit(kGeoExtrusion);
}

//______________________________________________________________________________
TGeoExtrusion::TGeoExtrusion(const char *name, TGeoVolume *vol, Int_t inode, Double_t ovlp)
              :TGeoOverlap(name, vol, ovlp)
{
// Ctor.      
   if (inode<0 || inode>vol->GetNdaughters()-1) {
      Error("Ctor", "invalid daughter number %i for volume %s", inode, vol->GetName());
      return;
   }
   fNode = vol->GetNode(inode);   
   TObject::SetBit(kGeoExtrusion);
}

//______________________________________________________________________________
void TGeoExtrusion::Draw(Option_t *option)
{
// Draw the extrusion. Mother volume will be blue, extruding daughter green,
// extruding points red.
   Int_t nd = fVolume->GetNdaughters();
   TGeoNode *current;
   for (Int_t i=0; i<nd; i++) {
      current = fVolume->GetNode(i);
      if (current==fNode) {
         current->SetVisibility(kTRUE);
         current->GetVolume()->SetVisibility(kTRUE);
         current->GetVolume()->SetLineColor(3);
      } else {
         current->SetVisibility(kFALSE);
      }
   }
   fVolume->SetVisibility(kTRUE);
   fVolume->SetLineColor(4);
   gGeoManager->SetTopVisible();
   gGeoManager->SetVisLevel(1);
   gGeoManager->SetVisOption(0);
   fVolume->Draw();      
   fMarker->Draw("SAME");
   gGeoManager->ModifiedPad();
   PrintInfo();
}

//______________________________________________________________________________
void TGeoExtrusion::PrintInfo() const
{
   printf("* extrusion %s/%s: vol=%s node=%s extr=%g\n", GetName(), GetTitle(), 
          fVolume->GetName(), fNode->GetName(), fOverlap);
}

ClassImp(TGeoNodeOverlap)
/*************************************************************************
 *   TGeoNodeOverlap - class representing the overlap of 2 positioned 
 *      nodes inside a mother volume.
 ************************************************************************/

//______________________________________________________________________________
TGeoNodeOverlap::TGeoNodeOverlap()   
{
// Default ctor.
   fNode1 = 0;
   fNode2 = 0;
   TObject::SetBit(kGeoNodeOverlap);
}
     
//______________________________________________________________________________
TGeoNodeOverlap::TGeoNodeOverlap(const char *name, TGeoVolume *vol, Int_t inode1, Int_t inode2, Double_t ovlp)
              :TGeoOverlap(name, vol, ovlp)
{
// Ctor.      
   if (inode1<0 || inode1>vol->GetNdaughters()-1) {
      Error("Ctor", "invalid daughter number %i for volume %s", inode1, vol->GetName());
      return;
   }
   fNode1 = vol->GetNode(inode1);   
   if (inode2<0 || inode2>vol->GetNdaughters()-1) {
      Error("Ctor", "invalid daughter number %i for volume %s", inode2, vol->GetName());
      return;
   }
   fNode2 = vol->GetNode(inode2);   
   TObject::SetBit(kGeoNodeOverlap);
}

//______________________________________________________________________________
void TGeoNodeOverlap::Draw(Option_t *option)
{
// Draw the overlap. One daughter will be blue, the other green,
// extruding points red.
   Int_t nd = fVolume->GetNdaughters();
   gGeoManager->SetTopVisible(kFALSE);
   gGeoManager->SetVisLevel(1);
   gGeoManager->SetVisOption(0);
   TGeoNode *current;
   for (Int_t i=0; i<nd; i++) {
      current = fVolume->GetNode(i);
      if (current==fNode1 || current==fNode2) {
         current->SetVisibility(kTRUE);
         current->GetVolume()->SetVisibility(kTRUE);
         if (current==fNode1) 
            current->GetVolume()->SetLineColor(3);
         else 
            current->GetVolume()->SetLineColor(4);
      } else {
         current->SetVisibility(kFALSE);
      }
   }
   fVolume->Draw();      
   fMarker->Draw("SAME");
   gGeoManager->ModifiedPad();
   PrintInfo();
}
 
//______________________________________________________________________________
void TGeoNodeOverlap::PrintInfo() const
{
   printf("* overlap %s/%s: vol=%s <%s<->%s> ovlp=%g\n", GetName(), GetTitle(), 
          fVolume->GetName(), fNode1->GetName(), fNode2->GetName(), fOverlap);
}
   
