// @(#)root/geom:$Name:  $:$Id: TGeoOverlap.cxx,v 1.3 2003/02/12 17:20:55 brun Exp $
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
   if (IsExtrusion()) {
      if (other->IsExtrusion()) return (fOverlap<=other->GetOverlap())?1:-1;
      return -1;
   } else {   
      if (other->IsExtrusion()) return 1;
      return (fOverlap<=other->GetOverlap())?1:-1;
   }
   return 0;   
}

//______________________________________________________________________________
Int_t TGeoOverlap::DistancetoPrimitive(Int_t px, Int_t py)
{
   return gGeoManager->GetGeomPainter()->DistanceToPrimitiveVol(fVolume, px, py);
}

//______________________________________________________________________________
void TGeoOverlap::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   gGeoManager->GetGeomPainter()->ExecuteVolumeEvent(fVolume, event, px, py);
}

//______________________________________________________________________________
void TGeoOverlap::Paint(Option_t *option)
{
   gGeoManager->GetGeomPainter()->PaintOverlap(this, option);
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
}

//______________________________________________________________________________
TGeoNode *TGeoExtrusion::GetNode(Int_t /*iovlp*/) const
{
// Get extruding node
   return fNode;
}
   
//______________________________________________________________________________
void TGeoExtrusion::Draw(Option_t *option)
{
// Draw the extrusion. Mother volume will be blue, extruding daughter green,
// extruding points red.
   gGeoManager->GetGeomPainter()->DrawOverlap(this, option);
   PrintInfo();
}

//______________________________________________________________________________
void TGeoExtrusion::PrintInfo() const
{
   printf("* extrusion %s/%s: vol=%s node=%s extr=%g\n", GetName(), GetTitle(), 
          fVolume->GetName(), fNode->GetName(), fOverlap);
}

//______________________________________________________________________________
void TGeoExtrusion::Sizeof3D() const
{
   fVolume->GetShape()->Sizeof3D();
   fNode->GetVolume()->GetShape()->Sizeof3D();
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
}

//______________________________________________________________________________
TGeoNode *TGeoNodeOverlap::GetNode(Int_t iovlp) const
{
// Get one of the overlapping nodes.
   switch (iovlp) {
      case 0:
         return fNode1;
      case 1:
         return fNode2;
      default:
         return 0;
   }            
}

//______________________________________________________________________________
void TGeoNodeOverlap::Draw(Option_t *option)
{
// Draw the overlap. One daughter will be blue, the other green,
// extruding points red.
   gGeoManager->GetGeomPainter()->DrawOverlap(this, option);
   PrintInfo();
}
 
//______________________________________________________________________________
void TGeoNodeOverlap::PrintInfo() const
{
   printf("* overlap %s/%s: vol=%s <%s<->%s> ovlp=%g\n", GetName(), GetTitle(), 
          fVolume->GetName(), fNode1->GetName(), fNode2->GetName(), fOverlap);
}
   
//______________________________________________________________________________
void TGeoNodeOverlap::Sizeof3D() const
{
   fNode1->GetVolume()->GetShape()->Sizeof3D();
   fNode2->GetVolume()->GetShape()->Sizeof3D();
}
