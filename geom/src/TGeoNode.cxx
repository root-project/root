// @(#)root/geom:$Name:$:$Id:$
// Author: Andrei Gheata   24/10/01

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////////
// TGeoNode
//----------
//   Nodes are positioned volumes. They have a pointer to the corresponding
// volume. Nodes always belong to a mother container volume, so they also
// have a pointer to this. The base class for nodes is TGeoNode, describing only
// the logical tree. The position of the node with respect to its mother is
// defined by classes deriving from TGeoNode and is stored as a transformation
// matrix by TGeoNodeMatrix or just an offset by division nodes (TGeoNodeXXX).
//   Nodes are invisible to user at build time : to create a node one has only
// to call TGeoVolume::AddNode() or TGeoVolume::Divide() methods. In the first
// case, the volume pointed by the node and the geometrical transformation with
// respect to the mother must be created a priori :
//
//   TGeoVolume *vol = new TGeoSphere("SPH", 200, 400);
//   TGeoVolume *mother = gGeoManager->MakeBox("HALL", "mat1", 1000, 1000, 3000);
//   TGeoTranslation *t1 = new TGeoTranslation(0, 0, 300);
//   mother->AddNode(vol, t1);
//
//   The last line will create a branch : HALL->SPH . A node named SPH:0 will
// be created. If trying to place the same volume many times inside the same
// mother, the automatic naming scheme for the corresponding nodes is just
// appending <:copy_number> to the name of the volume. Therefore :
//
//   TGeoTranslation *t2 = new TGeoTranslation(0,0,-300);
//   mother->AddNode(vol, t2);
//
// will create a TGeoNodeMatrix named SPH:1 inside HALL.
//
//   When creating division nodes (TGeoVolume::Divide()), one has to specify the
// number of divisions, optionally the range for dividing and an string option
// specifying the division type. A list of TGeoNodeOffset will be generated :
//
//   mother->Divide(5, "X");
//
// will create five TGeoNodeOffset nodes, pointing to the same basic cell volume
// which is automatically generated :
//
//   HALL:0 --|
//   HALL:1 --|
//   HALL:2 --|---> HALL_C = gGeoManager->MakeBox("HALL_C", "mat1", 200, 1000, 3000)
//   HALL:3 --|
//   HALL:4 --|
//
//   One can subsequently add usual nodes inside HALL_C cell or divide it, and the
// action will affect all nodes HALL:i .
//   If the basic cell volumes coming from a division operation are not identical,
// a volume will be generated per division node, and the naming sheme for them
// will be HALL_d1, HALL_d2, ... .
//
// Browsing nodes. (to be added)
//
// Node flags.(to be added)
//
//Begin_Html
/*
<img src="gif/t_node.jpg">
*/
//End_Html

#include "Riostream.h"

#include "TBrowser.h"
#include "TObjArray.h"
#include "TStyle.h"

#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TGeoShape.h"
#include "TGeoVolume.h"
#include "TGeoFinder.h"
#include "TGeoNode.h"

// statics and globals

ClassImp(TGeoNode)

//-----------------------------------------------------------------------------
TGeoNode::TGeoNode()
{
// Default constructor
   fVolume       = 0;
   fMother       = 0;
   fOverlaps     = 0;
   fNovlp        = 0;
}
//-----------------------------------------------------------------------------
TGeoNode::TGeoNode(TGeoVolume *vol)
{
// Constructor
   if (!vol) {
      Error("ctor", "volume not specified");
      return;
   }
   fVolume       = vol;
   fMother       = 0;
   fOverlaps     = 0;
   fNovlp        = 0;
}
//-----------------------------------------------------------------------------
TGeoNode::~TGeoNode()
{
// Destructor
   if (fOverlaps) delete fOverlaps;
}
//-----------------------------------------------------------------------------
void TGeoNode::Browse(TBrowser *b)
{
   if (!b) return;
//   if (!GetNdaughters()) b->Add(this);
   for (Int_t i=0; i<GetNdaughters(); i++)
      b->Add(GetDaughter(i));
}
//-----------------------------------------------------------------------------
Bool_t TGeoNode::IsOnScreen() const
{
// check if this node is drawn. Assumes that this node is current
   if (!IsVisible()) return kFALSE;
   if (!gGeoManager->GetTopVolume()->IsVisDaughters()) return kFALSE;
   Int_t level=gGeoManager->GetLevel();
   Int_t vis_level=gGeoManager->GetVisLevel();
   if ((!level) || (level>vis_level)) return kFALSE;
   Int_t vis_opt = gGeoManager->GetVisOption();
   Int_t nd=GetNdaughters();
   switch (vis_opt) {
      case TGeoManager::kGeoVisDefault:
         return kTRUE;
         break;
      case TGeoManager::kGeoVisLeaves:
         if ((nd==0) || (level==vis_level)) return kTRUE;
         if (!fVolume->IsVisDaughters()) return kTRUE;
         return kFALSE;
         break;
      case TGeoManager::kGeoVisOnly:
         return kFALSE;
         break;
      case TGeoManager::kGeoVisBranch:
         return kFALSE;
         break;
      default:
         return kFALSE;
   }
}
//-----------------------------------------------------------------------------
void TGeoNode::InspectNode() const
{
   printf("Inspecting node %s\n", GetName());
   if (fNovlp) printf("### node is MANY\n");
   if (fOverlaps && fMother) {
      printf("### possibly overlaping with :\n");
      for (Int_t i=0; i<fNovlp; i++)
         printf("###   node %s\n", fMother->GetNode(fOverlaps[i])->GetName());
   }
   printf("### transformation wrt mother\n");
   TGeoMatrix *matrix = GetMatrix();
   if (matrix) matrix->Print();
   if (fMother)
      printf("### mother volume %s\n", fMother->GetName());
   fVolume->InspectShape();
}
//-----------------------------------------------------------------------------
void TGeoNode::CheckShapes()
{
// check for wrong parameters in shapes
   fVolume->CheckShapes();
   Int_t nd = GetNdaughters();
   if (!nd) return;
   for (Int_t i=0; i<nd; i++) fVolume->GetNode(i)->CheckShapes();
}
//-----------------------------------------------------------------------------
void TGeoNode::DrawOnly(Option_t *option)
{
// draw only this node independently of its vis options
   fVolume->DrawOnly(option);
}
//-----------------------------------------------------------------------------
void TGeoNode::Draw(Option_t *option)
{
// draw current node according to option
   gGeoManager->FindNode();
   gGeoManager->CdUp();
   Double_t point[3];
   gGeoManager->MasterToLocal(gGeoManager->GetCurrentPoint(), &point[0]);
   gGeoManager->SetCurrentPoint(&point[0]);
   gGeoManager->GetCurrentVolume()->Draw(option);
}
//-----------------------------------------------------------------------------
void TGeoNode::DrawOverlaps()
{
   if (!fNovlp) {printf("node %s is ONLY\n", GetName()); return;}
   if (!fOverlaps) {printf("node %s no overlaps\n", GetName()); return;}
   fVolume->SetLineColor(3);
   fVolume->SetLineWidth(2);
   gGeoManager->RestoreMasterVolume();
   gGeoManager->GetTopVolume()->VisibleDaughters(kFALSE);
   fVolume->SetVisibility(kTRUE);
   fVolume->VisibleDaughters(kFALSE);
   TGeoNode *node;
   for (Int_t i=0; i<fNovlp; i++) {
      node = fMother->GetNode(fOverlaps[i]);
      node->GetVolume()->SetVisibility(kTRUE);
   }
   gGeoManager->GetTopVolume()->Draw();
}
//-----------------------------------------------------------------------------
Int_t TGeoNode::FindNode(TGeoNode *node, Int_t level)
{
   Int_t nd = GetNdaughters();
   if (!nd) return -1;
   TIter next(fVolume->GetNodes());
   TGeoNode *daughter;
   while ((daughter=(TGeoNode*)next())) {
      if (daughter==node) {
         gGeoManager->AddCheckedNode(node, level+1);
         return (level+1);
      }
   }
   next.Reset();
   Int_t new_level;
   while ((daughter=(TGeoNode*)next())) {
      new_level = daughter->FindNode(node, level+1);
      if (new_level>=0) {
         gGeoManager->AddCheckedNode(daughter, level+1);
         return new_level;
      }
   }
   return -1;
}
//-----------------------------------------------------------------------------
void TGeoNode::SaveAttributes(ofstream &out) const
{
// save attributes for this node
   if (fVolume->IsVisStreamed()) return;
   fVolume->SetVisStreamed(kTRUE);
   char quote='"';
   if ((!fVolume->IsStyleDefault()) && (fVolume->IsVisTouched())) {
      out << "   vol = gGeoManager->GetVolume("<<quote<<fVolume->GetName()<<quote<<");"<<endl;
      if (!fVolume->IsVisDaughters())
         out << "   vol->SetVisDaughters(kFALSE);"<<endl;
      if (fVolume->IsVisible()) {
         if (fVolume->GetLineColor() != gStyle->GetLineColor())
            out<<"   vol->SetLineColor("<<fVolume->GetLineColor()<<");"<<endl;
         if (fVolume->GetLineStyle() != gStyle->GetLineStyle())
            out<<"   vol->SetLineStyle("<<fVolume->GetLineStyle()<<");"<<endl;
         if (fVolume->GetLineWidth() != gStyle->GetLineWidth())
            out<<"   vol->SetLineWidth("<<fVolume->GetLineWidth()<<");"<<endl;
      } else {
         out <<"   vol->SetVisibility(kFALSE);"<<endl;
      }
   }
   if (!fVolume->IsVisDaughters()) return;
   Int_t nd = GetNdaughters();
   if (!nd) return;
   TGeoNode *node;
   for (Int_t i=0; i<nd; i++) {
      node = GetDaughter(i);
      node->SaveAttributes(out);
   }
}
//-----------------------------------------------------------------------------
void TGeoNode::MasterToLocal(const Double_t *master, Double_t *local) const
{
// Convert the point coordinates from mother reference to local reference system
   GetMatrix()->MasterToLocal(master, local);
}
//-----------------------------------------------------------------------------
void TGeoNode::MasterToLocalVect(const Double_t *master, Double_t *local) const
{
// Convert a vector from mother reference to local reference system
   GetMatrix()->MasterToLocalVect(master, local);
}
//-----------------------------------------------------------------------------
void TGeoNode::LocalToMaster(const Double_t *local, Double_t *master) const
{
// Convert the point coordinates from local reference system to mother reference
   GetMatrix()->LocalToMaster(local, master);
}
//-----------------------------------------------------------------------------
void TGeoNode::LocalToMasterVect(const Double_t *local, Double_t *master) const
{
// Convert a vector from local reference system to mother reference
   GetMatrix()->LocalToMasterVect(local, master);
}
//-----------------------------------------------------------------------------
void TGeoNode::ls(Option_t *option) const
{
// Print the path (A/B/C/...) to this node on stdout
}
//-----------------------------------------------------------------------------
void TGeoNode::Paint(Option_t *option)
{
// Paint this node with option specification
   Int_t vis_opt = gGeoManager->GetVisOption();
   TGeoNode *node = 0;
   Int_t nd = GetNdaughters();
   Bool_t last = kFALSE;
   Int_t level = gGeoManager->GetLevel();
   Int_t vis_level=gGeoManager->GetVisLevel();
   Bool_t vis=(IsVisible() && gGeoManager->GetLevel())?kTRUE:kFALSE;
   Int_t id;
   switch (vis_opt) {
      case TGeoManager::kGeoVisDefault:
         if (vis && (level<=vis_level))
            fVolume->GetShape()->Paint(option);
            // draw daughters
         if (level<vis_level) {
            if ((!nd) || (!fVolume->IsVisDaughters())) return;
            for (id=0; id<nd; id++) {
               node = GetDaughter(id);
               gGeoManager->CdDown(id);
               node->Paint(option);
               gGeoManager->CdUp();
            }
         }
         break;
      case TGeoManager::kGeoVisLeaves:
         if (level>vis_level) return;
         last = ((nd==0) || (level==vis_level) || (!fVolume->IsVisDaughters()))?kTRUE:kFALSE;
         if (vis && last)
            fVolume->GetShape()->Paint(option);
         if (last) return;
         for (id=0; id<nd; id++) {
            node = GetDaughter(id);
            gGeoManager->CdDown(id);
            node->Paint(option);
            gGeoManager->CdUp();
         }
         break;
      case TGeoManager::kGeoVisOnly:
         fVolume->GetShape()->Paint(option);
         break;
      case TGeoManager::kGeoVisBranch:
         gGeoManager->cd(gGeoManager->GetDrawPath());
         while (gGeoManager->GetLevel()) {
            if (gGeoManager->GetCurrentVolume()->IsVisible())
               gGeoManager->GetCurrentVolume()->GetShape()->Paint(option);
            gGeoManager->CdUp();
         }
         break;
      default:
         return;
   }
}
//-----------------------------------------------------------------------------
void TGeoNode::PrintCandidates() const
{
// print daughters candidates for containing current point
//   cd();
   Double_t point[3];
   gGeoManager->MasterToLocal(gGeoManager->GetCurrentPoint(), &point[0]);
   printf("   Local : %g, %g, %g\n", point[0], point[1], point[2]);
   if (!fVolume->Contains(&point[0])) {
      printf("current point not inside this\n");
      return;
   }
   TGeoPatternFinder *finder = fVolume->GetFinder();
   TGeoNode *node;
   if (finder) {
      printf("current node divided\n");
      node = finder->FindNode(&point[0]);
      if (!node) {
         printf("point not inside division element\n");
         return;
      }
      printf("inside division element %s\n", node->GetName());
      return;
   }
   TGeoVoxelFinder *voxels = fVolume->GetVoxels();
   if (!voxels) {
      printf("volume not voxelized\n");
      return;
   }
   Int_t ncheck = 0;
   Int_t *check_list = voxels->GetCheckList(&point[0], ncheck);
   voxels->PrintVoxelLimits(&point[0]);
   if (!check_list) {
      printf("no candidates for current point\n");
      return;
   }
   TString overlap = "ONLY";
   for (Int_t id=0; id<ncheck; id++) {
      node = fVolume->GetNode(check_list[id]);
      if (node->IsOverlapping()) overlap = "MANY";
      else overlap = "ONLY";
      printf("%i %s %s\n", check_list[id], node->GetName(), overlap.Data());
   }
   PrintOverlaps();
}
//-----------------------------------------------------------------------------
void TGeoNode::PrintOverlaps() const
{
// print possible overlapping nodes
   if (!fNovlp) {printf("node %s is ONLY\n", GetName()); return;}
   if (!fOverlaps) {printf("node %s no overlaps\n", GetName()); return;}
   printf("Overlaps for node %s :\n", GetName());
   TGeoNode *node;
   for (Int_t i=0; i<fNovlp; i++) {
      node = fMother->GetNode(fOverlaps[i]);
      printf("   %s\n", node->GetName());
   }
}
//-----------------------------------------------------------------------------
void TGeoNode::SetOverlaps(Int_t *ovlp, Int_t novlp)
{
// set the list of overlaps for this node (ovlp must be created with operator new)
   if (fOverlaps) delete fOverlaps;
   fOverlaps = ovlp;
   fNovlp = novlp;
}
//-----------------------------------------------------------------------------
void TGeoNode::VisibleDaughters(Bool_t vis)
{
   fVolume->SetVisibility(vis);
   if (!GetNdaughters()) return;
   TIter next(fVolume->GetNodes());
   TGeoNode *node;
   while ((node=(TGeoNode*)next())) node->VisibleDaughters(vis);
}

////////////////////////////////////////////////////////////////////////////////
// TGeoNodeMatrix - a node containing local transformation
//
//
//
//
//Begin_Html
/*
<img src=".gif">
*/
//End_Html

ClassImp(TGeoNodeMatrix)


//-----------------------------------------------------------------------------
TGeoNodeMatrix::TGeoNodeMatrix()
{
// Default constructor
   fMatrix       = 0;
}
//-----------------------------------------------------------------------------
TGeoNodeMatrix::TGeoNodeMatrix(TGeoVolume *vol, TGeoMatrix *matrix) :
             TGeoNode(vol)
{
// Constructor. Null pointer to matrix means identity transformation
   fMatrix = matrix;
}
//-----------------------------------------------------------------------------
TGeoNodeMatrix::~TGeoNodeMatrix()
{
// Destructor
}
//-----------------------------------------------------------------------------
Int_t TGeoNodeMatrix::GetByteCount() const
{
// return the total size in bytes of this node
   Int_t count = 40 + 4; // TGeoNode + fMatrix
//   if (fMatrix) count += fMatrix->GetByteCount();
   return count;
}
//-----------------------------------------------------------------------------
TGeoNode *TGeoNodeMatrix::MakeCopyNode() const
{
// make a copy of this node
   TGeoNodeMatrix *node = new TGeoNodeMatrix(fVolume, fMatrix);
   char *name = new char[strlen(GetName())];
   sprintf(name, "%s", GetName());
   node->SetName(name);
   // set the mother
   node->SetMotherVolume(fMother);
   // copy overlaps
   if (fNovlp>0) {
      if (fOverlaps) {
         Int_t *ovlps = new Int_t[fNovlp];
         memcpy(ovlps, fOverlaps, fNovlp*sizeof(Int_t));
         node->SetOverlaps(ovlps, fNovlp);
      } else {
         node->SetOverlaps(fOverlaps, fNovlp);
      }
   }
   // copy VC
   if (IsVirtual()) node->SetVirtual();
   return node;
}

/*************************************************************************
 * TGeoNodeOffset - node containing an offset
 *
 *************************************************************************/
ClassImp(TGeoNodeOffset)


//-----------------------------------------------------------------------------
TGeoNodeOffset::TGeoNodeOffset()
{
// Default constructor
   TObject::SetBit(kGeoNodeOffset);
   fOffset = 0;
   fIndex = 0;
   fFinder = 0;
}
//-----------------------------------------------------------------------------
TGeoNodeOffset::TGeoNodeOffset(TGeoVolume *vol, Int_t index, Double_t offset) :
           TGeoNode(vol)
{
// Constructor. Null pointer to matrix means identity transformation
   TObject::SetBit(kGeoNodeOffset);
   fOffset = offset;
   fIndex = index;
   fFinder = 0;
}
//-----------------------------------------------------------------------------
TGeoNodeOffset::~TGeoNodeOffset()
{
// Destructor
}
//-----------------------------------------------------------------------------
Int_t TGeoNodeOffset::GetIndex() const
{
   return (fIndex+fFinder->GetDivIndex());
}
//-----------------------------------------------------------------------------
TGeoNode *TGeoNodeOffset::MakeCopyNode() const
{
// make a copy of this node
   TGeoNodeOffset *node = new TGeoNodeOffset(fVolume, GetIndex(), fOffset);
   // set the mother
   node->SetMotherVolume(fMother);
   if (IsVirtual()) node->SetVirtual();
   // set the finder
   node->SetFinder(GetFinder());
   return node;
}

