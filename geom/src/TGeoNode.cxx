// @(#)root/geom:$Name:  $:$Id: TGeoNode.cxx,v 1.20 2003/10/06 15:15:01 brun Exp $
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
//_________
//   A node represent a volume positioned inside another.They store links to both
// volumes and to the TGeoMatrix representing the relative positioning. Node are
// never instanciated directly by users, but created as a result of volume operations.
// Adding a volume named A with a given user ID inside a volume B will create a node 
// node named A_ID. This will be added to the list of nodes stored by B. Also,
// when applying a division operation in N slices to a volume A, a list of nodes
// B_1, B_2, ..., B_N is also created. A node B_i does not represent a unique
// object in the geometry because its container A might be at its turn positioned
// as node inside several other volumes. Only when a complete branch of nodes
// is fully defined up to the top node in the geometry, a given path like:
//       /TOP_1/.../A_3/B_7 will represent an unique object. Its global transformation
// matrix can be computed as the pile-up of all local transformations in its
// branch. We will therefore call "logical graph" the hierarchy defined by nodes
// and volumes. The expansion of the logical graph by all possible paths defines
// a tree sructure where all nodes are unique "touchable" objects. We will call
// this the "physical tree". Unlike the logical graph, the physical tree can
// become a huge structure with several milions of nodes in case of complex
// geometries, therefore it is not always a good idea to keep it transient
// in memory. Since a the logical and physical structures are correlated, the
// modeller rather keeps track only of the current branch, updating the current
// global matrix at each change of the level in geometry. The current physical node
// is not an object that can be asked for at a given moment, but rather represented
// by the combination: current node + current global matrix. However, physical nodes
// have unique ID's that can be retreived for a given modeler state. These can be
// fed back to the modeler in order to force a physical node to become current.
// The advantage of this comes from the fact that all navigation queries check
// first the current node, therefore knowing the location of a point in the 
// geometry can be saved as a starting state for later use.
//
//   Nodes can be declared as "overlapping" in case they do overlap with other
// nodes inside the same container or extrude this container. Non-overlapping
// nodes can be created with:
//
//      TGeoVolume::AddNode(TGeoVolume *daughter, Int_t copy_No, TGeoMatrix *matr);
//
// The creation of overapping nodes can be done with a similar prototype:
//
//      TGeoVolume::AddNodeOverlap(same arguments);
//
// When closing the geometry, overlapping nodes perform a check of possible
// overlaps with their neighbours. These are stored and checked all the time
// during navigation, therefore navigation is slower when embedding such nodes
// into geometry.
//
//   Node have visualization attributes as volume have. When undefined by users,
// painting a node on a pad will take the corresponding volume attributes.
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
#include "TVirtualGeoPainter.h"
#include "TGeoNode.h"

// statics and globals

ClassImp(TGeoNode)

//_____________________________________________________________________________
TGeoNode::TGeoNode()
{
// Default constructor
   fVolume       = 0;
   fMother       = 0;
   fNumber       = 0;
   fOverlaps     = 0;
   fNovlp        = 0;
}

//_____________________________________________________________________________
TGeoNode::TGeoNode(const TGeoVolume *vol)
{
// Constructor
   if (!vol) {
      Error("ctor", "volume not specified");
      return;
   }
   fVolume       = (TGeoVolume*)vol;
   fMother       = 0;
   fNumber       = 0;
   fOverlaps     = 0;
   fNovlp        = 0;
}

//_____________________________________________________________________________
TGeoNode::~TGeoNode()
{
// Destructor
   if (fOverlaps) delete [] fOverlaps;
}

//_____________________________________________________________________________
void TGeoNode::Browse(TBrowser *b)
{
   if (!b) return;
//   if (!GetNdaughters()) b->Add(this);
   for (Int_t i=0; i<GetNdaughters(); i++)
      b->Add(GetDaughter(i));
}

//_____________________________________________________________________________
Bool_t TGeoNode::IsOnScreen() const
{
// check if this node is drawn. Assumes that this node is current
   
   if (fVolume->TGeoAtt::TestBit(TGeoAtt::kVisOnScreen)) return kTRUE;
   return kFALSE;
}

//_____________________________________________________________________________
void TGeoNode::InspectNode() const
{
   printf("Inspecting node %s\n", GetName());
   if (IsOverlapping()) printf("### node is MANY\n");
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

//_____________________________________________________________________________
void TGeoNode::CheckShapes()
{
// check for wrong parameters in shapes
   fVolume->CheckShapes();
   Int_t nd = GetNdaughters();
   if (!nd) return;
   for (Int_t i=0; i<nd; i++) fVolume->GetNode(i)->CheckShapes();
}

//_____________________________________________________________________________
void TGeoNode::DrawOnly(Option_t *option)
{
// draw only this node independently of its vis options
   fVolume->DrawOnly(option);
}

//_____________________________________________________________________________
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

//_____________________________________________________________________________
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

//_____________________________________________________________________________
void TGeoNode::FillIdArray(Int_t &ifree, Int_t &nodeid, Int_t *array) const
{
// Fill array with node id. Recursive on node branch.
   Int_t nd = GetNdaughters();
   if (!nd) return;
   TGeoNode *daughter;
   Int_t istart = ifree; // start index for daughters
   ifree += nd;
   for (Int_t id=0; id<nd; id++) {
      daughter = GetDaughter(id);
      array[istart+id] = ifree;
      array[ifree++] = ++nodeid;
      daughter->FillIdArray(ifree, nodeid, array);
   }
}      
   

//_____________________________________________________________________________
Int_t TGeoNode::FindNode(const TGeoNode *node, Int_t level)
{
   Int_t nd = GetNdaughters();
   if (!nd) return -1;
   TIter next(fVolume->GetNodes());
   TGeoNode *daughter;
   while ((daughter=(TGeoNode*)next())) {
      if (daughter==node) {
         gGeoManager->GetListOfNodes()->AddAt(daughter,level+1);
         return (level+1);
      }
   }
   next.Reset();
   Int_t new_level;
   while ((daughter=(TGeoNode*)next())) {
      new_level = daughter->FindNode(node, level+1);
      if (new_level>=0) {
         gGeoManager->GetListOfNodes()->AddAt(daughter, level+1);
         return new_level;
      }
   }
   return -1;
}

//_____________________________________________________________________________
void TGeoNode::SaveAttributes(ofstream &out)
{
// save attributes for this node
   if (IsVisStreamed()) return;
   SetVisStreamed(kTRUE);
   char quote='"';
   Bool_t voldef = kFALSE;
   if ((fVolume->IsVisTouched()) && (!fVolume->IsVisStreamed())) {
      fVolume->SetVisStreamed(kTRUE);
      out << "   vol = gGeoManager->GetVolume("<<quote<<fVolume->GetName()<<quote<<");"<<endl;
      voldef = kTRUE;
      if (!fVolume->IsVisDaughters())
         out << "   vol->SetVisDaughters(kFALSE);"<<endl;
      if (fVolume->IsVisible()) {
/*
         if (fVolume->GetLineColor() != gStyle->GetLineColor())
            out<<"   vol->SetLineColor("<<fVolume->GetLineColor()<<");"<<endl;
         if (fVolume->GetLineStyle() != gStyle->GetLineStyle())
            out<<"   vol->SetLineStyle("<<fVolume->GetLineStyle()<<");"<<endl;
         if (fVolume->GetLineWidth() != gStyle->GetLineWidth())
            out<<"   vol->SetLineWidth("<<fVolume->GetLineWidth()<<");"<<endl;
*/
      } else {
         out <<"   vol->SetVisibility(kFALSE);"<<endl;
      }
   }
   if (!IsVisDaughters()) return;
   Int_t nd = GetNdaughters();
   if (!nd) return;
   TGeoNode *node;
   for (Int_t i=0; i<nd; i++) {
      node = GetDaughter(i);
      if (node->IsVisStreamed()) continue;
      if (node->IsVisTouched()) {
         if (!voldef)
            out << "   vol = gGeoManager->GetVolume("<<quote<<fVolume->GetName()<<quote<<");"<<endl;
         out<<"   node = vol->GetNode("<<i<<");"<<endl;
         if (!node->IsVisDaughters()) {
            out<<"   node->VisibleDaughters(kFALSE);"<<endl;
            node->SetVisStreamed(kTRUE);
            continue;
         }
         if (!node->IsVisible()) 
            out<<"   node->SetVisibility(kFALSE);"<<endl;
      }         
      node->SaveAttributes(out);
      node->SetVisStreamed(kTRUE);
   }
}

//_____________________________________________________________________________
void TGeoNode::MasterToLocal(const Double_t *master, Double_t *local) const
{
// Convert the point coordinates from mother reference to local reference system
   GetMatrix()->MasterToLocal(master, local);
}

//_____________________________________________________________________________
void TGeoNode::MasterToLocalVect(const Double_t *master, Double_t *local) const
{
// Convert a vector from mother reference to local reference system
   GetMatrix()->MasterToLocalVect(master, local);
}

//_____________________________________________________________________________
void TGeoNode::LocalToMaster(const Double_t *local, Double_t *master) const
{
// Convert the point coordinates from local reference system to mother reference
   GetMatrix()->LocalToMaster(local, master);
}

//_____________________________________________________________________________
void TGeoNode::LocalToMasterVect(const Double_t *local, Double_t *master) const
{
// Convert a vector from local reference system to mother reference
   GetMatrix()->LocalToMasterVect(local, master);
}

//_____________________________________________________________________________
void TGeoNode::ls(Option_t * /*option*/) const
{
// Print the path (A/B/C/...) to this node on stdout
}

//_____________________________________________________________________________
void TGeoNode::Paint(Option_t *option)
{
// Paint this node and its content according to visualization settings.
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) return;
   painter->PaintNode(this, option);
}

//_____________________________________________________________________________
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

//_____________________________________________________________________________
void TGeoNode::PrintOverlaps() const
{
// print possible overlapping nodes
   if (!IsOverlapping()) {printf("node %s is ONLY\n", GetName()); return;}
   if (!fOverlaps) {printf("node %s no overlaps\n", GetName()); return;}
   printf("Overlaps for node %s :\n", GetName());
   TGeoNode *node;
   for (Int_t i=0; i<fNovlp; i++) {
      node = fMother->GetNode(fOverlaps[i]);
      printf("   %s\n", node->GetName());
   }
}

//_____________________________________________________________________________
Double_t TGeoNode::Safety(Double_t *point, Bool_t in) const
{
// computes the closest distance from given point to this shape

   Double_t local[3];
   GetMatrix()->MasterToLocal(point,local);
   return fVolume->GetShape()->Safety(local,in);
}

//_____________________________________________________________________________
void TGeoNode::SetOverlaps(Int_t *ovlp, Int_t novlp)
{
// set the list of overlaps for this node (ovlp must be created with operator new)
   if (fOverlaps) delete [] fOverlaps;
   fOverlaps = ovlp;
   fNovlp = novlp;
}

//_____________________________________________________________________________
void TGeoNode::SetVisibility(Bool_t vis)
{
   if (gGeoManager->IsClosed()) SetVisTouched(kTRUE);
   TGeoAtt::SetVisibility(vis);
   gGeoManager->ModifiedPad();
}

//_____________________________________________________________________________
void TGeoNode::VisibleDaughters(Bool_t vis)
{
   if (gGeoManager->IsClosed()) SetVisTouched(kTRUE);
   SetVisDaughters(vis);
   gGeoManager->ModifiedPad();
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


//_____________________________________________________________________________
TGeoNodeMatrix::TGeoNodeMatrix()
{
// Default constructor
   fMatrix       = 0;
}

//_____________________________________________________________________________
TGeoNodeMatrix::TGeoNodeMatrix(const TGeoVolume *vol, const TGeoMatrix *matrix) :
             TGeoNode(vol)
{
// Constructor. 
   fMatrix = (TGeoMatrix*)matrix;
   if (!fMatrix) fMatrix = gGeoIdentity;
}

//_____________________________________________________________________________
TGeoNodeMatrix::~TGeoNodeMatrix()
{
// Destructor
}

//_____________________________________________________________________________
Int_t TGeoNodeMatrix::GetByteCount() const
{
// return the total size in bytes of this node
   Int_t count = 40 + 4; // TGeoNode + fMatrix
//   if (fMatrix) count += fMatrix->GetByteCount();
   return count;
}

//_____________________________________________________________________________
Int_t TGeoNodeMatrix::GetOptimalVoxels() const
{
//--- Returns type of optimal voxelization for this node.
// type = 0 -> cartesian
// type = 1 -> cylindrical
   Bool_t type = fVolume->GetShape()->IsCylType();
   if (!type) return 0;
   if (!fMatrix->IsRotAboutZ()) return 0;
   const Double_t *transl = fMatrix->GetTranslation();
   if (TMath::Abs(transl[0])>1E-10) return 0;
   if (TMath::Abs(transl[1])>1E-10) return 0;
   return 1;
}   

//_____________________________________________________________________________
TGeoNode *TGeoNodeMatrix::MakeCopyNode() const
{
// make a copy of this node
//   printf("      Making a copy of node %s\n", GetName());
   TGeoNodeMatrix *node = new TGeoNodeMatrix(fVolume, fMatrix);
   char *name = new char[strlen(GetName())+1];
   sprintf(name, "%s", GetName());
   node->SetName(name);
   // set the mother
   node->SetMotherVolume(fMother);
   // set the copy number
   node->SetNumber(fNumber);
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


//_____________________________________________________________________________
TGeoNodeOffset::TGeoNodeOffset()
{
// Default constructor
   TObject::SetBit(kGeoNodeOffset);
   fOffset = 0;
   fIndex = 0;
   fFinder = 0;
}

//_____________________________________________________________________________
TGeoNodeOffset::TGeoNodeOffset(const TGeoVolume *vol, Int_t index, Double_t offset) :
           TGeoNode(vol)
{
// Constructor. Null pointer to matrix means identity transformation
   TObject::SetBit(kGeoNodeOffset);
   fOffset = offset;
   fIndex = index;
   fFinder = 0;
}

//_____________________________________________________________________________
TGeoNodeOffset::~TGeoNodeOffset()
{
// Destructor
}

//_____________________________________________________________________________
Int_t TGeoNodeOffset::GetIndex() const
{
   return (fIndex+fFinder->GetDivIndex());
}

//_____________________________________________________________________________
TGeoNode *TGeoNodeOffset::MakeCopyNode() const
{
// make a copy of this node
   TGeoNodeOffset *node = new TGeoNodeOffset(fVolume, GetIndex(), fOffset);
   char *name = new char[strlen(GetName())+1];
   sprintf(name, "%s", GetName());
   node->SetName(name);
   // set the mother
   node->SetMotherVolume(fMother);
   if (IsVirtual()) node->SetVirtual();
   // set the finder
   node->SetFinder(GetFinder());
   return node;
}

