// @(#)root/geom:$Id$
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
#include "TGeoVoxelFinder.h"
#include "TGeoNode.h"
#include "TMath.h"
#include "TStopwatch.h"

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
   if (fVolume->IsAdded()) fVolume->SetReplicated();
   fVolume->SetAdded();
   fMother       = 0;
   fNumber       = 0;
   fOverlaps     = 0;
   fNovlp        = 0;
}

//_____________________________________________________________________________
TGeoNode::TGeoNode(const TGeoNode& gn) :
  TNamed(gn),
  TGeoAtt(gn),
  fVolume(gn.fVolume),
  fMother(gn.fMother),
  fNumber(gn.fNumber),
  fNovlp(gn.fNovlp),
  fOverlaps(gn.fOverlaps)
{ 
   //copy constructor
}

//_____________________________________________________________________________
TGeoNode& TGeoNode::operator=(const TGeoNode& gn) 
{
   //assignment operator
   if(this!=&gn) {
      TNamed::operator=(gn);
      TGeoAtt::operator=(gn);
      fVolume=gn.fVolume;
      fMother=gn.fMother;
      fNumber=gn.fNumber;
      fNovlp=gn.fNovlp;
      fOverlaps=gn.fOverlaps;
   } 
   return *this;
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
// How-to-browse for a node.
   if (!b) return;
   if (!GetNdaughters()) return;
   TGeoNode *daughter;
   TString title;
   for (Int_t i=0; i<GetNdaughters(); i++) {
      daughter = GetDaughter(i);
      b->Add(daughter, daughter->GetName(), daughter->IsVisible());
   }      
}

//_____________________________________________________________________________
Int_t TGeoNode::CountDaughters(Bool_t unique_volumes)
{
// Returns the number of daughters. Nodes pointing to same volume counted
// once if unique_volumes is set.
   static Int_t icall = 0;   
   Int_t counter = 0;
   // Count this node
   if (unique_volumes) {
      if (!fVolume->IsSelected()) {
         counter++;
         fVolume->SelectVolume(kFALSE);
      }
   } else counter++;
   icall++;
   Int_t nd = fVolume->GetNdaughters();
   // Count daughters recursively
   for (Int_t i=0; i<nd; i++) counter += GetDaughter(i)->CountDaughters(unique_volumes);
   icall--;
   // Un-mark volumes
   if (icall == 0) fVolume->SelectVolume(kTRUE);
   return counter;
}      

//_____________________________________________________________________________
void TGeoNode::CheckOverlaps(Double_t ovlp, Option_t *option)
{
// Check overlaps bigger than OVLP hierarchically, starting with this node.
   Int_t icheck = 0;
   Int_t ncheck = 0;
   TStopwatch *timer;
   Int_t i;
   Bool_t sampling = kFALSE;
   TString opt(option);
   opt.ToLower();
   if (opt.Contains("s")) sampling = kTRUE;

   TGeoManager *geom = fVolume->GetGeoManager();
   ncheck = CountDaughters(kFALSE);
   timer = new TStopwatch();
   geom->ClearOverlaps();
   geom->SetCheckingOverlaps(kTRUE);
   Info("CheckOverlaps", "Checking overlaps for %s and daughters within %g", fVolume->GetName(),ovlp);
   if (sampling) {
      Info("CheckOverlaps", "Checking overlaps by sampling <%s> for %s and daughters", option, fVolume->GetName());
      Info("CheckOverlaps", "=== NOTE: Extrusions NOT checked with sampling option ! ===");
   }   
   timer->Start();
   geom->GetGeomPainter()->OpProgress(fVolume->GetName(),icheck,ncheck,timer,kFALSE);
   fVolume->CheckOverlaps(ovlp,option);
   icheck++;
   TGeoIterator next(fVolume);
   TGeoNode *node;
   TString path;
   while ((node=next())) {
      next.GetPath(path);
      icheck++;
      if (!node->GetVolume()->IsSelected()) {
         geom->GetGeomPainter()->OpProgress(node->GetVolume()->GetName(),icheck,ncheck,timer,kFALSE);
         node->GetVolume()->SelectVolume(kFALSE);
         node->GetVolume()->CheckOverlaps(ovlp,option);
      }   
   }   
   fVolume->SelectVolume(kTRUE);
   geom->SetCheckingOverlaps(kFALSE);
   geom->SortOverlaps();
   TObjArray *overlaps = geom->GetListOfOverlaps();
   Int_t novlps = overlaps->GetEntriesFast();     
   TNamed *obj;
   for (i=0; i<novlps; i++) {
      obj = (TNamed*)overlaps->At(i);
      obj->SetName(TString::Format("ov%05d",i));
   }
   geom->GetGeomPainter()->OpProgress("Check overlaps:",icheck,ncheck,timer,kTRUE);
   Info("CheckOverlaps", "Number of illegal overlaps/extrusions : %d\n", novlps);
   delete timer;
}      

//_____________________________________________________________________________
Int_t TGeoNode::DistancetoPrimitive(Int_t px, Int_t py)
{
// compute the closest distance of approach from point px,py to this node
   Int_t dist = 9999;
   if (!fVolume) return dist;
   if (gGeoManager != fVolume->GetGeoManager()) gGeoManager = fVolume->GetGeoManager();
   TVirtualGeoPainter *painter = gGeoManager->GetPainter();
   if (!painter) return dist;
   dist = painter->DistanceToPrimitiveVol(fVolume, px, py);
   return dist;
}
      
//_____________________________________________________________________________
void TGeoNode::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
// Execute mouse actions on this volume.
   if (!fVolume) return;
   TVirtualGeoPainter *painter = fVolume->GetGeoManager()->GetPainter();
   if (!painter) return;
   painter->ExecuteVolumeEvent(fVolume, event, px, py);
}

//_____________________________________________________________________________
char *TGeoNode::GetObjectInfo(Int_t px, Int_t py) const
{
// Get node info for the browser.
   if (!fVolume) return 0;
   TVirtualGeoPainter *painter = fVolume->GetGeoManager()->GetPainter();
   if (!painter) return 0;
   return (char*)painter->GetVolumeInfo(fVolume, px, py);
}

//_____________________________________________________________________________
Bool_t TGeoNode::IsOnScreen() const
{
// check if this node is drawn. Assumes that this node is current
   
   if (fVolume->TestAttBit(TGeoAtt::kVisOnScreen)) return kTRUE;
   return kFALSE;
}

//_____________________________________________________________________________
void TGeoNode::InspectNode() const
{
// Inspect this node.
   Info("InspectNode","Inspecting node %s", GetName());
   if (IsOverlapping()) Info("InspectNode","node is MANY");
   if (fOverlaps && fMother) {
      Info("InspectNode","possibly overlaping with :");
      for (Int_t i=0; i<fNovlp; i++)
         Info("InspectNode","   node %s", fMother->GetNode(fOverlaps[i])->GetName());
   }
   Info("InspectNode","Transformation matrix:\n");
   TGeoMatrix *matrix = GetMatrix();
   if (matrix) matrix->Print();
   if (fMother)
      Info("InspectNode","Mother volume %s\n", fMother->GetName());
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
// Method drawing the overlap candidates with this node.
   if (!fNovlp) {printf("node %s is ONLY\n", GetName()); return;}
   if (!fOverlaps) {printf("node %s no overlaps\n", GetName()); return;}
   TGeoNode *node;
   Int_t i;
   Int_t nd = fMother->GetNdaughters();
   for (i=0; i<nd; i++) {
      node = fMother->GetNode(i);
      node->GetVolume()->SetVisibility(kFALSE);
   }   
   fVolume->SetVisibility(kTRUE);
   for (i=0; i<fNovlp; i++) {
      node = fMother->GetNode(fOverlaps[i]);
      node->GetVolume()->SetVisibility(kTRUE);
   }
   gGeoManager->SetVisLevel(1);
   fMother->Draw();
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
// Search for a node within the branch of this one.
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
void TGeoNode::SaveAttributes(ostream &out)
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
Bool_t TGeoNode::MayOverlap(Int_t iother) const 
{
// Check the overlab between the bounding box of the node overlaps with the one
// the brother with index IOTHER.
   if (!fOverlaps) return kFALSE;
   for (Int_t i=0; i<fNovlp; i++) if (fOverlaps[i]==iother) return kTRUE;
   return kFALSE;
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
//   if (!IsOverlapping()) {printf("node %s is ONLY\n", GetName()); return;}
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
// Set visibility of the node (obsolete).
   if (gGeoManager->IsClosed()) SetVisTouched(kTRUE);
   TGeoAtt::SetVisibility(vis);
   if (vis && !fVolume->IsVisible()) fVolume->SetVisibility(vis);
   gGeoManager->ModifiedPad();
}

//_____________________________________________________________________________
void TGeoNode::VisibleDaughters(Bool_t vis)
{
// Set visibility of the daughters (obsolete).
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
TGeoNodeMatrix::TGeoNodeMatrix(const TGeoNodeMatrix& gnm)
               :TGeoNode(gnm), 
                fMatrix(gnm.fMatrix)
{
// Copy ctor.
}

//_____________________________________________________________________________
TGeoNodeMatrix& TGeoNodeMatrix::operator=(const TGeoNodeMatrix& gnm)
{
// Assignment.
   if (this!=&gnm) {
      TGeoNode::operator=(gnm); 
      fMatrix=gnm.fMatrix;
   }
   return *this;
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
// Make a copy of this node.
   TGeoNodeMatrix *node = new TGeoNodeMatrix(fVolume, fMatrix);
   node->SetName(GetName());
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

//_____________________________________________________________________________
void TGeoNodeMatrix::SetMatrix(const TGeoMatrix *matrix)
{
// Matrix setter.
   fMatrix = (TGeoMatrix*)matrix;
   if (!fMatrix) fMatrix = gGeoIdentity;
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
TGeoNodeOffset::TGeoNodeOffset(const TGeoNodeOffset& gno) :
  TGeoNode(gno),
  fOffset(gno.fOffset),
  fIndex(gno.fIndex),
  fFinder(gno.fFinder)
{ 
   //copy constructor
}

//_____________________________________________________________________________
TGeoNodeOffset& TGeoNodeOffset::operator=(const TGeoNodeOffset& gno)
{
   //assignment operator
   if(this!=&gno) {
      TGeoNode::operator=(gno);
      fOffset=gno.fOffset;
      fIndex=gno.fIndex;
      fFinder=gno.fFinder;
   } 
   return *this;
}

//_____________________________________________________________________________
TGeoNodeOffset::~TGeoNodeOffset()
{
// Destructor
}

//_____________________________________________________________________________
Int_t TGeoNodeOffset::GetIndex() const
{
// Get the index of this offset.
   return (fIndex+fFinder->GetDivIndex());
}

//_____________________________________________________________________________
TGeoNode *TGeoNodeOffset::MakeCopyNode() const
{
// make a copy of this node
   TGeoNodeOffset *node = new TGeoNodeOffset(fVolume, GetIndex(), fOffset);
   node->SetName(GetName());
   // set the mother
   node->SetMotherVolume(fMother);
   // set the copy number
   node->SetNumber(fNumber);
   if (IsVirtual()) node->SetVirtual();
   // set the finder
   node->SetFinder(GetFinder());
   return node;
}

/*************************************************************************
 * TGeoIterator - a geometry iterator
 *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////////
// TGeoIterator
//==============
// A geometry iterator that sequentially follows all nodes of the geometrical
// hierarchy of a volume. The iterator has to be initiated with a top volume 
// pointer:
//
//    TGeoIterator next(myVolume);
//
// One can use the iterator as any other in ROOT:
//
//    TGeoNode *node;
//    while ((node=next())) {
//       ...
//    }
// 
// The iterator can perform 2 types of iterations that can be selected via:
//
//    next.SetType(Int_t type);
//
// Here TYPE can be:
//    0 (default) - 'first daughter next' behavior
//    1           - iteration at the current level only
//
// Supposing the tree structure looks like:
//
// TOP ___ A_1 ___ A1_1 ___ A11_1
//    |       |        |___ A12_1
//    |      |_____A2_1 ___ A21_1
//    |                |___ A21_2
//    |___ B_1 ...
//
// The order of iteration for TYPE=0 is: A_1, A1_1, A11_1, A12_1, A2_1, A21_1,
// A21_2, B_1, ...
// The order of iteration for TYPE=1 is: A_1, B_1, ...
// At any moment during iteration, TYPE can be changed. If the last iterated node
// is for instance A1_1 and the iteration type was 0, one can do:
//
//    next.SetType(1);
// The next iterated nodes will be the rest of A daughters: A2,A3,... The iterator
// will return 0 after finishing all daughters of A.
//
// During iteration, the following can be retreived:
// - Top volume where iteration started:    TGeoIterator::GetTopVolume()
// - Node at level I in the current branch: TGeoIterator::GetNode(Int_t i)
// - Iteration type:                        TGeoIterator::GetType()
// - Global matrix of the current node with respect to the top volume:
//                                          TGeoIterator::GetCurrentMatrix()
//
// The iterator can be reset by changing (or not) the top volume:
//
//    TGeoIterator::Reset(TGeoVolume *top);
//
// Example:
//==========
// We want to find out a volume named "MyVol" in the hierarchy of TOP volume.
// 
//    TIter next(TOP);
//    TGeoNode *node;
//    TString name("MyVol");
//    while ((node=next())) 
//       if (name == node->GetVolume()->GetName()) return node->GetVolume();
//
////////////////////////////////////////////////////////////////////////////////

ClassImp(TGeoIteratorPlugin)
ClassImp(TGeoIterator)
//_____________________________________________________________________________
TGeoIterator::TGeoIterator(TGeoVolume *top)
{
// Geometry iterator for a branch starting with a TOP node.
   fTop = top;
   fLevel = 0;
   fMustResume = kFALSE;
   fMustStop = kFALSE;
   fType = 0;
   fArray = new Int_t[30];
   fMatrix = new TGeoHMatrix();
   fTopName = fTop->GetName();
   fPlugin = 0;
   fPluginAutoexec = kFALSE;
}   

//_____________________________________________________________________________
TGeoIterator::TGeoIterator(const TGeoIterator &iter)
{
// Copy ctor.
   fTop = iter.GetTopVolume();
   fLevel = iter.GetLevel();
   fMustResume = kFALSE;
   fMustStop = kFALSE;
   fType = iter.GetType();
   fArray = new Int_t[30+ 30*Int_t(fLevel/30)];
   for (Int_t i=0; i<fLevel+1; i++) fArray[i] = iter.GetIndex(i);
   fMatrix = new TGeoHMatrix(*iter.GetCurrentMatrix());
   fTopName = fTop->GetName();
   fPlugin = iter.fPlugin;
   fPluginAutoexec = iter.fPluginAutoexec;;
}

//_____________________________________________________________________________
TGeoIterator::~TGeoIterator()
{
// Destructor.
   if (fArray) delete [] fArray;
   delete fMatrix;
}   

//_____________________________________________________________________________
TGeoIterator &TGeoIterator::operator=(const TGeoIterator &iter)
{
// Assignment.
   if (&iter == this) return *this;
   fTop = iter.GetTopVolume();
   fLevel = iter.GetLevel();
   fMustResume = kFALSE;
   fMustStop = kFALSE;
   fType = iter.GetType();
   if (fArray) delete [] fArray;
   fArray = new Int_t[30+ 30*Int_t(fLevel/30)];
   for (Int_t i=0; i<fLevel+1; i++) fArray[i] = iter.GetIndex(i);
   if (!fMatrix) fMatrix = new TGeoHMatrix();
   *fMatrix = *iter.GetCurrentMatrix();
   fTopName = fTop->GetName();
   fPlugin = iter.fPlugin;
   fPluginAutoexec = iter.fPluginAutoexec;;
   return *this;   
}   

//_____________________________________________________________________________
TGeoNode *TGeoIterator::Next()
{
// Returns next node.
   if (fMustStop) return 0;
   TGeoNode *mother = 0;
   TGeoNode *next = 0;
   Int_t i;
   Int_t nd = fTop->GetNdaughters();
   if (!nd) {
      fMustStop = kTRUE;
      return 0;
   }   
   if (!fLevel) {
      fArray[++fLevel] = 0;
      next = fTop->GetNode(0);
      if (fPlugin && fPluginAutoexec) fPlugin->ProcessNode();
      return next;
   }   
   next = fTop->GetNode(fArray[1]);
   // Move to current node
   for (i=2; i<fLevel+1; i++) {
      mother = next;
      next = mother->GetDaughter(fArray[i]);
   }   
   if (fMustResume) {
      fMustResume = kFALSE;
      if (fPlugin && fPluginAutoexec) fPlugin->ProcessNode();
      return next;
   }   
   
   switch (fType) {
      case 0:  // default next daughter behavior
         nd = next->GetNdaughters();
         if (nd) {
            // First daughter next
            fLevel++;
            if ((fLevel%30)==0) IncreaseArray();
            fArray[fLevel] = 0;
            if (fPlugin && fPluginAutoexec) fPlugin->ProcessNode();
            return next->GetDaughter(0);
         }
         // cd up and pick next
         while (next) {
            next = GetNode(fLevel-1);
            if (!next) {
               nd = fTop->GetNdaughters();
               if (fArray[fLevel]<nd-1) {
                  fArray[fLevel]++;
                  if (fPlugin && fPluginAutoexec) fPlugin->ProcessNode();
                  return fTop->GetNode(fArray[fLevel]);
               }   
               fMustStop = kTRUE;
               return 0;
            } else {
               nd = next->GetNdaughters();
               if (fArray[fLevel]<nd-1) {
                  fArray[fLevel]++;
                  if (fPlugin && fPluginAutoexec) fPlugin->ProcessNode();
                  return next->GetDaughter(fArray[fLevel]);
               }   
            }   
            fLevel--;
         }   
         break;
      case 1:  // one level search
         if (mother) nd = mother->GetNdaughters();
         if (fArray[fLevel]<nd-1) {
            fArray[fLevel]++;
            if (fPlugin && fPluginAutoexec) fPlugin->ProcessNode();
            if (!mother) return fTop->GetNode(fArray[fLevel]);
            else return mother->GetDaughter(fArray[fLevel]);
         }
   }
   fMustStop = kTRUE;
   return 0;
}
   
//_____________________________________________________________________________
TGeoNode *TGeoIterator::operator()()
{
// Returns next node.
   return Next();
}   

//_____________________________________________________________________________
const TGeoMatrix *TGeoIterator::GetCurrentMatrix() const
{
// Returns global matrix for current node.
   fMatrix->Clear();
   if (!fLevel) return fMatrix;
   TGeoNode *node = fTop->GetNode(fArray[1]);
   fMatrix->Multiply(node->GetMatrix());
   for (Int_t i=2; i<fLevel+1; i++) {
      node = node->GetDaughter(fArray[i]);
      fMatrix->Multiply(node->GetMatrix());
   }
   return fMatrix;   
}   

//_____________________________________________________________________________
TGeoNode *TGeoIterator::GetNode(Int_t level) const
{
// Returns current node at a given level.
   if (!level || level>fLevel) return 0;
   TGeoNode *node = fTop->GetNode(fArray[1]);
   for (Int_t i=2; i<level+1; i++) node = node->GetDaughter(fArray[i]);
   return node;
}

//_____________________________________________________________________________
void TGeoIterator::GetPath(TString &path) const
{
// Returns the path for the current node.
   path = fTopName;
   if (!fLevel) return;
   TGeoNode *node = fTop->GetNode(fArray[1]);
   path += "/";
   path += node->GetName();
   for (Int_t i=2; i<fLevel+1; i++) {
      node = node->GetDaughter(fArray[i]);
      path += "/";
      path += node->GetName();
   }   
}      

//_____________________________________________________________________________
void TGeoIterator::IncreaseArray() 
{
// Increase by 30 the size of the array.
   Int_t *array = new Int_t[fLevel+30];
   memcpy(array, fArray, fLevel*sizeof(Int_t));
   delete [] fArray;
   fArray = array;
}   
 
//_____________________________________________________________________________
void TGeoIterator::Reset(TGeoVolume *top)
{
// Resets the iterator for volume TOP.
   if (top) fTop = top;
   fLevel = 0;
   fMustResume = kFALSE;
   fMustStop = kFALSE;
}      

//_____________________________________________________________________________
void TGeoIterator::SetTopName(const char *name)
{
// Set the top name for path
   fTopName = name;
}   

//_____________________________________________________________________________
void TGeoIterator::Skip()
{
// Stop iterating the current branch. The iteration of the next node will
// behave as if the branch starting from the current node (included) is not existing.
   fMustResume = kTRUE;
   TGeoNode *next = GetNode(fLevel);
   if (!next) return;
   Int_t nd;
   switch (fType) {
      case 0:  // default next daughter behavior
         // cd up and pick next
         while (next) {
            next = GetNode(fLevel-1);
            nd = (next==0)?fTop->GetNdaughters():next->GetNdaughters();
            if (fArray[fLevel]<nd-1) {
               ++fArray[fLevel];
               return;
            }
            fLevel--;
            if (!fLevel) {
               fMustStop = kTRUE;
               return;
            }   
         }     
         break;
      case 1:  // one level search
         next = GetNode(fLevel-1);
         nd = (next==0)?fTop->GetNdaughters():next->GetNdaughters();
         if (fArray[fLevel]<nd-1) {
            ++fArray[fLevel];
            return;
         }
         fMustStop = kTRUE;   
         break;
   }
}         

//_____________________________________________________________________________
void TGeoIterator::SetUserPlugin(TGeoIteratorPlugin *plugin)
{
// Set a plugin.
   fPlugin = plugin;
   if (plugin) plugin->SetIterator(this);
}   
