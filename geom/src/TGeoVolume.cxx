// @(#)root/geom:$Name:  $:$Id: TGeoVolume.cxx,v 1.3 2002/07/10 19:24:16 brun Exp $
// Author: Andrei Gheata   30/05/02
// Divide() implemented by Mihaela Gheata

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////////
//   TGeoVolume - class containing the full description of a geometrical object. 
//
//   A volume is a geometrical container possibly including other objects inside.
// The volume does not store any information about his own position/transformation 
// nor any link to the upper level in the geometry tree. Therefore, the same 
// volume can be referenced several times in the geometry.
//
//   Positioned volumes are called nodes (see class TGeoNode) and can be placed
// only inside other volume. In order to define a correct geometry, nodes should
// not extend beyond the boundaries of their mother volume and must not overlap
// between each other. These conditions can become critical when tracking a
// geometry, so the package is provided with a simple but efficient checking
// tool (class TGeoChecker). Note that the nodes representing the daughters of
// a volume does NOT overlap with their mother - any point belonging to the
// daughter will automatically NOT belong to the mother any more. The geometry 
// tree built in this fashion is a CSG tree with constraints.
//
//Begin_Html
/*
<img src="gif/t_volume.jpg">
*/
//End_Html
//
//   A volume is referencing a shape and a material. These have to built BEFORE the 
// volume itself - see TGeoMaterial::TGeoMaterial() , TGeoShape::TGeoShape() . 
// Volumes must have unique names and any positioned volume (node) will append a
// copy number to the volume's name. For instance if a volume named PAD is 
// referenced in several nodes, their names will become : PAD_1, PAD_2, ...
//
//   A volume can be created with the sequence :
//
//        TGeoSphere   *sph = new TGeoSphere("sph1", 10.0, 11.0);
//        TGeoMaterial *mat = gGeoManager->GetMaterial("lead");
//        TGeoVolume   *vol = new TGeoVolume("shield", sph, mat);
//   
//   The volume is registering itself to the current TGeoManager and can be
// retreived at any time with :
//
//        TGeoVolume *vol = gGeoManager->GetVolume("shield");
//
// Deletion of volumes is also handled by TGeoManager class.
//   Positioning of other geometry nodes inside a volume is done by :
//        TGeoVolume::AddNode() method. The node to be placed does not have to 
// be created before :
//
//        TGeoVolume      *vol_in = ...;
//        TGeoTranslation *tr     = new TGeoTranslation(x, y, z);
//        TGeoNodeMatrix  *node   = vol->AddNodeMatrix (vol_in, tr, copy_number);
//
//   A volume can be divided according a pattern. The most simple division can
// be done along an axis, in cartesian, cylindrical or spherical coordinates. 
// For each division type there are corresponding TGeoVolume::AddNodeOffset() and
// TGeoVolume::Divide() methods. The option argument passed tothese methods can
// be :
//
//        X, Y, Z - for cartesian axis divisions;
//        CylR, CylPhi - for cylindrical divisions;
//        SphR, SphPhi, SphTheta - for spherical divisions;
//        honeycomb - for honeycomb structures
//
// For instance, dividing a volume in N segments along X axis, starting from Xmin 
// up to Xmax can be done with :
//        TGeoVolume::Divide(N, Xmin, Xmax, "X"); 
//
//   The GEANT3 option MANY is supported by TGeoVolumeOverlap class. An overlapping
// volume is in fact a virtual container that does not represent a physical object.
// It contains a list of nodes that are not his daughters but that must be checked 
// always before the container itself. This list must be defined by users and it 
// is checked and resolved in a priority order. Note that the feature is non-standard
// to geometrical modelers and it was introduced just to support conversions of 
// GEANT3 geometries, therefore its extensive usage should be avoided.
//
//   The following picture represent how a simple geometry tree is built in
// memory.
//Begin_Html
/*
<img src="gif/t_example.jpg">
*/
//End_Html

#include "TString.h"
#include "TBrowser.h"
#include "TStyle.h"

#include "TGeoManager.h"
#include "TGeoNode.h"
#include "TGeoMatrix.h"
#include "TGeoFinder.h"
#include "TVirtualGeoPainter.h"
#include "TGeoVolume.h"

ClassImp(TGeoVolume)

//-----------------------------------------------------------------------------
TGeoVolume::TGeoVolume()
{ 
// dummy constructor
   fUsageCount[0] = fUsageCount[1] = 0;
   fNodes    = 0;
   fShape    = 0;
   fMaterial = 0;
   fFinder   = 0;
   fVoxels   = 0;
   fField    = 0;
   fOption   = "";
}
//-----------------------------------------------------------------------------
TGeoVolume::TGeoVolume(const char *name, TGeoShape *shape, TGeoMaterial *mat)
           :TNamed(name, "")
{
// default constructor
   fUsageCount[0] = fUsageCount[1] = 0;
   fNodes    = 0;
   fShape    = shape;
   fMaterial = mat;
   fFinder   = 0;
   fVoxels   = 0;
   fField    = 0;
   fOption   = "";
   if (gGeoManager) gGeoManager->AddVolume(this);
}
//-----------------------------------------------------------------------------
TGeoVolume::~TGeoVolume()
{
// Destructor
   if (fNodes) { 
      if (!TObject::TestBit(kVolumeImportNodes)) {
         fNodes->Delete();
      }   
      delete fNodes;
   }
   if (fFinder) delete fFinder;
   if (fVoxels) delete fVoxels;
}
//-----------------------------------------------------------------------------
void TGeoVolume::Browse(TBrowser *b)
{
// How to browse a volume
   if (!b) return;
   if (!GetNdaughters()) b->Add(this);
   for (Int_t i=0; i<GetNdaughters(); i++) 
      b->Add(GetNode(i)->GetVolume());
}
//-----------------------------------------------------------------------------
void TGeoVolume::CleanAll()
{
   ClearNodes();
   ClearShape();
}
//-----------------------------------------------------------------------------
void TGeoVolume::ClearShape()
{
   gGeoManager->ClearShape(fShape);
}   
//-----------------------------------------------------------------------------
void TGeoVolume::CheckShapes()
{
// check for negative parameters in shapes.
//   printf("Checking %s\n", GetName());
   if (!fNodes) return;
   Int_t nd=fNodes->GetEntriesFast();
   TGeoNode *node = 0;
   const TGeoShape *shape = 0;
   TGeoVolume *old_vol;
   for (Int_t i=0; i<nd; i++) {
      node=(TGeoNode*)fNodes->At(i);
      // check if node has name
      if (!strlen(node->GetName())) printf("Daughter %i of volume %s - NO NAME!!!\n",
                                           i, GetName());
      old_vol = node->GetVolume();
      shape = old_vol->GetShape();
      if (shape->IsRunTimeShape()) {
//         printf("Node %s/%s has shape with negative parameters. \n", 
//                 GetName(), node->GetName());
//         old_vol->InspectShape();
         TGeoShape *new_shape = shape->GetMakeRuntimeShape(fShape);
         if (!new_shape) {
            printf("***ERROR - could not resolve runtime shape for volume %s\n",
                   old_vol->GetName());
            continue;
         }         
         TGeoVolume *new_volume = old_vol->MakeCopyVolume();
         new_volume->SetShape(new_shape);
//         printf(" new volume %s shape params :\n", new_volume->GetName());
//         new_volume->InspectShape();
         node->SetVolume(new_volume);
      }
   }
}     
//-----------------------------------------------------------------------------
Int_t TGeoVolume::CountNodes(Int_t nlevels) const
{
// count total number of subnodes starting from this volume, nlevels down
   Int_t count = 1;
   if (!fNodes || !nlevels) return 1;
   TIter next(fNodes);
   TGeoNode *node;
   TGeoVolume *vol;
   while ((node=(TGeoNode*)next())) {
      vol = node->GetVolume();
      count += vol->CountNodes(nlevels-1);
   }
   return count;
}
//-----------------------------------------------------------------------------
Bool_t TGeoVolume::IsFolder() const
{
// Return TRUE if volume contains nodes
   if (fNodes) return kTRUE;
   else return kFALSE;
}
//-----------------------------------------------------------------------------
Bool_t TGeoVolume::IsStyleDefault() const
{
// check if the visibility and attributes are the default ones
   if (!IsVisible()) return kFALSE;
   if (GetLineColor() != gStyle->GetLineColor()) return kFALSE;
   if (GetLineStyle() != gStyle->GetLineStyle()) return kFALSE;
   if (GetLineWidth() != gStyle->GetLineWidth()) return kFALSE;
   return kTRUE;
}
//-----------------------------------------------------------------------------
void TGeoVolume::InspectMaterial() const
{
   fMaterial->Print();
}
//-----------------------------------------------------------------------------
void TGeoVolume::cd(Int_t inode) const
{
// Actualize matrix of node indexed <inode>
   if (fFinder) fFinder->cd(inode-fFinder->GetDivIndex());
}
//-----------------------------------------------------------------------------
void TGeoVolume::AddNode(TGeoVolume *vol, Int_t copy_no, TGeoMatrix *mat, Option_t *option)
{
// Add a TGeoNodePos to the list of nodes. This is the usual method for adding
// daughters inside the container volume.
   if (!vol || !mat) {
      Error("AddNodeMatrix", "Volume/matrix not defined");
      return;
   }
   if (!vol->IsValid()) {
      Error("AddNode", "Won't add node with invalid shape");
      printf("### invalid volume was : %s\n", vol->GetName());
      return;
   }
   if (!fNodes) fNodes = new TObjArray();

   if (fFinder) {
   // volume already divided. Add the node to all its divisions.
      TGeoVolume *div_vol;
      for (Int_t idiv=0; idiv<fFinder->GetNdiv(); idiv++) {
         div_vol = fFinder->GetNodeOffset(idiv)->GetVolume();
         div_vol->AddNode(vol, copy_no, mat, option);
      }
      return;
   }

   TGeoNodeMatrix *node = new TGeoNodeMatrix(vol, mat);
   node->SetMotherVolume(this);
   fNodes->Add(node);
   char *name = new char[strlen(vol->GetName())+7];
   sprintf(name, "%s_%i", vol->GetName(), copy_no);
   node->SetName(name);
}
//-----------------------------------------------------------------------------
void TGeoVolume::AddNodeOffset(TGeoVolume *vol, Int_t copy_no, Double_t offset, Option_t *option)
{
// Add a division node to the list of nodes. The method is called by
// TGeoVolume::Divide() for creating the division nodes.
   if (!vol) {
      Error("AddNodeOffset", "invalid volume");
      return;
   }
   if (!vol->IsValid()) {
      Error("AddNode", "Won't add node with invalid shape");
      printf("### invalid volume was : %s\n", vol->GetName());
      return;
   }   
   if (!fNodes) fNodes = new TObjArray();
   TGeoNode *node = new TGeoNodeOffset(vol, copy_no, offset);
   node->SetMotherVolume(this);
   fNodes->Add(node);
   char *name = new char[strlen(vol->GetName())+7];
   sprintf(name, "%s_%i", vol->GetName(), copy_no+1);
   node->SetName(name);
}
//-----------------------------------------------------------------------------
void TGeoVolume::AddNodeOverlap(TGeoVolume *vol, Int_t copy_no, TGeoMatrix *mat, Option_t *option)
{
   if (!fFinder) {
      AddNode(vol, copy_no, mat, option);
      TGeoNode *node = (TGeoNode*)fNodes->At(GetNdaughters()-1);
      node->SetOverlaps(0, 1);
      if (vol->GetMaterial()->GetMedia() == fMaterial->GetMedia())
         node->SetVirtual();
      return;   
   } 
   TGeoVolume *div_vol;
   for (Int_t idiv=0; idiv<fFinder->GetNdiv(); idiv++) {
      div_vol = fFinder->GetNodeOffset(idiv)->GetVolume();
      div_vol->AddNodeOverlap(vol,copy_no,mat,option);
   }        
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoVolume::Divide(const char *divname, Int_t ndiv, Option_t *option)
{
   Error("Divide", "This type of division not implemenetd");
   return this; 
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoVolume::Divide(const char *divname, Int_t ndiv, Double_t start, Double_t step, Option_t *option)
{
// divide this volume in ndiv pieces from start, with given step
   Error("Divide", "This type of division not implemenetd");
   return this; 
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoVolume::Divide(const char *divname, Double_t start, Double_t end, Double_t step, Option_t *option)
{
// divide this volume from start to end in pieces of length step
   Error("Divide", "This type of division not implemenetd");
   return this; 
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoVolume::Divide(const char *divname, TObject *userdiv, Double_t *params, Option_t *)
{
// divide this volume according to userdiv
   Error("Divide", "This type of division not implemenetd");
   return this; 
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoVolume::Divide(const char *divname, Int_t iaxis, Double_t step)
{
// Divide all range of iaxis in range/step cells 
   return fShape->Divide(this, divname, iaxis, step);
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoVolume::Divide(const char *divname, Int_t iaxis, Int_t ndiv, Double_t start, Double_t step)
{
// division a la G3
   TString stype = fShape->ClassName();
   TGeoVolume *vol = 0;
   if (!ndiv && start) {
      printf("Error : Divide %s type %s into %s- ndivisions=0\n",GetName(), stype.Data(), divname);
      return this;
   }
   if (!fNodes) fNodes = new TObjArray();
   if ((!ndiv) && (!start)) return fShape->Divide(this, divname, iaxis, step);
   if (fFinder) {
   // volume already divided. Divide again all its divisions.
      for (Int_t idiv=0; idiv<fFinder->GetNdiv(); idiv++) {
         vol = fFinder->GetNodeOffset(idiv)->GetVolume();
         vol->Divide(divname, iaxis, ndiv, start, step);
      }
      return this;
   }
   return fShape->Divide(this, divname, iaxis, ndiv, start, step); 
}
//-----------------------------------------------------------------------------
Int_t TGeoVolume::DistancetoPrimitive(Int_t px, Int_t py)
{
// compute the closest distance of approach from point px,py to this volume
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) return 9999;
   return painter->DistanceToPrimitiveVol(this, px, py);
}
//-----------------------------------------------------------------------------
void TGeoVolume::Draw(Option_t *option)
{
// draw top volume according to option
   TGeoVolume *old_vol = gGeoManager->GetTopVolume();
   if (old_vol!=this) gGeoManager->SetTopVolume(this);
   else old_vol=0;
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) return;
   painter->Draw(option);   
}
//-----------------------------------------------------------------------------
void TGeoVolume::DrawOnly(Option_t *option)
{
// draw only this volume
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) return;
   painter->DrawOnly(option);   
}
//-----------------------------------------------------------------------------
void TGeoVolume::Paint(Option_t *option)
{
// paint volume
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) return;
   painter->Paint(option);   
}
//-----------------------------------------------------------------------------
void TGeoVolume::PrintVoxels() const
{
   if (fVoxels) fVoxels->Print();
}
//-----------------------------------------------------------------------------
void TGeoVolume::PrintNodes() const
{
// print nodes
   Int_t nd = GetNdaughters();
   for (Int_t i=0; i<nd; i++) {
      printf("%s\n", GetNode(i)->GetName());
      cd(i);
      GetNode(i)->GetMatrix()->Print();
   }   
}
//-----------------------------------------------------------------------------
void TGeoVolume::RandomPoints(Int_t npoints, Option_t *option)
{
// Draw random points in the bounding box of this volume.
   gGeoManager->RandomPoints(this, npoints, option);
}
//-----------------------------------------------------------------------------
void TGeoVolume::RandomRays(Int_t nrays)
{
// Random raytracing method.
   TGeoVolume *old_vol = gGeoManager->GetTopVolume();
   if (old_vol!=this) gGeoManager->SetTopVolume(this);
   else old_vol=0;
   gGeoManager->RandomRays(nrays);
}
//-----------------------------------------------------------------------------
void TGeoVolume::RenameCopy(Int_t copy_no)
{
// add a copy number to this volume in order to handle gsposp

   TString name = GetName();
   Int_t digits = 1;
   Int_t num = 10;
   while ((Int_t)(copy_no/num)) {
      digits++;
      num *= 10;
   }
   name += '_';
   char *newname = new char[name.Length()+digits];
   sprintf(newname, "%s%i", name.Data(), copy_no);
   SetName(newname);
}
//-----------------------------------------------------------------------------
void TGeoVolume::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
// Execute mouse actions on this volume.
   gGeoManager->GetGeomPainter()->ExecuteVolumeEvent(this, event, px, py);
}
//-----------------------------------------------------------------------------
TGeoNode *TGeoVolume::FindNode(const char *name) const
{
// search a daughter inside the list of nodes
   return ((TGeoNode*)fNodes->FindObject(name));
}
//-----------------------------------------------------------------------------
Int_t TGeoVolume::GetNodeIndex(TGeoNode *node, Int_t *check_list, Int_t ncheck) const
{
   TGeoNode *current = 0;
   for (Int_t i=0; i<ncheck; i++) {
      current = (TGeoNode*)fNodes->At(check_list[i]);
      if (current==node) return check_list[i];
   }
   return -1;
}
//-----------------------------------------------------------------------------
Int_t TGeoVolume::GetIndex(TGeoNode *node) const
{
// get index number for a given daughter
   TGeoNode *current = 0;
   Int_t nd = GetNdaughters();
   if (!nd) return -1;
   for (Int_t i=0; i<nd; i++) {
      current = (TGeoNode*)fNodes->At(i);
      if (current==node) return i;
   }
   return -1;
}
//-----------------------------------------------------------------------------
char *TGeoVolume::GetObjectInfo(Int_t px, Int_t py) const
{
   TGeoVolume *vol = (TGeoVolume*)this;
   return gGeoManager->GetGeomPainter()->GetVolumeInfo(vol, px, py);
}
//-----------------------------------------------------------------------------
void TGeoVolume::MakeCopyNodes(TGeoVolume *other)
{
// make a new list of nodes and copy all nodes of other volume inside
   Int_t nd = other->GetNdaughters();
   if (!nd) return;
   if (fNodes) {
      printf("Warning : volume %s had already nodes\n", GetName());
      delete fNodes;
   }
   fNodes = new TObjArray();
//   printf("other : %s\n nd=%i", other->GetName(), nd);
   for (Int_t i=0; i<nd; i++) fNodes->Add(other->GetNode(i));
}      
//-----------------------------------------------------------------------------
TGeoVolume *TGeoVolume::MakeCopyVolume()
{
    // make a copy of this volume
    char *name = new char[strlen(GetName())];
    sprintf(name, "%s", GetName());
    // build a volume with same name, shape and material
    Bool_t is_runtime = fShape->IsRunTimeShape();
    if (is_runtime) fShape->SetRuntime(kFALSE);
    TGeoVolume *vol = new TGeoVolume(name, fShape, fMaterial);
    if (is_runtime) fShape->SetRuntime();
    Int_t i=0;
    // copy volume attributes
    vol->SetVisibility(IsVisible());
    vol->SetLineColor(GetLineColor());
    vol->SetLineStyle(GetLineStyle());
    vol->SetLineWidth(GetLineWidth());
    vol->SetFillColor(GetFillColor());
    vol->SetFillStyle(GetFillStyle());
    // copy field
    vol->SetField(fField);
    // if divided, copy division object
    if (fFinder) {
       Error("MakeCopyVolume", "volume divided");
       vol->SetFinder(fFinder);
    }   
    if (!fNodes) return vol;
    TGeoNode *node;
    Int_t nd = fNodes->GetEntriesFast();
    if (!nd) return vol;
    // create new list of nodes
    TObjArray *list = new TObjArray();
    // attach it to new volume
    vol->SetNodes(list);
    for (i=0; i<nd; i++) {
       //create copies of nodes and add them to list
       node = GetNode(i)->MakeCopyNode();
       node->SetMotherVolume(vol);
       list->Add(node);
    }
    return vol;       
}    
//-----------------------------------------------------------------------------
void TGeoVolume::SetAsTopVolume()
{
   gGeoManager->SetTopVolume(this);
}
//-----------------------------------------------------------------------------
void TGeoVolume::SetCurrentPoint(Double_t x, Double_t y, Double_t z)
{
   gGeoManager->SetCurrentPoint(x,y,z);
}
//-----------------------------------------------------------------------------
void TGeoVolume::SetMaterial(TGeoMaterial *material)
{
// set the material associated with this volume
   if (!material) {
      Error("SetMaterial", "No material");
      return;
   }
   fMaterial = material;   
}
//-----------------------------------------------------------------------------
void TGeoVolume::SetShape(TGeoShape *shape)
{
// set the shape associated with this volume
   if (!shape) {
      Error("SetShape", "No shape");
      return;
   }
   fShape = shape;  
}
//-----------------------------------------------------------------------------
void TGeoVolume::Sizeof3D() const
{
//   Compute size of this 3d object.
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) return;
   painter->Sizeof3D(this);
}
//-----------------------------------------------------------------------------
void TGeoVolume::SortNodes()
{
// sort nodes by decreasing volume of the bounding box. ONLY nodes comes first,
// then overlapping nodes and finally division nodes.
   if (!Valid()) {
      Error("SortNodes", "Bounding box not valid");
      return;
   }
   Int_t nd = GetNdaughters();
//   printf("volume : %s, nd=%i\n", GetName(), nd);
   if (!nd) return;
   if (fFinder) return;
//   printf("Nodes for %s\n", GetName());
   Int_t id = 0;
   TGeoNode *node = 0;
   TObjArray *nodes = new TObjArray(nd);
   Int_t inode = 0;
   // first put ONLY's
   for (id=0; id<nd; id++) {
      node = GetNode(id);
      if (node->InheritsFrom("TGeoNodeOffset") || node->IsOverlapping()) continue;
      nodes->Add(node);
//      printf("inode %i ONLY\n", inode);
      inode++;
   }
   // second put overlapping nodes
   for (id=0; id<nd; id++) {
      node = GetNode(id);
      if (node->InheritsFrom("TGeoNodeOffset") || (!node->IsOverlapping())) continue;
      nodes->Add(node);
//      printf("inode %i MANY\n", inode);
      inode++;
   }
   // third put the divided nodes
   if (fFinder) {
      fFinder->SetDivIndex(inode);
      for (id=0; id<nd; id++) {
         node = GetNode(id);
         if (!node->InheritsFrom("TGeoNodeOffset")) continue;
         nodes->Add(node);
//         printf("inode %i DIV\n", inode);
         inode++;
      }
   }
   if (inode != nd) printf(" volume %s : number of nodes does not match!!!\n", GetName());
   delete fNodes;
   fNodes = nodes;
}
//-----------------------------------------------------------------------------
void TGeoVolume::SetOption(const char *option)
{
// set the current options  
}
//-----------------------------------------------------------------------------
void TGeoVolume::SetLineColor(Color_t lcolor) 
{
   TAttLine::SetLineColor(lcolor);
   if (gGeoManager->IsClosed()) SetVisTouched(kTRUE);
}   
//-----------------------------------------------------------------------------
void TGeoVolume::SetLineStyle(Style_t lstyle) 
{
   TAttLine::SetLineStyle(lstyle);
   if (gGeoManager->IsClosed()) SetVisTouched(kTRUE);
}   
//-----------------------------------------------------------------------------
void TGeoVolume::SetLineWidth(Style_t lwidth) 
{
   TAttLine::SetLineWidth(lwidth);
   if (gGeoManager->IsClosed()) SetVisTouched(kTRUE);
}   
//-----------------------------------------------------------------------------
TGeoNode *TGeoVolume::GetNode(const char *name) const
{
// get the pointer to a daughter node
   Int_t nd = fNodes->GetEntriesFast();
   TGeoNode *node;
   for (Int_t i=0; i<nd; i++) {
      node = (TGeoNode*)fNodes->At(i);
      if (!strcmp(node->GetName(), name)) return node;
   }   
   return 0;
}
//-----------------------------------------------------------------------------
Double_t TGeoVolume::GetUsageCount(Int_t i) const
{
// check usage count 
   return 0;
}
//-----------------------------------------------------------------------------
Int_t TGeoVolume::GetByteCount() const
{
// get the total size in bytes for this volume
   Int_t count = 28+2+6+4+0;    // TNamed+TGeoAtt+TAttLine+TAttFill+TAtt3D
   count += strlen(GetName()) + strlen(GetTitle()); // name+title
   count += 8+4+4+4+4+4; // fUsageCount[2] + fShape + fMaterial + fFinder + fField + fNodes
   count += 8 + strlen(fOption.Data()); // fOption
   if (fShape) count += fShape->GetByteCount();
//   if (fMaterial) count += fMaterial->GetByteCount();
   if (fFinder) count += fFinder->GetByteCount();
   if (fNodes) {
      count += 32 + 4*fNodes->GetEntries(); // TObjArray
      TIter next(fNodes);
      TGeoNode *node;
      while ((node=(TGeoNode*)next())) count += node->GetByteCount();
   }
   return count;
}
//-----------------------------------------------------------------------------
void TGeoVolume::FindOverlaps() const
{
// loop all nodes marked as overlaps and find overlaping brothers
   if (!Valid()) {
      Error("FindOverlaps","Bounding box not valid");
      return;
   }   
   if (!fVoxels) return;
   TIter next(fNodes);
   TGeoNode *node=0;
   Int_t inode = 0;
   while ((node=(TGeoNode*)next())) {
      if (!node->IsOverlapping()) {inode++; continue;}
      fVoxels->FindOverlaps(inode);
      inode++;
   }
}
//-----------------------------------------------------------------------------
Bool_t TGeoVolume::Valid() const
{
   return fShape->IsValidBox();
}
//-----------------------------------------------------------------------------
void TGeoVolume::VisibleDaughters(Bool_t vis)
{
// set visibility for daughters
   SetVisDaughters(vis);
   gGeoManager->ModifiedPad();
}
//-----------------------------------------------------------------------------
void TGeoVolume::Voxelize(Option_t *option)
{
// build the voxels for this volume 
   if (!Valid()) {
      Error("Voxelize", "Bounding box not valid");
      return; 
   }   
   if (fFinder || (!GetNdaughters())) return;
   if (fVoxels) delete fVoxels;
   fVoxels = new TGeoVoxelFinder(this);
   fVoxels->Voxelize(option);
//   if (fVoxels) fVoxels->Print();
}

ClassImp(TGeoVolumeMulti)

//-----------------------------------------------------------------------------
TGeoVolumeMulti::TGeoVolumeMulti()
{ 
// dummy constructor
   fVolumes   = 0;
   fAttSet = kFALSE;
   TObject::SetBit(kVolumeMulti);
}
//-----------------------------------------------------------------------------
TGeoVolumeMulti::TGeoVolumeMulti(const char *name, TGeoMaterial *mat)
{
// default constructor
   fVolumes = new TObjArray();
   fAttSet = kFALSE;
   TObject::SetBit(kVolumeMulti);
   SetName(name);
   SetMaterial(mat);
   gGeoManager->GetListOfGVolumes()->Add(this);
}
//-----------------------------------------------------------------------------
TGeoVolumeMulti::~TGeoVolumeMulti()
{
// Destructor
   delete fVolumes;
}
//-----------------------------------------------------------------------------
void TGeoVolumeMulti::AddNode(TGeoVolume *vol, Int_t copy_no, TGeoMatrix *mat, Option_t *option)
{
// Add a TGeoNodePos to the list of volumes. This is the usual method for adding
// daughters inside the container volume.
   Int_t nvolumes = fVolumes->GetEntriesFast();
   TGeoVolume *volume = 0;
   for (Int_t ivo=0; ivo<nvolumes; ivo++) {
      volume = GetVolume(ivo);
      if (!fAttSet) {
         volume->SetLineColor(GetLineColor());
         volume->SetLineStyle(GetLineStyle());
         volume->SetLineWidth(GetLineWidth());
         volume->SetVisibility(IsVisible());
         fAttSet = kTRUE;
      }   
      volume->AddNode(vol, copy_no, mat, option); 
   }
}
//-----------------------------------------------------------------------------
void TGeoVolumeMulti::AddNodeOverlap(TGeoVolume *vol, Int_t copy_no, TGeoMatrix *mat, Option_t *option)
{
   Int_t nvolumes = fVolumes->GetEntriesFast();
   TGeoVolume *volume = 0;
   for (Int_t ivo=0; ivo<nvolumes; ivo++) {
      volume = GetVolume(ivo);
      if (!fAttSet) {
         volume->SetLineColor(GetLineColor());
         volume->SetLineStyle(GetLineStyle());
         volume->SetLineWidth(GetLineWidth());
         volume->SetVisibility(IsVisible());
         fAttSet = kTRUE;
      }   
      volume->AddNodeOverlap(vol, copy_no, mat, option); 
   }
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoVolumeMulti::Divide(const char *divname, Int_t iaxis, Int_t ndiv, Double_t start, Double_t step)
{
// division a la G3
   Int_t nvolumes = fVolumes->GetEntriesFast();
   TGeoVolume *vol = 0;
   TGeoVolumeMulti *div = new TGeoVolumeMulti(divname, fMaterial);
   for (Int_t ivo=0; ivo<nvolumes; ivo++) {
      vol = GetVolume(ivo);
      if (!fAttSet) {
         vol->SetLineColor(GetLineColor());
         vol->SetLineStyle(GetLineStyle());
         vol->SetLineWidth(GetLineWidth());
         vol->SetVisibility(IsVisible());
         fAttSet = kTRUE;
      }   
      div->AddVolume(vol->Divide(divname,iaxis,ndiv,start,step)); 
   }
   return div;
}
//-----------------------------------------------------------------------------
void TGeoVolumeMulti::SetLineColor(Color_t lcolor) 
{
   TGeoVolume::SetLineColor(lcolor);
   Int_t nvolumes = fVolumes->GetEntriesFast();
   TGeoVolume *vol = 0;
   for (Int_t ivo=0; ivo<nvolumes; ivo++) {
      vol = GetVolume(ivo);
      vol->SetLineColor(lcolor); 
   }
}
//-----------------------------------------------------------------------------
void TGeoVolumeMulti::SetLineStyle(Style_t lstyle) 
{
   TGeoVolume::SetLineStyle(lstyle); 
   Int_t nvolumes = fVolumes->GetEntriesFast();
   TGeoVolume *vol = 0;
   for (Int_t ivo=0; ivo<nvolumes; ivo++) {
      vol = GetVolume(ivo);
      vol->SetLineStyle(lstyle); 
   }
}
//-----------------------------------------------------------------------------
void TGeoVolumeMulti::SetLineWidth(Width_t lwidth) 
{
   TGeoVolume::SetLineWidth(lwidth);
   Int_t nvolumes = fVolumes->GetEntriesFast();
   TGeoVolume *vol = 0;
   for (Int_t ivo=0; ivo<nvolumes; ivo++) {
      vol = GetVolume(ivo);
      vol->SetLineWidth(lwidth); 
   }
}
//-----------------------------------------------------------------------------
void TGeoVolumeMulti::SetVisibility(Bool_t vis) 
{
   TGeoVolume::SetVisibility(vis); 
   Int_t nvolumes = fVolumes->GetEntriesFast();
   TGeoVolume *vol = 0;
   for (Int_t ivo=0; ivo<nvolumes; ivo++) {
      vol = GetVolume(ivo);
      vol->SetVisibility(vis); 
   }
}
