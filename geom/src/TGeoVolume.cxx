// @(#)root/geom:$Name:  $:$Id: TGeoVolume.cxx,v 1.19 2003/01/15 18:43:44 brun Exp $
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
//   A volume is referencing a shape and a medium. These have to built BEFORE the 
// volume itself - see TGeoMaterial::TGeoMaterial() , TGeoShape::TGeoShape() . 
// Volumes must have unique names and any positioned volume (node) will append a
// copy number to the volume's name. For instance if a volume named PAD is 
// referenced in several nodes, their names will become : PAD_1, PAD_2, ...
//
//   A volume can be created with the sequence :
//
//        TGeoSphere   *sph = new TGeoSphere("sph1", 10.0, 11.0);
//        TGeoMedium   *med = gGeoManager->GetMedium("lead");
//        TGeoVolume   *vol = new TGeoVolume("shield", sph, med);
//   
//   The volume is registering itself to the current TGeoManager and can be
// retrieved at any time with :
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
#include "TH2F.h"

#include "TGeoManager.h"
#include "TGeoNode.h"
#include "TGeoMatrix.h"
#include "TVirtualGeoPainter.h"
#include "TGeoVolume.h"

ClassImp(TGeoVolume)

//-----------------------------------------------------------------------------
TGeoVolume::TGeoVolume()
{ 
// dummy constructor
   fNodes    = 0;
   fShape    = 0;
   fFinder   = 0;
   fVoxels   = 0;
   fField    = 0;
   fMedium   = 0;
   fOption   = "";
   TObject::ResetBit(kVolumeImportNodes);
}
//-----------------------------------------------------------------------------
TGeoVolume::TGeoVolume(const char *name, const TGeoShape *shape, const TGeoMedium *med)
           :TNamed(name, "")
{
// default constructor
   fNodes    = 0;
   fShape    = (TGeoShape*)shape;
   fFinder   = 0;
   fVoxels   = 0;
   fField    = 0;
   fOption   = "";
   fMedium   = (TGeoMedium*)med;
   if (gGeoManager) gGeoManager->AddVolume(this);
   TObject::ResetBit(kVolumeImportNodes);
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
   if (fFinder && !TObject::TestBit(kVolumeImportNodes)) delete fFinder;
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
void TGeoVolume::CheckGeometry(Int_t nrays, Double_t startx, Double_t starty, Double_t startz) const
{
// Shoot nrays with random directions from starting point (startx, starty, startz)
// in the reference frame of this volume. Track each ray until exiting geometry, then
// shoot backwards from exiting point and compare boundary crossing points.
   TGeoVolume *old_vol = gGeoManager->GetTopVolume();
   if (old_vol!=this) gGeoManager->SetTopVolume((TGeoVolume*)this);
   else old_vol=0;
   gGeoManager->GetTopVolume()->Draw();
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) {
      Error("CheckGeometry", "Could not instanciate painter");
      return;
   }
   painter->CheckGeometry(nrays, startx, starty, startz);
//   if (old_vol) gGeoManager->SetTopVolume(old_vol);
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
// THIS METHOD LEAVES SOME GARBAGE NODES -> memory leak, to be fixed
//   printf("---Checking daughters of volume %s\n", GetName());
   if (!fNodes) return;
   Int_t nd=fNodes->GetEntriesFast();
   TGeoNode *node = 0;
   TGeoNode *new_node;
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
//         printf("   Node %s/%s has shape with negative parameters. \n", 
//                 GetName(), node->GetName());
//         old_vol->InspectShape();
         // make a copy of the node
         new_node = node->MakeCopyNode();
         TGeoShape *new_shape = shape->GetMakeRuntimeShape(fShape);
         if (!new_shape) {
            Error("CheckShapes","cannot resolve runtime shape for volume %s/%s\n",
                   GetName(),old_vol->GetName());
            continue;
         }         
         TGeoVolume *new_volume = old_vol->MakeCopyVolume();
         new_volume->SetShape(new_shape);
//         printf(" new volume %s shape params :\n", new_volume->GetName());
//         new_volume->InspectShape();
         new_node->SetVolume(new_volume);
         // decouple the old node and put the new one instead
         fNodes->AddAt(new_node, i);
//         new_volume->CheckShapes();
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
   fMedium->GetMaterial()->Print();
}
//-----------------------------------------------------------------------------
void TGeoVolume::cd(Int_t inode) const
{
// Actualize matrix of node indexed <inode>
   if (fFinder) fFinder->cd(inode-fFinder->GetDivIndex());
}
//-----------------------------------------------------------------------------
void TGeoVolume::AddNode(const TGeoVolume *vol, Int_t copy_no, const TGeoMatrix *mat, Option_t * /*option*/)
{
// Add a TGeoNode to the list of nodes. This is the usual method for adding
// daughters inside the container volume.
   TGeoMatrix *matrix = (mat==0)?gGeoIdentity:(TGeoMatrix*)mat;
   if (!vol) {
      Error("AddNode", "Volume is NULL");
      return;
   }
   if (!vol->IsValid()) {
      Error("AddNode", "Won't add node with invalid shape");
      printf("### invalid volume was : %s\n", vol->GetName());
      return;
   }
   if (!fNodes) fNodes = new TObjArray();   

   if (fFinder) {
      // volume already divided.
      Error("AddNode", "Cannot add node %s_%i into divided volume %s", vol->GetName(), copy_no, GetName());
      return;
   }

   TGeoNodeMatrix *node = new TGeoNodeMatrix(vol, matrix);
   node->SetMotherVolume(this);
   fNodes->Add(node);
   char *name = new char[strlen(vol->GetName())+7];
   sprintf(name, "%s_%i", vol->GetName(), copy_no);
   if (fNodes->FindObject(name))
      Warning("AddNode", "Volume %s : added node %s with same name", GetName(), name);
   node->SetName(name);
}
//-----------------------------------------------------------------------------
void TGeoVolume::AddNodeOffset(const TGeoVolume *vol, Int_t copy_no, Double_t offset, Option_t * /*option*/)
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
void TGeoVolume::AddNodeOverlap(const TGeoVolume *vol, Int_t copy_no, const TGeoMatrix *mat, Option_t * /*option*/)
{
// Add a TGeoNode to the list of nodes. This is the usual method for adding
// daughters inside the container volume.
   TGeoMatrix *matrix = (mat==0)?gGeoIdentity:(TGeoMatrix*)mat;
   if (!vol) {
      Error("AddNodeOverlap", "Volume is NULL");
      return;
   }
   if (!vol->IsValid()) {
      Error("AddNodeOverlap", "Won't add node with invalid shape");
      printf("### invalid volume was : %s\n", vol->GetName());
      return;
   }
   if (!fNodes) fNodes = new TObjArray();   

   if (fFinder) {
      // volume already divided.
      Error("AddNodeOverlap", "Cannot add node %s_%i into divided volume %s", vol->GetName(), copy_no, GetName());
      return;
   }

   TGeoNodeMatrix *node = new TGeoNodeMatrix(vol, matrix);
   node->SetMotherVolume(this);
   fNodes->Add(node);
   char *name = new char[strlen(vol->GetName())+7];
   sprintf(name, "%s_%i", vol->GetName(), copy_no);
   if (fNodes->FindObject(name))
      Warning("AddNode", "Volume %s : added node %s with same name", GetName(), name);
   node->SetName(name);
   node->SetOverlapping();
   if (vol->GetMedium() == fMedium)
   node->SetVirtual();
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoVolume::Divide(const char * /*divname*/, Int_t /*ndiv*/, Option_t * /*option*/)
{
   Error("Divide", "This type of division not implemenetd");
   return this; 
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoVolume::Divide(const char * /*divname*/, Int_t /*ndiv*/, Double_t /*start*/, Double_t /*step*/, Option_t * /*option*/)
{
// divide this volume in ndiv pieces from start, with given step
   Error("Divide", "This type of division not implemenetd");
   return this; 
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoVolume::Divide(const char * /*divname*/, Double_t /*start*/, Double_t /*end*/, Double_t /*step*/, Option_t * /*option*/)
{
// divide this volume from start to end in pieces of length step
   Error("Divide", "This type of division not implemenetd");
   return this; 
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoVolume::Divide(const char * /*divname*/, TObject * /*userdiv*/, Double_t * /*params*/, Option_t *)
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
//   printf("--- dividing %s into %s, ndiv=%i\n", GetName(), divname, ndiv);
   TString stype = fShape->ClassName();
//   TGeoVolume *vol = 0;
   if (!ndiv && start != 0) {
      printf("Error : Divide %s type %s into %s- ndivisions=0\n",GetName(), stype.Data(), divname);
      return this;
   }
   if (!fNodes) fNodes = new TObjArray();
   if ((!ndiv) && (start == 0)) return fShape->Divide(this, divname, iaxis, step);
   if (fFinder) {
   // volume already divided.
      Error("Divide","volume %s already divided", GetName());
      return 0;
//      for (Int_t idiv=0; idiv<fFinder->GetNdiv(); idiv++) {
//         vol = fFinder->GetNodeOffset(idiv)->GetVolume();
//         vol->Divide(divname, iaxis, ndiv, start, step);
//      }
//      return this;
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
   TGeoVolume *old_vol = gGeoManager->GetTopVolume();
   if (old_vol!=this) gGeoManager->SetTopVolume(this);
   else old_vol=0;
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) return;
   painter->DrawOnly(option);   
}
//-----------------------------------------------------------------------------
Bool_t TGeoVolume::OptimizeVoxels()
{
// Perform an exensive sampling to find which type of voxelization is
// most efficient.
   printf("Optimizing volume %s ...\n", GetName());
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) return kFALSE;
   return painter->TestVoxels(this);   
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
//______________________________________________________________________________
TH2F *TGeoVolume::LegoPlot(Int_t ntheta, Double_t themin, Double_t themax,
                            Int_t nphi,   Double_t phimin, Double_t phimax,
                            Double_t rmin, Double_t rmax, Option_t *option)
{
// Generate a lego plot fot the top volume, according to option.
   TVirtualGeoPainter *p = gGeoManager->GetGeomPainter();
   if (!p) return 0;
   TGeoVolume *old_vol = gGeoManager->GetTopVolume();
   if (old_vol!=this) gGeoManager->SetTopVolume(this);
   else old_vol=0;
   TH2F *hist = p->LegoPlot(ntheta, themin, themax, nphi, phimin, phimax, rmin, rmax, option);   
   hist->Draw("lego1sph");
   return hist;
}
//-----------------------------------------------------------------------------
void TGeoVolume::RandomPoints(Int_t npoints, Option_t *option)
{
// Draw random points in the bounding box of this volume.
   gGeoManager->RandomPoints(this, npoints, option);
}
//-----------------------------------------------------------------------------
void TGeoVolume::RandomRays(Int_t nrays, Double_t startx, Double_t starty, Double_t startz)
{
// Random raytracing method.
   TGeoVolume *old_vol = gGeoManager->GetTopVolume();
   if (old_vol!=this) gGeoManager->SetTopVolume(this);
   else old_vol=0;
   gGeoManager->RandomRays(nrays, startx, starty, startz);
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
Int_t TGeoVolume::GetNodeIndex(const TGeoNode *node, Int_t *check_list, Int_t ncheck) const
{
   TGeoNode *current = 0;
   for (Int_t i=0; i<ncheck; i++) {
      current = (TGeoNode*)fNodes->At(check_list[i]);
      if (current==node) return check_list[i];
   }
   return -1;
}
//-----------------------------------------------------------------------------
Int_t TGeoVolume::GetIndex(const TGeoNode *node) const
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
Bool_t TGeoVolume::GetOptimalVoxels() const
{
//--- Returns true if cylindrical voxelization is optimal.
   Int_t nd = GetNdaughters();
   if (!nd) return kFALSE;
   Int_t id;
   Int_t ncyl = 0;
   TGeoNode *node;
   for (id=0; id<nd; id++) {
      node = (TGeoNode*)fNodes->At(id);
      ncyl += node->GetOptimalVoxels();
   }
   if (ncyl>(nd/2)) return kTRUE;
   return kFALSE;
}      
//-----------------------------------------------------------------------------
void TGeoVolume::MakeCopyNodes(const TGeoVolume *other)
{
// make a new list of nodes and copy all nodes of other volume inside
   Int_t nd = other->GetNdaughters();
   if (!nd) return;
   if (fNodes) {
//      printf("Warning : volume %s had already nodes -> replace them\n", GetName());
      delete fNodes;
   }
   fNodes = new TObjArray();
//   printf("other : %s\n nd=%i", other->GetName(), nd);
   for (Int_t i=0; i<nd; i++) fNodes->Add(other->GetNode(i));
   TObject::SetBit(kVolumeImportNodes);
}      
//-----------------------------------------------------------------------------
void TGeoVolume::GrabFocus()
{
// Move perspective view focus to this volume
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (painter) painter->GrabFocus();
}   
//-----------------------------------------------------------------------------
TGeoVolume *TGeoVolume::MakeCopyVolume()
{
    // make a copy of this volume
//    printf("   Making a copy of %s\n", GetName());
    char *name = new char[strlen(GetName())+1];
    sprintf(name, "%s", GetName());
    // build a volume with same name, shape and medium
    Bool_t is_runtime = fShape->IsRunTimeShape();
    if (is_runtime) fShape->SetRuntime(kFALSE);
    TGeoVolume *vol = new TGeoVolume(name, fShape, fMedium);
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
//       Error("MakeCopyVolume", "volume %s divided", GetName());
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
    ((TObject*)vol)->SetBit(kVolumeImportNodes);
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
void TGeoVolume::SetShape(const TGeoShape *shape)
{
// set the shape associated with this volume
   if (!shape) {
      Error("SetShape", "No shape");
      return;
   }
   fShape = (TGeoShape*)shape;  
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
void TGeoVolume::Streamer(TBuffer &R__b)
{
   // Stream an object of class TGeoVolume.
   if (R__b.IsReading()) {
      TGeoVolume::Class()->ReadBuffer(R__b, this);
   } else {
      if (!fVoxels) {
         TGeoVolume::Class()->WriteBuffer(R__b, this);
      } else {
         if (!gGeoManager->IsStreamingVoxels()) {
            TGeoVoxelFinder *voxels = fVoxels;
            fVoxels = 0;
            TGeoVolume::Class()->WriteBuffer(R__b, this);
            fVoxels = voxels;
         } else {
            TGeoVolume::Class()->WriteBuffer(R__b, this);
         }
      }
   }
}

//-----------------------------------------------------------------------------
void TGeoVolume::SetOption(const char * /*option*/)
{
// set the current options  
}
//-----------------------------------------------------------------------------
void TGeoVolume::SetLineColor(Color_t lcolor) 
{
   TAttLine::SetLineColor(lcolor);
   //if (gGeoManager->IsClosed()) SetVisTouched(kTRUE);
}   
//-----------------------------------------------------------------------------
void TGeoVolume::SetLineStyle(Style_t lstyle) 
{
   TAttLine::SetLineStyle(lstyle);
   //if (gGeoManager->IsClosed()) SetVisTouched(kTRUE);
}   
//-----------------------------------------------------------------------------
void TGeoVolume::SetLineWidth(Style_t lwidth) 
{
   TAttLine::SetLineWidth(lwidth);
   //if (gGeoManager->IsClosed()) SetVisTouched(kTRUE);
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
Int_t TGeoVolume::GetByteCount() const
{
// get the total size in bytes for this volume
   Int_t count = 28+2+6+4+0;    // TNamed+TGeoAtt+TAttLine+TAttFill+TAtt3D
   count += strlen(GetName()) + strlen(GetTitle()); // name+title
   count += 4+4+4+4+4; // fShape + fMedium + fFinder + fField + fNodes
   count += 8 + strlen(fOption.Data()); // fOption
   if (fShape)  count += fShape->GetByteCount();
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
void TGeoVolume::SetVisibility(Bool_t vis)
{
// set visibility of this volume
   TGeoAtt::SetVisibility(vis);
   if (gGeoManager->IsClosed()) SetVisTouched(kTRUE);
   gGeoManager->ModifiedPad();
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
   if (gGeoManager->IsClosed()) SetVisTouched(kTRUE);
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
   // do not voxelize divided volumes
   if (fFinder) return;
   // or final leaves
   Int_t nd = GetNdaughters();
   if (!nd) return;
   // delete old voxelization if any
   if (fVoxels) {
      delete fVoxels;
      fVoxels = 0;
   }   
   // see if a given voxelization type is enforced
   if (IsCylVoxels()) {
      fVoxels = new TGeoCylVoxels(this);
      fVoxels->Voxelize(option);
      return;
   } else {
      if (IsXYZVoxels()) {
         fVoxels = new TGeoVoxelFinder(this);
         fVoxels->Voxelize(option);
         return;
      }
   }      
   // find optimal voxelization
   Bool_t cyltype = GetOptimalVoxels();
   if (cyltype) {
//      fVoxels = new TGeoCylVoxels(this);
      fVoxels = new TGeoVoxelFinder(this);
//      printf("%s cyl. voxels\n", GetName());
   } else {
      fVoxels = new TGeoVoxelFinder(this);
   }   
   fVoxels->Voxelize(option);
//   if (fVoxels) fVoxels->Print();
}

ClassImp(TGeoVolumeMulti)

//-----------------------------------------------------------------------------
TGeoVolumeMulti::TGeoVolumeMulti()
{ 
// dummy constructor
   fVolumes   = 0;
   fDivision = 0;
   fNdiv = 0;
   fAxis = 0;
   fStart = 0;
   fStep = 0;
   fAttSet = kFALSE;
   TObject::SetBit(kVolumeMulti);
}
//-----------------------------------------------------------------------------
TGeoVolumeMulti::TGeoVolumeMulti(const char *name, const TGeoMedium *med)
{
// default constructor
   fVolumes = new TObjArray();
   fDivision = 0;
   fNdiv = 0;
   fAxis = 0;
   fStart = 0;
   fStep = 0;
   fAttSet = kFALSE;
   TObject::SetBit(kVolumeMulti);
   SetName(name);
   SetMedium(med);
   gGeoManager->GetListOfGVolumes()->Add(this);
//   printf("--- volume multi %s created\n", name);
}
//-----------------------------------------------------------------------------
TGeoVolumeMulti::~TGeoVolumeMulti()
{
// Destructor
   if (fVolumes) delete fVolumes;
}
//-----------------------------------------------------------------------------
void TGeoVolumeMulti::AddVolume(TGeoVolume *vol) 
{
// Add a volume with valid shape to the list of volumes. Copy all existing nodes
// to this volume
   fVolumes->Add(vol);
   TGeoVolumeMulti *div;
   TGeoVolume *cell;
   if (fDivision) {
      div = (TGeoVolumeMulti*)vol->Divide(fDivision->GetName(), fAxis, fNdiv, fStart, fStep);
      div->MakeCopyNodes(fDivision);
      for (Int_t i=0; i<div->GetNvolumes(); i++) {
         cell = div->GetVolume(i);
         cell->MakeCopyNodes(fDivision);
      }
   }      
   if (fNodes)
      vol->MakeCopyNodes(this);
}
   
//-----------------------------------------------------------------------------
void TGeoVolumeMulti::AddNode(const TGeoVolume *vol, Int_t copy_no, const TGeoMatrix *mat, Option_t *option)
{
// Add a new node to the list of nodes. This is the usual method for adding
// daughters inside the container volume.
   TGeoVolume::AddNode(vol, copy_no, mat, option);
   Int_t nvolumes = fVolumes->GetEntriesFast();
   TGeoVolume *volume = 0;
   for (Int_t ivo=0; ivo<nvolumes; ivo++) {
      volume = GetVolume(ivo);
      volume->SetLineColor(GetLineColor());
      volume->SetLineStyle(GetLineStyle());
      volume->SetLineWidth(GetLineWidth());
      volume->SetVisibility(IsVisible());
      volume->AddNode(vol, copy_no, mat, option); 
   }
//   printf("--- vmulti %s : node %s added to %i components\n", GetName(), vol->GetName(), nvolumes);
}
//-----------------------------------------------------------------------------
void TGeoVolumeMulti::AddNodeOverlap(const TGeoVolume *vol, Int_t copy_no, const TGeoMatrix *mat, Option_t *option)
{
   TGeoVolume::AddNodeOverlap(vol, copy_no, mat, option);
   Int_t nvolumes = fVolumes->GetEntriesFast();
   TGeoVolume *volume = 0;
   for (Int_t ivo=0; ivo<nvolumes; ivo++) {
      volume = GetVolume(ivo);
      volume->SetLineColor(GetLineColor());
      volume->SetLineStyle(GetLineStyle());
      volume->SetLineWidth(GetLineWidth());
      volume->SetVisibility(IsVisible());
      volume->AddNodeOverlap(vol, copy_no, mat, option); 
   }
//   printf("--- vmulti %s : node ovlp %s added to %i components\n", GetName(), vol->GetName(), nvolumes);
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoVolumeMulti::Divide(const char *divname, Int_t iaxis, Int_t ndiv, Double_t start, Double_t step)
{
// division of multiple volumes
   if (fDivision) {
      Error("Divide", "volume %s already divided", GetName());
      return 0;
   }   
   Int_t nvolumes = fVolumes->GetEntriesFast();
   if (!nvolumes) {
      // this is a virtual volume
      fDivision = new TGeoVolumeMulti(divname, fMedium);
      fAxis = iaxis;
      fNdiv = ndiv;
      fStart = start;
      fStep = step;
      // nothing else to do at this stage
      return fDivision;
   }   
      
   TGeoVolume *vol = 0;
   fDivision = new TGeoVolumeMulti(divname, fMedium);
   fAxis = iaxis;
   fNdiv = ndiv;
   fStart = start;
   fStep = step;
   for (Int_t ivo=0; ivo<nvolumes; ivo++) {
      vol = GetVolume(ivo);
      vol->SetLineColor(GetLineColor());
      vol->SetLineStyle(GetLineStyle());
      vol->SetLineWidth(GetLineWidth());
      vol->SetVisibility(IsVisible());
      fDivision->AddVolume(vol->Divide(divname,iaxis,ndiv,start,step)); 
   }
//   printf("--- volume multi %s (%i volumes) divided\n", GetName(), nvolumes);
   return fDivision;
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
