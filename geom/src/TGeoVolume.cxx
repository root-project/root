// @(#)root/geom:$Name:$:$Id:$
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
#include "TObjArray.h"
#include "TBrowser.h"
#include "TPolyMarker3D.h"
#include "TPad.h"
#include "TView.h"
#include "TStyle.h"
#include "TRandom3.h"

#include "TGeoManager.h"
#include "TGeoPara.h"
#include "TGeoTube.h"
#include "TGeoCone.h"
#include "TGeoSphere.h"
#include "TGeoArb8.h"
#include "TGeoPgon.h"
#include "TGeoTrd1.h"
#include "TGeoTrd2.h"
#include "TGeoCompositeShape.h"
#include "TGeoNode.h"
#include "TGeoMatrix.h"
#include "TGeoFinder.h"
#include "TGeoVolume.h"
#include "TVirtualGeoPainter.h"

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
void TGeoVolume::CheckPoint() const
{
   if (!gPad) return;
   Double_t point[3];
   memcpy(&point[0], gGeoManager->GetCurrentPoint(), 3*sizeof(Double_t));
   TPolyMarker3D *pm = new TPolyMarker3D();
   pm->SetMarkerColor(2);
   pm->SetMarkerStyle(4);
   pm->SetNextPoint(point[0], point[1], point[2]);
   Double_t dx = 1;
   Double_t dmin = 100;
   Double_t ox, oy, oz, x, y, z;
   ox = point[0];
   oy = point[1];
   oz = point[2];
   gRandom = new TRandom3();
   TGeoNode *node = gGeoManager->FindNode();
   TGeoNode *last, *closest=0;
   for (Int_t i=0; i<10000; i++) {
      x = ox-dx+2.*dx*gRandom->Rndm();
      y = oy-dx+2.*dx*gRandom->Rndm();
      z = oz-dx+2.*dx*gRandom->Rndm();
      gGeoManager->SetCurrentPoint(x,y,z);
      last = gGeoManager->FindNode();
      if (last!=node) { 
         dmin=TMath::Min(dmin,TMath::Sqrt((x-ox)*(x-ox)+(y-oy)*(y-oy)+(z-oz)*(z-oz)));
         dx = dmin;
         closest = last;
//         pm->SetNextPoint(x,y,z);
//         printf("%f %s\n", dmin, gGeoManager->GetPath());
      }
   }
   pm->Draw("SAME");
   gPad->Update();
   printf("Distance to boundary : %f\n", dmin);
   if (closest) printf("closest node : %s\n", closest->GetName());
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
/*
   if (!gGeoManager->IsLoopingVolumes()) {
      gGeoManager->SetLoopVolumes();   
      TGeoVolume *vcurrent;
      TList *volumes = gGeoManager->GetListOfVolumes();
      Int_t nvol = volumes->GetSize();
      Int_t ivol = 0;
      for (ivol=0; ivol<nvol; ivol++) {
         vcurrent = (TGeoVolume*)volumes->At(ivol);
         if (strcmp(vcurrent->GetName(), GetName())) continue;
         vcurrent->AddNode(vol, copy_no, mat, option);
      }
      volumes = gGeoManager->GetListOfGVolumes();
      nvol = volumes->GetSize();
      for (ivol=0; ivol<nvol; ivol++) {
         vcurrent = (TGeoVolume*)volumes->At(ivol);
         if (strcmp(vcurrent->GetName(), GetName())) continue;
         vcurrent->AddNode(vol, copy_no, mat, option);
      }
      gGeoManager->SetLoopVolumes(kFALSE);
      return;
   }      
*/
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
/*
   if (!gGeoManager->IsLoopingVolumes()) {
      gGeoManager->SetLoopVolumes();   
      TGeoVolume *vcurrent;
      TList *volumes = gGeoManager->GetListOfVolumes();
      Int_t nvol = volumes->GetSize();
      Int_t ivol = 0;
      for (ivol=0; ivol<nvol; ivol++) {
         vcurrent = (TGeoVolume*)volumes->At(ivol);
         if (strcmp(vcurrent->GetName(), GetName())) continue;
         vcurrent->AddNodeOffset(vol, copy_no, offset, option);
      }
      volumes = gGeoManager->GetListOfGVolumes();
      nvol = volumes->GetSize();
      for (ivol=0; ivol<nvol; ivol++) {
         vcurrent = (TGeoVolume*)volumes->At(ivol);
         if (strcmp(vcurrent->GetName(), GetName())) continue;
         vcurrent->AddNodeOffset(vol, copy_no, offset, option);
      }
      gGeoManager->SetLoopVolumes(kFALSE);
      return;
   }      
*/
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
//   vol->SetVisibility(kFALSE);
/*
   if (!gGeoManager->IsLoopingVolumes()) {
      gGeoManager->SetLoopVolumes();   
      TGeoVolume *vcurrent;
      TList *volumes = gGeoManager->GetListOfVolumes();
      Int_t nvol = volumes->GetSize();
      Int_t ivol = 0;
      for (ivol=0; ivol<nvol; ivol++) {
         vcurrent = (TGeoVolume*)volumes->At(ivol);
         if (strcmp(vcurrent->GetName(), GetName())) continue;
         vcurrent->AddNodeOverlap(vol, copy_no, mat, option);
      }
      volumes = gGeoManager->GetListOfGVolumes();
      nvol = volumes->GetSize();
      for (ivol=0; ivol<nvol; ivol++) {
         vcurrent = (TGeoVolume*)volumes->At(ivol);
         if (strcmp(vcurrent->GetName(), GetName())) continue;
         vcurrent->AddNodeOverlap(vol, copy_no, mat, option);
      }
      gGeoManager->SetLoopVolumes(kFALSE);
      return;
   }      
*/
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
   return 0; 
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoVolume::Divide(const char *divname, Int_t ndiv, Double_t start, Double_t step, Option_t *option)
{
// divide this volume in ndiv pieces from start, with given step
   return 0;
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoVolume::Divide(const char *divname, Double_t start, Double_t end, Double_t step, Option_t *option)
{
// divide this volume from start to end in pieces of length step
   return 0;
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoVolume::Divide(const char *divname, TObject *userdiv, Double_t *params, Option_t *)
{
// divide this volume according to userdiv
   return 0;
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoVolume::Divide(const char *divname, Int_t iaxis, Double_t step)
{
// divide all range of iaxis in range/step cells (G3 matching from ZEBRA
   Double_t start=0, end=0;
   Int_t ndiv;
   TString stype = fShape->ClassName();
   if (stype == "TGeoBBox") {
      switch (iaxis) {
         case 1:
            start=-((TGeoBBox*)fShape)->GetDX();
            end=-start;
            break;
         case 2:
            start=-((TGeoBBox*)fShape)->GetDY();
            end=-start;
            break;
         case 3:
            start=-((TGeoBBox*)fShape)->GetDZ();
            end=-start;
            break;
         default:
            Error("Divide", "Wrong division axis");
            return 0;   
      }      
   }
   if (stype == "TGeoTube") {
      switch (iaxis) {
         case 1:
            start = ((TGeoTube*)fShape)->GetRmin();
            end = ((TGeoTube*)fShape)->GetRmax();
            break;
         case 2:
            start = 0.;
            end = 360.;
            break;
         case 3:
            start = -((TGeoTube*)fShape)->GetDz();
            end = -start;
            break;
         default:
            Error("Divide", "Wrong division axis");
            return 0;   
      }      
   }
   if (stype == "TGeoTubeSeg") {
      switch (iaxis) {
         case 1:
            start = ((TGeoTube*)fShape)->GetRmin();
            end = ((TGeoTube*)fShape)->GetRmax();
            break;
         case 2:
            start = ((TGeoTubeSeg*)fShape)->GetPhi1();
            end = ((TGeoTubeSeg*)fShape)->GetPhi2();
            if (end<start) end+=360.;
            break;
         case 3:
            start = -((TGeoTube*)fShape)->GetDz();
            end = -start;
            break;
         default:
            Error("Divide", "Wrong division axis");
            return 0;   
      }      
   }
   Double_t range = end - start;
   ndiv = Int_t((range+0.01*step)/step);   
   if (ndiv<=0) {
      Error("Divide", "ndivisions=0, wrong type");
      return 0;
   }
   Double_t err = range-ndiv*step;
   if (err>(0.01*step)) {
      start+=0.5*err;
      end-=0.5*err;
   }   
//   printf("Divide : start=%f end=%f ndiv=%i step=%f err=%f\n", start, end,ndiv,step,err);
   return Divide(divname, iaxis, ndiv, start, step);
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoVolume::Divide(const char *divname, Int_t iaxis, Int_t ndiv, Double_t start, Double_t step)
{
// division a la G3
   TString stype = fShape->ClassName();
   TGeoShape *shape = 0;
   TGeoVolume *vol = 0;
   if (!ndiv && start) {
      printf("Error : Divide %s type %s into %s- ndivisions=0\n",GetName(), stype.Data(), divname);
//      vol = new TGeoVolume(divname, fShape, fMaterial);
//      AddNode(vol, 1, gGeoIdentity);
//      return vol;
      return this;
   }
/*
   if (!gGeoManager->IsLoopingVolumes()) {
      gGeoManager->SetLoopVolumes();   
      TGeoVolume *vcurrent;
      TGeoVolume *cell = 0;
      TList *volumes = gGeoManager->GetListOfVolumes();
      Int_t nvol = volumes->GetSize();
      Int_t ivol = 0;
      for (ivol=0; ivol<nvol; ivol++) {
         vcurrent = (TGeoVolume*)volumes->At(ivol);
         if (strcmp(vcurrent->GetName(), GetName())) continue;
         cell = vcurrent->Divide(divname, iaxis, ndiv, start, step);
      }
      volumes = gGeoManager->GetListOfGVolumes();
      nvol = volumes->GetSize();
      for (ivol=0; ivol<nvol; ivol++) {
         vcurrent = (TGeoVolume*)volumes->At(ivol);
         if (strcmp(vcurrent->GetName(), GetName())) continue;
         cell = vcurrent->Divide(divname, iaxis, ndiv, start, step);
      }
      gGeoManager->SetLoopVolumes(kFALSE);
      return cell;
   }      
*/
   if ((!ndiv) && (!start)) return Divide(divname, iaxis, step);
   if (!fNodes) fNodes = new TObjArray();
   if (fFinder) {
   // volume already divided. Divide again all its divisions.
      for (Int_t idiv=0; idiv<fFinder->GetNdiv(); idiv++) {
         vol = fFinder->GetNodeOffset(idiv)->GetVolume();
         vol->Divide(divname, iaxis, ndiv, start, step);
      }
      return this;
   }
   TString opt = "";
   Int_t id, is, ipl, idiv;
   
   if (stype == "TGeoBBox") {
//      printf("Dividing box %s on %i axis\n", GetName(), iaxis);
      Double_t dx = ((TGeoBBox*)fShape)->GetDX();
      Double_t dy = ((TGeoBBox*)fShape)->GetDY();
      Double_t dz = ((TGeoBBox*)fShape)->GetDZ();
      switch (iaxis) {
         case 1:
            if (step<=0) {step=2*dx/ndiv; start=-dx;}
            if (((start+dx)<-1E-4) || ((start+ndiv*step-dx)>1E-4)) {
               Warning("Divide", "box x division exceed shape range");
               printf("   volume was %s\n", GetName());
            }
            shape = new TGeoBBox(step/2., dy, dz); 
            fFinder = new TGeoPatternX(this, ndiv, start, start+ndiv*step);
            opt = "X";
            break;
         case 2:
            if (step<=0) {step=2*dy/ndiv; start=-dy;}
            if (((start+dy)<-1E-4) || ((start+ndiv*step-dy)>1E-4)) {
               Warning("Divide", "box y division exceed shape range");
               printf("   volume was %s\n", GetName());
            }
            shape = new TGeoBBox(dx, step/2., dz); 
            fFinder = new TGeoPatternY(this, ndiv, start, start+ndiv*step);
            opt = "Y";
            break;
         case 3:
            if (step<=0) {step=2*dz/ndiv; start=-dz;}
            if (((start+dz)<-1E-4) || ((start+ndiv*step-dz)>1E-4)) {
               Warning("Divide", "box z division exceed shape range");
               printf("   volume was %s\n", GetName());
            }
            shape = new TGeoBBox(dx, dy, step/2.); 
            fFinder = new TGeoPatternZ(this, ndiv, start, start+ndiv*step);
            opt = "Z";
            break;
         default:
            Error("Divide", "Wrong axis type for division");
            return this;
      }
      vol = new TGeoVolume(divname, shape, fMaterial);
      fFinder->SetBasicVolume(vol);
      fFinder->SetDivIndex(GetNdaughters());
      for (Int_t ic=0; ic<ndiv; ic++) {
         AddNodeOffset(vol, ic, start+step/2.+ic*step, opt.Data());
         ((TGeoNodeOffset*)fNodes->At(GetNdaughters()-1))->SetFinder(fFinder);    
      }
      return vol;
   }
   if (stype == "TGeoTube") {
//      printf("Dividing tube %s on %i axis\n", GetName(), iaxis);
      Double_t rmin = ((TGeoTube*)fShape)->GetRmin();
      Double_t rmax = ((TGeoTube*)fShape)->GetRmax();
      Double_t dz   = ((TGeoTube*)fShape)->GetDz();
      switch (iaxis) {
         case 1:  // R division
            if (step<=0) {step=(rmax-rmin)/ndiv; start=rmin;}
            if (((start-rmin)<-1E-4) || ((start-rmax)>1E-4) || 
                 ((start+ndiv*step-rmin)<-1E-4) ||((start+ndiv*step-rmax)>1E-4)) {
               Warning("Divide", "cyl R division exceed shape range");
               printf("   volume was %s\n", GetName());
            }
            fFinder = new TGeoPatternCylR(this, ndiv, start, start+ndiv*step);
            fFinder->SetDivIndex(GetNdaughters());
            for (id=0; id<ndiv; id++) {
               shape = new TGeoTube(start+id*step, start+(id+1)*step, dz);
//               char *name = new char[20];
//               sprintf(name, "%s_%i", divname, id+1);
               vol = new TGeoVolume(divname, shape, fMaterial);
               opt = "R";
               AddNodeOffset(vol, id, 0, opt.Data());
               ((TGeoNodeOffset*)fNodes->At(GetNdaughters()-1))->SetFinder(fFinder);
            }
            return this;
         case 2:  // phi division
            if (step<=0) step=360./ndiv;
            fFinder = new TGeoPatternCylPhi(this, ndiv, start, start+ndiv*step);
            fFinder->SetDivIndex(GetNdaughters());            
            shape = new TGeoTubeSeg(rmin, rmax, dz, -step/2, step/2);
            vol = new TGeoVolume(divname, shape, fMaterial);
            opt = "Phi";
            for (id=0; id<ndiv; id++) {
               AddNodeOffset(vol, id, start+id*step+step/2, opt.Data());
               ((TGeoNodeOffset*)fNodes->At(GetNdaughters()-1))->SetFinder(fFinder);
            }
            return vol;
         case 3: // Z division
            if (step<=0) {step=2*dz/ndiv; start=-dz;}
            if (((start+dz)<-1E-4) || ((start+ndiv*step-dz)>1E-4)) {
               Warning("Divide", "cyl z division exceed shape range");
               printf("   volume was %s\n", GetName());
            }
            fFinder = new TGeoPatternZ(this, ndiv, start, start+ndiv*step);
            fFinder->SetDivIndex(GetNdaughters());            
            shape = new TGeoTube(rmin, rmax, step/2);
            vol = new TGeoVolume(divname, shape, fMaterial);
            opt = "Z";
            for (id=0; id<ndiv; id++) {
               AddNodeOffset(vol, id, start+step/2+id*step, opt.Data());
               ((TGeoNodeOffset*)fNodes->At(GetNdaughters()-1))->SetFinder(fFinder);
            }
            return vol;
         default:
            Error("Divide", "Wrong axis type for division");
            return this;            
      }
   }
   if (stype == "TGeoTubeSeg") {
//      printf("Dividing tube segment %s on %i axis\n", GetName(), iaxis);
      Double_t rmin = ((TGeoTube*)fShape)->GetRmin();
      Double_t rmax = ((TGeoTube*)fShape)->GetRmax();
      Double_t dz   = ((TGeoTube*)fShape)->GetDz();
      Double_t phi1 = ((TGeoTubeSeg*)fShape)->GetPhi1();
      Double_t phi2 = ((TGeoTubeSeg*)fShape)->GetPhi2();
      Double_t dphi;
      switch (iaxis) {
         case 1:  // R division
            if (step<=0) {step=(rmax-rmin)/ndiv; start=rmin;}
            if (((start-rmin)<-1E-4) || ((start-rmax)>1E-4) || 
                 ((start+ndiv*step-rmin)<-1E-4) ||((start+ndiv*step-rmax)>1E-4)) {
               Warning("Divide", "cyl seg R division exceed shape range");
               printf("   volume was %s\n", GetName());
            }
            fFinder = new TGeoPatternCylR(this, ndiv, start, start+ndiv*step);
            fFinder->SetDivIndex(GetNdaughters());
            for (id=0; id<ndiv; id++) {
               shape = new TGeoTubeSeg(start+id*step, start+(id+1)*step, dz, phi1, phi2);
//               char *name = new char[20];
//               sprintf(name, "%s_%i", divname, id+1);
               vol = new TGeoVolume(divname, shape, fMaterial);
               opt = "R";
               AddNodeOffset(vol, id, 0, opt.Data());
               ((TGeoNodeOffset*)fNodes->At(GetNdaughters()-1))->SetFinder(fFinder);
            }
            return this;
         case 2:  // phi division
            dphi = phi2-phi1;
            if (dphi<0) dphi+=360.;
            if (step<=0) {step=dphi/ndiv; start=phi1;}
            fFinder = new TGeoPatternCylPhi(this, ndiv, start, start+ndiv*step);
            fFinder->SetDivIndex(GetNdaughters());            
            shape = new TGeoTubeSeg(rmin, rmax, dz, -step/2, step/2);
            vol = new TGeoVolume(divname, shape, fMaterial);
            opt = "Phi";
            for (id=0; id<ndiv; id++) {
               AddNodeOffset(vol, id, start+id*step+step/2, opt.Data());
               ((TGeoNodeOffset*)fNodes->At(GetNdaughters()-1))->SetFinder(fFinder);
            }
            return vol;
         case 3: // Z division
            if (step<=0) {step=2*dz/ndiv; start=-dz;}
            if (((start+dz)<-1E-4) || ((start+ndiv*step-dz)>1E-4)) {
               Warning("Divide", "cyl seg Z division exceed shape range"); 
               printf("   volume was %s\n", GetName());
            }
            fFinder = new TGeoPatternZ(this, ndiv, start, start+ndiv*step);
            fFinder->SetDivIndex(GetNdaughters());            
            shape = new TGeoTubeSeg(rmin, rmax, step/2, phi1, phi2);
            vol = new TGeoVolume(divname, shape, fMaterial);
            opt = "Z";
            for (id=0; id<ndiv; id++) {
               AddNodeOffset(vol, id, start+step/2+id*step, opt.Data());
               ((TGeoNodeOffset*)fNodes->At(GetNdaughters()-1))->SetFinder(fFinder);
            }
            return vol;
         default:
            Error("Divide", "Wrong axis type for division");
            return this;            
      }
   }
   if (stype == "TGeoCone") {
//      printf("Dividing cone %s on %i axis\n", GetName(), iaxis);
      Double_t rmin1 = ((TGeoCone*)fShape)->GetRmin1();
      Double_t rmax1 = ((TGeoCone*)fShape)->GetRmax1();
      Double_t rmin2 = ((TGeoCone*)fShape)->GetRmin2();
      Double_t rmax2 = ((TGeoCone*)fShape)->GetRmax2();
      Double_t dz   = ((TGeoCone*)fShape)->GetDz();
      switch (iaxis) {
         case 1:  // R division
            Error("Divide","division of a cone on R not implemented");
            return this;
         case 2:  // phi division
            if (step<=0) step=360./ndiv;
            fFinder = new TGeoPatternCylPhi(this, ndiv, start, start+ndiv*step);
            fFinder->SetDivIndex(GetNdaughters());            
            shape = new TGeoConeSeg(dz, rmin1, rmax1, rmin2, rmax2, -step/2, step/2);
            vol = new TGeoVolume(divname, shape, fMaterial);
            opt = "Phi";
            for (id=0; id<ndiv; id++) {
               AddNodeOffset(vol, id, start+id*step+step/2, opt.Data());
               ((TGeoNodeOffset*)fNodes->At(GetNdaughters()-1))->SetFinder(fFinder);
            }
            return vol;
         case 3: // Z division
            if (step<=0) {step=2*dz/ndiv; start=-dz;}
            if (((start+dz)<-1E-4) || ((start+ndiv*step-dz)>1E-4)) {
               Warning("Divide", "cone Z division exceed shape range");
               printf("   volume was %s\n", GetName());
            }
            fFinder = new TGeoPatternZ(this, ndiv, start, start+ndiv*step);
            fFinder->SetDivIndex(GetNdaughters());            
            for (id=0; id<ndiv; id++) {
               Double_t z1 = start+id*step;
               Double_t z2 = start+(id+1)*step;
               Double_t rmin1n = 0.5*(rmin1*(dz-z1)+rmin2*(dz+z1))/dz;
               Double_t rmax1n = 0.5*(rmax1*(dz-z1)+rmax2*(dz+z1))/dz;
               Double_t rmin2n = 0.5*(rmin1*(dz-z2)+rmin2*(dz+z2))/dz;
               Double_t rmax2n = 0.5*(rmax1*(dz-z2)+rmax2*(dz+z2))/dz;
               shape = new TGeoCone(rmin1n, rmax1n, rmin2n, rmax2n, step/2); 
               vol = new TGeoVolume(divname, shape, fMaterial);
               opt = "Z";
               AddNodeOffset(vol, id, start+id*step+step/2, opt.Data());
               ((TGeoNodeOffset*)fNodes->At(GetNdaughters()-1))->SetFinder(fFinder);
            }
            return this;
         default:
            Error("Divide", "Wrong axis type for division");
            return this;            
      }
   }
   if (stype == "TGeoConeSeg") {
//      printf("Dividing cone segment %s on %i axis\n", GetName(), iaxis);
      Double_t rmin1 = ((TGeoCone*)fShape)->GetRmin1();
      Double_t rmax1 = ((TGeoCone*)fShape)->GetRmax1();
      Double_t rmin2 = ((TGeoCone*)fShape)->GetRmin2();
      Double_t rmax2 = ((TGeoCone*)fShape)->GetRmax2();
      Double_t phi1 = ((TGeoConeSeg*)fShape)->GetPhi1();
      Double_t phi2 = ((TGeoConeSeg*)fShape)->GetPhi2();
      Double_t dz   = ((TGeoCone*)fShape)->GetDz();
      Double_t dphi;
      switch (iaxis) {
         case 1:  // R division
            Error("Divide","division of a cone segment on R not implemented");
            return this;
         case 2:  // phi division
            dphi = phi2-phi1;
            if (dphi<0) dphi+=360.;
            if (step<=0) {step=dphi/ndiv; start=phi1;}
            fFinder = new TGeoPatternCylPhi(this, ndiv, start, start+ndiv*step);
            fFinder->SetDivIndex(GetNdaughters());            
            shape = new TGeoConeSeg(dz, rmin1, rmax1, rmin2, rmax2, -step/2, step/2);
            vol = new TGeoVolume(divname, shape, fMaterial);
            opt = "Phi";
            for (id=0; id<ndiv; id++) {
               AddNodeOffset(vol, id, start+id*step+step/2, opt.Data());
               ((TGeoNodeOffset*)fNodes->At(GetNdaughters()-1))->SetFinder(fFinder);
            }
            return vol;
         case 3: // Z division
            if (step<=0) {step=2*dz/ndiv; start=-dz;}
            if (((start+dz)<-1E-4) || ((start+ndiv*step-dz)>1E-4)) {
               Warning("Divide", "cone seg Z division exceed shape range");
               printf("   volume was %s\n", GetName());
            }
            fFinder = new TGeoPatternZ(this, ndiv, start, start+ndiv*step);
            fFinder->SetDivIndex(GetNdaughters());            
            for (id=0; id<ndiv; id++) {
               Double_t z1 = start+id*step;
               Double_t z2 = start+(id+1)*step;
               Double_t rmin1n = 0.5*(rmin1*(dz-z1)+rmin2*(dz+z1))/dz;
               Double_t rmax1n = 0.5*(rmax1*(dz-z1)+rmax2*(dz+z1))/dz;
               Double_t rmin2n = 0.5*(rmin1*(dz-z2)+rmin2*(dz+z2))/dz;
               Double_t rmax2n = 0.5*(rmax1*(dz-z2)+rmax2*(dz+z2))/dz;
               shape = new TGeoConeSeg(step/2, rmin1n, rmax1n, rmin2n, rmax2n, phi1, phi2); 
               vol = new TGeoVolume(divname, shape, fMaterial);
               opt = "Z";
               AddNodeOffset(vol, id, start+id*step+step/2, opt.Data());
               ((TGeoNodeOffset*)fNodes->At(GetNdaughters()-1))->SetFinder(fFinder);
             }
             return this;
         default:
            Error("Divide", "Wrong axis type for division");
            return this;            
      }
   }
   if (stype == "TGeoPara") {
//      printf("Dividing para %s on %i axis\n", GetName(), iaxis);
      Double_t dx = ((TGeoPara*)fShape)->GetX();
      Double_t dy = ((TGeoPara*)fShape)->GetY();
      Double_t dz = ((TGeoPara*)fShape)->GetZ();
      Double_t alpha = ((TGeoPara*)fShape)->GetAlpha();
      Double_t theta = ((TGeoPara*)fShape)->GetTheta();
      Double_t phi = ((TGeoPara*)fShape)->GetPhi();
      switch (iaxis) {
         case 1:
            if (step<=0) {step=2*dx/ndiv; start=-dx;}
            if (((start+dx)<-1E-4) || ((start+ndiv*step-dx)>1E-4)) {
               Warning("Divide", "para X division exceed shape range");
               printf("   volume was %s\n", GetName());
               printf("start=%f end=%f dx=%f\n", start, start+ndiv*step, dx);
            }
            shape = new TGeoPara(step/2, dy, dz, alpha, theta, phi);
            fFinder = new TGeoPatternParaX(this, ndiv, start, start+ndiv*step);
            opt = "X";
            break;
         case 2:
            if (step<=0) {step=2*dy/ndiv; start=-dy;}
            if (((start+dy)<-1E-4) || ((start+ndiv*step-dy)>1E-4)) {
               Warning("Divide", "para Y division exceed shape range");
               printf("   volume was %s\n", GetName());
            }
            shape = new TGeoPara(dx, step/2, dz, alpha, theta, phi);
            fFinder = new TGeoPatternParaY(this, ndiv, start, start+ndiv*step);
            opt = "Y";
            break;
         case 3:
            if (step<=0) {step=2*dz/ndiv; start=-dz;}
            if (((start+dz)<-1E-4) || ((start+ndiv*step-dz)>1E-4)) {
               Warning("Divide", "para Z division exceed shape range");
               printf("   volume was %s\n", GetName());
            }
            shape = new TGeoPara(dx, dy, step/2, alpha, theta, phi);
            fFinder = new TGeoPatternParaZ(this, ndiv, start, start+ndiv*step);
            opt = "Z";
            break;
         default:
            Error("Divide", "Wrong axis type for division");
            return this;            
      }
      vol = new TGeoVolume(divname, shape, fMaterial);
      fFinder->SetBasicVolume(vol);
      fFinder->SetDivIndex(GetNdaughters());
      for (Int_t ic=0; ic<ndiv; ic++) {
         AddNodeOffset(vol, ic, start+step/2.+ic*step, opt.Data());
         ((TGeoNodeOffset*)fNodes->At(GetNdaughters()-1))->SetFinder(fFinder);    
      }
      return vol;
   }
   if (stype == "TGeoTrd1") {
//      printf("Dividing trd1 %s on %i axis\n", GetName(), iaxis);
      Double_t dx1, dx2, dy, dz, zmin, zmax, dx1n, dx2n;
      dy = ((TGeoTrd1*)fShape)->GetDy();
      dz = ((TGeoTrd1*)fShape)->GetDz();
      dx1 = ((TGeoTrd1*)fShape)->GetDx1();
      dx2 = ((TGeoTrd1*)fShape)->GetDx2();
      switch (iaxis) {
         case 1:
            Warning("Divide", "dividing a Trd1 on X not implemented");
            break;
         case 2:
            if (step<=0) {step=2*dy/ndiv; start=-dy;}
            if (((start+dy)<-1E-4) || ((start+ndiv*step-dy)>1E-4)) {
               Warning("Divide", "trd1 Y division exceed shape range");
               printf("   volume was %s\n", GetName());
               printf("start=%f end=%f, dy=%f\n", start, start+ndiv*step, dy);
            }
            fFinder = new TGeoPatternY(this, ndiv, start, start+ndiv*step);
            fFinder->SetDivIndex(GetNdaughters());            
            shape = new TGeoTrd1(dx1, dx2, step/2, dz);
            vol = new TGeoVolume(divname, shape, fMaterial); 
            opt = "Y";
            for (id=0; id<ndiv; id++) {
               AddNodeOffset(vol, id, start+step/2+id*step, opt.Data());
               ((TGeoNodeOffset*)fNodes->At(GetNdaughters()-1))->SetFinder(fFinder);
            }
            return vol;
         case 3:
            if (step<=0) {step=2*dz/ndiv; start=-dz;}
            if (((start+dz)<-1E-4) || ((start+ndiv*step-dz)>1E-4)) {
               Warning("Divide", "trd1 Z division exceed shape range");
               printf("   volume was %s\n", GetName());
            }
            fFinder = new TGeoPatternZ(this, ndiv, start, start+ndiv*step);
            fFinder->SetDivIndex(GetNdaughters());            
            for (id=0; id<ndiv; id++) {
               zmin = start+id*step;
               zmax = start+(id+1)*step;
               dx1n = 0.5*(dx1*(dz-zmin)+dx2*(dz+zmin))/dz;
               dx2n = 0.5*(dx1*(dz-zmax)+dx2*(dz+zmax))/dz;
               shape = new TGeoTrd1(dx1n, dx2n, dy, step/2.);
               vol = new TGeoVolume(divname, shape, fMaterial); 
               opt = "Z";             
               AddNodeOffset(vol, id, start+step/2+id*step, opt.Data());
               ((TGeoNodeOffset*)fNodes->At(GetNdaughters()-1))->SetFinder(fFinder);
            }
            return this;
         default:
            Error("Divide", "Wrong axis type for division");
            return this;
      }
   }
   if (stype == "TGeoTrd2") {
//      printf("Dividing trd2 %s on %i axis\n", GetName(), iaxis);
      Double_t dx1, dx2, dy1, dy2, dz, zmin, zmax, dx1n, dx2n, dy1n, dy2n;
      dz = ((TGeoTrd2*)fShape)->GetDz();
      dx1 = ((TGeoTrd2*)fShape)->GetDx1();
      dx2 = ((TGeoTrd2*)fShape)->GetDx2();
      dy1 = ((TGeoTrd2*)fShape)->GetDy1();
      dy2 = ((TGeoTrd2*)fShape)->GetDy2();
      switch (iaxis) {
         case 1:
            Warning("Divide", "dividing a Trd2 on X not implemented");
            break;
         case 2:
            Warning("Divide", "dividing a Trd2 on Y not implemented");
            break;
         case 3:
            if (step<=0) {step=2*dz/ndiv; start=-dz;}
            if (((start+dz)<-1E-4) || ((start+ndiv*step-dz)>1E-4)) {
               Warning("Divide", "trd2 Z division exceed shape range");
               printf("   volume was %s\n", GetName());
            }
            fFinder = new TGeoPatternZ(this, ndiv, start, start+ndiv*step);
            fFinder->SetDivIndex(GetNdaughters());            
            for (id=0; id<ndiv; id++) {
               zmin = start+id*step;
               zmax = start+(id+1)*step;
               dx1n = 0.5*(dx1*(dz-zmin)+dx2*(dz+zmin))/dz;
               dx2n = 0.5*(dx1*(dz-zmax)+dx2*(dz+zmax))/dz;
               dy1n = 0.5*(dy1*(dz-zmin)+dy2*(dz+zmin))/dz;
               dy2n = 0.5*(dy1*(dz-zmax)+dy2*(dz+zmax))/dz;
               shape = new TGeoTrd2(dx1n, dx2n, dy1n, dy2n, step/2.);
               vol = new TGeoVolume(divname, shape, fMaterial); 
               opt = "Z";             
               AddNodeOffset(vol, id, start+step/2+id*step, opt.Data());
               ((TGeoNodeOffset*)fNodes->At(GetNdaughters()-1))->SetFinder(fFinder);
            }
            return this;
         default:
            Error("Divide", "Wrong axis type for division");
            return this;
      }
   }
   if (stype == "TGeoPcon") {
//      printf("Dividing pcon %s on %i axis\n", GetName(), iaxis);
      Int_t nz = ((TGeoPcon*)fShape)->GetNz();
      Double_t phi1 = ((TGeoPcon*)fShape)->GetPhi1();
      Double_t dphi = ((TGeoPcon*)fShape)->GetDphi();
      Double_t *rmin = ((TGeoPcon*)fShape)->GetRmin();
      Double_t *rmax = ((TGeoPcon*)fShape)->GetRmax();
      Double_t *zpl = ((TGeoPcon*)fShape)->GetZ();
      Double_t zmin = start;
      Double_t zmax = start+ndiv*step;            
      Int_t isect = -1;
      switch (iaxis) {
         case 1:
            Error("Divide", "cannot divide a pcon on radius");
            break;
         case 2:  // phi division
            fFinder = new TGeoPatternCylPhi(this, ndiv, start, start+ndiv*step);
            fFinder->SetDivIndex(GetNdaughters());            
            shape = new TGeoPcon(-step/2, step, nz);
            for (is=0; is<nz; is++)
               ((TGeoPcon*)shape)->DefineSection(is, zpl[is], rmin[is], rmax[is]); 
            vol = new TGeoVolume(divname, shape, fMaterial);
            opt = "Phi";
            for (id=0; id<ndiv; id++) {
               AddNodeOffset(vol, id, start+id*step+step/2, opt.Data());
               ((TGeoNodeOffset*)fNodes->At(GetNdaughters()-1))->SetFinder(fFinder);
            }
            return vol;
         case 3: // Z division
            // find start plane
            for (ipl=0; ipl<nz-1; ipl++) {
               if (start<zpl[ipl]) continue;
               else {if ((start+ndiv*step)>zpl[ipl+1]) continue;}
               isect = ipl;
               break;
            }
            if (isect<0) {
               Error("Divide", "cannot divide pcon on Z if divided region is not between 2 planes");
               break;
            }
            fFinder = new TGeoPatternZ(this, ndiv, start, start+ndiv*step);
            fFinder->SetDivIndex(GetNdaughters());
            opt = "Z";
            for (id=0; id<ndiv; id++) {
               Double_t z1 = start+id*step;
               Double_t z2 = start+(id+1)*step;
               Double_t rmin1 = (rmin[isect]*(zmax-z1)-rmin[isect+1]*(zmin-z1))/(zmax-zmin);
               Double_t rmax1 = (rmax[isect]*(zmax-z1)-rmax[isect+1]*(zmin-z1))/(zmax-zmin);
               Double_t rmin2 = (rmin[isect]*(zmax-z2)-rmin[isect+1]*(zmin-z2))/(zmax-zmin);
               Double_t rmax2 = (rmax[isect]*(zmax-z2)-rmax[isect+1]*(zmin-z2))/(zmax-zmin);
               shape = new TGeoConeSeg(step/2, rmin1, rmax1, rmin2, rmax2, phi1, phi1+dphi); 
               vol = new TGeoVolume(divname, shape, fMaterial);
               AddNodeOffset(vol, id, start+id*step+step/2, opt.Data());
               ((TGeoNodeOffset*)fNodes->At(GetNdaughters()-1))->SetFinder(fFinder);
             }
             return this;
         default:
            Error("Divide", "Wrong axis type for division");
            return this;            
      }
      return this;
   }
   if (stype == "TGeoPgon") {
//      printf("Dividing pgon %s on %i axis\n", GetName(), iaxis);
      Int_t nz = ((TGeoPcon*)fShape)->GetNz();
      Double_t *rmin = ((TGeoPcon*)fShape)->GetRmin();
      Double_t *rmax = ((TGeoPcon*)fShape)->GetRmax();
      Double_t *zpl = ((TGeoPcon*)fShape)->GetZ();
      Int_t     nedges = ((TGeoPgon*)fShape)->GetNedges();
      Double_t phi1 = ((TGeoPcon*)fShape)->GetPhi1();
      Double_t dphi = ((TGeoPcon*)fShape)->GetDphi();
      Double_t zmin = start;
      Double_t zmax = start+ndiv*step;            
      Int_t isect = -1;
      switch (iaxis) {
         case 1:
            Error("Divide", "makes no sense dividing a pgon on radius");
            break;
         case 2:  // phi division
            if (nedges%ndiv) {
               Error("Divide", "cannot divide pgon like this");
               break;
            }
            nedges = nedges/ndiv;
            fFinder = new TGeoPatternCylPhi(this, ndiv, start, start+ndiv*step);
            fFinder->SetDivIndex(GetNdaughters());            
            shape = new TGeoPgon(-step/2, step, nedges, nz);
            for (is=0; is<nz; is++)
               ((TGeoPgon*)shape)->DefineSection(is, zpl[is], rmin[is], rmax[is]); 
            vol = new TGeoVolume(divname, shape, fMaterial);
            opt = "Phi";
            for (id=0; id<ndiv; id++) {
               AddNodeOffset(vol, id, start+id*step+step/2, opt.Data());
               ((TGeoNodeOffset*)fNodes->At(GetNdaughters()-1))->SetFinder(fFinder);
            }
            return vol;
         case 3: // Z division
            // find start plane
            for (ipl=0; ipl<nz-1; ipl++) {
               if (start<zpl[ipl]) continue;
               else {if ((start+ndiv*step)>zpl[ipl+1]) continue;}
               isect = ipl;
               break;
            }
            if (isect<0) {
               Error("Divide", "cannot divide pcon on Z if divided region is not between 2 planes");
               break;
            }
            fFinder = new TGeoPatternZ(this, ndiv, start, start+ndiv*step);
            fFinder->SetDivIndex(GetNdaughters());
            opt = "Z";
            for (id=0; id<ndiv; id++) {
               Double_t z1 = start+id*step;
               Double_t z2 = start+(id+1)*step;
               Double_t rmin1 = (rmin[isect]*(zmax-z1)-rmin[isect+1]*(zmin-z1))/(zmax-zmin);
               Double_t rmax1 = (rmax[isect]*(zmax-z1)-rmax[isect+1]*(zmin-z1))/(zmax-zmin);
               Double_t rmin2 = (rmin[isect]*(zmax-z2)-rmin[isect+1]*(zmin-z2))/(zmax-zmin);
               Double_t rmax2 = (rmax[isect]*(zmax-z2)-rmax[isect+1]*(zmin-z2))/(zmax-zmin);
               shape = new TGeoPgon(phi1, dphi, nedges, 2); 
               ((TGeoPgon*)shape)->DefineSection(0, -step/2, rmin1, rmax1); 
               ((TGeoPgon*)shape)->DefineSection(1,  step/2, rmin2, rmax2); 
               vol = new TGeoVolume(divname, shape, fMaterial);
               AddNodeOffset(vol, id, start+id*step+step/2, opt.Data());
               ((TGeoNodeOffset*)fNodes->At(GetNdaughters()-1))->SetFinder(fFinder);
             }
             return this;
         default:
            Error("Divide", "Wrong axis type for division");
            return this;            
      }
   }
   if (fShape->InheritsFrom("TGeoTrap")) {
      if (iaxis!=3) {
         Error("Divide", "cannot divide Arb8 on other axis than Z");
         return this;
      }
//      printf("Dividing %s Arb8 on Z\n", GetName());
      Double_t points_lo[8];
      Double_t points_hi[8];
      fFinder = new TGeoPatternTrapZ(this, ndiv, start, start+ndiv*step);
      fFinder->SetDivIndex(GetNdaughters());
      opt = "Z";
      Double_t theta = ((TGeoTrap*)fShape)->GetTheta();
      Double_t phi =   ((TGeoTrap*)fShape)->GetPhi();
      Double_t txz = ((TGeoPatternTrapZ*)fFinder)->GetTxz();
      Double_t tyz = ((TGeoPatternTrapZ*)fFinder)->GetTyz();
      Double_t zmin, zmax, ox,oy,oz;
      for (idiv=0; idiv<ndiv; idiv++) {
         zmin = start+idiv*step;
         zmax = start+(idiv+1)*step;
         oz = start+idiv*step+step/2;
         ox = oz*txz;
         oy = oz*tyz;
         ((TGeoArb8*)fShape)->SetPlaneVertices(zmin, &points_lo[0]);
         ((TGeoArb8*)fShape)->SetPlaneVertices(zmax, &points_hi[0]);
         shape = new TGeoTrap(step/2, theta, phi);
         for (Int_t vert1=0; vert1<4; vert1++)
            ((TGeoArb8*)shape)->SetVertex(vert1, points_lo[2*vert1]-ox, points_lo[2*vert1+1]-oy);
         for (Int_t vert2=0; vert2<4; vert2++)
            ((TGeoArb8*)shape)->SetVertex(vert2+4, points_hi[2*vert2]-ox, points_hi[2*vert2+1]-oy);
         vol = new TGeoVolume(divname, shape, fMaterial);
         AddNodeOffset(vol, idiv, oz, opt.Data());
         ((TGeoNodeOffset*)fNodes->At(GetNdaughters()-1))->SetFinder(fFinder);
      }
      return this;
   }  
   Error("Divide", "this type of division not implemented");
   printf("Volume was : %s shape %s iaxis=%i\n", GetName(), stype.Data(), iaxis);
   return this;
}
//-----------------------------------------------------------------------------
Int_t TGeoVolume::DistancetoPrimitive(Int_t px, Int_t py)
{
// compute the closest distance of approach from point px,py to this 
   const Int_t big = 9999;
   const Int_t inaxis = 7;
   const Int_t maxdist = 5;
   
   Int_t puxmin = gPad->XtoAbsPixel(gPad->GetUxmin());
   Int_t puymin = gPad->YtoAbsPixel(gPad->GetUymin());
   Int_t puxmax = gPad->XtoAbsPixel(gPad->GetUxmax());
   Int_t puymax = gPad->YtoAbsPixel(gPad->GetUymax());
   // return if point not in user area
   if (px < puxmin - inaxis) return big;
   if (py > puymin + inaxis) return big;
   if (px > puxmax + inaxis) return big;
   if (py < puymax - inaxis) return big;
   
   TView *view = gPad->GetView();
   if (!view) return big;
   Int_t dist = big;
   Int_t id;
   
   if (gGeoManager->GetTopVolume() == this) gGeoManager->CdTop();
   Int_t vis_opt = gGeoManager->GetVisOption();
   Int_t level = gGeoManager->GetLevel();
   Int_t vis_level=gGeoManager->GetVisLevel();
   Bool_t vis=(IsVisible() && gGeoManager->GetLevel())?kTRUE:kFALSE;
   TGeoNode *node = 0;
   Int_t nd = GetNdaughters();
   Bool_t last = kFALSE;
   switch (vis_opt) {
      case TGeoManager::kGeoVisDefault:
         if (vis && (level<=vis_level)) { 
            dist = fShape->DistancetoPrimitive(px,py);
            if (dist<maxdist) {
               gPad->SetSelected(this);
               return 0;
            }
         }
         // check daughters
         if (level<vis_level) {
            if ((!nd) || (!IsVisDaughters())) return dist;
            for (id=0; id<nd; id++) {
               node = GetNode(id);
               gGeoManager->CdDown(id);
               dist = node->GetVolume()->DistancetoPrimitive(px, py);
               if (dist==0) return 0;
               gGeoManager->CdUp();
            }
         }
         break;
      case TGeoManager::kGeoVisLeaves:
         last = ((nd==0) || (level==vis_level) || (!IsVisDaughters()))?kTRUE:kFALSE;
         if (vis && last) {
            dist = fShape->DistancetoPrimitive(px, py);
            if (dist<maxdist) {
               gPad->SetSelected(this);
               return 0;
            }
         }
         if (last) return dist;
         for (id=0; id<nd; id++) {
            node = GetNode(id);
            gGeoManager->CdDown(id);
            dist = node->GetVolume()->DistancetoPrimitive(px,py);
            if (dist==0) return 0;
            gGeoManager->CdUp();
         }
         break;
      case TGeoManager::kGeoVisOnly:
         dist = fShape->DistancetoPrimitive(px, py);
         if (dist<maxdist) {
            gPad->SetSelected(this);
            return 0;
         }
         break;
      case TGeoManager::kGeoVisBranch:
         gGeoManager->cd(gGeoManager->GetDrawPath());
         while (gGeoManager->GetLevel()) {
            if (gGeoManager->GetCurrentVolume()->IsVisible()) {
               dist = gGeoManager->GetCurrentVolume()->GetShape()->DistancetoPrimitive(px, py);
               if (dist<maxdist) {
                  gPad->SetSelected(gGeoManager->GetCurrentVolume());
                  return 0;
               }
            }   
            gGeoManager->CdUp();
         }
         gPad->SetSelected(view);      
         return big;   
      default:
         return big;
   }       
   if ((dist>maxdist) && !gGeoManager->GetLevel()) gPad->SetSelected(view);
   return dist;
}
//-----------------------------------------------------------------------------
void TGeoVolume::Draw(Option_t *option)
{
// draw top volume according to option
   TGeoVolume *old_vol = gGeoManager->GetTopVolume();
   if (old_vol!=this) gGeoManager->SetTopVolume(this);
   else old_vol=0;
   TVirtualGeoPainter *painter = gGeoManager->GetMakeDefPainter();
   if (!painter) return;
   painter->Draw(option);   
}
//-----------------------------------------------------------------------------
void TGeoVolume::DrawOnly(Option_t *option)
{
// draw only this volume
   TVirtualGeoPainter *painter = gGeoManager->GetMakeDefPainter();
   if (!painter) return;
   painter->DrawOnly(option);   
}
//-----------------------------------------------------------------------------
void TGeoVolume::DrawPoints(Int_t npoints, Option_t *option)
{
   gGeoManager->DrawPoints(this, npoints, option);
}
//-----------------------------------------------------------------------------
void TGeoVolume::Paint(Option_t *option)
{
// paint volume
   TVirtualGeoPainter *painter = gGeoManager->GetMakeDefPainter();
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
void TGeoVolume::RandomRays(Int_t nrays)
{
// draw top volume according to option
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
// description
//   if (gPad->GetView()) {
//      gPad->GetView()->ExecuteRotateView(event, px, py);
//   }
   gPad->SetCursor(kHand);
   switch (event) {
   case kMouseEnter:
      SetLineWidth(3);
      gPad->Modified();
      gPad->Update();
      break;
   
   case kMouseLeave:
      SetLineWidth(1);
      gPad->Modified();
      gPad->Update();
      break;
   
   case kButton1Double:
      gPad->SetCursor(kWatch);
      Draw();
      break;
   }
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
   const char *snull = "";
   if (!gPad) return (char*)snull;
   static char info[128];
   sprintf(info,"%s, shape=%s", gGeoManager->GetPath(), fShape->ClassName());
   return info;
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
//   return size of this 3d object
   if (gGeoManager->GetTopVolume() == this) gGeoManager->CdTop();
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
            fShape->Sizeof3D();
            // draw daughters
         if (level<vis_level) {
            if ((!nd) || (!IsVisDaughters())) return;
            for (id=0; id<nd; id++) {
               node = GetNode(id);
               gGeoManager->CdDown(id);
               node->GetVolume()->Sizeof3D();
               gGeoManager->CdUp();
            }
         }
         break;
      case TGeoManager::kGeoVisLeaves:
         last = ((nd==0) || (level==vis_level) || (!IsVisDaughters()))?kTRUE:kFALSE;
         if (vis && last)
            fShape->Sizeof3D();
         if (last) return;
         for (id=0; id<nd; id++) {
            node = GetNode(id);
            gGeoManager->CdDown(id);
            node->GetVolume()->Sizeof3D();
            gGeoManager->CdUp();
         }
         break;
      case TGeoManager::kGeoVisOnly:
         fShape->Sizeof3D();
         break;
      case TGeoManager::kGeoVisBranch:
         gGeoManager->cd(gGeoManager->GetDrawPath());
         while (gGeoManager->GetLevel()) {
            if (gGeoManager->GetCurrentVolume()->IsVisible()) 
               gGeoManager->GetCurrentVolume()->GetShape()->Sizeof3D();
            gGeoManager->CdUp();   
         }   
         break;
      default:
         return;
   }       
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
   Double_t dx = ((TGeoBBox*)fShape)->GetDX();
   Double_t dy = ((TGeoBBox*)fShape)->GetDY();
   Double_t dz = ((TGeoBBox*)fShape)->GetDZ();
   if ((dx<0) || (dy<0) || (dz<0)) return kFALSE;
   return kTRUE;
}
//-----------------------------------------------------------------------------
void TGeoVolume::VisibleDaughters(Bool_t vis)
{
// set visibility for daughters
   SetVisDaughters(vis);
   if (!gPad) return;
   if (!gPad->GetView()) return;
   gPad->Modified();
   gPad->Update();
//   if (!GetNdaughters()) return;
//   TIter next(fNodes);
//   TGeoNode *node;
//   while ((node=(TGeoNode*)next())) node->VisibleDaughters(vis);
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
