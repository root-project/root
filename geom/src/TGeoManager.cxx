// @(#)root/geom:$Name:  $:$Id: TGeoManager.cxx,v 1.6 2002/07/15 15:32:25 brun Exp $
// Author: Andrei Gheata   25/10/01

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////////
// The geometry package
// --------------------
//
//   The new ROOT geometry package is a tool designed for building, browsing,
// tracking and visualizing a detector geometry. The code is independent from
// other external MC for simulation, therefore it does not contain any
// constraints related to physics. However, the package defines a number of
// hooks for physics, such as materials or magnetic field, in order to allow
// interfaces to tracking MC's. The final purpose is to be able to use the same
// geometry for several purposes, such as tracking, reconstruction or visualization,
// taking advantage of the ROOT features related to bookkeeping, I/O,
// histograming, browsing and GUI's.
//
//   The geometrical modeler is the most important component of the package and
// it provides answers to the basic questions like "where am I" or "how far
// from the next boundary", but also to more complex ones like "how far from
// the closest surface" or "which is the next crossing along a parametric curve".
// It can provide the current material and allows an user-defined stack of the
// last classified points in order to fasten tracking.
//
//   The architecture of the modeler is a combination between a GEANT-like
// containment scheme - for speeding up tracking - and a normal CSG binary tree
// at the level of shapes - for allowing building of more complex shapes from a
// set of primitives via bolean operations. The base classes used for building
// the GEANT-like tree are TGeoVolume and TGeoNode. These allow replicating a
// given volume several times in the geometry. A volume contains no information
// of his position in the geometry nor of his container, but only about its
// daughter nodes. On the other hand, nodes are unique non-overlapping volumes
// that are holding a transformation in the local reference system and know
// the volume in which they are contained. A geometry tree made of volumes and
// nodes is browsed starting with the top level in order to answer a geometrical
// query.
//
//   A volume can be divided according default or user-defined patterns, creating
// automatically the list of division nodes inside. The elementary volumes
// created during the dividing process follow the same scheme as usual volumes,
// therefore it is possible to position further geometrical structures inside or
// to divide them further more - see TGeoVolume::Divide().
//
//   The primitive shapes supported by the package are basically the GEANT3
// shapes (see class TGeoShape), extruded shapes and arbitrary wedges with eight
// vertices on two paralel planes. In order to build a TGeoCompositeShape, one has
// to define first the primitive components. The object that handle boolean
// operations among components is called TGeoBoolFinder and it has to be
// constructed providing a string boolean expression between the components names.
//
//  Example for building a simple geometry :
//______________________________________________________________________________
//   TGeometry *geom = new TGeometry("Geometry", "Simple geometry");
//   //--------------- build the materials ----------------
//   TGeoMaterial *mat, *mix;
//   // materials can be retreived by name
//   mat = new TGeoMaterial("mat1","HYDROGEN",1.01,1,0.7080000E-01);
//   mat = new TGeoMaterial("mat2","DEUTERIUM",2.01,1,0.162);
//   mat = new TGeoMaterial("mat3","ALUMINIUM",26.98,13,2.7);
//   mat = new TGeoMaterial("mat4","LEAD",207.19,82,11.35);
//   mix = new TGeoMixture("mix1","SCINT",2);
//      mix->DefineElement(0,12.01,6,0.922427);
//      mix->DefineElement(1,1.01,1,0.7757296E-01);
//   // ---materials can be retreived also by pointer
//   TGeoMaterial *mat_ptr = new TGeoMaterial("mat5","VACUUM",0,0,0);
//   //--------------- build the rotations ----------------
//   TGeoRotation *rot1 = new TGeoRotation("rot1",90,0,90,90,0,0);
//   TGeoRotation *rot2 = new TGeoRotation("rot2",90,180,90,90,180,0);
//   TGeoRotation *rot3 = new TGeoRotation("rot3",90,180,90,270,0,0);
//   TGeoRotation *rot4 = new TGeoRotation("rot4",90,90,90,180,0,0);
//   TGeoRotation *rot5 = new TGeoRotation("rot5",90,198,90,90,0,0);
//   //--------------- build the shapes ----------------
//   TGeoBBox *box = new TGeoBBox("HALL_out", 700, 700, 1500);
//   TGeoTube *tube = new TGeoTube("HALL_in", 0, 20, 2000);
//   TGeoTrd1 *trd = new TGeoTrd1("abso", 100, 100, 50 ,300);
//   //--- a composite shape
//   TGeoCompositeShape *hall = new TGeoCompositeShape("HALL");
//   hall->AddShape(box, gIdentity, "A");
//   hall->AddShape(tube,gIdentity, "B");
//   hall->MakeCombination("A\B");
//   //--------------- build some volumes ----------------
//   TGeoVolume *vol, *abso;
//   vol = new TGeoVolume("Hall", hall, mat_ptr);
//   abso = new TGeoVolume("absorber", box, "mat4");
//   //--- volumes having a primitive shape can be built in one step
//   TGeoVolume *pipe = gGeoManager->MakeTube("beam_pipe", "mat4", 17, 20, 1500);
//   TGeoVolume *target = gGeoManager->MakeBox("target_box", "mat3", 10, 10, 0.5);
//   TGeoVolume *detector = gGeoManager->MakeBox("detector", "mat1", 15, 15, 50.);
//   //---detector has 50 layers a layer has 15x15 cells
//   TGeoVolume *layer = gGeoManager->MakeBox("layer", "mix1", 15, 15, 1.);
//   //--------------- build the nodes ----------------
//   TGeoVolume *HALL = gGeoManager->GetVolume("Hall")
//   gGeoManager->SetTopVolume(HALL);
//   HALL->AddNode(pipe, gGeoIdentity);
//   HALL->AddNode(target, new TGeoTranslation(0,0,100));
//   HALL->AddNode(abso, new TGeoTranslation(0,0,-600));
//   detector->Divide(50, "Z");
//      vol = target->GetBasicCell();
//      vol->Divide(15, "X");
//      vol = vol->GetBasicCell();
//      vol->Divide(15, "Y");
//   HALL->AddNode(detector, new TGeoCombiTrans(0,0, -400, rot1));
//   HALL->AddNode(detector, new TGeoCombiTrans(0,0, 400, rot1));
//______________________________________________________________________________
//
//
// TGeoManager - the manager class for the geometry package.
// ---------------------------------------------------------
//   Contains the lists (arrays) of all user defined or default objects used
// in building the geometry : materials, geometrical transformations, shapes,
// volumes. The user can navigate downwords through the tree of nodes
// starting from any point. In order to navigate upwords, the current branch
// is stored in a TObjArray .
//   Materials, shapes and volumes can be retreived by name :
//      TGeoManager::GetMaterial(const char *name);
//      TGeoManager::GetShape(const char *name);
//      TGeoManager::GetVolume(const char *name);
//   The top level volume must be specified with TGeoManager::SetTopVolume();
// All objects like transformations, materials, shapes or volumes can be created
// with new operator and they will register themselves to TGeoManager class.
// The user does not have to take care of their deletion.
//
//
//  Drawing the geometry
// ----------------------
//   Any logical volume can be drawn via TGeoVolume::Draw() member function.
// This can be direcly accessed from the context menu of the volume object
// directly from the browser.
//   There are several drawing options that can be set with
// TGeoManager::SetVisOption(Int_t opt) method :
// opt=1 - only the content of the volume is drawn, N levels down (default N=3).
//    This is the default behavior. The number of levels to be drawn can be changed
//    via TGeoManager::SetVisLevel(Int_t level) method.
// opt=2 - the final leaves (e.g. daughters with no containment) of the branch
//    starting from volume are drawn. WARNING : This mode is memory consuming
//    depending of the size of geometry, so drawing from top level within this mode
//    should be handled with care. In future there will be a limitation on the
//    maximum number of nodes to be visualized.
// opt=3 - only a given path is visualized. This is automatically set by
//    TGeoVolume::DrawPath(const char *path) method
//
//    The current view can be exploded in cartesian, cylindrical or spherical
// coordinates :
//   TGeoManager::SetExplodedView(Option_t *option). Options may be :
// - "NONE" - default
// - "XYZ"  - cartesian coordinates. The bomb factor on each axis can be set with
//   TGeoManager::SetBombX(Double_t bomb) and corresponding Y and Z.
// - "CYL"  - bomb in cylindrical coordinates. Only the bomb factors on Z and R
//   are considered
// - "SPH"  - bomb in radial spherical coordinate : TGeoManager::SetBombR()+5
//
//Begin_Html
/*
<img src="gif/t_mgr.jpg">
*/
//End_Html

#include "Riostream.h"

#include "TROOT.h"
#include "TSystem.h"
#include "TStyle.h"
#include "TVirtualPad.h"
#include "TBrowser.h"
#include "TGeoMaterial.h"
#include "TGeoMatrix.h"
#include "TGeoManager.h"
#include "TGeoPara.h"
#include "TGeoTube.h"
#include "TGeoEltu.h"
#include "TGeoCone.h"
#include "TGeoSphere.h"
#include "TGeoArb8.h"
#include "TGeoPgon.h"
#include "TGeoTrd1.h"
#include "TGeoTrd2.h"
#include "TGeoCompositeShape.h"
#include "TGeoFinder.h"
#include "TVirtualGeoPainter.h"

#include "TGeoManager.h"

// statics and globals

TGeoManager *gGeoManager = 0;

//Int_t TGeoManager::kGeoVisLevel = 3;
const char *kGeoOutsidePath = " ";

ClassImp(TGeoManager)

//-----------------------------------------------------------------------------
TGeoManager::TGeoManager()
{
// dummy constructor
   fBits = 0;
   fMaterials = 0;
   fMatrices = 0;
   fPoint = 0;
   fDirection = 0;
   fNormalChecked = 0;
   fCldirChecked = 0;
   fNormal = 0;
   fCldir = 0;
   fGlobalMatrices = 0;
   fNodes = 0;
   fNNodes = 0;
   fVolumes = 0;
   fShapes = 0;
   fTopVolume = 0;
   fTopNode = 0;
   fCurrentVolume = 0;
   fMasterVolume = 0;
   fCurrentNode = 0;
   fLastNode = 0;
   fPath = "";
   fCache = 0;
   fLevel = 0;
   fPainter = 0;
   fGVolumes = 0;
   fGShapes = 0;
   fSearchOverlaps = kFALSE;
   fCurrentOverlapping = kFALSE;
   fLoopVolumes = kFALSE;
   fStartSafe = kFALSE;
   fSafety = 0;
   fStep = 0;
   fIsEntering = kFALSE;
   fIsExiting  = kFALSE;
   fIsStepEntering = kFALSE;
   fIsStepExiting  = kFALSE;
   fIsOutside  = kFALSE;
   gGeoIdentity = 0;
}
//-----------------------------------------------------------------------------
TGeoManager::TGeoManager(const char *name, const char *title)
            :TNamed(name, title)
{
// constructor
   gGeoManager = this;
   fSearchOverlaps = kFALSE;
   fLoopVolumes = kFALSE;
   fStartSafe = kTRUE;
   fSafety = 0;
   fStep = 0;
   fBits = new UChar_t[50000]; // max 25000 nodes per volume
   fMaterials = new TList();
   fMatrices = new TList();
   fNodes = new TObjArray(30);
   fNNodes = 0;
   fGlobalMatrices = new TObjArray(30);
   for (Int_t level=0; level<30; level++) {
      char *name = new char[20];
      name[0] = '\0';
      sprintf(name, "global%i", level);
      fGlobalMatrices->AddAt(new TGeoHMatrix(name), level);
   }
   fLevel = 0;
   fPoint = new Double_t[3];
   fDirection = new Double_t[3];
   fNormalChecked = new Double_t[3];
   fCldirChecked = new Double_t[3];
   fNormal = new Double_t[3];
   fCldir = new Double_t[3];
   fVolumes = new TList();
   fShapes = new TList();
   fGVolumes = new TList();
   fGShapes = new TList();
   fTopVolume = 0;
   fTopNode = 0;
   fCurrentVolume = 0;
   fMasterVolume = 0;
   fCurrentNode = 0;
   fLastNode = 0;
   fCurrentOverlapping = kFALSE;
   fPath = "";
   fCache = 0;
   fPainter = 0;
   fIsEntering = kFALSE;
   fIsExiting = kFALSE;
   fIsStepEntering = kFALSE;
   fIsStepExiting = kFALSE;
   fIsOutside = kFALSE;

   gGeoIdentity = new TGeoIdentity("Identity");
   BuildDefaultMaterials();
   gROOT->GetListOfGeometries()->Add(this);
}
//-----------------------------------------------------------------------------
TGeoManager::~TGeoManager()
{
// Destructor
   delete [] fBits;
   printf("deleting cache...\n");
   if (fCache) delete fCache;
//   printf("deleting top node...\n");
//   if (fTopNode) delete fTopNode;
   printf("deleting matrices...\n");
   if (fMatrices) {fMatrices->Delete(); delete fMatrices;}
//   printf("deleting global matrices...\n");
   if (fGlobalMatrices) {fGlobalMatrices->Delete(); delete fGlobalMatrices;}
   if (fNodes) delete fNodes;
   printf("deleting materials...\n");
   if (fMaterials) {fMaterials->Delete(); delete fMaterials;}
   printf("deleting shapes...\n");
   if (fShapes) {fShapes->Delete(); delete fShapes;}
   printf("deleting volumes...\n");
   if (fVolumes) {fVolumes->Delete(); delete fVolumes;}
   printf("cleaning garbage...\n");
   CleanGarbage();
   delete [] fPoint;
   delete [] fDirection;
   delete [] fNormalChecked;
   delete [] fCldirChecked;
   delete [] fNormal;
   delete [] fCldir;
   delete fGVolumes;
   delete fGShapes;
   gGeoIdentity = 0;
   gGeoManager = 0;
}
//-----------------------------------------------------------------------------
Int_t TGeoManager::AddMaterial(TGeoMaterial *material)
{
// Add a material to the list. Returns index of the material in list
   if (!material) {
      Error("AddMaterial", "invalid material");
      return -1;
   }
//   printf("adding material %s\n", material->GetTitle());
   Int_t index = GetMaterialIndex(material->GetName());
   if (index >= 0) return index;
   index = fMaterials->GetSize();
   fMaterials->Add(material);
   return index;
}
//-----------------------------------------------------------------------------
Int_t TGeoManager::AddTransformation(TGeoMatrix *matrix)
{
// Add a matrix to the list. Returns index of the matrix in list
   if (!matrix) {
      Error("AddMatrix", "invalid matrix");
      return -1;
   }
//   printf("adding matrix %s\n", matrix->GetName());
   Int_t index = fMatrices->GetSize();
   fMatrices->Add(matrix);
   return index;
}
//-----------------------------------------------------------------------------
Int_t TGeoManager::AddShape(TGeoShape *shape)
{
// Add a shape to the list. Returns index of the shape in list
   if (!shape) {
      Error("AddShape", "invalid shape");
      return -1;
   }
//   if (fShapes->FindObject(shape->GetName())) {
//      Error("AddShape", "a shape with this name already defined");
//      return -1;
//   }
   TList *list = fShapes;
   if (shape->IsRunTimeShape()) list = fGShapes;;
   Int_t index = list->GetSize();
//   printf("adding shape %i\n", index);
   list->Add(shape);
   return index;
}
//-----------------------------------------------------------------------------
Int_t TGeoManager::AddVolume(TGeoVolume *volume)
{
// Add a volume to the list. Returns index of the volume in list
   if (!volume) {
      Error("AddVolume", "invalid volume");
      return -1;
   }
//      Warning("AddVolume", "a volume with this name already defined");
   TList *list = fVolumes;
   if (volume->IsRunTime()) list = fGVolumes;
   Int_t index = list->GetSize();
   list->Add(volume);
   return index;
}
//-----------------------------------------------------------------------------
void TGeoManager::Browse(TBrowser *b)
{
   if (!b) return;
   if (fMaterials) b->Add(fMaterials, "Materials");
   if (fMatrices)  b->Add(fMatrices, "Local transformations");
   if (fTopVolume) b->Add(fTopVolume);
   if (fTopNode)   b->Add(fTopNode);
}
//-----------------------------------------------------------------------------
void TGeoManager::BombTranslation(const Double_t *tr, Double_t *bombtr)
{
// get the new 'bombed' translation vector according current exploded view mode
   if (fPainter) fPainter->BombTranslation(tr, bombtr);
   return;
}
//-----------------------------------------------------------------------------
void TGeoManager::UnbombTranslation(const Double_t *tr, Double_t *bombtr)
{
// get the new 'unbombed' translation vector according current exploded view mode
   if (fPainter) fPainter->UnbombTranslation(tr, bombtr);
   return;
}
//-----------------------------------------------------------------------------
void TGeoManager::BuildCache()
{
// builds the cache memory and the default number of physical nodes
// within
   if (!fCache) {
      if (fNNodes>5000000)
         fCache = new TGeoCacheDummy(fTopNode);
      else
         fCache = new TGeoNodeCache(0);
   }
}
//-----------------------------------------------------------------------------
void TGeoManager::ClearPad()
{
// clear pad if any
   if (gPad) delete gPad;
   gPad = 0;
}
//-----------------------------------------------------------------------------
void TGeoManager::ClearAttributes()
{
// reset all attributes to default ones
   ClearPad();
   SetVisOption(0);
   SetVisLevel(3);
   SetExplodedView(0);
   SetBombFactors();
   if (!gStyle) return;
   TIter next(fVolumes);
   TGeoVolume *vol = 0;
   while ((vol=(TGeoVolume*)next())) {
      if (!vol->IsVisTouched()) continue;
      vol->SetVisibility(kTRUE);
      vol->SetVisDaughters(kTRUE);
//      vol->SetLineColor(gStyle->GetLineColor());
      vol->SetLineStyle(gStyle->GetLineStyle());
      vol->SetLineWidth(gStyle->GetLineWidth());
      vol->SetVisTouched(kFALSE);
   }
}
//-----------------------------------------------------------------------------
void TGeoManager::CloseGeometry()
{
// closing geometry implies building the physical nodes and voxels
   SelectTrackingMedia();
   printf("Fixing runtime shapes...\n");
   CheckGeometry();
//   printf("Fixing runtime shapes...\n");
//   CheckGeometry();
   printf("Counting nodes...\n");
   fNNodes = gGeoManager->CountNodes();
   Voxelize("ALL");
   printf("Building caches for nodes and matrices...\n");
   BuildCache();
   printf("### nodes in %s : %i\n", gGeoManager->GetTitle(), fNNodes);
   gROOT->GetListOfBrowsables()->Add(this);
   printf("----------------modeler ready----------------\n");
}
//-----------------------------------------------------------------------------
void TGeoManager::ClearShape(TGeoShape *shape)
{
   if (fShapes->FindObject(shape)) fShapes->Remove(shape);
   delete shape;
}
//-----------------------------------------------------------------------------
void TGeoManager::CleanGarbage()
{
// clean volumes and shapes from garbage collection
   TIter nextv(fGVolumes);
   TGeoVolume *vol = 0;
   while ((vol=(TGeoVolume*)nextv()))
      vol->SetFinder(0);
   fGVolumes->Delete();
   fGShapes->Delete();
}
//-----------------------------------------------------------------------------
void TGeoManager::CdTop()
{
// make top level node current
//----this is for no cache
//   Top();
//   return;
//-----------------------
   fLevel = 0;
   if (fCurrentOverlapping) fLastNode = fCurrentNode;
   fCurrentNode = fTopNode;
   fCache->CdTop();
   fCurrentOverlapping = fCurrentNode->IsOverlapping();
}
//-----------------------------------------------------------------------------
void TGeoManager::CdUp()
{
// go one level up in geometry
//----this is for no cache
//   Up();
//   return;
//-----------------------
   if (!fLevel) return;
   fLevel--;
   if (!fLevel) {
      CdTop();
      return;
   }
   fCache->CdUp();
   if (fCurrentOverlapping) fLastNode = fCurrentNode;
   fCurrentNode = fCache->GetNode();
   if (!fCurrentNode->IsOffset()) fCurrentOverlapping = fCurrentNode->IsOverlapping();
}
//-----------------------------------------------------------------------------
void TGeoManager::CdDown(Int_t index)
{
// cd to daughter. Can be called only with a valid daughter
//----this is for no cache
//   Down(index);
//   return;
//-----------------------
   TGeoNode *node = fCurrentNode->GetDaughter(index);
   Bool_t is_offset = node->IsOffset();
   if (is_offset)
      node->GetFinder()->cd(node->GetIndex());
   else
      fCurrentOverlapping = node->IsOverlapping();
   fCache->CdDown(index);
   fCurrentNode = node;
   fLevel++;
}
//-----------------------------------------------------------------------------
Bool_t TGeoManager::cd(const char *path)
{
// Browse the tree of nodes starting from fTopNode according to pathname.
// Changes the path accordingly.
   if (!strlen(path)) return kFALSE;
   CdTop();
   TString spath = path;
   TGeoVolume *vol;
   Int_t length = spath.Length();
   Int_t ind1 = spath.Index("/");
   Int_t ind2 = 0;
   Bool_t end = kFALSE;
   TString name;
   TGeoNode *node;
   while (!end) {
      ind2 = spath.Index("/", ind1+1);
      if (ind2<0) {
         ind2 = length;
         end  = kTRUE;
      }
      name = spath(ind1+1, ind2-ind1-1);
      if (name==fTopNode->GetName()) {
         ind1 = ind2;
         continue;
      }
      vol = fCurrentNode->GetVolume();
      if (vol) {
         node = vol->GetNode(name.Data());
      } else node = 0;
      if (!node) {
         Error("cd", "path not valid");
         return kFALSE;
      }
      CdDown(fCurrentNode->GetVolume()->GetIndex(node));
      ind1 = ind2;
   }
   return kTRUE;
}
//-----------------------------------------------------------------------------
Int_t TGeoManager::CountNodes(TGeoVolume *vol, Int_t nlevels)
{
// Count the total number of nodes starting from a volume, nlevels down
   TGeoVolume *top;
   if (!vol) {
      top = fTopVolume;
   } else {
      top = vol;
   }
   Int_t count = top->CountNodes(nlevels);
   return count;
}
//-----------------------------------------------------------------------------
void TGeoManager::DefaultAngles()
{
// Set default angles for a given view.
   if (fPainter) fPainter->DefaultAngles();
}
//-----------------------------------------------------------------------------
void TGeoManager::DrawCurrentPoint(Int_t color)
{
// Draw current point in the same view.
   if (fPainter) fPainter->DrawCurrentPoint(color);
}
//-----------------------------------------------------------------------------
void TGeoManager::RandomPoints(TGeoVolume *vol, Int_t npoints, Option_t *option)
{
// Draw random points in the bounding box of a volume.
   GetGeomPainter()->RandomPoints(vol, npoints, option);
}
//-----------------------------------------------------------------------------
void TGeoManager::Test(Int_t npoints, Option_t *option)
{
// Check time of finding "Where am I" for n points.
   GetGeomPainter()->Test(npoints, option); 
}
//-----------------------------------------------------------------------------
void TGeoManager::TestOverlaps(const char* path)
{
//--- Geometry overlap checker based on sampling. 
   GetGeomPainter()->TestOverlaps(path);
}
//-----------------------------------------------------------------------------
void TGeoManager::GetBombFactors(Double_t &bombx, Double_t &bomby, Double_t &bombz, Double_t &bombr) const
{
// Retreive cartesian and radial bomb factors.
   if (fPainter) {
      fPainter->GetBombFactors(bombx, bomby, bombz, bombr);
      return;
   }
   bombx = bomby = bombz = bombr = 1.3;
}
//-----------------------------------------------------------------------------
Int_t TGeoManager::GetVisLevel() const
{
// Returns current depth to which geometry is drawn.
   if (fPainter) return fPainter->GetVisLevel();
   return TVirtualGeoPainter::kGeoVisLevel;
}
//-----------------------------------------------------------------------------
Int_t TGeoManager::GetVisOption() const
{
// Returns current depth to which geometry is drawn.
   if (fPainter) return fPainter->GetVisOption();
   return TVirtualGeoPainter::kGeoVisDefault;
}
//-----------------------------------------------------------------------------
Int_t TGeoManager::GetVirtualLevel()
{
// Find level of virtuality of current overlapping node (number of levels
// up having the same tracking media.
   
   // return if the current node is ONLY
   if (!fCurrentOverlapping) return 0;
   Int_t new_media = 0;
   Int_t imedia = fCurrentNode->GetMedia();
   Int_t virtual_level = 1;
   TGeoNode *mother = 0;

   while ((mother=GetMother(virtual_level))) {
      if (!mother->IsOverlapping() && !mother->IsOffset()) {
         if (!new_media) new_media=(mother->GetMedia()==imedia)?0:virtual_level;
         break;
      }
      if (!new_media) new_media=(mother->GetMedia()==imedia)?0:virtual_level;
      virtual_level++;
   }
   return (new_media==0)?virtual_level:(new_media-1);
}
//-----------------------------------------------------------------------------
Bool_t TGeoManager::GotoSafeLevel()
{
// Go upwards the tree until a non-overlaping node
   while (fCurrentOverlapping && fLevel) CdUp();
   return kTRUE;
}
//-----------------------------------------------------------------------------
TGeoNode *TGeoManager::FindInCluster(Int_t *cluster, Int_t nc)
{
// Find a node inside a cluster of overlapping nodes. Current node must
// be on top of all the nodes in cluster. Always nc>1
   TGeoNode *clnode = 0;
   TGeoNode *priority = fLastNode;
   // save current node
   TGeoNode *current = fCurrentNode;
   TGeoNode *found = 0;
   // save path
   Int_t ipop = PushPath();
   // mark this search
   fSearchOverlaps = kTRUE;
   Int_t deepest = fLevel;
   Int_t deepest_virtual = fLevel-GetVirtualLevel();
   Int_t found_virtual = 0;
   Bool_t replace = kFALSE;
   Bool_t added = kFALSE;
   Int_t i;
   for (i=0; i<nc; i++) {
      clnode = current->GetDaughter(cluster[i]);
      CdDown(cluster[i]);
      found = SearchNode(kTRUE, clnode);
      if (!fSearchOverlaps) {
      // an only was found during the search -> exiting
         PopDummy(ipop);
         return found;
      }
      found_virtual = fLevel-GetVirtualLevel();
      if (added) {
      // we have put something in stack -> check it
         if (found_virtual>deepest_virtual) {
            replace = kTRUE;
         } else {
            if (found_virtual==deepest_virtual) {
               if (fLevel>deepest) {
                  replace = kTRUE;
               } else {
                  if ((fLevel==deepest) && (clnode==priority)) replace=kTRUE;
                  else                                          replace = kFALSE;
               }
            } else                 replace = kFALSE;
         }
         // if this was the last checked node
         if (i==(nc-1)) {
            if (replace) {
               PopDummy(ipop);
               return found;
            } else {
               fCurrentOverlapping = PopPath();
               PopDummy(ipop);
               return fCurrentNode;
            }
         }
         // we still have to go on
         if (replace) {
            // reset stack
            PopDummy();
            PushPath();
            deepest = fLevel;
            deepest_virtual = found_virtual;
         }
         // restore top of cluster
         fCurrentOverlapping = PopPath(ipop);
      } else {
      // the stack was clean, push new one
         PushPath();
         added = kTRUE;
         deepest = fLevel;
         deepest_virtual = found_virtual;
         // restore original path
         fCurrentOverlapping = PopPath(ipop);
      }
   }
   PopDummy(ipop);
   return fCurrentNode;
}
//-----------------------------------------------------------------------------
Int_t TGeoManager::GetTouchedCluster(Int_t start, Double_t *point,
                              Int_t *check_list, Int_t ncheck, Int_t *result)
{
// Make the cluster of overlapping nodes in a voxel, containing point in reference
// of the mother. Returns number of nodes containing the point. Nodes should not be
// offsets.

   // we are in the mother reference system
   TGeoNode *current = fCurrentNode->GetDaughter(check_list[start]);
   Int_t novlps = 0;
   Int_t *ovlps = current->GetOverlaps(novlps);
   if (!ovlps) return 0;
   Double_t local[3];
   // intersect check list with overlap list
   Int_t ntotal = 0;
   current->MasterToLocal(point, &local[0]);
   if (current->GetVolume()->Contains(&local[0])) {
      result[ntotal++]=check_list[start];
   }

   Int_t jst=0, i, j;
   while ((ovlps[jst]<=check_list[start]) && (jst<novlps))  jst++;
   if (jst==novlps) return 0;
   for (i=start; i<ncheck; i++) {
      for (j=jst; j<novlps; j++) {
         if (check_list[i]==ovlps[j]) {
         // overlapping node in voxel -> check if touched
            current = fCurrentNode->GetDaughter(check_list[i]);
            current->MasterToLocal(point, &local[0]);
            if (current->GetVolume()->Contains(&local[0])) {
               result[ntotal++]=check_list[i];
            }
         }
      }
   }
   return ntotal;
}
//-----------------------------------------------------------------------------
void TGeoManager::DefaultColors()
{
// Set default volume colors according to tracking media.
   if (fPainter) {
      fPainter->DefaultColors();
      return;
   }   
   TIter next(fVolumes);
   TGeoVolume *vol;
   while ((vol=(TGeoVolume*)next()))
      vol->SetLineColor(vol->GetMaterial()->GetDefaultColor());
}
//-----------------------------------------------------------------------------
void TGeoManager::SetBombFactors(Double_t bombx, Double_t bomby, Double_t bombz, Double_t bombr) 
{
// Set factors that will "bomb" all translations in cartesian and cylindrical coordinates.
   if (fPainter) fPainter->SetBombFactors(bombx, bomby, bombz, bombr);
}   
//-----------------------------------------------------------------------------
void TGeoManager::SetVisOption(Int_t option) {
// set drawing mode :
// option=0 (default) all nodes drawn down to vislevel
// option=1           leaves and nodes at vislevel drawn
// option=2           path is drawn
   GetGeomPainter();
   fPainter->SetVisOption(option);
}
//-----------------------------------------------------------------------------
void TGeoManager::SetVisLevel(Int_t level) {
// set default level down to which visualization is performed
   GetGeomPainter();
   fPainter->SetVisLevel(level);
}
//-----------------------------------------------------------------------------
void TGeoManager::SaveAttributes(const char *filename)
{
// Save current attributes in a macro
   if (!fTopNode) {
      printf("SaveAttributes - geometry must be closed\n");
      return;
   }
   ofstream out;
   char *fname = new char[20];
   char quote = '"';
   if (!strlen(filename))
      sprintf(fname, "tgeoatt.C");
   else
      sprintf(fname, "%s", filename);
   out.open(fname, ios::out);
   if (!out.good()) {
      Error("SaveAttributes", "cannot open file");
      delete [] fname;
      return;
   }
   // write header
   TDatime t;
   TString sname(fname);
   sname.ReplaceAll(".C", "");
   out << sname.Data()<<"()"<<endl;
   out << "{" << endl;
   out << "//=== Macro generated by ROOT version "<< gROOT->GetVersion()<<" : "<<t.AsString()<<endl;
   out << "//=== Attributes for " << GetTitle() << " geometry"<<endl;
   out << "//===== <run this macro AFTER loading the geometry in memory>"<<endl;
   // save current top volume
   out << "   TGeoVolume *top = gGeoManager->GetVolume("<<quote<<gGeoManager->GetTopVolume()->GetName()<<quote<<");"<<endl;
   out << "   TGeoVolume *vol = 0;"<<endl;
   out << "   // clear all volume attributes and get painter"<<endl;
   out << "   gGeoManager->ClearAttributes();"<<endl;
   out << "   gGeoManager->GetGeomPainter();"<<endl;
   out << "   // set visualization modes and bomb factors"<<endl;
   out << "   gGeoManager->SetVisOption("<<gGeoManager->GetVisOption()<<");"<<endl;
   out << "   gGeoManager->SetVisLevel("<<gGeoManager->GetVisLevel()<<");"<<endl;
   out << "   gGeoManager->SetExplodedView("<<gGeoManager->GetBombMode()<<");"<<endl;
   Double_t bombx, bomby, bombz, bombr;
   GetBombFactors(bombx, bomby, bombz, bombr);
   out << "   gGeoManager->SetBombFactors("<<bombx<<","<<bomby<<","<<bombz<<","<<bombr<<");"<<endl;
   out << "   // iterate volumes coontainer and set new attributes"<<endl;
//   out << "   TIter next(gGeoManager->GetListOfVolumes());"<<endl;
   TGeoVolume *vol = 0;
   fTopNode->SaveAttributes(out);

   TIter next(fVolumes);
   while ((vol=(TGeoVolume*)next())) {
      vol->SetVisStreamed(kFALSE);
   }
   out << "   // draw top volume with new settings"<<endl;
   out << "   top->Draw();"<<endl;
   out << "}" << endl;
   out.close();
   delete [] fname;
}
//-----------------------------------------------------------------------------
TGeoNode *TGeoManager::SearchNode(Bool_t downwards, TGeoNode *skipnode)
{
// Returns the deepest node containing fPoint, which must be set a priori.
   Double_t point[3];
   TGeoVolume *vol = 0;
   Bool_t inside_current = kFALSE;
   if (!downwards) {
   // we are looking upwards until inside current node or exit
      if (fStartSafe) GotoSafeLevel();
      vol=fCurrentNode->GetVolume();
      MasterToLocal(fPoint, &point[0]);
      inside_current = vol->Contains(&point[0]);
      if (!inside_current) {
         TGeoNode *skip = fCurrentNode;
         // check if we can go up
         if (!fLevel) {
            fIsOutside = kTRUE;
            return 0;
         }
         CdUp();
         return SearchNode(kFALSE, skip);
      }
   }
   if (!inside_current) {
   // we are looking downwards
      vol = fCurrentNode->GetVolume();
      MasterToLocal(fPoint, &point[0]);
      if (fCurrentNode==skipnode) {
      // in case searching down and skipping this
         inside_current = kTRUE;
      } else {
         inside_current = vol->Contains(&point[0]);
         if (!inside_current) return 0;
      }
   }
   // point inside current (safe) node -> search downwards
   TGeoNode *node;
   Int_t ncheck = 0;
   // if inside an non-overlapping node, reset overlap searches
   if (!fCurrentOverlapping) {
      fSearchOverlaps = kFALSE;
   }

   Int_t nd = vol->GetNdaughters();
   // in case there are no daughters
   if (!nd) return fCurrentNode;

   TGeoPatternFinder *finder = vol->GetFinder();
   // point is inside the current node
   // first check if inside a division
   if (finder) {
      node=finder->FindNode(&point[0]);
      if (node) {
         // go inside the division cell and search downwards
         CdDown(node->GetIndex());
         return SearchNode(kTRUE, node);
      }
      // point is not inside the division, but might be in other nodes
      // at the same level (NOT SUPPORTED YET)
      return fCurrentNode;
   }
   // second, look if current volume is voxelized
   TGeoVoxelFinder *voxels = vol->GetVoxels();
   Int_t *check_list = 0;
   if (voxels) {
      // get the list of nodes passing thorough the current voxel
      check_list = voxels->GetCheckList(&point[0], ncheck);
      // if none in voxel, see if this is the last one
      if (!check_list) return fCurrentNode;
      // loop all nodes in voxel
      for (Int_t id=0; id<ncheck; id++) {
         node = vol->GetNode(check_list[id]);
         if (node==skipnode) continue;
         if ((id<(ncheck-1)) && node->IsOverlapping()) {
         // make the cluster of overlaps
            Int_t *cluster = new Int_t[ncheck-id];
            Int_t nc = GetTouchedCluster(id, &point[0], check_list, ncheck, cluster);
            if (nc>1) {
               node = FindInCluster(cluster, nc);
               delete [] cluster;
               return node;
            }
         }
         CdDown(check_list[id]);
         node = SearchNode(kTRUE);
         if (node) return node;
         CdUp();
      }
      return fCurrentNode;
   }
   // if there are no voxels just loop all daughters
   Int_t id = 0;
   while ((node=fCurrentNode->GetDaughter(id++))) {
      if (node==skipnode) {
         if (id==nd) return fCurrentNode;
         continue;
      }
      CdDown(id-1);
      node = SearchNode(kTRUE);
      if (node) return node;
      CdUp();
      if (id == nd) return fCurrentNode;
   }
   // point is not inside one of the daughters, so it is in the current vol
   return fCurrentNode;
}
//-----------------------------------------------------------------------------
TGeoNode *TGeoManager::FindNextBoundary(const char *path)
{
// Find distance to target node given by path boundary on current direction. If no target
// is specified, find distance to next boundary from current point to current direction
// and store this in fStep. Returns node having this boundary. Find also
// distance to closest boundary and store it in fSafety.

   // convert current point and direction to local reference
   fStep = TGeoShape::kBig;
   Double_t point[3];
   Double_t dir[3];
   if (strlen(path)) {
      PushPath();
      if (!cd(path)) {
         PopPath();
         return 0;
      }
      TGeoNode *target=fCurrentNode;
      TGeoVolume *tvol=fCurrentNode->GetVolume();
      MasterToLocal(fPoint, &point[0]);
      MasterToLocalVect(fDirection, &dir[0]);
      if (tvol->Contains(&point[0]))
         fStep=tvol->GetShape()->DistToOut(&point[0], &dir[0], 3, TGeoShape::kBig, &fSafety);
      else
         fStep=tvol->GetShape()->DistToIn(&point[0], &dir[0], 3, TGeoShape::kBig, &fSafety);
      PopPath();
      return target;
   }
   MasterToLocal(fPoint, &point[0]);
   MasterToLocalVect(fDirection, &dir[0]);
   // compute distance to exit point from current node and the distance to its
   // closest boundary
   TGeoVolume *vol = fCurrentNode->GetVolume();
   // if point is outside, just check the top node
   if (fIsOutside) {
      fStep = vol->GetShape()->DistToIn(&point[0], &dir[0], 3, TGeoShape::kBig, &fSafety);
      return fTopNode;
   }
   if (fIsEntering || fIsExiting) {
      fStep = vol->GetShape()->DistToOut(&point[0], &dir[0], 3, TGeoShape::kBig, &fSafety);
      if (fIsExiting) return fCurrentNode;
   } else {
      fStep = vol->GetShape()->DistToOut(&point[0], &dir[0], 2, TGeoShape::kBig, &fSafety);
   }
   // get number of daughters. If no daughters we are done.
   Int_t nd = vol->GetNdaughters();
   if (!nd) return fCurrentNode;
   TGeoNode *current = 0;
   TGeoNode *clnode = fCurrentNode;
   Double_t lpoint[3];
   Double_t ldir[3];
   Double_t safety = TGeoShape::kBig;
   Double_t snext  = TGeoShape::kBig;
   Int_t i=0;
   // if only one daughter, check it and exit
   if (nd<3) {
      for (i=0; i<nd; i++) {
         current = vol->GetNode(i);
         current->cd();
         current->MasterToLocal(&point[0], &lpoint[0]);
         current->MasterToLocalVect(&dir[0], &ldir[0]);
         snext = current->GetVolume()->GetShape()->DistToIn(&lpoint[0], &ldir[0], 2, TGeoShape::kBig, &safety);
         fSafety = TMath::Min(fSafety, safety);
         if (snext<fStep) {
            fStep=snext;
            clnode = current;
         }
      }
   return clnode;
   }
   // if current volume is divided, we are in the non-divided region. We
   // check only the first and the last cell
   TGeoPatternFinder *finder = vol->GetFinder();
   if (finder) {
      Int_t ifirst = finder->GetDivIndex();
      current = vol->GetNode(ifirst);
      current->cd();
      current->MasterToLocal(&point[0], &lpoint[0]);
      current->MasterToLocalVect(&dir[0], &ldir[0]);
      snext = current->GetVolume()->GetShape()->DistToIn(&lpoint[0], &ldir[0], 2, TGeoShape::kBig, &safety);
      fSafety = TMath::Min(fSafety, safety);
      if (snext<fStep) {
         fStep=snext;
         clnode = current;
      }
      Int_t ilast = ifirst+finder->GetNdiv()-1;
      if (ilast==ifirst) return clnode;
      current = vol->GetNode(ilast);
      current->cd();
      current->MasterToLocal(&point[0], &lpoint[0]);
      current->MasterToLocalVect(&dir[0], &ldir[0]);
      snext = current->GetVolume()->GetShape()->DistToIn(&lpoint[0], &ldir[0], 2, TGeoShape::kBig, &safety);
      fSafety = TMath::Min(fSafety, safety);
      if (snext<fStep) {
         fStep=snext;
         return current;
      }
   }
   // if current volume is voxelized, first get current voxel
   TGeoVoxelFinder *voxels = vol->GetVoxels();
   if (voxels) {
      Int_t ncheck = 0;
      Int_t *vlist = 0;
      voxels->SortCrossedVoxels(&point[0], &dir[0]);
      Bool_t first = kTRUE;
      while ((vlist=voxels->GetNextVoxel(&point[0], &dir[0], ncheck))) {
         for (i=0; i<ncheck; i++) {
            current = vol->GetNode(vlist[i]);
            current->cd();
            current->MasterToLocal(&point[0], &lpoint[0]);
            current->MasterToLocalVect(&dir[0], &ldir[0]);
            if (first) {
            // compute also safety if we are in the starting voxel
               snext = current->GetVolume()->GetShape()->DistToIn(&lpoint[0], &ldir[0], 2, TGeoShape::kBig, &safety);
               if (safety<fSafety) fSafety=safety;
            }
            else
               snext = current->GetVolume()->GetShape()->DistToIn(&lpoint[0], &ldir[0], 3, TGeoShape::kBig, &safety);
            if (snext<fStep) {
               fStep=snext;
               clnode = current;
            }
         }
         first=kFALSE;
      }
   }
   return clnode;
}
//-----------------------------------------------------------------------------
void TGeoManager::InitTrack(Double_t *point, Double_t *dir)
{
// initialize current point and current direction vector (normalized)
// in MARS
   SetCurrentPoint(point);
   SetCurrentDirection(dir);
   FindNode();
}
//-----------------------------------------------------------------------------
void TGeoManager::InitTrack(Double_t x, Double_t y, Double_t z, Double_t nx, Double_t ny, Double_t nz)
{
// initialize current point and current direction vector (normalized)
// in MARS
   SetCurrentPoint(x,y,z);
   SetCurrentDirection(nx,ny,nz);
   FindNode();
}
//-----------------------------------------------------------------------------
const char *TGeoManager::GetPath() const
{
   if (fIsOutside) return kGeoOutsidePath;
   return fCache->GetPath();
}
//-----------------------------------------------------------------------------
Int_t TGeoManager::GetByteCount(Option_t *option)
{
// Get total size of geometry in bytes
   Int_t count = 0;
   TIter next(fVolumes);
   TGeoVolume *vol;
   while ((vol=(TGeoVolume*)next())) count += vol->GetByteCount();
   TIter next1(fMatrices);
   TGeoMatrix *matrix;
   while ((matrix=(TGeoMatrix*)next1())) count += matrix->GetByteCount();
   TIter next2(fMaterials);
   TGeoMaterial *mat;
   while ((mat=(TGeoMaterial*)next2())) count += mat->GetByteCount();
   printf("Total size of logical tree : %i bytes\n", count);
   return count;
}
//-----------------------------------------------------------------------------
TVirtualGeoPainter *TGeoManager::GetGeomPainter()
{
// make a default painter if none present
    if (!fPainter) fPainter=TVirtualGeoPainter::GeoPainter();
    return fPainter;
}
//-----------------------------------------------------------------------------
TGeoMaterial *TGeoManager::GetMaterial(const char *matname) const
{
// search for given material
   TGeoMaterial *mat = (TGeoMaterial*)fMaterials->FindObject(matname);
   return mat;
}
//-----------------------------------------------------------------------------
TGeoMaterial *TGeoManager::GetMaterial(Int_t id) const
{
// return material at position id
   if (id >= fMaterials->GetSize()) return 0;
   TGeoMaterial *mat = (TGeoMaterial*)fMaterials->At(id);
   return mat;
}
//-----------------------------------------------------------------------------
Int_t TGeoManager::GetMaterialIndex(const char *matname) const
{
// return index of given material
   TIter next(fMaterials);
   TGeoMaterial *mat;
   Int_t id = 0;
   while ((mat = (TGeoMaterial*)next())) {
      if (mat->GetName() == matname)
         return id;
      id++;
   }
   return -1;  // fail
}
//-----------------------------------------------------------------------------
void TGeoManager::RandomRays(Int_t nrays)
{
// randomly shoot nrays and plot intersections with surfaces for current
// top node
   GetGeomPainter()->RandomRays(nrays);
}
//-----------------------------------------------------------------------------
void TGeoManager::RemoveMaterial(Int_t index)
{
// remove material at given index
   TObject *obj = fMaterials->At(index);
   if (obj) fMaterials->Remove(obj);
}
//-----------------------------------------------------------------------------
void TGeoManager::RestoreMasterVolume()
{
   if (fTopVolume == fMasterVolume) return;
   if (fMasterVolume) SetTopVolume(fMasterVolume);
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoManager::GetVolume(const char *name) const
{
// retrieves a named volume
   return ((TGeoVolume*)fVolumes->FindObject(name));
}
//-----------------------------------------------------------------------------
void TGeoManager::Voxelize(Option_t *option)
{
   // voxelize all non-divided volumes
   TGeoVolume *vol;
   printf("Voxelizing...\n");
   Int_t nentries = fVolumes->GetSize();
   for (Int_t i=0; i<nentries; i++) {
      vol = (TGeoVolume*)fVolumes->At(i);
      vol->SortNodes();
      vol->Voxelize(option);
      vol->FindOverlaps();
   }
}
//-----------------------------------------------------------------------------
void TGeoManager::ModifiedPad() const
{
// Send "Modified" signal to painter.
   if (!fPainter) return;
   fPainter->ModifiedPad();
}   
//-----------------------------------------------------------------------------
TGeoVolume *TGeoManager::MakeArb8(const char *name, const char *material,
                                  Double_t dz, Double_t *vertices)
{
   // Make an arb8 shape
   TGeoVolume *old = 0;
   old=(TGeoVolume*)fVolumes->FindObject(name);
   TGeoMaterial *mat = GetMaterial(material);
   if (!mat) {
      printf("%s\n", material);
      Warning("MakeArb8", "Material  unknown");
      mat = GetMaterial("default");
   }
   TGeoArb8 *arb = new TGeoArb8(dz, vertices);
   TGeoVolume *vol = new TGeoVolume(name, arb, mat);
   if (old) vol->MakeCopyNodes(old);
   return vol;
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoManager::MakeBox(const char *name, const char *material,
                                    Double_t dx, Double_t dy, Double_t dz)
{
// Make in one step a volume pointing to a box shape with given material
   TGeoVolume *old = 0;
   old=(TGeoVolume*)fVolumes->FindObject(name);
   TGeoMaterial *mat = GetMaterial(material);
   if (!mat) {
      printf("%s\n", material);
      Warning("MakeBox", "Material  unknown");
      mat = GetMaterial("default");
   }
   TGeoBBox *box = new TGeoBBox(dx, dy, dz);
   TGeoVolume *vol = new TGeoVolume(name, box, mat);
   if (old) vol->MakeCopyNodes(old);
   return vol;
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoManager::MakePara(const char *name, const char *material,
                                    Double_t dx, Double_t dy, Double_t dz,
                                    Double_t alpha, Double_t theta, Double_t phi)
{
// Make in one step a volume pointing to a box shape with given material
   if ((alpha==0) && (theta==0)) {
      printf("Warning : para %s with alpha=0, theta=0 -> making box instead\n", name);
      return MakeBox(name, material, dx, dy, dz);
   }
   TGeoVolume *old = 0;
   old=(TGeoVolume*)fVolumes->FindObject(name);
   TGeoMaterial *mat = GetMaterial(material);
   if (!mat) {
      printf("%s\n", material);
      Warning("MakePara", "Material  unknown");
      mat = GetMaterial("default");
   }
   TGeoPara *para=0;
   para = new TGeoPara(dx, dy, dz, alpha, theta, phi);
   TGeoVolume *vol = new TGeoVolume(name, para, mat);
   if (old) vol->MakeCopyNodes(old);
   return vol;
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoManager::MakeSphere(const char *name, const char *material,
                                    Double_t rmin, Double_t rmax, Double_t themin, Double_t themax,
                                    Double_t phimin, Double_t phimax)
{
// Make in one step a volume pointing to a sphere shape with given material
   TGeoVolume *old = 0;
   old=(TGeoVolume*)fVolumes->FindObject(name);
   TGeoMaterial *mat = GetMaterial(material);
   if (!mat) {
      printf("%s\n", material);
      Warning("MakeSphere", " unknown");
      mat = GetMaterial("default");
   }
   TGeoSphere *sph = new TGeoSphere(rmin, rmax, themin, themax, phimin, phimax);
   TGeoVolume *vol = new TGeoVolume(name, sph, mat);
   if (old) vol->MakeCopyNodes(old);
   return vol;
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoManager::MakeTube(const char *name, const char *material,
                                     Double_t rmin, Double_t rmax, Double_t dz)
{
// Make in one step a volume pointing to a tube shape with given material
   TGeoVolume *old = 0;
   old=(TGeoVolume*)fVolumes->FindObject(name);
   TGeoMaterial *mat = GetMaterial(material);
   if (!mat) {
      printf("%s\n", material);
      Warning("MakeTube", "Material unknown");
      mat = GetMaterial("default");
   }
   TGeoTube *tube = new TGeoTube(rmin, rmax, dz);
   TGeoVolume *vol = new TGeoVolume(name, tube, mat);
   if (old) vol->MakeCopyNodes(old);
   return vol;
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoManager::MakeTubs(const char *name, const char *material,
                                     Double_t rmin, Double_t rmax, Double_t dz,
                                     Double_t phi1, Double_t phi2)
{
// Make in one step a volume pointing to a tube segment shape with given material
   TGeoVolume *old = 0;
   old=(TGeoVolume*)fVolumes->FindObject(name);
   TGeoMaterial *mat = GetMaterial(material);
   if (!mat) {
      printf("%s\n", material);
      Warning("MakeTubs", "Material unknown");
      mat = GetMaterial("default");
   }
   TGeoTubeSeg *tubs = new TGeoTubeSeg(rmin, rmax, dz, phi1, phi2);
   TGeoVolume *vol = new TGeoVolume(name, tubs, mat);
   if (old) vol->MakeCopyNodes(old);
   return vol;
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoManager::MakeEltu(const char *name, const char *material,
                                     Double_t a, Double_t b, Double_t dz)
{
// Make in one step a volume pointing to a tube shape with given material
   TGeoVolume *old = 0;
   old=(TGeoVolume*)fVolumes->FindObject(name);
   TGeoMaterial *mat = GetMaterial(material);
   if (!mat) {
      printf("%s\n", material);
      Warning("MakeTube", "Material unknown");
      mat = GetMaterial("default");
   }
   TGeoEltu *eltu = new TGeoEltu(a, b, dz);
   TGeoVolume *vol = new TGeoVolume(name, eltu, mat);
   if (old) vol->MakeCopyNodes(old);
   return vol;
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoManager::MakeCtub(const char *name, const char *material,
                                     Double_t rmin, Double_t rmax, Double_t dz, Double_t phi1, Double_t phi2,
                                     Double_t lx, Double_t ly, Double_t lz, Double_t tx, Double_t ty, Double_t tz)
{
// Make in one step a volume pointing to a tube segment shape with given material
   TGeoVolume *old = 0;
   old=(TGeoVolume*)fVolumes->FindObject(name);
   TGeoMaterial *mat = GetMaterial(material);
   if (!mat) {
      printf("%s\n", material);
      Warning("MakeTubs", "Material unknown");
      mat = GetMaterial("default");
   }
   TGeoCtub *ctub = new TGeoCtub(rmin, rmax, dz, phi1, phi2, lx, ly, lz, tx, ty, tz);
   TGeoVolume *vol = new TGeoVolume(name, ctub, mat);
   if (old) vol->MakeCopyNodes(old);
   return vol;
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoManager::MakeCone(const char *name, const char *material,
                                     Double_t dz, Double_t rmin1, Double_t rmax1,
                                     Double_t rmin2, Double_t rmax2)
{
// Make in one step a volume pointing to a cone shape with given material
   TGeoVolume *old = 0;
   old=(TGeoVolume*)fVolumes->FindObject(name);
   TGeoMaterial *mat = GetMaterial(material);
   if (!mat) {
      printf("%s\n", material);
      Warning("MakeCone", "Material unknown");
      mat = GetMaterial("default");
   }
   TGeoCone *cone = new TGeoCone(dz, rmin1, rmax1, rmin2, rmax2);
   TGeoVolume *vol = new TGeoVolume(name, cone, mat);
   if (old) vol->MakeCopyNodes(old);
   return vol;
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoManager::MakeCons(const char *name, const char *material,
                                     Double_t dz, Double_t rmin1, Double_t rmax1,
                                     Double_t rmin2, Double_t rmax2,
                                     Double_t phi1, Double_t phi2)
{
// Make in one step a volume pointing to a cone segment shape with given material
   TGeoVolume *old = 0;
   old=(TGeoVolume*)fVolumes->FindObject(name);
   TGeoMaterial *mat = GetMaterial(material);
   if (!mat) {
      printf("%s\n", material);
      Warning("MakeCons", "Material unknown");
      mat = GetMaterial("default");
   }
   TGeoConeSeg *cons = new TGeoConeSeg(dz, rmin1, rmax1, rmin2, rmax2, phi1, phi2);
   TGeoVolume *vol = new TGeoVolume(name, cons, mat);
   if (old) vol->MakeCopyNodes(old);
   return vol;
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoManager::MakePcon(const char *name, const char *material,
                                     Double_t phi, Double_t dphi, Int_t nz)
{
// Make in one step a volume pointing to a pcon shape with given material
   TGeoVolume *old = 0;
   old=(TGeoVolume*)fVolumes->FindObject(name);
   TGeoMaterial *mat = GetMaterial(material);
   if (!mat) {
      printf("%s\n", material);
      Warning("MakePcon", "Material unknown");
      mat = GetMaterial("default");
   }
   TGeoPcon *pcon = new TGeoPcon(phi, dphi, nz);
   TGeoVolume *vol = new TGeoVolume(name, pcon, mat);
   if (old) vol->MakeCopyNodes(old);
   return vol;
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoManager::MakePgon(const char *name, const char *material,
                                     Double_t phi, Double_t dphi, Int_t nedges, Int_t nz)
{
// Make in one step a volume pointing to a pgon shape with given material
   TGeoVolume *old = 0;
   old=(TGeoVolume*)fVolumes->FindObject(name);
   TGeoMaterial *mat = GetMaterial(material);
   if (!mat) {
      printf("%s\n", material);
      Warning("MakePgon", "Material unknown");
      mat = GetMaterial("default");
   }
   TGeoPgon *pgon = new TGeoPgon(phi, dphi, nedges, nz);
   TGeoVolume *vol = new TGeoVolume(name, pgon, mat);
   if (old) vol->MakeCopyNodes(old);
   return vol;
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoManager::MakeTrd1(const char *name, const char *material,
                                  Double_t dx1, Double_t dx2, Double_t dy, Double_t dz)
{
// Make in one step a volume pointing to a trd1 shape with given material
   TGeoVolume *old = 0;
   old=(TGeoVolume*)fVolumes->FindObject(name);
   TGeoMaterial *mat = GetMaterial(material);
   if (!mat) {
      printf("%s\n", material);
      Warning("MakeTrd1", "Material unknown");
      mat = GetMaterial("default");
   }
   TGeoTrd1 *trd1 = new TGeoTrd1(dx1, dx2, dy, dz);
   TGeoVolume *vol = new TGeoVolume(name, trd1, mat);
   if (old) vol->MakeCopyNodes(old);
   return vol;
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoManager::MakeTrd2(const char *name, const char *material,
                                  Double_t dx1, Double_t dx2, Double_t dy1, Double_t dy2,
                                  Double_t dz)
{
// Make in one step a volume pointing to a trd2 shape with given material
   TGeoVolume *old = 0;
   old=(TGeoVolume*)fVolumes->FindObject(name);
   TGeoMaterial *mat = GetMaterial(material);
   if (!mat) {
      printf("%s\n", material);
      Warning("MakeTrd2", "Material unknown");
      mat = GetMaterial("default");
   }
   TGeoTrd2 *trd2 = new TGeoTrd2(dx1, dx2, dy1, dy2, dz);
   TGeoVolume *vol = new TGeoVolume(name, trd2, mat);
   if (old) vol->MakeCopyNodes(old);
   return vol;
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoManager::MakeTrap(const char *name, const char *material,
                                  Double_t dz, Double_t theta, Double_t phi, Double_t h1,
                                  Double_t bl1, Double_t tl1, Double_t alpha1, Double_t h2, Double_t bl2,
                                  Double_t tl2, Double_t alpha2)
{
// Make in one step a volume pointing to a trd2 shape with given material
   TGeoVolume *old = 0;
   old=(TGeoVolume*)fVolumes->FindObject(name);
   TGeoMaterial *mat = GetMaterial(material);
   if (!mat) {
      printf("%s\n", material);
      Warning("MakeTrap", "Material unknown");
      mat = GetMaterial("default");
   }
   TGeoTrap *trap = new TGeoTrap(dz, theta, phi, h1, bl1, tl1, alpha1, h2, bl2,
                                 tl2, alpha2);
   TGeoVolume *vol = new TGeoVolume(name, trap, mat);
   if (old) vol->MakeCopyNodes(old);
   return vol;
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoManager::MakeGtra(const char *name, const char *material,
                                  Double_t dz, Double_t theta, Double_t phi, Double_t twist, Double_t h1,
                                  Double_t bl1, Double_t tl1, Double_t alpha1, Double_t h2, Double_t bl2,
                                  Double_t tl2, Double_t alpha2)
{
// Make in one step a volume pointing to a trd2 shape with given material
   TGeoVolume *old = 0;
   old=(TGeoVolume*)fVolumes->FindObject(name);
   TGeoMaterial *mat = GetMaterial(material);
   if (!mat) {
      printf("%s\n", material);
      Warning("MakeTrap", "Material unknown");
      mat = GetMaterial("default");
   }
   TGeoGtra *gtra = new TGeoGtra(dz, theta, phi, twist, h1, bl1, tl1, alpha1, h2, bl2,
                                 tl2, alpha2);
   TGeoVolume *vol = new TGeoVolume(name, gtra, mat);
   if (old) vol->MakeCopyNodes(old);
   return vol;
}
//-----------------------------------------------------------------------------
TGeoVolumeMulti *TGeoManager::MakeVolumeMulti(const char *name, const char *material)
{
   TGeoMaterial *mat = GetMaterial(material);
   if (!mat) {
      printf("%s\n", material);
      Warning("MakeVolumeMulti", "Material unknown");
      mat = GetMaterial("default");
   }
   return (new TGeoVolumeMulti(name, mat));
}

//-----------------------------------------------------------------------------
void TGeoManager::SetExplodedView(UInt_t ibomb)
{
   // set type of exploding view
   GetGeomPainter();
   fPainter->SetExplodedView(ibomb);
}

//-----------------------------------------------------------------------------
void TGeoManager::SetNsegments(Int_t nseg)
{
// Set number of segments for approximating circles
   if (nseg < 3) return;
   TVirtualGeoPainter *painter = GetGeomPainter();
   if (painter) painter->SetNsegments(nseg);
}

//-----------------------------------------------------------------------------
Int_t TGeoManager::GetNsegments() const
{
// Get number of segments approximating circles
   TVirtualGeoPainter *painter = ((TGeoManager*)this)->GetGeomPainter();
   if (painter) return painter->GetNsegments();
   return 0;
}
//-----------------------------------------------------------------------------
void TGeoManager::ComputeGlobalMatrices(Option_t *option)
{   
// compute global matrices according to option
}

//-----------------------------------------------------------------------------
void TGeoManager::BuildDefaultMaterials()
{
// build the default materials. A list of those can be found in ...
   new TGeoMaterial("default", "Air", 14.61, 7.3, 0.001205);
}
//-----------------------------------------------------------------------------
TGeoNode *TGeoManager::Step(Bool_t is_geom, Bool_t cross)
{
// make a rectiliniar step of length fStep from current point (fPoint) on current
// direction (fDirection). If the step is imposed by geometry, is_geom flag
// must be true (default). The cross flag specifies if the boundary should be
// crossed in case of a geometry step (default true). Returns new node after step.
   Double_t epsil = 0;
   if (is_geom) {
      epsil=(cross)?1E-9:-1E-9;
      fIsEntering = cross;
      fIsExiting  = !cross;
   } else {
      fIsEntering = fIsExiting = kFALSE;
   }
   for (Int_t i=0; i<3; i++)
      fPoint[i]+=(fStep+epsil)*fDirection[i];
   return FindNode();
}
//-----------------------------------------------------------------------------
TGeoNode *TGeoManager::SamplePoints(Int_t npoints, Double_t &dist, Double_t epsil,
                                    const char* g3path)
{
// shoot npoints randomly in a box of 1E-5 arround current point.
// return minimum distance to points outside
   return GetGeomPainter()->SamplePoints(npoints, dist, epsil, g3path);
}
//-----------------------------------------------------------------------------
void TGeoManager::SetTopVolume(TGeoVolume *vol)
{
// set the top volume and corresponding node
   if (fTopVolume) fTopVolume->SetTitle("");
   fTopVolume = vol;
   vol->SetTitle("Top volume");
   if (fTopNode) delete fTopNode;
   else fMasterVolume = vol;
   fTopNode = new TGeoNodeMatrix(vol, gGeoIdentity);
   char *name = new char[strlen(vol->GetName()+2)];
   sprintf(name, "%s_1", vol->GetName());
   fTopNode->SetName(name);
   fTopNode->SetTitle("Top logical node");
   fCurrentNode = fTopNode;
   fNodes->AddAt(fTopNode, 0);
   fLevel = 0;
   *((TGeoHMatrix*)fGlobalMatrices->At(0)) = gGeoIdentity;
   if (fCache) {
      delete fCache;
      fCache = 0;
      BuildCache();
   }
   printf("Top volume is %s. Master volume is %s\n", fTopVolume->GetName(),
           fMasterVolume->GetName());
}
//-----------------------------------------------------------------------------
void TGeoManager::SelectTrackingMedia()
{
// define different tracking media
   printf("List of materials :\n");
   Int_t nmat = fMaterials->GetSize();
   if (!nmat) {printf(" No materials !\n"); return;}
   Int_t *media = new Int_t[nmat];
   memset(media, 0, nmat*sizeof(Int_t));
   Int_t imedia = 1;
   TGeoMaterial *mat, *matref;
   mat = (TGeoMaterial*)fMaterials->At(0);
   mat->SetMedia(imedia);
   media[0] = imedia++;
   mat->Print();
   for (Int_t i=0; i<nmat; i++) {
      mat = (TGeoMaterial*)fMaterials->At(i);
      for (Int_t j=0; j<i; j++) {
         matref = (TGeoMaterial*)fMaterials->At(j);
         if (mat->IsEq(matref)) {
            mat->SetMedia(media[j]);
            break;
         }
         if (j==(i-1)) {
         // different material
            mat->SetMedia(imedia);
            media[i] = imedia++;
            mat->Print();
         }
      }
   }
}
//-----------------------------------------------------------------------------
void TGeoManager::CheckPoint(Double_t x, Double_t y, Double_t z, Option_t *option)
{
   GetGeomPainter()->CheckPoint(x,y,z,option);
}
//-----------------------------------------------------------------------------
void TGeoManager::CheckGeometry(Option_t *option)
{
// instances a TGeoChecker object and investigates the geometry according to
// option
   // check shapes first
   fTopNode->CheckShapes();
}
//-----------------------------------------------------------------------------
void TGeoManager::UpdateCurrentPosition(Double_t *nextpoint)
{
// computes and changes the current node according to the new position
}
//-----------------------------------------------------------------------------
ULong_t TGeoManager::SizeOf(TGeoNode *node, Option_t *option)
{
// computes the total size in bytes of the branch starting with node.
// The option can specify if all the branch has to be parsed or only the node
   return 0;
}
