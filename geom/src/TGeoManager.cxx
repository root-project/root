/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// Author : Andrei Gheata - date

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
#include <fstream.h>
#include <iostream.h>

#include "TROOT.h"
#include "TSystem.h"
#include "TStyle.h"
#include "TPad.h"
#include "TView.h"
#include "TRandom3.h"
#include "TNtuple.h"
#include "TPolyMarker3D.h"
#include "TStopwatch.h"
#include "TDatime.h"
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
#include "TGeoVolume.h"
#include "TGeoNode.h"
#include "TGeoFinder.h"
#include "TGeoCache.h"
#include "TVirtualGeoPainter.h"

#include "TGeoManager.h"

// statics and globals

TGeoManager *gGeoManager = 0;
TGeoNodeCache *gGeoNodeCache = 0;

Int_t TGeoManager::kGeoDefaultNsegments = 20;
Int_t TGeoManager::kGeoVisLevel = 3;
Double_t *TGeoManager::kGeoSinTable = 0;
Double_t *TGeoManager::kGeoCosTable = 0;
const char *TGeoManager::kGeoOutsidePath = " ";

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
   fRandomBox = 0;
   fGlobalMatrices = 0;
   fNodes = 0;
   fNNodes = 0;
   fVolumes = 0;
   fShapes = 0;
   fTopVolume = 0;
   fNsegments = 0;
   fSegStep = 0;
   fTopNode = 0;
   fCurrentVolume = 0;
   fMasterVolume = 0;
   fCurrentNode = 0;
   fLastNode = 0;
   fPath = "";
   fCache = 0;
   fLevel = 0;
   fVisLevel = kGeoVisLevel;
   fVisOption = kGeoVisDefault;
   fExplodedView = 0;
   fBombX = 0;
   fBombY = 0;
   fBombZ = 0;
   fBombR = 0;
   fVisBranch = "";
   fIgnored = 0;
   fNignored = 0;
   fILevel = 0;
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
   fIsOutside  = kFALSE;
   fExplodedView = 0;
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
   fVisLevel = kGeoVisLevel;
   fVisOption = kGeoVisDefault;
   fVisBranch = "";
   fExplodedView = TGeoManager::kGeoNoBomb;
   fBombX = 1.3;
   fBombY = 1.3;
   fBombZ = 1.3;
   fBombR = 1.3;
   fIgnored = new Int_t[10];
   fNignored = 0;
   fILevel = 0;
   fPoint = new Double_t[3];
   fDirection = new Double_t[3];
   fNormalChecked = new Double_t[3];
   fCldirChecked = new Double_t[3];
   fNormal = new Double_t[3];
   fCldir = new Double_t[3];
   fRandomBox = new Double_t[6];
   fVolumes = new TList();
   fShapes = new TList();
   fGVolumes = new TList();
   fGShapes = new TList();
   fTopVolume = 0;
   SetNsegments(10000);
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
   delete [] fRandomBox;
   delete [] kGeoSinTable;
   delete [] kGeoCosTable;
   delete fGVolumes;
   delete fGShapes;
   kGeoSinTable = 0;
   kGeoCosTable = 0;
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
   memcpy(bombtr, tr, 3*sizeof(Double_t));
   switch (fExplodedView) {
      case kGeoNoBomb:
         return;
      case kGeoBombXYZ:
         bombtr[0] *= fBombX;
         bombtr[1] *= fBombY;
         bombtr[2] *= fBombZ;
         return;
      case kGeoBombCyl:
         bombtr[0] *= fBombR;      
         bombtr[1] *= fBombR;      
         bombtr[2] *= fBombZ;
         return;
      case kGeoBombSph:
         bombtr[0] *= fBombR;      
         bombtr[1] *= fBombR;      
         bombtr[2] *= fBombR;
         return;
      default:
         return;      
   }
}
//-----------------------------------------------------------------------------
void TGeoManager::UnbombTranslation(const Double_t *tr, Double_t *bombtr)
{
// get the new 'unbombed' translation vector according current exploded view mode 
   memcpy(bombtr, tr, 3*sizeof(Double_t));
   switch (fExplodedView) {
      case kGeoNoBomb:
         return;
      case kGeoBombXYZ:
         bombtr[0] /= fBombX;
         bombtr[1] /= fBombY;
         bombtr[2] /= fBombZ;
         return;
      case kGeoBombCyl:
         bombtr[0] /= fBombR;      
         bombtr[1] /= fBombR;      
         bombtr[2] /= fBombZ;
         return;
      case kGeoBombSph:
         bombtr[0] /= fBombR;      
         bombtr[1] /= fBombR;      
         bombtr[2] /= fBombR;
         return;
      default:
         return;      
   }
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
   gGeoNodeCache->CdTop();
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
   gGeoNodeCache->CdUp();
   if (fCurrentOverlapping) fLastNode = fCurrentNode;
   fCurrentNode = gGeoNodeCache->GetNode();
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
   gGeoNodeCache->CdDown(index);
   fCurrentNode = node;
   fLevel++; 
}     
//-----------------------------------------------------------------------------
Bool_t TGeoManager::cd(const char *path)
{
// browse the tree of nodes starting from fTopNode according to pathname
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
//         printf("current vol : %s\n", vol->GetName());
         node = vol->GetNode(name.Data());
//         printf("current node : %s\n", node->GetName());
      } else node = 0;   
      if (!node) {
         Error("cd", "path not valid");
         return kFALSE;
      }
      CdDown(fCurrentNode->GetVolume()->GetIndex(node));
      ind1 = ind2;
   }
   return kTRUE;
//   printf("CurrentPath : %s\n", GetPath());
}
//-----------------------------------------------------------------------------
Int_t TGeoManager::CountNodes(TGeoVolume *vol, Int_t nlevels)
{
// count the total number of nodes starting from a volume, nlevels down
   TGeoVolume *top;
   if (!vol) {
      top = fTopVolume;
   } else {     
      top = vol;
   }
   Int_t count = top->CountNodes(nlevels);
//   printf("Number of physical nodes in %s %i levels down : %i\n", GetName(), nlevels,  count);
   return count;
}
//-----------------------------------------------------------------------------
void TGeoManager::DefaultAngles()
{
// set default angles for a given view
   if (gPad) {
      Int_t irep;
      TView *view = gPad->GetView();
      if (!view) return;
      view->SetView(-206,126,75,irep);
      gPad->Modified();
      gPad->Update();
   }   
}
//-----------------------------------------------------------------------------
void TGeoManager::Down(Int_t id)
{
// go down one level to daughter id of current node
   fCurrentNode = fCurrentNode->GetDaughter(id);
   fNodes->AddAt(fCurrentNode, ++fLevel);
   // now compute the matrix
   TGeoHMatrix *gmat  = (TGeoHMatrix*)fGlobalMatrices->At(fLevel);
   *gmat = (TGeoHMatrix*)fGlobalMatrices->At(fLevel-1);
   gmat->Multiply(fCurrentNode->GetMatrix());
}
//-----------------------------------------------------------------------------
void TGeoManager::DrawCurrentPoint(Int_t color)
{
// draw current point in the same view
   TPolyMarker3D *pm = new TPolyMarker3D();
   pm->SetMarkerColor(color);
   pm->SetNextPoint(fPoint[0], fPoint[1], fPoint[2]);
   pm->SetMarkerStyle(2);
   pm->SetMarkerSize(2);
   pm->Draw("SAME");
}
//-----------------------------------------------------------------------------
void TGeoManager::DrawPoint(Double_t x, Double_t y, Double_t z)
{
// draw a point in the context of the geom
   RestoreMasterVolume();
   if ((x!=0) || (y!=0) || (z!=0)) SetCurrentPoint(x,y,z);
   TGeoNode *node = FindNode();
   if (!node) return;
   printf("Path for point : %f %f %f LEVEL=%i:\n", x,y,z, fLevel);
   printf("    %s\n", GetPath());
   CheckPoint(x,y,z);
   Int_t nd = node->GetNdaughters();
   if (!nd) CdUp();
   node->GetVolume()->VisibleDaughters(kFALSE);
   TGeoNode *node1;
   for (Int_t i=0; i<nd; i++) {
      node1 = node->GetDaughter(i);
      node1->GetVolume()->SetVisibility(kTRUE);
   }   
   Double_t point[3];
   Double_t current[3];
   memcpy(&point[0], fPoint, 3*sizeof(Double_t));
   MasterToLocal(fPoint, &current[0]);
   memcpy(fPoint, &current[0], 3*sizeof(Double_t));
   GetCurrentVolume()->Draw();
   DrawCurrentPoint(2);
   memcpy(fPoint, &point[0], 3*sizeof(Double_t));
   
//   node->GetVolume()->SetLineColor(2);
//      node = fCurrentNode;
//      node->Draw("");
//      node->CheckPoint();
}
//-----------------------------------------------------------------------------
void TGeoManager::DrawPoints(TGeoVolume *vol, Int_t npoints, Option_t *option)
{
// draw points in the bbox of a volume using FindNode
   if (!vol) {
      if (fTopVolume != fMasterVolume) RestoreMasterVolume();
      vol = fMasterVolume;
   }
   TGeoVolume *old_vol = 0;
   if (vol != fTopVolume) {
      old_vol = fTopVolume;
      SetTopVolume(vol);
   }
   TString opt = option;
   opt.ToLower();
   Double_t xmin = 1000000;
   Double_t xmax = -1000000;
   Double_t ymin = 1000000;
   Double_t ymax = -1000000;
   Double_t zmin = 1000000;
   Double_t zmax = -1000000;
   TObjArray *pm = new TObjArray(128);
   TPolyMarker3D *marker = 0;
   TPolyMarker3D *omarker = new TPolyMarker3D();
   omarker->SetMarkerColor(2);
   omarker->SetMarkerStyle(3);
   TNtuple *ntpl = new TNtuple("ntpl","random points","x:y:z");
   gRandom = new TRandom3();
   const TGeoShape *shape = vol->GetShape();
   Double_t dx = ((TGeoBBox*)shape)->GetDX();
   Double_t dy = ((TGeoBBox*)shape)->GetDY();
   Double_t dz = ((TGeoBBox*)shape)->GetDZ();
   Double_t ox = (((TGeoBBox*)shape)->GetOrigin())[0];
   Double_t oy = (((TGeoBBox*)shape)->GetOrigin())[1];
   Double_t oz = (((TGeoBBox*)shape)->GetOrigin())[2];
   if (fRandomBox[1]!=0) {
      ox = fRandomBox[0];
      dx = fRandomBox[1];
      oy = fRandomBox[2];
      dy = fRandomBox[3];
      oz = fRandomBox[4];
      dz = fRandomBox[5];
   }
   omarker->SetNextPoint(ox,oy,oz);
   Double_t *xyz = new Double_t[3];
   TStopwatch *timer = new TStopwatch();
   printf("Random box : %f, %f, %f\n", dx, dy, dz);
   TGeoNode *node;
   printf("Start... %i points\n", npoints);
   Int_t i=0;
   Int_t igen=0;
   Int_t ic = 0;
   Double_t ratio=0;
   timer->Start(kFALSE);
   while (i<npoints) {
      xyz[0] = ox-dx+2*dx*gRandom->Rndm();
      xyz[1] = oy-dy+2*dy*gRandom->Rndm();
      xyz[2] = oz-dz+2*dz*gRandom->Rndm();
      SetCurrentPoint(xyz);
      igen++;

//      i++; /// to remove
      if ((igen%1000)==0) {
         ratio = (Double_t)i/(Double_t)igen;
         if (ratio<0.00001) break;
      }
      node = FindNode();
      if (!node) continue;
      if (!node->IsVisible()) continue;
      // draw only points in overlapping volumes
      if (opt.Contains("many") && !node->IsOverlapping()) continue;
      if (opt.Contains("only") && node->IsOverlapping()) continue;
      ic = node->GetColour();
      if (ic >= 128) ic = 0;
      marker = (TPolyMarker3D*)pm->At(ic);
      if (!marker) {
         marker = new TPolyMarker3D();
         marker->SetMarkerColor(ic);
         pm->AddAt(marker, ic);
      }
      if (vol->GetNdaughters()==0) {
         xmin = TMath::Min(xmin, xyz[0]);
         xmax = TMath::Max(xmax, xyz[0]);
         ymin = TMath::Min(ymin, xyz[1]);
         ymax = TMath::Max(ymax, xyz[1]);
         zmin = TMath::Min(zmin, xyz[2]);
         zmax = TMath::Max(zmax, xyz[2]);   
//         ntpl->Fill(xyz[0], xyz[1], xyz[2]);
         marker->SetNextPoint(xyz[0], xyz[1], xyz[2]);
         i++;
         continue;
      }
      if ((fLevel==0)) continue;
      xmin = TMath::Min(xmin, xyz[0]);
      xmax = TMath::Max(xmax, xyz[0]);
      ymin = TMath::Min(ymin, xyz[1]);
      ymax = TMath::Max(ymax, xyz[1]);
      zmin = TMath::Min(zmin, xyz[2]);
      zmax = TMath::Max(zmax, xyz[2]);   
//      ntpl->Fill(xyz[0], xyz[1], xyz[2]);
      marker->SetNextPoint(xyz[0], xyz[1], xyz[2]);
      if ((i%1000) == 0) printf("%i\n", i);
      i++; 
      ratio = (Double_t)i/(Double_t)igen;
   }
   timer->Stop();
   timer->Print();
   printf("ratio : %g  volume of inner structures for %s : %g[cm3]\n", 
           ratio, vol->GetName(), ratio*dx*dy*dz);
   ntpl->Fill(xmin,ymin,zmin);
   ntpl->Fill(xmax,ymin,zmin);
   ntpl->Fill(xmin,ymax,zmin);
   ntpl->Fill(xmax,ymax,zmin);
   ntpl->Fill(xmin,ymin,zmax);
   ntpl->Fill(xmax,ymin,zmax);
   ntpl->Fill(xmin,ymax,zmax);
   ntpl->Fill(xmax,ymax,zmax);
   ntpl->Draw("z:y:x");
   for (Int_t m=0; m<128; m++) {
      marker = (TPolyMarker3D*)pm->At(m);
      if (marker) marker->Draw("SAME");
   }
   omarker->Draw("SAME");
   if (gPad) gPad->Update();
   delete ntpl;
//   pm->Delete();
   delete pm;
   delete xyz;
   delete timer;
   if (old_vol) SetTopVolume(old_vol);
}
//-----------------------------------------------------------------------------
void TGeoManager::Test(Int_t npoints, Option_t *option)
{
   gRandom= new TRandom3();
   Bool_t recheck = !strcmp(option, "RECHECK");
   if (recheck) printf("RECHECK\n");
   const TGeoShape *shape = fTopVolume->GetShape();
   Double_t dx = ((TGeoBBox*)shape)->GetDX();
   Double_t dy = ((TGeoBBox*)shape)->GetDY();
   Double_t dz = ((TGeoBBox*)shape)->GetDZ();
   Double_t *xyz = new Double_t[3*npoints];
   TStopwatch *timer = new TStopwatch();
   printf("Random box : %f, %f, %f\n", dx, dy, dz);
   timer->Start(kFALSE);
   for (Int_t i=0; i<npoints; i++) {
      xyz[3*i] = -dx+2*dx*gRandom->Rndm();
      xyz[3*i+1] = -dy+2*dy*gRandom->Rndm();
      xyz[3*i+2] = -dz+2*dz*gRandom->Rndm();
   }
   timer->Stop();
   printf("Generation time :\n");
   timer->Print();
   timer->Reset();
   TGeoNode *node, *node1;
   printf("Start... %i points\n", npoints);
   timer->Start(kFALSE);
   for (Int_t i=0; i<npoints; i++) {
      SetCurrentPoint(xyz+3*i);
      if (recheck) CdTop();
      node = FindNode();
      if (recheck) {
         node1 = FindNode();
         if (node1 != node) {
            printf("Difference for x=%g y=%g z=%g\n", xyz[3*i], xyz[3*i+1], xyz[3*i+2]);
            printf(" from top : %s\n", node->GetName());
            printf(" redo     : %s\n", GetPath());
         }
      }
   }
   timer->Stop();
   timer->Print();
   delete xyz;
   delete timer;
}
//-----------------------------------------------------------------------------
void TGeoManager::TestOverlaps(const char* path)
{
   if (fTopVolume!=fMasterVolume) RestoreMasterVolume();
   printf("Checking overlaps for path :\n");
   cd(path);
   TGeoNode *checked = fCurrentNode;
   checked->InspectNode();
   // shoot 1E4 points in the shape of the current volume
   gRandom= new TRandom3();
   Int_t npoints = 1000000;
   Double_t big = 1E6;
   Double_t xmin = big;
   Double_t xmax = -big;
   Double_t ymin = big;
   Double_t ymax = -big;
   Double_t zmin = big;
   Double_t zmax = -big;
   TObjArray *pm = new TObjArray(128);
   TPolyMarker3D *marker = 0;
   TPolyMarker3D *markthis = new TPolyMarker3D();
   markthis->SetMarkerColor(5);
   TNtuple *ntpl = new TNtuple("ntpl","random points","x:y:z");
   TGeoShape *shape = fCurrentNode->GetVolume()->GetShape();
   Double_t *point = new Double_t[3];
   Double_t dx = ((TGeoBBox*)shape)->GetDX();
   Double_t dy = ((TGeoBBox*)shape)->GetDY();
   Double_t dz = ((TGeoBBox*)shape)->GetDZ();
   Double_t ox = (((TGeoBBox*)shape)->GetOrigin())[0];
   Double_t oy = (((TGeoBBox*)shape)->GetOrigin())[1];
   Double_t oz = (((TGeoBBox*)shape)->GetOrigin())[2];
   Double_t *xyz = new Double_t[3*npoints];
   Int_t i=0;
   printf("Generating %i points inside %s\n", npoints, GetPath());
   while (i<npoints) {
      point[0] = ox-dx+2*dx*gRandom->Rndm();
      point[1] = oy-dy+2*dy*gRandom->Rndm();
      point[2] = oz-dz+2*dz*gRandom->Rndm();
      if (!shape->Contains(point)) continue;
      // convert each point to MARS
//      printf("local  %9.3f %9.3f %9.3f\n", point[0], point[1], point[2]);
      LocalToMaster(point, &xyz[3*i]);
//      printf("master %9.3f %9.3f %9.3f\n", xyz[3*i], xyz[3*i+1], xyz[3*i+2]);
      xmin = TMath::Min(xmin, xyz[3*i]);
      xmax = TMath::Max(xmax, xyz[3*i]);
      ymin = TMath::Min(ymin, xyz[3*i+1]);
      ymax = TMath::Max(ymax, xyz[3*i+1]);
      zmin = TMath::Min(zmin, xyz[3*i+2]);
      zmax = TMath::Max(zmax, xyz[3*i+2]);
      i++;
   }
   delete point;
   ntpl->Fill(xmin,ymin,zmin);
   ntpl->Fill(xmax,ymin,zmin);
   ntpl->Fill(xmin,ymax,zmin);
   ntpl->Fill(xmax,ymax,zmin);
   ntpl->Fill(xmin,ymin,zmax);
   ntpl->Fill(xmax,ymin,zmax);
   ntpl->Fill(xmin,ymax,zmax);
   ntpl->Fill(xmax,ymax,zmax);
   ntpl->Draw("z:y:x");
   
   // shoot the poins in the geometry
   TGeoNode *node;
   TString cpath;
   Int_t ic=0;
   TObjArray *overlaps = new TObjArray();
   printf("using FindNode...\n");
   for (Int_t j=0; j<npoints; j++) {
      // always start from top level (testing only)
      CdTop();
      SetCurrentPoint(&xyz[3*j]);
      node = FindNode();
      cpath = GetPath();
      if (cpath.Contains(path)) {
         markthis->SetNextPoint(xyz[3*j], xyz[3*j+1], xyz[3*j+2]);
         continue;
      }
      // current point is found in an overlapping node
      if (!node) ic=128;
      else ic = node->GetColour();
      if (ic >= 128) ic = 0;
      marker = (TPolyMarker3D*)pm->At(ic);
      if (!marker) {
         marker = new TPolyMarker3D();
         marker->SetMarkerColor(ic);
         pm->AddAt(marker, ic);
      }
      // draw the overlapping point
      marker->SetNextPoint(xyz[3*j], xyz[3*j+1], xyz[3*j+2]);
      if (node) { 
         if (overlaps->IndexOf(node) < 0) overlaps->Add(node);
      }
   }
   // draw all overlapping points
   for (Int_t m=0; m<128; m++) {
      marker = (TPolyMarker3D*)pm->At(m);
//      if (marker) marker->Draw("SAME");
   }
   markthis->Draw("SAME");
   if (gPad) gPad->Update();
   // display overlaps 
   if (overlaps->GetEntriesFast()) {
      printf("list of overlapping nodes :\n");
      for (i=0; i<overlaps->GetEntriesFast(); i++) {
         node = (TGeoNode*)overlaps->At(i);
         if (node->IsOverlapping()) printf("%s  MANY\n", node->GetName());
         else printf("%s  ONLY\n", node->GetName());
      }
   } else printf("No overlaps\n");
   delete ntpl;
   delete pm;
   delete xyz;
   delete overlaps;
}
//-----------------------------------------------------------------------------
Int_t TGeoManager::GetVirtualLevel()
{
// find level of virtuality of current overlapping node (number of levels
// up having the same tracking media
   // return if the current node is ONLY
   if (!fCurrentOverlapping) return 0;
//   Bool_t is_overlapping = kTRUE;
//   Bool_t is_offset = fCurrentNode->IsOffset();
//   if (!is_overlapping && !is_offset) return 0;
   //Int_t last_many = 0;
   Int_t new_media = 0;
   Int_t imedia = fCurrentNode->GetMedia();
   Int_t virtual_level = 1;
   TGeoNode *mother = 0;

   while ((mother=GetMother(virtual_level))) {
      if (!mother->IsOverlapping() && !mother->IsOffset()) {
         if (!new_media) new_media=(mother->GetMedia()==imedia)?0:virtual_level;
         break;
      }   
      //if (mother->IsOverlapping()) 
         //last_many=virtual_level;
      if (!new_media) new_media=(mother->GetMedia()==imedia)?0:virtual_level;
      virtual_level++;
   }  
//   if (last_many<0) return 0;
   return (new_media==0)?virtual_level:(new_media-1);    
}   
//-----------------------------------------------------------------------------
Bool_t TGeoManager::GotoSafeLevel()
{
// go upwards the tree until a non-overlaping node
/*
   Bool_t is_overlapping = fCurrentNode->IsOverlapping();
   Bool_t is_offset = fCurrentNode->IsOffset();
   if (!is_overlapping && !is_offset) return kFALSE;
   Int_t last_many = (is_overlapping)?0:-1;
   Int_t virtual_level = 1;
   TGeoNode *mother = 0;

   while ((mother=GetMother(virtual_level))) {
      if (!mother->IsOverlapping() && !mother->IsOffset()) 
         break;
      if (mother->IsOverlapping()) 
         last_many=virtual_level;
      virtual_level++;
   }  
   if (last_many<0) return kFALSE;
   while (virtual_level--) CdUp();
*/
   while (fCurrentOverlapping && fLevel) CdUp();
   return kTRUE;
} 
//-----------------------------------------------------------------------------
TGeoNode *TGeoManager::FindInCluster(Int_t *cluster, Int_t nc)
{
// find a node inside a cluster of overlapping nodes. Current node must
// be on top of all the nodes in cluster. Always nc>1
//   printf("Start findcluster   nc=%i\n", nc);
   TGeoNode *clnode = 0;
   TGeoNode *priority = fLastNode;
//   if (priority) printf("we are in %s priority : %s\n", fCurrentNode->GetName(),priority->GetName());
   // save current node
   TGeoNode *current = fCurrentNode;
   TGeoNode *found = 0;
   // save path
   Int_t ipop = PushPath();
//   printf("pushed ipop=%i\n", ipop);
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
//      printf("Searching cluster node : %s\n", clnode->GetName());
      found = SearchNode(kTRUE, clnode);
      if (!fSearchOverlaps) {
//         printf(" cluster search STOPPED\n");
      // an only was found during the search -> exiting
         PopDummy(ipop);
//         printf("Final node in cluster: %s\n", found->GetName());
         return found;
      }   
//      printf("node found in cluster: %s\n", found->GetName());
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
//         if (replace) printf("...will become current candidate\n");         
         // if this was the last checked node
         if (i==(nc-1)) {
            if (replace) {
               PopDummy(ipop);
//               printf("Final node in cluster: %s\n", found->GetName());
               return found;
            } else {
               fCurrentOverlapping = PopPath();
               PopDummy(ipop);
//               printf("Final node in cluster: %s\n", fCurrentNode->GetName());
               return fCurrentNode;
            }      
         }
         // we still have to go on
         if (replace) {
            // reset stack
            PopDummy();
            PushPath();
//            printf("popped old found, pushed new one, ipop=%i stack=%i\n", ipop,GetStackLevel());
            deepest = fLevel;
            deepest_virtual = found_virtual;
         }
         // restore top of cluster
         fCurrentOverlapping = PopPath(ipop);
//         printf("popped ipop=%i\n", ipop);
//         continue;   
      } else {
      // the stack was clean, push new one
         PushPath();
//         printf("pushed new found ipop=%i stack=%i\n", ipop, GetStackLevel());
         added = kTRUE;
         deepest = fLevel;
         deepest_virtual = found_virtual;           
         // restore original path
         fCurrentOverlapping = PopPath(ipop);
//         printf("popped ipop=%i\n", ipop);
      }      
   }
//   printf("woops - abnormal return\n");
   PopDummy(ipop);
//   printf("Final node in cluster: %s\n", fCurrentNode->GetName());
   return fCurrentNode;
}   
//-----------------------------------------------------------------------------
Int_t TGeoManager::GetTouchedCluster(Int_t start, Double_t *point, 
                              Int_t *check_list, Int_t ncheck, Int_t *result)
{
// make the cluster of overlapping nodes in a voxel, containing point in reference
// of the mother. Returns number of nodes containing the point. Nodes should not be
// offsets
   // we are in the mother reference system
//   printf("GetTouchedCluster start, start=%i ncheck=%i\n", start, ncheck);
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
//      printf("CLUSTER node : %s\n", current->GetName());
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
//               printf("CLUSTER node : %s\n", current->GetName());
            }  
         }  
      }
   }          
   return ntotal;
}   
//-----------------------------------------------------------------------------
void TGeoManager::DefaultColors()
{
// Set default volume colors according to tracking media
   TIter next(fVolumes);
   TGeoVolume *vol;
   while ((vol=(TGeoVolume*)next()))
      vol->SetLineColor(vol->GetMaterial()->GetDefaultColor());
   if (gPad) {
      if (gPad->GetView()) {
         gPad->Modified();
         gPad->Update();
      }
   }         
}
//-----------------------------------------------------------------------------
void TGeoManager::SetVisOption(Int_t option) {
// set drawing mode :
// option=0 (default) all nodes drawn down to vislevel
// option=1           leaves and nodes at vislevel drawn
// option=2           path is drawn
   fVisOption=option;
   if (!gPad) return;
   if (gPad->GetView()) {
      gPad->Modified();
      gPad->Update();
   }
}      
//-----------------------------------------------------------------------------
void TGeoManager::SetVisLevel(Int_t level) {
// set default level down to which visualization is performed
   fVisLevel=level;
   if (!gPad) return;
   if (gPad->GetView()) {
      gPad->Modified();
      gPad->Update();
   }
}      
//-----------------------------------------------------------------------------
void TGeoManager::SaveAttributes(const char *filename)
{
// save current attributes in a macro
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
   out << "   // clear all volume attributes"<<endl;
   out << "   gGeoManager->ClearAttributes();"<<endl;
   out << "   // set visualization modes and bomb factors"<<endl;
   out << "   gGeoManager->SetVisOption("<<gGeoManager->GetVisOption()<<");"<<endl;
   out << "   gGeoManager->SetVisLevel("<<gGeoManager->GetVisLevel()<<");"<<endl;
   out << "   gGeoManager->SetExplodedView("<<gGeoManager->GetBombMode()<<");"<<endl;
   out << "   gGeoManager->SetBombFactors("<<fBombX<<","<<fBombY<<","<<fBombZ<<","<<fBombR<<");"<<endl;
   out << "   // iterate volumes coontainer and set new attributes"<<endl;
//   out << "   TIter next(gGeoManager->GetListOfVolumes());"<<endl;
   TGeoVolume *vol = 0;
   fTopNode->SaveAttributes(out);

   TIter next(fVolumes);
   while ((vol=(TGeoVolume*)next())) {
      vol->SetVisStreamed(kFALSE);
/*
      if (vol->IsStyleDefault()) continue;
      out << "   vol = gGeoManager->GetVolume("<<quote<<vol->GetName()<<quote<<");"<<endl;
      if (!vol->IsVisible()) 
         out << "   vol->SetVisibility(kFALSE);"<<endl;
      if (vol->GetLineColor() != gStyle->GetLineColor())   
         out << "   vol->SetLineColor("<<vol->GetLineColor()<<");"<<endl;
      if (vol->GetLineStyle() != gStyle->GetLineStyle())   
         out << "   vol->SetLineStyle("<<vol->GetLineStyle()<<");"<<endl;
      if (vol->GetLineWidth() != gStyle->GetLineWidth())   
         out << "   vol->SetLineWidth("<<vol->GetLineWidth()<<");"<<endl;
*/
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
// returns the deepest node containing fPoint, which must be set a priori
   Double_t point[3];
   TGeoVolume *vol = 0;
   Bool_t inside_current = kFALSE;
   if (!downwards) {
   // we are looking upwards until inside current node or exit
//      printf("Searching upwards : %s\n", GetPath());
      if (fStartSafe) GotoSafeLevel();
      vol=fCurrentNode->GetVolume();
      MasterToLocal(fPoint, &point[0]);
      inside_current = vol->Contains(&point[0]);
      if (!inside_current) {
//         printf("NOT here -> go up\n");
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
//      printf("looking downwards in : %s\n", GetPath());
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
//   printf("Point is inside\n");
//   printf("current safe node : %s\n", fCurrentNode->GetName());
   // point inside current (safe) node -> search downwards   
   TGeoNode *node;
   Int_t ncheck = 0;
   // if inside an non-overlapping node, reset overlap searches
   if (!fCurrentOverlapping) {
//      printf("current node is ONLY -> reset ovlp srch. : %s\n", fCurrentNode->GetName());
      fSearchOverlaps = kFALSE;
//      printf("=== overlap search reset ===\n");
   } //else printf("current node is MANY\n");  

   Int_t nd = vol->GetNdaughters();
   // in case there are no daughters
   if (!nd) return fCurrentNode;

   TGeoPatternFinder *finder = vol->GetFinder();
   // point is inside the current node
//   gGeoNodeCache->IncrementUsageCount();
   // first check if inside a division
   if (finder) {
//      printf("Current node divided\n");
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
//      printf("VOXELS. Ncandidates=%i\n", ncheck);
      // if none in voxel, see if this is the last one
      if (!check_list) return fCurrentNode;
      // loop all nodes in voxel
      for (Int_t id=0; id<ncheck; id++) {
         node = vol->GetNode(check_list[id]);
         if (node==skipnode) continue;
//         printf("Searching in node %s\n", node->GetName());
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
//         printf("     down %i\n", check_list[id]);
         CdDown(check_list[id]);
         node = SearchNode(kTRUE);
         if (node) return node;
//         printf("NOT HERE\n");
         CdUp();
      }
      return fCurrentNode;
   }
   // if there are no voxels just loop all daughters
   Int_t id = 0;
//   printf("woops --- no voxels\n");
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
//   printf("master p: %f %f %f  local p:%f %f %f\n", fPoint[0], fPoint[1], fPoint[2], point[0], point[1], point[2]);
//   printf("master d: %f %f %f  local d:%f %f %f\n", fDirection[0], fDirection[1], fDirection[2], dir[0], dir[1], dir[2]);
   // compute distance to exit point from current node and the distance to its 
   // closest boundary
   TGeoVolume *vol = fCurrentNode->GetVolume();
   // if point is outside, just check the top node
   if (fIsOutside) {
      fStep = vol->GetShape()->DistToOut(&point[0], &dir[0], 3, TGeoShape::kBig, &fSafety);
      return fTopNode;
   }   
//   printf("shape is : %s\n", vol->GetShape()->ClassName());
   if (fIsEntering || fIsExiting) {
      fStep = vol->GetShape()->DistToOut(&point[0], &dir[0], 3, TGeoShape::kBig, &fSafety);
      if (fIsExiting) return fCurrentNode;
   } else {
      fStep = vol->GetShape()->DistToOut(&point[0], &dir[0], 2, TGeoShape::kBig, &fSafety);
//      printf("step to out : %f\n", fStep);
   }   
   // get number of daughters. If no daughters we are done.
   Int_t nd = vol->GetNdaughters();
//   printf("Vol : %s number of daughters : %i\n", vol->GetName(),nd);
//   printf("TO OUT : %f\n", fStep);
   if (!nd) return fCurrentNode;
   TGeoNode *current = 0;
   TGeoNode *clnode = fCurrentNode;
   Double_t lpoint[3];
   Double_t ldir[3];
   Double_t safety = TGeoShape::kBig;
   Double_t snext  = TGeoShape::kBig;
   Int_t i=0;
   // if only one daughter, check it and exit
/*
   if (nd==1) {
      current = vol->GetNode(0);
      current->cd();
      current->MasterToLocal(&point[0], &lpoint[0]);
      current->MasterToLocalVect(&dir[0], &ldir[0]);
      snext = current->GetVolume()->GetShape()->DistToIn(&lpoint[0], &ldir[0], 2, TGeoShape::kBig, &safety);
      fSafety = TMath::Min(fSafety, safety);
      if (snext<fStep) {
         fStep=snext;
         return current;
      }
      return clnode;   
   }
*/
   if (nd<3) {
//      printf("LOOPING\n");
      for (i=0; i<nd; i++) {
         current = vol->GetNode(i);
         current->cd();
         current->MasterToLocal(&point[0], &lpoint[0]);
         current->MasterToLocalVect(&dir[0], &ldir[0]);
         snext = current->GetVolume()->GetShape()->DistToIn(&lpoint[0], &ldir[0], 2, TGeoShape::kBig, &safety);
//         printf(" checking %s : %f\n", current->GetName(), snext);
         fSafety = TMath::Min(fSafety, safety);
         if (snext<fStep) {
            fStep=snext;
            clnode = current;
//            printf("CLOSER: %s AT %f\n", clnode->GetName(), fStep);
         }
      }
   return clnode;   
   }         
   // if current volume is divided, we are in the non-divided region. We
   // check only the first and the last cell
   TGeoPatternFinder *finder = vol->GetFinder();
   if (finder) {
//      printf("DIVIDED\n");
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
//         printf("CLOSER: %s AT %f\n", clnode->GetName(), fStep);
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
//         printf("CLOSER: %s AT %f\n", current->GetName(), fStep);
         return current;
      }
   }
   // if current volume is voxelized, first get current voxel
   TGeoVoxelFinder *voxels = vol->GetVoxels();
   if (voxels) {
//      printf("VOXELS\n");
      Int_t ncheck = 0;
      Int_t *vlist = 0;   
//      printf("Entering voxels of %s, path=%s\n", vol->GetName(), GetPath());  
      voxels->SortCrossedVoxels(&point[0], &dir[0]);
      Bool_t first = kTRUE;
//      Bool_t end = kFALSE;
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
//            printf(" checking %s : %f\n", current->GetName(), snext);
            if (snext<fStep) {
               fStep=snext;
//               end = kTRUE;
               clnode = current;
//               printf("CLOSER: %s AT %f\n", current->GetName(), fStep);
            }   
         }
         first=kFALSE;
//         if (end) return clnode;
      }
   }         
   // if no voxels and more than 1 daughter (should never happen) just
   // loop on volumes
   return clnode;
}
//-----------------------------------------------------------------------------
TGeoNode *TGeoManager::FindNodeLast(Bool_t downwards)
{
// we have reached the deepest node containing the point, now
// look if this is the final solution

   // first check if current node is declared overapping
   if (!fCurrentNode->IsOverlapping()) {
      ResetIgnored();
      return fCurrentNode;
   }   
   // now check if current node might have real overlaps
   if (fILevel != (fLevel-1)) ResetIgnored();
   TGeoNode *node = FindNodeOverlap();
   // if something was found this is the final solution
   if (node) return node;
   // if we were looking downwards we can skip stage 2
   if (downwards) return fCurrentNode;
   // if the mother of this is not overlapping, current node is the solution
   if (!GetMother()->IsOverlapping()) return fCurrentNode;
   // mother is also overapping node, push current path and go up
   PushPath();
   CdUp();
   while (fLevel>0) {
      node = FindNodeOverlap();
      if (node) {
         PopDummy();
         ResetIgnored();
         return node;
      }
      if (!GetMother()->IsOverlapping()) {
         PopPath();
         ResetIgnored();
         return fCurrentNode;
      }
      CdUp();
   }
   PopPath();
   return fCurrentNode;
}
//-----------------------------------------------------------------------------
TGeoNode *TGeoManager::FindNodeOverlap()
{
   // point is in current node, which might overlap with other nodes
   // from the same level
//   printf("Checking OVERLAPS for %s\n", fCurrentNode->GetName());

   Int_t novlps = 0;
   // if no real overlaps with other bboxes, return NULL
   Int_t *ovlps = fCurrentNode->GetOverlaps(novlps);
   if (!ovlps) return 0;
   TGeoVolume *vol = fCurrentNode->GetMotherVolume();
//   if (!vol) return fCurrentNode;
   // if division, return
//   TGeoPatternFinder *finder = vol->GetFinder();
//   if (finder) return fCurrentNode;
   TGeoVoxelFinder *voxels = vol->GetVoxels();
   if (!voxels) return 0;
   Double_t point[3];
   Int_t *check_list = 0;
   Int_t ncheck = 0; 
   TGeoNode *last_node = fCurrentNode;
   Int_t last_level = fLevel;
   PushPath();
   CdUp();
   MasterToLocal(fPoint, &point[0]);
   // check if we are still in the mother volume or in protruding MANY
   if (!vol->GetShape()->Contains(point)) {
//      printf("WOOPS - protruding MANY\n");
      ResetIgnored();
      CdUp();
      PopDummy();
      return SearchNode();
   }
   check_list = voxels->GetCheckList(&point[0], ncheck);
   if (!ncheck) {
      PopPath();
      return 0;
   }   
//   if (!ncheck) printf("WOOPS ncheck... lost it\n");
   Int_t current = vol->GetNodeIndex(last_node, check_list, ncheck);
   if (current<0) {
      PopPath();
      return 0;
//      printf("WOOPS current node not in mother's voxel ???\n");
//      printf("mother is %s  node is %s\n", vol->GetName(), last_node->GetName());
//      printf("check list :\n");
//      for (Int_t i=0; i<ncheck; i++) 
//         printf("%s\n", vol->GetNode(check_list[ncheck])->GetName());
   }   
   Bool_t is_virtual = kFALSE;
   TGeoNode *node = 0;
   if (ncheck>1) {
   // loop the check list
//      printf("Ncheck=%i  Novlp=%i\n", ncheck, novlps);
      fIgnored[fNignored++] = current;
      fILevel = fLevel;
      for (Int_t id=0; id<ncheck; id++) {
         if (IsIgnored(check_list[id])) continue;
         for (Int_t iov=0; iov<novlps; iov++) {
            if (check_list[id]!=ovlps[iov]) continue;
//            printf("down to checklist[id]=%i\n", check_list[id]);
            CdDown(check_list[id]);
            node = SearchNode(kTRUE);
            if (node) {
               if (!node->IsOverlapping()) {
               // an ONLY was found
                  PopDummy();
                  return node;
               }   
               if (fLevel>last_level) {
               // deeper node found               
                  if (last_node->IsVirtual()) {
                     PopDummy();
                     return node;
                  }   
                  is_virtual = node->IsVirtual();                                    
                  if (!is_virtual) {
                     PopDummy();
                     return node;
                  }   
                  for (Int_t i=1; i<(fLevel-last_level+2); i++)                  
                     is_virtual = is_virtual & GetMother(i)->IsVirtual();                  
                  if (!is_virtual) {
                     PopDummy();
                     return node;
                  }   
                  // branch to new node virtual, if old node also virtual choose new
                  PopPath();
                  return 0;
               }                                       
               // new level equal to old one               
               if (node->IsVirtual()) {
               // keep old one, new node virtual
                  PopPath();
                  return 0;
               }   
               if (last_node->IsVirtual()) {
               // the new node is not virtual - this is the solution
                  PopDummy();
                  return node;
               } else {
               // both nodes not virtual at same level               
                  if (last_node->GetMedia()!=node->GetMedia()) 
                     printf("ERROR in geom : 2 MANY's with different media at same level overlapping\n");
                  PopPath();
                  return 0;
               }                  
            } else {
               PopPath();
               PushPath();
               CdUp();
            }   
         }
      }
   }
   PopPath();
   return 0;
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
const char *TGeoManager::Path()
{
// returns current path
   fPath = "";
   for (Int_t level=0; level<fLevel+1; level++) {
      fPath += "/";
      fPath += ((TGeoNode*)fNodes->At(level))->GetName();
   }
   return fPath.Data();
}
//-----------------------------------------------------------------------------
const char *TGeoManager::GetPath()
{
   if (fIsOutside) return kGeoOutsidePath;
   return gGeoNodeCache->GetPath();
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
TVirtualGeoPainter *TGeoManager::GetMakeDefPainter()
{
// make a default painter if none present
    if (!fPainter) fPainter=TVirtualGeoPainter::GeoPainter();
    return fPainter;
}
//-----------------------------------------------------------------------------
TGeoMaterial *TGeoManager::GetMaterial(const char *matname)
{
// search for given material
   TGeoMaterial *mat = (TGeoMaterial*)fMaterials->FindObject(matname);
   return mat;
}
//-----------------------------------------------------------------------------
TGeoMaterial *TGeoManager::GetMaterial(Int_t id)
{
// return material at position id
   if (id >= fMaterials->GetSize()) return 0;
   TGeoMaterial *mat = (TGeoMaterial*)fMaterials->At(id);
   return mat;
}
//-----------------------------------------------------------------------------
Int_t TGeoManager::GetMaterialIndex(const char *matname)
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
   TObjArray *pm = new TObjArray(128);
   TPolyMarker3D *marker = 0;
   Int_t ic=0;
   gRandom = new TRandom3();
   TGeoVolume *vol=fTopVolume;
   TGeoBBox *box = (TGeoBBox*)vol->GetShape();
   Double_t dx = box->GetDX();
   Double_t dy = box->GetDY();
   Double_t dz = box->GetDZ();
   Double_t ox = (box->GetOrigin())[0];
   Double_t oy = (box->GetOrigin())[1];
   Double_t oz = (box->GetOrigin())[2];

   Double_t start[3];
   Double_t dir[3];
   vol->Draw();
   printf("Random box : %f, %f, %f\n", dx, dy, dz);
   TGeoNode *node, *startnode, *endnode;
   Int_t i=0;
   Double_t theta,phi;
   while (i<nrays) {
      start[0] = ox-dx+2*dx*gRandom->Rndm();
      start[1] = oy-dy+2*dy*gRandom->Rndm();
      start[2] = oz-dz+2*dz*gRandom->Rndm();
      phi = 2*TMath::Pi()*gRandom->Rndm();
      theta= TMath::ACos(1.-2.*gRandom->Rndm());
      dir[0]=TMath::Sin(theta)*TMath::Cos(phi);
      dir[1]=TMath::Sin(theta)*TMath::Sin(phi);
      dir[2]=TMath::Cos(theta);
      InitTrack(&start[0], &dir[0]);
      startnode = fCurrentNode;
      Bool_t vis1,vis2, draw;
      vis1 = (startnode)?(startnode->IsOnScreen()):kFALSE;
      node = FindNextBoundary();
      if (fStep<1E10) {
         endnode = Step();
         vis2=(endnode)?(endnode->IsOnScreen()):kFALSE;
         // if we did not cross the boundary, skip this
         if (endnode==startnode) continue;
         // if exiting top volume, ignore point
         if (endnode==0) continue;
         i++;
         // current path is to end node
         while ((fStep<1E10) && (startnode!=endnode) && (endnode!=0)) {
            if (node) {
               // case endnode=node -> entering node
               if (node==endnode) {
                  if (vis2) {
                     draw=kTRUE;
                     ic=node->GetColour();
                  } else {
                     draw=kFALSE;
                  }
               } else {
                  // case exiting node
                  if (vis1) {
                     draw=kTRUE;
                     ic=node->GetColour();
                  } else {
                     draw=kFALSE;
                  }
               }   
                        
               if (draw) {
                  if (ic >= 128) ic = 0;
                  marker = (TPolyMarker3D*)pm->At(ic);
                  if (!marker) {
                     marker = new TPolyMarker3D();
                     marker->SetMarkerColor(ic);
                     marker->SetMarkerStyle(8);
                     marker->SetMarkerSize(0.2);
                     pm->AddAt(marker, ic);
                  }
                  marker->SetNextPoint(fPoint[0], fPoint[1], fPoint[2]);
               }
            }
            startnode=endnode;
            vis1=vis2;      
            node = FindNextBoundary();
//            if (fStep<1E-9) break;
            endnode = Step();
            vis2=(endnode)?(endnode->IsOnScreen()):kFALSE;
         }
      }
   }         
   for (Int_t m=0; m<128; m++) {
      marker = (TPolyMarker3D*)pm->At(m);
      if (marker) marker->Draw("SAME");
   }
   gPad->Update();
   delete pm;
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
TGeoVolume *TGeoManager::GetVolume(const char *name)
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
//      vol->CheckShapes();
      vol->SortNodes();
      vol->Voxelize(option);
      vol->FindOverlaps();
//      if (vol->GetVoxels()) vol->GetVoxels()->Print();
   }
}
/*
//-----------------------------------------------------------------------------
void TGeoManager::LocalToMaster(Double_t *local, Double_t *master)
{
//  convert a point from local reference system of the current node
//  to MARS
   gGeoMatrixCache->LocalToMaster(local, master);
}
//-----------------------------------------------------------------------------
void TGeoManager::MasterToLocal(Double_t *master, Double_t *local)
{
//  convert a point from MARS to the local reference system of the current node
   gGeoMatrixCache->MasterToLocal(master, local);
}
*/
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
                                     Double_t lx, Double_t ly, Double_t lz, Double_t hx, Double_t hy, Double_t hz)
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
   TGeoCtub *ctub = new TGeoCtub(rmin, rmax, dz, phi1, phi2, lx, ly, lz, hx, hy, hz);
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
   Bool_t change = (gPad==0)?kFALSE:kTRUE;

   if (ibomb==kGeoNoBomb) {
      change &= ((fExplodedView==kGeoNoBomb)?kFALSE:kTRUE);
   }      
   if (ibomb==kGeoBombXYZ) {
      change &= ((fExplodedView==kGeoBombXYZ)?kFALSE:kTRUE);
   }
   if (ibomb==kGeoBombCyl) {
      change &= ((fExplodedView==kGeoBombCyl)?kFALSE:kTRUE);
   }
   if (ibomb==kGeoBombSph) {
      change &= ((fExplodedView==kGeoBombSph)?kFALSE:kTRUE);
   }
   fExplodedView = ibomb;
   if (change && gPad->GetView()) {
      gPad->Modified();
      gPad->Update();  
   }       
}
//-----------------------------------------------------------------------------
void TGeoManager::SetNsegments(Int_t nseg)
{
// (re)compute tables of Sin and Cos
   if (nseg < 3) return;
   fNsegments = nseg;
   if (kGeoSinTable) {
      delete [] kGeoSinTable;
      delete [] kGeoCosTable;
   }
   kGeoSinTable = new Double_t[nseg];
   kGeoCosTable = new Double_t[nseg];
   Double_t step = 360./nseg;
   fSegStep = step;
   Double_t phi = 0;
   for (Int_t i=0; i<nseg; i++) {
      phi = i*step*TMath::Pi()/180;
      kGeoSinTable[i] = TMath::Sin(phi);
      kGeoCosTable[i] = TMath::Cos(phi);
   }
}
//-----------------------------------------------------------------------------
Double_t TGeoManager::CoSin(Double_t phi, Bool_t icos)
{
   if (phi<0) phi+=360;
   Int_t n1 = ((Int_t)(phi/fSegStep))%fNsegments;
   if (icos) 
      return kGeoCosTable[n1];
   else
      return kGeoSinTable[n1];
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
   // make sure that path to current node is updated
   // get the response of tgeo
   TGeoNode *node = FindNode();
   TGeoNode *nodegeo = 0;
   TGeoNode *nodeg3 = 0;
   TGeoNode *solg3 = 0;
   if (!node) {dist=-1; return 0;}
   Bool_t hasg3 = kFALSE;
   if (strlen(g3path)) hasg3 = kTRUE;
   char geopath[200];
   sprintf(geopath, "%s\n", gGeoManager->GetPath());
   dist = 1E10;
   TString common = "";
   // cd to common path
   Double_t point[3];
   Double_t closest[3];
   TGeoNode *node1 = 0;
   TGeoNode *node_close = 0;
   dist = 1E10;
   Double_t dist1 = 0;
   // initialize size of random box to epsil
   Double_t eps[3];
   eps[0] = epsil; eps[1]=epsil; eps[2]=epsil;
   if (hasg3) {
      TString spath = geopath;
      TString name = "";
      Int_t index=0;
      while (index>=0) {
         index = spath.Index("/", index+1);
         if (index>0) {
            name = spath(0, index);
            if (strstr(g3path, name.Data())) {
               common = name;
               continue;
            } else break;
         }   
      }
      // if g3 response was given, cd to common path
      if (strlen(common.Data())) {
         while (strcmp(gGeoManager->GetPath(), common.Data()) && fLevel) {
            nodegeo = fCurrentNode;
            CdUp();
         }   
         cd(g3path);
         solg3 = fCurrentNode;
         while (strcmp(gGeoManager->GetPath(), common.Data()) && fLevel) {
            nodeg3 = fCurrentNode;
            CdUp();
         }   
         if (!nodegeo) return 0;
         if (!nodeg3) return 0;
         cd(common.Data());
//         printf("common path : %s\n", common.Data());
//         printf("node geo : %s\n", nodegeo->GetName());
//         printf("node g3  : %s\n", nodeg3->GetName());
         gGeoManager->MasterToLocal(fPoint, &point[0]);
         Double_t xyz[3], local[3];
         for (Int_t i=0; i<npoints; i++) {
            xyz[0] = point[0] - eps[0] + 2*eps[0]*gRandom->Rndm();
            xyz[1] = point[1] - eps[1] + 2*eps[1]*gRandom->Rndm();
            xyz[2] = point[2] - eps[2] + 2*eps[2]*gRandom->Rndm();
//            nodegeo->MasterToLocal(&xyz[0], &local[0]);
//            if (nodegeo->GetVolume()->Contains(&local[0])) continue;
//            printf("out nodegeo\n");
            nodeg3->MasterToLocal(&xyz[0], &local[0]);
            if (!nodeg3->GetVolume()->Contains(&local[0])) continue;
            dist1 = TMath::Sqrt((xyz[0]-point[0])*(xyz[0]-point[0])+
                   (xyz[1]-point[1])*(xyz[1]-point[1])+(xyz[2]-point[2])*(xyz[2]-point[2]));
            if (dist1<dist) {
            // save node and closest point
               dist = dist1;
               node_close = solg3;
               // make the random box smaller
               eps[0] = TMath::Abs(point[0]-fPoint[0]);
               eps[1] = TMath::Abs(point[1]-fPoint[1]);
               eps[2] = TMath::Abs(point[2]-fPoint[2]);
            }   
         }            
      }
      if (!node_close) dist = -1;
      return node_close;
   }
         
//   gRandom = new TRandom3();
   // save current point
   memcpy(&point[0], fPoint, 3*sizeof(Double_t));
   for (Int_t i=0; i<npoints; i++) {
      // generate a random point in MARS
      fPoint[0] = point[0] - eps[0] + 2*eps[0]*gRandom->Rndm();
      fPoint[1] = point[1] - eps[1] + 2*eps[1]*gRandom->Rndm();
      fPoint[2] = point[2] - eps[2] + 2*eps[2]*gRandom->Rndm();
      // check if new node is different from the old one
      if (node1!=node) {
         dist1 = TMath::Sqrt((point[0]-fPoint[0])*(point[0]-fPoint[0])+
                 (point[1]-fPoint[1])*(point[1]-fPoint[1])+(point[2]-fPoint[2])*(point[2]-fPoint[2]));
         if (dist1<dist) {
            dist = dist1;
            node_close = node1;
            memcpy(&closest[0], fPoint, 3*sizeof(Double_t));
            // make the random box smaller
            eps[0] = TMath::Abs(point[0]-fPoint[0]);
            eps[1] = TMath::Abs(point[1]-fPoint[1]);
            eps[2] = TMath::Abs(point[2]-fPoint[2]);
         }
      }         
   }
//   delete [] geopath;
   // restore the original point and path
   memcpy(fPoint, &point[0], 3*sizeof(Double_t));
   FindNode();  // really needed ?
   if (!node_close) dist=-1;
   return node_close;            
}
//-----------------------------------------------------------------------------
void TGeoManager::SetRandomBox(Double_t ox, Double_t dx, Double_t oy, Double_t dy,
                               Double_t oz, Double_t dz)
{
// set the random box parameters for point sampling
   fRandomBox[0] = ox;
   fRandomBox[1] = dx;
   fRandomBox[2] = oy;
   fRandomBox[3] = dy;
   fRandomBox[4] = oz;
   fRandomBox[5] = dz;
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
   if (gGeoNodeCache) {
      delete gGeoNodeCache;
      gGeoNodeCache = 0;
      gGeoMatrixCache = 0;
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
void TGeoManager::Top()
{
// go to top level
   fLevel = 0;
   fCurrentNode = fTopNode;
}
//-----------------------------------------------------------------------------
void TGeoManager::Up()
{
// go up one level
   if (!fLevel) return;
   fCurrentNode = (TGeoNode*)fNodes->At(--fLevel);
}
//-----------------------------------------------------------------------------
void TGeoManager::CheckPoint(Double_t x, Double_t y, Double_t z, Option_t *option)
{
   TGeoNode *current = 0;
   TGeoNode *node = 0;
   Double_t dist;
   SetCurrentPoint(x,y,z);
   current = FindNode();
   printf("point: x=%f y=%f z=%f\n", x,y,z);
   if (current) 
      printf("node containing point : %s\n", current->GetName());
   else
      printf("node containing point : none\n");   
   node = SamplePoints(10000, dist, 0.01);
   if (dist>1E9) {
      printf("No boundary closer than 0.01\n");
      return;
   }
   printf("     closest boundary : %f sampled with %d points\n", dist, 10000);    
   if (node)
      printf("         closest node : %s\n", node->GetName());
   else
      printf("         closest node : NULL\n");   
      
/*
   TString opt = option;
   opt.ToLower();
   if (gPad) gPad->Clear();
   if (opt == "+") {
      fRandomBox[1] *= 2;
      fRandomBox[3] *= 2;
      fRandomBox[5] *= 2;
      DrawPoints();
      return;
   }
   if (opt == "-") {
      fRandomBox[1] /= 2;
      fRandomBox[3] /= 2;
      fRandomBox[5] /= 2;
      DrawPoints();
      return;
   }
   if (fTopVolume != fMasterVolume) RestoreMasterVolume();
   SetCurrentPoint(x,y,z);
   SetRandomBox(x, 0.1, y, 0.1, z, 0.1); 
   DrawPoints();
*/
}
//-----------------------------------------------------------------------------
void TGeoManager::CheckGeometry(Option_t *option)
{
// instances a TGeoChecker object and investigates the geometry according to 
// option 
   // check shapes first
   fTopNode->CheckShapes();
//   CleanGarbage();
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
