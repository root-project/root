// @(#)root/geom:$Name:  $:$Id: TGeoManager.cxx,v 1.10 2002/07/17 15:13:08 brun Exp $
// Author: Andrei Gheata   25/10/01

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////////
// General architecture
// --------------------
//
//   The new ROOT geometry package is a tool designed for building, browsing,
// tracking and visualizing a detector geometry. The code is independent from
// other external MC for simulation, therefore it does not contain any 
// constraints related to physics. However, the package defines a number of 
// hooks for tracking, such as materials, magnetic field or track state flags, 
// in order to allow interfacing to tracking MC's. The final goal is to be 
// able to use the same geometry for several purposes, such as tracking, 
// reconstruction or visualization, taking advantage of the ROOT features 
// related to bookkeeping, I/O, histograming, browsing and GUI's.  
//    
//   The geometrical modeler is the most important component of the package and
// it provides answers to the basic questions like "Where am I ?" or "How far
// from the next boundary ?", but also to more complex ones like "How far from
// the closest surface ?" or "Which is the next crossing along a helix ?".
//
//   The architecture of the modeler is a combination between a GEANT-like
// containment scheme and a normal CSG binary tree at the level of shapes. An
// important common feature of all detector geometry descriptions is the 
// mother-daughter concept. This is the most natural approach when tracking
// is concerned and imposes a set of constraints to the way geometry is defined. 
// Constructive solid geometry composition is used only in order to create more
// complex shapes from an existing set of primitives through boolean operations. 
// This feature is not implemented yet but in future full definition of boolean
// expressions will be supported. 
//
//   Practically every geometry defined in GEANT style can be mapped by the modeler.
// The basic components used for building the logical hierarchy of the geometry 
// are called "volumes" and "nodes". Volumes (sometimes called "solids") are fully 
// defined geometrical objects having a given shape and material and possibly 
// containing a list of nodes. Nodes represent just positioned instances of volumes 
// inside a container volume and they are not directly defined by user. They are 
// automatically created as a result of adding one volume inside other or dividing 
// a volume. The geometrical transformation hold by nodes is always defined with 
// respect to their mother (relative positioning). Reflection matrices are allowed.
// All volumes have to be fully aware of their containees when the geometry is
// closed. They will build aditional structures (voxels) in order to fasten-up
// the search algorithms. Finally, nodes can be regarded as bidirectional links 
// between containers and containees objects. 
//
//   The structure defined in this way is a graph structure since volumes are 
// replicable (same volume can become daughter node of several other volumes), 
// every volume becoming a branch in this graph. Any volume in the logical graph
// can become the actual top volume at run time (see TGeoManager::SetTopVolume()).
// All functionalities of the modeler will behave in this case as if only the
// corresponding branch starting from this volume is the registered geometry.
//
//Begin_Html
/*
<img src="gif/t_graf.jpg">
*/
//End_Html
//
//   A given volume can be positioned several times in the geometry. A volume 
// can be divided according default or user-defined patterns, creating automatically 
// the list of division nodes inside. The elementary volumes created during the 
// dividing process follow the same scheme as usual volumes, therefore it is possible 
// to position further geometrical structures inside or to divide them further more 
// (see TGeoVolume::Divide()).
//
//   The primitive shapes supported by the package are basically the GEANT3
// shapes (see class TGeoShape), arbitrary wedges with eight vertices on two parallel 
// planes. All basic primitives inherits from class TGeoBBox since the bounding box
// of a solid is essential for the tracking algorithms. They also implement the
// virtual methods defined in the virtual class TGeoShape (point and segment
// classification). User-defined primitives can be direcly plugged into the modeler
// provided that they override these methods. Composite shapes will be soon supported
// by the modeler. In order to build a TGeoCompositeShape, one will have to define 
// first the primitive components. The object that handle boolean 
// operations among components is called TGeoBoolCombinator and it has to be 
// constructed providing a string boolean expression between the components names.
// 
//
// Example for building a simple geometry :
//-----------------------------------------
//
//______________________________________________________________________________
//void rootgeom()
//{
////--- Definition of a simple geometry
//   gSystem->Load("libGeom");
//   TGeoManager *geom = new TGeoManager("simple1", "Simple geometry");
//   
//   //--- define some materials
//   TGeoMaterial *mat;
//   mat = new TGeoMaterial("mat1", "Vacuum", 0,0,0);
//   mat = new TGeoMaterial("mat2", "Al", 26.98,13,2.7);
//
//   //--- define the transformations
//   TGeoTranslation *tr1 = new TGeoTranslation(20., 0, 0.);
//   TGeoTranslation *tr2 = new TGeoTranslation(10., 0., 0.);
//   TGeoTranslation *tr3 = new TGeoTranslation(10., 20., 0.);
//   TGeoTranslation *tr4 = new TGeoTranslation(5., 10., 0.);
//   TGeoTranslation *tr5 = new TGeoTranslation(20., 0., 0.);
//   TGeoTranslation *tr6 = new TGeoTranslation(-5., 0., 0.);
//   TGeoTranslation *tr7 = new TGeoTranslation(7.5, 7.5, 0.);
//   TGeoRotation   *rot1 = new TGeoRotation("rot1", 90., 0., 90., 270., 0., 0.);
//   TGeoCombiTrans *combi1 = new TGeoCombiTrans(7.5, -7.5, 0., rot1);
//   TGeoTranslation *tr8 = new TGeoTranslation(7.5, -5., 0.);
//   TGeoTranslation *tr9 = new TGeoTranslation(7.5, 20., 0.);
//   TGeoTranslation *tr10 = new TGeoTranslation(85., 0., 0.);
//   TGeoTranslation *tr11 = new TGeoTranslation(35., 0., 0.);
//   TGeoTranslation *tr12 = new TGeoTranslation(-15., 0., 0.);
//   TGeoTranslation *tr13 = new TGeoTranslation(-65., 0., 0.);
//   
//   TGeoTranslation  *tr14 = new TGeoTranslation(0,0,-100);
//   TGeoCombiTrans *combi2 = new TGeoCombiTrans(0,0,100, 
//                                   new TGeoRotation("rot2",90,180,90,90,180,0));
//   TGeoCombiTrans *combi3 = new TGeoCombiTrans(100,0,0, 
//                                   new TGeoRotation("rot3",90,270,0,0,90,180));
//   TGeoCombiTrans *combi4 = new TGeoCombiTrans(-100,0,0, 
//                                   new TGeoRotation("rot4",90,90,0,0,90,0));
//   TGeoCombiTrans *combi5 = new TGeoCombiTrans(0,100,0, 
//                                   new TGeoRotation("rot5",0,0,90,180,90,270));
//   TGeoCombiTrans *combi6 = new TGeoCombiTrans(0,-100,0, 
//                                   new TGeoRotation("rot6",180,0,90,180,90,90));
//   
//   //--- make the top container volume
//   Double_t worldx = 110.;
//   Double_t worldy = 50.;
//   Double_t worldz = 5.;
//   TGeoVolume *top = geom->MakeBox("TOP", "mat1", 270., 270., 120.);
//   geom->SetTopVolume(top); // mandatory !
//   //--- build other container volumes
//   TGeoVolume *replica = geom->MakeBox("REPLICA", "mat1",120,120,120);
//   replica->SetVisibility(kFALSE);
//   TGeoVolume *rootbox = geom->MakeBox("ROOT", "mat1", 110., 50., 5.);
//   rootbox->SetVisibility(kFALSE); // this will hold word 'ROOT'
//   
//   //--- make letter 'R'
//   TGeoVolume *R = geom->MakeBox("R", "mat1", 25., 25., 5.);
//   R->SetVisibility(kFALSE);
//   TGeoVolume *bar1 = geom->MakeBox("bar1", "mat2", 5., 25, 5.);
//   bar1->SetLineColor(kRed);
//   R->AddNode(bar1, 1, tr1);
//   TGeoVolume *bar2 = geom->MakeBox("bar2", "mat2", 5., 5., 5.);
//   bar2->SetLineColor(kRed);
//   R->AddNode(bar2, 1, tr2);
//   R->AddNode(bar2, 2, tr3);
//   TGeoVolume *tub1 = geom->MakeTubs("tub1", "mat2", 5., 15., 5., 90., 270.);
//   tub1->SetLineColor(kRed);
//   R->AddNode(tub1, 1, tr4);
//   TGeoVolume *bar3 = geom->MakeArb8("bar3", "mat2", 5.);
//   bar3->SetLineColor(kRed);
//   TGeoArb8 *arb = (TGeoArb8*)bar3->GetShape();
//   arb->SetVertex(0, 15., -5.);
//   arb->SetVertex(1, 5., -5.);
//   arb->SetVertex(2, -10., -25.);
//   arb->SetVertex(3, 0., -25.);
//   arb->SetVertex(4, 15., -5.);
//   arb->SetVertex(5, 5., -5.);
//   arb->SetVertex(6, -10., -25.);
//   arb->SetVertex(7, 0., -25.);
//   R->AddNode(bar3, 1, gGeoIdentity);
//   
//   //--- make letter 'O'
//   TGeoVolume *O = geom->MakeBox("O", "mat1", 25., 25., 5.);
//   O->SetVisibility(kFALSE);
//   TGeoVolume *bar4 = geom->MakeBox("bar4", "mat2", 5., 7.5, 5.);
//   bar4->SetLineColor(kYellow);
//   O->AddNode(bar4, 1, tr5);
//   O->AddNode(bar4, 2, tr6);
//   TGeoVolume *tub2 = geom->MakeTubs("tub1", "mat2", 7.5, 17.5, 5., 0., 180.);
//   tub2->SetLineColor(kYellow);
//   O->AddNode(tub2, 1, tr7);
//   O->AddNode(tub2, 2, combi1);
//   
//   //--- make letter 'T'
//   TGeoVolume *T = geom->MakeBox("T", "mat1", 25., 25., 5.);
//   T->SetVisibility(kFALSE);
//   TGeoVolume *bar5 = geom->MakeBox("bar5", "mat2", 5., 20., 5.);
//   bar5->SetLineColor(kBlue);
//   T->AddNode(bar5, 1, tr8);
//   TGeoVolume *bar6 = geom->MakeBox("bar6", "mat2", 17.5, 5., 5.);
//   bar6->SetLineColor(kBlue);
//   T->AddNode(bar6, 1, tr9);
//
//   //--- add letters to 'ROOT' container         
//   rootbox->AddNode(R, 1, tr10);
//   rootbox->AddNode(O, 1, tr11);
//   rootbox->AddNode(O, 2, tr12);
//   rootbox->AddNode(T, 1, tr13);
//
//   //--- add word 'ROOT' on each face of a cube
//   replica->AddNode(rootbox, 1, tr14);
//   replica->AddNode(rootbox, 2, combi2);
//   replica->AddNode(rootbox, 3, combi3);
//   replica->AddNode(rootbox, 4, combi4);
//   replica->AddNode(rootbox, 5, combi5);
//   replica->AddNode(rootbox, 6, combi6);
//
//   //--- add four replicas of this cube to top volume
//   top->AddNode(replica, 1, new TGeoTranslation(-150, -150, 0));
//   top->AddNode(replica, 2, new TGeoTranslation(150, -150, 0));
//   top->AddNode(replica, 3, new TGeoTranslation(150, 150, 0));
//   top->AddNode(replica, 4, new TGeoTranslation(-150, 150, 0));
//
//   //--- close the geometry
//   geom->CloseGeometry();
//   
//   //--- draw the ROOT box
//   geom->SetVisLevel(4); 
//   top->Draw();
//   if (gPad) gPad->x3d();
//}   
//______________________________________________________________________________
//
//
//Begin_Html
/*
<img src="gif/t_root.jpg">
*/
//End_Html
//
//
// TGeoManager - the manager class for the geometry package.
// ---------------------------------------------------------
//
//   TGeoManager class is embedding all the API needed for building and tracking
// a geometry. It defines a global pointer (gGeoManager) in order to be fully
// accessible from external code. The mechanism of handling multiple geometries
// at the same time will be soon implemented. 
//
//   TGeoManager is the owner of all geometry objects defined in a session, 
// therefore users must not try to control their deletion. It contains lists of 
// materials, transformations, shapes and volumes. Logical nodes (positioned 
// volumes) are created and destroyed by the TGeoVolume class. Physical
// nodes and their global transformations are subjected to a caching mechanism
// due to the sometimes very large memory requirements of logical graph expansion.
// The caching mechanism is triggered by the total number of physical instances 
// of volumes and the cache manager is a client of TGeoManager. The manager class
// also controls the painter client. This is linked with ROOT graphical libraries 
// loaded on demand in order to control visualization actions.
//
// Rules for building a valid geometry
// -----------------------------------
//
//   A given geometry can be built in various ways, but there are mandatory steps
// that have to be followed in order to be validated by the modeler. There are
// general rules : volumes needs materials and shapes in order to be created,
// both container an containee volumes must be created before linking them together,
// and the relative transformation matrix must be provided. All branches must
// have an upper link point otherwise they will not be considered as part of the 
// geometry. Visibility or tracking properties of volumes can be provided both
// at build time or after geometry is closed, but global visualization settings
// (see TGeoPainter class) should not be provided at build time, otherwise the
// drawing package will be loaded. There is also a list of specific rules : 
// positioned daughters should not extrude their mother or intersect with sisters
// unless this is specified (see TGeoVolume::AddNodeOverlap()), the top volume
// (containing all geometry tree) must be specified before closing the geometry
// and must not be positioned - it represents the global reference frame. After
// building the full geometry tree, the geometry must be closed 
// (see TGeoManager::CloseGeometry()). Voxelization can be redone per volume after
// this process.
// 
//
//   Below is the general scheme of the manager class.
//
//Begin_Html
/*
<img src="gif/t_mgr.jpg">
*/
//End_Html
//
//  An interactive session
// ------------------------
//
//   Provided that a geometry was successfully built and closed (for instance the 
// previous example $ROOTSYS/tutorials/rootgeom.C ), the manager class will register 
// itself to ROOT and the logical/physical structures will become immediately browsable. 
// The ROOT browser will display starting from the geometry folder : the list of 
// transformations and materials, the top volume and the top logical node. These last 
// two can be fully expanded, any intermediate volume/node in the browser being subject 
// of direct access context menu operations (right mouse button click). All user
// utilities of classes TGeoManager, TGeoVolume and TGeoNode can be called via the
// context menu.
//
//Begin_Html
/*
<img src="gif/t_browser.jpg">
*/
//End_Html
//
//  --- Drawing the geometry
//
//   Any logical volume can be drawn via TGeoVolume::Draw() member function.
// This can be direcly accessed from the context menu of the volume object
// directly from the browser. 
//   There are several drawing options that can be set with
// TGeoManager::SetVisOption(Int_t opt) method :
// opt=0 - only the content of the volume is drawn, N levels down (default N=3).
//    This is the default behavior. The number of levels to be drawn can be changed
//    via TGeoManager::SetVisLevel(Int_t level) method.
//
//Begin_Html
/*
<img src="gif/t_frame0.jpg">
*/
//End_Html
//
// opt=1 - the final leaves (e.g. daughters with no containment) of the branch
//    starting from volume are drawn down to the current number of levels. 
//                                     WARNING : This mode is memory consuming
//    depending of the size of geometry, so drawing from top level within this mode
//    should be handled with care for expensive geometries. In future there will be 
//    a limitation on the maximum number of nodes to be visualized.
//
//Begin_Html
/*
<img src="gif/t_frame1.jpg">
*/
//End_Html
//
// opt=2 - only the clicked volume is visualized. This is automatically set by
//    TGeoVolume::DrawOnly() method
// opt=3 - only a given path is visualized. This is automatically set by
//    TGeoVolume::DrawPath(const char *path) method
//
//    The current view can be exploded in cartesian, cylindrical or spherical
// coordinates :
//   TGeoManager::SetExplodedView(Int_t opt). Options may be :
// - 0  - default (no bombing)
// - 1  - cartesian coordinates. The bomb factor on each axis can be set with
//        TGeoManager::SetBombX(Double_t bomb) and corresponding Y and Z.
// - 2  - bomb in cylindrical coordinates. Only the bomb factors on Z and R
//        are considered
//
//Begin_Html
/*
<img src="gif/t_frameexp.jpg">
*/
//End_Html
//
// - 3  - bomb in radial spherical coordinate : TGeoManager::SetBombR()
//
// Volumes themselves support different visualization settings :
//    - TGeoVolume::SetVisibility() : set volume visibility.
//    - TGeoVolume::VisibleDaughters() : set daughters visibility.
// All these actions automatically updates the current view if any.
//
//  --- Checking the geometry
//
//  Several checking methods are accessible from the volume context menu. They
// generally apply only to the visible parts of the drawn geometry in order to
// ease geometry checking, and their implementation is in the TGeoChecker class
// from the painting package.
//
// 1. Checking a given point.
//   Can be called from TGeoManager::CheckPoint(Double_t x, Double_t y, Double_t z).
// This method is drawing the daughters of the volume containing the point one
// level down, printing the path to the deepest physical node holding this point.
// It also computes the closest distance to any boundary. The point will be drawn
// in red.
//
//Begin_Html
/*
<img src="gif/t_checkpoint.jpg">
*/
//End_Html
//
//  2. Shooting random points.
//   Can be called from TGeoVolume::RandomPoints() (context menu function) and 
// it will draw this volume with current visualization settings. Random points
// are generated in the bounding box of the top drawn volume. The points are 
// classified and drawn with the color of their deepest container. Only points
// in visible nodes will be drawn.
//
//Begin_Html
/*
<img src="gif/t_random1.jpg">
*/
//End_Html
//
//
//  3. Raytracing.
//   Can be called from TGeoVolume::RandomRays() (context menu of volumes) and
// will shoot rays from a given point in the local reference frame with random
// directions. The intersections with displayed nodes will appear as segments
// having the color of the touched node. Drawn geometry will be then made invisible
// in order to enhance rays.
//
//Begin_Html
/*
<img src="gif/t_random2.jpg">
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

const char *kGeoOutsidePath = " ";

ClassImp(TGeoManager)

//-----------------------------------------------------------------------------
TGeoManager::TGeoManager()
{
// Default constructor.
   fBits = 0;
   fMaterials = 0;
   fMatrices = 0;
   fPoint = 0;
   fDirection = 0;
   fNormalChecked = 0;
   fCldirChecked = 0;
   fNormal = 0;
   fCldir = 0;
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
   fIsOnBoundary = kFALSE;
   fIsNullStep = kFALSE;
   gGeoIdentity = 0;
}
//-----------------------------------------------------------------------------
TGeoManager::TGeoManager(const char *name, const char *title)
            :TNamed(name, title)
{
// Constructor.
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
   fIsOnBoundary = kFALSE;
   fIsNullStep = kFALSE;

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
   printf("deleting matrices...\n");
   if (fMatrices) {fMatrices->Delete(); delete fMatrices;}
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
// Add a material to the list. Returns index of the material in list.
   if (!material) {
      Error("AddMaterial", "invalid material");
      return -1;
   }
   Int_t index = GetMaterialIndex(material->GetName());
   if (index >= 0) return index;
   index = fMaterials->GetSize();
   fMaterials->Add(material);
   return index;
}
//-----------------------------------------------------------------------------
Int_t TGeoManager::AddTransformation(TGeoMatrix *matrix)
{
// Add a matrix to the list. Returns index of the matrix in list.
   if (!matrix) {
      Error("AddMatrix", "invalid matrix");
      return -1;
   }
   Int_t index = fMatrices->GetSize();
   fMatrices->Add(matrix);
   return index;
}
//-----------------------------------------------------------------------------
Int_t TGeoManager::AddShape(TGeoShape *shape)
{
// Add a shape to the list. Returns index of the shape in list.
   if (!shape) {
      Error("AddShape", "invalid shape");
      return -1;
   }
   TList *list = fShapes;
   if (shape->IsRunTimeShape()) list = fGShapes;;
   Int_t index = list->GetSize();
   list->Add(shape);
   return index;
}
//-----------------------------------------------------------------------------
Int_t TGeoManager::AddVolume(TGeoVolume *volume)
{
// Add a volume to the list. Returns index of the volume in list.
   if (!volume) {
      Error("AddVolume", "invalid volume");
      return -1;
   }
   TList *list = fVolumes;
   if (volume->IsRunTime()) list = fGVolumes;
   Int_t index = list->GetSize();
   list->Add(volume);
   return index;
}
//-----------------------------------------------------------------------------
void TGeoManager::Browse(TBrowser *b)
{
// Describe how to browse this object.
   if (!b) return;
   if (fMaterials) b->Add(fMaterials, "Materials");
   if (fMatrices)  b->Add(fMatrices, "Local transformations");
   if (fTopVolume) b->Add(fTopVolume);
   if (fTopNode)   b->Add(fTopNode);
}
//-----------------------------------------------------------------------------
void TGeoManager::BombTranslation(const Double_t *tr, Double_t *bombtr)
{
// Get the new 'bombed' translation vector according current exploded view mode.
   if (fPainter) fPainter->BombTranslation(tr, bombtr);
   return;
}
//-----------------------------------------------------------------------------
void TGeoManager::UnbombTranslation(const Double_t *tr, Double_t *bombtr)
{
// Get the new 'unbombed' translation vector according current exploded view mode.
   if (fPainter) fPainter->UnbombTranslation(tr, bombtr);
   return;
}
//-----------------------------------------------------------------------------
void TGeoManager::BuildCache()
{
// Builds the cache for physical nodes and global matrices.
   if (!fCache) {
      if (fNNodes>5000000)  // temporary - works without
         // build dummy cache 
         fCache = new TGeoCacheDummy(fTopNode);
      else
         // build real cache
         fCache = new TGeoNodeCache(0);
   }
}
//-----------------------------------------------------------------------------
void TGeoManager::ClearAttributes()
{
// Reset all attributes to default ones. Default attributes for visualization
// are those defined before closing the geometry. 
   if (gPad) delete gPad;
   gPad = 0;
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
      vol->SetLineStyle(gStyle->GetLineStyle());
      vol->SetLineWidth(gStyle->GetLineWidth());
      vol->SetVisTouched(kFALSE);
   }
}
//-----------------------------------------------------------------------------
void TGeoManager::CloseGeometry()
{
// Closing geometry implies checking the geometry validity, fixing shapes 
// with negative parameters (run-time shapes)building the cache manager, 
// voxelizing all volumes, counting the total number of physical nodes and
// registring the manager class to the browser.
   SelectTrackingMedia();
   printf("Fixing runtime shapes...\n");
   CheckGeometry();
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
// Remove a shape from the list of shapes.
   if (fShapes->FindObject(shape)) fShapes->Remove(shape);
   delete shape;
}
//-----------------------------------------------------------------------------
void TGeoManager::CleanGarbage()
{
// Clean temporary volumes and shapes from garbage collection.
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
// Make top level node the current node. Updates the cache accordingly.
// Determine the overlapping state of current node.
   fLevel = 0;
   if (fCurrentOverlapping) fLastNode = fCurrentNode;
   fCurrentNode = fTopNode;
   fCache->CdTop();
   fCurrentOverlapping = fCurrentNode->IsOverlapping();
}
//-----------------------------------------------------------------------------
void TGeoManager::CdUp()
{
// Go one level up in geometry. Updates cache accordingly.
// Determine the overlapping state of current node.
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
// Make a daughter of current node current. Can be called only with a valid 
// daughter index (no check). Updates cache accordingly.
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
// Count the total number of nodes starting from a volume, nlevels down.
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
// Geometry overlap checker based on sampling. 
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
// be on top of all the nodes in cluster. Always nc>1.
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
// distance to closest boundary and store it in fSafety. Set flags 
// fIsStepEntering/fIsStepExiting according to the fact that current ray will enter
// or exit next node.

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
      if (tvol->Contains(&point[0])) {
         fStep=tvol->GetShape()->DistToOut(&point[0], &dir[0], 3, TGeoShape::kBig, &fSafety);
         fIsStepEntering=kFALSE;
         fIsStepExiting=kTRUE;
      } else {
         fStep=tvol->GetShape()->DistToIn(&point[0], &dir[0], 3, TGeoShape::kBig, &fSafety);
         fIsStepEntering=kTRUE;
         fIsStepExiting=kFALSE;
      }   
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
      fStep = fTopVolume->GetShape()->DistToIn(fPoint, fDirection, 3, TGeoShape::kBig, &fSafety);
      fIsStepEntering=kTRUE;
      fIsStepExiting=kFALSE;
      return fTopNode;
   }
   // find distance to exiting current node
   fIsStepEntering=kFALSE;
   fIsStepExiting=kTRUE;
   if (fIsOnBoundary) {
      fStep = vol->GetShape()->DistToOut(&point[0], &dir[0], 2, TGeoShape::kBig, &fSafety);
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
            fIsStepEntering=kTRUE;
            fIsStepExiting=kFALSE;
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
         fIsStepEntering=kTRUE;
         fIsStepExiting=kFALSE;
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
         fIsStepEntering=kTRUE;
         fIsStepExiting=kFALSE;
         return current;
      }
   }
   // if current volume is voxelized, first get current voxel
   TGeoVoxelFinder *voxels = vol->GetVoxels();
//   printf("---check voxels\n");
   if (voxels) {
      Int_t ncheck = 0;
      Int_t *vlist = 0;
      voxels->SortCrossedVoxels(&point[0], &dir[0]);
      Bool_t first = kTRUE;
      while ((vlist=voxels->GetNextVoxel(&point[0], &dir[0], ncheck))) {
//         printf("---ncheck : %i\n", ncheck);
         for (i=0; i<ncheck; i++) {
            current = vol->GetNode(vlist[i]);
            current->cd();
            current->MasterToLocal(&point[0], &lpoint[0]);
            current->MasterToLocalVect(&dir[0], &ldir[0]);
            if (first) {
            // compute also safety if we are in the starting voxel
               snext = current->GetVolume()->GetShape()->DistToIn(&lpoint[0], &ldir[0], 2, TGeoShape::kBig, &safety);
               if (safety<fSafety) fSafety=safety;
            } else {
               snext = current->GetVolume()->GetShape()->DistToIn(&lpoint[0], &ldir[0], 3, TGeoShape::kBig, &safety);
            } 
            if (snext<fStep) {
               fStep=snext;
               fIsStepEntering=kTRUE;
               fIsStepExiting=kFALSE;
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
// Initialize current point and current direction vector (normalized)
// in MARS.
   SetCurrentPoint(point);
   SetCurrentDirection(dir);
   FindNode();
}
//-----------------------------------------------------------------------------
void TGeoManager::InitTrack(Double_t x, Double_t y, Double_t z, Double_t nx, Double_t ny, Double_t nz)
{
// Initialize current point and current direction vector (normalized)
// in MARS.
   SetCurrentPoint(x,y,z);
   SetCurrentDirection(nx,ny,nz);
   FindNode();
}
//-----------------------------------------------------------------------------
const char *TGeoManager::GetPath() const
{
// Get path to the current node in the form /node0/node1/...
   if (fIsOutside) return kGeoOutsidePath;
   return fCache->GetPath();
}
//-----------------------------------------------------------------------------
Int_t TGeoManager::GetByteCount(Option_t *option)
{
// Get total size of geometry in bytes.
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
// Make a default painter if none present. Returns pointer to it.
    if (!fPainter) fPainter=TVirtualGeoPainter::GeoPainter();
    return fPainter;
}
//-----------------------------------------------------------------------------
TGeoMaterial *TGeoManager::GetMaterial(const char *matname) const
{
// Search for a named material.
   TGeoMaterial *mat = (TGeoMaterial*)fMaterials->FindObject(matname);
   return mat;
}
//-----------------------------------------------------------------------------
TGeoMaterial *TGeoManager::GetMaterial(Int_t id) const
{
// Return material at position id.
   if (id >= fMaterials->GetSize()) return 0;
   TGeoMaterial *mat = (TGeoMaterial*)fMaterials->At(id);
   return mat;
}
//-----------------------------------------------------------------------------
Int_t TGeoManager::GetMaterialIndex(const char *matname) const
{
// Return index of named material.
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
void TGeoManager::RandomRays(Int_t nrays, Double_t startx, Double_t starty, Double_t startz)
{
// Randomly shoot nrays and plot intersections with surfaces for current
// top node.
   GetGeomPainter()->RandomRays(nrays, startx, starty, startz);
}
//-----------------------------------------------------------------------------
void TGeoManager::RemoveMaterial(Int_t index)
{
// Remove material at given index.
   TObject *obj = fMaterials->At(index);
   if (obj) fMaterials->Remove(obj);
}
//-----------------------------------------------------------------------------
void TGeoManager::RestoreMasterVolume()
{
// Restore the master volume of the geometry.
   if (fTopVolume == fMasterVolume) return;
   if (fMasterVolume) SetTopVolume(fMasterVolume);
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoManager::GetVolume(const char *name) const
{
// Retrieves a named volume.
   return ((TGeoVolume*)fVolumes->FindObject(name));
}
//-----------------------------------------------------------------------------
void TGeoManager::Voxelize(Option_t *option)
{
// Voxelize all non-divided volumes.
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
// Make an TGeoArb8 volume.
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
// Make in one step a volume pointing to a box shape with given material.
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
// Make in one step a volume pointing to a paralelipiped shape with given material.
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
// Make in one step a volume pointing to a tube shape with given material.
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
// Make in one step a volume pointing to a tube segment shape with given material.
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
// Make in one step a volume pointing to a cone shape with given material.
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
// Make in one step a volume pointing to a polycone shape with given material.
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
// Make in one step a volume pointing to a polygone shape with given material.
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
// Make in one step a volume pointing to a TGeoTrd1 shape with given material.
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
// Make in one step a volume pointing to a TGeoTrd2 shape with given material.
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
// Make in one step a volume pointing to a trapezoid shape with given material.
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
// Make in one step a volume pointing to a twisted trapezoid shape with given material.
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
// Make a TGeoVolumeMulti handling a list of volumes.
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
// Set type of exploding view (see TGeoPainter::SetExplodedView())
   GetGeomPainter();
   fPainter->SetExplodedView(ibomb);
}

//-----------------------------------------------------------------------------
void TGeoManager::SetNsegments(Int_t nseg)
{
// Set number of segments for approximating circles in drawing.
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
void TGeoManager::BuildDefaultMaterials()
{
// Build the default materials. A list of those can be found in ...
   new TGeoMaterial("default", "Air", 14.61, 7.3, 0.001205);
}
//-----------------------------------------------------------------------------
TGeoNode *TGeoManager::Step(Bool_t is_geom, Bool_t cross)
{
// Make a rectiliniar step of length fStep from current point (fPoint) on current
// direction (fDirection). If the step is imposed by geometry, is_geom flag
// must be true (default). The cross flag specifies if the boundary should be
// crossed in case of a geometry step (default true). Returns new node after step.
// Set also on boundary condition.
   Double_t epsil = 0;
   if (fStep<1E-9) fIsNullStep=kTRUE;
   else fIsNullStep=kFALSE; 
   if (is_geom) epsil=(cross)?1E-9:-1E-9;
   TGeoNode *old = fCurrentNode;
   if (fIsOutside) old = 0;
   for (Int_t i=0; i<3; i++) fPoint[i]+=(fStep+epsil)*fDirection[i];
   TGeoNode *current = FindNode();
   if (fIsOutside) current=0;
   if (is_geom) {
      fIsEntering = (current==old)?kFALSE:kTRUE;
      fIsExiting  = !fIsEntering;
      fIsOnBoundary = kTRUE;
   } else {
      fIsEntering = fIsExiting = kFALSE;
      fIsOnBoundary = kFALSE;
   }
   return current;
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
// Set the top volume and corresponding node as starting point of the geometry.
   if (fTopVolume==vol) return;
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
// Define different tracking media.
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
// Classify a given point. See TGeoChecker::CheckPoint().
   GetGeomPainter()->CheckPoint(x,y,z,option);
}

//-----------------------------------------------------------------------------
void TGeoManager::CheckGeometry(Option_t *option)
{
// Instanciate a TGeoChecker object and investigates the geometry according to
// option. Not implemented yet.
   // check shapes first
   fTopNode->CheckShapes();
}

//-----------------------------------------------------------------------------
void TGeoManager::UpdateCurrentPosition(Double_t *nextpoint)
{
// Computes and changes the current node according to the new position.
// Not implemented.
}

//-----------------------------------------------------------------------------
ULong_t TGeoManager::SizeOf(TGeoNode *node, Option_t *option)
{
// computes the total size in bytes of the branch starting with node.
// The option can specify if all the branch has to be parsed or only the node
   return 0;
}
