// @(#)root/geom:$Name:  $:$Id: TGeoManager.cxx,v 1.149 2006/05/02 18:25:35 brun Exp $
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
// hooks for tracking, such as media, materials, magnetic field or track state flags,
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
// defined geometrical objects having a given shape and medium and possibly
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
//   TGeoMaterial *matVacuum = new TGeoMaterial("Vacuum", 0,0,0);
//   TGeoMaterial *matAl = new TGeoMaterial("Al", 26.98,13,2.7);
//   //--- define some media
//   TGeoMedium *med;
//   TGeoMedium *Vacuum = new TGeoMedium(1, matVacuum);
//   TGeoMedium *Al = new TGeoMedium(2, matAl);
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
//   TGeoVolume *top = geom->MakeBox("TOP", Vacuum, 270., 270., 120.);
//   geom->SetTopVolume(top); // mandatory !
//   //--- build other container volumes
//   TGeoVolume *replica = geom->MakeBox("REPLICA", Vacuum,120,120,120);
//   replica->SetVisibility(kFALSE);
//   TGeoVolume *rootbox = geom->MakeBox("ROOT", Vacuum, 110., 50., 5.);
//   rootbox->SetVisibility(kFALSE); // this will hold word 'ROOT'
//
//   //--- make letter 'R'
//   TGeoVolume *R = geom->MakeBox("R", Vacuum, 25., 25., 5.);
//   R->SetVisibility(kFALSE);
//   TGeoVolume *bar1 = geom->MakeBox("bar1", Al, 5., 25, 5.);
//   bar1->SetLineColor(kRed);
//   R->AddNode(bar1, 1, tr1);
//   TGeoVolume *bar2 = geom->MakeBox("bar2", Al, 5., 5., 5.);
//   bar2->SetLineColor(kRed);
//   R->AddNode(bar2, 1, tr2);
//   R->AddNode(bar2, 2, tr3);
//   TGeoVolume *tub1 = geom->MakeTubs("tub1", Al, 5., 15., 5., 90., 270.);
//   tub1->SetLineColor(kRed);
//   R->AddNode(tub1, 1, tr4);
//   TGeoVolume *bar3 = geom->MakeArb8("bar3", Al, 5.);
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
//   TGeoVolume *O = geom->MakeBox("O", Vacuum, 25., 25., 5.);
//   O->SetVisibility(kFALSE);
//   TGeoVolume *bar4 = geom->MakeBox("bar4", Al, 5., 7.5, 5.);
//   bar4->SetLineColor(kYellow);
//   O->AddNode(bar4, 1, tr5);
//   O->AddNode(bar4, 2, tr6);
//   TGeoVolume *tub2 = geom->MakeTubs("tub1", Al, 7.5, 17.5, 5., 0., 180.);
//   tub2->SetLineColor(kYellow);
//   O->AddNode(tub2, 1, tr7);
//   O->AddNode(tub2, 2, combi1);
//
//   //--- make letter 'T'
//   TGeoVolume *T = geom->MakeBox("T", Vacuum, 25., 25., 5.);
//   T->SetVisibility(kFALSE);
//   TGeoVolume *bar5 = geom->MakeBox("bar5", Al, 5., 20., 5.);
//   bar5->SetLineColor(kBlue);
//   T->AddNode(bar5, 1, tr8);
//   TGeoVolume *bar6 = geom->MakeBox("bar6", Al, 17.5, 5., 5.);
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
// media, materials, transformations, shapes and volumes. Logical nodes (positioned
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
// general rules : volumes needs media and shapes in order to be created,
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
// transformations and media, the top volume and the top logical node. These last
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
#include "TGeoManager.h"
#include "TSystem.h"
#include "TStyle.h"
#include "TVirtualPad.h"
#include "TBrowser.h"
#include "TFile.h"
#include "TKey.h"
#include "THashList.h"
#include "TClass.h"

#include "TGeoElement.h"
#include "TGeoMaterial.h"
#include "TGeoMedium.h"
#include "TGeoMatrix.h"
#include "TGeoNode.h"
#include "TGeoPhysicalNode.h"
#include "TGeoManager.h"
#include "TGeoPara.h"
#include "TGeoParaboloid.h"
#include "TGeoTube.h"
#include "TGeoEltu.h"
#include "TGeoHype.h"
#include "TGeoCone.h"
#include "TGeoSphere.h"
#include "TGeoArb8.h"
#include "TGeoPgon.h"
#include "TGeoTrd1.h"
#include "TGeoTrd2.h"
#include "TGeoTorus.h"
#include "TGeoXtru.h"
#include "TGeoCompositeShape.h"
#include "TGeoBoolNode.h"
#include "TVirtualGeoPainter.h"
#include "TPluginManager.h"
#include "TVirtualGeoTrack.h"
#include "TQObject.h"

// statics and globals

TGeoManager *gGeoManager = 0;

const char *kGeoOutsidePath = " ";
const Int_t kN3 = 3*sizeof(Double_t);

ClassImp(TGeoManager)

Bool_t TGeoManager::fgLock = kFALSE;

//_____________________________________________________________________________
TGeoManager::TGeoManager()
{
// Default constructor.
   if (TClass::IsCallingNew() == TClass::kDummyNew) {
      fTimeCut = kFALSE;
      fTmin = 0.;
      fTmax = 999.;
      fPhiCut = kFALSE;
      fPhimin = 0;
      fPhimax = 360;
      fDrawExtra = kFALSE;
      fStreamVoxels = kFALSE;
      fIsGeomReading = kFALSE;
      fSearchOverlaps = kFALSE;
      fLoopVolumes = kFALSE;
      fStartSafe = kTRUE;
      fSafety = 0;
      fLastSafety = 0.;
      fStep = 0;
      fBits = 0;
      fMaterials = 0;
      fHashPNE = 0;
      fMatrices = 0;
      fNodes = 0;
      fOverlaps = 0;
      fNNodes = 0;
      fLevel = 0;
      fNmany = 0;
      fMaxVisNodes = 10000;
   
      memset(fLastPoint, 0, kN3);
      fPoint = 0;
      fDirection = 0;
      fCldirChecked = 0;
      memset(fNormal, 0, kN3);
      fCldir = 0;
      fVolumes = 0;
      fPhysicalNodes = 0;
      fShapes = 0;
      fGVolumes = 0;
      fGShapes = 0;
      fTracks = 0;
      fMedia = 0;
      fNtracks = 0;
      fNpdg = 0;
      fPdgNames = 0;
      fCurrentTrack = 0;
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
      fActivity = kFALSE;
      fIsEntering = kFALSE;
      fIsExiting = kFALSE;
      fIsNodeSelectable = kFALSE;
      fIsStepEntering = kFALSE;
      fIsStepExiting = kFALSE;
      fIsOutside = kFALSE;
      fIsOnBoundary = kFALSE;
      fIsSameLocation = kTRUE;
      fIsNullStep = kFALSE;
      fVisDensity = 0.;
      fVisLevel = 3;
      fVisOption = 1;
      fExplodedView = 0;
      fNsegments = 20;
      fNLevel = 0;
      fCurrentMatrix = 0;
      fUniqueVolumes = 0;
      fNodeIdArray = 0;
      fClippingShape = 0;
      fIntSize = fDblSize = 1000;
      fIntBuffer = 0;
      fDblBuffer = 0;
      fOverlapMark = 0;
      fOverlapSize = 1000;
      fOverlapClusters = 0;
      fMatrixTransform = kFALSE;
      fMatrixReflection = kFALSE;
      fGLMatrix = 0;
      fPaintVolume = 0;
      fElementTable = 0;
      fHashVolumes = 0;
      fHashGVolumes = 0;
      //gGeoManager = this;
   } else {
      Init();
      gGeoIdentity = 0;
   }
}

//_____________________________________________________________________________
TGeoManager::TGeoManager(const char *name, const char *title)
            :TNamed(name, title)
{
// Constructor.
   Init();
   gGeoIdentity = new TGeoIdentity("Identity");
   BuildDefaultMaterials();
   Info("TGeoManager","Geometry %s, %s created", GetName(), GetTitle());
}

//_____________________________________________________________________________
void TGeoManager::Init()
{
// Initialize manager class.

   if (gGeoManager) {
      Warning("Init","Deleting previous geometry: %s/%s",gGeoManager->GetName(),gGeoManager->GetTitle());
      delete gGeoManager;
      if (fgLock) Fatal("Init", "New geometry created while the old one locked !!!");
   }

   gGeoManager = this;
   fTimeCut = kFALSE;
   fTmin = 0.;
   fTmax = 999.;
   fPhiCut = kFALSE;
   fPhimin = 0;
   fPhimax = 360;
   fDrawExtra = kFALSE;
   fStreamVoxels = kFALSE;
   fIsGeomReading = kFALSE;
   fSearchOverlaps = kFALSE;
   fLoopVolumes = kFALSE;
   fStartSafe = kTRUE;
   fSafety = 0;
   fLastSafety = 0.;
   fStep = 0;
   fBits = new UChar_t[50000]; // max 25000 nodes per volume
   fHashPNE = new THashList(256,3);
   fMaterials = new THashList(200,3);
   fMatrices = new TObjArray(256);
   fNodes = new TObjArray(30);
   fOverlaps = new TObjArray(256);
   fNNodes = 0;
   fLevel = 0;
   fNmany = 0;
   fMaxVisNodes = 10000;
   
   memset(fLastPoint, 0, kN3);
   fPoint = new Double_t[3];
   fDirection = new Double_t[3];
   fCldirChecked = new Double_t[3];
   memset(fNormal, 0, kN3);
   fCldir = new Double_t[3];
   fVolumes = new TObjArray(256);
   fPhysicalNodes = new TObjArray(256);
   fShapes = new TObjArray(256);
   fGVolumes = new TObjArray(256);
   fGShapes = new TObjArray(256);
   fTracks = new TObjArray(256);
   fMedia = new THashList(200,3);
   fNtracks = 0;
   fNpdg = 0;
   fPdgNames = 0;
   fCurrentTrack = 0;
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
   fActivity = kFALSE;
   fIsEntering = kFALSE;
   fIsExiting = kFALSE;
   fIsStepEntering = kFALSE;
   fIsStepExiting = kFALSE;
   fIsNodeSelectable = kFALSE;
   fIsOutside = kFALSE;
   fIsOnBoundary = kFALSE;
   fIsSameLocation = kTRUE;
   fIsNullStep = kFALSE;
   fVisDensity = 0.;
   fVisLevel = 3;
   fVisOption = 1;
   fExplodedView = 0;
   fNsegments = 20;
   fNLevel = 0;
   fCurrentMatrix = 0;
   fUniqueVolumes = new TObjArray(256);
   fNodeIdArray = 0;
   fClippingShape = 0;
   fIntSize = fDblSize = 1000;
   fIntBuffer = new Int_t[1000];
   fDblBuffer = new Double_t[1000];
   fOverlapMark = 0;
   fOverlapSize = 1000;
   fOverlapClusters = new Int_t[fOverlapSize];
   fMatrixTransform = kFALSE;
   fMatrixReflection = kFALSE;
   fGLMatrix = new TGeoHMatrix();
   fPaintVolume = 0;
   fElementTable = 0;
   fHashVolumes = 0;
   fHashGVolumes = 0;
}

//_____________________________________________________________________________
TGeoManager::~TGeoManager()
{
// Destructor
   if (gGeoManager != this) gGeoManager = this;

   gROOT->GetListOfBrowsables()->Remove(this);
   TSeqCollection *brlist = gROOT->GetListOfBrowsers();
   TIter next(brlist);
   TBrowser *browser = 0;
   while ((browser=(TBrowser*)next())) browser->RecursiveRemove(this);
   delete [] fBits;
   if (fCache) delete fCache;
   if (fNodes) delete fNodes;
   if (fTopNode) delete fTopNode;
   if (fOverlaps) {fOverlaps->Delete(); delete fOverlaps;}
   if (fMaterials) {fMaterials->Delete(); delete fMaterials;}
   if (fElementTable) delete fElementTable;
   if (fMedia) {fMedia->Delete(); delete fMedia;}
   if (fHashVolumes) delete fHashVolumes;
   if (fHashGVolumes) delete fHashGVolumes;
   if (fHashPNE) {fHashPNE->Delete(); delete fHashPNE;}
   if (fVolumes) {fVolumes->Delete(); delete fVolumes;}
   fVolumes = 0;
   if (fShapes) {fShapes->Delete(); delete fShapes;}
   if (fPhysicalNodes) {fPhysicalNodes->Delete(); delete fPhysicalNodes;}
   if (fMatrices) {fMatrices->Delete(); delete fMatrices;}
   if (fTracks) {fTracks->Delete(); delete fTracks;}
   if (fUniqueVolumes) delete fUniqueVolumes;
   if (fPdgNames) {fPdgNames->Delete(); delete fPdgNames;}
   CleanGarbage();
   if (fPainter) delete fPainter;
   delete [] fPoint;
   delete [] fDirection;
   delete [] fCldirChecked;
   delete [] fCldir;
   delete [] fDblBuffer;
   delete [] fIntBuffer;
   delete [] fOverlapClusters;
   delete fGLMatrix;
   gGeoIdentity = 0;
   gGeoManager = 0;
}

//_____________________________________________________________________________
Int_t TGeoManager::AddMaterial(const TGeoMaterial *material)
{
// Add a material to the list. Returns index of the material in list.
   if (!material) {
      Error("AddMaterial", "invalid material");
      return -1;
   }
   Int_t index = fMaterials->GetSize();
   ((TGeoMaterial*)material)->SetIndex(index);
   fMaterials->Add((TGeoMaterial*)material);
   return index;
}

//_____________________________________________________________________________
Int_t TGeoManager::AddOverlap(const TNamed *ovlp)
{
// Add an illegal overlap/extrusion to the list.
   Int_t size = fOverlaps->GetEntriesFast();
   fOverlaps->Add((TObject*)ovlp);
   return size;
}

//_____________________________________________________________________________
Int_t TGeoManager::AddTransformation(const TGeoMatrix *matrix)
{
// Add a matrix to the list. Returns index of the matrix in list.
   if (!matrix) {
      Error("AddMatrix", "invalid matrix");
      return -1;
   }
   Int_t index = fMatrices->GetEntriesFast();
   fMatrices->AddAtAndExpand((TGeoMatrix*)matrix,index);
   return index;
}
//_____________________________________________________________________________
Int_t TGeoManager::AddShape(const TGeoShape *shape)
{
// Add a shape to the list. Returns index of the shape in list.
   if (!shape) {
      Error("AddShape", "invalid shape");
      return -1;
   }
   TObjArray *list = fShapes;
   if (shape->IsRunTimeShape()) list = fGShapes;;
   Int_t index = list->GetEntriesFast();
   list->AddAtAndExpand((TGeoShape*)shape,index);
   return index;
}
//_____________________________________________________________________________
Int_t TGeoManager::AddTrack(Int_t id, Int_t pdgcode, TObject *particle)
{
// Add a track to the list of tracks
   Int_t index = fNtracks;
   fTracks->AddAtAndExpand(GetGeomPainter()->AddTrack(id,pdgcode,particle),fNtracks++);
   return index;
}

//_____________________________________________________________________________
TVirtualGeoTrack *TGeoManager::MakeTrack(Int_t id, Int_t pdgcode, TObject *particle)
{
// Makes a primary track but do not attach it to the list of tracks. The track
// can be attached as daughter to another one with TVirtualGeoTrack::AddTrack
   TVirtualGeoTrack *track = GetGeomPainter()->AddTrack(id,pdgcode,particle);
   return track;
}

//_____________________________________________________________________________
Int_t TGeoManager::AddVolume(TGeoVolume *volume)
{
// Add a volume to the list. Returns index of the volume in list.
   if (!volume) {
      Error("AddVolume", "invalid volume");
      return -1;
   }
   Int_t uid = fUniqueVolumes->GetEntriesFast();
   if (!uid) uid++;
   if (!fCurrentVolume) {
      fCurrentVolume = volume;
      fUniqueVolumes->AddAtAndExpand(volume,uid);
   } else {
      if (!strcmp(volume->GetName(), fCurrentVolume->GetName())) {
         uid = fCurrentVolume->GetNumber();
      } else {
         fCurrentVolume = volume;
         Int_t olduid = GetUID(volume->GetName());
         if (olduid<0) {
            fUniqueVolumes->AddAtAndExpand(volume,uid);
         } else {
            uid = olduid;
         }
      }
   }
   volume->SetNumber(uid);                           
   if (!fHashVolumes) {
      fHashVolumes = new THashList(256);
      fHashGVolumes = new THashList(256);
   }   
   TObjArray *list = fVolumes;
   if (!volume->GetShape() || volume->IsRunTime() || volume->IsVolumeMulti()) {
      list = fGVolumes;
      fHashGVolumes->Add(volume);
   } else {
      fHashVolumes->Add(volume);   
   }   
   Int_t index = list->GetEntriesFast();
   list->AddAtAndExpand(volume,index);
   return uid;
}
//_____________________________________________________________________________
void TGeoManager::Browse(TBrowser *b)
{
// Describe how to browse this object.
   if (!b) return;
   if (fMaterials) b->Add(fMaterials, "Materials");
   if (fMedia)     b->Add(fMedia,     "Media");
   if (fMatrices)  b->Add(fMatrices, "Local transformations");
   if (fOverlaps)  b->Add(fOverlaps, "Illegal overlaps");
   if (fTracks)    b->Add(fTracks,   "Tracks");
   if (fMasterVolume) b->Add(fMasterVolume, "Master Volume", fMasterVolume->IsVisible());
   if (fTopVolume) b->Add(fTopVolume, "Top Volume", fTopVolume->IsVisible());
   if (fTopNode)   b->Add(fTopNode);
   TQObject::Connect("TRootBrowser", "Checked(TObject*,Bool_t)", 
                     "TGeoManager", this, "SetVisibility(TObject*,Bool_t)");
}

//_____________________________________________________________________________
void TGeoManager::Edit(Option_t *option) {
// Append a pad for this geometry.
   AppendPad("");
   GetGeomPainter()->EditGeometry(option);
}   

//_____________________________________________________________________________
void TGeoManager::SetVisibility(TObject *obj, Bool_t vis)
{
// Set visibility for a volume.
   if(obj->IsA() != TGeoVolume::Class()) return;
   TGeoVolume *vol = (TGeoVolume *) obj;
   vol->SetVisibility(vis);
}  

//_____________________________________________________________________________
void TGeoManager::BombTranslation(const Double_t *tr, Double_t *bombtr)
{
// Get the new 'bombed' translation vector according current exploded view mode.
   if (fPainter) fPainter->BombTranslation(tr, bombtr);
   return;
}

//_____________________________________________________________________________
void TGeoManager::UnbombTranslation(const Double_t *tr, Double_t *bombtr)
{
// Get the new 'unbombed' translation vector according current exploded view mode.
   if (fPainter) fPainter->UnbombTranslation(tr, bombtr);
   return;
}

//_____________________________________________________________________________
void TGeoManager::BuildCache(Bool_t dummy, Bool_t nodeid)
{
// Builds the cache for physical nodes and global matrices.
   static Bool_t first = kTRUE;
   if (!fCache) {
      if (!fNLevel) {
         fNLevel = 100;
         if (first) Info("BuildCache","--- Maximum geometry depth set to 100");
      } else {
         if (first) Info("BuildCache","--- Maximum geometry depth is %i", fNLevel);   
      }   
      if (fNNodes>5000000 || dummy)  // temporary - works without
         // build dummy cache
         fCache = new TGeoCacheDummy(fTopNode, nodeid, fNLevel+1);
      else
         // build real cache
         fCache = new TGeoNodeCache(fNLevel+1,nodeid);
   }
   first = kFALSE;
}

//_____________________________________________________________________________
void TGeoManager::BuildIdArray()
{
// Builds node id array.
   if (fCache) fCache->BuildIdArray();
}

//_____________________________________________________________________________
void TGeoManager::RegisterMatrix(const TGeoMatrix *matrix)
{
// Register a matrix to the list of matrices. It will be cleaned-up at the
// destruction TGeoManager.
   if (matrix->IsRegistered()) return;
   TGeoMatrix *mat = (TGeoMatrix*)matrix;
   Int_t nmat = fMatrices->GetEntriesFast();
   fMatrices->AddAtAndExpand(mat, nmat);
}

//_____________________________________________________________________________
Int_t TGeoManager::ReplaceVolume(TGeoVolume *vorig, TGeoVolume *vnew) 
{
// Replaces all occurences of VORIG with VNEW in the geometry tree. The volume VORIG
// is not replaced from the list of volumes, but all node referencing it will reference
// VNEW instead. Returns number of occurences changed.
   Int_t nref = 0;
   if (!vorig || !vnew) return nref;
   TGeoMedium *morig = vorig->GetMedium();
   Bool_t checkmed = kFALSE;
   if (morig) checkmed = kTRUE;
   TGeoMedium *mnew = vnew->GetMedium();
   // Try to limit the damage produced by incorrect usage.
   if (!mnew && !vnew->IsAssembly()) {
      Error("ReplaceVolume","Replacement volume %s has no medium and it is not an assembly",
             vnew->GetName());              
      return nref;       
   }          
   if (mnew && checkmed) {
      if (mnew->GetId() != morig->GetId())
         Warning("ReplaceVolume","Replacement volume %s has different medium than original volume %s",
                 vnew->GetName(), vorig->GetName());
      checkmed = kFALSE;
   }
   
   // Medium checking now performed only if replacement is an assembly and old volume a real one.
   // Check result is dependent on positioning.
   Int_t nvol = fVolumes->GetEntriesFast();
   Int_t i,j,nd;
   Int_t ierr = 0;
   TGeoVolume *vol;
   TGeoNode *node;
   TGeoVoxelFinder *voxels;
   for (i=0; i<nvol; i++) {
      vol = (TGeoVolume*)fVolumes->At(i);
      if (!vol) continue;
      if (vol==vorig || vol==vnew) continue;
      nd = vol->GetNdaughters();
      for (j=0; j<nd; j++) {
         node = vol->GetNode(j);
         if (node->GetVolume() == vorig) {
            if (checkmed) {
               mnew = node->GetMotherVolume()->GetMedium();
               if (mnew && mnew->GetId()!=morig->GetId()) ierr++;
            }
            nref++;
            if (node->IsOverlapping()) {
               node->SetOverlapping(kFALSE);
               Info("ReplaceVolume","%s replaced with assembly and declared NON-OVERLAPPING!",node->GetName());
            }   
            node->SetVolume(vnew);
            voxels = node->GetMotherVolume()->GetVoxels();
            if (voxels) voxels->SetNeedRebuild();
         } else {
            if (node->GetMotherVolume() == vorig) {
               nref++;
               node->SetMotherVolume(vnew);
               if (node->IsOverlapping()) {
                  node->SetOverlapping(kFALSE);
                  Info("ReplaceVolume","%s inside substitute assembly %s declared NON-OVERLAPPING!",node->GetName(),vnew->GetName());
               }   
            }
         }      
      }
   }
   if (ierr) Warning("ReplaceVolume", "Volumes should not be replaced with assemblies if they are positioned in containers having a different medium ID.\n %i occurences for assembly replacing volume %s", 
                     ierr, vorig->GetName());
   return nref;
}         
         
//_____________________________________________________________________________
Int_t TGeoManager::TransformVolumeToAssembly(const char *vname)
{
// Transform all volumes named VNAME to assemblies. The volumes must be virtual.
   TGeoVolume *toTransform = FindVolumeFast(vname);
   if (!toTransform) {
      Warning("TransformVolumeToAssembly", "Volume %s not found", vname);
      return 0;
   }
   Int_t index = fVolumes->IndexOf(toTransform);
   Int_t count = 0;
   Int_t indmax = fVolumes->GetEntries();     
   Bool_t replace = kTRUE;
   TGeoVolume *transformed;
   while (index<indmax) {
      if (replace) {
         replace = kFALSE;
         transformed = TGeoVolumeAssembly::MakeAssemblyFromVolume(toTransform);
         if (transformed) {
            ReplaceVolume(toTransform, transformed);
            count++;
         } else {
            if (toTransform->IsAssembly())
               Warning("TransformVolumeToAssembly", "Volume %s already assembly", toTransform->GetName());
            if (!toTransform->GetNdaughters())
               Warning("TransformVolumeToAssembly", "Volume %s has no daughters, cannot transform", toTransform->GetName());
            if (toTransform->IsVolumeMulti())
               Warning("TransformVolumeToAssembly", "Volume %s divided, cannot transform", toTransform->GetName());
         }   
      }   
      index++;
      if (index >= indmax) return count;
      toTransform = (TGeoVolume*)fVolumes->At(index);
      if (!strcmp(toTransform->GetName(),vname)) replace = kTRUE;
   }
   return count;   
}

//_____________________________________________________________________________
TGeoVolume *TGeoManager::Division(const char *name, const char *mother, Int_t iaxis,
                                  Int_t ndiv, Double_t start, Double_t step, Int_t numed, Option_t *option)
{
// Create a new volume by dividing an existing one (GEANT3 like)
//
// Divides MOTHER into NDIV divisions called NAME
// along axis IAXIS starting at coordinate value START
// and having size STEP. The created volumes will have tracking
// media ID=NUMED (if NUMED=0 -> same media as MOTHER)
//    The behavior of the division operation can be triggered using OPTION :
// OPTION (case insensitive) :
//  N  - divide all range in NDIV cells (same effect as STEP<=0) (GSDVN in G3)
//  NX - divide range starting with START in NDIV cells          (GSDVN2 in G3)
//  S  - divide all range with given STEP. NDIV is computed and divisions will be centered
//         in full range (same effect as NDIV<=0)                (GSDVS, GSDVT in G3)
//  SX - same as DVS, but from START position.                   (GSDVS2, GSDVT2 in G3)

   TGeoVolume *amother;
   TString sname = name;
   sname = sname.Strip();
   const char *vname = sname.Data();
   TString smname = mother;
   smname = smname.Strip();
   const char *mname = smname.Data();
   
   amother = (TGeoVolume*)fGVolumes->FindObject(mname);
   if (!amother) amother = GetVolume(mname);
   if (amother) return amother->Divide(vname,iaxis,ndiv,start,step,numed, option);

   Error("Division","VOLUME: \"%s\" not defined",mname);

   return 0;
}
//_____________________________________________________________________________
void TGeoManager::Matrix(Int_t index, Double_t theta1, Double_t phi1,
                         Double_t theta2, Double_t phi2,
                         Double_t theta3, Double_t phi3)
{
// Create rotation matrix named 'mat<index>'.
//
//  index    rotation matrix number
//  theta1   polar angle for axis X
//  phi1     azimuthal angle for axis X
//  theta2   polar angle for axis Y
//  phi2     azimuthal angle for axis Y
//  theta3   polar angle for axis Z
//  phi3     azimuthal angle for axis Z
//
   TGeoRotation * rot = new TGeoRotation("",theta1,phi1,theta2,phi2,theta3,phi3);
   rot->SetUniqueID(index);
   rot->RegisterYourself();
}

//_____________________________________________________________________________
TGeoMaterial *TGeoManager::Material(const char *name, Double_t a, Double_t z, Double_t dens, Int_t uid,Double_t radlen, Double_t intlen)
{
// Create material with given A, Z and density, having an unique id.
   TGeoMaterial *material = new TGeoMaterial(name,a,z,dens,radlen,intlen);
   material->SetUniqueID(uid);
   return material;
}

//_____________________________________________________________________________
TGeoMaterial *TGeoManager::Mixture(const char *name, Float_t *a, Float_t *z, Double_t dens,
                                   Int_t nelem, Float_t *wmat, Int_t uid)
{
// Create mixture OR COMPOUND IMAT as composed by THE BASIC nelem
// materials defined by arrays A,Z and WMAT, having an unique id.
   TGeoMixture *mix = new TGeoMixture(name,nelem,dens);
   mix->SetUniqueID(uid);
   Int_t i;
   for (i=0;i<nelem;i++) {
      mix->DefineElement(i,a[i],z[i],wmat[i]);
   }
   return (TGeoMaterial*)mix;
}

//_____________________________________________________________________________
TGeoMaterial *TGeoManager::Mixture(const char *name, Double_t *a, Double_t *z, Double_t dens,
                                   Int_t nelem, Double_t *wmat, Int_t uid)
{
// Create mixture OR COMPOUND IMAT as composed by THE BASIC nelem
// materials defined by arrays A,Z and WMAT, having an unique id.
   TGeoMixture *mix = new TGeoMixture(name,nelem,dens);
   mix->SetUniqueID(uid);
   Int_t i;
   for (i=0;i<nelem;i++) {
      mix->DefineElement(i,a[i],z[i],wmat[i]);
   }
   return (TGeoMaterial*)mix;
}

//_____________________________________________________________________________
TGeoMedium *TGeoManager::Medium(const char *name, Int_t numed, Int_t nmat, Int_t isvol,
                                Int_t ifield, Double_t fieldm, Double_t tmaxfd,
                                Double_t stemax, Double_t deemax, Double_t epsil,
                                Double_t stmin)
{
// Create tracking medium
  //
  //  numed      tracking medium number assigned
  //  name      tracking medium name
  //  nmat      material number
  //  isvol     sensitive volume flag
  //  ifield    magnetic field
  //  fieldm    max. field value (kilogauss)
  //  tmaxfd    max. angle due to field (deg/step)
  //  stemax    max. step allowed
  //  deemax    max. fraction of energy lost in a step
  //  epsil     tracking precision (cm)
  //  stmin     min. step due to continuous processes (cm)
  //
  //  ifield = 0 if no magnetic field; ifield = -1 if user decision in guswim;
  //  ifield = 1 if tracking performed with g3rkuta; ifield = 2 if tracking
  //  performed with g3helix; ifield = 3 if tracking performed with g3helx3.
  //
   return new TGeoMedium(name,numed,nmat,isvol,ifield,fieldm,tmaxfd,stemax,deemax,epsil,stmin);
}

//_____________________________________________________________________________
void TGeoManager::Node(const char *name, Int_t nr, const char *mother,
                       Double_t x, Double_t y, Double_t z, Int_t irot,
                       Bool_t isOnly, Float_t *upar, Int_t npar)
{
// Create a node called <name_nr> pointing to the volume called <name>
// as daughter of the volume called <mother> (gspos). The relative matrix is
// made of : a translation (x,y,z) and a rotation matrix named <matIROT>.
// In case npar>0, create the volume to be positioned in mother, according
// its actual parameters (gsposp).
//  NAME   Volume name
//  NUMBER Copy number of the volume
//  MOTHER Mother volume name
//  X      X coord. of the volume in mother ref. sys.
//  Y      Y coord. of the volume in mother ref. sys.
//  Z      Z coord. of the volume in mother ref. sys.
//  IROT   Rotation matrix number w.r.t. mother ref. sys.
//  ISONLY ONLY/MANY flag
   TGeoVolume *amother= 0;
   TGeoVolume *volume = 0;
   
   // look into special volume list first
   amother = FindVolumeFast(mother,kTRUE);
   if (!amother) amother = FindVolumeFast(mother);
   if (!amother) {
      TString mname = mother;
      mname = mname.Strip();
      Error("Node","Mother VOLUME \"%s\" not defined",mname.Data());
      return;
   }
   Int_t i;
   if (npar<=0) {
   //---> acting as G3 gspos
      if (gDebug > 0) Info("Node","Calling gspos, mother=%s, name=%s, nr=%d, x=%g, y=%g, z=%g, irot=%d, konly=%i",mother,name,nr,x,y,z,irot,(Int_t)isOnly);
      // look into special volume list first
      volume  = FindVolumeFast(name,kTRUE);
      if (!volume) volume = FindVolumeFast(name);
      if (!volume) {
         TString vname = name;
         vname = vname.Strip();
         Error("Node","VOLUME: \"%s\" not defined",vname.Data());
         return;
      }
      if (((TObject*)volume)->TestBit(TGeoVolume::kVolumeMulti) && !volume->GetShape()) {
         Error("Node", "cannot add multiple-volume object %s as node", volume->GetName());
         return;
      }
   } else {
   //---> acting as G3 gsposp
      TGeoVolumeMulti *vmulti  = (TGeoVolumeMulti*)FindVolumeFast(name, kTRUE);
      if (!vmulti) {
         volume = FindVolumeFast(name);
         if (volume) {
            Warning("Node", "volume: %s is defined as single -> ignoring shape parameters", volume->GetName());
            Node(name,nr,mother,x,y,z,irot,isOnly, upar);
            return;
         }
         TString vname = name;
         vname = vname.Strip();         
         Error("Node","VOLUME: \"%s\" not defined ",vname.Data());
         return;
      }
      TGeoMedium *medium = vmulti->GetMedium();
      TString sh    = vmulti->GetTitle();
      sh.ToLower();
      if (sh.Contains("box")) {
         volume = MakeBox(name,medium,upar[0],upar[1],upar[2]);
      } else if (sh.Contains("trd1")) {
         volume = MakeTrd1(name,medium,upar[0],upar[1],upar[2],upar[3]);
      } else if (sh.Contains("trd2")) {
         volume = MakeTrd2(name,medium,upar[0],upar[1],upar[2],upar[3],upar[4]);
      } else if (sh.Contains("trap")) {
         volume = MakeTrap(name,medium,upar[0],upar[1],upar[2],upar[3],upar[4],upar[5],upar[6],upar[7],upar[8],upar[9],upar[10]);
      } else if (sh.Contains("gtra")) {
         volume = MakeGtra(name,medium,upar[0],upar[1],upar[2],upar[3],upar[4],upar[5],upar[6],upar[7],upar[8],upar[9],upar[10],upar[11]);
      } else if (sh.Contains("tube")) {
         volume = MakeTube(name,medium,upar[0],upar[1],upar[2]);
      } else if (sh.Contains("tubs")) {
         volume = MakeTubs(name,medium,upar[0],upar[1],upar[2],upar[3],upar[4]);
      } else if (sh.Contains("cone")) {
         volume = MakeCone(name,medium,upar[0],upar[1],upar[2],upar[3],upar[4]);
      } else if (sh.Contains("cons")) {
         volume = MakeCons(name,medium,upar[0],upar[1],upar[2],upar[3],upar[4],upar[5],upar[6]);
      } else if (sh.Contains("pgon")) {
         volume = MakePgon(name,medium,upar[0],upar[1],(Int_t)upar[2],(Int_t)upar[3]);
         Int_t nz = (Int_t)upar[3];
         for (i=0;i<nz;i++) {
            ((TGeoPgon*)volume->GetShape())->DefineSection(i,upar[3*i+4],upar[3*i+5],upar[3*i+6]);
         }
      } else if (sh.Contains("pcon")) {
         volume = MakePcon(name,medium,upar[0],upar[1],(Int_t)upar[2]);
         Int_t nz = (Int_t)upar[2];
         for (i=0;i<nz;i++) {
            ((TGeoPcon*)volume->GetShape())->DefineSection(i,upar[3*i+3],upar[3*i+4],upar[3*i+5]);
         }
      } else if (sh.Contains("eltu")) {
         volume = MakeEltu(name,medium,upar[0],upar[1],upar[2]);
      } else if (sh.Contains("sphe")) {
         volume = MakeSphere(name,medium,upar[0],upar[1],upar[2],upar[3],upar[4],upar[5]);
      } else if (sh.Contains("ctub")) {
         volume = MakeCtub(name,medium,upar[0],upar[1],upar[2],upar[3],upar[4],upar[5],upar[6],upar[7],upar[8],upar[9],upar[10]);
      } else if (sh.Contains("para")) {
         volume = MakePara(name,medium,upar[0],upar[1],upar[2],upar[3],upar[4],upar[5]);
      } else {
         Error("Node","cannot create shape %s",sh.Data());
      }

      if (!volume) return;
      vmulti->AddVolume(volume);
   }
   if (irot) {
      TGeoRotation *matrix = 0;
      TGeoMatrix *mat;
      TIter next(fMatrices);
      while ((mat=(TGeoMatrix*)next())) {
         if (mat->GetUniqueID()==UInt_t(irot)) {
            matrix = (TGeoRotation*)mat;
            break;
         }
      }
      if (!matrix) {
         Error("Node", "rotation %i not found", irot);
         return;
      }
      if (isOnly) amother->AddNode(volume,nr,new TGeoCombiTrans(x,y,z,matrix));
      else        amother->AddNodeOverlap(volume,nr,new TGeoCombiTrans(x,y,z,matrix));
   } else {
      if (x == 0 && y== 0 && z == 0) {
         if (isOnly) amother->AddNode(volume,nr);
         else        amother->AddNodeOverlap(volume,nr);
      } else {
         if (isOnly) amother->AddNode(volume,nr,new TGeoTranslation(x,y,z));
         else        amother->AddNodeOverlap(volume,nr,new TGeoTranslation(x,y,z));
      }
   }
}

//_____________________________________________________________________________
void TGeoManager::Node(const char *name, Int_t nr, const char *mother,
                       Double_t x, Double_t y, Double_t z, Int_t irot,
                       Bool_t isOnly, Double_t *upar, Int_t npar)
{
// Create a node called <name_nr> pointing to the volume called <name>
// as daughter of the volume called <mother> (gspos). The relative matrix is
// made of : a translation (x,y,z) and a rotation matrix named <matIROT>.
// In case npar>0, create the volume to be positioned in mother, according
// its actual parameters (gsposp).
//  NAME   Volume name
//  NUMBER Copy number of the volume
//  MOTHER Mother volume name
//  X      X coord. of the volume in mother ref. sys.
//  Y      Y coord. of the volume in mother ref. sys.
//  Z      Z coord. of the volume in mother ref. sys.
//  IROT   Rotation matrix number w.r.t. mother ref. sys.
//  ISONLY ONLY/MANY flag
   TGeoVolume *amother= 0;
   TGeoVolume *volume = 0;
   
   // look into special volume list first
   amother = FindVolumeFast(mother,kTRUE);
   if (!amother) amother = FindVolumeFast(mother);
   if (!amother) {
      TString mname = mother;
      mname = mname.Strip();
      Error("Node","Mother VOLUME \"%s\" not defined",mname.Data());
      return;
   }
   Int_t i;
   if (npar<=0) {
   //---> acting as G3 gspos
      if (gDebug > 0) Info("Node","Calling gspos, mother=%s, name=%s, nr=%d, x=%g, y=%g, z=%g, irot=%d, konly=%i",mother,name,nr,x,y,z,irot,(Int_t)isOnly);
      // look into special volume list first
      volume  = FindVolumeFast(name,kTRUE);
      if (!volume) volume = FindVolumeFast(name);
      if (!volume) {
         TString vname = name;
         vname = vname.Strip();
         Error("Node","VOLUME: \"%s\" not defined",vname.Data());
         return;
      }
      if (((TObject*)volume)->TestBit(TGeoVolume::kVolumeMulti) && !volume->GetShape()) {
         Error("Node", "cannot add multiple-volume object %s as node", volume->GetName());
         return;
      }
   } else {
   //---> acting as G3 gsposp
      TGeoVolumeMulti *vmulti  = (TGeoVolumeMulti*)FindVolumeFast(name, kTRUE);
      if (!vmulti) {
         volume = FindVolumeFast(name);
         if (volume) {
            Warning("Node", "volume: %s is defined as single -> ignoring shape parameters", volume->GetName());
            Node(name,nr,mother,x,y,z,irot,isOnly, upar);
            return;
         }
         TString vname = name;
         vname = vname.Strip();         
         Error("Node","VOLUME: \"%s\" not defined ",vname.Data());
         return;
      }
      TGeoMedium *medium = vmulti->GetMedium();
      TString sh    = vmulti->GetTitle();
      sh.ToLower();
      if (sh.Contains("box")) {
         volume = MakeBox(name,medium,upar[0],upar[1],upar[2]);
      } else if (sh.Contains("trd1")) {
         volume = MakeTrd1(name,medium,upar[0],upar[1],upar[2],upar[3]);
      } else if (sh.Contains("trd2")) {
         volume = MakeTrd2(name,medium,upar[0],upar[1],upar[2],upar[3],upar[4]);
      } else if (sh.Contains("trap")) {
         volume = MakeTrap(name,medium,upar[0],upar[1],upar[2],upar[3],upar[4],upar[5],upar[6],upar[7],upar[8],upar[9],upar[10]);
      } else if (sh.Contains("gtra")) {
         volume = MakeGtra(name,medium,upar[0],upar[1],upar[2],upar[3],upar[4],upar[5],upar[6],upar[7],upar[8],upar[9],upar[10],upar[11]);
      } else if (sh.Contains("tube")) {
         volume = MakeTube(name,medium,upar[0],upar[1],upar[2]);
      } else if (sh.Contains("tubs")) {
         volume = MakeTubs(name,medium,upar[0],upar[1],upar[2],upar[3],upar[4]);
      } else if (sh.Contains("cone")) {
         volume = MakeCone(name,medium,upar[0],upar[1],upar[2],upar[3],upar[4]);
      } else if (sh.Contains("cons")) {
         volume = MakeCons(name,medium,upar[0],upar[1],upar[2],upar[3],upar[4],upar[5],upar[6]);
      } else if (sh.Contains("pgon")) {
         volume = MakePgon(name,medium,upar[0],upar[1],(Int_t)upar[2],(Int_t)upar[3]);
         Int_t nz = (Int_t)upar[3];
         for (i=0;i<nz;i++) {
            ((TGeoPgon*)volume->GetShape())->DefineSection(i,upar[3*i+4],upar[3*i+5],upar[3*i+6]);
         }
      } else if (sh.Contains("pcon")) {
         volume = MakePcon(name,medium,upar[0],upar[1],(Int_t)upar[2]);
         Int_t nz = (Int_t)upar[2];
         for (i=0;i<nz;i++) {
            ((TGeoPcon*)volume->GetShape())->DefineSection(i,upar[3*i+3],upar[3*i+4],upar[3*i+5]);
         }
      } else if (sh.Contains("eltu")) {
         volume = MakeEltu(name,medium,upar[0],upar[1],upar[2]);
      } else if (sh.Contains("sphe")) {
         volume = MakeSphere(name,medium,upar[0],upar[1],upar[2],upar[3],upar[4],upar[5]);
      } else if (sh.Contains("ctub")) {
         volume = MakeCtub(name,medium,upar[0],upar[1],upar[2],upar[3],upar[4],upar[5],upar[6],upar[7],upar[8],upar[9],upar[10]);
      } else if (sh.Contains("para")) {
         volume = MakePara(name,medium,upar[0],upar[1],upar[2],upar[3],upar[4],upar[5]);
      } else {
         Error("Node","cannot create shape %s",sh.Data());
      }

      if (!volume) return;
      vmulti->AddVolume(volume);
   }
   if (irot) {
      TGeoRotation *matrix = 0;
      TGeoMatrix *mat;
      TIter next(fMatrices);
      while ((mat=(TGeoMatrix*)next())) {
         if (mat->GetUniqueID()==UInt_t(irot)) {
            matrix = (TGeoRotation*)mat;
            break;
         }
      }
      if (!matrix) {
         Error("Node", "rotation %i not found", irot);
         return;
      }
      if (isOnly) amother->AddNode(volume,nr,new TGeoCombiTrans(x,y,z,matrix));
      else        amother->AddNodeOverlap(volume,nr,new TGeoCombiTrans(x,y,z,matrix));
   } else {
      if (x == 0 && y== 0 && z == 0) {
         if (isOnly) amother->AddNode(volume,nr);
         else        amother->AddNodeOverlap(volume,nr);
      } else {
         if (isOnly) amother->AddNode(volume,nr,new TGeoTranslation(x,y,z));
         else        amother->AddNodeOverlap(volume,nr,new TGeoTranslation(x,y,z));
      }
   }
}

//_____________________________________________________________________________
TGeoVolume *TGeoManager::Volume(const char *name, const char *shape, Int_t nmed,
                                Float_t *upar, Int_t npar)
{
// Create a volume in GEANT3 style.
//  NAME   Volume name
//  SHAPE  Volume type
//  NMED   Tracking medium number
//  NPAR   Number of shape parameters
//  UPAR   Vector containing shape parameters
   Int_t i;
   TGeoVolume *volume = 0;
   TGeoMedium *medium = GetMedium(nmed);
   if (!medium) {
      Error("Volume","cannot create volume: %s, medium: %d is unknown",name,nmed);
      return 0;
   }
   TString sh = shape;
   TString sname = name;
   sname = sname.Strip();
   const char *vname = sname.Data();
   if (npar <= 0) {
      //--- create a TGeoVolumeMulti
      volume = MakeVolumeMulti(vname,medium);
      volume->SetTitle(shape);
      TGeoVolumeMulti *vmulti  = (TGeoVolumeMulti*)fGVolumes->FindObject(vname);
      if (!vmulti) {
         Error("Volume","volume multi: %s not created",vname);
         return 0;
      }
      return vmulti;
   }
   //---> create a normal volume
   sh.ToLower();
   if (sh.Contains("box")) {
      volume = MakeBox(vname,medium,upar[0],upar[1],upar[2]);
   } else if (sh.Contains("trd1")) {
      volume = MakeTrd1(vname,medium,upar[0],upar[1],upar[2],upar[3]);
   } else if (sh.Contains("trd2")) {
      volume = MakeTrd2(vname,medium,upar[0],upar[1],upar[2],upar[3],upar[4]);
   } else if (sh.Contains("trap")) {
      volume = MakeTrap(vname,medium,upar[0],upar[1],upar[2],upar[3],upar[4],upar[5],upar[6],upar[7],upar[8],upar[9],upar[10]);
   } else if (sh.Contains("gtra")) {
      volume = MakeGtra(vname,medium,upar[0],upar[1],upar[2],upar[3],upar[4],upar[5],upar[6],upar[7],upar[8],upar[9],upar[10],upar[11]);
   } else if (sh.Contains("tube")) {
      volume = MakeTube(vname,medium,upar[0],upar[1],upar[2]);
   } else if (sh.Contains("tubs")) {
      volume = MakeTubs(vname,medium,upar[0],upar[1],upar[2],upar[3],upar[4]);
   } else if (sh.Contains("cone")) {
      volume = MakeCone(vname,medium,upar[0],upar[1],upar[2],upar[3],upar[4]);
   } else if (sh.Contains("cons")) {
      volume = MakeCons(vname,medium,upar[0],upar[1],upar[2],upar[3],upar[4],upar[5],upar[6]);
   } else if (sh.Contains("pgon")) {
      volume = MakePgon(vname,medium,upar[0],upar[1],(Int_t)upar[2],(Int_t)upar[3]);
      Int_t nz = (Int_t)upar[3];
      for (i=0;i<nz;i++) {
         ((TGeoPgon*)volume->GetShape())->DefineSection(i,upar[3*i+4],upar[3*i+5],upar[3*i+6]);
      }
   } else if (sh.Contains("pcon")) {
      volume = MakePcon(vname,medium,upar[0],upar[1],(Int_t)upar[2]);
      Int_t nz = (Int_t)upar[2];
      for (i=0;i<nz;i++) {
         ((TGeoPcon*)volume->GetShape())->DefineSection(i,upar[3*i+3],upar[3*i+4],upar[3*i+5]);
      }
   } else if (sh.Contains("eltu")) {
      volume = MakeEltu(vname,medium,upar[0],upar[1],upar[2]);
   } else if (sh.Contains("sphe")) {
      volume = MakeSphere(vname,medium,upar[0],upar[1],upar[2],upar[3],upar[4],upar[5]);
   } else if (sh.Contains("ctub")) {
      volume = MakeCtub(vname,medium,upar[0],upar[1],upar[2],upar[3],upar[4],upar[5],upar[6],upar[7],upar[8],upar[9],upar[10]);
   } else if (sh.Contains("para")) {
      volume = MakePara(vname,medium,upar[0],upar[1],upar[2],upar[3],upar[4],upar[5]);
   }
   if (!volume) {
      Error("Volume","volume: %s not created",vname);
      return 0;
   }
   return volume;
}

//_____________________________________________________________________________
TGeoVolume *TGeoManager::Volume(const char *name, const char *shape, Int_t nmed,
                                Double_t *upar, Int_t npar)
{
// Create a volume in GEANT3 style.
//  NAME   Volume name
//  SHAPE  Volume type
//  NMED   Tracking medium number
//  NPAR   Number of shape parameters
//  UPAR   Vector containing shape parameters
   Int_t i;
   TGeoVolume *volume = 0;
   TGeoMedium *medium = GetMedium(nmed);
   if (!medium) {
      Error("Volume","cannot create volume: %s, medium: %d is unknown",name,nmed);
      return 0;
   }
   TString sh = shape;
   TString sname = name;
   sname = sname.Strip();
   const char *vname = sname.Data();
   if (npar <= 0) {
      //--- create a TGeoVolumeMulti
      volume = MakeVolumeMulti(vname,medium);
      volume->SetTitle(shape);
      TGeoVolumeMulti *vmulti  = (TGeoVolumeMulti*)fGVolumes->FindObject(vname);
      if (!vmulti) {
         Error("Volume","volume multi: %s not created",vname);
         return 0;
      }
      return vmulti;
   }
   //---> create a normal volume
   sh.ToLower();
   if (sh.Contains("box")) {
      volume = MakeBox(vname,medium,upar[0],upar[1],upar[2]);
   } else if (sh.Contains("trd1")) {
      volume = MakeTrd1(vname,medium,upar[0],upar[1],upar[2],upar[3]);
   } else if (sh.Contains("trd2")) {
      volume = MakeTrd2(vname,medium,upar[0],upar[1],upar[2],upar[3],upar[4]);
   } else if (sh.Contains("trap")) {
      volume = MakeTrap(vname,medium,upar[0],upar[1],upar[2],upar[3],upar[4],upar[5],upar[6],upar[7],upar[8],upar[9],upar[10]);
   } else if (sh.Contains("gtra")) {
      volume = MakeGtra(vname,medium,upar[0],upar[1],upar[2],upar[3],upar[4],upar[5],upar[6],upar[7],upar[8],upar[9],upar[10],upar[11]);
   } else if (sh.Contains("tube")) {
      volume = MakeTube(vname,medium,upar[0],upar[1],upar[2]);
   } else if (sh.Contains("tubs")) {
      volume = MakeTubs(vname,medium,upar[0],upar[1],upar[2],upar[3],upar[4]);
   } else if (sh.Contains("cone")) {
      volume = MakeCone(vname,medium,upar[0],upar[1],upar[2],upar[3],upar[4]);
   } else if (sh.Contains("cons")) {
      volume = MakeCons(vname,medium,upar[0],upar[1],upar[2],upar[3],upar[4],upar[5],upar[6]);
   } else if (sh.Contains("pgon")) {
      volume = MakePgon(vname,medium,upar[0],upar[1],(Int_t)upar[2],(Int_t)upar[3]);
      Int_t nz = (Int_t)upar[3];
      for (i=0;i<nz;i++) {
         ((TGeoPgon*)volume->GetShape())->DefineSection(i,upar[3*i+4],upar[3*i+5],upar[3*i+6]);
      }
   } else if (sh.Contains("pcon")) {
      volume = MakePcon(vname,medium,upar[0],upar[1],(Int_t)upar[2]);
      Int_t nz = (Int_t)upar[2];
      for (i=0;i<nz;i++) {
         ((TGeoPcon*)volume->GetShape())->DefineSection(i,upar[3*i+3],upar[3*i+4],upar[3*i+5]);
      }
   } else if (sh.Contains("eltu")) {
      volume = MakeEltu(vname,medium,upar[0],upar[1],upar[2]);
   } else if (sh.Contains("sphe")) {
      volume = MakeSphere(vname,medium,upar[0],upar[1],upar[2],upar[3],upar[4],upar[5]);
   } else if (sh.Contains("ctub")) {
      volume = MakeCtub(vname,medium,upar[0],upar[1],upar[2],upar[3],upar[4],upar[5],upar[6],upar[7],upar[8],upar[9],upar[10]);
   } else if (sh.Contains("para")) {
      volume = MakePara(vname,medium,upar[0],upar[1],upar[2],upar[3],upar[4],upar[5]);
   }
   if (!volume) {
      Error("Volume","volume: %s not created",vname);
      return 0;
   }
   return volume;
}

//_____________________________________________________________________________
void TGeoManager::SetAllIndex()
{
// Assigns uid's for all materials,media and matrices.
   Int_t index = 1;
   TIter next(fMaterials);
   TGeoMaterial *mater;
   while ((mater=(TGeoMaterial*)next())) {
      mater->SetUniqueID(index++);
      mater->ResetBit(TGeoMaterial::kMatSavePrimitive);
   }   
   index = 1;
   TIter next1(fMedia);
   TGeoMedium *med;
   while ((med=(TGeoMedium*)next1())) {
      med->SetUniqueID(index++);
      med->ResetBit(TGeoMedium::kMedSavePrimitive);
   }   
   index = 1;
   TIter next2(fShapes);
   TGeoShape *shape;
   while ((shape=(TGeoShape*)next2())) {
      shape->SetUniqueID(index++);
      if (shape->IsComposite()) ((TGeoCompositeShape*)shape)->GetBoolNode()->RegisterMatrices();
   }
      
   TIter next3(fMatrices);
   TGeoMatrix *matrix;
   while ((matrix=(TGeoMatrix*)next3())) {
      matrix->RegisterYourself();   
   }
   TIter next4(fMatrices);
   index = 1;
   while ((matrix=(TGeoMatrix*)next4())) {
      matrix->SetUniqueID(index++);   
      matrix->ResetBit(TGeoMatrix::kGeoSavePrimitive);
   }
   TIter next5(fVolumes);
   TGeoVolume *vol;
   while ((vol=(TGeoVolume*)next5())) vol->UnmarkSaved();
}

//_____________________________________________________________________________
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
//      vol->SetVisibility(kTRUE);
//      vol->SetVisDaughters(kTRUE);
//      vol->SetLineStyle(gStyle->GetLineStyle());
//      vol->SetLineWidth(gStyle->GetLineWidth());
      vol->SetVisTouched(kFALSE);
   }
}
//_____________________________________________________________________________
void TGeoManager::CloseGeometry(Option_t *option)
{
// Closing geometry implies checking the geometry validity, fixing shapes
// with negative parameters (run-time shapes)building the cache manager,
// voxelizing all volumes, counting the total number of physical nodes and
// registring the manager class to the browser.
   if (IsClosed()) {
      Warning("CloseGeometry", "geometry already closed");
      return;
   }
   if (!fMasterVolume) {
      Error("CloseGeometry","you MUST call SetTopVolume() first !");
      return;
   }
   if (!gROOT->GetListOfBrowsables()->FindObject(this)) gROOT->GetListOfBrowsables()->Add(this);
   TSeqCollection *brlist = gROOT->GetListOfBrowsers();
   TIter next(brlist);
   TBrowser *browser = 0;
   while ((browser=(TBrowser*)next())) browser->Refresh();
   TString opt(option);
   opt.ToLower();
   Bool_t dummy = opt.Contains("d");
   Bool_t nodeid = opt.Contains("i");
   if (fIsGeomReading) {
      Info("CloseGeometry","Geometry loaded from file...");
      gGeoIdentity=(TGeoIdentity *)fMatrices->At(0);
      if (!fElementTable) fElementTable = new TGeoElementTable(200);
      if (!fTopNode) {
         if (!fMasterVolume) {
            Error("CloseGeometry", "Master volume not streamed");
            return;
         }
         SetTopVolume(fMasterVolume);
         if (fStreamVoxels) Info("CloseGeometry","Voxelization retrieved from file");
         Voxelize("ALL");
         if (!fCache) BuildCache(dummy,nodeid);
      } else {
         Warning("CloseGeometry", "top node was streamed!");
         Voxelize("ALL");
         if (!fCache) BuildCache(dummy,nodeid);
      }
//      BuildIdArray();
      Info("CloseGeometry","%i nodes/ %i volume UID's in %s", fNNodes, fUniqueVolumes->GetEntriesFast()-1, GetTitle());
      Info("CloseGeometry","----------------modeler ready----------------");
      return;
   }

   SelectTrackingMedia();
   CheckGeometry();
   Info("CloseGeometry","Counting nodes...");
   fNNodes = CountNodes();
   fNLevel = fMasterVolume->CountNodes(1,3)+1;
//   BuildIdArray();
   Voxelize("ALL");
   Info("CloseGeometry","Building cache...");
   BuildCache(dummy,nodeid);
   Info("CloseGeometry","%i nodes/ %i volume UID's in %s", fNNodes, fUniqueVolumes->GetEntriesFast()-1, GetTitle());
   Info("CloseGeometry","----------------modeler ready----------------");
}

//_____________________________________________________________________________
void TGeoManager::ClearOverlaps()
{
// Clear the list of overlaps.
   if (fOverlaps) {
      fOverlaps->Delete();
      delete fOverlaps;
   }
   fOverlaps = new TObjArray();
}

//_____________________________________________________________________________
void TGeoManager::ClearShape(const TGeoShape *shape)
{
// Remove a shape from the list of shapes.
   if (fShapes->FindObject(shape)) fShapes->Remove((TGeoShape*)shape);
   delete shape;
}
//_____________________________________________________________________________
void TGeoManager::CleanGarbage()
{
// Clean temporary volumes and shapes from garbage collection.
   if (!fGVolumes && !fGShapes) return;
   Int_t i,nentries;
   if (fGVolumes) {
      nentries = fGVolumes->GetEntries();
      TGeoVolume *vol = 0;
      for (i=0; i<nentries; i++) {
         vol=(TGeoVolume*)fGVolumes->At(i);
         if (vol) vol->SetFinder(0);
      }   
      fGVolumes->Delete();
      delete fGVolumes;
      fGVolumes = 0;
   }   
   if (fGShapes) {
      fGShapes->Delete();
      delete fGShapes;
      fGShapes = 0;
   }   
}

//_____________________________________________________________________________
void TGeoManager::CdNode(Int_t nodeid)
{
// Change current path to point to the node having this id.
// Node id has to be in range : 0 to fNNodes-1 (no check for performance reasons)
   fCache->CdNode(nodeid);
}

//_____________________________________________________________________________
Int_t TGeoManager::GetCurrentNodeId() const
{
// Get the unique ID of the current node.
   return fCache->GetCurrentNodeId();
}

//_____________________________________________________________________________
void TGeoManager::CdTop()
{
// Make top level node the current node. Updates the cache accordingly.
// Determine the overlapping state of current node.
   fLevel = 0;
   fNmany = 0;
   if (fCurrentOverlapping) fLastNode = fCurrentNode;
   fCurrentNode = fTopNode;
   fCache->CdTop();
   fCurrentOverlapping = fCurrentNode->IsOverlapping();
   if (fCurrentOverlapping) fNmany++;
}
//_____________________________________________________________________________
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
   if (fCurrentOverlapping) {
      fLastNode = fCurrentNode;
      fNmany--;
   }   
   fCurrentNode = fCache->GetNode();
   if (!fCurrentNode->IsOffset()) {
      fCurrentOverlapping = fCurrentNode->IsOverlapping();
   } else {
      Int_t up = 1;
      Bool_t offset = kTRUE;
      TGeoNode *mother = 0;
      while  (offset) {
         mother = GetMother(up++);
         offset = mother->IsOffset();
      }
      fCurrentOverlapping = mother->IsOverlapping();
   }      
}
//_____________________________________________________________________________
void TGeoManager::CdDown(Int_t index)
{
// Make a daughter of current node current. Can be called only with a valid
// daughter index (no check). Updates cache accordingly.
   TGeoNode *node = fCurrentNode->GetDaughter(index);
   Bool_t is_offset = node->IsOffset();
   if (is_offset)
      node->cd();
   else
      fCurrentOverlapping = node->IsOverlapping();
   fCache->CdDown(index);
   fCurrentNode = node;
   if (fCurrentOverlapping) fNmany++;
   fLevel++;
}
//_____________________________________________________________________________
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
         Error("cd", "Path %s not valid", path);
         return kFALSE;
      }
      CdDown(fCurrentNode->GetVolume()->GetIndex(node));
      ind1 = ind2;
   }
   return kTRUE;
}
//_____________________________________________________________________________
Int_t TGeoManager::CountNodes(const TGeoVolume *vol, Int_t nlevels, Int_t option)
{
// Count the total number of nodes starting from a volume, nlevels down.
   TGeoVolume *top;
   if (!vol) {
      top = fTopVolume;
   } else {
      top = (TGeoVolume*)vol;
   }
   Int_t count = top->CountNodes(nlevels, option);
   return count;
}

//_____________________________________________________________________________
void TGeoManager::DefaultAngles()
{
// Set default angles for a given view.
   if (fPainter) fPainter->DefaultAngles();
}

//_____________________________________________________________________________
void TGeoManager::DrawCurrentPoint(Int_t color)
{
// Draw current point in the same view.
   if (fPainter) fPainter->DrawCurrentPoint(color);
}

//_____________________________________________________________________________
void TGeoManager::AnimateTracks(Double_t tmin, Double_t tmax, Int_t nframes, Option_t *option)
{
// Draw animation of tracks
   SetAnimateTracks();
   GetGeomPainter();
   if (tmin<0 || tmin>=tmax || nframes<1) return;
   Double_t *box = fPainter->GetViewBox();
   box[0] = box[1] = box[2] = 0;
   box[3] = box[4] = box[5] = 100;
   Double_t dt = (tmax-tmin)/Double_t(nframes);
   Double_t delt = 2E-9;
   Double_t t = tmin;
   Int_t i, j;
   TString opt(option);
   Bool_t save = kFALSE, geomanim=kFALSE;
   char fname[15];
   if (opt.Contains("/S")) save = kTRUE;

   if (opt.Contains("/G")) geomanim = kTRUE;
   SetTminTmax(0,0);
   DrawTracks(opt.Data());
   Double_t start[6], end[6];
   Double_t dd[6] = {0,0,0,0,0,0};
   Double_t dlat=0, dlong=0, dpsi=0;
   if (geomanim) {
      fPainter->EstimateCameraMove(tmin+5*dt, tmin+15*dt, start, end);
      for (i=0; i<3; i++) {
         start[i+3] = 20 + 1.3*start[i+3];
         end[i+3] = 20 + 0.9*end[i+3];
      }
      for (i=0; i<6; i++) {
         dd[i] = (end[i]-start[i])/10.;
      }
      memcpy(box, start, 6*sizeof(Double_t));
      fPainter->GetViewAngles(dlong,dlat,dpsi);
      dlong = (-206-dlong)/Double_t(nframes);
      dlat  = (126-dlat)/Double_t(nframes);
      dpsi  = (75-dpsi)/Double_t(nframes);
      fPainter->GrabFocus();
   }

   for (i=0; i<nframes; i++) {
      if (t-delt<0) SetTminTmax(t-delt,t);
      else gGeoManager->SetTminTmax(t-delt,t);
      if (geomanim) {
         for (j=0; j<6; j++) box[j]+=dd[j];
         fPainter->GrabFocus(1,dlong,dlat,dpsi);
      } else {
         ModifiedPad();
      }
      if (save) {
         Int_t ndigits=1;
         Int_t result=i;
         while ((result /= 10)) ndigits++;
         sprintf(fname, "anim0000.gif");
         char *fpos = fname+8-ndigits;
         sprintf(fpos, "%d.gif", i);
         gPad->Print(fname);
      }
      t += dt;
   }
   SetAnimateTracks(kFALSE);
}

//_____________________________________________________________________________
void TGeoManager::DrawTracks(Option_t *option)
{
// Draw tracks over the geometry, according to option. By default, only
// primaries are drawn. See TGeoTrack::Draw() for additional options.
   TVirtualGeoTrack *track;
   //SetVisLevel(1);
   //SetVisOption(1);
   SetAnimateTracks();
   for (Int_t i=0; i<fNtracks; i++) {
      track = GetTrack(i);
      track->Draw(option);
   }
   SetAnimateTracks(kFALSE);
   ModifiedPad();
}

//_____________________________________________________________________________
void TGeoManager::DrawPath(const char *path)
{
// Draw current path
   if (!fTopVolume) return;
   fTopVolume->SetVisBranch();
   GetGeomPainter()->DrawPath(path);
}
//_____________________________________________________________________________
void TGeoManager::RandomPoints(const TGeoVolume *vol, Int_t npoints, Option_t *option)
{
// Draw random points in the bounding box of a volume.
   GetGeomPainter()->RandomPoints((TGeoVolume*)vol, npoints, option);
}
//_____________________________________________________________________________
void TGeoManager::Test(Int_t npoints, Option_t *option)
{
// Check time of finding "Where am I" for n points.
   GetGeomPainter()->Test(npoints, option);
}
//_____________________________________________________________________________
void TGeoManager::TestOverlaps(const char* path)
{
// Geometry overlap checker based on sampling.
   GetGeomPainter()->TestOverlaps(path);
}
//_____________________________________________________________________________
void TGeoManager::GetBranchNames(Int_t *names) const
{
// Fill volume names of current branch into an array.
   fCache->GetBranchNames(names);
}
//_____________________________________________________________________________
const char *TGeoManager::GetPdgName(Int_t pdg) const
{
// Get name for given pdg code;
   static char *defaultname = "XXX";
   if (!fPdgNames) return defaultname;
   for (Int_t i=0; i<fNpdg; i++) {
      if (fPdgId[i]==pdg) return fPdgNames->At(i)->GetName();
   }
   return defaultname;
}

//_____________________________________________________________________________
void TGeoManager::SetPdgName(Int_t pdg, const char *name)
{
// Set a name for a particle having a given pdg.
   if (!fPdgNames) {
      fPdgNames = new TObjArray(256);
   }
   if (!strcmp(name, GetPdgName(pdg))) return;
   // store pdg name
   fPdgId[fNpdg] = pdg;
   TNamed *pdgname = new TNamed(name, "");
   fPdgNames->AddAt(pdgname, fNpdg++);
}

//_____________________________________________________________________________
void TGeoManager::GetBranchNumbers(Int_t *copyNumbers, Int_t *volumeNumbers) const
{
// Fill node copy numbers of current branch into an array.
   fCache->GetBranchNumbers(copyNumbers, volumeNumbers);
}
//_____________________________________________________________________________
void TGeoManager::GetBranchOnlys(Int_t *isonly) const
{
// Fill node copy numbers of current branch into an array.
   fCache->GetBranchOnlys(isonly);
}
//_____________________________________________________________________________
void TGeoManager::GetBombFactors(Double_t &bombx, Double_t &bomby, Double_t &bombz, Double_t &bombr) const
{
// Retrieve cartesian and radial bomb factors.
   if (fPainter) {
      fPainter->GetBombFactors(bombx, bomby, bombz, bombr);
      return;
   }
   bombx = bomby = bombz = bombr = 1.3;
}
//_____________________________________________________________________________
TGeoHMatrix *TGeoManager::GetHMatrix()
{
// Return stored current matrix (global matrix of the next touched node).
   if (!fCurrentMatrix) {
      fCurrentMatrix = new TGeoHMatrix();
      fCurrentMatrix->RegisterYourself();
   }   
   return fCurrentMatrix;
}
//_____________________________________________________________________________
Int_t TGeoManager::GetVisLevel() const
{
// Returns current depth to which geometry is drawn.
   return fVisLevel;
}
//_____________________________________________________________________________
Int_t TGeoManager::GetVisOption() const
{
// Returns current depth to which geometry is drawn.
   return fVisOption;
}
//_____________________________________________________________________________
Int_t TGeoManager::GetVirtualLevel()
{
// Find level of virtuality of current overlapping node (number of levels
// up having the same tracking media.

   // return if the current node is ONLY
   if (!fCurrentOverlapping) return 0;
   Int_t new_media = 0;
   TGeoMedium *medium = fCurrentNode->GetMedium();
   Int_t virtual_level = 1;
   TGeoNode *mother = 0;

   while ((mother=GetMother(virtual_level))) {
      if (!mother->IsOverlapping() && !mother->IsOffset()) {
         if (!new_media) new_media=(mother->GetMedium()==medium)?0:virtual_level;
         break;
      }
      if (!new_media) new_media=(mother->GetMedium()==medium)?0:virtual_level;
      virtual_level++;
   }
   return (new_media==0)?virtual_level:(new_media-1);
}
//_____________________________________________________________________________
TVirtualGeoTrack *TGeoManager::GetTrackOfId(Int_t id) const
{
// Get track with a given ID.
   TVirtualGeoTrack *track;
   for (Int_t i=0; i<fNtracks; i++) {
      if ((track = (TVirtualGeoTrack *)fTracks->UncheckedAt(i))) {
         if (track->GetId() == id) return track;
      }
   }
   return 0;
}

//_____________________________________________________________________________
TVirtualGeoTrack *TGeoManager::GetParentTrackOfId(Int_t id) const
{
// Get parent track with a given ID.
   TVirtualGeoTrack *track = fCurrentTrack;
   while ((track=track->GetMother())) {
      if (track->GetId()==id) return track;
   }
   return 0;
}

//_____________________________________________________________________________
Int_t TGeoManager::GetTrackIndex(Int_t id) const
{
// Get index for track id, -1 if not found.
   TVirtualGeoTrack *track;
   for (Int_t i=0; i<fNtracks; i++) {
      if ((track = (TVirtualGeoTrack *)fTracks->UncheckedAt(i))) {
         if (track->GetId() == id) return i;
      }
   }
   return -1;
}

//_____________________________________________________________________________
Bool_t TGeoManager::GotoSafeLevel()
{
// Go upwards the tree until a non-overlaping node
   while (fCurrentOverlapping && fLevel) CdUp();
   Double_t point[3];
   fCache->MasterToLocal(fPoint, point);
   if (!fCurrentNode->GetVolume()->Contains(point)) return kFALSE;
   if (fNmany) {
   // We still have overlaps on the branch
      Int_t up = 1;
      Int_t imother;
      Int_t nmany = fNmany;
      Bool_t ovlp = kFALSE;
      Bool_t nextovlp = kFALSE;
      TGeoNode *current = fCurrentNode;
      TGeoNode *mother, *mup;
      TGeoHMatrix *matrix;
      while (nmany) {
         mother = GetMother(up);
         if (!mother) return kTRUE;
         mup = mother;
         imother = up+1;
         while (mup->IsOffset()) mup = GetMother(imother++);
         nextovlp = mup->IsOverlapping();
         if (ovlp) nmany--;
         if (ovlp || nextovlp) {
         // check if the point is in the next node up
            matrix = GetMotherMatrix(up);
            matrix->MasterToLocal(fPoint,point);
            if (!mother->GetVolume()->Contains(point)) {
               up++;
               while (up--) CdUp();
               return GotoSafeLevel();
            }
         }   
         current = mother;
         ovlp = nextovlp;
         up++;
      }            
   }      
   return kTRUE;         
}

//_____________________________________________________________________________
Int_t TGeoManager::GetSafeLevel() const
{
// Go upwards the tree until a non-overlaping node
   Bool_t overlapping = fCurrentOverlapping;
   if (!overlapping) return fLevel;
   Int_t level = fLevel;
   TGeoNode *node;
   while (overlapping && level) {
      level--;
      node = GetMother(fLevel-level);
      if (!node->IsOffset()) overlapping = node->IsOverlapping();
   }
   return level;
}

//_____________________________________________________________________________
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
//_____________________________________________________________________________
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
            if (fActivity && !current->GetVolume()->IsActive()) continue;
            current->MasterToLocal(point, &local[0]);
            if (current->GetVolume()->Contains(&local[0])) {
               result[ntotal++]=check_list[i];
            }
         }
      }
   }
   return ntotal;
}
//_____________________________________________________________________________
void TGeoManager::DefaultColors()
{
// Set default volume colors according to A of material
   
   const Int_t nmax = 250;
   Int_t col[nmax];
   for (Int_t i=0;i<nmax;i++) col[i] = 18;
        
   //here we should create a new TColor with the same rgb as in the default
   //ROOT colors used below
   col[  8] = 15;
   col[  9] = 16;
   col[ 10] = 17;
   col[ 11] = 21;
   col[ 12] = 20;
   col[ 13] = 18;
   col[ 14] = 23;
   col[ 15] = 24;
   col[ 16] = 24+100;
   col[ 17] = 24+150;
   col[ 18] = 23+150;
   col[ 19] = 23+100;
   col[ 20] = 25;
   col[ 21] = 26;
   col[ 22] = 26+100;
   col[ 23] = 26+150;
   col[ 24] = 27;
   col[ 25] = 28;
   col[ 26] = 17; //29;
   col[ 27] = 30;
   col[ 28] = 30+100;
   col[ 29] = 30+150;
   col[ 30] = 14;
   col[ 31] = 31;
   col[ 32] = 31+100;
   col[ 33] = 31+150;
   col[ 38] = 33;
   col[ 39] = 2;
   col[ 41] = 38;
   col[ 42] = 40;
   col[ 45] = 37;
   col[ 55] = 41;
   col[ 63] = 42;
   col[ 64] = 44;
   col[169] = 45;
   col[170] = 50;
   col[207] = 38;

   TGeoVolume *vol;
   TIter next(fVolumes);
   while ((vol=(TGeoVolume*)next())) {
      TGeoMedium *med = vol->GetMedium();
      if (!med) continue;
      TGeoMaterial *mat = med->GetMaterial();
      Int_t matA = (Int_t)mat->GetA();
      vol->SetLineColor(col[matA]);
   }
}

//_____________________________________________________________________________
Double_t TGeoManager::Safety(Bool_t inside)
{
// Compute safe distance from the current point. This represent the distance
// from POINT to the closest boundary.

   if (fIsOnBoundary) {
      fSafety = 0;
      return fSafety;
   }
   Double_t point[3];
   if (!inside) fSafety = TGeoShape::Big();
   if (fIsOutside) {
      fSafety = fTopVolume->GetShape()->Safety(fPoint,kFALSE);
      if (fSafety < TGeoShape::Tolerance()) {
         fSafety = 0;
         fIsOnBoundary = kTRUE;
      }   
      return fSafety;
   }
   //---> convert point to local reference frame of current node
   fCache->MasterToLocal(fPoint, point);

   //---> compute safety to current node
   TGeoVolume *vol = fCurrentNode->GetVolume();
   if (!inside) {
      fSafety = vol->GetShape()->Safety(point, kTRUE);
      //---> if we were just entering, return this safety
      if (fSafety < TGeoShape::Tolerance()) {
         fSafety = 0;
         fIsOnBoundary = kTRUE;
         return fSafety;
      }
   }

   //---> now check the safety to the last node

   //---> if we were just exiting, return this safety
   TObjArray *nodes = vol->GetNodes();
   Int_t nd = fCurrentNode->GetNdaughters();
   if (!nd && !fCurrentOverlapping) return fSafety;
   TGeoNode *node;
   Double_t safe;
   Int_t id;

   // if current volume is divided, we are in the non-divided region. We
   // check only the first and the last cell
   TGeoPatternFinder *finder = vol->GetFinder();
   if (finder) {
      Int_t ifirst = finder->GetDivIndex();
      node = (TGeoNode*)nodes->UncheckedAt(ifirst);
      node->cd();
      safe = node->Safety(point, kFALSE);
      if (safe < TGeoShape::Tolerance()) {
         fSafety=0;
         fIsOnBoundary = kTRUE;
         return fSafety;
      }
      if (safe<fSafety) fSafety=safe;
      Int_t ilast = ifirst+finder->GetNdiv()-1;
      if (ilast==ifirst) return fSafety;
      node = (TGeoNode*)nodes->UncheckedAt(ilast);
      node->cd();
      safe = node->Safety(point, kFALSE);
      if (safe < TGeoShape::Tolerance()) {
         fSafety=0;
         fIsOnBoundary = kTRUE;
         return fSafety;
      }
      if (safe<fSafety) fSafety=safe;
      if (fCurrentOverlapping  && !inside) SafetyOverlaps();
      return fSafety;
   }

   //---> If no voxels just loop daughters
   TGeoVoxelFinder *voxels = vol->GetVoxels();
   if (!voxels) {
      for (id=0; id<nd; id++) {
         node = (TGeoNode*)nodes->UncheckedAt(id);
         safe = node->Safety(point, kFALSE);
         if (safe < TGeoShape::Tolerance()) {
            fSafety=0;
            fIsOnBoundary = kTRUE;
            return fSafety;
         }
         if (safe<fSafety) fSafety=safe;
      }
      if (fNmany && !inside) SafetyOverlaps();
      return fSafety;
   }

   //---> check fast unsafe voxels
   Double_t *boxes = voxels->GetBoxes();
   for (id=0; id<nd; id++) {
      Int_t ist = 6*id;
      Double_t dxyz = 0.;
      Double_t dxyz0 = TMath::Abs(point[0]-boxes[ist+3])-boxes[ist];
      if (dxyz0 > fSafety) continue;
      Double_t dxyz1 = TMath::Abs(point[1]-boxes[ist+4])-boxes[ist+1];
      if (dxyz1 > fSafety) continue;
      Double_t dxyz2 = TMath::Abs(point[2]-boxes[ist+5])-boxes[ist+2];
      if (dxyz2 > fSafety) continue;
      if (dxyz0>0) dxyz+=dxyz0*dxyz0;
      if (dxyz1>0) dxyz+=dxyz1*dxyz1;
      if (dxyz2>0) dxyz+=dxyz2*dxyz2;
      if (dxyz >= fSafety*fSafety) continue;
      node = (TGeoNode*)nodes->UncheckedAt(id);
      safe = node->Safety(point, kFALSE);
      if (safe<TGeoShape::Tolerance()) {
         fSafety=0;
         fIsOnBoundary = kTRUE;
         return fSafety;
      }
      if (safe<fSafety) fSafety = safe;
   }
   if (fNmany  && !inside) SafetyOverlaps();
   return fSafety;
}

//_____________________________________________________________________________
void TGeoManager::SafetyOverlaps()
{
// Compute safe distance from the current point within an overlapping node
   Double_t point[3], local[3];
   Double_t safe;
   Bool_t contains;
   TGeoNode *nodeovlp;
   TGeoVolume *vol;
   Int_t novlp, io;
   Int_t *ovlp;
   Int_t safelevel = GetSafeLevel();
   PushPath(safelevel+1);
   while (fCurrentOverlapping) {
      ovlp = fCurrentNode->GetOverlaps(novlp);
      CdUp();
      vol = fCurrentNode->GetVolume();
      gGeoManager->MasterToLocal(fPoint, point);
      contains = fCurrentNode->GetVolume()->Contains(point);
      safe = fCurrentNode->GetVolume()->GetShape()->Safety(point, contains);
      if (safe<fSafety && safe>=0) fSafety=safe;
      if (!novlp || !contains) continue;
      // we are now in the container, check safety to all candidates
      for (io=0; io<novlp; io++) {
         nodeovlp = vol->GetNode(ovlp[io]);
         nodeovlp->GetMatrix()->MasterToLocal(point,local);
         contains = nodeovlp->GetVolume()->Contains(local);
         if (contains) {
            CdDown(ovlp[io]);
            safe = Safety(kTRUE);
            CdUp();
         } else {
            safe = nodeovlp->GetVolume()->GetShape()->Safety(local, kFALSE);
         }
         if (safe<fSafety && safe>=0) fSafety=safe;
      }
   }
   if (fNmany) {
   // We have overlaps up in the branch, check distance to exit
      Int_t up = 1;
      Int_t imother;
      Int_t nmany = fNmany;
      Bool_t ovlp = kFALSE;
      Bool_t nextovlp = kFALSE;
      TGeoNode *current = fCurrentNode;
      TGeoNode *mother, *mup;
      TGeoHMatrix *matrix;
      while (nmany) {
         mother = GetMother(up);
         mup = mother;
         imother = up+1;
         while (mup->IsOffset()) mup = GetMother(imother++);
         nextovlp = mup->IsOverlapping();
         if (ovlp) nmany--;
         if (ovlp || nextovlp) {
            matrix = GetMotherMatrix(up);
            matrix->MasterToLocal(fPoint,local);
            safe = mother->GetVolume()->GetShape()->Safety(local,kTRUE);
            if (safe<fSafety) fSafety = safe;
            current = mother;
            ovlp = nextovlp;
         }
         up++;
      }      
   }
   PopPath();
   if (fSafety < TGeoShape::Tolerance()) {
      fSafety = 0.;
      fIsOnBoundary = kTRUE;
   }   
}

//_____________________________________________________________________________
void TGeoManager::SetVolumeAttribute(const char *name, const char *att, Int_t val)
{
// Set volume attributes in G3 style.
   TGeoVolume *volume;
   Bool_t all = kFALSE;
   if (strstr(name,"*")) all=kTRUE;
   Int_t ivo=0;
   TIter next(fVolumes);
   TString chatt = att;
   chatt.ToLower();
   while ((volume=(TGeoVolume*)next())) {
      if (strcmp(volume->GetName(), name) && !all) continue;
      ivo++;
      if (chatt.Contains("colo")) volume->SetLineColor(val);
      if (chatt.Contains("lsty")) volume->SetLineStyle(val);
      if (chatt.Contains("lwid")) volume->SetLineWidth(val);
      if (chatt.Contains("fill")) volume->SetFillColor(val);
      if (chatt.Contains("seen")) volume->SetVisibility(val);
   }
   TIter next1(fGVolumes);
   while ((volume=(TGeoVolume*)next1())) {
      if (strcmp(volume->GetName(), name) && !all) continue;
      ivo++;
      if (chatt.Contains("colo")) volume->SetLineColor(val);
      if (chatt.Contains("lsty")) volume->SetLineStyle(val);
      if (chatt.Contains("lwid")) volume->SetLineWidth(val);
      if (chatt.Contains("fill")) volume->SetFillColor(val);
      if (chatt.Contains("seen")) volume->SetVisibility(val);
   }
   if (!ivo) {
      Warning("SetVolumeAttribute","volume: %s does not exist",name);
   }
}
//_____________________________________________________________________________
void TGeoManager::SetBombFactors(Double_t bombx, Double_t bomby, Double_t bombz, Double_t bombr)
{
// Set factors that will "bomb" all translations in cartesian and cylindrical coordinates.
   if (fPainter) fPainter->SetBombFactors(bombx, bomby, bombz, bombr);
}

//_____________________________________________________________________________
void TGeoManager::SetClippingShape(TGeoShape *shape)
{
// Set a user-defined shape as clipping for ray tracing.
   TVirtualGeoPainter *painter = GetGeomPainter();
   if (shape) {
      if (fClippingShape && (fClippingShape!=shape)) ClearShape(fClippingShape);
      fClippingShape = shape;
   }
   painter->SetClippingShape(shape);
}

//_____________________________________________________________________________
void TGeoManager::SetMaxVisNodes(Int_t maxnodes) {
// set the maximum number of visible nodes.   
   fMaxVisNodes = maxnodes;
   if (maxnodes>0) Info("SetMaxVisNodes","Automatic visible depth for %d visible nodes", maxnodes);
   if (!fPainter) return;
   fPainter->CountVisibleNodes();
   Int_t level = fPainter->GetVisLevel();
   if (level != fVisLevel) fVisLevel = level;
}

//_____________________________________________________________________________
void TGeoManager::SetTopVisible(Bool_t vis) {
// make top volume visible on screen
   GetGeomPainter();
   fPainter->SetTopVisible(vis);
}
//_____________________________________________________________________________
void TGeoManager::SetVisOption(Int_t option) {
// set drawing mode :
// option=0 (default) all nodes drawn down to vislevel
// option=1           leaves and nodes at vislevel drawn
// option=2           path is drawn
   if ((option>=0) && (option<3)) fVisOption=option;
   if (fPainter) fPainter->SetVisOption(option);
}

//_____________________________________________________________________________
void TGeoManager::ViewLeaves(Bool_t flag)
{
// Set visualization option (leaves only OR all volumes)
   if (flag) SetVisOption(1);
   else      SetVisOption(0);
}

//_____________________________________________________________________________
void TGeoManager::SetVisDensity(Double_t density)
{
// Set density threshold. Volumes with densities lower than this become
// transparent.
   fVisDensity = density;
   if (fPainter) fPainter->ModifiedPad();
}      

//_____________________________________________________________________________
void TGeoManager::SetVisLevel(Int_t level) {
// set default level down to which visualization is performed
   if (level>0) {
      fVisLevel = level;
      fMaxVisNodes = 0;
      Info("SetVisLevel","Automatic visible depth disabled");
      if (fPainter) fPainter->CountVisibleNodes();
   } else {
      SetMaxVisNodes();
   }      
}

//_____________________________________________________________________________
void TGeoManager::SortOverlaps()
{
// Sort overlaps by decreasing overlap distance. Extrusions comes first.
   fOverlaps->Sort();
}

//_____________________________________________________________________________
void TGeoManager::OptimizeVoxels(const char *filename)
{
// Optimize voxelization type for all volumes. Save best choice in a macro.
   if (!fTopNode) {
      Error("OptimizeVoxels","Geometry must be closed first");
      return;
   }
   ofstream out;
   char *fname = new char[20];
   char quote = '"';
   if (!strlen(filename))
      sprintf(fname, "tgeovox.C");
   else
      sprintf(fname, "%s", filename);
   out.open(fname, ios::out);
   if (!out.good()) {
      Error("OptimizeVoxels", "cannot open file");
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
   out << "//=== Voxel optimization for " << GetTitle() << " geometry"<<endl;
   out << "//===== <run this macro JUST BEFORE closing the geometry>"<<endl;
   out << "   TGeoVolume *vol = 0;"<<endl;
   out << "   // parse all voxelized volumes"<<endl;
   TGeoVolume *vol = 0;
   Bool_t cyltype;
   TIter next(fVolumes);
   while ((vol=(TGeoVolume*)next())) {
      if (!vol->GetVoxels()) continue;
      out<<"   vol = gGeoManager->GetVolume("<<quote<<vol->GetName()<<quote<<");"<<endl;
      cyltype = vol->OptimizeVoxels();
      if (cyltype) {
         out<<"   vol->SetCylVoxels();"<<endl;
      } else {
         out<<"   vol->SetCylVoxels(kFALSE);"<<endl;
      }
   }
   out << "}" << endl;
   out.close();
   delete [] fname;
}
//_____________________________________________________________________________
Int_t TGeoManager::Parse(const char *expr, TString &expr1, TString &expr2, TString &expr3)
{
// Parse a string boolean expression and do a syntax check. Find top
// level boolean operator and returns its type. Fill the two
// substrings to which this operator applies. The returned integer is :
// -1 : parse error
//  0 : no boolean operator
//  1 : union - represented as '+' in expression
//  2 : difference (subtraction) - represented as '-' in expression
//  3 : intersection - represented as '*' in expression.
// Paranthesys should be used to avoid ambiguites. For instance :
//    A+B-C will be interpreted as (A+B)-C which is not the same as A+(B-C)
   // eliminate not needed paranthesys
   TString startstr(expr);
   Int_t len = startstr.Length();
   Int_t i;
   TString e0 = "";
   expr3 = "";
   // eliminate blanks
   for (i=0; i< len; i++) {
      if (startstr(i)==' ') continue;
      e0 += startstr(i, 1);
   }
   Int_t level = 0;
   Int_t levmin = 999;
   Int_t boolop = 0;
   Int_t indop = 0;
   Int_t iloop = 1;
   Int_t lastop = 0;
   Int_t lastdp = 0;
   Int_t lastpp = 0;
   Bool_t foundmat = kFALSE;
   // check/eliminate paranthesys
   while (iloop==1) {
      iloop = 0;
      lastop = 0;
      lastdp = 0;
      lastpp = 0;
      len = e0.Length();
      for (i=0; i<len; i++) {
         if (e0(i)=='(') {
            if (!level) iloop++;
            level++;
            continue;
         }
         if  (e0(i)==')') {
            level--;
            if (level==0) lastpp=i;
            continue;
         }
         if ((e0(i)=='+') || (e0(i)=='-') || (e0(i)=='*')) {
            lastop = i;
            if (level<levmin) {
               levmin = level;
               indop = i;
            }
            continue;
         }
         if  ((e0(i)==':') && (level==0)) {
            lastdp = i;
            continue;
         }
      }
      if (level!=0) {
         if (gGeoManager) gGeoManager->Error("Parse","paranthesys does not match");
         return -1;
      }
      if (iloop==1 && (e0(0)=='(') && (e0(len-1)==')')) {
         // eliminate extra paranthesys
         e0=e0(1, len-2);
         continue;
      }
      if (foundmat) break;
      if (((lastop==0) && (lastdp>0)) || ((lastpp>0) && (lastdp>lastpp) && (indop<lastpp))) {
         expr3 = e0(lastdp+1, len-lastdp);
         e0=e0(0, lastdp);
         foundmat = kTRUE;
         iloop = 1;
         continue;
      } else break;
   }
   // loop expression and search paranthesys/operators
   levmin = 999;
   for (i=0; i<len; i++) {
      if (e0(i)=='(') {
         level++;
         continue;
      }
      if  (e0(i)==')') {
         level--;
         continue;
      }
      if (level<levmin) {
         if (e0(i)=='+') {
            boolop = 1; // union
            levmin = level;
            indop = i;
         }
         if (e0(i)=='-') {
            boolop = 2; // difference
            levmin = level;
            indop = i;
         }
         if (e0(i)=='*') {
            boolop = 3; // intersection
            levmin = level;
            indop = i;
         }
      }
   }
   if (indop==0) {
      expr1=e0;
      return indop;
   }
   expr1 = e0(0, indop);
   expr2 = e0(indop+1, len-indop);
   return boolop;
}


//_____________________________________________________________________________
void TGeoManager::SaveAttributes(const char *filename)
{
// Save current attributes in a macro
   if (!fTopNode) {
      Error("SaveAttributes","geometry must be closed first\n");
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
   out << "   TGeoVolume *top = gGeoManager->GetVolume("<<quote<<fTopVolume->GetName()<<quote<<");"<<endl;
   out << "   TGeoVolume *vol = 0;"<<endl;
   out << "   TGeoNode *node = 0;"<<endl;
   out << "   // clear all volume attributes and get painter"<<endl;
   out << "   gGeoManager->ClearAttributes();"<<endl;
   out << "   gGeoManager->GetGeomPainter();"<<endl;
   out << "   // set visualization modes and bomb factors"<<endl;
   out << "   gGeoManager->SetVisOption("<<GetVisOption()<<");"<<endl;
   out << "   gGeoManager->SetVisLevel("<<GetVisLevel()<<");"<<endl;
   out << "   gGeoManager->SetExplodedView("<<GetBombMode()<<");"<<endl;
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
   out << "   gPad->x3d();"<<endl;
   out << "}" << endl;
   out.close();
   delete [] fname;
}
//_____________________________________________________________________________
TGeoNode *TGeoManager::SearchNode(Bool_t downwards, const TGeoNode *skipnode)
{
// Returns the deepest node containing fPoint, which must be set a priori.
   Double_t point[3];
   TGeoVolume *vol = 0;
   Bool_t inside_current = (fCurrentNode==skipnode)?kTRUE:kFALSE;
   if (!downwards) {
   // we are looking upwards until inside current node or exit
      if (fActivity && !vol->IsActive()) {
         // We are inside an inactive volume-> go upwards
         CdUp();
         fIsSameLocation = kFALSE;
         return SearchNode(kFALSE, skipnode);
      }
      // Check if the current point is still inside the current volume
      vol=fCurrentNode->GetVolume();
      if (vol->IsAssembly()) inside_current=kTRUE;
      // If the current node is not to be skipped
      if (!inside_current) {
         fCache->MasterToLocal(fPoint, point);
         inside_current = vol->Contains(point);
      }   
      // Point might be inside an overlapping node
      if (fNmany) {
         inside_current = GotoSafeLevel();
      }   
      if (!inside_current) {
         // If not, go upwards
         fIsSameLocation = kFALSE;
         TGeoNode *skip = fCurrentNode;  // skip current node at next search
         // check if we can go up
         if (!fLevel) {
            fIsOutside = kTRUE;
            return 0;
         }
         CdUp();
         return SearchNode(kFALSE, skip);
      }
   }
   vol = fCurrentNode->GetVolume();
   fCache->MasterToLocal(fPoint, point);
   if (!inside_current && downwards) {
   // we are looking downwards
      inside_current = vol->Contains(point);
      if (!inside_current) {
         fIsSameLocation = kFALSE;
         return 0;
      }
   }
   // point inside current (safe) node -> search downwards
   TGeoNode *node;
   Int_t ncheck = 0;
   // if inside an non-overlapping node, reset overlap searches
   if (!fCurrentOverlapping) {
      fSearchOverlaps = kFALSE;
   }

   Int_t crtindex = vol->GetCurrentNodeIndex();
   while (crtindex>=0 && downwards) {
      CdDown(crtindex);
      vol = fCurrentNode->GetVolume();
      crtindex = vol->GetCurrentNodeIndex();
      if (crtindex<0) fCache->MasterToLocal(fPoint, point);
   }   
      
   Int_t nd = vol->GetNdaughters();
   // in case there are no daughters
   if (!nd) return fCurrentNode;
   if (fActivity && !vol->IsActiveDaughters()) return fCurrentNode;

   TGeoPatternFinder *finder = vol->GetFinder();
   // point is inside the current node
   // first check if inside a division
   if (finder) {
      node=finder->FindNode(&point[0]);
      if (node) {
         // go inside the division cell and search downwards
         fIsSameLocation = kFALSE;
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
   Int_t id;
   if (voxels) {
      // get the list of nodes passing thorough the current voxel
      check_list = voxels->GetCheckList(&point[0], ncheck);
      // if none in voxel, see if this is the last one
      if (!check_list) {
         if (!fCurrentNode->GetVolume()->IsAssembly()) return fCurrentNode;
         node = fCurrentNode;
         if (!fLevel) {
            fIsOutside = kTRUE;
            return 0;
         }
         CdUp();
         return SearchNode(kFALSE,node);
      }   
      // loop all nodes in voxel
      for (id=0; id<ncheck; id++) {
         node = vol->GetNode(check_list[id]);
         if (node==skipnode) continue;
         if (fActivity && !node->GetVolume()->IsActive()) continue;
         if ((id<(ncheck-1)) && node->IsOverlapping()) {
         // make the cluster of overlaps
            if (ncheck+fOverlapMark > fOverlapSize) {
               fOverlapSize = 2*(ncheck+fOverlapMark);
               delete [] fOverlapClusters;
               fOverlapClusters = new Int_t[fOverlapSize];
            }
            Int_t *cluster = fOverlapClusters + fOverlapMark;
            Int_t nc = GetTouchedCluster(id, &point[0], check_list, ncheck, cluster);
            if (nc>1) {
               fOverlapMark += nc;
               node = FindInCluster(cluster, nc);
               fOverlapMark -= nc;
               return node;
            }
         }
         CdDown(check_list[id]);
         node = SearchNode(kTRUE);
         if (node) {
            fIsSameLocation = kFALSE;
            return node;
         }
         CdUp();
      }
      if (!fCurrentNode->GetVolume()->IsAssembly()) return fCurrentNode;
      node = fCurrentNode;
      if (!fLevel) {
         fIsOutside = kTRUE;
         return 0;
      }
      CdUp();
      return SearchNode(kFALSE,node);
   }
   // if there are no voxels just loop all daughters
   for (id=0; id<nd; id++) {
      node=fCurrentNode->GetDaughter(id);
      if (node==skipnode) continue;  
      if (fActivity && !node->GetVolume()->IsActive()) continue;
      CdDown(id);
      node = SearchNode(kTRUE);
      if (node) {
         fIsSameLocation = kFALSE;
         return node;
      }
      CdUp();
   }      
   // point is not inside one of the daughters, so it is in the current vol
   if (fCurrentNode->GetVolume()->IsAssembly()) {
      node = fCurrentNode;
      if (!fLevel) {
         fIsOutside = kTRUE;
         return 0;
      }
      CdUp();
      return SearchNode(kFALSE,node);
   }      
   return fCurrentNode;
}

//_____________________________________________________________________________
TGeoNode *TGeoManager::FindNextBoundaryAndStep(Double_t stepmax, Bool_t compsafe)
{
// Compute distance to next boundary within STEPMAX. If no boundary is found,
// propagate current point along current direction with fStep=STEPMAX. Otherwise
// propagate with fStep=SNEXT (distance to boundary) and locate/return the next 
// node.
   static Int_t icount = 0;
   icount++;
   Int_t iact = 3;
   Int_t nextindex;
   Bool_t is_assembly;
   TGeoNode *skip;
   fIsStepEntering = kFALSE;
   fStep = stepmax;
   Double_t snext = TGeoShape::Big();
   if (compsafe) {
      fIsOnBoundary = kFALSE;
      Safety();
   }   
   Double_t extra = (fIsOnBoundary)?TGeoShape::Tolerance():0.0;
   fIsOnBoundary = kFALSE;
   fPoint[0] += extra*fDirection[0];
   fPoint[1] += extra*fDirection[1];
   fPoint[2] += extra*fDirection[2];
   *fCurrentMatrix = GetCurrentMatrix();
   
   if (fIsOutside) {
      snext = fTopVolume->GetShape()->DistFromOutside(fPoint, fDirection, iact, fStep);
      if (snext < fStep) {
         if (snext<=0) {
            snext = 0.0;
            fStep = snext;
            fPoint[0] -= extra*fDirection[0];
            fPoint[1] -= extra*fDirection[1];
            fPoint[2] -= extra*fDirection[2];
         } else {
            fStep = snext+extra;
         }   
         fIsStepEntering = kTRUE;
         fNextNode = fTopNode;
         nextindex = fNextNode->GetVolume()->GetNextNodeIndex();
         while (nextindex>=0) {
            CdDown(nextindex);
            fNextNode = fCurrentNode;
            nextindex = fNextNode->GetVolume()->GetNextNodeIndex();
            if (nextindex<0) *fCurrentMatrix = GetCurrentMatrix();
         }   
         // Update global point
         fPoint[0] += snext*fDirection[0];
         fPoint[1] += snext*fDirection[1];
         fPoint[2] += snext*fDirection[2];
         fIsOnBoundary = kTRUE;
         fIsOutside = kFALSE;
         return CrossBoundaryAndLocate(kTRUE, fCurrentNode);
      }
      if (snext<TGeoShape::Big()) {
         // New point still outside, but the top node is reachable
         fNextNode = fTopNode;
         fPoint[0] += (fStep-extra)*fDirection[0];
         fPoint[1] += (fStep-extra)*fDirection[1];
         fPoint[2] += (fStep-extra)*fDirection[2];
         return fNextNode;
      }      
      // top node not reachable from current point/direction
      fNextNode = 0;
      fIsOnBoundary = kFALSE;
      return 0;
   }
   Double_t point[3],dir[3];
   Int_t icrossed = -2;
   fCache->MasterToLocal(fPoint, &point[0]);
   fCache->MasterToLocalVect(fDirection, &dir[0]);
   TGeoVolume *vol = fCurrentNode->GetVolume();
   // find distance to exiting current node
   snext = vol->GetShape()->DistFromInside(point, dir, iact, fStep);
   fNextNode = fCurrentNode;
   if (snext <= TGeoShape::Tolerance()) {
      snext = TGeoShape::Tolerance();
      fStep = snext;
      fIsOnBoundary = kTRUE;
      fIsStepEntering = kTRUE;
      skip = fCurrentNode;
      is_assembly = fCurrentNode->GetVolume()->IsAssembly();
      if (!fLevel && !is_assembly) {
         fIsOutside = kTRUE;
         return 0;
      }   
      if (fLevel) CdUp();
      else        skip = 0;
      return CrossBoundaryAndLocate(kFALSE, skip);
   }   

   if (snext < fStep) {
      icrossed = -1;
      fStep = snext;
      fIsStepEntering = kTRUE;
   }
   // Find next daughter boundary for the current volume
   Int_t idaughter = -1;
   TGeoNode *crossed = FindNextDaughterBoundary(point,dir, idaughter, kTRUE);
   if (crossed) {
      icrossed = idaughter;
      fIsStepEntering = kTRUE;
   }   
   TGeoNode *current = 0;
   TGeoNode *dnode = 0;
   TGeoVolume *mother = 0;
   // if we are in an overlapping node, check also the mother(s)
   if (fNmany) {
      Double_t mothpt[3];
      Double_t vecpt[3];
      Double_t dpt[3], dvec[3];
      Int_t novlps;
      Int_t safelevel = GetSafeLevel();
      PushPath(safelevel+1);
      while (fCurrentOverlapping) {
         Int_t *ovlps = fCurrentNode->GetOverlaps(novlps);
         CdUp();
         mother = fCurrentNode->GetVolume();
         fCache->MasterToLocal(fPoint, &mothpt[0]);
         fCache->MasterToLocalVect(fDirection, &vecpt[0]);
         // check distance to out
         snext = TGeoShape::Big();
         if (!mother->IsAssembly()) snext = mother->GetShape()->DistFromInside(mothpt, vecpt, iact, fStep);
         if (snext<fStep) {
            // exiting mother first (extrusion)
            icrossed = -1;
            PopDummy();
            PushPath(safelevel+1);
            fIsStepEntering = kTRUE;
            fStep = snext;
            *fCurrentMatrix = GetCurrentMatrix();
            fNextNode = fCurrentNode;
         }
         // check overlapping nodes
         for (Int_t i=0; i<novlps; i++) {
            current = mother->GetNode(ovlps[i]);
            if (!current->IsOverlapping()) {
               current->cd();
               current->MasterToLocal(&mothpt[0], &dpt[0]);
               current->MasterToLocalVect(&vecpt[0], &dvec[0]);
               snext = current->GetVolume()->GetShape()->DistFromOutside(dpt, dvec, iact, fStep);
               if (snext<fStep) {
                  PopDummy();
                  PushPath(safelevel+1);
                 *fCurrentMatrix = GetCurrentMatrix();
                  fCurrentMatrix->Multiply(current->GetMatrix());
                  fIsStepEntering = kTRUE;
                  icrossed = ovlps[i];
                  fStep = snext;
                  fNextNode = current;
               }
            } else {
               // another many - check if point is in or out
               current->cd();
               current->MasterToLocal(&mothpt[0], &dpt[0]);
               current->MasterToLocalVect(&vecpt[0], &dvec[0]);
               if (current->GetVolume()->Contains(dpt)) {
                  if (current->GetVolume()->GetNdaughters()) {
                     CdDown(ovlps[i]);
                     dnode = FindNextDaughterBoundary(dpt,dvec,idaughter,kFALSE);
                     if (dnode) {
                        *fCurrentMatrix = GetCurrentMatrix();
                        fCurrentMatrix->Multiply(dnode->GetMatrix());
                        icrossed = idaughter;
                        PopDummy();
                        PushPath(safelevel+1);
                        fIsStepEntering = kTRUE;
                        fNextNode = dnode;
                     }   
                     CdUp();
                  }   
               } else {
                  snext = current->GetVolume()->GetShape()->DistFromOutside(dpt, dvec, iact, fStep);
                  if (snext<fStep) {
                     *fCurrentMatrix = GetCurrentMatrix();
                     fCurrentMatrix->Multiply(current->GetMatrix());
                     fIsStepEntering = kTRUE;
                     fStep = snext;
                     fNextNode = current;
                     icrossed = ovlps[i];
                     PopDummy();
                     PushPath(safelevel+1);
                  }               
               }  
            }
         }
      }
      // Now we are in a non-overlapping node
      if (fNmany) {
      // We have overlaps up in the branch, check distance to exit
         Int_t up = 1;
         Int_t imother;
         Int_t nmany = fNmany;
         Bool_t ovlp = kFALSE;
         Bool_t nextovlp = kFALSE;
         Bool_t offset = kFALSE;
         TGeoNode *current = fCurrentNode;
         TGeoNode *mother, *mup;
         TGeoHMatrix *matrix;
         while (nmany) {
            mother = GetMother(up);
            mup = mother;
            imother = up+1;
            offset = kFALSE;
            while (mup->IsOffset()) {
               mup = GetMother(imother++);
               offset = kTRUE;
            }   
            nextovlp = mup->IsOverlapping();
            if (offset) {
               mother = mup;
               if (nextovlp) nmany -= imother-up;
               up = imother-1;
            } else {    
               if (ovlp) nmany--;
            }
            if (ovlp || nextovlp) {
               matrix = GetMotherMatrix(up);
               matrix->MasterToLocal(fPoint,dpt);
               matrix->MasterToLocalVect(fDirection,dvec);
               snext = TGeoShape::Big();
               if (!mother->GetVolume()->IsAssembly()) snext = mother->GetVolume()->GetShape()->DistFromInside(dpt,dvec,iact,fStep);
               if (snext<fStep) {
                  fNextNode = mother;
                  *fCurrentMatrix = matrix;
                  fStep = snext;
                  while (up--) CdUp();
                  PopDummy();
                  PushPath();
                  icrossed = -1;
                  up = 1;
                  current = fCurrentNode;
                  ovlp = current->IsOverlapping();
                  continue;
               }   
            }   
            current = mother;
            ovlp = nextovlp;
            up++;            
         }
      }      
      PopPath();
   }
   fPoint[0] += fStep*fDirection[0];
   fPoint[1] += fStep*fDirection[1];
   fPoint[2] += fStep*fDirection[2];
   fStep += extra;
   if (icrossed == -2) {
      // Nothing crossed within stepmax -> propagate and return same location   
      return fCurrentNode;
   }
   fIsOnBoundary = kTRUE;
   if (icrossed == -1) {
      TGeoNode *skip = fCurrentNode;
      is_assembly = fCurrentNode->GetVolume()->IsAssembly();
      if (!fLevel && !is_assembly) {
         fIsOutside = kTRUE;
         return 0;
      }   
      if (fLevel) CdUp();
      else        skip = 0;
//      is_assembly = fCurrentNode->GetVolume()->IsAssembly();
//      while (fLevel && is_assembly) {
//         CdUp();
//         is_assembly = fCurrentNode->GetVolume()->IsAssembly();
//         skip = fCurrentNode;
//      }   
      return CrossBoundaryAndLocate(kFALSE, skip);
   }   
   current = fCurrentNode;   
   CdDown(icrossed);
   nextindex = fCurrentNode->GetVolume()->GetNextNodeIndex();
   while (nextindex>=0) {
      current = fCurrentNode;
      CdDown(nextindex);
      nextindex = fCurrentNode->GetVolume()->GetNextNodeIndex();
   }   

   return CrossBoundaryAndLocate(kTRUE, current);
}   

//_____________________________________________________________________________
TGeoNode *TGeoManager::CrossBoundaryAndLocate(Bool_t downwards, TGeoNode *skipnode)
{
// Cross next boundary and locate within current node
// The current point must be on the boundary of fCurrentNode.

// Extrapolate current point with shape tolerance.
   Double_t extra = TGeoShape::Tolerance();
   fPoint[0] += extra*fDirection[0];
   fPoint[1] += extra*fDirection[1];
   fPoint[2] += extra*fDirection[2];
   TGeoNode *current = SearchNode(downwards, skipnode);
   fPoint[0] -= extra*fDirection[0];
   fPoint[1] -= extra*fDirection[1];
   fPoint[2] -= extra*fDirection[2];
   if (!current) return 0;
   if (downwards) {
      Int_t nextindex = current->GetVolume()->GetNextNodeIndex();
      while (nextindex>=0) {
         CdDown(nextindex);
         current = fCurrentNode;
         nextindex = fCurrentNode->GetVolume()->GetNextNodeIndex();
      }
      return current;   
   }   
     
   if ((skipnode && current == skipnode) || current->GetVolume()->IsAssembly()) {
      if (!fLevel) {
         fIsOutside = kTRUE;
         return fTopNode;
      }
      CdUp();
      while (fLevel && fCurrentNode->GetVolume()->IsAssembly()) CdUp();
      if (!fLevel && fCurrentNode->GetVolume()->IsAssembly()) {
         fIsOutside = kTRUE;
         return fTopNode;
      }
      return fTopNode;   
   }
   return current;
}   

//_____________________________________________________________________________
TGeoNode *TGeoManager::FindNextBoundary(Double_t stepmax, const char *path)
{
// Find distance to next boundary and store it in fStep. Returns node to which this
// boundary belongs. If PATH is specified, compute only distance to the node to which
// PATH points. If STEPMAX is specified, compute distance only in case fSafety is smaller
// than this value. STEPMAX represent the step to be made imposed by other reasons than
// geometry (usually physics processes). Therefore in this case this method provides the
// answer to the question : "Is STEPMAX a safe step ?" returning a NULL node and filling
// fStep with a big number.

// Note : safety distance for the current point is computed ONLY in case STEPMAX is
//        specified, otherwise users have to call explicitly TGeoManager::Safety() if
//        they want this computed for the current point.

   // convert current point and direction to local reference
   Int_t iact = 3;
   fStep = TGeoShape::Big();
   Bool_t computeGlobal = kFALSE;
   fIsOnBoundary = kFALSE;
   if (stepmax<1E20) {
      if (stepmax <= 0) {
         stepmax = - stepmax;
         computeGlobal = kTRUE;
      }
      if (IsSamePoint(fPoint[0], fPoint[1], fPoint[2])) fSafety = fLastSafety;
      else fSafety = Safety();
      fSafety = TMath::Abs(fSafety);
      memcpy(fLastPoint, fPoint, kN3);
      fLastSafety = fSafety;
      if (fSafety<TGeoShape::Tolerance()) fIsOnBoundary = kTRUE;
      else fIsOnBoundary = kFALSE;
      fStep = stepmax;
      if (stepmax<fSafety) {
         fStep = stepmax;
         return fCurrentNode;
      }
   }
   if (computeGlobal) *fCurrentMatrix = GetCurrentMatrix();
   Double_t snext  = TGeoShape::Big();
   Double_t safe;
   Double_t point[3];
   Double_t dir[3];
   if (strlen(path)) {
      PushPath();
      if (!cd(path)) {
         PopPath();
         return 0;
      }
      if (computeGlobal) *fCurrentMatrix = GetCurrentMatrix();
      fNextNode = fCurrentNode;
      TGeoVolume *tvol=fCurrentNode->GetVolume();
      fCache->MasterToLocal(fPoint, &point[0]);
      fCache->MasterToLocalVect(fDirection, &dir[0]);
      if (tvol->Contains(&point[0])) {
         fStep=tvol->GetShape()->DistFromInside(&point[0], &dir[0], iact, fStep, &safe);
      } else {
         fStep=tvol->GetShape()->DistFromOutside(&point[0], &dir[0], iact, fStep, &safe);
      }
      PopPath();
      return fNextNode;
   }
   // compute distance to exit point from current node and the distance to its
   // closest boundary
   // if point is outside, just check the top node
   if (fIsOutside) {
      snext = fTopVolume->GetShape()->DistFromOutside(fPoint, fDirection, iact, fStep, &safe);
      fNextNode = fTopNode;
      if (snext < fStep) {
         fIsStepEntering = kTRUE;
         fStep = snext;
         Int_t indnext = fNextNode->GetVolume()->GetNextNodeIndex();;
         while (indnext>=0) {
            fNextNode = fNextNode->GetDaughter(indnext);
            if (computeGlobal) fCurrentMatrix->Multiply(fNextNode->GetMatrix());
            indnext = fNextNode->GetVolume()->GetNextNodeIndex();
         }
         return fNextNode;
      }
      return 0;
   }
   fCache->MasterToLocal(fPoint, &point[0]);
   fCache->MasterToLocalVect(fDirection, &dir[0]);
   TGeoVolume *vol = fCurrentNode->GetVolume();
   // find distance to exiting current node
   snext = vol->GetShape()->DistFromInside(&point[0], &dir[0], iact, fStep, &safe);
   if (snext < fStep) {
      fNextNode = fCurrentNode;
      fStep = snext;
      fIsStepEntering = kFALSE;
      if (fStep<1E-6) return fCurrentNode;
   }
   fNextNode = (fStep<1E20)?fCurrentNode:0;
   // Find next daughter boundary for the current volume
   Int_t idaughter = -1;
   FindNextDaughterBoundary(point,dir,idaughter,computeGlobal);
   
   TGeoNode *current = 0;
   TGeoNode *dnode = 0;
   TGeoVolume *mother = 0;
   // if we are in an overlapping node, check also the mother(s)
   if (fNmany) {
      Double_t mothpt[3];
      Double_t vecpt[3];
      Double_t dpt[3], dvec[3];
      Int_t novlps;
      Int_t idovlp = -1;
      Int_t safelevel = GetSafeLevel();
      PushPath(safelevel+1);
      while (fCurrentOverlapping) {
         Int_t *ovlps = fCurrentNode->GetOverlaps(novlps);
         CdUp();
         mother = fCurrentNode->GetVolume();
         fCache->MasterToLocal(fPoint, &mothpt[0]);
         fCache->MasterToLocalVect(fDirection, &vecpt[0]);
         // check distance to out
         snext = TGeoShape::Big();
         if (!mother->IsAssembly()) snext = mother->GetShape()->DistFromInside(&mothpt[0], &vecpt[0], iact, fStep, &safe);
         if (snext<fStep) {
            fIsStepEntering = kFALSE;
            fStep = snext;
            if (computeGlobal) *fCurrentMatrix = GetCurrentMatrix();
            fNextNode = fCurrentNode;
         }
         // check overlapping nodes
         for (Int_t i=0; i<novlps; i++) {
            current = mother->GetNode(ovlps[i]);
            if (!current->IsOverlapping()) {
               current->cd();
               current->MasterToLocal(&mothpt[0], &dpt[0]);
               current->MasterToLocalVect(&vecpt[0], &dvec[0]);
               snext = current->GetVolume()->GetShape()->DistFromOutside(&dpt[0], &dvec[0], iact, fStep, &safe);
               if (snext<fStep) {
                  if (computeGlobal) {
                     *fCurrentMatrix = GetCurrentMatrix();
                     fCurrentMatrix->Multiply(current->GetMatrix());
                  }
                  fIsStepEntering = kFALSE;
                  fStep = snext;
                  fNextNode = current;
               }
            } else {
               // another many - check if point is in or out
               current->cd();
               current->MasterToLocal(&mothpt[0], &dpt[0]);
               current->MasterToLocalVect(&vecpt[0], &dvec[0]);
               if (current->GetVolume()->Contains(dpt)) {
                  if (current->GetVolume()->GetNdaughters()) {
                     CdDown(ovlps[i]);
                     dnode = FindNextDaughterBoundary(dpt,dvec,idovlp,computeGlobal);
                     if (dnode && computeGlobal) {
                        *fCurrentMatrix = GetCurrentMatrix();
                        fCurrentMatrix->Multiply(dnode->GetMatrix());
                     }
                     fNextNode = dnode;
                     CdUp();
                  }   
               } else {
                  snext = current->GetVolume()->GetShape()->DistFromOutside(&dpt[0], &dvec[0], iact, fStep, &safe);
                  if (snext<fStep) {
                     if (computeGlobal) {
                        *fCurrentMatrix = GetCurrentMatrix();
                        fCurrentMatrix->Multiply(current->GetMatrix());
                     }
                     fIsStepEntering = kFALSE;
                     fStep = snext;
                     fNextNode = current;
                  }               
               }  
            }
         }
      }
      // Now we are in a non-overlapping node
      if (fNmany) {
      // We have overlaps up in the branch, check distance to exit
         Int_t up = 1;
         Int_t imother;
         Int_t nmany = fNmany;
         Bool_t ovlp = kFALSE;
         Bool_t nextovlp = kFALSE;
         Bool_t offset = kFALSE;
         TGeoNode *current = fCurrentNode;
         TGeoNode *mother, *mup;
         TGeoHMatrix *matrix;
         while (nmany) {
            mother = GetMother(up);
            mup = mother;
            imother = up+1;
            offset = kFALSE;
            while (mup->IsOffset()) {
               mup = GetMother(imother++);
               offset = kTRUE;
            }   
            nextovlp = mup->IsOverlapping();
            if (offset) {
               mother = mup;
               if (nextovlp) nmany -= imother-up;
               up = imother-1;
            } else {    
               if (ovlp) nmany--;
            }   
            if (ovlp || nextovlp) {
               matrix = GetMotherMatrix(up);
               matrix->MasterToLocal(fPoint,dpt);
               matrix->MasterToLocalVect(fDirection,dvec);
               snext = TGeoShape::Big();
               if (!mother->GetVolume()->IsAssembly()) snext = mother->GetVolume()->GetShape()->DistFromInside(dpt,dvec,iact,fStep);
               if (snext<fStep) {
                  fStep = snext;
                  fNextNode = mother;
                  if (computeGlobal) *fCurrentMatrix = matrix;
                  while (up--) CdUp();
                  up = 1;
                  current = fCurrentNode;
                  ovlp = current->IsOverlapping();
                  continue;
               }   
            }   
            current = mother;
            ovlp = nextovlp;
            up++;            
         }
      }      
      PopPath();
   }
   return fNextNode;
}

//_____________________________________________________________________________
TGeoNode *TGeoManager::FindNextDaughterBoundary(Double_t *point, Double_t *dir, Int_t &idaughter, Bool_t compmatrix)
{
// Computes as fStep the distance to next daughter of the current volume. 
// The point and direction must be converted in the coordinate system of the current volume.
// The proposed step limit is fStep.

   Double_t snext = TGeoShape::Big();
   idaughter = -1; // nothing crossed
   TGeoNode *nodefound = 0;
   // Get number of daughters. If no daughters we are done.

   TGeoVolume *vol = fCurrentNode->GetVolume();
   Int_t nd = vol->GetNdaughters();
   if (!nd) return 0;  // No daughter 
   if (fActivity && !vol->IsActiveDaughters()) return 0;
   Double_t lpoint[3], ldir[3];
   TGeoNode *current = 0;
   Int_t i=0;
   // if current volume is divided, we are in the non-divided region. We
   // check only the first and the last cell
   TGeoPatternFinder *finder = vol->GetFinder();
   if (finder) {
      Int_t ifirst = finder->GetDivIndex();
      current = vol->GetNode(ifirst);
      current->cd();
      current->MasterToLocal(&point[0], lpoint);
      current->MasterToLocalVect(&dir[0], ldir);
      snext = current->GetVolume()->GetShape()->DistFromOutside(lpoint, ldir, 3, fStep);
      if (snext<fStep) {
         if (compmatrix) {
            *fCurrentMatrix = GetCurrentMatrix();
            fCurrentMatrix->Multiply(current->GetMatrix());
         }
         fIsStepEntering = kTRUE;
         fStep=snext;
         fNextNode = current;
         nodefound = current;
         idaughter = ifirst;
      }
      Int_t ilast = ifirst+finder->GetNdiv()-1;
      if (ilast==ifirst) return fNextNode;
      current = vol->GetNode(ilast);
      current->cd();
      current->MasterToLocal(&point[0], lpoint);
      current->MasterToLocalVect(&dir[0], ldir);
      snext = current->GetVolume()->GetShape()->DistFromOutside(lpoint, ldir, 3, fStep);
      if (snext<fStep) {
         if (compmatrix) {
            *fCurrentMatrix = GetCurrentMatrix();
            fCurrentMatrix->Multiply(current->GetMatrix());
         }
         fIsStepEntering = kTRUE;
         fStep=snext;
         fNextNode = current;
         nodefound = current;
         idaughter = ilast;
      }
      return nodefound;
   }
   // if only few daughters, check all and exit
   TGeoVoxelFinder *voxels = vol->GetVoxels();
   Int_t indnext;
   if (nd<5 || !voxels) {
      for (i=0; i<nd; i++) {
         current = vol->GetNode(i);
         if (fActivity && !current->GetVolume()->IsActive()) continue;
         current->cd();
         if (voxels && voxels->IsSafeVoxel(point, i, fStep)) continue;
         current->MasterToLocal(point, lpoint);
         current->MasterToLocalVect(dir, ldir);
         if (current->IsOverlapping() && current->GetVolume()->Contains(lpoint)) continue;
         snext = current->GetVolume()->GetShape()->DistFromOutside(lpoint, ldir, 3, fStep);
         if (snext<fStep) {
            indnext = current->GetVolume()->GetNextNodeIndex();
            if (compmatrix) {
               *fCurrentMatrix = GetCurrentMatrix();
               fCurrentMatrix->Multiply(current->GetMatrix());
            }    
            fIsStepEntering = kTRUE;
            fStep=snext;
            fNextNode = current;
            nodefound = fNextNode;   
            idaughter = i;   
            while (indnext>=0) {
               current = current->GetDaughter(indnext);
               if (compmatrix) fCurrentMatrix->Multiply(current->GetMatrix());
               fNextNode = current;
               nodefound = current;
               indnext = current->GetVolume()->GetNextNodeIndex();
            }
         }
      }
      return nodefound;
   }
   // if current volume is voxelized, first get current voxel
   Int_t ncheck = 0;
   Int_t *vlist = 0;
   voxels->SortCrossedVoxels(point, dir);
   while ((vlist=voxels->GetNextVoxel(point, dir, ncheck))) {
      for (i=0; i<ncheck; i++) {
         current = vol->GetNode(vlist[i]);
         if (fActivity && !current->GetVolume()->IsActive()) continue;
         current->cd();
         current->MasterToLocal(point, lpoint);
         current->MasterToLocalVect(dir, ldir);
         if (current->IsOverlapping() && current->GetVolume()->Contains(lpoint)) continue;
         snext = current->GetVolume()->GetShape()->DistFromOutside(lpoint, ldir, 3, fStep);
         if (snext<fStep) {
            indnext = current->GetVolume()->GetNextNodeIndex();
            if (compmatrix) {
               *fCurrentMatrix = GetCurrentMatrix();
               fCurrentMatrix->Multiply(current->GetMatrix());
            }
            fIsStepEntering = kTRUE;
            fStep=snext;
            fNextNode = current;
            nodefound = fNextNode;
            idaughter = vlist[i];
            while (indnext>=0) {
               current = current->GetDaughter(indnext);
               if (compmatrix) fCurrentMatrix->Multiply(current->GetMatrix());
               fNextNode = current;
               nodefound = current;
               indnext = current->GetVolume()->GetNextNodeIndex();
            }
         }
      }
   }
   return nodefound;
}

//_____________________________________________________________________________
TGeoNode *TGeoManager::FindNode(Bool_t safe_start)
{
// Returns deepest node containing current point.
   fSafety = 0;
   fSearchOverlaps = kFALSE;
   fIsOutside = kFALSE;
   fIsEntering = fIsExiting = kFALSE;
   fIsOnBoundary = kFALSE;
   fStartSafe = safe_start;
   fIsSameLocation = kTRUE;
   TGeoNode *last = fCurrentNode;
   TGeoNode *found = SearchNode();
   if (found != last) {
      fIsSameLocation = kFALSE;
   } else {   
      if (last->IsOverlapping()) fIsSameLocation = kTRUE;
   }   
   return found;
}

//_____________________________________________________________________________
TGeoNode *TGeoManager::FindNode(Double_t x, Double_t y, Double_t z)
{
// Returns deepest node containing current point.
   fPoint[0] = x;
   fPoint[1] = y;
   fPoint[2] = z;
   fSafety = 0;
   fSearchOverlaps = kFALSE;
   fIsOutside = kFALSE;
   fIsEntering = fIsExiting = kFALSE;
   fIsOnBoundary = kFALSE;
   fStartSafe = kTRUE;
   fIsSameLocation = kTRUE;
   TGeoNode *last = fCurrentNode;
   TGeoNode *found = SearchNode();
   if (found != last) {
      fIsSameLocation = kFALSE;
   } else {   
      if (last->IsOverlapping()) fIsSameLocation = kTRUE;
   }   
   return found;
}

//_____________________________________________________________________________
Double_t *TGeoManager::FindNormalFast()
{
// Computes fast normal to next crossed boundary, assuming that the current point
// is close enough to the boundary. Works only after calling FindNextBoundary.
   if (!fNextNode) return 0;
   Double_t local[3];
   Double_t ldir[3];
   Double_t lnorm[3];
   fCurrentMatrix->MasterToLocal(fPoint, local);
   fCurrentMatrix->MasterToLocalVect(fDirection, ldir);
   fNextNode->GetVolume()->GetShape()->ComputeNormal(local, ldir,lnorm);
   fCurrentMatrix->LocalToMasterVect(lnorm, fNormal);
   return fNormal;
}

//_____________________________________________________________________________
Double_t *TGeoManager::FindNormal(Bool_t forward)
{
// Computes normal vector to the next surface that will be or was already
// crossed when propagating on a straight line from a given point/direction.
// Returns the normal vector cosines in the MASTER coordinate system. The dot
// product of the normal and the current direction is positive defined.
   Double_t lnorm[3];
   Double_t lpt[3];
   Double_t ldir[3];
   if (fIsStepEntering && !forward) {
      MasterToLocal(fPoint, lpt);
      MasterToLocalVect(fDirection, ldir);
      fCurrentNode->GetVolume()->GetShape()->ComputeNormal(lpt,ldir,lnorm);
      LocalToMasterVect(lnorm, fNormal);
      return fNormal;
   }

   // we have to compute the normal to the current node
   Double_t saved_point[3];
   Double_t saved_direction[3];
   Double_t saved_step = fStep;
   Double_t is_entering = fIsEntering;
   Double_t bigstep = 1.E6;
   Int_t i, istep;
   Int_t start = PushPath();
   memcpy(saved_point, fPoint, kN3);
   memcpy(saved_direction, fDirection, kN3);
//   PushPath();

   if (!forward) {
      // change to opposite direction
      for (i=0; i<3; i++) fDirection[i] *= -1.;
      // compute distance to already crossed boundary
      FindNextBoundary();
      // if it is nothing backwards, restore state and  return NULL
      if (fStep>bigstep) {
         memcpy(fPoint, saved_point, kN3);
         memcpy(fDirection, saved_direction, kN3);
         fStep = saved_step;
         fIsEntering = is_entering;
         PopPath();
         Warning("FindNormal","nothing backwards\n");
         return 0;
      }
      // try to cross the boundary
      istep = 0;
      Step();
      // if this fails, do extra small steps (up to a total of 1cm)
      while (!fIsEntering) {
         istep++;
         if (istep>1E3) {
            // we have a big problem not being able to reach the boundary
            Error("FindNormal", "cannot reach backward boundary");
            Error("FindNormal", "starting point was : (%f, %f, %f)", saved_point[0], saved_point[1], saved_point[2]);
            Error("FindNormal", "direction was      : (%f, %f, %f)", fDirection[0], fDirection[1], fDirection[2]);
            memcpy(fPoint, saved_point, kN3);
            memcpy(fDirection, saved_direction, kN3);
            fStep = saved_step;
            fIsEntering = is_entering;
            PopPath();
            return 0;
         }
         fStep = 1E-2;
         Step();
      }
      if (!fIsStepEntering) PopPath(start);
      // restore initial direction
      memcpy(fDirection, saved_direction, kN3);
   } else {
      FindNextBoundary();
      if (fStep>bigstep) {
         memcpy(fPoint, saved_point, kN3);
         memcpy(fDirection, saved_direction, kN3);
         fStep = saved_step;
         fIsEntering = is_entering;
         PopPath();
         Error("FindNormal","nothing forward...");
         return 0;
      }
      // try to cross the boundary
      istep = 0;
      Step();
      // if this fails, do extra small steps (up to a total of 10cm)
      while (!fIsEntering) {
         istep++;
         if (istep>1E3) {
            // we have a big problem not being able to reach the boundary
            Error("FindNormal", "cannot reach forward boundary");
            Error("FindNormal", "starting point was : (%f, %f, %f)", saved_point[0], saved_point[1], saved_point[2]);
            Error("FindNormal", "direction was      : (%f, %f, %f)", fDirection[0], fDirection[1], fDirection[2]);
            memcpy(fPoint, saved_point, kN3);
            memcpy(fDirection, saved_direction, kN3);
            fStep = saved_step;
            fIsEntering = is_entering;
            PopPath();
            return 0;
         }
         fStep = 1E-2;
         Step();
      }
      if (!fIsStepEntering) PopPath(start);
   }
   // now the current point is close to the boundary of the current node
   // we compute the normal
   MasterToLocal(fPoint, lpt);
   MasterToLocalVect(fDirection, ldir);
   fCurrentNode->GetVolume()->GetShape()->ComputeNormal(lpt,ldir,lnorm);
   LocalToMasterVect(lnorm, fNormal);
   memcpy(fPoint, saved_point, kN3);
   memcpy(fDirection, saved_direction, kN3);
   fStep = saved_step;
   fIsEntering = is_entering;
   PopPath();
   return fNormal;
}

//_____________________________________________________________________________
Bool_t TGeoManager::IsSameLocation(Double_t x, Double_t y, Double_t z, Bool_t change)
{
// Checks if point (x,y,z) is still in the current node.
   // check if this is an overlapping node
   Double_t oldpt[3];
   if (fLastSafety>0) {
      Double_t dx = (x-fLastPoint[0]);
      Double_t dy = (y-fLastPoint[1]);
      Double_t dz = (z-fLastPoint[2]);
      Double_t dsq = dx*dx+dy*dy+dz*dz;
      if (dsq<fLastSafety*fLastSafety) return kTRUE;
   }
   if (fCurrentOverlapping) {
//      TGeoNode *current = fCurrentNode;
      Int_t cid = GetCurrentNodeId();
      if (!change) PushPoint();
      memcpy(oldpt, fPoint, kN3);
      gGeoManager->SetCurrentPoint(x,y,z);
      gGeoManager->SearchNode();
      memcpy(fPoint, oldpt, kN3);
      Bool_t same = (cid==GetCurrentNodeId())?kTRUE:kFALSE;
      if (!change) PopPoint();
      return same;
   }

   Double_t point[3];
   point[0] = x;
   point[1] = y;
   point[2] = z;
   if (change) memcpy(fPoint, point, kN3);
   TGeoVolume *vol = fCurrentNode->GetVolume();
   if (fIsOutside) {
      if (vol->GetShape()->Contains(point)) {
         if (!change) return kFALSE;
         FindNode(x,y,z);
         return kFALSE;
      }
      return kTRUE;
   }
   Double_t local[3];
   // convert to local frame
   MasterToLocal(point,local);
   // check if still in current volume.
   if (!vol->GetShape()->Contains(local)) {
      if (!change) return kFALSE;
      CdUp();
      FindNode(x,y,z);
      return kFALSE;
   }

   // check if there are daughters
   Int_t nd = vol->GetNdaughters();
   if (!nd) return kTRUE;

   TGeoNode *node;
   TGeoPatternFinder *finder = vol->GetFinder();
   if (finder) {
      node=finder->FindNode(local);
      if (node) {
         if (!change) return kFALSE;
         CdDown(node->GetIndex());
         SearchNode(kTRUE,node);
         return kFALSE;
      }
      return kTRUE;
   }
   // if we are not allowed to do changes, save the current path
   TGeoVoxelFinder *voxels = vol->GetVoxels();
   Int_t *check_list = 0;
   Int_t ncheck = 0;
   Double_t local1[3];
   if (voxels) {
      check_list = voxels->GetCheckList(local, ncheck);
      if (!check_list) return kTRUE;
      if (!change) PushPath();
      for (Int_t id=0; id<ncheck; id++) {
         node = vol->GetNode(check_list[id]);
         CdDown(check_list[id]);
         fCurrentNode->GetMatrix()->MasterToLocal(local,local1);
         if (fCurrentNode->GetVolume()->GetShape()->Contains(local1)) {
            if (!change) {
               PopPath();
               return kFALSE;
            }
            SearchNode(kTRUE);
            return kFALSE;
         }
         CdUp();
      }
      if (!change) PopPath();
      return kTRUE;
   }
   Int_t id = 0;
   if (!change) PushPath();
   while ((node=fCurrentNode->GetDaughter(id++))) {
      CdDown(id-1);
      fCurrentNode->GetMatrix()->MasterToLocal(local,local1);
      if (fCurrentNode->GetVolume()->GetShape()->Contains(local1)) {
         if (!change) {
            PopPath();
            return kFALSE;
         }
         SearchNode(kTRUE);
         return kFALSE;
      }
      CdUp();
      if (id == nd) {
         if (!change) PopPath();
         return kTRUE;
      }
   }
   if (!change) PopPath();
   return kTRUE;
}
//_____________________________________________________________________________
Bool_t TGeoManager::IsSamePoint(Double_t x, Double_t y, Double_t z) const
{
// Check if a new point with given coordinates is the same as the last located one.
   if (x==fLastPoint[0]) {
      if (y==fLastPoint[1]) {
         if (z==fLastPoint[2]) return kTRUE;
      }
   }
   return kFALSE;
}

//_____________________________________________________________________________
Bool_t TGeoManager::IsInPhiRange() const
{
// True if current node is in phi range
   if (!fPhiCut) return kTRUE;
   const Double_t *origin;
   if (!fCurrentNode) return kFALSE;
   origin = ((TGeoBBox*)fCurrentNode->GetVolume()->GetShape())->GetOrigin();
   Double_t point[3];
   LocalToMaster(origin, &point[0]);
   Double_t phi = TMath::ATan2(point[1], point[0])*TMath::RadToDeg();
   if (phi<0) phi+=360.;
   if ((phi>=fPhimin) && (phi<=fPhimax)) return kFALSE;
   return kTRUE;
}
//_____________________________________________________________________________
TGeoNode *TGeoManager::InitTrack(Double_t *point, Double_t *dir)
{
// Initialize current point and current direction vector (normalized)
// in MARS. Return corresponding node.
   SetCurrentPoint(point);
   SetCurrentDirection(dir);
   return FindNode();
}
//_____________________________________________________________________________
TGeoNode *TGeoManager::InitTrack(Double_t x, Double_t y, Double_t z, Double_t nx, Double_t ny, Double_t nz)
{
// Initialize current point and current direction vector (normalized)
// in MARS. Return corresponding node.
   SetCurrentPoint(x,y,z);
   SetCurrentDirection(nx,ny,nz);
   return FindNode();
}

//_____________________________________________________________________________
void TGeoManager::InspectState() const
{
// Inspects path and all flags for the current state.
   Info("InspectState","Current path is: %s",GetPath());
   Int_t level;
   TGeoNode *node;
   Bool_t is_offset, is_overlapping;
   for (level=0; level<fLevel+1; level++) {
      node = GetMother(fLevel-level);
      if (!node) continue;
      is_offset = node->IsOffset();
      is_overlapping = node->IsOverlapping();
      Info("InspectState","level %i: %s  div=%i  many=%i",level,node->GetName(),is_offset,is_overlapping);
   }
   Info("InspectState","on_bound=%i   entering=%i", fIsOnBoundary, fIsEntering);
}      

//_____________________________________________________________________________
const char *TGeoManager::GetPath() const
{
// Get path to the current node in the form /node0/node1/...
   if (fIsOutside) return kGeoOutsidePath;
   return fCache->GetPath();
}
//_____________________________________________________________________________
Int_t TGeoManager::GetByteCount(Option_t * /*option*/)
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
   TIter next3(fMedia);
   TGeoMedium *med;
   while ((med=(TGeoMedium*)next3())) count += med->GetByteCount();
   Info("GetByteCount","Total size of logical tree : %i bytes", count);
   return count;
}
//_____________________________________________________________________________
TVirtualGeoPainter *TGeoManager::GetGeomPainter()
{
// Make a default painter if none present. Returns pointer to it.
   if (!fPainter) {
      TPluginHandler *h;
      if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualGeoPainter"))) {
         if (h->LoadPlugin() == -1)
            return 0;
         fPainter = (TVirtualGeoPainter*)h->ExecPlugin(1,this);
         if (!fPainter) {
            Error("GetGeomPainter", "could not create painter");
            return 0;
         }
      }
   }
   return fPainter;
}
//_____________________________________________________________________________
TGeoVolume *TGeoManager::GetVolume(const char *name) const
{
// Search for a named volume. All trailing blanks stripped.
   TString sname = name;
   sname = sname.Strip();
   TGeoVolume *vol = (TGeoVolume*)fVolumes->FindObject(sname.Data());
   return vol;
}

//_____________________________________________________________________________
TGeoVolume *TGeoManager::FindVolumeFast(const char *name, Bool_t multi)
{
// Fast search for a named volume. All trailing blanks stripped.
   if (!fHashVolumes) {
      Int_t nvol = fVolumes->GetEntriesFast();
      Int_t ngvol = fGVolumes->GetEntriesFast();
      fHashVolumes = new THashList(nvol+1);
      fHashGVolumes = new THashList(ngvol+1);
      Int_t i;
      for (i=0; i<ngvol; i++) fHashGVolumes->AddAt(fGVolumes->At(i),i);
      for (i=0; i<nvol; i++) fHashVolumes->AddAt(fVolumes->At(i),i);
   }   
   TString sname = name;
   sname = sname.Strip();
   THashList *list = fHashVolumes;
   if (multi) list = fHashGVolumes;
   TGeoVolume *vol = (TGeoVolume*)list->FindObject(sname.Data());
   return vol;
}

//_____________________________________________________________________________
Int_t TGeoManager::GetUID(const char *volname) const
{
// Retreive unique id for a volume name. Return -1 if name not found.
   TGeoManager *geom = (TGeoManager*)this;
   TGeoVolume *vol = geom->FindVolumeFast(volname, kFALSE);
   if (!vol) vol = geom->FindVolumeFast(volname, kTRUE);
   if (!vol) return -1;
   return vol->GetNumber();
}

//_____________________________________________________________________________
TGeoMaterial *TGeoManager::FindDuplicateMaterial(const TGeoMaterial *mat) const
{
// Find if a given material duplicates an existing one.
   Int_t index = fMaterials->IndexOf(mat);
   if (index <= 0) return 0;
   TGeoMaterial *other;
   for (Int_t i=0; i<index; i++) {
      other = (TGeoMaterial*)fMaterials->At(i);
      if (other == mat) continue;
      if (other->IsEq(mat)) return other;
   }
   return 0;
}

//_____________________________________________________________________________
TGeoMaterial *TGeoManager::GetMaterial(const char *matname) const
{
// Search for a named material. All trailing blanks stripped.
   TString sname = matname;
   sname = sname.Strip();
   TGeoMaterial *mat = (TGeoMaterial*)fMaterials->FindObject(sname.Data());
   return mat;
}

//_____________________________________________________________________________
TGeoMedium *TGeoManager::GetMedium(const char *medium) const
{
// Search for a named tracking medium. All trailing blanks stripped.
   TString sname = medium;
   sname = sname.Strip();
   TGeoMedium *med = (TGeoMedium*)fMedia->FindObject(sname.Data());
   return med;
}

//_____________________________________________________________________________
TGeoMedium *TGeoManager::GetMedium(Int_t numed) const
{
// Search for a tracking medium with a given ID.
   TIter next(fMedia);
   TGeoMedium *med;
   while ((med=(TGeoMedium*)next())) {
      if (med->GetId()==numed) return med;
   }
   return 0;
}

//_____________________________________________________________________________
TGeoMaterial *TGeoManager::GetMaterial(Int_t id) const
{
// Return material at position id.
   if (id<0 || id >= fMaterials->GetSize()) return 0;
   TGeoMaterial *mat = (TGeoMaterial*)fMaterials->At(id);
   return mat;
}
//_____________________________________________________________________________
Int_t TGeoManager::GetMaterialIndex(const char *matname) const
{
// Return index of named material.
   TIter next(fMaterials);
   TGeoMaterial *mat;
   Int_t id = 0;
   TString sname = matname;
   sname = sname.Strip();
   while ((mat = (TGeoMaterial*)next())) {
      if (!strcmp(mat->GetName(),sname.Data()))
         return id;
      id++;
   }
   return -1;  // fail
}
//_____________________________________________________________________________
void TGeoManager::RandomRays(Int_t nrays, Double_t startx, Double_t starty, Double_t startz)
{
// Randomly shoot nrays and plot intersections with surfaces for current
// top node.
   GetGeomPainter()->RandomRays(nrays, startx, starty, startz);
}

//_____________________________________________________________________________
void TGeoManager::RemoveMaterial(Int_t index)
{
// Remove material at given index.
   TObject *obj = fMaterials->At(index);
   if (obj) fMaterials->Remove(obj);
}

//_____________________________________________________________________________
void TGeoManager::ResetUserData()
{
// Sets all pointers TGeoVolume::fField to NULL. User data becomes decoupled 
// from geometry. Deletion has to be managed by users.
   TIter next(fVolumes);
   TGeoVolume *vol;
   while ((vol=(TGeoVolume*)next())) vol->SetField(0);
}   

//_____________________________________________________________________________
void TGeoManager::RestoreMasterVolume()
{
// Restore the master volume of the geometry.
   if (fTopVolume == fMasterVolume) return;
   if (fMasterVolume) SetTopVolume(fMasterVolume);
}
//_____________________________________________________________________________
void TGeoManager::Voxelize(Option_t *option)
{
// Voxelize all non-divided volumes.
   TGeoVolume *vol;
   TGeoVoxelFinder *vox = 0;
   if (!fStreamVoxels) Info("Voxelize","Voxelizing...");
//   Int_t nentries = fVolumes->GetSize();
   TIter next(fVolumes);
   while ((vol = (TGeoVolume*)next())) {
      if (!fIsGeomReading) vol->SortNodes();
      if (!fStreamVoxels) {
         vol->Voxelize(option);
      } else {
         vox = vol->GetVoxels();
         if (vox) vox->CreateCheckList();
      }
      if (!fIsGeomReading) vol->FindOverlaps();
   }
}
//_____________________________________________________________________________
void TGeoManager::ModifiedPad() const
{
// Send "Modified" signal to painter.
   if (!fPainter) return;
   fPainter->ModifiedPad();
}
//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakeArb8(const char *name, const TGeoMedium *medium,
                                  Double_t dz, Double_t *vertices)
{
// Make an TGeoArb8 volume.
   TGeoArb8 *arb = new TGeoArb8(dz, vertices);
   TGeoVolume *vol = new TGeoVolume(name, arb, medium);
   return vol;
}
//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakeBox(const char *name, const TGeoMedium *medium,
                                    Double_t dx, Double_t dy, Double_t dz)
{
// Make in one step a volume pointing to a box shape with given medium.
   TGeoBBox *box = new TGeoBBox(dx, dy, dz);
   TGeoVolume *vol = 0;
   if (box->IsRunTimeShape()) {
      vol = MakeVolumeMulti(name, medium);
      vol->SetShape(box);
   } else {
      vol = new TGeoVolume(name, box, medium);
   }
   return vol;
}
//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakePara(const char *name, const TGeoMedium *medium,
                                    Double_t dx, Double_t dy, Double_t dz,
                                    Double_t alpha, Double_t theta, Double_t phi)
{
// Make in one step a volume pointing to a paralelipiped shape with given medium.
   if ((alpha==0) && (theta==0)) {
      Warning("MakePara","parallelipiped %s having alpha=0, theta=0 -> making box instead", name);
      return MakeBox(name, medium, dx, dy, dz);
   }
   TGeoPara *para=0;
   para = new TGeoPara(dx, dy, dz, alpha, theta, phi);
   TGeoVolume *vol = 0;
   if (para->IsRunTimeShape()) {
      vol = MakeVolumeMulti(name, medium);
      vol->SetShape(para);
   } else {
      vol = new TGeoVolume(name, para, medium);
   }
   return vol;
}
//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakeSphere(const char *name, const TGeoMedium *medium,
                                    Double_t rmin, Double_t rmax, Double_t themin, Double_t themax,
                                    Double_t phimin, Double_t phimax)
{
// Make in one step a volume pointing to a sphere shape with given medium
   TGeoSphere *sph = new TGeoSphere(rmin, rmax, themin, themax, phimin, phimax);
   TGeoVolume *vol = new TGeoVolume(name, sph, medium);
   return vol;
}

//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakeTorus(const char *name, const TGeoMedium *medium, Double_t r,
                                   Double_t rmin, Double_t rmax, Double_t phi1, Double_t dphi)
{
// Make in one step a volume pointing to a torus shape with given medium.
   TGeoTorus *tor = new TGeoTorus(name,r,rmin,rmax,phi1,dphi);
   TGeoVolume *vol = new TGeoVolume(name, tor, medium);
   return vol;
}

//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakeTube(const char *name, const TGeoMedium *medium,
                                     Double_t rmin, Double_t rmax, Double_t dz)
{
// Make in one step a volume pointing to a tube shape with given medium.
   if (rmin>rmax) {
      Error("MakeTube", "tube %s, Rmin=%g greater than Rmax=%g", name,rmin,rmax);
   }
   TGeoTube *tube = new TGeoTube(rmin, rmax, dz);
   TGeoVolume *vol = 0;
   if (tube->IsRunTimeShape()) {
      vol = MakeVolumeMulti(name, medium);
      vol->SetShape(tube);
   } else {
      vol = new TGeoVolume(name, tube, medium);
   }
   return vol;
}
//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakeTubs(const char *name, const TGeoMedium *medium,
                                     Double_t rmin, Double_t rmax, Double_t dz,
                                     Double_t phi1, Double_t phi2)
{
// Make in one step a volume pointing to a tube segment shape with given medium.
   TGeoTubeSeg *tubs = new TGeoTubeSeg(rmin, rmax, dz, phi1, phi2);
   TGeoVolume *vol = 0;
   if (tubs->IsRunTimeShape()) {
      vol = MakeVolumeMulti(name, medium);
      vol->SetShape(tubs);
   } else {
      vol = new TGeoVolume(name, tubs, medium);
   }
   return vol;
}
//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakeEltu(const char *name, const TGeoMedium *medium,
                                     Double_t a, Double_t b, Double_t dz)
{
// Make in one step a volume pointing to a tube shape with given medium
   TGeoEltu *eltu = new TGeoEltu(a, b, dz);
   TGeoVolume *vol = 0;
   if (eltu->IsRunTimeShape()) {
      vol = MakeVolumeMulti(name, medium);
      vol->SetShape(eltu);
   } else {
      vol = new TGeoVolume(name, eltu, medium);
   }
   return vol;
}
//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakeHype(const char *name, const TGeoMedium *medium,
                                        Double_t rin, Double_t stin, Double_t rout, Double_t stout, Double_t dz)
{
// Make in one step a volume pointing to a tube shape with given medium
   TGeoHype * hype =  new TGeoHype(name, rin,stin,rout,stout,dz);
   TGeoVolume *vol = 0;
   if (hype->IsRunTimeShape()) {
      vol = MakeVolumeMulti(name, medium);
      vol->SetShape(hype);
   } else {
      vol = new TGeoVolume(name, hype, medium);
   }
   return vol;
}

//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakeParaboloid(const char *name, const TGeoMedium *medium,
                                        Double_t rlo, Double_t rhi, Double_t dz)
{
// Make in one step a volume pointing to a tube shape with given medium
   TGeoParaboloid *parab = new TGeoParaboloid(rlo, rhi, dz);
   TGeoVolume *vol = 0;
   if (parab->IsRunTimeShape()) {
      vol = MakeVolumeMulti(name, medium);
      vol->SetShape(parab);
   } else {
      vol = new TGeoVolume(name, parab, medium);
   }
   return vol;
}
//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakeCtub(const char *name, const TGeoMedium *medium,
                                     Double_t rmin, Double_t rmax, Double_t dz, Double_t phi1, Double_t phi2,
                                     Double_t lx, Double_t ly, Double_t lz, Double_t tx, Double_t ty, Double_t tz)
{
// Make in one step a volume pointing to a tube segment shape with given medium
   TGeoCtub *ctub = new TGeoCtub(rmin, rmax, dz, phi1, phi2, lx, ly, lz, tx, ty, tz);
   TGeoVolume *vol = new TGeoVolume(name, ctub, medium);
   return vol;
}
//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakeCone(const char *name, const TGeoMedium *medium,
                                     Double_t dz, Double_t rmin1, Double_t rmax1,
                                     Double_t rmin2, Double_t rmax2)
{
// Make in one step a volume pointing to a cone shape with given medium.
   TGeoCone *cone = new TGeoCone(dz, rmin1, rmax1, rmin2, rmax2);
   TGeoVolume *vol = 0;
   if (cone->IsRunTimeShape()) {
      vol = MakeVolumeMulti(name, medium);
      vol->SetShape(cone);
   } else {
      vol = new TGeoVolume(name, cone, medium);
   }
   return vol;
}
//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakeCons(const char *name, const TGeoMedium *medium,
                                     Double_t dz, Double_t rmin1, Double_t rmax1,
                                     Double_t rmin2, Double_t rmax2,
                                     Double_t phi1, Double_t phi2)
{
// Make in one step a volume pointing to a cone segment shape with given medium
   TGeoConeSeg *cons = new TGeoConeSeg(dz, rmin1, rmax1, rmin2, rmax2, phi1, phi2);
   TGeoVolume *vol = 0;
   if (cons->IsRunTimeShape()) {
      vol = MakeVolumeMulti(name, medium);
      vol->SetShape(cons);
   } else {
      vol = new TGeoVolume(name, cons, medium);
   }
   return vol;
}
//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakePcon(const char *name, const TGeoMedium *medium,
                                     Double_t phi, Double_t dphi, Int_t nz)
{
// Make in one step a volume pointing to a polycone shape with given medium.
   TGeoPcon *pcon = new TGeoPcon(phi, dphi, nz);
   TGeoVolume *vol = new TGeoVolume(name, pcon, medium);
   return vol;
}
//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakePgon(const char *name, const TGeoMedium *medium,
                                     Double_t phi, Double_t dphi, Int_t nedges, Int_t nz)
{
// Make in one step a volume pointing to a polygone shape with given medium.
   TGeoPgon *pgon = new TGeoPgon(phi, dphi, nedges, nz);
   TGeoVolume *vol = new TGeoVolume(name, pgon, medium);
   return vol;
}
//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakeTrd1(const char *name, const TGeoMedium *medium,
                                  Double_t dx1, Double_t dx2, Double_t dy, Double_t dz)
{
// Make in one step a volume pointing to a TGeoTrd1 shape with given medium.
   TGeoTrd1 *trd1 = new TGeoTrd1(dx1, dx2, dy, dz);
   TGeoVolume *vol = 0;
   if (trd1->IsRunTimeShape()) {
      vol = MakeVolumeMulti(name, medium);
      vol->SetShape(trd1);
   } else {
      vol = new TGeoVolume(name, trd1, medium);
   }
   return vol;
}
//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakeTrd2(const char *name, const TGeoMedium *medium,
                                  Double_t dx1, Double_t dx2, Double_t dy1, Double_t dy2,
                                  Double_t dz)
{
// Make in one step a volume pointing to a TGeoTrd2 shape with given medium.
   TGeoTrd2 *trd2 = new TGeoTrd2(dx1, dx2, dy1, dy2, dz);
   TGeoVolume *vol = 0;
   if (trd2->IsRunTimeShape()) {
      vol = MakeVolumeMulti(name, medium);
      vol->SetShape(trd2);
   } else {
      vol = new TGeoVolume(name, trd2, medium);
   }
   return vol;
}
//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakeTrap(const char *name, const TGeoMedium *medium,
                                  Double_t dz, Double_t theta, Double_t phi, Double_t h1,
                                  Double_t bl1, Double_t tl1, Double_t alpha1, Double_t h2, Double_t bl2,
                                  Double_t tl2, Double_t alpha2)
{
// Make in one step a volume pointing to a trapezoid shape with given medium.
   TGeoTrap *trap = new TGeoTrap(dz, theta, phi, h1, bl1, tl1, alpha1, h2, bl2,
                                 tl2, alpha2);
   TGeoVolume *vol = new TGeoVolume(name, trap, medium);
   return vol;
}
//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakeGtra(const char *name, const TGeoMedium *medium,
                                  Double_t dz, Double_t theta, Double_t phi, Double_t twist, Double_t h1,
                                  Double_t bl1, Double_t tl1, Double_t alpha1, Double_t h2, Double_t bl2,
                                  Double_t tl2, Double_t alpha2)
{
// Make in one step a volume pointing to a twisted trapezoid shape with given medium.
   TGeoGtra *gtra = new TGeoGtra(dz, theta, phi, twist, h1, bl1, tl1, alpha1, h2, bl2,
                                 tl2, alpha2);
   TGeoVolume *vol = new TGeoVolume(name, gtra, medium);
   return vol;
}
//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakeXtru(const char *name, const TGeoMedium *medium, Int_t nz)
{
// Make a TGeoXtru-shaped volume with nz planes
   TGeoXtru *xtru = new TGeoXtru(nz);

   TGeoVolume *vol = new TGeoVolume(name, xtru, medium);
   return vol;
}

//_____________________________________________________________________________
TGeoPNEntry *TGeoManager::SetAlignableEntry(const char *unique_name, const char *path)
{
// Creates an aligneable object with unique name corresponding to a path
// and adds it to the list of alignables.
   if (!fHashPNE) fHashPNE = new THashList(256,3);
   TGeoPNEntry *entry = GetAlignableEntry(unique_name);
   if (entry) {
      Error("SetAlignableEntry", "An alignable object with name %s already existing. NOT ADDED !", unique_name);
      return 0;
   }
   entry = new TGeoPNEntry(unique_name, path);
   fHashPNE->Add(entry);
   return entry;
}

//_____________________________________________________________________________
TGeoPNEntry *TGeoManager::GetAlignableEntry(const char *name) const
{
// Retreives an existing alignable object.
   if (!fHashPNE) return 0;
   return (TGeoPNEntry*)fHashPNE->FindObject(name);
}   

//_____________________________________________________________________________
TGeoPNEntry *TGeoManager::GetAlignableEntry(Int_t index) const
{
// Retreives an existing alignable object at a given index.
   if (!fHashPNE) return 0;
   return (TGeoPNEntry*)fHashPNE->At(index);
}   

//_____________________________________________________________________________
Int_t TGeoManager::GetNAlignable() const
{
// Retreives an existing alignable object at a given index.
   if (!fHashPNE) return 0;
   return fHashPNE->GetSize();
}   

//_____________________________________________________________________________
TGeoPhysicalNode *TGeoManager::MakeAlignablePN(const char *name)
{
// Make a physical node from the path pointed by an alignable object with a given name.
   TGeoPNEntry *entry = GetAlignableEntry(name);
   if (!entry) {
      Error("MakeAlignablePN","No alignable object named %s found !", name);
      return 0;
   }
   return MakeAlignablePN(entry);
}
      
//_____________________________________________________________________________
TGeoPhysicalNode *TGeoManager::MakeAlignablePN(TGeoPNEntry *entry)
{
// Make a physical node from the path pointed by a given alignable object.
   if (!entry) {
      Error("MakeAlignablePN","No alignable object specified !");
      return 0;
   }
   const char *path = entry->GetTitle();
   if (!cd(path)) {
      Error("MakeAlignablePN", "Alignable object %s poins to invalid path: %s",
            entry->GetName(), path);
      return 0;
   }
   TGeoPhysicalNode *node = MakePhysicalNode(path);
   entry->SetPhysicalNode(node);
   return node;
}        

//_____________________________________________________________________________
TGeoPhysicalNode *TGeoManager::MakePhysicalNode(const char *path)
{
// Makes a physical node corresponding to a path. If PATH is not specified,
// makes physical node matching current modeller state.
   TGeoPhysicalNode *node;
   if (path) {
      node = new TGeoPhysicalNode(path);
   } else {
      node = new TGeoPhysicalNode();
      node->SetBranchAsState();
   }
   fPhysicalNodes->Add(node);
   return node;
}

//_____________________________________________________________________________
void TGeoManager::ClearPhysicalNodes(Bool_t mustdelete)
{
// Clear the current list of physical nodes, so that we can start over with a new list.
// If MUSTDELETE is true, delete previous nodes.
   if (mustdelete) fPhysicalNodes->Delete();
   else fPhysicalNodes->Clear();
}

//_____________________________________________________________________________
TGeoVolumeAssembly *TGeoManager::MakeVolumeAssembly(const char *name)
{
// Make an assembly of volumes.
   return (new TGeoVolumeAssembly(name));
}

//_____________________________________________________________________________
TGeoVolumeMulti *TGeoManager::MakeVolumeMulti(const char *name, const TGeoMedium *medium)
{
// Make a TGeoVolumeMulti handling a list of volumes.
   return (new TGeoVolumeMulti(name, medium));
}

//_____________________________________________________________________________
void TGeoManager::SetExplodedView(Int_t ibomb)
{
// Set type of exploding view (see TGeoPainter::SetExplodedView())
   if ((ibomb>=0) && (ibomb<4)) fExplodedView = ibomb;
   if (fPainter) fPainter->SetExplodedView(ibomb);
}

//_____________________________________________________________________________
void TGeoManager::SetPhiRange(Double_t phimin, Double_t phimax)
{
// Set cut phi range
   if ((phimin==0) && (phimax==360)) {
      fPhiCut = kFALSE;
      return;
   }
   fPhiCut = kTRUE;
   fPhimin = phimin;
   fPhimax = phimax;
}

//_____________________________________________________________________________
void TGeoManager::SetNsegments(Int_t nseg)
{
// Set number of segments for approximating circles in drawing.
   if (fNsegments==nseg) return;
   if (nseg>2) fNsegments = nseg;
   if (fPainter) fPainter->SetNsegments(nseg);
}

//_____________________________________________________________________________
Int_t TGeoManager::GetNsegments() const
{
// Get number of segments approximating circles
   return fNsegments;
}

//_____________________________________________________________________________
void TGeoManager::BuildDefaultMaterials()
{
// Build the default materials. A list of those can be found in ...
//   new TGeoMaterial("Air", 14.61, 7.3, 0.001205);
   fElementTable = new TGeoElementTable(200);
}
//_____________________________________________________________________________
TGeoNode *TGeoManager::Step(Bool_t is_geom, Bool_t cross)
{
// Make a rectiliniar step of length fStep from current point (fPoint) on current
// direction (fDirection). If the step is imposed by geometry, is_geom flag
// must be true (default). The cross flag specifies if the boundary should be
// crossed in case of a geometry step (default true). Returns new node after step.
// Set also on boundary condition.
   Double_t epsil = 0;
   if (fStep<1E-6) {
      fIsNullStep=kTRUE;
      if (fStep<0) fStep = 0.;
   } else {
      fIsNullStep=kFALSE;
   }
   if (is_geom) epsil=(cross)?1E-6:-1E-6;
   TGeoNode *old = fCurrentNode;
   Int_t idold = GetNodeId();
   if (fIsOutside) old = 0;
   fStep += epsil;
   for (Int_t i=0; i<3; i++) fPoint[i]+=fStep*fDirection[i];
   TGeoNode *current = FindNode();
   if (is_geom) {
      fIsEntering = (current==old)?kFALSE:kTRUE;
      if (!fIsEntering) {
         Int_t id = GetNodeId();
         fIsEntering = (id==idold)?kFALSE:kTRUE;
      }
      fIsExiting  = !fIsEntering;
      if (fIsEntering && fIsNullStep) fIsNullStep = kFALSE;
      fIsOnBoundary = kTRUE;
   } else {
      fIsEntering = fIsExiting = kFALSE;
      fIsOnBoundary = kFALSE;
   }
   return current;
}
//_____________________________________________________________________________
TGeoNode *TGeoManager::SamplePoints(Int_t npoints, Double_t &dist, Double_t epsil,
                                    const char* g3path)
{
// shoot npoints randomly in a box of 1E-5 arround current point.
// return minimum distance to points outside
   return GetGeomPainter()->SamplePoints(npoints, dist, epsil, g3path);
}
//_____________________________________________________________________________
void TGeoManager::SetTopVolume(TGeoVolume *vol)
{
// Set the top volume and corresponding node as starting point of the geometry.
   if (fTopVolume==vol) return;

   TSeqCollection *brlist = gROOT->GetListOfBrowsers();
   TIter next(brlist);
   TBrowser *browser = 0;

   if (fTopVolume) fTopVolume->SetTitle("");
   fTopVolume = vol;
   vol->SetTitle("Top volume");
   if (fTopNode) {
      TGeoNode *topn = fTopNode;
      fTopNode = 0;
      while ((browser=(TBrowser*)next())) browser->RecursiveRemove(topn);
      delete topn;
   } else {
      fMasterVolume = vol;
      fUniqueVolumes->AddAtAndExpand(vol,0);
      GetHMatrix();
      Info("SetTopVolume","Top volume is %s. Master volume is %s", fTopVolume->GetName(),
           fMasterVolume->GetName());
   }
   *fCurrentMatrix = gGeoIdentity;
//   fMasterVolume->FindMatrixOfDaughterVolume(vol);
//   fCurrentMatrix->Print();
   fTopNode = new TGeoNodeMatrix(vol, gGeoIdentity);
   char *name = new char[strlen(vol->GetName())+3];
   sprintf(name, "%s_1", vol->GetName());
   fTopNode->SetName(name);
   delete [] name;
   fTopNode->SetNumber(1);
   fTopNode->SetTitle("Top logical node");
   fCurrentNode = fTopNode;
   fNodes->AddAt(fTopNode, 0);
   fLevel = 0;
   if (fCache) {
      Bool_t dummy=fCache->IsDummy();
      Bool_t nodeid = fCache->HasIdArray();
      delete fCache;
      fCache = 0;
      BuildCache(dummy,nodeid);
   }
}
//_____________________________________________________________________________
void TGeoManager::SelectTrackingMedia()
{
// Define different tracking media.
//   printf("List of materials :\n");
/*
   Int_t nmat = fMaterials->GetSize();
   if (!nmat) {printf(" No materials !\n"); return;}
   Int_t *media = new Int_t[nmat];
   memset(media, 0, nmat*sizeof(Int_t));
   Int_t imedia = 1;
   TGeoMaterial *mat, *matref;
   mat = (TGeoMaterial*)fMaterials->At(0);
   if (mat->GetMedia()) {
      for (Int_t i=0; i<nmat; i++) {
         mat = (TGeoMaterial*)fMaterials->At(i);
         mat->Print();
      }
      return;
   }
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
*/
}

//_____________________________________________________________________________
void TGeoManager::CheckPoint(Double_t x, Double_t y, Double_t z, Option_t *option)
{
// Classify a given point. See TGeoChecker::CheckPoint().
   GetGeomPainter()->CheckPoint(x,y,z,option);
}

//_____________________________________________________________________________
void TGeoManager::CheckGeometry(Option_t * /*option*/)
{
// Instanciate a TGeoChecker object and investigates the geometry according to
// option. Not implemented yet.
   // check shapes first
   Info("CheckGeometry","Fixing runtime shapes...");
   TIter next(fShapes);
   TGeoShape *shape;
   Bool_t has_runtime = kFALSE;
   while ((shape = (TGeoShape*)next())) {
      if (shape->IsRunTimeShape()) {
         has_runtime = kTRUE;
      }
      if (shape->TestShapeBit(TGeoShape::kGeoPcon) || shape->TestShapeBit(TGeoShape::kGeoArb8))
         if (!shape->TestShapeBit(TGeoShape::kGeoClosedShape)) shape->ComputeBBox();
   }
   if (has_runtime) fTopNode->CheckShapes();
   else Info("CheckGeometry","...Nothing to fix");
}

//_____________________________________________________________________________
void TGeoManager::CheckOverlaps(Double_t ovlp, Option_t * option)
{
// Check all geometry for illegal overlaps within a limit OVLP.
   ClearOverlaps();
   Info("CheckOverlaps", "Checking overlaps for %s within a limit of %g", GetName(),ovlp);
   fSearchOverlaps = kTRUE;
   Int_t nvol = fVolumes->GetEntriesFast();
   Int_t i10 = nvol/10;
   Int_t iv=0;
   TIter next(fVolumes);
   TGeoVolume *vol;
   while ((vol=(TGeoVolume*)next())) {
      iv++;
      if (i10 && nvol>1000) {
         if ((iv%i10) == 0) printf("%i percent\n", Int_t(10*iv/i10));
      }
      if (!vol->GetNdaughters() || vol->GetFinder()) continue;
      vol->CheckOverlaps(ovlp, option);
   }
   SortOverlaps();
   Int_t novlps = fOverlaps->GetEntriesFast();
   TNamed *obj;
   char name[15];
   char num[15];
   Int_t ndigits=1;
   Int_t i,j, result=novlps;
   while ((result /= 10)) ndigits++;
   for (i=0; i<novlps; i++) {
      obj = (TNamed*)fOverlaps->At(i);
      result = i;
      name[0] = 'o';
      name[1] = 'v';
      for (j=0; j<ndigits; j++) name[j+2]='0';
      name[ndigits+2] = 0;
      sprintf(num,"%i", i);
      memcpy(name+2+ndigits-strlen(num), num, strlen(num));
      obj->SetName(name);
   }
   fSearchOverlaps = kFALSE;
   Info("CheckOverlaps","number of illegal overlaps/extrusions : %d", novlps);
}

//_____________________________________________________________________________
void TGeoManager::PrintOverlaps() const
{
// Prints the current list of overlaps.
   if (!fOverlaps) return;
   Int_t novlp = fOverlaps->GetEntriesFast();
   if (!novlp) return;
   fPainter->PrintOverlaps();
}
//_____________________________________________________________________________
void TGeoManager::UpdateCurrentPosition(Double_t * /*nextpoint*/)
{
// Computes and changes the current node according to the new position.
// Not implemented.
}

//_____________________________________________________________________________
Double_t TGeoManager::Weight(Double_t precision, Option_t *option)
{
// Estimate weight of volume VOL with a precision SIGMA(W)/W better than PRECISION.
// Option can be "v" - verbose (default)
   GetGeomPainter();
   TString opt(option);
   opt.ToLower();
   Double_t weight;
   TGeoVolume *volume = fTopVolume;
   if (opt.Contains("v")) {
      if (opt.Contains("a")) {
         Info("Weight", "Computing analytically weight of %s", volume->GetName());
         weight = volume->WeightA();
         Info("Weight", "Computed weight: %f [kg]\n", weight);
         return weight;
      }            
      Info("Weight", "Estimating weight of %s with %g %% precision", fTopVolume->GetName(), 100.*precision);
      printf("    event         weight         err\n");
      printf("========================================\n");
   }
   weight = fPainter->Weight(precision, option);
   return weight;
}

//_____________________________________________________________________________
ULong_t TGeoManager::SizeOf(const TGeoNode * /*node*/, Option_t * /*option*/)
{
// computes the total size in bytes of the branch starting with node.
// The option can specify if all the branch has to be parsed or only the node
   return 0;
}

//______________________________________________________________________________
void TGeoManager::Streamer(TBuffer &R__b)
{
   // Stream an object of class TGeoManager.

   if (R__b.IsReading()) {
      TGeoManager::Class()->ReadBuffer(R__b, this);
      fIsGeomReading = kTRUE;
      CloseGeometry();
      fStreamVoxels = kFALSE;
      fIsGeomReading = kFALSE;
   } else {
      TGeoManager::Class()->WriteBuffer(R__b, this);
   }
}

//_____________________________________________________________________________
void TGeoManager::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
// Execute mouse actions on this manager.
   if (!fPainter) return;
   fPainter->ExecuteManagerEvent(this, event, px, py);
}

//______________________________________________________________________________
Int_t TGeoManager::Export(const char *filename, const char *name, Option_t *option)
{
   // Export this geometry on filename with a key=name
   // By default the geometry is saved without the voxelisation info.
   // Use option 'v" to save the voxelisation info.
   // If file extension is not .root, export as C++ code

   TString sfile(filename);
   if (!sfile.Contains(".root")) {
      Info("Export","Exporting %s %s as C++ code", GetName(), GetTitle());
      fTopVolume->SaveAs(filename);
      return 1;
   }   
   TFile *f = TFile::Open(filename,"recreate");
   if (!f || f->IsZombie()) {
      Error("Export","Cannot open file");
      return 0;
   }   
   char keyname[256];
   if (name) strcpy(keyname,name);
   if (strlen(keyname) == 0) strcpy(keyname,GetName());
   TString opt = option;
   opt.ToLower();
   if (opt.Contains("v")) {
      fStreamVoxels = kTRUE;
      Info("Export","Exporting %s %s as root file. Optimizations streamed.", GetName(), GetTitle());
   } else {
      fStreamVoxels = kFALSE;
      Info("Export","Exporting %s %s as root file. Optimizations not streamed.", GetName(), GetTitle());
   }
   Int_t nbytes = Write(keyname);
   fStreamVoxels = kFALSE;
   delete f;
   return nbytes;
}
//______________________________________________________________________________
void TGeoManager::LockGeometry()
{
// Lock current geometry so that no other geometry can be imported.
   fgLock = kTRUE;
}

//______________________________________________________________________________
void TGeoManager::UnlockGeometry()
{
// Unlock current geometry.
   fgLock = kFALSE;
}

//______________________________________________________________________________
Bool_t TGeoManager::IsLocked()
{
// Check lock state.
   return fgLock;
}   
   
//______________________________________________________________________________
TGeoManager *TGeoManager::Import(const char *filename, const char *name, Option_t * /*option*/)
{
   //static function
   //Import in memory from filename the geometry with key=name.
   //if name="" (default), the first TGeoManager object in the file is returned.
   //Note that this function deletes the current gGeoManager (if one)
   //before importing the new object.
   if (fgLock) {
      printf("WARNING: TGeoManager::Import : TGeoMananager in lock mode. NOT IMPORTING new geometry\n");
      return NULL;
   }
   printf("Info: TGeoManager::Import : Reading geometry from file\n");
   TFile *old = gFile;
   TFile *f = TFile::Open(filename);
   if (!f || f->IsZombie()) {
      if (old) old->cd();
      printf("Error: TGeoManager::Import : Cannot open file\n");
      return 0;
   }
   if (gGeoManager) delete gGeoManager;
   gGeoManager = 0;
   if (name && strlen(name) > 0) {
      gGeoManager = (TGeoManager*)f->Get(name);
   } else {
      TIter next(f->GetListOfKeys());
      TKey *key;
      while ((key = (TKey*)next())) {
         if (strcmp(key->GetClassName(),"TGeoManager") != 0) continue;
         gGeoManager = (TGeoManager*)key->ReadObj();
         break;
      }
   }
   if (old) old->cd();
   delete f;
   if (gGeoManager && (!gROOT->GetListOfBrowsables()->FindObject(gGeoManager))) gROOT->GetListOfBrowsables()->Add(gGeoManager);
   return gGeoManager;
}
//______________________________________________________________________________
Int_t *TGeoManager::GetIntBuffer(Int_t length)
{
// Get a temporary buffer of Int_t*
   if (length>fIntSize) {
      delete [] fIntBuffer;
      fIntBuffer = new Int_t[length];
      fIntSize = length;
   }
   return fIntBuffer;
}

//______________________________________________________________________________
Double_t *TGeoManager::GetDblBuffer(Int_t length)
{
// Get a temporary buffer of Double_t*
   if (length>fDblSize) {
      delete [] fDblBuffer;
      fDblBuffer = new Double_t[length];
      fDblSize = length;
   }
   return fDblBuffer;
}

//______________________________________________________________________________
Bool_t TGeoManager::GetTminTmax(Double_t &tmin, Double_t &tmax) const
{
// Get time cut for drawing tracks.
   tmin = fTmin;
   tmax = fTmax;
   return fTimeCut;
}

//______________________________________________________________________________
void TGeoManager::SetTminTmax(Double_t tmin, Double_t tmax)
{
// Set time cut interval for drawing tracks. If called with no arguments, time
// cut will be disabled.
   fTmin = tmin;
   fTmax = tmax;
   if (tmin==0 && tmax==999) fTimeCut = kFALSE;
   else fTimeCut = kTRUE;
   if (fTracks && !IsAnimatingTracks()) ModifiedPad();
}

//______________________________________________________________________________
void TGeoManager::MasterToTop(const Double_t *master, Double_t *top) const
{
// Convert coordinates from master volume frame to top.
   fCurrentMatrix->MasterToLocal(master, top);
}

//______________________________________________________________________________
void TGeoManager::TopToMaster(const Double_t *top, Double_t *master) const
{
 // Convert coordinates from top volume frame to master.
   fCurrentMatrix->LocalToMaster(top, master);
}



