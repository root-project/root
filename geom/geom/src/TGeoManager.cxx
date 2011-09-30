// @(#)root/geom:$Id$
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
// Begin_Html <a href=http://root.cern.ch/root/html/tutorials/geom/rootgeom.C.html>rootgeom.C</a> //
// End_Html                                                             //
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
// previous example $ROOTSYS/tutorials/geom/rootgeom.C ), the manager class will register
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
#include "TThread.h"

#include "TGeoVoxelFinder.h"
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
#include "TGeoBuilder.h"
#include "TVirtualGeoPainter.h"
#include "TPluginManager.h"
#include "TVirtualGeoTrack.h"
#include "TQObject.h"
#include "TMath.h"
#include "TEnv.h"

// statics and globals

TGeoManager *gGeoManager = 0;

ClassImp(TGeoManager)

Bool_t TGeoManager::fgLock         = kFALSE;
Int_t  TGeoManager::fgVerboseLevel = 1;
Int_t  TGeoManager::fgNumThreads   = 0;
TGeoManager::ThreadsMap_t TGeoManager::fgThreadId;

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
      fClosed = kFALSE;
      fLoopVolumes = kFALSE;
      fBits = 0;
      fCurrentNavigator = 0;
      fMaterials = 0;
      fHashPNE = 0;
      fArrayPNE = 0;
      fMatrices = 0;
      fNodes = 0;
      fOverlaps = 0;
      fNNodes = 0;
      fMaxVisNodes = 10000;
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
      memset(fPdgId, 0, 256*sizeof(Int_t));
      fCurrentTrack = 0;
      fCurrentVolume = 0;
      fTopVolume = 0;
      fTopNode = 0;
      fMasterVolume = 0;
      fPainter = 0;
      fActivity = kFALSE;
      fIsNodeSelectable = kFALSE;
      fVisDensity = 0.;
      fVisLevel = 3;
      fVisOption = 1;
      fExplodedView = 0;
      fNsegments = 20;
      fNLevel = 0;
      fUniqueVolumes = 0;
      fNodeIdArray = 0;
      fClippingShape = 0;
      fIntSize = fDblSize = 1000;
      fIntBuffer = 0;
      fDblBuffer = 0;
      fMatrixTransform = kFALSE;
      fMatrixReflection = kFALSE;
      fGLMatrix = 0;
      fPaintVolume = 0;
      fElementTable = 0;
      fHashVolumes = 0;
      fHashGVolumes = 0;
      fSizePNEId = 0;
      fNPNEId = 0;
      fKeyPNEId = 0;
      fValuePNEId = 0;
      fMultiThread = kFALSE;
      ClearThreadsMap();
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
   if (!gROOT->GetListOfGeometries()->FindObject(this)) gROOT->GetListOfGeometries()->Add(this);
   if (!gROOT->GetListOfBrowsables()->FindObject(this)) gROOT->GetListOfBrowsables()->Add(this);
   Init();
   gGeoIdentity = new TGeoIdentity("Identity");
   BuildDefaultMaterials();
   if (fgVerboseLevel>0) Info("TGeoManager","Geometry %s, %s created", GetName(), GetTitle());
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
   fClosed = kFALSE;
   fLoopVolumes = kFALSE;
   fBits = new UChar_t[50000]; // max 25000 nodes per volume
   fCurrentNavigator = 0;
   fHashPNE = new THashList(256,3);
   fArrayPNE = 0;
   fMaterials = new THashList(200,3);
   fMatrices = new TObjArray(256);
   fNodes = new TObjArray(30);
   fOverlaps = new TObjArray(256);
   fNNodes = 0;
   fMaxVisNodes = 10000;
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
   memset(fPdgId, 0, 256*sizeof(Int_t));
   fCurrentTrack = 0;
   fCurrentVolume = 0;
   fTopVolume = 0;
   fTopNode = 0;
   fMasterVolume = 0;
   fPainter = 0;
   fActivity = kFALSE;
   fIsNodeSelectable = kFALSE;
   fVisDensity = 0.;
   fVisLevel = 3;
   fVisOption = 1;
   fExplodedView = 0;
   fNsegments = 20;
   fNLevel = 0;
   fUniqueVolumes = new TObjArray(256);
   fNodeIdArray = 0;
   fClippingShape = 0;
   fIntSize = fDblSize = 1000;
   fIntBuffer = new Int_t[1000];
   fDblBuffer = new Double_t[1000];
   fMatrixTransform = kFALSE;
   fMatrixReflection = kFALSE;
   fGLMatrix = new TGeoHMatrix();
   fPaintVolume = 0;
   fElementTable = 0;
   fHashVolumes = 0;
   fHashGVolumes = 0;
   fSizePNEId = 0;
   fNPNEId = 0;
   fKeyPNEId = 0;
   fValuePNEId = 0;
   fMultiThread = kFALSE;
   ClearThreadsMap();
}

//_____________________________________________________________________________
TGeoManager::TGeoManager(const TGeoManager& gm) :
  TNamed(gm),
  fPhimin(gm.fPhimin),
  fPhimax(gm.fPhimax),
  fTmin(gm.fTmin),
  fTmax(gm.fTmax),
  fNNodes(gm.fNNodes),
  fParticleName(gm.fParticleName),
  fVisDensity(gm.fVisDensity),
  fExplodedView(gm.fExplodedView),
  fVisOption(gm.fVisOption),
  fVisLevel(gm.fVisLevel),
  fNsegments(gm.fNsegments),
  fNtracks(gm.fNtracks),
  fMaxVisNodes(gm.fMaxVisNodes),
  fCurrentTrack(gm.fCurrentTrack),
  fNpdg(gm.fNpdg),
  fClosed(gm.fClosed),
  fLoopVolumes(gm.fLoopVolumes),
  fStreamVoxels(gm.fStreamVoxels),
  fIsGeomReading(gm.fIsGeomReading),
  fPhiCut(gm.fPhiCut),
  fTimeCut(gm.fTimeCut),
  fDrawExtra(gm.fDrawExtra),
  fMatrixTransform(gm.fMatrixTransform),
  fMatrixReflection(gm.fMatrixReflection),
  fActivity(gm.fActivity),
  fIsNodeSelectable(gm.fIsNodeSelectable),
  fPainter(gm.fPainter),
  fMatrices(gm.fMatrices),
  fShapes(gm.fShapes),
  fVolumes(gm.fVolumes),
  fPhysicalNodes(gm.fPhysicalNodes),
  fGShapes(gm.fGShapes),
  fGVolumes(gm.fGVolumes),
  fTracks(gm.fTracks),
  fPdgNames(gm.fPdgNames),
  fMaterials(gm.fMaterials),
  fMedia(gm.fMedia),
  fNodes(gm.fNodes),
  fOverlaps(gm.fOverlaps),
  fBits(gm.fBits),
  fCurrentNavigator(gm.fCurrentNavigator),
  fCurrentVolume(gm.fCurrentVolume),
  fTopVolume(gm.fTopVolume),
  fTopNode(gm.fTopNode),
  fMasterVolume(gm.fMasterVolume),
  fGLMatrix(gm.fGLMatrix),
  fUniqueVolumes(gm.fUniqueVolumes),
  fClippingShape(gm.fClippingShape),
  fElementTable(gm.fElementTable),
  fNodeIdArray(gm.fNodeIdArray),
  fIntSize(gm.fIntSize),
  fDblSize(gm.fDblSize),
  fIntBuffer(gm.fIntBuffer),
  fNLevel(gm.fNLevel),
  fDblBuffer(gm.fDblBuffer),
  fPaintVolume(gm.fPaintVolume),
  fHashVolumes(gm.fHashVolumes),
  fHashGVolumes(gm.fHashGVolumes),
  fHashPNE(gm.fHashPNE),
  fArrayPNE(gm.fArrayPNE),
  fSizePNEId(0),
  fNPNEId(0),
  fKeyPNEId(0),
  fValuePNEId(0),
  fMultiThread(kFALSE)
{
   //copy constructor
   for(Int_t i=0; i<256; i++)
      fPdgId[i]=gm.fPdgId[i];
   ClearThreadsMap();
}

//_____________________________________________________________________________
TGeoManager& TGeoManager::operator=(const TGeoManager& gm)
{
   //assignment operator
   if(this!=&gm) {
      TNamed::operator=(gm);
      fPhimin=gm.fPhimin;
      fPhimax=gm.fPhimax;
      fTmin=gm.fTmin;
      fTmax=gm.fTmax;
      fNNodes=gm.fNNodes;
      fParticleName=gm.fParticleName;
      fVisDensity=gm.fVisDensity;
      fExplodedView=gm.fExplodedView;
      fVisOption=gm.fVisOption;
      fVisLevel=gm.fVisLevel;
      fNsegments=gm.fNsegments;
      fNtracks=gm.fNtracks;
      fMaxVisNodes=gm.fMaxVisNodes;
      fCurrentTrack=gm.fCurrentTrack;
      fNpdg=gm.fNpdg;
      for(Int_t i=0; i<256; i++)
         fPdgId[i]=gm.fPdgId[i];
      fClosed=gm.fClosed;
      fLoopVolumes=gm.fLoopVolumes;
      fStreamVoxels=gm.fStreamVoxels;
      fIsGeomReading=gm.fIsGeomReading;
      fPhiCut=gm.fPhiCut;
      fTimeCut=gm.fTimeCut;
      fDrawExtra=gm.fDrawExtra;
      fMatrixTransform=gm.fMatrixTransform;
      fMatrixReflection=gm.fMatrixReflection;
      fActivity=gm.fActivity;
      fIsNodeSelectable=gm.fIsNodeSelectable;
      fPainter=gm.fPainter;
      fMatrices=gm.fMatrices;
      fShapes=gm.fShapes;
      fVolumes=gm.fVolumes;
      fPhysicalNodes=gm.fPhysicalNodes;
      fGShapes=gm.fGShapes;
      fGVolumes=gm.fGVolumes;
      fTracks=gm.fTracks;
      fPdgNames=gm.fPdgNames;
      fMaterials=gm.fMaterials;
      fMedia=gm.fMedia;
      fNodes=gm.fNodes;
      fOverlaps=gm.fOverlaps;
      fBits=gm.fBits;
      fCurrentNavigator=gm.fCurrentNavigator;
      fCurrentVolume = gm.fCurrentVolume;
      fTopVolume=gm.fTopVolume;
      fTopNode=gm.fTopNode;
      fMasterVolume=gm.fMasterVolume;
      fGLMatrix=gm.fGLMatrix;
      fUniqueVolumes=gm.fUniqueVolumes;
      fClippingShape=gm.fClippingShape;
      fElementTable=gm.fElementTable;
      fNodeIdArray=gm.fNodeIdArray;
      fIntSize=gm.fIntSize;
      fDblSize=gm.fDblSize;
      fIntBuffer=gm.fIntBuffer;
      fNLevel=gm.fNLevel;
      fDblBuffer=gm.fDblBuffer;
      fPaintVolume=gm.fPaintVolume;
      fHashVolumes=gm.fHashVolumes;
      fHashGVolumes=gm.fHashGVolumes;
      fHashPNE=gm.fHashPNE;
      fArrayPNE=gm.fArrayPNE;
      fSizePNEId = 0;
      fNPNEId = 0;
      fKeyPNEId = 0;
      fValuePNEId = 0;
      fMultiThread = kFALSE;
      ClearThreadsMap();
   }
   return *this;
}

//_____________________________________________________________________________
TGeoManager::~TGeoManager()
{
//   Destructor
   if (gGeoManager != this) gGeoManager = this;

   if (gROOT->GetListOfFiles()) { //in case this function is called from TROOT destructor
      gROOT->GetListOfGeometries()->Remove(this);
      gROOT->GetListOfBrowsables()->Remove(this);
   }
//   TSeqCollection *brlist = gROOT->GetListOfBrowsers();
//   TIter next(brlist);
//   TBrowser *browser = 0;
//   while ((browser=(TBrowser*)next())) browser->RecursiveRemove(this);
   delete TGeoBuilder::Instance(this);
   if (fBits)  delete [] fBits;
   SafeDelete(fNodes);
   SafeDelete(fTopNode);
   if (fOverlaps) {fOverlaps->Delete(); SafeDelete(fOverlaps);}
   if (fMaterials) {fMaterials->Delete(); SafeDelete(fMaterials);}
   SafeDelete(fElementTable);
   if (fMedia) {fMedia->Delete(); SafeDelete(fMedia);}
   SafeDelete(fHashVolumes);
   SafeDelete(fHashGVolumes);
   if (fHashPNE) {fHashPNE->Delete(); SafeDelete(fHashPNE);}
   if (fArrayPNE) {delete fArrayPNE;}
   if (fVolumes) {fVolumes->Delete(); SafeDelete(fVolumes);}
   if (fShapes) {fShapes->Delete(); SafeDelete( fShapes );}
   if (fPhysicalNodes) {fPhysicalNodes->Delete(); SafeDelete( fPhysicalNodes );}
   if (fMatrices) {fMatrices->Delete(); SafeDelete( fMatrices );}
   if (fTracks) {fTracks->Delete(); SafeDelete( fTracks );}
   SafeDelete( fUniqueVolumes );
   if (fPdgNames) {fPdgNames->Delete(); SafeDelete( fPdgNames );}
//   if (fNavigators) {fNavigators->Delete(); SafeDelete( fNavigators );}
   CleanGarbage();
   SafeDelete( fPainter ); 
   delete [] fDblBuffer;
   delete [] fIntBuffer;
   SafeDelete( fGLMatrix ); 
   if (fSizePNEId) {
      delete [] fKeyPNEId;
      delete [] fValuePNEId;
   }
   ClearThreadsMap();
   gGeoIdentity = 0;
   gGeoManager = 0;
}

//_____________________________________________________________________________
Int_t TGeoManager::AddMaterial(const TGeoMaterial *material)
{
// Add a material to the list. Returns index of the material in list.
   return TGeoBuilder::Instance(this)->AddMaterial((TGeoMaterial*)material);
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
   return TGeoBuilder::Instance(this)->AddTransformation((TGeoMatrix*)matrix);
}

//_____________________________________________________________________________
Int_t TGeoManager::AddShape(const TGeoShape *shape)
{
// Add a shape to the list. Returns index of the shape in list.
   return TGeoBuilder::Instance(this)->AddShape((TGeoShape*)shape);
}

//_____________________________________________________________________________
Int_t TGeoManager::AddTrack(Int_t id, Int_t pdgcode, TObject *particle)
{
// Add a track to the list of tracks. Use this for primaries only. For secondaries,
// add them to the parent track. The method create objects that are registered
// to the analysis manager but have to be cleaned-up by the user via ClearTracks().
   Int_t index = fNtracks;
   fTracks->AddAtAndExpand(GetGeomPainter()->AddTrack(id,pdgcode,particle),fNtracks++);
   return index;
}

//_____________________________________________________________________________
Int_t TGeoManager::AddTrack(TVirtualGeoTrack *track)
{
// Add a track to the list of tracks
   Int_t index = fNtracks;
   fTracks->AddAtAndExpand(track,fNtracks++);
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
TGeoNavigator *TGeoManager::AddNavigator()
{
// Add a navigator in the list of navigators. If it is the first one make it
// current navigator.
   Long_t threadId = (fMultiThread)?TThread::SelfId():999;
   NavigatorsMap_t::const_iterator it = fNavigators.find(threadId);
   TGeoNavigatorArray *array = 0;
   if (it != fNavigators.end()) array = it->second;
   else {
      array = new TGeoNavigatorArray(this);
      fNavigators.insert(NavigatorsMap_t::value_type(threadId, array));
   }
   return array->AddNavigator();   
}   

//_____________________________________________________________________________
TGeoNavigator *TGeoManager::GetCurrentNavigator() const
{
// Returns current navigator for the calling thread.
   if (!fMultiThread) return fCurrentNavigator;
   Long_t threadId = TThread::SelfId();
   NavigatorsMap_t::const_iterator it = fNavigators.find(threadId);
   if (it == fNavigators.end()) return 0;
   TGeoNavigatorArray *array = it->second;
   return array->GetCurrentNavigator();
}

//_____________________________________________________________________________
TGeoNavigatorArray *TGeoManager::GetListOfNavigators() const
{
// Get list of navigators for the calling thread.
   Long_t threadId = (fMultiThread)?TThread::SelfId():999;
   NavigatorsMap_t::const_iterator it = fNavigators.find(threadId);
   if (it == fNavigators.end()) return 0;
   TGeoNavigatorArray *array = it->second;
   return array;
}

//_____________________________________________________________________________
Bool_t TGeoManager::SetCurrentNavigator(Int_t index)
{
// Switch to another existing navigator for the calling thread.
   Long_t threadId = (fMultiThread)?TThread::SelfId():999;
   NavigatorsMap_t::const_iterator it = fNavigators.find(threadId);
   if (it == fNavigators.end()) {
      Error("SetCurrentNavigator", "No navigator defined for thread %ld\n", threadId);
      return kFALSE;
   }   
   TGeoNavigatorArray *array = it->second;
   TGeoNavigator *nav = array->SetCurrentNavigator(index);
   if (!nav) {
      Error("SetCurrentNavigator", "Navigator %d not existing for thread %ld\n", index, threadId);
      return kFALSE;
   }
   if (!fMultiThread) fCurrentNavigator = nav;
   return kTRUE;
}

//_____________________________________________________________________________
void TGeoManager::ClearThreadsMap()
{
// Clear the current map of threads. This will be filled again by the calling
// threads via ThreadId calls.
   TThread::Lock();
   fgThreadId.clear();
   fgNumThreads = 0;
   TThread::UnLock();
}

//_____________________________________________________________________________
Int_t TGeoManager::ThreadId()
{
// Translates the current thread id to an ordinal number. This can be used to
// manage data which is pspecific for a given thread.
   Int_t tid = 0;
   Long_t selfId = TThread::SelfId();
   TGeoManager::ThreadsMapIt_t it = fgThreadId.find(selfId);
   if (it != fgThreadId.end()) return it->second;
   // Map needs to be updated.
   fgThreadId[selfId] = fgNumThreads;
   tid = fgNumThreads++;
   TThread::UnLock();
   return tid;
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
   TString browserImp(gEnv->GetValue("Browser.Name", "TRootBrowserLite"));
   TQObject::Connect(browserImp.Data(), "Checked(TObject*,Bool_t)",
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
   if(obj->IsA() == TGeoVolume::Class()) {
      TGeoVolume *vol = (TGeoVolume *) obj;
      vol->SetVisibility(vis);
   } else {
      if (obj->InheritsFrom(TGeoNode::Class())) {
         TGeoNode *node = (TGeoNode *) obj;
         node->SetVisibility(vis);
      } else return;
   }
   GetGeomPainter()->ModifiedPad(kTRUE);
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
void TGeoManager::DoBackupState()
{
// Backup the current state without affecting the cache stack.
   GetCurrentNavigator()->DoBackupState();
}

//_____________________________________________________________________________
void TGeoManager::DoRestoreState()
{
// Restore a backed-up state without affecting the cache stack.
   GetCurrentNavigator()->DoRestoreState();
}

//_____________________________________________________________________________
void TGeoManager::RegisterMatrix(const TGeoMatrix *matrix)
{
// Register a matrix to the list of matrices. It will be cleaned-up at the
// destruction TGeoManager.
   return TGeoBuilder::Instance(this)->RegisterMatrix((TGeoMatrix*)matrix);
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

   return TGeoBuilder::Instance(this)->Division(name, mother, iaxis, ndiv, start, step, numed, option);
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
   TGeoBuilder::Instance(this)->Matrix(index, theta1, phi1, theta2, phi2, theta3, phi3);
}

//_____________________________________________________________________________
TGeoMaterial *TGeoManager::Material(const char *name, Double_t a, Double_t z, Double_t dens, Int_t uid,Double_t radlen, Double_t intlen)
{
// Create material with given A, Z and density, having an unique id.
   return TGeoBuilder::Instance(this)->Material(name, a, z, dens, uid, radlen, intlen);

}

//_____________________________________________________________________________
TGeoMaterial *TGeoManager::Mixture(const char *name, Float_t *a, Float_t *z, Double_t dens,
                                   Int_t nelem, Float_t *wmat, Int_t uid)
{
// Create mixture OR COMPOUND IMAT as composed by THE BASIC nelem
// materials defined by arrays A,Z and WMAT, having an unique id.
   return TGeoBuilder::Instance(this)->Mixture(name, a, z, dens, nelem, wmat, uid);
}

//_____________________________________________________________________________
TGeoMaterial *TGeoManager::Mixture(const char *name, Double_t *a, Double_t *z, Double_t dens,
                                   Int_t nelem, Double_t *wmat, Int_t uid)
{
// Create mixture OR COMPOUND IMAT as composed by THE BASIC nelem
// materials defined by arrays A,Z and WMAT, having an unique id.
   return TGeoBuilder::Instance(this)->Mixture(name, a, z, dens, nelem, wmat, uid);
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
   return TGeoBuilder::Instance(this)->Medium(name, numed, nmat, isvol, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin);
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

   TGeoBuilder::Instance(this)->Node(name, nr, mother, x, y, z, irot, isOnly, upar, npar);
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
   TGeoBuilder::Instance(this)->Node(name, nr, mother, x, y, z, irot, isOnly, upar, npar);

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
   return TGeoBuilder::Instance(this)->Volume(name, shape, nmed, upar, npar);
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
   return TGeoBuilder::Instance(this)->Volume(name, shape, nmed, upar, npar);
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
   if (fClosed) {
      Warning("CloseGeometry", "geometry already closed");
      return;
   }
   if (!fMasterVolume) {
      Error("CloseGeometry","you MUST call SetTopVolume() first !");
      return;
   }
   if (!gROOT->GetListOfGeometries()->FindObject(this)) gROOT->GetListOfGeometries()->Add(this);
   if (!gROOT->GetListOfBrowsables()->FindObject(this)) gROOT->GetListOfBrowsables()->Add(this);
//   TSeqCollection *brlist = gROOT->GetListOfBrowsers();
//   TIter next(brlist);
//   TBrowser *browser = 0;
//   while ((browser=(TBrowser*)next())) browser->Refresh();
   TString opt(option);
   opt.ToLower();
//   Bool_t dummy = opt.Contains("d");
   Bool_t nodeid = opt.Contains("i");
   // Create a geometry navigator if not present
   if (!GetCurrentNavigator()) fCurrentNavigator = AddNavigator();
   TGeoNavigator *nav = 0;
   Int_t nnavigators = GetListOfNavigators()->GetEntriesFast();
   // Check if the geometry is streamed from file
   if (fIsGeomReading) {
      if (fgVerboseLevel>0) Info("CloseGeometry","Geometry loaded from file...");
      gGeoIdentity=(TGeoIdentity *)fMatrices->At(0);
      if (!fElementTable) fElementTable = new TGeoElementTable(200);
      if (!fTopNode) {
         if (!fMasterVolume) {
            Error("CloseGeometry", "Master volume not streamed");
            return;
         }
         SetTopVolume(fMasterVolume);
         if (fStreamVoxels && fgVerboseLevel>0) Info("CloseGeometry","Voxelization retrieved from file");
         Voxelize("ALL");
         if (nodeid) {
            for (Int_t i=0; i<nnavigators; i++) {
               nav = (TGeoNavigator*)GetListOfNavigators()->At(i);
               nav->GetCache()->BuildIdArray();
            }   
         }
      } else {
         Warning("CloseGeometry", "top node was streamed!");
         Voxelize("ALL");
         if (nodeid) {
            for (Int_t i=0; i<nnavigators; i++) {
               nav = (TGeoNavigator*)GetListOfNavigators()->At(i);
               nav->GetCache()->BuildIdArray();
            }   
         }
      }
      if (!fHashVolumes) {
         Int_t nvol = fVolumes->GetEntriesFast();
         Int_t ngvol = fGVolumes->GetEntriesFast();
         fHashVolumes = new THashList(nvol+1);
         fHashGVolumes = new THashList(ngvol+1);
         Int_t i;
         for (i=0; i<ngvol; i++) fHashGVolumes->AddLast(fGVolumes->At(i));
         for (i=0; i<nvol; i++) fHashVolumes->AddLast(fVolumes->At(i));
      }

      if (fgVerboseLevel>0) Info("CloseGeometry","%i nodes/ %i volume UID's in %s", fNNodes, fUniqueVolumes->GetEntriesFast()-1, GetTitle());
      if (fgVerboseLevel>0) Info("CloseGeometry","----------------modeler ready----------------");
      fClosed = kTRUE;
      return;
   }

   SelectTrackingMedia();
   CheckGeometry();
   if (fgVerboseLevel>0) Info("CloseGeometry","Counting nodes...");
   fNNodes = CountNodes();
   fNLevel = fMasterVolume->CountNodes(1,3)+1;
   if (fNLevel<30) fNLevel = 100;

//   BuildIdArray();
   Voxelize("ALL");
   if (fgVerboseLevel>0) Info("CloseGeometry","Building cache...");
   if (nodeid) {
      for (Int_t i=0; i<nnavigators; i++) {
         nav = (TGeoNavigator*)GetListOfNavigators()->At(i);
         nav->GetCache()->BuildIdArray();
      }   
   }
   fClosed = kTRUE;
   if (fgVerboseLevel>0) {
      Info("CloseGeometry","%i nodes/ %i volume UID's in %s", fNNodes, fUniqueVolumes->GetEntriesFast()-1, GetTitle());
      Info("CloseGeometry","----------------modeler ready----------------");
   }   
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
   GetCurrentNavigator()->CdNode(nodeid);
}

//_____________________________________________________________________________
Int_t TGeoManager::GetCurrentNodeId() const
{
// Get the unique ID of the current node.
   return GetCurrentNavigator()->GetCurrentNodeId();
}

//_____________________________________________________________________________
void TGeoManager::CdTop()
{
// Make top level node the current node. Updates the cache accordingly.
// Determine the overlapping state of current node.
   GetCurrentNavigator()->CdTop();
}

//_____________________________________________________________________________
void TGeoManager::CdUp()
{
// Go one level up in geometry. Updates cache accordingly.
// Determine the overlapping state of current node.
   GetCurrentNavigator()->CdUp();
}
//_____________________________________________________________________________
void TGeoManager::CdDown(Int_t index)
{
// Make a daughter of current node current. Can be called only with a valid
// daughter index (no check). Updates cache accordingly.
   GetCurrentNavigator()->CdDown(index);
}

//_____________________________________________________________________________
void TGeoManager::CdNext()
{
// Do a cd to the node found next by FindNextBoundary
   GetCurrentNavigator()->CdNext();
}

//_____________________________________________________________________________
Bool_t TGeoManager::cd(const char *path)
{
// Browse the tree of nodes starting from fTopNode according to pathname.
// Changes the path accordingly.
   return GetCurrentNavigator()->cd(path);
}

//_____________________________________________________________________________
Bool_t TGeoManager::CheckPath(const char *path) const
{
// Check if a geometry path is valid without changing the state of the current navigator.
   return GetCurrentNavigator()->CheckPath(path);
}

//_____________________________________________________________________________
void TGeoManager::ConvertReflections()
{
// Convert all reflections in geometry to normal rotations + reflected shapes.
   if (!fTopNode) return;
   if (fgVerboseLevel>0) Info("ConvertReflections", "Converting reflections in: %s - %s ...", GetName(), GetTitle());
   TGeoIterator next(fTopVolume);
   TGeoNode *node;
   TGeoNodeMatrix *nodematrix;
   TGeoMatrix *matrix, *mclone;
   TGeoVolume *reflected;
   while ((node=next())) {
      matrix = node->GetMatrix();
      if (matrix->IsReflection()) {
//         printf("%s before\n", node->GetName());
//         matrix->Print();
         mclone = new TGeoCombiTrans(*matrix);
         mclone->RegisterYourself();
         // Reflect just the rotation component
         mclone->ReflectZ(kFALSE, kTRUE);
         nodematrix = (TGeoNodeMatrix*)node;
         nodematrix->SetMatrix(mclone);
//         printf("%s after\n", node->GetName());
//         node->GetMatrix()->Print();
         reflected = node->GetVolume()->MakeReflectedVolume();
         node->SetVolume(reflected);
      }
   }
   if (fgVerboseLevel>0) Info("ConvertReflections", "Done");
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
   TString fname;
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
         fname = TString::Format("anim%04d.gif", i);
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
   GetCurrentNavigator()->GetBranchNames(names);
}
//_____________________________________________________________________________
const char *TGeoManager::GetPdgName(Int_t pdg) const
{
// Get name for given pdg code;
   static char defaultname[5] = { "XXX" };
   if (!fPdgNames || !pdg) return defaultname;
   for (Int_t i=0; i<fNpdg; i++) {
      if (fPdgId[i]==pdg) return fPdgNames->At(i)->GetName();
   }
   return defaultname;
}

//_____________________________________________________________________________
void TGeoManager::SetPdgName(Int_t pdg, const char *name)
{
// Set a name for a particle having a given pdg.
   if (!pdg) return;
   if (!fPdgNames) {
      fPdgNames = new TObjArray(256);
   }
   if (!strcmp(name, GetPdgName(pdg))) return;
   // store pdg name
   if (fNpdg>255) {
      Warning("SetPdgName", "No more than 256 different pdg codes allowed");
      return;
   }
   fPdgId[fNpdg] = pdg;
   TNamed *pdgname = new TNamed(name, "");
   fPdgNames->AddAtAndExpand(pdgname, fNpdg++);
}

//_____________________________________________________________________________
void TGeoManager::GetBranchNumbers(Int_t *copyNumbers, Int_t *volumeNumbers) const
{
// Fill node copy numbers of current branch into an array.
   GetCurrentNavigator()->GetBranchNumbers(copyNumbers, volumeNumbers);
}

//_____________________________________________________________________________
void TGeoManager::GetBranchOnlys(Int_t *isonly) const
{
// Fill node copy numbers of current branch into an array.
   GetCurrentNavigator()->GetBranchOnlys(isonly);
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
   if (!GetCurrentNavigator()) return NULL;
   return GetCurrentNavigator()->GetHMatrix();
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

   return GetCurrentNavigator()->GetVirtualLevel();
}
//_____________________________________________________________________________
TVirtualGeoTrack *TGeoManager::FindTrackWithId(Int_t id) const
{
// Search the track hierarchy to find the track with the
// given id
//
// if 'primsFirst' is true, then:
// first tries TGeoManager::GetTrackOfId, then does a
// recursive search if that fails. this would be faster
// if the track is somehow known to be a primary
   TVirtualGeoTrack* trk = 0;
   trk = GetTrackOfId(id);
   if (trk) return trk;
   // need recursive search
   TIter next(fTracks);
   TVirtualGeoTrack* prim;
   while ((prim = (TVirtualGeoTrack*)next())) {
      trk = prim->FindTrackWithId(id);
      if (trk) return trk;
   }
   return NULL;
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
   return GetCurrentNavigator()->GotoSafeLevel();
}

//_____________________________________________________________________________
Int_t TGeoManager::GetSafeLevel() const
{
// Go upwards the tree until a non-overlaping node
   return GetCurrentNavigator()->GetSafeLevel();
}

//_____________________________________________________________________________
void TGeoManager::DefaultColors()
{
// Set default volume colors according to A of material

   const Int_t nmax = 110;
   Int_t col[nmax];
   for (Int_t i=0;i<nmax;i++) col[i] = kGray;

   //here we should create a new TColor with the same rgb as in the default
   //ROOT colors used below
   col[ 3] = kYellow-10;
   col[ 4] = col[ 5] = kGreen-10;
   col[ 6] = col[ 7] = kBlue-7;
   col[ 8] = col[ 9] = kMagenta-3;
   col[10] = col[11] = kRed-10;
   col[12] = kGray+1;
   col[13] = kBlue-10;
   col[14] = kOrange+7;
   col[16] = kYellow+1;
   col[20] = kYellow-10;
   col[24] = col[25] = col[26] = kBlue-8;
   col[29] = kOrange+9;
   col[79] = kOrange-2;

   TGeoVolume *vol;
   TIter next(fVolumes);
   while ((vol=(TGeoVolume*)next())) {
      TGeoMedium *med = vol->GetMedium();
      if (!med) continue;
      TGeoMaterial *mat = med->GetMaterial();
      Int_t matZ = (Int_t)mat->GetZ();
      vol->SetLineColor(col[matZ]);
      if (mat->GetDensity()<0.1) vol->SetTransparency(60);
   }
}

//_____________________________________________________________________________
Double_t TGeoManager::Safety(Bool_t inside)
{
// Compute safe distance from the current point. This represent the distance
// from POINT to the closest boundary.

   return GetCurrentNavigator()->Safety(inside);
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
   if (maxnodes>0 && fgVerboseLevel>0)
      Info("SetMaxVisNodes","Automatic visible depth for %d visible nodes", maxnodes);
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
void TGeoManager::SetCheckedNode(TGeoNode *node) {
// Assign a given node to be checked for ovelaps. Any other overlaps will be ignored.
   GetGeomPainter()->SetCheckedNode(node);
}   
//_____________________________________________________________________________
void TGeoManager::SetNmeshPoints(Int_t npoints)
{
// Set the number of points to be generated on the shape outline when checking
// for overlaps.
   GetGeomPainter()->SetNmeshPoints(npoints);
}
   
//_____________________________________________________________________________
void TGeoManager::SetVisOption(Int_t option) {
// set drawing mode :
// option=0 (default) all nodes drawn down to vislevel
// option=1           leaves and nodes at vislevel drawn
// option=2           path is drawn
// option=4           visibility changed
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
      if (fgVerboseLevel>0)
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
   TString fname = filename;
   if (fname.IsNull()) fname = "tgeovox.C";
   out.open(fname, ios::out);
   if (!out.good()) {
      Error("OptimizeVoxels", "cannot open file");
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
      out<<"   vol = gGeoManager->GetVolume(\""<<vol->GetName()<<"\");"<<endl;
      cyltype = vol->OptimizeVoxels();
      if (cyltype) {
         out<<"   vol->SetCylVoxels();"<<endl;
      } else {
         out<<"   vol->SetCylVoxels(kFALSE);"<<endl;
      }
   }
   out << "}" << endl;
   out.close();
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
      // Take LAST operator at lowest level (revision 28/07/08)
      if (level<=levmin) {
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
   TString fname(filename);
   if (fname.IsNull()) fname = "tgeoatt.C";
   out.open(fname, ios::out);
   if (!out.good()) {
      Error("SaveAttributes", "cannot open file");
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
   out << "   TGeoVolume *top = gGeoManager->GetVolume(\""<<fTopVolume->GetName()<<"\");"<<endl;
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
}
//_____________________________________________________________________________
TGeoNode *TGeoManager::SearchNode(Bool_t downwards, const TGeoNode *skipnode)
{
// Returns the deepest node containing fPoint, which must be set a priori.

   return GetCurrentNavigator()->SearchNode(downwards, skipnode);
}

//_____________________________________________________________________________
TGeoNode *TGeoManager::CrossBoundaryAndLocate(Bool_t downwards, TGeoNode *skipnode)
{
// Cross next boundary and locate within current node
// The current point must be on the boundary of fCurrentNode.
   return GetCurrentNavigator()->CrossBoundaryAndLocate(downwards, skipnode);
}

//_____________________________________________________________________________
TGeoNode *TGeoManager::FindNextBoundaryAndStep(Double_t stepmax, Bool_t compsafe)
{
// Compute distance to next boundary within STEPMAX. If no boundary is found,
// propagate current point along current direction with fStep=STEPMAX. Otherwise
// propagate with fStep=SNEXT (distance to boundary) and locate/return the next
// node.

   return GetCurrentNavigator()->FindNextBoundaryAndStep(stepmax, compsafe);
}

//_____________________________________________________________________________
TGeoNode *TGeoManager::FindNextBoundary(Double_t stepmax, const char *path, Bool_t frombdr)
{
// Find distance to next boundary and store it in fStep. Returns node to which this
// boundary belongs. If PATH is specified, compute only distance to the node to which
// PATH points. If STEPMAX is specified, compute distance only in case fSafety is smaller
// than this value. STEPMAX represent the step to be made imposed by other reasons than
// geometry (usually physics processes). Therefore in this case this method provides the
// answer to the question : "Is STEPMAX a safe step ?" returning a NULL node and filling
// fStep with a big number.
// In case frombdr=kTRUE, the isotropic safety is set to zero.
// Note : safety distance for the current point is computed ONLY in case STEPMAX is
//        specified, otherwise users have to call explicitly TGeoManager::Safety() if
//        they want this computed for the current point.

   // convert current point and direction to local reference
   return GetCurrentNavigator()->FindNextBoundary(stepmax,path, frombdr);
}

//_____________________________________________________________________________
TGeoNode *TGeoManager::FindNextDaughterBoundary(Double_t *point, Double_t *dir, Int_t &idaughter, Bool_t compmatrix)
{
// Computes as fStep the distance to next daughter of the current volume.
// The point and direction must be converted in the coordinate system of the current volume.
// The proposed step limit is fStep.

   return GetCurrentNavigator()->FindNextDaughterBoundary(point, dir, idaughter, compmatrix);
}

//_____________________________________________________________________________
void TGeoManager::ResetState()
{
// Reset current state flags.
   GetCurrentNavigator()->ResetState();
}

//_____________________________________________________________________________
TGeoNode *TGeoManager::FindNode(Bool_t safe_start)
{
// Returns deepest node containing current point.
   return GetCurrentNavigator()->FindNode(safe_start);
}

//_____________________________________________________________________________
TGeoNode *TGeoManager::FindNode(Double_t x, Double_t y, Double_t z)
{
// Returns deepest node containing current point.
   return GetCurrentNavigator()->FindNode(x, y, z);
}

//_____________________________________________________________________________
Double_t *TGeoManager::FindNormalFast()
{
// Computes fast normal to next crossed boundary, assuming that the current point
// is close enough to the boundary. Works only after calling FindNextBoundary.
   return GetCurrentNavigator()->FindNormalFast();
}

//_____________________________________________________________________________
Double_t *TGeoManager::FindNormal(Bool_t forward)
{
// Computes normal vector to the next surface that will be or was already
// crossed when propagating on a straight line from a given point/direction.
// Returns the normal vector cosines in the MASTER coordinate system. The dot
// product of the normal and the current direction is positive defined.
   return GetCurrentNavigator()->FindNormal(forward);
}

//_____________________________________________________________________________
Bool_t TGeoManager::IsSameLocation(Double_t x, Double_t y, Double_t z, Bool_t change)
{
// Checks if point (x,y,z) is still in the current node.
   return GetCurrentNavigator()->IsSameLocation(x,y,z,change);
}

//_____________________________________________________________________________
Bool_t TGeoManager::IsSamePoint(Double_t x, Double_t y, Double_t z) const
{
// Check if a new point with given coordinates is the same as the last located one.
   return GetCurrentNavigator()->IsSamePoint(x,y,z);
}

//_____________________________________________________________________________
Bool_t TGeoManager::IsInPhiRange() const
{
// True if current node is in phi range
   if (!fPhiCut) return kTRUE;
   const Double_t *origin;
   if (!GetCurrentNavigator() || !GetCurrentNavigator()->GetCurrentNode()) return kFALSE;
   origin = ((TGeoBBox*)GetCurrentNavigator()->GetCurrentVolume()->GetShape())->GetOrigin();
   Double_t point[3];
   LocalToMaster(origin, &point[0]);
   Double_t phi = TMath::ATan2(point[1], point[0])*TMath::RadToDeg();
   if (phi<0) phi+=360.;
   if ((phi>=fPhimin) && (phi<=fPhimax)) return kFALSE;
   return kTRUE;
}

//_____________________________________________________________________________
TGeoNode *TGeoManager::InitTrack(const Double_t *point, const Double_t *dir)
{
// Initialize current point and current direction vector (normalized)
// in MARS. Return corresponding node.
   return GetCurrentNavigator()->InitTrack(point, dir);
}

//_____________________________________________________________________________
TGeoNode *TGeoManager::InitTrack(Double_t x, Double_t y, Double_t z, Double_t nx, Double_t ny, Double_t nz)
{
// Initialize current point and current direction vector (normalized)
// in MARS. Return corresponding node.
   return GetCurrentNavigator()->InitTrack(x,y,z,nx,ny,nz);
}

//_____________________________________________________________________________
void TGeoManager::InspectState() const
{
// Inspects path and all flags for the current state.
   GetCurrentNavigator()->InspectState();
}

//_____________________________________________________________________________
const char *TGeoManager::GetPath() const
{
// Get path to the current node in the form /node0/node1/...
   return GetCurrentNavigator()->GetPath();
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
   if (fgVerboseLevel>0) Info("GetByteCount","Total size of logical tree : %i bytes", count);
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
      for (i=0; i<ngvol; i++) fHashGVolumes->AddLast(fGVolumes->At(i));
      for (i=0; i<nvol; i++) fHashVolumes->AddLast(fVolumes->At(i));
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
   if (!fStreamVoxels && fgVerboseLevel>0) Info("Voxelize","Voxelizing...");
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
TGeoVolume *TGeoManager::MakeArb8(const char *name, TGeoMedium *medium,
                                  Double_t dz, Double_t *vertices)
{
// Make an TGeoArb8 volume.
   return TGeoBuilder::Instance(this)->MakeArb8(name, medium, dz, vertices);
}

//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakeBox(const char *name, TGeoMedium *medium,
                                    Double_t dx, Double_t dy, Double_t dz)
{
// Make in one step a volume pointing to a box shape with given medium.
   return TGeoBuilder::Instance(this)->MakeBox(name, medium, dx, dy, dz);
}

//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakePara(const char *name, TGeoMedium *medium,
                                    Double_t dx, Double_t dy, Double_t dz,
                                    Double_t alpha, Double_t theta, Double_t phi)
{
// Make in one step a volume pointing to a paralelipiped shape with given medium.
   return TGeoBuilder::Instance(this)->MakePara(name, medium, dx, dy, dz, alpha, theta, phi);
}

//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakeSphere(const char *name, TGeoMedium *medium,
                                    Double_t rmin, Double_t rmax, Double_t themin, Double_t themax,
                                    Double_t phimin, Double_t phimax)
{
// Make in one step a volume pointing to a sphere shape with given medium
   return TGeoBuilder::Instance(this)->MakeSphere(name, medium, rmin, rmax, themin, themax, phimin, phimax);
}

//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakeTorus(const char *name, TGeoMedium *medium, Double_t r,
                                   Double_t rmin, Double_t rmax, Double_t phi1, Double_t dphi)
{
// Make in one step a volume pointing to a torus shape with given medium.
   return TGeoBuilder::Instance(this)->MakeTorus(name, medium, r, rmin, rmax, phi1, dphi);
}

//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakeTube(const char *name, TGeoMedium *medium,
                                     Double_t rmin, Double_t rmax, Double_t dz)
{
// Make in one step a volume pointing to a tube shape with given medium.
   return TGeoBuilder::Instance(this)->MakeTube(name, medium, rmin, rmax, dz);
}

//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakeTubs(const char *name, TGeoMedium *medium,
                                     Double_t rmin, Double_t rmax, Double_t dz,
                                     Double_t phi1, Double_t phi2)
{
// Make in one step a volume pointing to a tube segment shape with given medium.
   return TGeoBuilder::Instance(this)->MakeTubs(name, medium, rmin, rmax, dz, phi1, phi2);
}

//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakeEltu(const char *name, TGeoMedium *medium,
                                     Double_t a, Double_t b, Double_t dz)
{
// Make in one step a volume pointing to a tube shape with given medium
   return TGeoBuilder::Instance(this)->MakeEltu(name, medium, a, b, dz);
}

//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakeHype(const char *name, TGeoMedium *medium,
                                        Double_t rin, Double_t stin, Double_t rout, Double_t stout, Double_t dz)
{
// Make in one step a volume pointing to a tube shape with given medium
   return TGeoBuilder::Instance(this)->MakeHype(name, medium, rin, stin, rout, stout, dz);
}

//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakeParaboloid(const char *name, TGeoMedium *medium,
                                        Double_t rlo, Double_t rhi, Double_t dz)
{
// Make in one step a volume pointing to a tube shape with given medium
   return TGeoBuilder::Instance(this)->MakeParaboloid(name, medium, rlo, rhi, dz);
}

//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakeCtub(const char *name, TGeoMedium *medium,
                                     Double_t rmin, Double_t rmax, Double_t dz, Double_t phi1, Double_t phi2,
                                     Double_t lx, Double_t ly, Double_t lz, Double_t tx, Double_t ty, Double_t tz)
{
// Make in one step a volume pointing to a tube segment shape with given medium
   return TGeoBuilder::Instance(this)->MakeCtub(name, medium, rmin, rmax, dz, phi1, phi2, lx, ly, lz, tx, ty, tz);
}

//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakeCone(const char *name, TGeoMedium *medium,
                                     Double_t dz, Double_t rmin1, Double_t rmax1,
                                     Double_t rmin2, Double_t rmax2)
{
// Make in one step a volume pointing to a cone shape with given medium.
   return TGeoBuilder::Instance(this)->MakeCone(name, medium, dz, rmin1, rmax1, rmin2, rmax2);
}

//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakeCons(const char *name, TGeoMedium *medium,
                                     Double_t dz, Double_t rmin1, Double_t rmax1,
                                     Double_t rmin2, Double_t rmax2,
                                     Double_t phi1, Double_t phi2)
{
// Make in one step a volume pointing to a cone segment shape with given medium
   return TGeoBuilder::Instance(this)->MakeCons(name, medium, dz, rmin1, rmax1, rmin2, rmax2, phi1, phi2);
}

//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakePcon(const char *name, TGeoMedium *medium,
                                     Double_t phi, Double_t dphi, Int_t nz)
{
// Make in one step a volume pointing to a polycone shape with given medium.
   return TGeoBuilder::Instance(this)->MakePcon(name, medium, phi, dphi, nz);
}

//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakePgon(const char *name, TGeoMedium *medium,
                                     Double_t phi, Double_t dphi, Int_t nedges, Int_t nz)
{
// Make in one step a volume pointing to a polygone shape with given medium.
   return TGeoBuilder::Instance(this)->MakePgon(name, medium, phi, dphi, nedges, nz);
}

//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakeTrd1(const char *name, TGeoMedium *medium,
                                  Double_t dx1, Double_t dx2, Double_t dy, Double_t dz)
{
// Make in one step a volume pointing to a TGeoTrd1 shape with given medium.
   return TGeoBuilder::Instance(this)->MakeTrd1(name, medium, dx1, dx2, dy, dz);
}

//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakeTrd2(const char *name, TGeoMedium *medium,
                                  Double_t dx1, Double_t dx2, Double_t dy1, Double_t dy2,
                                  Double_t dz)
{
// Make in one step a volume pointing to a TGeoTrd2 shape with given medium.
   return TGeoBuilder::Instance(this)->MakeTrd2(name, medium, dx1, dx2, dy1, dy2, dz);
}

//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakeTrap(const char *name, TGeoMedium *medium,
                                  Double_t dz, Double_t theta, Double_t phi, Double_t h1,
                                  Double_t bl1, Double_t tl1, Double_t alpha1, Double_t h2, Double_t bl2,
                                  Double_t tl2, Double_t alpha2)
{
// Make in one step a volume pointing to a trapezoid shape with given medium.
   return TGeoBuilder::Instance(this)->MakeTrap(name, medium, dz, theta, phi, h1, bl1, tl1, alpha1, h2, bl2, tl2, alpha2);
}

//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakeGtra(const char *name, TGeoMedium *medium,
                                  Double_t dz, Double_t theta, Double_t phi, Double_t twist, Double_t h1,
                                  Double_t bl1, Double_t tl1, Double_t alpha1, Double_t h2, Double_t bl2,
                                  Double_t tl2, Double_t alpha2)
{
// Make in one step a volume pointing to a twisted trapezoid shape with given medium.
   return TGeoBuilder::Instance(this)->MakeGtra(name, medium, dz, theta, phi, twist, h1, bl1, tl1, alpha1, h2, bl2, tl2, alpha2);
}

//_____________________________________________________________________________
TGeoVolume *TGeoManager::MakeXtru(const char *name, TGeoMedium *medium, Int_t nz)
{
// Make a TGeoXtru-shaped volume with nz planes
   return TGeoBuilder::Instance(this)->MakeXtru(name, medium, nz);
}

//_____________________________________________________________________________
TGeoPNEntry *TGeoManager::SetAlignableEntry(const char *unique_name, const char *path,
                                            Int_t uid)
{
// Creates an aligneable object with unique name corresponding to a path
// and adds it to the list of alignables. An optional unique ID can be
// provided, in which case PN entries can be searched fast by uid.
   if (!CheckPath(path)) return NULL;
   if (!fHashPNE) fHashPNE = new THashList(256,3);
   if (!fArrayPNE) fArrayPNE = new TObjArray(256);
   TGeoPNEntry *entry = GetAlignableEntry(unique_name);
   if (entry) {
      Error("SetAlignableEntry", "An alignable object with name %s already existing. NOT ADDED !", unique_name);
      return 0;
   }
   entry = new TGeoPNEntry(unique_name, path);
   Int_t ientry = fHashPNE->GetSize();
   fHashPNE->Add(entry);
   fArrayPNE->AddAtAndExpand(entry, ientry);
   if (uid>=0) {
      Bool_t added = InsertPNEId(uid, ientry);
      if (!added) Error("SetAlignableEntry", "A PN entry: has already uid=%i", uid);
   }
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
   if (!fArrayPNE && !InitArrayPNE()) return 0;
   return (TGeoPNEntry*)fArrayPNE->At(index);
}

//_____________________________________________________________________________
TGeoPNEntry *TGeoManager::GetAlignableEntryByUID(Int_t uid) const
{
// Retreives an existing alignable object having a preset UID.
   if (!fNPNEId || (!fArrayPNE && !InitArrayPNE())) return NULL;
   Int_t index = TMath::BinarySearch(fNPNEId, fKeyPNEId, uid);
   if (index<0 || fKeyPNEId[index]!=uid) return NULL;
   return (TGeoPNEntry*)fArrayPNE->At(fValuePNEId[index]);
}

//_____________________________________________________________________________
Int_t TGeoManager::GetNAlignable(Bool_t with_uid) const
{
// Retreives number of PN entries with or without UID.
   if (!fHashPNE) return 0;
   if (with_uid) return fNPNEId;
   return fHashPNE->GetSize();
}

//_____________________________________________________________________________
Bool_t TGeoManager::InsertPNEId(Int_t uid, Int_t ientry)
{
// Insert a PN entry in the sorted array of indexes.
   if (!fSizePNEId) {
      // Create the arrays.
      fSizePNEId = 128;
      fKeyPNEId = new Int_t[fSizePNEId];
      memset(fKeyPNEId, 0, fSizePNEId*sizeof(Int_t));
      fValuePNEId = new Int_t[fSizePNEId];
      memset(fValuePNEId, 0, fSizePNEId*sizeof(Int_t));
      fKeyPNEId[fNPNEId] = uid;
      fValuePNEId[fNPNEId++] = ientry;
      return kTRUE;
   }
   // Search id in the existing array and return false if it already exists.
   Int_t index = TMath::BinarySearch(fNPNEId, fKeyPNEId, uid);
   if (index>0 && fKeyPNEId[index]==uid) return kFALSE;
   // Resize the arrays and insert the value
   Bool_t resize = (fNPNEId==fSizePNEId)?kTRUE:kFALSE;
   if (resize) {
      // Double the size of the array
      fSizePNEId *= 2;
      // Create new arrays of keys and values
      Int_t *keys = new Int_t[fSizePNEId];
      memset(keys, 0, fSizePNEId*sizeof(Int_t));
      Int_t *values = new Int_t[fSizePNEId];
      memset(values, 0, fSizePNEId*sizeof(Int_t));
      // Copy all keys<uid in the new keys array (0 to index)
      memcpy(keys,   fKeyPNEId,   (index+1)*sizeof(Int_t));
      memcpy(values, fValuePNEId, (index+1)*sizeof(Int_t));
      // Insert current key at index+1
      keys[index+1]   = uid;
      values[index+1] = ientry;
      // Copy all remaining keys from the old to new array
      memcpy(&keys[index+2],   &fKeyPNEId[index+1],   (fNPNEId-index-1)*sizeof(Int_t));
      memcpy(&values[index+2], &fValuePNEId[index+1], (fNPNEId-index-1)*sizeof(Int_t));
      delete [] fKeyPNEId;
      fKeyPNEId = keys;
      delete [] fValuePNEId;
      fValuePNEId = values;
      fNPNEId++;
      return kTRUE;
   }
   // Insert the value in the existing arrays
   Int_t i;
   for (i=fNPNEId-1; i>index; i--) {
      fKeyPNEId[i+1] = fKeyPNEId[i];
      fValuePNEId[i+1] = fValuePNEId[i];
   }
   fKeyPNEId[index+1] = uid;
   fValuePNEId[index+1] = ientry;
   fNPNEId++;
   return kTRUE;
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
      if (!CheckPath(path)) {
         Error("MakePhysicalNode", "path: %s not valid", path);
         return NULL;
      }
      node = new TGeoPhysicalNode(path);
   } else {
      node = new TGeoPhysicalNode(GetPath());
   }
   fPhysicalNodes->Add(node);
   return node;
}

//_____________________________________________________________________________
void TGeoManager::RefreshPhysicalNodes(Bool_t lock)
{
// Refresh physical nodes to reflect the actual geometry paths after alignment
// was applied. Optionally locks physical nodes (default).
   TIter next(gGeoManager->GetListOfPhysicalNodes());
   TGeoPhysicalNode *pn;
   while ((pn=(TGeoPhysicalNode*)next())) pn->Refresh();
   if (lock) LockGeometry();
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
   return TGeoBuilder::Instance(this)->MakeVolumeAssembly(name);
}

//_____________________________________________________________________________
TGeoVolumeMulti *TGeoManager::MakeVolumeMulti(const char *name, TGeoMedium *medium)
{
// Make a TGeoVolumeMulti handling a list of volumes.
   return TGeoBuilder::Instance(this)->MakeVolumeMulti(name, medium);
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
// Now just a shortcut for GetElementTable.
   GetElementTable();
}

//_____________________________________________________________________________
TGeoElementTable *TGeoManager::GetElementTable()
{
// Returns material table. Creates it if not existing.
   if (!fElementTable) fElementTable = new TGeoElementTable(200);
   return fElementTable;
}

//_____________________________________________________________________________
TGeoNode *TGeoManager::Step(Bool_t is_geom, Bool_t cross)
{
// Make a rectiliniar step of length fStep from current point (fPoint) on current
// direction (fDirection). If the step is imposed by geometry, is_geom flag
// must be true (default). The cross flag specifies if the boundary should be
// crossed in case of a geometry step (default true). Returns new node after step.
// Set also on boundary condition.
   return GetCurrentNavigator()->Step(is_geom, cross);
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
      if (fgVerboseLevel>0) Info("SetTopVolume","Top volume is %s. Master volume is %s", fTopVolume->GetName(),
           fMasterVolume->GetName());
   }
//   fMasterVolume->FindMatrixOfDaughterVolume(vol);
//   fCurrentMatrix->Print();
   fTopNode = new TGeoNodeMatrix(vol, gGeoIdentity);
   fTopNode->SetName(TString::Format("%s_1",vol->GetName()));
   fTopNode->SetNumber(1);
   fTopNode->SetTitle("Top logical node");
   fNodes->AddAt(fTopNode, 0);
   if (!GetCurrentNavigator()) fCurrentNavigator = AddNavigator();
   Int_t nnavigators = GetListOfNavigators()->GetEntriesFast();
   for (Int_t i=0; i<nnavigators; i++) {
      TGeoNavigator *nav = (TGeoNavigator*)GetListOfNavigators()->At(i);
      nav->ResetAll();
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
void TGeoManager::CheckBoundaryErrors(Int_t ntracks, Double_t radius)
{
// Check pushes and pulls needed to cross the next boundary with respect to the
// position given by FindNextBoundary. If radius is not mentioned the full bounding
// box will be sampled.
   GetGeomPainter()->CheckBoundaryErrors(ntracks, radius);
}   

//_____________________________________________________________________________
void TGeoManager::CheckBoundaryReference(Int_t icheck)
{
// Check the boundary errors reference file created by CheckBoundaryErrors method.
// The shape for which the crossing failed is drawn with the starting point in red
// and the extrapolated point to boundary (+/- failing push/pull) in yellow.
   GetGeomPainter()->CheckBoundaryReference(icheck);
}   

//_____________________________________________________________________________
void TGeoManager::CheckPoint(Double_t x, Double_t y, Double_t z, Option_t *option)
{
// Classify a given point. See TGeoChecker::CheckPoint().
   GetGeomPainter()->CheckPoint(x,y,z,option);
}

//_____________________________________________________________________________
void TGeoManager::CheckShape(TGeoShape *shape, Int_t testNo, Int_t nsamples, Option_t *option)
{
// Test for shape navigation methods. Summary for test numbers:
//  1: DistFromInside/Outside. Sample points inside the shape. Generate 
//    directions randomly in cos(theta). Compute DistFromInside and move the 
//    point with bigger distance. Compute DistFromOutside back from new point.
//    Plot d-(d1+d2)
//
   GetGeomPainter()->CheckShape(shape, testNo, nsamples, option);
}
    
//_____________________________________________________________________________
void TGeoManager::CheckGeometryFull(Int_t ntracks, Double_t vx, Double_t vy, Double_t vz, Option_t *option)
{
// Geometry checking.
// - if option contains 'o': Optional overlap checkings (by sampling and by mesh).
// - if option contains 'b': Optional boundary crossing check + timing per volume.
//
// STAGE 1: extensive overlap checking by sampling per volume. Stdout need to be
//  checked by user to get report, then TGeoVolume::CheckOverlaps(0.01, "s") can
//  be called for the suspicious volumes.
// STAGE2 : normal overlap checking using the shapes mesh - fills the list of
//  overlaps.
// STAGE3 : shooting NRAYS rays from VERTEX and counting the total number of
//  crossings per volume (rays propagated from boundary to boundary until
//  geometry exit). Timing computed and results stored in a histo.
// STAGE4 : shooting 1 mil. random rays inside EACH volume and calling
//  FindNextBoundary() + Safety() for each call. The timing is normalized by the
//  number of crossings computed at stage 2 and presented as percentage.
//  One can get a picture on which are the most "burned" volumes during
//  transportation from geometry point of view. Another plot of the timing per
//  volume vs. number of daughters is produced.
   TString opt(option);
   opt.ToLower();
   if (!opt.Length()) {
      Error("CheckGeometryFull","The option string must contain a letter. See method documentation.");
      return;
   }
   Bool_t checkoverlaps  = opt.Contains("o");
   Bool_t checkcrossings = opt.Contains("b");
   Double_t vertex[3];
   vertex[0] = vx;
   vertex[1] = vy;
   vertex[2] = vz;
   GetGeomPainter()->CheckGeometryFull(checkoverlaps,checkcrossings,ntracks,vertex);
}

//_____________________________________________________________________________
void TGeoManager::CheckGeometry(Option_t * /*option*/)
{
// Instanciate a TGeoChecker object and investigates the geometry according to
// option. Not implemented yet.
   // check shapes first
   if (fgVerboseLevel>0) Info("CheckGeometry","Fixing runtime shapes...");
   TIter next(fShapes);
   TIter nextv(fVolumes);
   TGeoShape *shape;
   TGeoVolume *vol;
   Bool_t has_runtime = kFALSE;
   while ((shape = (TGeoShape*)next())) {
      if (shape->IsRunTimeShape()) {
         has_runtime = kTRUE;
      }
      if (shape->TestShapeBit(TGeoShape::kGeoPcon) || shape->TestShapeBit(TGeoShape::kGeoArb8))
         if (!shape->TestShapeBit(TGeoShape::kGeoClosedShape)) shape->ComputeBBox();
   }
   if (has_runtime) fTopNode->CheckShapes();
   else if (fgVerboseLevel>0) Info("CheckGeometry","...Nothing to fix");
   // Compute bounding  box for assemblies
   while ((vol = (TGeoVolume*)nextv())) {
      if (vol->IsAssembly()) vol->GetShape()->ComputeBBox();
   }   
}

//_____________________________________________________________________________
void TGeoManager::CheckOverlaps(Double_t ovlp, Option_t * option)
{
// Check all geometry for illegal overlaps within a limit OVLP.
   if (!fTopNode) {
      Error("CheckOverlaps","Top node not set");
      return;
   }
   fTopNode->CheckOverlaps(ovlp,option);
}

//_____________________________________________________________________________
void TGeoManager::PrintOverlaps() const
{
// Prints the current list of overlaps.
   if (!fOverlaps) return;
   Int_t novlp = fOverlaps->GetEntriesFast();
   if (!novlp) return;
   TGeoManager *geom = (TGeoManager*)this;
   geom->GetGeomPainter()->PrintOverlaps();
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
         if (fgVerboseLevel>0) Info("Weight", "Computing analytically weight of %s", volume->GetName());
         weight = volume->WeightA();
         if (fgVerboseLevel>0) Info("Weight", "Computed weight: %f [kg]\n", weight);
         return weight;
      }
      if (fgVerboseLevel>0) {
         Info("Weight", "Estimating weight of %s with %g %% precision", fTopVolume->GetName(), 100.*precision);
         printf("    event         weight         err\n");
         printf("========================================\n");
      }   
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
      R__b.ReadClassBuffer(TGeoManager::Class(), this);
      fIsGeomReading = kTRUE;
      CloseGeometry();
      fStreamVoxels = kFALSE;
      fIsGeomReading = kFALSE;
   } else {
      R__b.WriteClassBuffer(TGeoManager::Class(), this);
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
   // Export this geometry to a file
   //
   // -Case 1: root file or root/xml file
   //  if filename end with ".root". The key will be named name
   //  By default the geometry is saved without the voxelisation info.
   //  Use option 'v" to save the voxelisation info.
   //  if filename end with ".xml" a root/xml file is produced.
   //
   // -Case 2: C++ script
   //  if filename end with ".C"
   //
   // -Case 3: gdml file
   //  if filename end with ".gdml"
   //  NOTE that to use this option, the PYTHONPATH must be defined like
   //      export PYTHONPATH=$ROOTSYS/lib:$ROOTSYS/geom/gdml
   //

   TString sfile(filename);
   if (sfile.Contains(".C")) {
      //Save geometry as a C++ script
      if (fgVerboseLevel>0) Info("Export","Exporting %s %s as C++ code", GetName(), GetTitle());
      fTopVolume->SaveAs(filename);
      return 1;
   }
   if (sfile.Contains(".gdml")) {
      //Save geometry as a gdml file
      if (fgVerboseLevel>0) Info("Export","Exporting %s %s as gdml code", GetName(), GetTitle());
      gROOT->ProcessLine("TPython::Exec(\"from math import *\")");

      gROOT->ProcessLine("TPython::Exec(\"import writer\")");
      gROOT->ProcessLine("TPython::Exec(\"import ROOTwriter\")");

      // get TGeoManager and top volume
      gROOT->ProcessLine("TPython::Exec(\"geomgr = ROOT.gGeoManager\")");
      gROOT->ProcessLine("TPython::Exec(\"topV = geomgr.GetTopVolume()\")");

      // instanciate writer
      TString cmd=TString::Format("TPython::Exec(\"gdmlwriter = writer.writer('%s')\")",filename);
      gROOT->ProcessLine(cmd);
      gROOT->ProcessLine("TPython::Exec(\"binding = ROOTwriter.ROOTwriter(gdmlwriter)\")");

      // dump materials
      gROOT->ProcessLine("TPython::Exec(\"matlist = geomgr.GetListOfMaterials()\")");
      gROOT->ProcessLine("TPython::Exec(\"binding.dumpMaterials(matlist)\")");

      // dump solids
      gROOT->ProcessLine("TPython::Exec(\"shapelist = geomgr.GetListOfShapes()\")");
      gROOT->ProcessLine("TPython::Exec(\"binding.dumpSolids(shapelist)\")");

      // dump geo tree
      gROOT->ProcessLine("TPython::Exec(\"print 'Info in <TPython::Exec>: Traversing geometry tree'\")");
      gROOT->ProcessLine("TPython::Exec(\"gdmlwriter.addSetup('default', '1.0', topV.GetName())\")");
      gROOT->ProcessLine("TPython::Exec(\"binding.examineVol(topV)\")");

      // write file
      gROOT->ProcessLine("TPython::Exec(\"gdmlwriter.writeFile()\")");
      if (fgVerboseLevel>0) printf("Info in <TPython::Exec>: GDML Export complete - %s is ready\n", filename);
      return 1;
   }
   if (sfile.Contains(".root") || sfile.Contains(".xml")) {
      //Save geometry as a root file
      TFile *f = TFile::Open(filename,"recreate");
      if (!f || f->IsZombie()) {
         Error("Export","Cannot open file");
         return 0;
      }
      TString keyname = name;
      if (keyname.IsNull()) keyname = GetName();
      TString opt = option;
      opt.ToLower();
      if (opt.Contains("v")) {
         fStreamVoxels = kTRUE;
         if (fgVerboseLevel>0) Info("Export","Exporting %s %s as root file. Optimizations streamed.", GetName(), GetTitle());
      } else {
         fStreamVoxels = kFALSE;
         if (fgVerboseLevel>0) Info("Export","Exporting %s %s as root file. Optimizations not streamed.", GetName(), GetTitle());
      }
      Int_t nbytes = Write(keyname);
      fStreamVoxels = kFALSE;
      delete f;
      return nbytes;
   }
   return 0;
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
Int_t TGeoManager::GetVerboseLevel()
{
// Set verbosity level (static function).
// 0 - suppress messages related to geom-painter visibility level
// 1 - default value
   return fgVerboseLevel;
}

//______________________________________________________________________________
void TGeoManager::SetVerboseLevel(Int_t vl)
{
// Return current verbosity level (static function).
   fgVerboseLevel = vl;
}

//______________________________________________________________________________
TGeoManager *TGeoManager::Import(const char *filename, const char *name, Option_t * /*option*/)
{
   //static function
   //Import a geometry from a gdml or ROOT file
   //
   // -Case 1: gdml
   //  if filename ends with ".gdml" the foreign geometry described with gdml
   //  is imported executing some python scripts in $ROOTSYS/gdml.
   //  NOTE that to use this option, the PYTHONPATH must be defined like
   //      export PYTHONPATH=$ROOTSYS/lib:$ROOTSYS/gdml
   //
   // -Case 2: root file (.root) or root/xml file (.xml)
   //  Import in memory from filename the geometry with key=name.
   //  if name="" (default), the first TGeoManager object in the file is returned.
   //
   //Note that this function deletes the current gGeoManager (if one)
   //before importing the new object.

   if (fgLock) {
      ::Warning("TGeoManager::Import", "TGeoMananager in lock mode. NOT IMPORTING new geometry");
      return NULL;
   }
   if (!filename) return 0;
   if (fgVerboseLevel>0) ::Info("TGeoManager::Import","Reading geometry from file: %s",filename);

   if (gGeoManager) delete gGeoManager;
   gGeoManager = 0;

   if (strstr(filename,".gdml")) {
      // import from a gdml file
      new TGeoManager("GDMLImport", "Geometry imported from GDML");
      TString cmd = TString::Format("TGDMLParse::StartGDML(\"%s\")", filename);
      TGeoVolume* world = (TGeoVolume*)gROOT->ProcessLineFast(cmd);

      if(world == 0) {
         ::Error("TGeoManager::Import", "Cannot open file");
      }
      else {
         gGeoManager->SetTopVolume(world);
         gGeoManager->CloseGeometry();
         gGeoManager->DefaultColors();
      }
   } else {
      // import from a root file
      TDirectory::TContext ctxt(0);
      // in case a web file is specified, use the cacheread option to cache
      // this file in the cache directory
      TFile *f = 0;
      if (strstr(filename,"http")) f = TFile::Open(filename,"CACHEREAD");
      else                         f = TFile::Open(filename);
      if (!f || f->IsZombie()) {
         ::Error("TGeoManager::Import", "Cannot open file");
         return 0;
      }
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
      delete f;
   }
   if (!gGeoManager) return 0;
   if (!gROOT->GetListOfGeometries()->FindObject(gGeoManager)) gROOT->GetListOfGeometries()->Add(gGeoManager);
   if (!gROOT->GetListOfBrowsables()->FindObject(gGeoManager)) gROOT->GetListOfBrowsables()->Add(gGeoManager);
   gGeoManager->UpdateElements();
   return gGeoManager;
}

//___________________________________________________________________________
void TGeoManager::UpdateElements()
{
// Update element flags when geometry is loaded from a file.
   if (!fElementTable) return;
   TIter next(fMaterials);
   TGeoMaterial *mat;
   TGeoMixture *mix;
   TGeoElement *elem, *elem_table;
   Int_t i, nelem;
   while ((mat=(TGeoMaterial*)next())) {
      if (mat->IsMixture()) {
         mix = (TGeoMixture*)mat;
         nelem = mix->GetNelements();
         for (i=0; i<nelem; i++) {
            elem = mix->GetElement(i);
            elem_table = fElementTable->GetElement(elem->Z());
            if (elem != elem_table) {
               elem_table->SetDefined(elem->IsDefined());
               elem_table->SetUsed(elem->IsUsed());
            } else {
               elem_table->SetDefined();
            }
         }
      } else {
         elem = mat->GetElement();
         elem_table = fElementTable->GetElement(elem->Z());
         if (elem != elem_table) {
            elem_table->SetDefined(elem->IsDefined());
            elem_table->SetUsed(elem->IsUsed());
         } else {
            elem_table->SetUsed();
         }
      }
   }
}

//___________________________________________________________________________
Bool_t TGeoManager::InitArrayPNE() const
{
// Initialize PNE array for fast access via index and unique-id.
   if (fHashPNE) {
     fArrayPNE = new TObjArray(fHashPNE->GetSize());
     TIter next(fHashPNE);
     TObject *obj;
     while ((obj = next())) {
       fArrayPNE->Add(obj);
     }
     return kTRUE;
   }
   return kFALSE;
}

//___________________________________________________________________________
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
   GetCurrentNavigator()->MasterToLocal(master, top);
}

//______________________________________________________________________________
void TGeoManager::TopToMaster(const Double_t *top, Double_t *master) const
{
 // Convert coordinates from top volume frame to master.
   GetCurrentNavigator()->LocalToMaster(top, master);
}



