// @(#)root/geom:$Name:  $:$Id: TGeoVolume.cxx,v 1.34 2003/08/08 09:22:18 brun Exp $
// Author: Andrei Gheata   30/05/02
// Divide(), CheckOverlaps() implemented by Mihaela Gheata

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//Begin_Html
/*
<img src="gif/t_volume.jpg">
*/
//End_Html

////////////////////////////////////////////////////////////////////////////////
//   TGeoVolume - the base class representing solids. 
//
//   Volumes are the basic objects used in building the geometrical hierarchy.
// They represent unpositioned objects but store all information about the
// placement of the other volumes they may contain. Therefore a volume can
// be replicated several times in the geometry. In order to create a volume, one
// has to put togeather a shape and a medium which are already defined. Volumes
// have to be named by users at creation time. Every different name may represent a 
// an unique volume object, but may also represent more general a family (class)
// of volume objects having the same shape type and medium, but possibly
// different shape parameters. It is the user's task to provide different names
// for different volume families in order to avoid ambiguities at tracking time.
// A generic family rather than a single volume is created only in two cases : 
// when a generic shape is provided to the volume constructor or when a division
// operation is applied. Each volume in the geometry stores an unique
// ID corresponding to its family. In order to ease-up their creation, the manager
// class is providing an API that allows making a shape and a volume in a single step.
//
//   Volumes are objects that can be visualized, therefore having visibility,
// colour, line and fill attributes that can be defined or modified any time after
// the volume creation. It is advisable however to define these properties just
// after the first creation of a volume namespace, since in case of volume families
// any new member created by the modeler inherits these properties. 
//
//    In order to provide navigation features, volumes have to be able to find
// the proper container of any point defined in the local reference frame. This
// can be the volume itself, one of its positioned daughter volumes or none if 
// the point is actually outside. On the other hand, volumes have to provide also
// other navigation methods such as finding the distances to its shape boundaries
// or which daughter will be crossed first. The implementation of these features
// is done at shape level, but the local mother-daughters management is handled
// by volumes that builds additional optimisation structures upon geometry closure.
// In order to have navigation features properly working one has to follow the
// general rules for building a valid geometry (see TGeoManager class).
//
//   Now let's make a simple volume representing a copper wire. We suppose that
// a medium is already created (see TGeoMedium class on how to create media). 
// We will create a TUBE shape for our wire, having Rmin=0cm, Rmax=0.01cm
// and a half-length dZ=1cm :
//
//   TGeoTube *tube = new TGeoTube("wire_tube", 0, 0.01, 1);
//
// One may ommit the name for the shape if no retreiving by name is further needed
// during geometry building. The same shape can be shared by different volumes 
// having different names and materials. Now let's make the volume for our wire.
// The prototype for volumes constructor looks like :
//
//   TGeoVolume::TGeoVolume(const char *name, TGeoShape *shape, TGeoMedium *med)
//
// Since TGeoTube derives brom the base shape class, we can provide it to the volume
// constructor :
//
//   TGeoVolume *wire_co = new TGeoVolume("WIRE_CO", tube, ptrCOPPER);
//
// Do not bother to delete neither the media, shapes or volumes that you have
// created since all will be automatically cleaned on exit by the manager class.
// If we would have taken a look inside TGeoManager::MakeTube() method, we would
// have been able to create our wire with a single line :
//
//   TGeoVolume *wire_co = gGeoManager->MakeTube("WIRE_CO", ptrCOPPER, 0, 0.01, 1);
//
// The same applies for all primitive shapes, for which there can be found
// corresponding MakeSHAPE() methods. Their usage is much more convenient unless 
// a shape has to be shared between more volumes. Let's make now an aluminium wire 
// having the same shape, supposing that we have created the copper wire with the 
// line above :
//
//   TGeoVolume *wire_al = new TGeoVolume("WIRE_AL", wire_co->GetShape(), ptrAL);
//
// Now that we have learned how to create elementary volumes, let's see how we
// can create a geometrical hierarchy.
//
//
//   Positioning volumes
// -----------------------
//
//   When creating a volume one does not specify if this will contain or not other
// volumes. Adding daughters to a volume implies creating those and adding them 
// one by one to the list of daughters. Since the volume has to know the position 
// of all its daughters, we will have to supply at the same time a geometrical 
// transformation with respect to its local reference frame for each of them.
// The objects referencing a volume and a transformation are called NODES and
// their creation is fully handled by the modeler. They represent the link 
// elements in the hierarchy of volumes. Nodes are unique and distinct geometrical
// objects ONLY from their container point of view. Since volumes can be replicated
// in the geometry, the same node may be found on different branches.
//
//Begin_Html
/*
<img src="gif/t_example.jpg">
*/
//End_Html
//
//   An important observation is that volume objects are owned by the TGeoManager
// class. This stores a list of all volumes in the geometry, that is cleaned
// upon destruction.
//
//   Let's consider positioning now our wire in the middle of a gas chamber. We 
// need first to define the gas chamber :
//
//   TGeoVolume *chamber = gGeoManager->MakeTube("CHAMBER", ptrGAS, 0, 1, 1);
// 
// Now we can put the wire inside :
//
//   chamber->AddNode(wire_co, 1);
//
// If we inspect now the chamber volume in a browser, we will notice that it has 
// one daughter. Of course the gas has some container also, but let's keep it like 
// that for the sake of simplicity. The full prototype of AddNode() is :
//
//   TGeoVolume::AddNode(TGeoVolume *daughter, Int_t usernumber, 
//                       TGeoMatrix *matrix=gGeoIdentity)
//
// Since we did not supplied the third argument, the wire will be positioned with
// an identity transformation inside the chamber. One will notice that the inner
// radii of the wire and chamber are both zero - therefore, aren't the two volumes
// overlapping ? The answer is no, the modeler is even relaying on the fact that
// any daughter is fully contained by its mother. On the other hand, neither of
// the nodes positioned inside a volume should overlap with each other. We will
// see that there are allowed some exceptions to those rules.
//
// Overlapping volumes
// --------------------
//
//   Positioning volumes that does not overlap their neighbours nor extrude
// their container is sometimes quite strong contrain. Some parts of the geometry
// might overlap naturally, e.g. two crossing tubes. The modeller supports such
// cases only if the overlapping nodes are declared by the user. In order to do
// that, one should use TGeoVolume::AddNodeOverlap() instead of TGeoVolume::AddNode().
//   When 2 or more positioned volumes are overlapping, not all of them have to
// be declared so, but at least one. A point inside an overlapping region equally
// belongs to all overlapping nodes, but the way these are defined can enforce
// the modeler to give priorities.
//   The general rule is that the deepest node in the hierarchy containing a point
// have the highest priority. For the same geometry level, non-overlapping is
// prioritized over overlapping. In order to illustrate this, we will consider 
// few examples. We will designate non-overlapping nodes as ONLY and the others
// MANY as in GEANT3, where this concept was introduced:
//   1. The part of a MANY node B extruding its container A will never be "seen" 
// during navigation, as if B was in fact the result of the intersection of A and B.
//   2. If we have two nodes A (ONLY) and B (MANY) inside the same container, all
// points in the overlapping region of A and B will be designated as belonging to A.
//   3. If A an B in the above case were both MANY, points in the overlapping 
// part will be designated to the one defined first. Both nodes must have the 
// same medium.
//   4. The silces of a divided MANY will be as well MANY.
//
// One needs to know that navigation inside geometry parts MANY nodes is much 
// slower. Any overlapping part can be defined based on composite shapes - this
// is always recommended. 

//   Replicating volumes
// -----------------------
//
//   What can we do if our chamber contains two identical wires instead of one ?
// What if then we would need 1000 chambers in our detector ? Should we create
// 2000 wires and 1000 chamber volumes ? No, we will just need to replicate the
// ones that we have already created.
//
//   chamber->AddNode(wire_co, 1, new TGeoTranslation(-0.2,0,0));
//   chamber->AddNode(wire_co, 2, new TGeoTranslation(0.2,0,0));
//
//   The 2 nodes that we have created inside chamber will both point to a wire_co
// object, but will be completely distinct : WIRE_CO_1 and WIRE_CO_2. We will
// want now to place symetrically 1000 chabmers on a pad, following a pattern
// of 20 rows and 50 columns. One way to do this will be to replicate our chamber
// by positioning it 1000 times in different positions of the pad. Unfortunatelly,
// this is far from being the optimal way of doing what we want.
// Imagine that we would like to find out which of the 1000 chambers is containing
// a (x,y,z) point defined in the pad reference. You will never have to do that,
// since the modeller will take care of it for you, but let's guess what it has 
// to do. The most simple algorithm will just loop over all daughters, convert
// the point from mother to local reference and check if the current chamber
// contains the point or not. This might be efficient for pads with few chambers,
// but definitely not for 1000. Fortunately the modeler is smarter than that and 
// create for each volume some optimization structures called voxels (see Voxelization) 
// to minimize the penalty having too many daughters, but if you have 100 pads like 
// this in your geometry you will anyway loose a lot in your tracking performance.
//
//   The way out when volumes can be arranged acording to simple patterns is the
// usage of divisions. We will describe them in detail later on. Let's think now
// at a different situation : instead of 1000 chambers of the same type, we may
// have several types of chambers. Let's say all chambers are cylindrical and have
// a wire inside, but their dimensions are different. However, we would like all
// to be represented by a single volume family, since they have the same properties.
//
//   Volume families
// ------------------
// A volume family is represented by the class TGeoVolumeMulti. It represents
// a class of volumes having the same shape type and each member will be 
// identified by the same name and volume ID. Any operation applied to a 
// TGeoVolume equally affects all volumes in that family. The creation of a 
// family is generally not a user task, but can be forced in particular cases:
//
//      TGeoManager::Volume(const char *vname, const char *shape, Int_t nmed);
//
// where VNAME is the family name, NMED is the medium number and SHAPE is the
// shape type that can be:
//   box    - for TGeoBBox
//   trd1   - for TGeoTrd1
//   trd2   - for TGeoTrd2
//   trap   - for TGeoTrap
//   gtra   - for TGeoGtra
//   para   - for TGeoPara
//   tube, tubs - for TGeoTube, TGeoTubeSeg
//   cone, cons - for TGeoCone, TgeoCons
//   eltu   - for TGeoEltu
//   ctub   - for TGeoCtub
//   pcon   - for TGeoPcon
//   pgon   - for TGeoPgon
//
// Volumes are then added to a given family upon adding the generic name as node
// inside other volume:
//   TGeoVolume *box_family = gGeoManager->Volume("BOXES", "box", nmed);
//   ...
//   gGeoManager->Node("BOXES", Int_t copy_no, "mother_name", 
//                     Double_t x, Double_t y, Double_t z, Int_t rot_index,
//                     Bool_t is_only, Double_t *upar, Int_t npar);
// here:
//   BOXES   - name of the family of boxes
//   copy_no - user node number for the created node
//   mother_name - name of the volume to which we want to add the node
//   x,y,z   - translation components
//   rot_index   - indx of a rotation matrix in the list of matrices
//   upar    - array of actual shape parameters
//   npar    - number of parameters
// The parameters order and number are the same as in the corresponding shape
// constructors.
//
//   An other particular case where volume families are used is when we want
// that a volume positioned inside a container to match one ore more container
// limits. Suppose we want to position the same box inside 2 different volumes
// and we want the Z size to match the one of each container:
//
//   TGeoVolume *container1 = gGeoManager->MakeBox("C1", imed, 10,10,30);
//   TGeoVolume *container2 = gGeoManager->MakeBox("C2", imed, 10,10,20);
//   TGeoVolume *pvol       = gGeoManager->MakeBox("PVOL", jmed, 3,3,-1);
//   container1->AddNode(pvol, 1);
//   container2->AddNode(pvol, 1);
//
//   Note that the third parameter of PVOL is negative, which does not make sense
// as half-length on Z. This is interpreted as: when positioned, create a box
// replacing all invalid parameters with the corresponding dimensions of the
// container. This is also internally handled by the TGeoVolumeMulti class, which
// does not need to be instanciated by users.
//
//   Dividing volumes
// ------------------
//
//   Volumes can be divided according a pattern. The most simple division can
// be done along one axis, that can be: X, Y, Z, Phi, Rxy or Rxyz. Let's take 
// the most simple case: we would like to divide a box in N equal slices along X
// coordinate, representing a new volume family. Supposing we already have created
// the initial box, this can be done like:
//
//      TGeoVolume *slicex = box->Divide("SLICEX", 1, N);
//
// where SLICE is the name of the new family representing all slices and 1 is the
// slicing axis. The meaning of the axis index is the following: for all volumes
// having shapes like box, trd1, trd2, trap, gtra or para - 1,2,3 means X,Y,Z; for
// tube, tubs, cone, cons - 1 means Rxy, 2 means phi and 3 means Z; for pcon and
// pgon - 2 means phi and 3 means Z; for spheres 1 means R and 2 means phi.
//   In fact, the division operation has the same effect as positioning volumes
// in a given order inside the divided container - the advantage being that the 
// navigation in such a structure is much faster. When a volume is divided, a
// volume family corresponding to the slices is created. In case all slices can
// be represented by a single shape, only one volume is added to the family and
// positioned N times inside the divided volume, otherwise, each slice will be 
// represented by a distinct volume in the family.
//   Divisions can be also performed in a given range of one axis. For that, one
// have to specify also the starting coordinate value and the step:
//
//      TGeoVolume *slicex = box->Divide("SLICEX", 1, N, start, step);
//
// A check is always done on the resulting division range : if not fitting into
// the container limits, an error message is posted. If we will browse the divided
// volume we will notice that it will contain N nodes starting with index 1 upto
// N. The first one has the lower X limit at START position, while the last one
// will have the upper X limit at START+N*STEP. The resulting slices cannot
// be positioned inside an other volume (they are by default positioned inside the
// divided one) but can be further divided and may contain other volumes:
//
//      TGeoVolume *slicey = slicex->Divide("SLICEY", 2, N1);
//      slicey->AddNode(other_vol, index, some_matrix);
//
//   When doing that, we have to remember that SLICEY represents a family, therefore
// all members of the family will be divided on Y and the other volume will be 
// added as node inside all.
//   In the example above all the resulting slices had the same shape as the
// divided volume (box). This is not always the case. For instance, dividing a
// volume with TUBE shape on PHI axis will create equal slices having TUBESEG 
// shape. Other divisions can alsoo create slices having shapes with different
// dimensins, e.g. the division of a TRD1 volume on Z. 
//   When positioning volumes inside slices, one can do it using the generic
// volume family (e.g. slicey). This should be done as if the coordinate system
// of the generic slice was the same as the one of the divided volume. The generic
// slice in case of PHI divisioned is centered with respect to X axis. If the
// family contains slices of different sizes, ani volume positioned inside should 
// fit into the smallest one.
//    Examples for specific divisions according to shape types can be found inside
// shape classes.
// 
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

//_____________________________________________________________________________
TGeoVolume::TGeoVolume()
{ 
// dummy constructor
   fNodes    = 0;
   fShape    = 0;
   fFinder   = 0;
   fVoxels   = 0;
   fField    = 0;
   fMedium   = 0;
   fNumber   = 0;
   fNtotal   = 0;
   fOption   = "";
   TObject::ResetBit(kVolumeImportNodes);
}

//_____________________________________________________________________________
TGeoVolume::TGeoVolume(const char *name, const TGeoShape *shape, const TGeoMedium *med)
           :TNamed(name, "")
{
// default constructor
   fNodes    = 0;
   fShape    = (TGeoShape*)shape;
   if (fShape->TestBit(TGeoShape::kGeoBad)) {
      Warning("Ctor", "volume %s has invalid shape", name);
   }   
   fFinder   = 0;
   fVoxels   = 0;
   fField    = 0;
   fOption   = "";
   fMedium   = (TGeoMedium*)med;
   fNumber   = 0;
   fNtotal   = 0;
   if (gGeoManager) fNumber = gGeoManager->AddVolume(this);
   TObject::ResetBit(kVolumeImportNodes);
}

//_____________________________________________________________________________
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

//_____________________________________________________________________________
void TGeoVolume::Browse(TBrowser *b)
{
// How to browse a volume
   if (!b) return;
   if (!GetNdaughters()) b->Add(this);
   for (Int_t i=0; i<GetNdaughters(); i++) 
      b->Add(GetNode(i)->GetVolume());
}

//_____________________________________________________________________________
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

//_____________________________________________________________________________
void TGeoVolume::CheckOverlaps(Double_t ovlp, Option_t *option) const
{
// Overlap checking tool. Check for illegal overlaps within a limit OVLP.
   if (!GetNdaughters() || fFinder) return;
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) {
      Error("CheckOverlaps", "Could not instanciate painter");
      return;
   }
   gGeoManager->SetNsegments(80);
   if (!gGeoManager->IsCheckingOverlaps()) {
      gGeoManager->ClearOverlaps();
      printf("=== Checking overlaps vor volume %s ===\n", GetName());
   }   
   painter->CheckOverlaps(this, ovlp, option);
   
   if (!gGeoManager->IsCheckingOverlaps()) {
      gGeoManager->SortOverlaps();
      TObjArray *overlaps = gGeoManager->GetListOfOverlaps();
      Int_t novlps = overlaps->GetEntriesFast();
      TNamed *obj;
      char *name;
      char num[10];
      Int_t ndigits=1;
      Int_t i,j, result=novlps;
      while ((result /= 10)) ndigits++;
      for (i=0; i<novlps; i++) {
         obj = (TNamed*)overlaps->At(i);
         result = i;
         name = new char[10];
         name[0] = 'o';
         name[1] = 'v';
         for (j=0; j<ndigits; j++) name[j+2]='0';
         name[ndigits+2] = 0;
         sprintf(num,"%i", i);
         memcpy(name+2+ndigits-strlen(num), num, strlen(num));
         obj->SetName(name);
      }   
      printf("   number of illegal overlaps/extrusions : %d\n", novlps);
   }   
}

//_____________________________________________________________________________
void TGeoVolume::CleanAll()
{
   ClearNodes();
   ClearShape();
}

//_____________________________________________________________________________
void TGeoVolume::ClearShape()
{
   gGeoManager->ClearShape(fShape);
}   

//_____________________________________________________________________________
void TGeoVolume::CheckShapes()
{
// check for negative parameters in shapes.
// THIS METHOD LEAVES SOME GARBAGE NODES -> memory leak, to be fixed
//   printf("---Checking daughters of volume %s\n", GetName());
   if (fShape->IsRunTimeShape()) {
      Error("CheckShapes", "volume %s has run-time shape", GetName());
      InspectShape();
      return;
   }   
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
         TGeoShape *new_shape = shape->GetMakeRuntimeShape(fShape, node->GetMatrix());
         if (!new_shape) {
            Error("CheckShapes","cannot resolve runtime shape for volume %s/%s\n",
                   GetName(),old_vol->GetName());
            continue;
         }         
         TGeoVolume *new_volume = old_vol->MakeCopyVolume(new_shape);
//         printf(" new volume %s shape params :\n", new_volume->GetName());
//         new_volume->InspectShape();
         new_node->SetVolume(new_volume);
         // decouple the old node and put the new one instead
         fNodes->AddAt(new_node, i);
//         new_volume->CheckShapes();
      }
   }
}     

//_____________________________________________________________________________
Int_t TGeoVolume::CountNodes(Int_t nlevels)
{
// count total number of subnodes starting from this volume, nlevels down
   if (fNtotal) return fNtotal;
   fNtotal = 1;
   if (!fNodes || !nlevels) return fNtotal;
   Int_t nd = GetNdaughters();
   TGeoNode *node;
   TGeoVolume *vol;
   for (Int_t i=0; i<nd; i++) {
      node = GetNode(i);
      vol = node->GetVolume();
      fNtotal += vol->CountNodes(nlevels-1);
   }
   return fNtotal;
}

//_____________________________________________________________________________
Bool_t TGeoVolume::IsFolder() const
{
// Return TRUE if volume contains nodes
   if (fNodes) return kTRUE;
   else return kFALSE;
}

//_____________________________________________________________________________
Bool_t TGeoVolume::IsStyleDefault() const
{
// check if the visibility and attributes are the default ones
   if (!IsVisible()) return kFALSE;
   if (GetLineColor() != gStyle->GetLineColor()) return kFALSE;
   if (GetLineStyle() != gStyle->GetLineStyle()) return kFALSE;
   if (GetLineWidth() != gStyle->GetLineWidth()) return kFALSE;
   return kTRUE;
}

//_____________________________________________________________________________
void TGeoVolume::InspectMaterial() const
{
   fMedium->GetMaterial()->Print();
}

//_____________________________________________________________________________
void TGeoVolume::cd(Int_t inode) const
{
// Actualize matrix of node indexed <inode>
   if (fFinder) fFinder->cd(inode-fFinder->GetDivIndex());
}

//_____________________________________________________________________________
void TGeoVolume::AddNode(const TGeoVolume *vol, Int_t copy_no, TGeoMatrix *mat, Option_t * /*option*/)
{
// Add a TGeoNode to the list of nodes. This is the usual method for adding
// daughters inside the container volume.
   TGeoMatrix *matrix = mat;
   if (matrix==0) matrix = gGeoIdentity;
   else           matrix->RegisterYourself();
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
   node->SetNumber(copy_no);
}

//_____________________________________________________________________________
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
   node->SetNumber(copy_no+1);
}

//_____________________________________________________________________________
void TGeoVolume::AddNodeOverlap(const TGeoVolume *vol, Int_t copy_no, TGeoMatrix *mat, Option_t * /*option*/)
{
// Add a TGeoNode to the list of nodes. This is the usual method for adding
// daughters inside the container volume.
   TGeoMatrix *matrix = mat;
   if (matrix==0) matrix = gGeoIdentity;
   else           matrix->RegisterYourself();
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
   node->SetNumber(copy_no);
   node->SetOverlapping();
   if (vol->GetMedium() == fMedium)
   node->SetVirtual();
}

//_____________________________________________________________________________
TGeoVolume *TGeoVolume::Divide(const char *divname, Int_t iaxis, Int_t ndiv, Double_t start, Double_t step, Int_t numed, Option_t *option)
{
// Division a la G3. The volume will be divided along IAXIS (see shape classes), in NDIV
// slices, from START with given STEP. The division volumes will have medium number NUMED.
// If NUMED=0 they will get the medium number of the divided volume (this). If NDIV<=0,
// all range of IAXIS will be divided and the resulting number of divisions will be centered on
// IAXIS. If STEP<=0, the real STEP will be computed as the full range of IAXIS divided by NDIV.
// Options (case insensitive):
//  N  - divide all range in NDIV cells (same effect as STEP<=0) (GSDVN in G3)
//  NX - divide range starting with START in NDIV cells          (GSDVN2 in G3)
//  S  - divide all range with given STEP. NDIV is computed and divisions will be centered
//         in full range (same effect as NDIV<=0)                (GSDVS, GSDVT in G3)
//  SX - same as DVS, but from START position.                   (GSDVS2, GSDVT2 in G3)

   if (fFinder) {
   // volume already divided.
      Fatal("Divide","volume %s already divided", GetName());
      return 0;
   }
   TString opt(option);
   opt.ToLower();
   TString stype = fShape->ClassName();
   if (!fNodes) fNodes = new TObjArray();
   Double_t xlo, xhi, range;
   range = fShape->GetAxisRange(iaxis, xlo, xhi);
   // for phi divisions correct the range
   if (!strcmp(fShape->GetAxisName(iaxis), "PHI")) {
      if ((start-xlo)<-1E-3) start+=360.;
      if (range==360) {
         xlo = start;
         xhi = start+range;
      }   
   }   
   if (range <=0) {
      InspectShape();
      Fatal("Divide", "cannot divide volume %s (%s) on %s axis", GetName(), stype.Data(), fShape->GetAxisName(iaxis));
      return 0;
   }
   if (ndiv<=0 || opt.Contains("s")) {
      if (step<=0) {
         Fatal("Divide", "invalid division type for volume %s : ndiv=%i, step=%g", GetName(), ndiv, step);
         return 0;
      }   
      if (opt.Contains("x")) {
         if ((xlo-start)>1E-3 || (xhi-start)<-1E-3) {
            Fatal("Divide", "invalid START=%g for division on axis %s of volume %s. Range is (%g, %g)",
                  start, fShape->GetAxisName(iaxis), GetName(), xlo, xhi);
            return 0;
         }
         xlo = start;
         range = xhi-xlo;
      }            
      ndiv = Int_t((range+0.1*step)/step);
      Double_t ddx = range - ndiv*step;
      // always center the division in this case
      if (ddx>1E-3) Warning("Divide", "division of volume %s on %s axis (ndiv=%d) will be centered in the full range",
                            GetName(), fShape->GetAxisName(iaxis), ndiv);
      start = xlo + 0.5*ddx;
   }
   if (step<=0 || opt.Contains("n")) {
      if (opt.Contains("x")) {
         if ((xlo-start)>1E-3 || (xhi-start)<-1E-3) {
            Fatal("Divide", "invalid START=%g for division on axis %s of volume %s. Range is (%g, %g)",
                  start, fShape->GetAxisName(iaxis), GetName(), xlo, xhi);
            return 0;
         }
         xlo = start;
         range = xhi-xlo;
      }     
      step  = range/ndiv;
      start = xlo;
   }
   
   Double_t end = start+ndiv*step;
   if (((start-xlo)<-1E-3) || ((end-xhi)>1E-3)) {
      Fatal("Divide", "division of volume %s on axis %s exceed range (%g, %g)",
            GetName(), fShape->GetAxisName(iaxis), xlo, xhi);
      return 0;
   }         
   TGeoVolume *voldiv = fShape->Divide(this, divname, iaxis, ndiv, start, step);
   if (numed) {
      TGeoMedium *medium = gGeoManager->GetMedium(numed);
      if (!medium) {
         Fatal("Divide", "invalid medium number %d for division volume %s", numed, divname);
         return voldiv;
      }   
      voldiv->SetMedium(medium);
   }   
   return voldiv; 
}

//_____________________________________________________________________________
Int_t TGeoVolume::DistancetoPrimitive(Int_t px, Int_t py)
{
// compute the closest distance of approach from point px,py to this volume
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) return 9999;
   return painter->DistanceToPrimitiveVol(this, px, py);
}

//_____________________________________________________________________________
void TGeoVolume::Draw(Option_t *option)
{
// draw top volume according to option
   TGeoVolume *old_vol = gGeoManager->GetTopVolume();
   if (old_vol!=this) gGeoManager->SetTopVolume(this);
   else old_vol=0;
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) return;
   //painter->SetRaytracing(kFALSE);
   painter->Draw(option);   
}

//_____________________________________________________________________________
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

//_____________________________________________________________________________
Bool_t TGeoVolume::OptimizeVoxels()
{
// Perform an exensive sampling to find which type of voxelization is
// most efficient.
   printf("Optimizing volume %s ...\n", GetName());
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) return kFALSE;
   return painter->TestVoxels(this);   
}

//_____________________________________________________________________________
void TGeoVolume::Paint(Option_t *option)
{
// paint volume
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) return;
   painter->Paint(option);   
}

//_____________________________________________________________________________
void TGeoVolume::PrintVoxels() const
{
   if (fVoxels) fVoxels->Print();
}

//_____________________________________________________________________________
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

//_____________________________________________________________________________
void TGeoVolume::RandomPoints(Int_t npoints, Option_t *option)
{
// Draw random points in the bounding box of this volume.
   gGeoManager->RandomPoints(this, npoints, option);
}

//_____________________________________________________________________________
void TGeoVolume::RandomRays(Int_t nrays, Double_t startx, Double_t starty, Double_t startz)
{
// Random raytracing method.
   TGeoVolume *old_vol = gGeoManager->GetTopVolume();
   if (old_vol!=this) gGeoManager->SetTopVolume(this);
   else old_vol=0;
   gGeoManager->RandomRays(nrays, startx, starty, startz);
}

//_____________________________________________________________________________
void TGeoVolume::Raytrace(Option_t * /*option*/)
{
// Draw this volume with current settings and perform raytracing in the pad.
//   Draw();
//   gGeoManager->GetGeomPainter()->SetRaytracing();
//   gGeoManager->GetGeomPainter()->ModifiedPad();
}   

//_____________________________________________________________________________
void TGeoVolume::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
// Execute mouse actions on this volume.
   gGeoManager->GetGeomPainter()->ExecuteVolumeEvent(this, event, px, py);
}

//_____________________________________________________________________________
TGeoNode *TGeoVolume::FindNode(const char *name) const
{
// search a daughter inside the list of nodes
   return ((TGeoNode*)fNodes->FindObject(name));
}

//_____________________________________________________________________________
Int_t TGeoVolume::GetNodeIndex(const TGeoNode *node, Int_t *check_list, Int_t ncheck) const
{
   TGeoNode *current = 0;
   for (Int_t i=0; i<ncheck; i++) {
      current = (TGeoNode*)fNodes->At(check_list[i]);
      if (current==node) return check_list[i];
   }
   return -1;
}

//_____________________________________________________________________________
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

//_____________________________________________________________________________
char *TGeoVolume::GetObjectInfo(Int_t px, Int_t py) const
{
   TGeoVolume *vol = (TGeoVolume*)this;
   return gGeoManager->GetGeomPainter()->GetVolumeInfo(vol, px, py);
}

//_____________________________________________________________________________
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

//_____________________________________________________________________________
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

//_____________________________________________________________________________
void TGeoVolume::GrabFocus()
{
// Move perspective view focus to this volume
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (painter) painter->GrabFocus();
}   

//_____________________________________________________________________________
TGeoVolume *TGeoVolume::MakeCopyVolume(TGeoShape *newshape)
{
    // make a copy of this volume
//    printf("   Making a copy of %s\n", GetName());
    char *name = new char[strlen(GetName())+1];
    sprintf(name, "%s", GetName());
    // build a volume with same name, shape and medium
    TGeoVolume *vol = new TGeoVolume(name, newshape, fMedium);
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

//_____________________________________________________________________________
void TGeoVolume::SetAsTopVolume()
{
   gGeoManager->SetTopVolume(this);
}

//_____________________________________________________________________________
void TGeoVolume::SetCurrentPoint(Double_t x, Double_t y, Double_t z)
{
   gGeoManager->SetCurrentPoint(x,y,z);
}

//_____________________________________________________________________________
void TGeoVolume::SetShape(const TGeoShape *shape)
{
// set the shape associated with this volume
   if (!shape) {
      Error("SetShape", "No shape");
      return;
   }
   fShape = (TGeoShape*)shape;  
}

//_____________________________________________________________________________
void TGeoVolume::Sizeof3D() const
{
//   Compute size of this 3d object.
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) return;
   painter->Sizeof3D(this);
}

//_____________________________________________________________________________
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

//_____________________________________________________________________________
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


//_____________________________________________________________________________
void TGeoVolume::SetOption(const char * /*option*/)
{
// set the current options  
}

//_____________________________________________________________________________
void TGeoVolume::SetLineColor(Color_t lcolor) 
{
   TAttLine::SetLineColor(lcolor);
   //if (gGeoManager->IsClosed()) SetVisTouched(kTRUE);
}   

//_____________________________________________________________________________
void TGeoVolume::SetLineStyle(Style_t lstyle) 
{
   TAttLine::SetLineStyle(lstyle);
   //if (gGeoManager->IsClosed()) SetVisTouched(kTRUE);
}   

//_____________________________________________________________________________
void TGeoVolume::SetLineWidth(Style_t lwidth) 
{
   TAttLine::SetLineWidth(lwidth);
   //if (gGeoManager->IsClosed()) SetVisTouched(kTRUE);
}   

//_____________________________________________________________________________
TGeoNode *TGeoVolume::GetNode(const char *name) const
{
// get the pointer to a daughter node
   if (!fNodes) return 0;
   TGeoNode *node = (TGeoNode *)fNodes->FindObject(name);
   return node;
}

//_____________________________________________________________________________
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

//_____________________________________________________________________________
void TGeoVolume::FindOverlaps() const
{
// loop all nodes marked as overlaps and find overlaping brothers
   if (!Valid()) {
      Error("FindOverlaps","Bounding box not valid");
      return;
   }   
   if (!fVoxels) return;
   Int_t nd = GetNdaughters();
   if (!nd) return;
   TGeoNode *node=0;
   Int_t inode = 0;
   for (inode=0; inode<nd; inode++) {
      node = GetNode(inode);
      if (!node->IsOverlapping()) continue;
      fVoxels->FindOverlaps(inode);
   }
}

//_____________________________________________________________________________
void TGeoVolume::SetVisibility(Bool_t vis)
{
// set visibility of this volume
   TGeoAtt::SetVisibility(vis);
   if (gGeoManager->IsClosed()) SetVisTouched(kTRUE);
   gGeoManager->ModifiedPad();
}   

//_____________________________________________________________________________
Bool_t TGeoVolume::Valid() const
{
   return fShape->IsValidBox();
}

//_____________________________________________________________________________
Bool_t TGeoVolume::FindMatrixOfDaughterVolume(TGeoVolume *vol) const
{
// Find a daughter node having VOL as volume and fill TGeoManager::fHMatrix
// with its global matrix.
   if (vol == this) return kTRUE;
   Int_t nd = GetNdaughters();
   if (!nd) return kFALSE;
   TGeoHMatrix *global = gGeoManager->GetHMatrix();
   TGeoNode *dnode;
   TGeoVolume *dvol;
   TGeoMatrix *local;
   Int_t i;
   for (i=0; i<nd; i++) {
      dnode = GetNode(i);
      dvol = dnode->GetVolume();
      if (dvol == vol) {
         local = dnode->GetMatrix();
         global->MultiplyLeft(local);
         return kTRUE;
      }
   }
   for (i=0; i<nd; i++) {
      dnode = GetNode(i);
      dvol = dnode->GetVolume();
      if (dvol->FindMatrixOfDaughterVolume(vol)) return kTRUE;
   }
   return kFALSE;
}                    

//_____________________________________________________________________________
void TGeoVolume::VisibleDaughters(Bool_t vis)
{
// set visibility for daughters
   SetVisDaughters(vis);
   if (gGeoManager->IsClosed()) SetVisTouched(kTRUE);
   gGeoManager->ModifiedPad();
}

//_____________________________________________________________________________
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

//_____________________________________________________________________________
Double_t TGeoVolume::Weight(Double_t precision, Option_t *option)
{
// Estimate the weight of a volume with SIGMA(M)/M better than PRECISION.
// Option can be : v - verbose (default)
   gGeoManager->SetTopVolume((TGeoVolume*)this);
   return gGeoManager->Weight(precision, option);
}

ClassImp(TGeoVolumeMulti)


//_____________________________________________________________________________
TGeoVolumeMulti::TGeoVolumeMulti()
{ 
// dummy constructor
   fVolumes   = 0;
   fDivision = 0;
   fNumed = 0;
   fNdiv = 0;
   fAxis = 0;
   fStart = 0;
   fStep = 0;
   fAttSet = kFALSE;
   TObject::SetBit(kVolumeMulti);
}

//_____________________________________________________________________________
TGeoVolumeMulti::TGeoVolumeMulti(const char *name, const TGeoMedium *med)
{
// default constructor
   fVolumes = new TObjArray();
   fDivision = 0;
   fNumed = 0;
   fNdiv = 0;
   fAxis = 0;
   fStart = 0;
   fStep = 0;
   fAttSet = kFALSE;
   TObject::SetBit(kVolumeMulti);
   SetName(name);
   SetMedium(med);
   gGeoManager->AddVolume(this);
//   printf("--- volume multi %s created\n", name);
}

//_____________________________________________________________________________
TGeoVolumeMulti::~TGeoVolumeMulti()
{
// Destructor
   if (fVolumes) delete fVolumes;
}

//_____________________________________________________________________________
void TGeoVolumeMulti::AddVolume(TGeoVolume *vol) 
{
// Add a volume with valid shape to the list of volumes. Copy all existing nodes
// to this volume
   Int_t idx = fVolumes->GetEntriesFast();
   fVolumes->AddAtAndExpand(vol,idx);
   TGeoVolumeMulti *div;
   TGeoVolume *cell;
   if (fDivision) {
      div = (TGeoVolumeMulti*)vol->Divide(fDivision->GetName(), fAxis, fNdiv, fStart, fStep, fNumed, fOption.Data());
//      div->MakeCopyNodes(fDivision);
      for (Int_t i=0; i<div->GetNvolumes(); i++) {
         cell = div->GetVolume(i);
         fDivision->AddVolume(cell);
//         cell->MakeCopyNodes(fDivision);
      }
   }      
   if (fNodes)
      vol->MakeCopyNodes(this);
}
   

//_____________________________________________________________________________
void TGeoVolumeMulti::AddNode(const TGeoVolume *vol, Int_t copy_no, TGeoMatrix *mat, Option_t *option)
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

//_____________________________________________________________________________
void TGeoVolumeMulti::AddNodeOverlap(const TGeoVolume *vol, Int_t copy_no, TGeoMatrix *mat, Option_t *option)
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


//_____________________________________________________________________________
TGeoVolume *TGeoVolumeMulti::Divide(const char *divname, Int_t iaxis, Int_t ndiv, Double_t start, Double_t step, Int_t numed, const char *option)
{
// division of multiple volumes
   if (fDivision) {
      Error("Divide", "volume %s already divided", GetName());
      return 0;
   }   
   Int_t nvolumes = fVolumes->GetEntriesFast();
   TGeoMedium *medium = fMedium;
   if (numed) {
      medium = gGeoManager->GetMedium(numed);
      if (!medium) {
         Error("Divide", "Invalid medium number %d for division volume %s", numed, divname);
         medium = fMedium;
      }
   }      
   if (!nvolumes) {
      // this is a virtual volume
      fDivision = new TGeoVolumeMulti(divname, medium);
      fNumed = medium->GetId();
      fOption = option;
      fAxis = iaxis;
      fNdiv = ndiv;
      fStart = start;
      fStep = step;
      // nothing else to do at this stage
      return fDivision;
   }      
   TGeoVolume *vol = 0;
   fDivision = new TGeoVolumeMulti(divname, medium);
   fNumed = medium->GetId();
   fOption = option;
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
      fDivision->AddVolume(vol->Divide(divname,iaxis,ndiv,start,step, numed, option)); 
   }
//   printf("--- volume multi %s (%i volumes) divided\n", GetName(), nvolumes);
   if (numed) fDivision->SetMedium(medium);
   return fDivision;
}

//_____________________________________________________________________________
TGeoVolume *TGeoVolumeMulti::MakeCopyVolume(TGeoShape *newshape)
{
    // make a copy of this volume
//    printf("   Making a copy of %s\n", GetName());
    char *name = new char[strlen(GetName())+1];
    sprintf(name, "%s", GetName());
    // build a volume with same name, shape and medium
    TGeoVolume *vol = new TGeoVolume(name, newshape, fMedium);
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
//    if (fFinder) {
//       Error("MakeCopyVolume", "volume %s divided", GetName());
//       vol->SetFinder(fFinder);
//    }
   if (fDivision) {
      TGeoVolume *cell;
      TGeoVolumeMulti *div = (TGeoVolumeMulti*)vol->Divide(fDivision->GetName(), fAxis, fNdiv, fStart, fStep, fNumed, fOption.Data());
//      div->MakeCopyNodes(fDivision);
      for (Int_t i=0; i<div->GetNvolumes(); i++) {
         cell = div->GetVolume(i);
         fDivision->AddVolume(cell);
//         cell->MakeCopyNodes(fDivision);
      }
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
//_____________________________________________________________________________
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

//_____________________________________________________________________________
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

//_____________________________________________________________________________
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

//_____________________________________________________________________________
void TGeoVolumeMulti::SetMedium(const TGeoMedium *med)
{
// Set medium for a multiple volume.
   TGeoVolume::SetMedium(med);
   Int_t nvolumes = fVolumes->GetEntriesFast();
   TGeoVolume *vol = 0;
   for (Int_t ivo=0; ivo<nvolumes; ivo++) {
      vol = GetVolume(ivo);
      vol->SetMedium(med); 
   }
}   


//_____________________________________________________________________________
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
