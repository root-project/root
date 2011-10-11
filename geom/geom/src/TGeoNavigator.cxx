// @(#)root/geom:$Id$
// Author: Mihaela Gheata   30/05/07

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
 
//_____________________________________________________________________________
// TGeoNavigator
//===============
//
//   Class providing navigation API for TGeo geometries. Several instances are 
// allowed for a single geometry.
// A default navigator is provided for any geometry but one may add several 
// others for parallel navigation:
//	
//	TGeoNavigator *navig = new TGeoNavigator(gGeoManager);
//	Int_t inav = gGeoManager->AddNavigator(navig);
//	gGeoManager->SetCurrentNavigator(inav);
//
//	.... and then switch back to the default navigator:
//
//	gGeoManager->SetCurrentNavigator(0);
//_____________________________________________________________________________

#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TGeoNode.h"
#include "TGeoVolume.h"
#include "TGeoPatternFinder.h"
#include "TGeoVoxelFinder.h"
#include "TMath.h"

#include "TGeoNavigator.h"

static Double_t gTolerance = TGeoShape::Tolerance();
const char *kGeoOutsidePath = " ";
const Int_t kN3 = 3*sizeof(Double_t);

ClassImp(TGeoNavigator)

//_____________________________________________________________________________
TGeoNavigator::TGeoNavigator()
              :fStep(0.),
               fSafety(0.),
               fLastSafety(0.),
               fThreadId(0),
               fLevel(0),
               fNmany(0),
               fNextDaughterIndex(0),
               fOverlapSize(0),
               fOverlapMark(0),
               fOverlapClusters(0),
               fSearchOverlaps(kFALSE),
               fCurrentOverlapping(kFALSE),
               fStartSafe(kFALSE),
               fIsEntering(kFALSE),
               fIsExiting(kFALSE),
               fIsStepEntering(kFALSE),
               fIsStepExiting(kFALSE),
               fIsOutside(kFALSE),
               fIsOnBoundary(kFALSE),
               fIsSameLocation(kFALSE),
               fIsNullStep(kFALSE),
               fGeometry(0),
               fCache(0),
               fCurrentVolume(0),
               fCurrentNode(0),
               fTopNode(0),
               fLastNode(0),
               fNextNode(0),
               fForcedNode(0),
               fBackupState(0),
               fCurrentMatrix(0),
               fGlobalMatrix(0),
               fDivMatrix(0),
               fPath()
                
{
// dummy constructor
   for (Int_t i=0; i<3; i++) {
      fNormal[i] = 0.;
      fCldir[i] = 0.;
      fCldirChecked[i] = 0.;
      fPoint[i] = 0.;
      fDirection[i] = 0.;
      fLastPoint[i] = 0.;
   }
}

//_____________________________________________________________________________
TGeoNavigator::TGeoNavigator(TGeoManager* geom)
              :fStep(0.),
               fSafety(0.),
               fLastSafety(0.),
               fThreadId(0),
               fLevel(0),
               fNmany(0),
               fNextDaughterIndex(-2),
               fOverlapSize(1000),
               fOverlapMark(0),
               fOverlapClusters(0),
               fSearchOverlaps(kFALSE),
               fCurrentOverlapping(kFALSE),
               fStartSafe(kTRUE),
               fIsEntering(kFALSE),
               fIsExiting(kFALSE),
               fIsStepEntering(kFALSE),
               fIsStepExiting(kFALSE),
               fIsOutside(kFALSE),
               fIsOnBoundary(kFALSE),
               fIsSameLocation(kTRUE),
               fIsNullStep(kFALSE),
               fGeometry(geom),
               fCache(0),
               fCurrentVolume(0),
               fCurrentNode(0),
               fTopNode(0),
               fLastNode(0),
               fNextNode(0),
               fForcedNode(0),
               fBackupState(0),
               fCurrentMatrix(0),
               fGlobalMatrix(0),
               fDivMatrix(0),
               fPath()
                
{
// Default constructor.
   fThreadId = gGeoManager->ThreadId();
   printf("Navigator: threadId=%d\n", fThreadId);
   for (Int_t i=0; i<3; i++) {
      fNormal[i] = 0.;
      fCldir[i] = 0.;
      fCldirChecked[i] = 0;
      fPoint[i] = 0.;
      fDirection[i] = 0.;
      fLastPoint[i] = 0.;
   }
   fCurrentMatrix = new TGeoHMatrix();
   fCurrentMatrix->RegisterYourself();
   fDivMatrix = new TGeoHMatrix();
   fDivMatrix->RegisterYourself();
   fOverlapClusters = new Int_t[fOverlapSize];
   ResetAll();
}      

//_____________________________________________________________________________
TGeoNavigator::TGeoNavigator(const TGeoNavigator& gm)
              :TObject(gm),
               fStep(gm.fStep),
               fSafety(gm.fSafety),
               fLastSafety(gm.fLastSafety),
               fThreadId(0),
               fLevel(gm.fLevel),
               fNmany(gm.fNmany),
               fNextDaughterIndex(gm.fNextDaughterIndex),
               fOverlapSize(gm.fOverlapSize),
               fOverlapMark(gm.fOverlapMark),
               fOverlapClusters(gm.fOverlapClusters),
               fSearchOverlaps(gm.fSearchOverlaps),
               fCurrentOverlapping(gm.fCurrentOverlapping),
               fStartSafe(gm.fStartSafe),
               fIsEntering(gm.fIsEntering),
               fIsExiting(gm.fIsExiting),
               fIsStepEntering(gm.fIsStepEntering),
               fIsStepExiting(gm.fIsStepExiting),
               fIsOutside(gm.fIsOutside),
               fIsOnBoundary(gm.fIsOnBoundary),
               fIsSameLocation(gm.fIsSameLocation),
               fIsNullStep(gm.fIsNullStep),
               fGeometry(gm.fGeometry),
               fCache(gm.fCache),
               fCurrentVolume(gm.fCurrentVolume),
               fCurrentNode(gm.fCurrentNode),
               fTopNode(gm.fTopNode),
               fLastNode(gm.fLastNode),
               fNextNode(gm.fNextNode),
               fForcedNode(gm.fForcedNode),
               fBackupState(gm.fBackupState),
               fCurrentMatrix(gm.fCurrentMatrix),
               fGlobalMatrix(gm.fGlobalMatrix),
               fPath(gm.fPath)               
{
// Copy constructor.
   fThreadId = gGeoManager->ThreadId();
   for (Int_t i=0; i<3; i++) {
      fNormal[i] = gm.fNormal[i];
      fCldir[i] = gm.fCldir[i];
      fCldirChecked[i] = gm.fCldirChecked[i];
      fPoint[i] = gm.fPoint[i];
      fDirection[i] = gm.fDirection[i];
      fLastPoint[i] = gm.fLastPoint[i];
   }
   fDivMatrix = new TGeoHMatrix();
   fDivMatrix->RegisterYourself();
}      

//_____________________________________________________________________________
TGeoNavigator& TGeoNavigator::operator=(const TGeoNavigator& gm)
{
   //assignment operator
   if(this!=&gm) {
      TObject::operator=(gm);
      fStep = gm.fStep;
      fSafety = gm.fSafety;
      fLastSafety = gm.fLastSafety;
      fThreadId = gGeoManager->ThreadId();
      fLevel = gm.fLevel;
      fNmany = gm.fNmany;
      fNextDaughterIndex = gm.fNextDaughterIndex;
      fOverlapSize=gm.fOverlapSize;
      fOverlapMark=gm.fOverlapMark;
      fOverlapClusters=gm.fOverlapClusters;
      fSearchOverlaps = gm.fSearchOverlaps;
      fCurrentOverlapping = gm.fCurrentOverlapping;
      fStartSafe = gm.fStartSafe;
      fIsEntering = gm.fIsEntering;
      fIsExiting = gm.fIsExiting;
      fIsStepEntering = gm.fIsStepEntering;
      fIsStepExiting = gm.fIsStepExiting;
      fIsOutside = gm.fIsOutside;
      fIsOnBoundary = gm.fIsOnBoundary;
      fIsSameLocation = gm.fIsSameLocation;
      fIsNullStep = gm.fIsNullStep;
      fGeometry = gm.fGeometry;
      fCache = gm.fCache;
      fCurrentVolume = gm.fCurrentVolume;
      fCurrentNode = gm.fCurrentNode;
      fTopNode = gm.fTopNode;
      fLastNode = gm.fLastNode;
      fNextNode = gm.fNextNode;
      fForcedNode = gm.fForcedNode;
      fBackupState = gm.fBackupState;
      fCurrentMatrix = gm.fCurrentMatrix;
      fGlobalMatrix = gm.fGlobalMatrix;
      fPath = gm.fPath;
      for (Int_t i=0; i<3; i++) {
         fNormal[i] = gm.fNormal[i];
         fCldir[i] = gm.fCldir[i];
         fCldirChecked[i] = gm.fCldirChecked[i];
         fPoint[i] = gm.fPoint[i];
         fDirection[i] = gm.fDirection[i];
         fLastPoint[i] = gm.fLastPoint[i];
      }
      fDivMatrix = new TGeoHMatrix();
      fDivMatrix->RegisterYourself();
   }
   return *this;   
}

//_____________________________________________________________________________
TGeoNavigator::~TGeoNavigator()
{
// Destructor.
   if (fCache) delete fCache;
   if (fBackupState) delete fBackupState;
   if (fOverlapClusters) delete [] fOverlapClusters;
}
   
//_____________________________________________________________________________
void TGeoNavigator::BuildCache(Bool_t /*dummy*/, Bool_t nodeid)
{
// Builds the cache for physical nodes and global matrices.
   static Bool_t first = kTRUE;
   Int_t nlevel = fGeometry->GetMaxLevel();
   if (nlevel<=0) nlevel = 100;
   if (!fCache) {
      if (nlevel==100) {
         if (first) Info("BuildCache","--- Maximum geometry depth set to 100");
      } else {
         if (first) Info("BuildCache","--- Maximum geometry depth is %i", nlevel);   
      }   
      // build cache
      fCache = new TGeoNodeCache(fGeometry->GetTopNode(), nodeid, nlevel+1);
      fGlobalMatrix = fCache->GetCurrentMatrix();
      fBackupState = new TGeoCacheState(nlevel+1);
   }
   first = kFALSE;
}

//_____________________________________________________________________________
Bool_t TGeoNavigator::cd(const char *path)
{
// Browse the tree of nodes starting from top node according to pathname.
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
      if (name==fGeometry->GetTopNode()->GetName()) {
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
Bool_t TGeoNavigator::CheckPath(const char *path) const
{
// Check if a geometry path is valid without changing the state of the navigator.
   Int_t length = strlen(path);
   if (!length) return kFALSE;
   TString spath = path;
   TGeoVolume *vol;
   TGeoNode *top = fGeometry->GetTopNode();
   // Check first occurance of a '/'
   Int_t ind1 = spath.Index("/");
   if (ind1<0) {
      // No '/' so we check directly the path against the name of the top
      if (strcmp(path,top->GetName())) return kFALSE;
      return kTRUE;
   }   
   Int_t ind2 = ind1;
   Bool_t end = kFALSE;
   if (ind1>0) ind1 = -1;   // no trailing '/'
   else ind2 = spath.Index("/", ind1+1);
   if (ind2<0) ind2 = length;
   TString name(spath(ind1+1, ind2-ind1-1));
   if (name==top->GetName()) {
      if (ind2>=length-1) return kTRUE;
      ind1 = ind2;
   } else return kFALSE;  
   TGeoNode *node = top;
   // Deeper than just top level
   while (!end) {
      ind2 = spath.Index("/", ind1+1);
      if (ind2<0) {
         ind2 = length;
         end  = kTRUE;
      }
      vol = node->GetVolume();
      name = spath(ind1+1, ind2-ind1-1);
      node = vol->GetNode(name.Data());
      if (!node) return kFALSE;
      if (ind2>=length-1) return kTRUE;
      ind1 = ind2;
   }
   return kTRUE;
}

//_____________________________________________________________________________
void TGeoNavigator::CdNode(Int_t nodeid)
{
// Change current path to point to the node having this id.
// Node id has to be in range : 0 to fNNodes-1 (no check for performance reasons)
   if (fCache) {
      fCache->CdNode(nodeid);
      fGlobalMatrix = fCache->GetCurrentMatrix();
   }   
}

//_____________________________________________________________________________
void TGeoNavigator::CdDown(Int_t index)
{
// Make a daughter of current node current. Can be called only with a valid
// daughter index (no check). Updates cache accordingly.
   if (!fCache) return;
   TGeoNode *node = fCurrentNode->GetDaughter(index);
   Bool_t is_offset = node->IsOffset();
   if (is_offset)
      node->cd();
   else
      fCurrentOverlapping = node->IsOverlapping();
   fCache->CdDown(index);
   fCurrentNode = node;
   fGlobalMatrix = fCache->GetCurrentMatrix();
   if (fCurrentOverlapping) fNmany++;
   fLevel++;
}


//_____________________________________________________________________________
void TGeoNavigator::CdUp()
{
// Go one level up in geometry. Updates cache accordingly.
// Determine the overlapping state of current node.
   if (!fLevel || !fCache) return;
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
   fGlobalMatrix = fCache->GetCurrentMatrix();
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
void TGeoNavigator::CdTop()
{
// Make top level node the current node. Updates the cache accordingly.
// Determine the overlapping state of current node.
   if (!fCache) return;
   fLevel = 0;
   fNmany = 0;
   if (fCurrentOverlapping) fLastNode = fCurrentNode;
   fCurrentNode = fGeometry->GetTopNode();
   fCache->CdTop();
   fGlobalMatrix = fCache->GetCurrentMatrix();
   fCurrentOverlapping = fCurrentNode->IsOverlapping();
   if (fCurrentOverlapping) fNmany++;
}

//_____________________________________________________________________________
void TGeoNavigator::CdNext()
{
// Do a cd to the node found next by FindNextBoundary
   if (fNextDaughterIndex == -2 || !fCache) return;
   if (fNextDaughterIndex ==  -3) {
      // Next node is a many - restore it
      DoRestoreState();
      fNextDaughterIndex = -2;
      return;
   }   
   if (fNextDaughterIndex == -1) {
      CdUp();
      while (fCurrentNode->GetVolume()->IsAssembly()) CdUp();
      fNextDaughterIndex--;
      return;
   }
   if (fCurrentNode && fNextDaughterIndex<fCurrentNode->GetNdaughters()) {
      CdDown(fNextDaughterIndex);
      Int_t nextindex = fCurrentNode->GetVolume()->GetNextNodeIndex();
      while (nextindex>=0) {
         CdDown(nextindex);
         nextindex = fCurrentNode->GetVolume()->GetNextNodeIndex();
      }   
   }
   fNextDaughterIndex = -2;
}   

//_____________________________________________________________________________
void TGeoNavigator::GetBranchNames(Int_t *names) const
{
// Fill volume names of current branch into an array.
   fCache->GetBranchNames(names);
}

//_____________________________________________________________________________
void TGeoNavigator::GetBranchNumbers(Int_t *copyNumbers, Int_t *volumeNumbers) const
{
// Fill node copy numbers of current branch into an array.
   fCache->GetBranchNumbers(copyNumbers, volumeNumbers);
}

//_____________________________________________________________________________
void TGeoNavigator::GetBranchOnlys(Int_t *isonly) const
{
// Fill node copy numbers of current branch into an array.
   fCache->GetBranchOnlys(isonly);
}

//_____________________________________________________________________________
TGeoNode *TGeoNavigator::CrossDivisionCell()
{
// Cross a division cell. Distance to exit contained in fStep, current node
// points to the cell node.
   TGeoPatternFinder *finder = fCurrentNode->GetFinder();
   if (!finder) {
      Fatal("CrossDivisionCell", "Volume has no pattern finder");
      return 0;
   }
   // Mark current node and go up to the level of the divided volume
   TGeoNode *skip = fCurrentNode;
   CdUp();
   Double_t point[3], newpoint[3], dir[3];
   fGlobalMatrix->MasterToLocal(fPoint, newpoint);
   fGlobalMatrix->MasterToLocalVect(fDirection, dir);
   // Does step cross a boundary along division axis ?
   Bool_t onbound = finder->IsOnBoundary(newpoint);
   if (onbound) {
      // Work along division axis
      // Get the starting point
      point[0] = newpoint[0] - dir[0]*fStep*(1.-gTolerance);
      point[1] = newpoint[1] - dir[1]*fStep*(1.-gTolerance);
      point[2] = newpoint[2] - dir[2]*fStep*(1.-gTolerance);
      // Find which is the next crossed cell.
      finder->FindNode(point, dir);
      Int_t inext = finder->GetNext();
      if (inext<0) {
         // step fully exits the division along the division axis
         // Do step exits in a mother cell ?
         if (fCurrentNode->IsOffset()) {
            Double_t dist = fCurrentNode->GetVolume()->GetShape()->DistFromInside(point,dir,3);
            // Do step exit also from mother cell ?
            if (dist < fStep+2.*gTolerance) {
               // Step exits mother on its own division axis
               return CrossDivisionCell();
            }
            // We end up here
            return fCurrentNode;
         }   
         // Exiting in a non-divided volume
         while (fCurrentNode->GetVolume()->IsAssembly()) {
            // Move always to mother for assemblies
            skip = fCurrentNode;
            if (!fLevel) break;
            CdUp();
         } 
         return CrossBoundaryAndLocate(kFALSE, skip);
      }
      // step enters a new cell
      CdDown(inext+finder->GetDivIndex());
      skip = fCurrentNode;
      return CrossBoundaryAndLocate(kTRUE, skip);
   }
   // step exits on an axis other than the division axis -> get next slice
   if (fCurrentNode->IsOffset()) return CrossDivisionCell();
   return CrossBoundaryAndLocate(kFALSE, skip);
}
   
//_____________________________________________________________________________
TGeoNode *TGeoNavigator::CrossBoundaryAndLocate(Bool_t downwards, TGeoNode *skipnode)
{
// Cross next boundary and locate within current node
// The current point must be on the boundary of fCurrentNode.

// Extrapolate current point with estimated error.
   Double_t *tr = fGlobalMatrix->GetTranslation();
   Double_t trmax = 1.+TMath::Abs(tr[0])+TMath::Abs(tr[1])+TMath::Abs(tr[2]);
   Double_t extra = 100.*(trmax+fStep)*gTolerance;
   fPoint[0] += extra*fDirection[0];
   fPoint[1] += extra*fDirection[1];
   fPoint[2] += extra*fDirection[2];
   TGeoNode *current = SearchNode(downwards, skipnode);
   fForcedNode = 0;
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
         return fGeometry->GetCurrentNode();
      }
      CdUp();
      while (fLevel && fCurrentNode->GetVolume()->IsAssembly()) CdUp();
      if (!fLevel && fCurrentNode->GetVolume()->IsAssembly()) {
         fIsOutside = kTRUE;
         return fCurrentNode;
      }
      return fCurrentNode;
   }
   return current;
}   
   
//_____________________________________________________________________________
TGeoNode *TGeoNavigator::FindNextBoundary(Double_t stepmax, const char *path, Bool_t frombdr)
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
   Int_t iact = 3;
   Int_t idebug = TGeoManager::GetVerboseLevel();
   fNextDaughterIndex = -2;
   fStep = TGeoShape::Big();
   fIsStepEntering = kFALSE;
   fIsStepExiting = kFALSE;
   fForcedNode = 0;
   Bool_t computeGlobal = kFALSE;
   fIsOnBoundary = frombdr;
   fSafety = 0.;
   TGeoNode *top_node = fGeometry->GetTopNode();
   TGeoVolume *top_volume = top_node->GetVolume();
   if (stepmax<1E29) {
      if (stepmax <= 0) {
         stepmax = - stepmax;
         computeGlobal = kTRUE;
      }
      // Try to get out easy if proposed step within safe region
      if (!frombdr && IsSafeStep(stepmax+gTolerance, fSafety)) {
         fStep = stepmax;
         fNextNode = fCurrentNode;
         return fCurrentNode;
      }   
//      if (fLastSafety>0 && IsSamePoint(fPoint[0], fPoint[1], fPoint[2])) fSafety = fLastSafety;
      fSafety = Safety();
      fSafety = TMath::Abs(fSafety);
      memcpy(fLastPoint, fPoint, kN3);
      fLastSafety = fSafety;
      if (fSafety<gTolerance) fIsOnBoundary = kTRUE;
      else fIsOnBoundary = kFALSE;
      fStep = stepmax;
      if (stepmax+gTolerance<fSafety) {
         fNextNode = fCurrentNode;
         return fCurrentNode;
      }   
   }
   if (computeGlobal) fCurrentMatrix->CopyFrom(fGlobalMatrix);
   Double_t snext  = TGeoShape::Big();
   Double_t safe;
   Double_t point[3];
   Double_t dir[3];
   if (idebug>4) {
      printf("TGeoManager::FindNextBoundary:  point=(%19.16f, %19.16f, %19.16f)\n",
             fPoint[0],fPoint[1],fPoint[2]);
      printf("                                dir=  (%19.16f, %19.16f, %19.16f)\n",
             fDirection[0], fDirection[1], fDirection[2]);
      printf("  pstep=%9.6g  path=%s\n", stepmax, GetPath());
   }   
   if (strlen(path)) {
      PushPath();
      if (!cd(path)) {
         PopPath();
         return 0;
      }
      if (computeGlobal) fCurrentMatrix->CopyFrom(fGlobalMatrix);
      fNextNode = fCurrentNode;
      TGeoVolume *tvol=fCurrentNode->GetVolume();
      fGlobalMatrix->MasterToLocal(fPoint, &point[0]);
      fGlobalMatrix->MasterToLocalVect(fDirection, &dir[0]);
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
      snext = top_volume->GetShape()->DistFromOutside(fPoint, fDirection, iact, fStep, &safe);
      fNextNode = top_node;
      if (snext < fStep-gTolerance) {
         fIsStepEntering = kTRUE;
         fStep = snext;
         Int_t indnext = fNextNode->GetVolume()->GetNextNodeIndex();
         fNextDaughterIndex = indnext;
         while (indnext>=0) {
            fNextNode = fNextNode->GetDaughter(indnext);
            if (computeGlobal) fCurrentMatrix->Multiply(fNextNode->GetMatrix());
            indnext = fNextNode->GetVolume()->GetNextNodeIndex();
         }
         return fNextNode;
      }
      return 0;
   }
   fGlobalMatrix->MasterToLocal(fPoint, &point[0]);
   fGlobalMatrix->MasterToLocalVect(fDirection, &dir[0]);
   TGeoVolume *vol = fCurrentNode->GetVolume();
   if (idebug>4) {
      printf("   -> from local=(%19.16f, %19.16f, %19.16f)\n",
             point[0],point[1],point[2]);
      printf("           ldir =(%19.16f, %19.16f, %19.16f)\n",
             dir[0],dir[1],dir[2]);             
   }           
   // find distance to exiting current node
   snext = vol->GetShape()->DistFromInside(&point[0], &dir[0], iact, fStep, &safe);
   if (idebug>4) {
      printf("      exiting %s shape %s at snext=%g\n", vol->GetName(), vol->GetShape()->ClassName(),snext);
   }   
   if (snext < fStep-gTolerance) {
      fNextNode = fCurrentNode;
      fNextDaughterIndex = -1;
      fIsStepExiting  = kTRUE;
      fStep = snext;
      fIsStepEntering = kFALSE;
      if (fStep<1E-6) return fCurrentNode;
   }
   fNextNode = (fStep<1E20)?fCurrentNode:0;
   // Find next daughter boundary for the current volume
   Int_t idaughter = -1;
   FindNextDaughterBoundary(point,dir,idaughter,computeGlobal);
   if (idaughter>=0) fNextDaughterIndex = idaughter;
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
         fGlobalMatrix->MasterToLocal(fPoint, &mothpt[0]);
         fGlobalMatrix->MasterToLocalVect(fDirection, &vecpt[0]);
         // check distance to out
         snext = TGeoShape::Big();
         if (!mother->IsAssembly()) snext = mother->GetShape()->DistFromInside(&mothpt[0], &vecpt[0], iact, fStep, &safe);
         if (snext<fStep-gTolerance) {
            fIsStepExiting  = kTRUE;
            fIsStepEntering = kFALSE;
            fStep = snext;
            if (computeGlobal) fCurrentMatrix->CopyFrom(fGlobalMatrix);
            fNextNode = fCurrentNode;
            fNextDaughterIndex = -3;
            DoBackupState();
         }
         // check overlapping nodes
         for (Int_t i=0; i<novlps; i++) {
            current = mother->GetNode(ovlps[i]);
            if (!current->IsOverlapping()) {
               current->cd();
               current->MasterToLocal(&mothpt[0], &dpt[0]);
               current->MasterToLocalVect(&vecpt[0], &dvec[0]);
               // Current point may be inside the other node - geometry error that we ignore
               snext = TGeoShape::Big();
               if (!current->GetVolume()->Contains(dpt))
                  snext = current->GetVolume()->GetShape()->DistFromOutside(&dpt[0], &dvec[0], iact, fStep, &safe);
               if (snext<fStep-gTolerance) {
                  if (computeGlobal) {
                     fCurrentMatrix->CopyFrom(fGlobalMatrix);
                     fCurrentMatrix->Multiply(current->GetMatrix());
                  }
                  fIsStepExiting  = kTRUE;
                  fIsStepEntering = kFALSE;
                  fStep = snext;
                  fNextNode = current;
                  fNextDaughterIndex = -3;
                  CdDown(ovlps[i]);
                  DoBackupState();
                  CdUp();
               }
            } else {
               // another many - check if point is in or out
               current->cd();
               current->MasterToLocal(&mothpt[0], &dpt[0]);
               current->MasterToLocalVect(&vecpt[0], &dvec[0]);
               if (current->GetVolume()->Contains(dpt)) {
                  if (current->GetVolume()->GetNdaughters()) {
                     CdDown(ovlps[i]);
                     fIsStepEntering  = kFALSE;
                     fIsStepExiting  = kTRUE;
                     dnode = FindNextDaughterBoundary(dpt,dvec,idovlp,computeGlobal);
                     if (dnode) {
                        if (computeGlobal) {
                           fCurrentMatrix->CopyFrom(fGlobalMatrix);
                           fCurrentMatrix->Multiply(dnode->GetMatrix());
                        }   
                        fNextNode = dnode;
                        fNextDaughterIndex = -3;
                        CdDown(idovlp);
                        Int_t indnext = fCurrentNode->GetVolume()->GetNextNodeIndex();
                        Int_t iup=0;
                        while (indnext>=0) {
                           CdDown(indnext);
                           iup++;
                           indnext = fCurrentNode->GetVolume()->GetNextNodeIndex();
                        }   
                        DoBackupState();
                        while (iup>0) {
                           CdUp();
                           iup--;
                        }   
                        CdUp();
                     }
                     CdUp();
                  }   
               } else {
                  snext = current->GetVolume()->GetShape()->DistFromOutside(&dpt[0], &dvec[0], iact, fStep, &safe);
                  if (snext<fStep-gTolerance) {
                     if (computeGlobal) {
                        fCurrentMatrix->CopyFrom(fGlobalMatrix);
                        fCurrentMatrix->Multiply(current->GetMatrix());
                     }
                     fIsStepExiting  = kTRUE;
                     fIsStepEntering = kFALSE;
                     fStep = snext;
                     fNextNode = current;
                     fNextDaughterIndex = -3;
                     CdDown(ovlps[i]);
                     DoBackupState();
                     CdUp();
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
         TGeoNode *currentnode = fCurrentNode;
         TGeoNode *mothernode, *mup;
         TGeoHMatrix *matrix;
         while (nmany) {
            mothernode = GetMother(up);
            mup = mothernode;
            imother = up+1;
            offset = kFALSE;
            while (mup->IsOffset()) {
               mup = GetMother(imother++);
               offset = kTRUE;
            }   
            nextovlp = mup->IsOverlapping();
            if (offset) {
               mothernode = mup;
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
               if (!mothernode->GetVolume()->IsAssembly()) snext = mothernode->GetVolume()->GetShape()->DistFromInside(dpt,dvec,iact,fStep);
               if (snext<fStep-gTolerance) {
                  fIsStepEntering  = kFALSE;
                  fIsStepExiting  = kTRUE;
                  fStep = snext;
                  fNextNode = mothernode;
                  fNextDaughterIndex = -3;
                  if (computeGlobal) fCurrentMatrix->CopyFrom(matrix);
                  while (up--) CdUp();
                  DoBackupState();
                  up = 1;
                  currentnode = fCurrentNode;
                  ovlp = currentnode->IsOverlapping();
                  continue;
               }   
            }   
            currentnode = mothernode;
            ovlp = nextovlp;
            up++;            
         }
      }      
      PopPath();
   }
   return fNextNode;
}

//_____________________________________________________________________________
TGeoNode *TGeoNavigator::FindNextDaughterBoundary(Double_t *point, Double_t *dir, Int_t &idaughter, Bool_t compmatrix)
{
// Computes as fStep the distance to next daughter of the current volume. 
// The point and direction must be converted in the coordinate system of the current volume.
// The proposed step limit is fStep.

   Double_t snext = TGeoShape::Big();
   Int_t idebug = TGeoManager::GetVerboseLevel();
   idaughter = -1; // nothing crossed
   TGeoNode *nodefound = 0;
   // Get number of daughters. If no daughters we are done.

   TGeoVolume *vol = fCurrentNode->GetVolume();
   Int_t nd = vol->GetNdaughters();
   if (!nd) return 0;  // No daughter 
   if (fGeometry->IsActivityEnabled() && !vol->IsActiveDaughters()) return 0;
   Double_t lpoint[3], ldir[3];
   TGeoNode *current = 0;
   Int_t i=0;
   // if current volume is divided, we are in the non-divided region. We
   // check first if we are inside a cell in which case compute distance to next cell
   TGeoPatternFinder *finder = vol->GetFinder();
   if (finder) {
      Int_t ifirst = finder->GetDivIndex();
      Int_t ilast = ifirst+finder->GetNdiv()-1;
      current = finder->FindNode(point);
      if (current) {
         // Point inside a cell: find distance to next cell
         Int_t index = current->GetIndex();
         if ((index-1) >= ifirst) ifirst = index-1;
         else                     ifirst = -1;
         if ((index+1) <= ilast)  ilast  = index+1;
         else                     ilast  = -1;
      }
      if (ifirst>=0) {   
         current = vol->GetNode(ifirst);
         current->cd();
         current->MasterToLocal(&point[0], lpoint);
         current->MasterToLocalVect(&dir[0], ldir);
         snext = current->GetVolume()->GetShape()->DistFromOutside(lpoint, ldir, 3, fStep);
         if (snext<fStep-gTolerance) {
            if (compmatrix) {
               fCurrentMatrix->CopyFrom(fGlobalMatrix);
               fCurrentMatrix->Multiply(current->GetMatrix());
            }
            fIsStepExiting  = kFALSE;
            fIsStepEntering = kTRUE;
            fStep=snext;
            fNextNode = current;
            nodefound = current;
            idaughter = ifirst;
         }
      }   
      if (ilast==ifirst) return nodefound;
      if (ilast>=0) { 
         current = vol->GetNode(ilast);
         current->cd();
         current->MasterToLocal(&point[0], lpoint);
         current->MasterToLocalVect(&dir[0], ldir);
         snext = current->GetVolume()->GetShape()->DistFromOutside(lpoint, ldir, 3, fStep);
         if (snext<fStep-gTolerance) {
            if (compmatrix) {
               fCurrentMatrix->CopyFrom(fGlobalMatrix);
               fCurrentMatrix->Multiply(current->GetMatrix());
            }
            fIsStepExiting  = kFALSE;
            fIsStepEntering = kTRUE;
            fStep=snext;
            fNextNode = current;
            nodefound = current;
            idaughter = ilast;
         }
      }   
      return nodefound;
   }
   // if only few daughters, check all and exit
   TGeoVoxelFinder *voxels = vol->GetVoxels();
   Int_t indnext;
   if (idebug>4) printf("   Checking distance to %d daughters...\n",nd);
   if (nd<5 || !voxels) {
      for (i=0; i<nd; i++) {
         current = vol->GetNode(i);
         if (fGeometry->IsActivityEnabled() && !current->GetVolume()->IsActive()) continue;
         current->cd();
         if (voxels && voxels->IsSafeVoxel(point, i, fStep)) continue;
         current->MasterToLocal(point, lpoint);
         current->MasterToLocalVect(dir, ldir);
         if (current->IsOverlapping() && current->GetVolume()->Contains(lpoint)) continue;
         snext = current->GetVolume()->GetShape()->DistFromOutside(lpoint, ldir, 3, fStep);
         if (snext<fStep-gTolerance) {
            if (idebug>4) {
               printf("   -> from local=(%19.16f, %19.16f, %19.16f)\n",
                      lpoint[0],lpoint[1],lpoint[2]);
               printf("           ldir =(%19.16f, %19.16f, %19.16f)\n",
                      ldir[0],ldir[1],ldir[2]);             
               printf("   -> to: %s shape %s snext=%g\n", current->GetName(),
                      current->GetVolume()->GetShape()->ClassName(), snext);        
            }           
            indnext = current->GetVolume()->GetNextNodeIndex();
            if (compmatrix) {
               fCurrentMatrix->CopyFrom(fGlobalMatrix);
               fCurrentMatrix->Multiply(current->GetMatrix());
            }    
            fIsStepExiting  = kFALSE;
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
      if (vol->IsAssembly()) ((TGeoVolumeAssembly*)vol)->SetNextNodeIndex(idaughter);
      return nodefound;
   }
   // if current volume is voxelized, first get current voxel
   Int_t ncheck = 0;
   Int_t sumchecked = 0;
   Int_t *vlist = 0;
   voxels->SortCrossedVoxels(point, dir, fThreadId);
   while ((sumchecked<nd) && (vlist=voxels->GetNextVoxel(point, dir, ncheck, fThreadId))) {
      for (i=0; i<ncheck; i++) {
         current = vol->GetNode(vlist[i]);
         if (fGeometry->IsActivityEnabled() && !current->GetVolume()->IsActive()) continue;
         current->cd();
         current->MasterToLocal(point, lpoint);
         current->MasterToLocalVect(dir, ldir);
         if (current->IsOverlapping() && current->GetVolume()->Contains(lpoint)) continue;
         snext = current->GetVolume()->GetShape()->DistFromOutside(lpoint, ldir, 3, fStep);
         sumchecked++;
//         printf("checked %d from %d : snext=%g\n", sumchecked, nd, snext);
         if (snext<fStep-gTolerance) {
            if (idebug>4) {
               printf("   -> from local=(%19.16f, %19.16f, %19.16f)\n",
                      lpoint[0],lpoint[1],lpoint[2]);
               printf("           ldir =(%19.16f, %19.16f, %19.16f)\n",
                      ldir[0],ldir[1],ldir[2]);             
               printf("   -> to: %s shape %s snext=%g\n", current->GetName(),
                      current->GetVolume()->GetShape()->ClassName(), snext);        
            }           
            indnext = current->GetVolume()->GetNextNodeIndex();
            if (compmatrix) {
               fCurrentMatrix->CopyFrom(fGlobalMatrix);
               fCurrentMatrix->Multiply(current->GetMatrix());
            }
            fIsStepExiting  = kFALSE;
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
   if (vol->IsAssembly()) ((TGeoVolumeAssembly*)vol)->SetNextNodeIndex(idaughter);
   return nodefound;
}

//_____________________________________________________________________________
TGeoNode *TGeoNavigator::FindNextBoundaryAndStep(Double_t stepmax, Bool_t compsafe)
{
// Compute distance to next boundary within STEPMAX. If no boundary is found,
// propagate current point along current direction with fStep=STEPMAX. Otherwise
// propagate with fStep=SNEXT (distance to boundary) and locate/return the next 
// node.
   static Int_t icount = 0;
   icount++;
   Int_t iact = 3;
   Int_t idebug = TGeoManager::GetVerboseLevel();
   Int_t nextindex;
   Bool_t is_assembly;
   fForcedNode = 0;
   fIsStepExiting  = kFALSE;
   TGeoNode *skip;
   fIsStepEntering = kFALSE;
   fStep = stepmax;
   Double_t snext = TGeoShape::Big();
   if (compsafe) {
      // Try to get out easy if proposed step within safe region
      fIsOnBoundary = kFALSE;
      if (IsSafeStep(stepmax+gTolerance, fSafety)) {
         fPoint[0] += stepmax*fDirection[0];
         fPoint[1] += stepmax*fDirection[1];
         fPoint[2] += stepmax*fDirection[2];
         return fCurrentNode;
      }   
      Safety();
      fLastSafety = fSafety;
      memcpy(fLastPoint, fPoint, kN3);
      // If proposed step less than safety, nothing to check
      if (fSafety > stepmax+gTolerance) {
         fPoint[0] += stepmax*fDirection[0];
         fPoint[1] += stepmax*fDirection[1];
         fPoint[2] += stepmax*fDirection[2];
         return fCurrentNode;
      }   
   }   
   Double_t extra = (fIsOnBoundary)?gTolerance:0.0;
   fIsOnBoundary = kFALSE;
   fPoint[0] += extra*fDirection[0];
   fPoint[1] += extra*fDirection[1];
   fPoint[2] += extra*fDirection[2];
   fCurrentMatrix->CopyFrom(fGlobalMatrix);
   if (idebug>4) {
      printf("TGeoManager::FindNextBAndStep:  point=(%19.16f, %19.16f, %19.16f)\n",
             fPoint[0],fPoint[1],fPoint[2]);
      printf("                                dir=  (%19.16f, %19.16f, %19.16f)\n",
             fDirection[0], fDirection[1], fDirection[2]);
      printf("  pstep=%9.6g  path=%s\n", stepmax, GetPath());
   }   
   
   if (fIsOutside) {
      snext = fGeometry->GetTopVolume()->GetShape()->DistFromOutside(fPoint, fDirection, iact, fStep);
      if (snext < fStep-gTolerance) {
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
         fNextNode = fGeometry->GetTopNode();
         nextindex = fNextNode->GetVolume()->GetNextNodeIndex();
         while (nextindex>=0) {
            CdDown(nextindex);
            fNextNode = fCurrentNode;
            nextindex = fNextNode->GetVolume()->GetNextNodeIndex();
            if (nextindex<0) fCurrentMatrix->CopyFrom(fGlobalMatrix);
         }   
         // Update global point
         fPoint[0] += snext*fDirection[0];
         fPoint[1] += snext*fDirection[1];
         fPoint[2] += snext*fDirection[2];
         fIsOnBoundary = kTRUE;
         fIsOutside = kFALSE;
         fForcedNode = fCurrentNode;
         return CrossBoundaryAndLocate(kTRUE, fCurrentNode);
      }
      if (snext<TGeoShape::Big()) {
         // New point still outside, but the top node is reachable
         fNextNode = fGeometry->GetTopNode();
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
   fGlobalMatrix->MasterToLocal(fPoint, &point[0]);
   fGlobalMatrix->MasterToLocalVect(fDirection, &dir[0]);
   TGeoVolume *vol = fCurrentNode->GetVolume();
   // find distance to exiting current node
   if (idebug>4) {
      printf("   -> from local=(%19.16f, %19.16f, %19.16f)\n",
             point[0],point[1],point[2]);
      printf("           ldir =(%19.16f, %19.16f, %19.16f)\n",
             dir[0],dir[1],dir[2]);             
   }           
   // find distance to exiting current node
   snext = vol->GetShape()->DistFromInside(point, dir, iact, fStep);
   if (idebug>4) {
      printf("      exiting %s shape %s at snext=%g\n", vol->GetName(), vol->GetShape()->ClassName(),snext);
   }   
   fNextNode = fCurrentNode;
   if (snext <= gTolerance) {
      // Current point on the boundary while track exiting
      snext = gTolerance;
      fStep = snext;
      fIsOnBoundary = kTRUE;
      fIsStepEntering = kFALSE;
      fIsStepExiting = kTRUE;
      skip = fCurrentNode;
      fPoint[0] += fStep*fDirection[0];
      fPoint[1] += fStep*fDirection[1];
      fPoint[2] += fStep*fDirection[2];
      is_assembly = fCurrentNode->GetVolume()->IsAssembly();
      if (!fLevel && !is_assembly) {
         fIsOutside = kTRUE;
         return 0;
      }
      if (fCurrentNode->IsOffset()) return CrossDivisionCell();   
      if (fLevel) CdUp();
      else        skip = 0;
      return CrossBoundaryAndLocate(kFALSE, skip);
   }   

   if (snext < fStep-gTolerance) {
      // Currently the minimum step chosen is the exiting one
      icrossed = -1;
      fStep = snext;
      fIsStepEntering = kFALSE;
      fIsStepExiting = kTRUE;
   }
   // Find next daughter boundary for the current volume
   Int_t idaughter = -1;
   TGeoNode *crossed = FindNextDaughterBoundary(point,dir, idaughter, kTRUE);
   if (crossed) {
      fIsStepExiting = kFALSE;
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
         fGlobalMatrix->MasterToLocal(fPoint, &mothpt[0]);
         fGlobalMatrix->MasterToLocalVect(fDirection, &vecpt[0]);
         // check distance to out
         snext = TGeoShape::Big();
         if (!mother->IsAssembly()) snext = mother->GetShape()->DistFromInside(mothpt, vecpt, iact, fStep);
         if (snext<fStep-gTolerance) {
            // exiting mother first (extrusion)
            icrossed = -1;
            PopDummy();
            PushPath(safelevel+1);
            fIsStepEntering = kFALSE;
            fIsStepExiting = kTRUE;
            fStep = snext;
            fCurrentMatrix->CopyFrom(fGlobalMatrix);
            fNextNode = fCurrentNode;
         }
         // check overlapping nodes
         for (Int_t i=0; i<novlps; i++) {
            current = mother->GetNode(ovlps[i]);
            if (!current->IsOverlapping()) {
               current->cd();
               current->MasterToLocal(&mothpt[0], &dpt[0]);
               current->MasterToLocalVect(&vecpt[0], &dvec[0]);
               // Current point may be inside the other node - geometry error that we ignore
               snext = TGeoShape::Big();
               if (!current->GetVolume()->Contains(dpt))
                  snext = current->GetVolume()->GetShape()->DistFromOutside(dpt, dvec, iact, fStep);
               if (snext<fStep-gTolerance) {
                  PopDummy();
                  PushPath(safelevel+1);
                  fCurrentMatrix->CopyFrom(fGlobalMatrix);
                  fCurrentMatrix->Multiply(current->GetMatrix());
                  fIsStepEntering = kFALSE;
                  fIsStepExiting = kTRUE;
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
                        fCurrentMatrix->CopyFrom(fGlobalMatrix);
                        fCurrentMatrix->Multiply(dnode->GetMatrix());
                        icrossed = idaughter;
                        PopDummy();
                        PushPath(safelevel+1);
                        fIsStepEntering = kFALSE;
                        fIsStepExiting = kTRUE;
                        fNextNode = dnode;
                     }   
                     CdUp();
                  }   
               } else {
                  snext = current->GetVolume()->GetShape()->DistFromOutside(dpt, dvec, iact, fStep);
                  if (snext<fStep-gTolerance) {
                     fCurrentMatrix->CopyFrom(fGlobalMatrix);
                     fCurrentMatrix->Multiply(current->GetMatrix());
                     fIsStepEntering = kFALSE;
                     fIsStepExiting = kTRUE;
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
         TGeoNode *currentnode = fCurrentNode;
         TGeoNode *mothernode, *mup;
         TGeoHMatrix *matrix;
         while (nmany) {
            mothernode = GetMother(up);
            mup = mothernode;
            imother = up+1;
            offset = kFALSE;
            while (mup->IsOffset()) {
               mup = GetMother(imother++);
               offset = kTRUE;
            }   
            nextovlp = mup->IsOverlapping();
            if (offset) {
               mothernode = mup;
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
               if (!mothernode->GetVolume()->IsAssembly()) snext = mothernode->GetVolume()->GetShape()->DistFromInside(dpt,dvec,iact,fStep);
                  fIsStepEntering = kFALSE;
                  fIsStepExiting  = kTRUE;
               if (snext<fStep-gTolerance) {
                  fNextNode = mothernode;
                  fCurrentMatrix->CopyFrom(matrix);
                  fStep = snext;
                  while (up--) CdUp();
                  PopDummy();
                  PushPath();
                  icrossed = -1;
                  up = 1;
                  currentnode = fCurrentNode;
                  ovlp = currentnode->IsOverlapping();
                  continue;
               }   
            }   
            currentnode = mothernode;
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
      fIsOnBoundary = kFALSE;
      return fCurrentNode;
   }
   fIsOnBoundary = kTRUE;
   if (icrossed == -1) {
      // Exiting current node.
      skip = fCurrentNode;
      is_assembly = fCurrentNode->GetVolume()->IsAssembly();
      if (!fLevel && !is_assembly) {
         fIsOutside = kTRUE;
         return 0;
      }   
      if (fCurrentNode->IsOffset()) return CrossDivisionCell();   
      if (fLevel) CdUp();
      else        skip = 0;
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
   fForcedNode = fCurrentNode;
   return CrossBoundaryAndLocate(kTRUE, current);
}   

//_____________________________________________________________________________
TGeoNode *TGeoNavigator::FindNode(Bool_t safe_start)
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
TGeoNode *TGeoNavigator::FindNode(Double_t x, Double_t y, Double_t z)
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
Double_t *TGeoNavigator::FindNormalFast()
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
Double_t *TGeoNavigator::FindNormal(Bool_t /*forward*/)
{
// Computes normal vector to the next surface that will be or was already
// crossed when propagating on a straight line from a given point/direction.
// Returns the normal vector cosines in the MASTER coordinate system. The dot
// product of the normal and the current direction is positive defined.
   return FindNormalFast();
}

//_____________________________________________________________________________
TGeoNode *TGeoNavigator::InitTrack(const Double_t *point, const Double_t *dir)
{
// Initialize current point and current direction vector (normalized)
// in MARS. Return corresponding node.
   SetCurrentPoint(point);
   SetCurrentDirection(dir);
   return FindNode();
}

//_____________________________________________________________________________
TGeoNode *TGeoNavigator::InitTrack(Double_t x, Double_t y, Double_t z, Double_t nx, Double_t ny, Double_t nz)
{
// Initialize current point and current direction vector (normalized)
// in MARS. Return corresponding node.
   SetCurrentPoint(x,y,z);
   SetCurrentDirection(nx,ny,nz);
   return FindNode();
}

//_____________________________________________________________________________
void TGeoNavigator::ResetState()
{
// Reset current state flags.
   fSearchOverlaps = kFALSE;
   fIsOutside = kFALSE;
   fIsEntering = fIsExiting = kFALSE;
   fIsOnBoundary = kFALSE;
   fIsStepEntering = fIsStepExiting = kFALSE;
}

//_____________________________________________________________________________
Double_t TGeoNavigator::Safety(Bool_t inside)
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
      fSafety = fGeometry->GetTopVolume()->GetShape()->Safety(fPoint,kFALSE);
      if (fSafety < gTolerance) {
         fSafety = 0;
         fIsOnBoundary = kTRUE;
      }   
      return fSafety;
   }
   //---> convert point to local reference frame of current node
   fGlobalMatrix->MasterToLocal(fPoint, point);

   //---> compute safety to current node
   TGeoVolume *vol = fCurrentNode->GetVolume();
   if (!inside) {
      fSafety = vol->GetShape()->Safety(point, kTRUE);
      //---> if we were just entering, return this safety
      if (fSafety < gTolerance) {
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
      if (safe < gTolerance) {
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
      if (safe < gTolerance) {
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
         if (safe < gTolerance) {
            fSafety=0;
            fIsOnBoundary = kTRUE;
            return fSafety;
         }
         if (safe<fSafety) fSafety=safe;
      }
      if (fNmany && !inside) SafetyOverlaps();
      return fSafety;
   } else {
      if (voxels->NeedRebuild()) {
         voxels->Voxelize();
         vol->FindOverlaps();
      }
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
      if (safe<gTolerance) {
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
void TGeoNavigator::SafetyOverlaps()
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
      fGeometry->MasterToLocal(fPoint, point);
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
      Bool_t crtovlp = kFALSE;
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
         if (crtovlp) nmany--;
         if (crtovlp || nextovlp) {
            matrix = GetMotherMatrix(up);
            matrix->MasterToLocal(fPoint,local);
            safe = mother->GetVolume()->GetShape()->Safety(local,kTRUE);
            if (safe<fSafety) fSafety = safe;
            current = mother;
            crtovlp = nextovlp;
         }
         up++;
      }      
   }
   PopPath();
   if (fSafety < gTolerance) {
      fSafety = 0.;
      fIsOnBoundary = kTRUE;
   }   
}

//_____________________________________________________________________________
TGeoNode *TGeoNavigator::SearchNode(Bool_t downwards, const TGeoNode *skipnode)
{
// Returns the deepest node containing fPoint, which must be set a priori.
   Double_t point[3];
   fNextDaughterIndex = -2;
   TGeoVolume *vol = 0;
   Int_t idebug = TGeoManager::GetVerboseLevel();
   Bool_t inside_current = (fCurrentNode==skipnode)?kTRUE:kFALSE;
   if (!downwards) {
   // we are looking upwards until inside current node or exit
      if (fGeometry->IsActivityEnabled() && !fCurrentNode->GetVolume()->IsActive()) {
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
         fGlobalMatrix->MasterToLocal(fPoint, point);
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
   fGlobalMatrix->MasterToLocal(fPoint, point);
   if (!inside_current && downwards) {
   // we are looking downwards
      if (fCurrentNode == fForcedNode) inside_current = kTRUE;
      else inside_current = vol->Contains(point);
      if (!inside_current) {
         fIsSameLocation = kFALSE;
         return 0;
      } else {
         if (fIsOutside) {
            fIsOutside = kFALSE;
            fIsSameLocation = kFALSE;
         }
         if (idebug>4) {
            printf("Search node local=(%19.16f, %19.16f, %19.16f) -> %s\n",
                   point[0],point[1],point[2], fCurrentNode->GetName());
         }                    
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
      // Make sure we did not end up in an assembly.
      CdDown(crtindex);
      vol = fCurrentNode->GetVolume();
      crtindex = vol->GetCurrentNodeIndex();
      if (crtindex<0) fGlobalMatrix->MasterToLocal(fPoint, point);
   }   
      
   Int_t nd = vol->GetNdaughters();
   // in case there are no daughters
   if (!nd) return fCurrentNode;
   if (fGeometry->IsActivityEnabled() && !vol->IsActiveDaughters()) return fCurrentNode;

   TGeoPatternFinder *finder = vol->GetFinder();
   // point is inside the current node
   // first check if inside a division
   if (finder) {
      node=finder->FindNode(point);
      if (!node && fForcedNode) {
         // Point *HAS* to be inside a cell
         Double_t dir[3];
         fGlobalMatrix->MasterToLocalVect(fDirection, dir);
         finder->FindNode(point,dir);
         node = finder->CdNext();
         if (!node) return fCurrentNode;  // inside divided volume but not in a cell
      }   
      if (node && node!=skipnode) {
         // go inside the division cell and search downwards
         fIsSameLocation = kFALSE;
         CdDown(node->GetIndex());
         fForcedNode = 0;
         return SearchNode(kTRUE, node);
      }
      // point is not inside the division, but might be in other nodes
      // at the same level (NOT SUPPORTED YET)
      while (fCurrentNode && fCurrentNode->IsOffset()) CdUp();
      return fCurrentNode;
   }
   // second, look if current volume is voxelized
   TGeoVoxelFinder *voxels = vol->GetVoxels();
   Int_t *check_list = 0;
   Int_t id;
   if (voxels) {
      // get the list of nodes passing thorough the current voxel
      check_list = voxels->GetCheckList(&point[0], ncheck, fThreadId);
      // if none in voxel, see if this is the last one
      if (!check_list) {
         if (!fCurrentNode->GetVolume()->IsAssembly()) return fCurrentNode;
         // Point in assembly - go up
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
         if (fGeometry->IsActivityEnabled() && !node->GetVolume()->IsActive()) continue;
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
         fForcedNode = 0;
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
      if (fGeometry->IsActivityEnabled() && !node->GetVolume()->IsActive()) continue;
      CdDown(id);
      fForcedNode = 0;
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
TGeoNode *TGeoNavigator::FindInCluster(Int_t *cluster, Int_t nc)
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
      Bool_t max_priority = (clnode==fNextNode)?kTRUE:kFALSE;
      found = SearchNode(kTRUE, clnode);
      if (!fSearchOverlaps || max_priority) {
      // an only was found during the search -> exiting
      // The node given by FindNextBoundary returned -> exiting
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
Int_t TGeoNavigator::GetTouchedCluster(Int_t start, Double_t *point,
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
            if (fGeometry->IsActivityEnabled() && !current->GetVolume()->IsActive()) continue;
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
TGeoNode *TGeoNavigator::Step(Bool_t is_geom, Bool_t cross)
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
Int_t TGeoNavigator::GetVirtualLevel()
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
Bool_t TGeoNavigator::GotoSafeLevel()
{
// Go upwards the tree until a non-overlaping node
   while (fCurrentOverlapping && fLevel) CdUp();
   Double_t point[3];
   fGlobalMatrix->MasterToLocal(fPoint, point);
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
Int_t TGeoNavigator::GetSafeLevel() const
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
void TGeoNavigator::InspectState() const
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
Bool_t TGeoNavigator::IsSameLocation(Double_t x, Double_t y, Double_t z, Bool_t change)
{
// Checks if point (x,y,z) is still in the current node.
   // check if this is an overlapping node
   Double_t oldpt[3];
   if (fLastSafety>0) {
      Double_t dx = (x-fLastPoint[0]);
      Double_t dy = (y-fLastPoint[1]);
      Double_t dz = (z-fLastPoint[2]);
      Double_t dsq = dx*dx+dy*dy+dz*dz;
      if (dsq<fLastSafety*fLastSafety) {
         if (change) {
            fPoint[0] = x;
            fPoint[1] = y;
            fPoint[2] = z;
            memcpy(fLastPoint, fPoint, 3*sizeof(Double_t));
            fLastSafety -= TMath::Sqrt(dsq);
         }
         return kTRUE;
      }  
      if (change) fLastSafety = 0;
   }
   if (fCurrentOverlapping) {
//      TGeoNode *current = fCurrentNode;
      Int_t cid = GetCurrentNodeId();
      if (!change) PushPoint();
      memcpy(oldpt, fPoint, kN3);
      SetCurrentPoint(x,y,z);
      SearchNode();
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
   fGlobalMatrix->MasterToLocal(point,local);
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
      check_list = voxels->GetCheckList(local, ncheck, fThreadId);
      if (!check_list) return kTRUE;
      if (!change) PushPath();
      for (Int_t id=0; id<ncheck; id++) {
//         node = vol->GetNode(check_list[id]);
         CdDown(check_list[id]);
         fGlobalMatrix->MasterToLocal(point,local1);
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
   while (fCurrentNode && fCurrentNode->GetDaughter(id++)) {
      CdDown(id-1);
      fGlobalMatrix->MasterToLocal(point,local1);
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
Bool_t TGeoNavigator::IsSafeStep(Double_t proposed, Double_t &newsafety) const
{
// In case a previous safety value was computed, check if the safety region is
// still safe for the current point and proposed step. Return value changed only
// if proposed distance is safe.

   // Last safety not computed.
   if (fLastSafety < gTolerance) return kFALSE;
   // Proposed step too small
   if (proposed < gTolerance) {
      newsafety = fLastSafety - proposed;
      return kTRUE;
   }
   // Normal step
   Double_t dist = (fPoint[0]-fLastPoint[0])*(fPoint[0]-fLastPoint[0])+
                   (fPoint[1]-fLastPoint[1])*(fPoint[1]-fLastPoint[1])+
                   (fPoint[2]-fLastPoint[2])*(fPoint[2]-fLastPoint[2]);
   dist = TMath::Sqrt(dist);
   Double_t safe =  fLastSafety - dist;
   if (safe < proposed) return kFALSE;
   newsafety = safe;
   return kTRUE;
}

//_____________________________________________________________________________
Bool_t TGeoNavigator::IsSamePoint(Double_t x, Double_t y, Double_t z) const
{
// Check if a new point with given coordinates is the same as the last located one.
   if (TMath::Abs(x-fLastPoint[0]) < 1.E-20) {
      if (TMath::Abs(y-fLastPoint[1]) < 1.E-20) {
         if (TMath::Abs(z-fLastPoint[2]) < 1.E-20) return kTRUE;
      }
   }
   return kFALSE;
}
  
//_____________________________________________________________________________
void TGeoNavigator::DoBackupState()
{
// Backup the current state without affecting the cache stack.
   if (fBackupState) fBackupState->SetState(fLevel,0, fNmany, fCurrentOverlapping);
}

//_____________________________________________________________________________
void TGeoNavigator::DoRestoreState()
{
// Restore a backed-up state without affecting the cache stack.
   if (fBackupState && fCache) {
      fCurrentOverlapping = fCache->RestoreState(fNmany, fBackupState);
      fCurrentNode=fCache->GetNode();
      fGlobalMatrix = fCache->GetCurrentMatrix();
      fLevel=fCache->GetLevel();
   }   
}

//_____________________________________________________________________________
TGeoHMatrix *TGeoNavigator::GetHMatrix()
{
// Return stored current matrix (global matrix of the next touched node).
   if (!fCurrentMatrix) {
      fCurrentMatrix = new TGeoHMatrix();
      fCurrentMatrix->RegisterYourself();
   }   
   return fCurrentMatrix;
}

//_____________________________________________________________________________
const char *TGeoNavigator::GetPath() const
{
// Get path to the current node in the form /node0/node1/...
   if (fIsOutside) return kGeoOutsidePath;
   return fCache->GetPath();
}

//______________________________________________________________________________
void TGeoNavigator::MasterToTop(const Double_t *master, Double_t *top) const
{
// Convert coordinates from master volume frame to top.
   fCurrentMatrix->MasterToLocal(master, top);
}

//______________________________________________________________________________
void TGeoNavigator::TopToMaster(const Double_t *top, Double_t *master) const
{
// Convert coordinates from top volume frame to master.
   fCurrentMatrix->LocalToMaster(top, master);
}

//______________________________________________________________________________
void TGeoNavigator::ResetAll()
{
// Reset the navigator.
   GetHMatrix();
   *fCurrentMatrix = gGeoIdentity;
   fCurrentNode = fGeometry->GetTopNode();
   ResetState();
   fStep = 0.;
   fSafety = 0.;
   fLastSafety = 0.;
   fLevel = 0;
   fNmany = 0;
   fNextDaughterIndex = -2;
   fCurrentOverlapping = kFALSE;
   fStartSafe = kFALSE;
   fIsSameLocation = kFALSE;
   fIsNullStep = kFALSE;
   fCurrentVolume = fGeometry->GetTopVolume();
   fCurrentNode = fGeometry->GetTopNode();
   fLastNode = 0;
   fNextNode = 0;
   fPath = "";
   if (fCache) {
      Bool_t dummy=fCache->IsDummy();
      Bool_t nodeid = fCache->HasIdArray();
      delete fCache;
      delete fBackupState;
      fCache = 0;
      BuildCache(dummy,nodeid);
   }
}

ClassImp(TGeoNavigatorArray)

//______________________________________________________________________________
TGeoNavigator *TGeoNavigatorArray::AddNavigator()
{
// Add a new navigator to the array.
   SetOwner(kTRUE);
   TGeoNavigator *nav = new TGeoNavigator(fGeoManager);
   nav->BuildCache(kTRUE, kFALSE);
   Add(nav);
   SetCurrentNavigator(GetEntriesFast()-1);
   return nav;
}
