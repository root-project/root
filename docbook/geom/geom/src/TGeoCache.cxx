// @(#)root/geom:$Id$
// Author: Andrei Gheata   18/03/02

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////////
// Physical tree description.
//
//
//
//
//Begin_Html
/*
<img src=".gif">
*/
//End_Html

#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TGeoVolume.h"
#include "TGeoCache.h"

const Int_t kN3 = 3*sizeof(Double_t);


ClassImp(TGeoNodeCache)

/*************************************************************************
 * TGeoNodeCache - special pool of reusable nodes
 *
 *
 *************************************************************************/

//_____________________________________________________________________________
TGeoNodeCache::TGeoNodeCache()
{
// Dummy constructor
   fGeoCacheMaxLevels    = 100;
   fGeoCacheStackSize    = 1000;
   fLevel       = 0;
   fStackLevel  = 0;
   fCurrentID   = 0;
   fIndex       = 0;
   fPath        = "";
   fTop         = 0;
   fNode        = 0;
   fMatrix      = 0;
   fStack       = 0;
   fMatrixBranch = 0;
   fMPB         = 0;
   fNodeBranch  = 0;
   fNodeIdArray = 0;
   for (Int_t i=0; i<100; i++) fIdBranch[i] = 0;
}

//_____________________________________________________________________________
TGeoNodeCache::TGeoNodeCache(TGeoNode *top, Bool_t nodeid, Int_t capacity)
{
// Default constructor
   fGeoCacheMaxLevels    = capacity;
   fGeoCacheStackSize    = 1000;
   fLevel       = 0;
   fStackLevel  = 0;
   fCurrentID   = 0;
   fIndex       = 0;
   fPath        = "";
   fTop         = top;
   fNode        = top;
   fStack = new TObjArray(fGeoCacheStackSize);
   for (Int_t ist=0; ist<fGeoCacheStackSize; ist++)
      fStack->Add(new TGeoCacheState(fGeoCacheMaxLevels)); // !obsolete 100
   fMatrixBranch = new TGeoHMatrix *[fGeoCacheMaxLevels];
   fMPB = new TGeoHMatrix *[fGeoCacheMaxLevels];
   for (Int_t i=0; i<fGeoCacheMaxLevels; i++) {
      fMPB[i] = new TGeoHMatrix(TString::Format("global_%d",i));
      fMatrixBranch[i] = 0;
   }
   fMatrix = fMatrixBranch[0] = fMPB[0];
   fNodeBranch  = new TGeoNode *[fGeoCacheMaxLevels];
   fNodeBranch[0] = top;
   fNodeIdArray = 0;
   for (Int_t i=0; i<100; i++) fIdBranch[i] = 0;
   if (nodeid) BuildIdArray();
   CdTop();
}

//_____________________________________________________________________________
TGeoNodeCache::TGeoNodeCache(const TGeoNodeCache& gnc)
              :TObject(gnc),
               fGeoCacheMaxLevels(gnc.fGeoCacheMaxLevels),
               fGeoCacheStackSize(gnc.fGeoCacheStackSize),
               fLevel(gnc.fLevel),
               fStackLevel(gnc.fStackLevel),
               fCurrentID(gnc.fCurrentID),
               fIndex(gnc.fIndex),
               fPath(gnc.fPath),
               fTop(gnc.fTop),
               fNode(gnc.fNode),
               fMatrix(gnc.fMatrix),
               fStack(gnc.fStack),
               fMatrixBranch(gnc.fMatrixBranch),
               fMPB(gnc.fMPB),
               fNodeBranch(gnc.fNodeBranch),
               fNodeIdArray(gnc.fNodeIdArray)
{
   // Copy constructor
   Error("CC","Not implemented!");
}

//_____________________________________________________________________________
TGeoNodeCache& TGeoNodeCache::operator=(const TGeoNodeCache&)
{
   // Assignment operator
   Error("operator=","Assignment not allowed");
   return *this;
}

//_____________________________________________________________________________
TGeoNodeCache::~TGeoNodeCache()
{
// Destructor
   if (fStack) {
      fStack->Delete();
      delete fStack;
   }
   if (fMatrixBranch) delete [] fMatrixBranch;
   if (fMPB) {
      for (Int_t i=0; i<fGeoCacheMaxLevels; i++) delete fMPB[i];
      delete [] fMPB;
   }
   if (fNodeBranch)   delete [] fNodeBranch;
   if (fNodeIdArray)  delete [] fNodeIdArray;
}

//_____________________________________________________________________________
void TGeoNodeCache::BuildIdArray()
{
// Builds node id array.
   Int_t nnodes = gGeoManager->GetNNodes();
   //if (nnodes>3E7) return;
   if (fNodeIdArray) delete [] fNodeIdArray;
   Info("BuildIDArray","--- node ID tracking enabled, size=%lu Bytes\n", ULong_t((2*nnodes+1)*sizeof(Int_t)));
   fNodeIdArray = new Int_t[2*nnodes+1];
   fNodeIdArray[0] = 0;
   Int_t ifree  = 1;
   Int_t nodeid = 0;
   gGeoManager->GetTopNode()->FillIdArray(ifree, nodeid, fNodeIdArray);
   fIdBranch[0] = 0;
}

//_____________________________________________________________________________
void TGeoNodeCache::CdNode(Int_t nodeid) {
// Change current path to point to the node having this id.
// Node id has to be in range : 0 to fNNodes-1 (no check for performance reasons)
   if (!fNodeIdArray) {
      printf("WARNING:CdNode() disabled - too many nodes\n");
      return;
   }
   Int_t *arr = fNodeIdArray;
   if (nodeid == arr[fIndex]) return;
   while (fLevel>0) {
      gGeoManager->CdUp();
      if (nodeid == arr[fIndex]) return;
   }
   gGeoManager->CdTop();
   Int_t currentID = 0;
   Int_t nd = GetNode()->GetNdaughters();
   Int_t nabove, nbelow, middle;
   while (nodeid!=currentID && nd) {
      nabove = nd+1;
      nbelow = 0;
      while (nabove-nbelow > 1) {
         middle = (nabove+nbelow)>>1;
         currentID = arr[arr[fIndex+middle]];
         if (nodeid == currentID) {
            gGeoManager->CdDown(middle-1);
            return;
         }
         if (nodeid < currentID) nabove = middle;
         else                    nbelow = middle;
      }
      gGeoManager->CdDown(nbelow-1);
      currentID = arr[fIndex];
      nd = GetNode()->GetNdaughters();
   }
}

//_____________________________________________________________________________
Bool_t TGeoNodeCache::CdDown(Int_t index)
{
// Make daughter INDEX of current node the active state. Compute global matrix.
   TGeoNode *newnode = fNode->GetDaughter(index);
   if (!newnode) return kFALSE;
   fLevel++;
   if (fNodeIdArray) {
      fIndex = fNodeIdArray[fIndex+index+1];
      fIdBranch[fLevel] = fIndex;
   }
   fNode = newnode;
   fNodeBranch[fLevel] = fNode;
   TGeoMatrix  *local = newnode->GetMatrix();
   TGeoHMatrix *newmat = fMPB[fLevel];
   if (!local->IsIdentity()) {
      newmat->CopyFrom(fMatrix);
      newmat->Multiply(local);
      fMatrix = newmat;
   }
   fMatrixBranch[fLevel] = fMatrix;
   return kTRUE;
}

//_____________________________________________________________________________
void TGeoNodeCache::CdUp()
{
// Make mother of current node the active state.
   if (!fLevel) return;
   fLevel--;
   if (fNodeIdArray) fIndex = fIdBranch[fLevel];
   fNode = fNodeBranch[fLevel];
   fMatrix = fMatrixBranch[fLevel];
}

//_____________________________________________________________________________
Int_t TGeoNodeCache::GetCurrentNodeId() const
{
// Returns a fixed ID for current physical node
   if (fNodeIdArray) return fNodeIdArray[fIndex];
   return GetNodeId();
}

//_____________________________________________________________________________
Int_t TGeoNodeCache::GetNodeId() const
{
// Get unique node id.
   Long_t id=0;
   for (Int_t level=0;level<fLevel+1; level++)
      id += (Long_t)fNodeBranch[level];
   return (Int_t)id;
}

//_____________________________________________________________________________
void TGeoNodeCache::GetBranchNames(Int_t *names) const
{
// Fill names with current branch volume names (4 char - used by GEANT3 interface).
   const char *name;
   for (Int_t i=0; i<fLevel+1; i++) {
      name = fNodeBranch[i]->GetVolume()->GetName();
      memcpy(&names[i], name, sizeof(Int_t));
   }
}

//_____________________________________________________________________________
void TGeoNodeCache::GetBranchNumbers(Int_t *copyNumbers, Int_t *volumeNumbers) const
{
// Fill copy numbers of current branch nodes.
   for (Int_t i=0; i<fLevel+1; i++) {
      copyNumbers[i]   = fNodeBranch[i]->GetNumber();
      volumeNumbers[i] = fNodeBranch[i]->GetVolume()->GetNumber();
   }
}

//_____________________________________________________________________________
void TGeoNodeCache::GetBranchOnlys(Int_t *isonly) const
{
// Fill copy numbers of current branch nodes.
   Bool_t ismany = kFALSE;
   for (Int_t i=0; i<fLevel+1; i++) {
      if (!fNodeBranch[i]->IsOffset()) ismany=fNodeBranch[i]->IsOverlapping();
      isonly[i] = (ismany)?0:1;
   }
}

//_____________________________________________________________________________
const char *TGeoNodeCache::GetPath()
{
// Returns the current geometry path.
   fPath = "";
   for (Int_t level=0;level<fLevel+1; level++) {
      fPath += "/";
      fPath += fNodeBranch[level]->GetName();
   }
   return fPath.Data();
}

//_____________________________________________________________________________
Int_t TGeoNodeCache::PushState(Bool_t ovlp, Int_t startlevel, Int_t nmany, Double_t *point)
{
// Push current state into heap.
   if (fStackLevel>=fGeoCacheStackSize) {
      printf("ERROR TGeoNodeCach::PushSate() : stack of states full\n");
      return 0;
   }
   ((TGeoCacheState*)fStack->At(fStackLevel))->SetState(fLevel,startlevel,nmany,ovlp,point);
   return ++fStackLevel;
}

//_____________________________________________________________________________
Bool_t TGeoNodeCache::PopState(Int_t &nmany, Double_t *point)
{
// Pop next state/point from heap.
   if (!fStackLevel) return 0;
   Bool_t ovlp = ((TGeoCacheState*)fStack->At(--fStackLevel))->GetState(fLevel,nmany,point);
   Refresh();
//   return (fStackLevel+1);
   return ovlp;
}

//_____________________________________________________________________________
Bool_t TGeoNodeCache::PopState(Int_t &nmany, Int_t level, Double_t *point)
{
// Pop next state/point from heap and restore matrices starting from LEVEL.
   if (level<=0) return 0;
   Bool_t ovlp = ((TGeoCacheState*)fStack->At(level-1))->GetState(fLevel,nmany,point);
   Refresh();
   return ovlp;
}

//_____________________________________________________________________________
Bool_t TGeoNodeCache::RestoreState(Int_t &nmany, TGeoCacheState *state, Double_t *point)
{
// Pop next state/point from a backed-up state.
   Bool_t ovlp = state->GetState(fLevel,nmany,point);
   Refresh();
   return ovlp;
}

//_____________________________________________________________________________
void TGeoNodeCache::LocalToMaster(const Double_t *local, Double_t *master) const
{
// Local point converted to master frame defined by current matrix.
   fMatrix->LocalToMaster(local, master);
}

//_____________________________________________________________________________
void TGeoNodeCache::MasterToLocal(const Double_t *master, Double_t *local) const
{
// Point in master frame defined by current matrix converted to local one.
   fMatrix->MasterToLocal(master, local);
}

//_____________________________________________________________________________
void TGeoNodeCache::LocalToMasterVect(const Double_t *local, Double_t *master) const
{
// Local vector converted to master frame defined by current matrix.
   fMatrix->LocalToMasterVect(local, master);
}

//_____________________________________________________________________________
void TGeoNodeCache::MasterToLocalVect(const Double_t *master, Double_t *local) const
{
// Vector in master frame defined by current matrix converted to local one.
   fMatrix->MasterToLocalVect(master,local);
}

//_____________________________________________________________________________
void TGeoNodeCache::LocalToMasterBomb(const Double_t *local, Double_t *master) const
{
// Local point converted to master frame defined by current matrix and rescaled with bomb factor.
   fMatrix->LocalToMasterBomb(local, master);
}

//_____________________________________________________________________________
void TGeoNodeCache::MasterToLocalBomb(const Double_t *master, Double_t *local) const
{
// Point in master frame defined by current matrix converted to local one and rescaled with bomb factor.
   fMatrix->MasterToLocalBomb(master, local);
}

ClassImp(TGeoCacheState)

/*************************************************************************
* TGeoCacheState - class storing the state of the cache at a given moment
*
*
*************************************************************************/

//_____________________________________________________________________________
TGeoCacheState::TGeoCacheState()
{
// Default ctor.
   fCapacity = 0;
   fLevel = 0;
   fNmany = 0;
   fStart = 0;
   memset(fIdBranch, 0, 30*sizeof(Int_t));
   memset(fPoint, 0, 3*sizeof(Int_t));
   fOverlapping = kFALSE;
   fNodeBranch = 0;
   fMatrixBranch = 0;
   fMatPtr = 0;
}

//_____________________________________________________________________________
TGeoCacheState::TGeoCacheState(Int_t capacity)
{
// Ctor.
   fCapacity = capacity;
   fLevel = 0;
   fNmany = 0;
   fStart = 0;
   memset(fIdBranch, 0, 30*sizeof(Int_t));
   memset(fPoint, 0, 3*sizeof(Int_t));
   fOverlapping = kFALSE;
   fNodeBranch = new TGeoNode *[capacity];
   fMatrixBranch = new TGeoHMatrix *[capacity];
   fMatPtr = new TGeoHMatrix *[capacity];
   for (Int_t i=0; i<capacity; i++)
      fMatrixBranch[i] = new TGeoHMatrix("global");
}

//_____________________________________________________________________________
TGeoCacheState::TGeoCacheState(const TGeoCacheState& gcs) :
  TObject(gcs),
  fCapacity(gcs.fCapacity),
  fLevel(gcs.fLevel),
  fNmany(gcs.fNmany),
  fStart(gcs.fStart),
  fOverlapping(gcs.fOverlapping)
{
   //copy constructor
   Int_t i;
   for (i=0; i<3; i++) fPoint[i]=gcs.fPoint[i];
   for(i=0; i<30; i++) fIdBranch[i]=gcs.fIdBranch[i];
   fNodeBranch = new TGeoNode *[fCapacity];
   fMatrixBranch = new TGeoHMatrix *[fCapacity];
   fMatPtr = new TGeoHMatrix *[fCapacity];
   for (i=0; i<fCapacity; i++) {
      fNodeBranch[i] = gcs.fNodeBranch[i];
      fMatrixBranch[i] = new TGeoHMatrix(*gcs.fMatrixBranch[i]);
      fMatPtr[i] = gcs.fMatPtr[i];
   }
}

//_____________________________________________________________________________
TGeoCacheState& TGeoCacheState::operator=(const TGeoCacheState& gcs)
{
   //assignment operator
   Int_t i;
   if(this!=&gcs) {
      TObject::operator=(gcs);
      fCapacity=gcs.fCapacity;
      fLevel=gcs.fLevel;
      fNmany=gcs.fNmany;
      fStart=gcs.fStart;
      for(i=0; i<30; i++) fIdBranch[i]=gcs.fIdBranch[i];
      for(i=0; i<3; i++) fPoint[i]=gcs.fPoint[i];
      fOverlapping=gcs.fOverlapping;
      fNodeBranch = new TGeoNode *[fCapacity];
      fMatrixBranch = new TGeoHMatrix *[fCapacity];
      fMatPtr = new TGeoHMatrix *[fCapacity];
      for (i=0; i<fCapacity; i++) {
         fNodeBranch[i] = gcs.fNodeBranch[i];
         fMatrixBranch[i] = new TGeoHMatrix(*gcs.fMatrixBranch[i]);
         fMatPtr[i] = gcs.fMatPtr[i];
      }
   }
   return *this;
}

//_____________________________________________________________________________
TGeoCacheState::~TGeoCacheState()
{
// Dtor.
   if (fNodeBranch) {
      delete [] fNodeBranch;
      for (Int_t i=0; i<fCapacity; i++)
         delete fMatrixBranch[i];
      delete [] fMatrixBranch;
      delete [] fMatPtr;
   }
}

//_____________________________________________________________________________
void TGeoCacheState::SetState(Int_t level, Int_t startlevel, Int_t nmany, Bool_t ovlp, Double_t *point)
{
// Fill current modeller state.
   fLevel = level;
   fStart = startlevel;
   fNmany = nmany;
   TGeoNodeCache *cache = gGeoManager->GetCache();
   if (cache->HasIdArray()) memcpy(fIdBranch, cache->GetIdBranch()+fStart, (level+1-fStart)*sizeof(Int_t));
   TGeoNode **node_branch = (TGeoNode **) cache->GetBranch();
   TGeoHMatrix **mat_branch  = (TGeoHMatrix **) cache->GetMatrices();

   memcpy(fNodeBranch, node_branch+fStart, (level+1-fStart)*sizeof(TGeoNode *));
   memcpy(fMatPtr, mat_branch+fStart, (level+1-fStart)*sizeof(TGeoHMatrix *));
   TGeoHMatrix *last = 0;
   TGeoHMatrix *current;
   for (Int_t i=0; i<level+1-fStart; i++) {
      current = mat_branch[i+fStart];
      if (current == last) continue;
      *fMatrixBranch[i] = current;
      last = current;
   }
   fOverlapping = ovlp;
   if (point) memcpy(fPoint, point, 3*sizeof(Double_t));
}

//_____________________________________________________________________________
Bool_t TGeoCacheState::GetState(Int_t &level, Int_t &nmany, Double_t *point) const
{
// Restore a modeler state.
   level = fLevel;
   nmany = fNmany;
   TGeoNodeCache *cache = gGeoManager->GetCache();
   if (cache->HasIdArray()) cache->FillIdBranch(fIdBranch, fStart);
   TGeoNode **node_branch = (TGeoNode **) cache->GetBranch();
   TGeoHMatrix **mat_branch  = (TGeoHMatrix **) cache->GetMatrices();

   memcpy(node_branch+fStart, fNodeBranch, (level+1-fStart)*sizeof(TGeoNode *));
   memcpy(mat_branch+fStart, fMatPtr, (level+1-fStart)*sizeof(TGeoHMatrix *));
   TGeoHMatrix *last = 0;
   TGeoHMatrix *current;
   for (Int_t i=0; i<level+1-fStart; i++) {
      current = mat_branch[i+fStart];
      if (current == last) continue;
      *current = fMatrixBranch[i];
      last = current;
   }
   if (point) memcpy(point, fPoint, 3*sizeof(Double_t));
   return fOverlapping;
}
