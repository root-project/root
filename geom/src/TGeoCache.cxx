// @(#)root/geom:$Name:  $:$Id: TGeoCache.cxx,v 1.45 2006/07/09 05:27:53 brun Exp $
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
#include "TBits.h"

#include "TGeoManager.h"
#include "TGeoShape.h"
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
// dummy constructor
   fGeoCacheMaxDaughters = 128;
   fGeoCacheMaxSize      = 1000000;
   fGeoCacheStackSize    = 1000;
   fGeoCacheDefaultLevel = 4;
   fGeoCacheMaxLevels    = 30;
   fGeoCacheObjArrayInd  = 0xFF;
   fGeoCacheUsageRatio   = 0.01;
   fSize        = 0;
   fNused       = 0;
   fLevel       = 0;
   fStackLevel  = 0;
   fStack       = 0;
   fDefaultLevel= 0;
   fCache       = 0;
   fPath        = "";
   fTopNode     = 0;
   fCurrentNode = 0;
   fCurrentCache = 0;
   fCurrentIndex = 0;
   fCurrentID   = 0;
   fBranch      = 0;
   fMatrices    = 0;
   fGlobalMatrix= 0;
   fMatrixPool  = 0;
   fNodeIdArray = 0;
   fIndex = 0;
//   BuildIdArray();
}

//_____________________________________________________________________________
TGeoNodeCache::TGeoNodeCache(Bool_t nodeid)
{
// dummy constructor
   fGeoCacheMaxDaughters = 128;
   fGeoCacheMaxSize      = 1000000;
   fGeoCacheStackSize    = 1000;
   fGeoCacheDefaultLevel = 4;
   fGeoCacheMaxLevels    = 30;
   fGeoCacheObjArrayInd  = 0xFF;
   fGeoCacheUsageRatio   = 0.01;
   fSize        = 0;
   fNused       = 0;
   fLevel       = 0;
   fStackLevel  = 0;
   fStack       = 0;
   fDefaultLevel= 0;
   fCache       = 0;
   fPath        = "";
   fTopNode     = 0;
   fCurrentNode = 0;
   fCurrentCache = 0;
   fCurrentIndex = 0;
   fCurrentID   = 0;
   fBranch      = 0;
   fMatrices    = 0;
   fGlobalMatrix= 0;
   fMatrixPool  = 0;
   fNodeIdArray = 0;
   fIndex = 0;
   if (nodeid) BuildIdArray();
}

//_____________________________________________________________________________
TGeoNodeCache::TGeoNodeCache(Int_t size, Bool_t nodeid)
{
// constructor
   fGeoCacheMaxDaughters = 128;
   fGeoCacheMaxSize      = 1000000;
   fGeoCacheStackSize    = 1000;
   fGeoCacheDefaultLevel = 4;
   fGeoCacheMaxLevels    = size;
   fGeoCacheObjArrayInd  = 0xFF;
   fGeoCacheUsageRatio   = 0.01;
   gGeoManager->SetCache(this);
   fDefaultLevel = fGeoCacheDefaultLevel;
   fSize = 0;
   fNused = 0;
   fLevel = 0;
   fStackLevel = 0;
   fCache = new TGeoNodeArray *[256];
   memset(fCache, 0, 0xFF*sizeof(TGeoNodeArray*));
   for (Int_t ic=0; ic<fGeoCacheMaxDaughters+1; ic++) {
      fCache[ic] = new TGeoNodeArray(ic);
      fSize += fCache[ic]->GetSize();
   }
   fCache[fGeoCacheObjArrayInd] = new TGeoNodeObjArray(0);
   fSize += fCache[fGeoCacheObjArrayInd]->GetSize();

   fPath = "";
   fPath.Resize(400);
   fCurrentCache = gGeoManager->GetTopNode()->GetNdaughters();
   if (fCurrentCache>fGeoCacheMaxDaughters)
      fCurrentCache = fGeoCacheObjArrayInd;
   fBranch = new Int_t[fGeoCacheMaxLevels];
   memset(fBranch, 0, fGeoCacheMaxLevels*sizeof(Int_t));
   memset(fIdBranch, 0, fGeoCacheMaxLevels*sizeof(Int_t));
   fMatrices = new Int_t[fGeoCacheMaxLevels];
   memset(fMatrices, 0, fGeoCacheMaxLevels*sizeof(Int_t));
   fGlobalMatrix = new TGeoHMatrix("current_global");
   fTopNode = AddNode(gGeoManager->GetTopNode());
   fCurrentNode = fTopNode;
   fBranch[0] = fTopNode;
   fCache[fCurrentCache]->SetPersistency();
   fStack = new TObjArray(fGeoCacheStackSize);
   for (Int_t ist=0; ist<fGeoCacheStackSize; ist++)
      fStack->Add(new TGeoCacheState(fGeoCacheMaxLevels)); // !obsolete 100
   printf("### nodes stored in cache %i ###\n", fSize);
   fMatrixPool = new TGeoMatrixCache(0);
   fCurrentID   = 0;
   fNodeIdArray = 0;
   fIndex = 0;
   if (nodeid) BuildIdArray();
   else        printf("--- node ID tracking disabled\n");
   CdTop();
}

//_____________________________________________________________________________
TGeoNodeCache::TGeoNodeCache(const TGeoNodeCache& gnc) :
  fGeoCacheUsageRatio(gnc.fGeoCacheUsageRatio),
  fGeoCacheMaxDaughters(gnc.fGeoCacheMaxDaughters),
  fGeoCacheMaxSize(gnc.fGeoCacheMaxSize),
  fGeoCacheDefaultLevel(gnc.fGeoCacheDefaultLevel),
  fGeoCacheMaxLevels(gnc.fGeoCacheMaxLevels),
  fGeoCacheObjArrayInd(gnc.fGeoCacheObjArrayInd),
  fGeoCacheStackSize(gnc.fGeoCacheStackSize),
  fLevel(gnc.fLevel),
  fCurrentID(gnc.fCurrentID),
  fPath(gnc.fPath),
  fStack(gnc.fStack),
  fNodeIdArray(gnc.fNodeIdArray),
  fIndex(gnc.fIndex),
  fSize(gnc.fSize),
  fNused(gnc.fNused),
  fDefaultLevel(gnc.fDefaultLevel),
  fTopNode(gnc.fTopNode),
  fCount(gnc.fCount),
  fCountLimit(gnc.fCountLimit),
  fCurrentNode(gnc.fCurrentNode),
  fCurrentCache(gnc.fCurrentCache),
  fCurrentIndex(gnc.fCurrentIndex),
  fBranch(gnc.fBranch),
  fMatrices(gnc.fMatrices),
  fStackLevel(gnc.fStackLevel),
  fGlobalMatrix(gnc.fGlobalMatrix),
  fCache(gnc.fCache),
  fMatrixPool(gnc.fMatrixPool)
{
   //copy constructor
   for(Int_t i=0; i<30; i++) fIdBranch[i]=gnc.fIdBranch[i];
}

//_____________________________________________________________________________
TGeoNodeCache& TGeoNodeCache::operator=(const TGeoNodeCache& gnc) 
{
   //assignment operator
   if(this!=&gnc) {
      fGeoCacheUsageRatio=gnc.fGeoCacheUsageRatio;
      fGeoCacheMaxDaughters=gnc.fGeoCacheMaxDaughters;
      fGeoCacheMaxSize=gnc.fGeoCacheMaxSize;
      fGeoCacheDefaultLevel=gnc.fGeoCacheDefaultLevel;
      fGeoCacheMaxLevels=gnc.fGeoCacheMaxLevels;
      fGeoCacheObjArrayInd=gnc.fGeoCacheObjArrayInd;
      fGeoCacheStackSize=gnc.fGeoCacheStackSize;
      fLevel=gnc.fLevel;
      fCurrentID=gnc.fCurrentID;
      fPath=gnc.fPath;
      fStack=gnc.fStack;
      fNodeIdArray=gnc.fNodeIdArray;
      fIndex=gnc.fIndex;
      fSize=gnc.fSize;
      fNused=gnc.fNused;
      fDefaultLevel=gnc.fDefaultLevel;
      fTopNode=gnc.fTopNode;
      fCount=gnc.fCount;
      fCountLimit=gnc.fCountLimit;
      fCurrentNode=gnc.fCurrentNode;
      fCurrentCache=gnc.fCurrentCache;
      fCurrentIndex=gnc.fCurrentIndex;
      fBranch=gnc.fBranch;
      fMatrices=gnc.fMatrices;
      fStackLevel=gnc.fStackLevel;
      fGlobalMatrix=gnc.fGlobalMatrix;
      fCache=gnc.fCache;
      fMatrixPool=gnc.fMatrixPool;
      for(Int_t i=0; i<30; i++) fIdBranch[i]=gnc.fIdBranch[i];
   } 
   return *this;
}

//_____________________________________________________________________________
TGeoNodeCache::~TGeoNodeCache()
{
// destructor
   if (fCache) {
      DeleteCaches();
      delete [] fBranch;
      delete [] fMatrices;
      delete fGlobalMatrix;
      delete fMatrixPool;
   }
   if (fStack) {
      fStack->Delete();
      delete fStack;
   }
   if (fNodeIdArray) delete [] fNodeIdArray;
}

//_____________________________________________________________________________
void TGeoNodeCache::BuildIdArray()
{
// Builds node id array.
   Int_t nnodes = gGeoManager->GetNNodes();
   //if (nnodes>3E7) return;
   if (fNodeIdArray) delete [] fNodeIdArray;
   printf("--- node ID tracking enabled, size=%d Bytes\n", (Int_t)((2*nnodes+1)*sizeof(Int_t)));
   fNodeIdArray = new Int_t[2*nnodes+1];
   fNodeIdArray[0] = 0;
   Int_t ifree  = 1;
   Int_t nodeid = 0;
   gGeoManager->GetTopNode()->FillIdArray(ifree, nodeid, fNodeIdArray);
   fIdBranch[0] = 0;
}

//_____________________________________________________________________________
Int_t TGeoNodeCache::GetCurrentNodeId() const
{
// Returns a fixed ID for current physical node
   if (fNodeIdArray) return fNodeIdArray[fIndex];
   return GetNodeId();
}

//_____________________________________________________________________________
void TGeoNodeCache::Compact()
{
// Compact arrays
   Int_t old_size, new_size;
   for (Int_t ic=0; ic<fGeoCacheMaxDaughters+1; ic++) {
      old_size = fCache[ic]->GetSize();
      fCache[ic]->Compact();
      new_size = fCache[ic]->GetSize();
      fSize -= (old_size-new_size);
   }
}

//_____________________________________________________________________________
void TGeoNodeCache::DeleteCaches()
{
// Delete all node caches.
   if (!fCache) return;
   for (Int_t ic=0; ic<fGeoCacheMaxDaughters+1; ic++) {
      fCache[ic]->DeleteArray();
      delete fCache[ic];
   }
   delete fCache[fGeoCacheObjArrayInd];
   delete [] fCache;
}

//_____________________________________________________________________________
Int_t TGeoNodeCache::AddNode(TGeoNode *node)
{
// Add a logical node in the cache corresponding to ndaughters.
   Int_t ic = node->GetNdaughters();
   if (ic > fGeoCacheMaxDaughters) ic = fGeoCacheObjArrayInd;
   return fCache[ic]->AddNode(node);
   //fNused++;
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
//   printf("%s nd=%d current=%d fIndex=%d\n", GetNode()->GetName(),nd,currentID,fIndex);
   Int_t nabove, nbelow, middle;
   while (nodeid!=currentID && nd) {
      nabove = nd+1;
      nbelow = 0;
//      for (Int_t j=0; j<nd; j++) printf("   %d id=%d\n", j, arr[fIndex+j+1]);
      while (nabove-nbelow > 1) {
         middle = (nabove+nbelow)>>1;
         currentID = arr[arr[fIndex+middle]];
//         printf("   nabove=%d nbelow=%d, middle=%d cid=%d\n", nabove,nbelow,middle,currentID);
         if (nodeid == currentID) {
            gGeoManager->CdDown(middle-1);
//            printf("final id : %d\n", nodeid);
            return;
         }
         if (nodeid < currentID) nabove = middle;
         else                    nbelow = middle;
      }
      gGeoManager->CdDown(nbelow-1);
      currentID = arr[fIndex];
//      printf("current=%d\n", currentID);
      nd = GetNode()->GetNdaughters();
   }
}

//_____________________________________________________________________________
Bool_t TGeoNodeCache::CdDown(Int_t index, Bool_t make)
{
// Make daughter 'index' of current node the current one.
   // first make sure that current node is also current in its cache
   fCache[fCurrentCache]->cd(fCurrentIndex);
   Int_t nind_d = fCache[fCurrentCache]->GetDaughter(index);
   Bool_t persistent = kFALSE;
   // if daughter is not stored, create it
   if (!nind_d) {
      if (!make) return kFALSE;
      TGeoNode *node = GetNode()->GetDaughter(index);
      nind_d = fCache[fCurrentCache]->AddDaughter(node, index);
      fNused++;
      if (fLevel < fGeoCacheDefaultLevel) persistent=kTRUE;
   }
   // make daughter current
   fBranch[++fLevel] = nind_d;
   if (fNodeIdArray) {
      fIndex = fNodeIdArray[fIndex+index+1];
      fIdBranch[fLevel] = fIndex;
   }
   fCurrentNode = nind_d;
   fCurrentCache = CacheId(nind_d);
   fCurrentIndex = Index(nind_d);
   fCache[fCurrentCache]->cd(fCurrentIndex);
   if (!make) return kTRUE;
   // set persistency
   if (persistent) fCache[fCurrentCache]->SetPersistency();
   // check if its global matrix is computed
   fMatrices[fLevel] = fCache[fCurrentCache]->GetMatrixInd();
   if (fMatrices[fLevel]) {
      fMatrixPool->cd(fMatrices[fLevel]);
      return kTRUE;
   }
   // compute matrix and add it to cache
   // get the local matrix
   TGeoMatrix *local = GetNode()->GetMatrix();
   if (local->IsIdentity()) {
   // just copy the matrix from fLevel-1
      fMatrices[fLevel] = fMatrices[fLevel-1];
   // bookkeep the matrix location
      fCache[fCurrentCache]->SetMatrix(fMatrices[fLevel]);
      return kTRUE;
   }
   fMatrixPool->GetMatrix(fGlobalMatrix);
   fGlobalMatrix->Multiply(local);
   // store it in cache and bookkeep its location
   fMatrices[fLevel] = fCache[fCurrentCache]->AddMatrix(fGlobalMatrix);
   return kTRUE;
}

//_____________________________________________________________________________
void TGeoNodeCache::CdUp()
{
// Change current path to mother.
   if (!fLevel) return;
   fLevel--;
   if (fNodeIdArray) fIndex = fIdBranch[fLevel];
   fCurrentNode = fBranch[fLevel];
   fCurrentCache = CacheId(fCurrentNode);
   fCurrentIndex = Index(fCurrentNode);
   fCache[fCurrentCache]->cd(fCurrentIndex);
   fMatrixPool->cd(fMatrices[fLevel]);
}

//_____________________________________________________________________________
void TGeoNodeCache::CleanCache()
{
// Free nodes which are not persistent from cache except the current branch.
   // first compute count limit for persistency
   printf("Cleaning cache...\n");
   fCountLimit = Int_t(fGeoCacheUsageRatio*(Double_t)fCount);
   // save level and node branch
   Int_t level = fLevel;
   Int_t *branch = new Int_t[fGeoCacheMaxLevels];
   Bool_t *flags = new Bool_t[fGeoCacheMaxLevels];
   memcpy(&branch[0], fBranch, fGeoCacheMaxLevels*sizeof(Int_t));
   // mark all nodes in the current branch as not-dumpable
   Int_t i;
   for (i=0; i<level+1; i++) {
      Int_t ic = CacheId(branch[i]);
      Int_t index = Index(branch[i]);
      fCache[ic]->cd(index);
      flags[i] = fCache[ic]->IsPersistent();
      fCache[ic]->SetPersistency();
   }
   // now go to top level and dump nodes
   CdTop();
   DumpNodes();
   // copy back the current branch
   memcpy(fBranch, &branch[0], fGeoCacheMaxLevels*sizeof(Int_t));
   // restore persistency flags
   for (i=0; i<level+1; i++) {
      Int_t ic = CacheId(branch[i]);
      Int_t index = Index(branch[i]);
      fCache[ic]->cd(index);
      fCache[ic]->SetPersistency(flags[i]);
   }
   // restore current level
   fLevel = level;
   fCurrentNode = fBranch[fLevel];
   fCurrentCache = CacheId(fCurrentNode);
   fCurrentIndex = Index(fCurrentNode);
   Status();
   delete [] branch;
   delete [] flags;
}

//_____________________________________________________________________________
Bool_t TGeoNodeCache::DumpNodes()
{
// Dump all non-persistent branches.
   Int_t ndaughters = fCache[fCurrentCache]->GetNdaughters();
   fCache[fCurrentCache]->cd(fCurrentIndex);
   if (!SetPersistency()) return kTRUE;
   for (Int_t id=0; id<ndaughters; id++) {
      if (!CdDown(id, kFALSE)) continue;
      if (DumpNodes()) {
         CdUp();
         ClearDaughter(id);
      } else {
         CdUp();
      }
   }
   return kFALSE;
}

//_____________________________________________________________________________
void TGeoNodeCache::ClearNode(Int_t nindex)
{
// Clear only the node nindex.
   Int_t ic = CacheId(nindex);
   Int_t index = Index(nindex);
   fCache[ic]->cd(index);
   fNused--;
   fCount-=GetUsageCount();
   fCache[ic]->ClearNode();
}

//_____________________________________________________________________________
TGeoNode *TGeoNodeCache::GetMother(Int_t up) const
{
// Get mother of current logical node, <up> levels up.
   if (!fLevel || (up>fLevel)) return 0;
   Int_t inode = fBranch[fLevel-up];
   Int_t id = CacheId(inode);
   fCache[id]->cd(Index(inode));
   TGeoNode *mother = fCache[id]->GetNode();
   if (fCurrentCache == id) fCache[fCurrentCache]->cd(fCurrentIndex);
   return mother;
}

//_____________________________________________________________________________
Int_t TGeoNodeCache::GetNodeId() const
{
// Get unique node id.
   return fBranch[fLevel];
}

//_____________________________________________________________________________
const char *TGeoNodeCache::GetPath()
{
// Returns the current path.
   fPath = "";
   for (Int_t level=0;level<fLevel+1; level++) {
      Int_t nindex = fBranch[level];
      Int_t ic = CacheId(nindex);
      Int_t index = Index(nindex);
      fCache[ic]->cd(index);
      fPath += "/";
      fPath += fCache[ic]->GetNode()->GetName();
   }
   return fPath.Data();
}

//_____________________________________________________________________________
void TGeoNodeCache::PrintNode() const
{
// Print some info about current node.
   TGeoNode *node = GetNode();
   printf("***********************************************\n");
   printf(" Node : %s\n", node->GetName());
   Int_t mat = fMatrices[fLevel];
   printf("   Global matrix : %i\n", mat);
   Bool_t persistency = fCache[fCurrentCache]->IsPersistent();
   UInt_t count = fCache[fCurrentCache]->GetUsageCount();
   printf("   persistency=%i  usage=%i\n", (Int_t)persistency, count);
   Int_t nd = fCache[fCurrentCache]->GetNdaughters();
   printf("   daughters : %i from %i\n", nd, node->GetNdaughters());
   for (Int_t i=0; i<nd; i++) {
      Int_t nindex_d = fCache[fCurrentCache]->GetDaughter(i);
      if (!nindex_d) continue;
      Int_t ic = CacheId(nindex_d);
      Int_t index = Index(nindex_d);
      fCache[ic]->cd(index);
      TGeoNode *dght = fCache[ic]->GetNode();
      if (dght) printf("      %i : %s\n", i, dght->GetName());
   }
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
void TGeoNodeCache::Refresh()
{
// Refresh current state.
   if (fLevel<0) {
      gGeoManager->SetOutside();
      fLevel = 0;
   }
   fCurrentNode=fBranch[fLevel];
   fCurrentCache=CacheId(fCurrentNode);
   fCurrentIndex=Index(fCurrentNode);
   fCache[fCurrentCache]->cd(fCurrentIndex);
   fMatrixPool->cd(fMatrices[fLevel]);
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
Bool_t TGeoNodeCache::SetPersistency()
{
// Set matrix cache persistent.
   if (fCache[fCurrentCache]->IsPersistent()) return kTRUE;
   Int_t usage = GetUsageCount();
   if (usage>fCountLimit) {
      fCache[fCurrentCache]->SetPersistency();
      return kTRUE;
   }
   return kFALSE;
}

//_____________________________________________________________________________
void TGeoNodeCache::Status() const
{
// Print status of cache.
   printf("Cache status : total %i   used %i   free %i nodes\n",
          fSize, fNused, fSize-fNused);
}

//_____________________________________________________________________________


/*************************************************************************
 * TGeoCacheDummy - a dummy cache for physical nodes
 *
 *
 *************************************************************************/
ClassImp(TGeoCacheDummy)

//_____________________________________________________________________________
TGeoCacheDummy::TGeoCacheDummy()
{
// Default ctor.
   fTop = 0;
   fNode = 0;
   fNodeBranch = 0;
   fMatrixBranch = 0;
   fMPB = 0;
   fMatrix = 0;
}

//_____________________________________________________________________________
TGeoCacheDummy::TGeoCacheDummy(TGeoNode *top, Bool_t nodeid, Int_t capacity)
               :TGeoNodeCache(nodeid)
{
// Constructor specifying the top node.
   fGeoCacheMaxLevels = capacity;
   fTop = top;
   fNode = top;
   fNodeBranch = new TGeoNode *[fGeoCacheMaxLevels];
   fNodeBranch[0] = top;
   fMatrixBranch = new TGeoHMatrix *[fGeoCacheMaxLevels];
   fMPB = new TGeoHMatrix *[fGeoCacheMaxLevels];
   for (Int_t i=0; i<fGeoCacheMaxLevels; i++) {
      fMPB[i] = new TGeoHMatrix("global");
      fMatrixBranch[i] = 0;
   }
   fMatrix = fMatrixBranch[0] = fMPB[0];
   fStack = new TObjArray(fGeoCacheStackSize);
   for (Int_t ist=0; ist<fGeoCacheStackSize; ist++)
      fStack->Add(new TGeoCacheStateDummy(fGeoCacheMaxLevels)); // !obsolete 100
   fMatrixPool = 0;
}

//_____________________________________________________________________________
TGeoCacheDummy::TGeoCacheDummy(const TGeoCacheDummy& gcd) :
  TGeoNodeCache(gcd),
  fTop(gcd.fTop),
  fNode(gcd.fNode),
  fMatrix(gcd.fMatrix),
  fMatrixBranch(gcd.fMatrixBranch),
  fMPB(gcd.fMPB),
  fNodeBranch(gcd.fNodeBranch)
{ 
   //copy constructor
}

//_____________________________________________________________________________
TGeoCacheDummy& TGeoCacheDummy::operator=(const TGeoCacheDummy& gcd) 
{
   //assignment operator
   if(this!=&gcd) {
      TGeoNodeCache::operator=(gcd);
      fTop=gcd.fTop;
      fNode=gcd.fNode;
      fMatrix=gcd.fMatrix;
      fMatrixBranch=gcd.fMatrixBranch;
      fMPB=gcd.fMPB;
      fNodeBranch=gcd.fNodeBranch;
   } 
   return *this;
}

//_____________________________________________________________________________
TGeoCacheDummy::~TGeoCacheDummy()
{
// Destructor.
   if (fNodeBranch) delete [] fNodeBranch;
   if (fMPB) {
      for (Int_t i=0; i<fGeoCacheMaxLevels; i++)
         delete fMPB[i];
      delete [] fMPB;
   }
   if (fMatrixBranch) delete [] fMatrixBranch;
}

//_____________________________________________________________________________
Bool_t TGeoCacheDummy::CdDown(Int_t index, Bool_t /*make*/)
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
      *newmat = fMatrix;
      newmat->Multiply(local);
      fMatrix = newmat;
   }
   fMatrixBranch[fLevel] = fMatrix;
   return kTRUE;
}

//_____________________________________________________________________________
void TGeoCacheDummy::CdUp()
{
// Make mother of current node the active state.
   if (!fLevel) return;
   fLevel--;
   if (fNodeIdArray) fIndex = fIdBranch[fLevel];
   fNode = fNodeBranch[fLevel];
   fMatrix = fMatrixBranch[fLevel];
}

//_____________________________________________________________________________
Int_t TGeoCacheDummy::GetNodeId() const
{
// Get unique node id.
   Long_t id=0;
   for (Int_t level=0;level<fLevel+1; level++)
      id += (Long_t)fNodeBranch[level];
   return (Int_t)id;
}

//_____________________________________________________________________________
void TGeoCacheDummy::GetBranchNames(Int_t *names) const
{
// Fill names with current branch volume names (4 char - used by GEANT3 interface).
   const char *name;
   for (Int_t i=0; i<fLevel+1; i++) {
      name = fNodeBranch[i]->GetVolume()->GetName();
      memcpy(&names[i], name, sizeof(Int_t));
   }
}

//_____________________________________________________________________________
void TGeoCacheDummy::GetBranchNumbers(Int_t *copyNumbers, Int_t *volumeNumbers) const
{
// Fill copy numbers of current branch nodes.
   for (Int_t i=0; i<fLevel+1; i++) {
      copyNumbers[i]   = fNodeBranch[i]->GetNumber();
      volumeNumbers[i] = fNodeBranch[i]->GetVolume()->GetNumber();
   }
}

//_____________________________________________________________________________
void TGeoCacheDummy::GetBranchOnlys(Int_t *isonly) const
{
// Fill copy numbers of current branch nodes.
   Bool_t ismany = kFALSE;
   for (Int_t i=0; i<fLevel+1; i++) {
      if (!fNodeBranch[i]->IsOffset()) ismany=fNodeBranch[i]->IsOverlapping();
      isonly[i] = (ismany)?0:1;
   }
}

//_____________________________________________________________________________
const char *TGeoCacheDummy::GetPath()
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
void TGeoCacheDummy::LocalToMaster(const Double_t *local, Double_t *master) const
{
// Local to master point conversion.
   fMatrix->LocalToMaster(local, master);
}

//_____________________________________________________________________________
void TGeoCacheDummy::LocalToMasterVect(const Double_t *local, Double_t *master) const
{
// Local to master vector conversion.
   fMatrix->LocalToMasterVect(local, master);
}

//_____________________________________________________________________________
void TGeoCacheDummy::LocalToMasterBomb(const Double_t *local, Double_t *master) const
{
// Local to master point conversion in exploded view.
   fMatrix->LocalToMasterBomb(local, master);
}

//_____________________________________________________________________________
void TGeoCacheDummy::MasterToLocal(const Double_t *master, Double_t *local) const
{
// Master to local point conversion.
   if (fMatrix->IsIdentity()) {
      memcpy(local, master, kN3);
      return;
   }
   const Double_t *tr  = fMatrix->GetTranslation();
   const Double_t *rot = fMatrix->GetRotationMatrix();
   Double_t mt0  = master[0]-tr[0];
   Double_t mt1  = master[1]-tr[1];
   Double_t mt2  = master[2]-tr[2];
   local[0] = mt0*rot[0] + mt1*rot[3] + mt2*rot[6];
   local[1] = mt0*rot[1] + mt1*rot[4] + mt2*rot[7];
   local[2] = mt0*rot[2] + mt1*rot[5] + mt2*rot[8];
}

//_____________________________________________________________________________
void TGeoCacheDummy::MasterToLocalVect(const Double_t *master, Double_t *local) const
{
// Master to local vector conversion.
   if (fMatrix->IsIdentity()) {
      memcpy(local, master, kN3);
      return;
   }
   const Double_t *rot = fMatrix->GetRotationMatrix();
   for (Int_t i=0; i<3; i++) {
      local[i] =  master[0]*rot[i]
                + master[1]*rot[i+3]
                + master[2]*rot[i+6];
   }
}

//_____________________________________________________________________________
void TGeoCacheDummy::MasterToLocalBomb(const Double_t *master, Double_t *local) const
{
// Master to local point conversion in exploded view.
   fMatrix->MasterToLocalBomb(master, local);
}


const Int_t TGeoNodeArray::fgGeoArrayMaxSize  = 1000000;
const Int_t TGeoNodeArray::fgGeoArrayInitSize = 1000;
const Int_t TGeoNodeArray::fgGeoReleasedSpace = 1000;

/*************************************************************************
 * TGeoNodeArray - base class for physical nodes arrays
 *    The structure of a node is stored in the following way :
 *    Int_t *offset = fArray+inode*nodesize position of node 'inode' in fArray
 *      ->offset+0   - pointer to physical node : fNode-gSystem
 *
 *                            |bit7 | b6  | b5 | b4  | b3 | b2  | b1 | b0 |
 *      ->offset+1   - Byte0= |scale|rot  | Z  | Y   | X  |matrix cache id|
 *                     | B3 | B2 | B1 | - matrix index in cache
 *      ->offset+2   - B0|b7 = node persistency ; b6 = has daughters
 *                     B3|B2|B1|B0 - usage count
 *      ->offset+3+i - Byte0=daughter array index,
 *                   |B3|B2|B1| - index of daughter i
 *      Total length : nodesize = (3+fNdaughters)*sizeof(Int_t)
 *
 *************************************************************************/

ClassImp(TGeoNodeArray)

//_____________________________________________________________________________
TGeoNodeArray::TGeoNodeArray()
{
// dummy ctor
   fNdaughters = 0;
   fSize = 0;
   fNodeSize = 0;
   fFirstFree = 0;
   fCurrent = 0;
   fNused = 0;
   fOffset = 0;
   fBitsArray = 0;
   fArray = 0;
}

//_____________________________________________________________________________
TGeoNodeArray::TGeoNodeArray(Int_t ndaughters, Int_t size)
{
// default constructor
   if ((ndaughters<0) || (ndaughters>254)) return;
   fNdaughters = ndaughters;
   fSize = size;
   if ((size<fgGeoArrayInitSize) || (size>fgGeoArrayMaxSize))
      fSize = fgGeoArrayInitSize;
   // number of integers stored in a node
   fNodeSize = 3+fNdaughters;
   fFirstFree = 0;
   fCurrent = 0;
   fNused = 0;
   fOffset = 0;
   fBitsArray = new TBits(fSize);
   fArray = new Int_t[fSize*fNodeSize];
   memset(fArray, 0, fSize*fNodeSize*sizeof(Int_t));
   if (!fNdaughters) {
      // never use first location of array with no daughters
      fFirstFree = fCurrent = 1;
      fBitsArray->SetBitNumber(0);
   }
}

//_____________________________________________________________________________
TGeoNodeArray::TGeoNodeArray(const TGeoNodeArray& gna) : 
  TObject(gna),
  fNodeSize(gna.fNodeSize),
  fNdaughters(gna.fNdaughters),
  fOffset(gna.fOffset),
  fSize(gna.fSize),
  fFirstFree(gna.fFirstFree),
  fCurrent(gna.fCurrent),
  fNused(gna.fNused),
  fBitsArray(gna.fBitsArray),
  fArray(gna.fArray)
{ 
   //copy constructor
}

//_____________________________________________________________________________
TGeoNodeArray& TGeoNodeArray::operator=(const TGeoNodeArray& gna) 
{
   //assignment operator
   if(this!=&gna) {
      TObject::operator=(gna);
      fNodeSize=gna.fNodeSize;
      fNdaughters=gna.fNdaughters;
      fOffset=gna.fOffset;
      fSize=gna.fSize;
      fFirstFree=gna.fFirstFree;
      fCurrent=gna.fCurrent;
      fNused=gna.fNused;
      fBitsArray=gna.fBitsArray;
      fArray=gna.fArray;
   } 
   return *this;
}

//_____________________________________________________________________________
TGeoNodeArray::~TGeoNodeArray()
{
// destructor
}

//_____________________________________________________________________________
Int_t TGeoNodeArray::AddDaughter(TGeoNode *node, Int_t i)
{
// Add node as i-th daughter of current node of this array.
   return (fOffset[3+i]=gGeoManager->GetCache()->AddNode(node));
}

//_____________________________________________________________________________
Int_t TGeoNodeArray::AddMatrix(TGeoMatrix *global)
{
// Adds a global matrix to the current node in this array.
   return (fOffset[1]=gGeoManager->GetCache()->GetMatrixPool()->AddMatrix(global));
}

//_____________________________________________________________________________
void TGeoNodeArray::Compact()
{
// Compact the array.
   fBitsArray->Compact();
   Int_t new_size = fBitsArray->GetNbits();
   Int_t *old_array = fArray;
   fArray = new Int_t[new_size*fNodeSize];
   memcpy(fArray, old_array, new_size*fNodeSize*sizeof(Int_t));
   delete [] old_array;
   fSize = new_size;
}

//_____________________________________________________________________________
void TGeoNodeArray::DeleteArray()
{
// Deletes the array of nodes.
   if (fArray) delete [] fArray;
   fArray = 0;
   if (fBitsArray) delete fBitsArray;
   fBitsArray = 0;
}

//_____________________________________________________________________________
Int_t TGeoNodeArray::AddNode(TGeoNode *node)
{
// Add node in the node array. The number of daughters
// MUST be equal to fNdaughters (no check for speed reasons)
// It does not check if node is already in the cache. This is
// done by AddDaughter
   // first compute the offset of the first free location
   Int_t index = fFirstFree;
   Int_t current = fCurrent;
   cd(index);
   // store the pointer of the node
   memset(fOffset, 0, fNodeSize*sizeof(Int_t));
   fOffset[0] = (ULong_t)node - (ULong_t)gSystem;
   // mark the location as used and compute first free
   fBitsArray->SetBitNumber(fFirstFree);
   fFirstFree = fBitsArray->FirstNullBit(fFirstFree);
   fNused++;
   if (fFirstFree >= fSize-1) IncreaseArray();
   UChar_t *cache = (UChar_t*)&index;
   cache[3] = (UChar_t)fNdaughters;
   cd(current);
   return index;
}

//_____________________________________________________________________________
void TGeoNodeArray::ClearDaughter(Int_t ind)
{
// Clear the daughter ind from the list of the current node. Send the
// signal back to TGeoNodeCache, that proceeds with dispatching the
// clear signal for all the branch.
   Int_t nindex_d = GetDaughter(ind);
   if (!nindex_d) return;
   fOffset[3+ind] = 0;
   gGeoManager->GetCache()->ClearNode(nindex_d);
}

//_____________________________________________________________________________
void TGeoNodeArray::ClearMatrix()
{
// Clears the global matrix of this node from matrix cache.
   Int_t ind_mat = fOffset[1];
   if (ind_mat && !(GetNode()->GetMatrix()->IsIdentity()))
      gGeoManager->GetCache()->GetMatrixPool()->ClearMatrix(ind_mat);
}

//_____________________________________________________________________________
void TGeoNodeArray::ClearNode()
{
// Clear the current node. All branch from this point downwords will be deleted.
   // remember the current node
   Int_t inode = fCurrent;
   // clear the daughters
   for (Int_t ind=0; ind<fNdaughters; ind++) ClearDaughter(ind);
   cd(inode);
   // clear the global matrix from matrix cache
   ClearMatrix();
   if (fCurrent<fFirstFree) fFirstFree = fCurrent;
   fBitsArray->SetBitNumber(fCurrent, kFALSE);
   fNused--;
   // empty all locations of current node
   memset(fOffset, 0, fNodeSize*sizeof(Int_t));
}

//_____________________________________________________________________________
Bool_t TGeoNodeArray::HasDaughters() const
{
// Check if current node has daughters.
   for (Int_t ind=0; ind<fNdaughters; ind++) {
      if (fOffset[3+ind]) return kTRUE;
   }
   return kFALSE;
}

//_____________________________________________________________________________
void TGeoNodeArray::IncreaseArray()
{
// Doubles the array size unless maximum cache limit is reached or
// global cache limit is reached. In this case forces the cache
// manager to do the garbage collection.
   Int_t new_size = 2*fSize;
   Int_t free_space = gGeoManager->GetCache()->GetFreeSpace();
   if (free_space<10) {
      gGeoManager->GetCache()->CleanCache();
      return;
   }
   if (free_space<fSize) new_size = fSize+free_space;
   // Increase the cache size and the TBits size
   fBitsArray->SetBitNumber(new_size-1, kFALSE);
   Int_t *new_array = new Int_t[new_size*fNodeSize];
   memset(new_array, 0, new_size*fNodeSize*sizeof(Int_t));
   memcpy(new_array, fArray, fSize*fNodeSize*sizeof(Int_t));
   delete [] fArray;
   fArray = new_array;
   gGeoManager->GetCache()->IncreasePool(new_size-fSize);
   fSize = new_size;
}

//_____________________________________________________________________________
Bool_t TGeoNodeArray::IsPersistent() const
{
// Returns persistency flag of the node.
   return ((fOffset[2] & 0x80000000)==0)?kFALSE:kTRUE;
}

//_____________________________________________________________________________
void TGeoNodeArray::SetPersistency(Bool_t flag)
{
// Set array of nodes as persistent in memory.
   if (flag) fOffset[2] |= 0x80000000;
   else      fOffset[2] &= 0x7FFFFFFF;
}

/*************************************************************************
 * TGeoNodeObjArray - container class for nodes with more than 254
 *     daughters.
 *
 *************************************************************************/

ClassImp(TGeoNodeObjArray)

//_____________________________________________________________________________
TGeoNodeObjArray::TGeoNodeObjArray()
{
// Default ctor.
   fObjArray = 0;
   fCurrent  = 0;
   fIndex = 0;
}

//_____________________________________________________________________________
TGeoNodeObjArray::TGeoNodeObjArray(Int_t size)
{
// Constructor.
   fSize = size;
   fIndex = 0;
   if (size<TGeoNodeArray::fgGeoArrayInitSize)
      fSize = TGeoNodeArray::fgGeoArrayInitSize;
   fObjArray = new TObjArray(fSize);
   for (Int_t i=0; i<fSize; i++) fObjArray->AddAt(new TGeoNodePos(), i);
   fBitsArray  = new TBits(fSize);
   fCurrent = 0;
}

//_____________________________________________________________________________
TGeoNodeObjArray::TGeoNodeObjArray(const TGeoNodeObjArray& noa) :
  TGeoNodeArray(noa),
  fIndex(noa.fIndex),
  fObjArray(noa.fObjArray),
  fCurrent(noa.fCurrent)
{ 
   //copy constructor
}

//_____________________________________________________________________________
TGeoNodeObjArray& TGeoNodeObjArray::operator=(const TGeoNodeObjArray& noa) 
{
   //assignment operator
   if(this!=&noa) {
      TGeoNodeArray::operator=(noa);
      fIndex=noa.fIndex;
      fObjArray=noa.fObjArray;
      fCurrent=noa.fCurrent;
   } 
   return *this;
}

//_____________________________________________________________________________
TGeoNodeObjArray::~TGeoNodeObjArray()
{
// Destructor.
   if (!fObjArray) return;
   fObjArray->Delete();
   delete fObjArray;
}

//_____________________________________________________________________________
Int_t TGeoNodeObjArray::AddDaughter(TGeoNode *node, Int_t i)
{
// Add i-th daughter of current node in the array. Node must be the i'th daughter of current node (inode, fOffset)
// This is called ONLY after GetDaughter(i) returns 0
   return fCurrent->AddDaughter(i, gGeoManager->GetCache()->AddNode(node));
}

//_____________________________________________________________________________
Int_t TGeoNodeObjArray::AddNode(TGeoNode *node)
{
// Add node in the node array.
   // first map the node to the first free location which becomes current
   Int_t index = fFirstFree;
   Int_t oldindex = fIndex;
   cd(index);
   fCurrent->Map(node);
   // mark the location as used and compute first free
   fBitsArray->SetBitNumber(fFirstFree);
   fFirstFree = fBitsArray->FirstNullBit(fFirstFree);
   fNused++;
   if (fFirstFree >= fSize-1) IncreaseArray();
   UChar_t *cache = (UChar_t*)&index;
   cache[3] = 0xFF;
   cd(oldindex);
   return index;
}

//_____________________________________________________________________________
Int_t TGeoNodeObjArray::AddMatrix(TGeoMatrix *global)
{
// Store the global matrix for the current node.
   return fCurrent->AddMatrix(global);
}

//_____________________________________________________________________________
void TGeoNodeObjArray::cd(Int_t inode)
{
// make inode the current node
   fCurrent = (TGeoNodePos*)fObjArray->At(inode);
   fIndex = inode;
}

//_____________________________________________________________________________
void TGeoNodeObjArray::ClearDaughter(Int_t ind)
{
// Clear the daughter ind from the list of the current node. Send the
// signal back to TGeoNodeCache, that proceeds with dispatching the
// clear signal for all the branch.
   Int_t nindex = fCurrent->GetDaughter(ind);
   if (!nindex) return;
   fCurrent->ClearDaughter(ind);
   gGeoManager->GetCache()->ClearNode(nindex);
}

//_____________________________________________________________________________
void TGeoNodeObjArray::ClearMatrix()
{
// Clear the global matrix of this node from matrix cache.
   Int_t ind_mat = fCurrent->GetMatrixInd();
   if (ind_mat && !fCurrent->GetNode()->GetMatrix()->IsIdentity())
      gGeoManager->GetCache()->GetMatrixPool()->ClearMatrix(ind_mat);
}

//_____________________________________________________________________________
void TGeoNodeObjArray::ClearNode()
{
// Clear the current node. All branch from this point downwords will be deleted.
   Int_t inode = fIndex;
   Int_t nd = GetNdaughters();
   // clear the daughters
   for (Int_t ind=0; ind<nd; ind++) ClearDaughter(ind);
   cd(inode);
   // clear the global matrix from matrix cache
   ClearMatrix();
   if (fIndex<fFirstFree) fFirstFree = fIndex;
   fBitsArray->SetBitNumber(fIndex, kFALSE);
   fNused--;
   // mapping this node to a new logical node is the task of AddNode
}

//_____________________________________________________________________________
void TGeoNodeObjArray::IncreaseArray()
{
// Doubles the array size unless maximum cache limit is reached or
// global cache limit is reached. In this case forces the cache
// manager to do the garbage collection.

//   printf("Increasing ARRAY\n");
   Int_t new_size = 2*fSize;
   Int_t free_space = gGeoManager->GetCache()->GetFreeSpace();
   if (free_space<10) {
      gGeoManager->GetCache()->CleanCache();
      return;
   }
   if (free_space<fSize) new_size = fSize+free_space;

   // Increase the cache size and the TBits size
   fBitsArray->SetBitNumber(new_size-1, kFALSE);
   fObjArray->Expand(new_size);
   for (Int_t i=fSize; i<new_size; i++) fObjArray->AddAt(new TGeoNodePos(), i);
   gGeoManager->GetCache()->IncreasePool(new_size-fSize);
   fSize = new_size;
}



/*************************************************************************
 * TGeoNodePos - the physical geometry node with links to mother and
 *   daughters.
 *
 *************************************************************************/

const Int_t   TGeoNodePos::fgPersistentNodeMask   = 0x80000000;
const UChar_t TGeoNodePos::fgPersistentMatrixMask = 64;
const UInt_t  TGeoNodePos::fgNoMatrix = 1000000000;

ClassImp(TGeoNodePos)

//_____________________________________________________________________________
TGeoNodePos::TGeoNodePos()
{
// Default ctor.
   fNdaughters = 0;
   fDaughters = 0;
   fMatrix = 0;
   fCount = 0;
   fNode = 0;
}

//_____________________________________________________________________________
TGeoNodePos::TGeoNodePos(const TGeoNodePos& gnp) :
  TObject(gnp),
  fNdaughters(gnp.fNdaughters),
  fMatrix(gnp.fMatrix),
  fCount(gnp.fCount),
  fDaughters(gnp.fDaughters),
  fNode(gnp.fNode)
{ 
   //copy constructor
}

//_____________________________________________________________________________
TGeoNodePos& TGeoNodePos::operator=(const TGeoNodePos& gnp) 
{
   //assignment operator
   if(this!=&gnp) {
      TObject::operator=(gnp);
      fNdaughters=gnp.fNdaughters;
      fMatrix=gnp.fMatrix;
      fCount=gnp.fCount;
      fDaughters=gnp.fDaughters;
      fNode=gnp.fNode;
   } 
   return *this;
}

//_____________________________________________________________________________
TGeoNodePos::TGeoNodePos(Int_t ndaughters)
{
// Constructor with ndaughters.
   fNdaughters = ndaughters;
   if (ndaughters < 0xFF) return;
   if (ndaughters) {
      fDaughters = new Int_t[ndaughters];
      memset(fDaughters, 0, ndaughters*sizeof(Int_t));
   } else {
      fDaughters = 0;
   }
   fMatrix = 0;
   fCount = 0;
   fNode = 0;
}

//_____________________________________________________________________________
TGeoNodePos::~TGeoNodePos()
{
// Destructor. It deletes the daughters also.
   if (fDaughters) delete [] fDaughters;
}

//_____________________________________________________________________________
Int_t TGeoNodePos::AddMatrix(TGeoMatrix *global)
{
// Cache the global matrix.
   return (fMatrix=gGeoManager->GetCache()->GetMatrixPool()->AddMatrix(global));
}

//_____________________________________________________________________________
void TGeoNodePos::ClearMatrix()
{
// Clear the matrix if not used by other nodes.
   if (fMatrix && !fNode->GetMatrix()->IsIdentity()) {
      gGeoManager->GetCache()->GetMatrixPool()->ClearMatrix(fMatrix);
      fMatrix = 0;
   }
}

//_____________________________________________________________________________
Int_t TGeoNodePos::GetDaughter(Int_t ind) const
{
// Get the i-th daughter.
   if (fDaughters) return fDaughters[ind];
   return 0;
}

//_____________________________________________________________________________
Bool_t TGeoNodePos::HasDaughters() const
{
// Check if current node has daughters.
   for (Int_t i=0; i<fNdaughters; i++) {
      if (fDaughters[i]!=0) return kTRUE;
   }
   return kFALSE;
}

//_____________________________________________________________________________
void TGeoNodePos::Map(TGeoNode *node)
{
// Map this nodepos to a logical node.
   fNdaughters = node->GetNdaughters();
   if (fDaughters) delete [] fDaughters;
   fDaughters = new Int_t[fNdaughters];
   memset(fDaughters, 0, fNdaughters*sizeof(Int_t));
   fMatrix = 0;
   fCount = 0;
   fNode = node;
}

//_____________________________________________________________________________
void TGeoNodePos::SetPersistency(Bool_t flag)
{
// Set this node persistent in cache.
   if (flag) fCount |= fgPersistentNodeMask;
   else      fCount &= !fgPersistentNodeMask;
}

/*************************************************************************
 * TGeoMatrixCache - cache of global matrices
 *
 *
 *************************************************************************/

ClassImp(TGeoMatrixCache)


//_____________________________________________________________________________
TGeoMatrixCache::TGeoMatrixCache()
{
// Default ctor.
   for (Int_t i=0; i<7; i++) {
      fSize[i]  = 0;
      fCache[i] = 0;
      fFree[i]  = 0;
      fBitsArray[i]  = 0;
   }
   fGeoMinCacheSize = 1000;
   fMatrix = 0;
   fHandler = 0;
   fCacheId = 0;
   fLength = 0;
   fHandlers = 0;
}

//_____________________________________________________________________________
TGeoMatrixCache::TGeoMatrixCache(Int_t size)
{
// Constructor with cache size.
   fGeoMinCacheSize = 1000;
   Int_t length;
   for (Int_t i=0; i<7; i++) {
      if (size < fGeoMinCacheSize) {
         fSize[i] = fGeoMinCacheSize;
      } else {
         fSize[i] = size;
      }
      length = 3*(i-1);
      if (length == 0) length=2;
      if (length < 0) length=1;
      fCache[i] = new Double_t[fSize[i]*length];
      fBitsArray[i]  = new TBits(fSize[i]);
      fFree[i]  = 0;
      if (i==0) {
         fBitsArray[i]->SetBitNumber(0);
         fFree[i] = 1;
      }
   }
   fMatrix = 0;
   fHandler = 0;
   fCacheId = 0;
   fLength = 0;
   fHandlers = new TGeoMatHandler *[14];
   fHandlers[0] = new TGeoMatHandlerX();
   fHandlers[1] = new TGeoMatHandlerY();
   fHandlers[2] = new TGeoMatHandlerZ();
   fHandlers[3] = new TGeoMatHandlerXY();
   fHandlers[4] = new TGeoMatHandlerXZ();
   fHandlers[5] = new TGeoMatHandlerYZ();
   fHandlers[6] = new TGeoMatHandlerXYZ();
   fHandlers[7] = new TGeoMatHandlerRot();
   fHandlers[8] = new TGeoMatHandlerRotTr();
   fHandlers[9] = new TGeoMatHandlerScl();
   fHandlers[10] = new TGeoMatHandlerTrScl();
   fHandlers[11] = new TGeoMatHandlerRotScl();
   fHandlers[12] = new TGeoMatHandlerRotTrScl();
   fHandlers[13] = new TGeoMatHandlerId();
   printf("### matrix caches of size %i built ###\n", fSize[0]);
}

//_____________________________________________________________________________
TGeoMatrixCache::TGeoMatrixCache(const TGeoMatrixCache& gmc) :
  fMatrix(gmc.fMatrix),
  fHandler(gmc.fHandler),
  fCacheId(gmc.fCacheId),
  fLength(gmc.fLength),
  fHandlers(gmc.fHandlers),
  fGeoMinCacheSize(gmc.fGeoMinCacheSize)
{
   //copy constructor
   for(Int_t i=0; i<7; i++) {
      fSize[i]=gmc.fSize[i];
      fFree[i]=gmc.fFree[i];
      fCache[i]=gmc.fCache[i];
      fBitsArray[i]=gmc.fBitsArray[i];
   }
}

//_____________________________________________________________________________
TGeoMatrixCache& TGeoMatrixCache::operator=(const TGeoMatrixCache& gmc) 
{
   //assignment operator
   if(this!=&gmc) {
      fMatrix=gmc.fMatrix;
      fHandler=gmc.fHandler;
      fCacheId=gmc.fCacheId;
      fLength=gmc.fLength;
      for(Int_t i=0; i<7; i++) {
         fSize[i]=gmc.fSize[i];
         fFree[i]=gmc.fFree[i];
         fCache[i]=gmc.fCache[i];
         fBitsArray[i]=gmc.fBitsArray[i];
      }
      fHandlers=gmc.fHandlers;
      fGeoMinCacheSize=gmc.fGeoMinCacheSize;
   } 
   return *this;
}

//_____________________________________________________________________________
TGeoMatrixCache::~TGeoMatrixCache()
{
// Destructor.
   if (fSize[0]) {
      for (Int_t i=0; i<7; i++) {
         delete fCache[i];
         delete fBitsArray[i];
      }
      for (Int_t j=0; j<14; j++)
         delete fHandlers[j];
      delete [] fHandlers;
   }
}

//_____________________________________________________________________________
Int_t TGeoMatrixCache::AddMatrix(TGeoMatrix *matrix)
{
// Add a global matrix to the first free array of corresponding type.
   if (matrix->IsIdentity()) {fHandler=13; return (fMatrix=0);}

   const Double_t *translation = matrix->GetTranslation();

   UChar_t type = 0;
   if (matrix->IsRotation()) type |= 8;
   if (matrix->IsScale())    type |= 16;
   if (matrix->IsTranslation()) {
      if (translation[0]!=0)  type |= 1;
      if (translation[1]!=0)  type |= 2;
      if (translation[2]!=0)  type |= 4;
   }
   const Int_t cache_id[32] = {
      0, 0, 0, 1, 0, 1, 1, 2, 4, 5, 5, 5, 5, 5, 5, 5,
      2, 3, 3, 3, 3, 3, 3, 3, 5, 6, 6, 6, 6, 6, 6, 6};
   const Int_t cache_len[32] = {
      0, 1, 1, 2, 1, 2, 2, 3, 9,12,12,12,12,12,12,12,
      3, 6, 6, 6, 6, 6, 6, 6,12,15,15,15,15,15,15,15};
   const Int_t handler_id[32] = {
      0, 0, 1, 3, 2, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8,
      9,10,10,10,10,10,10,10,11,12,12,12,12,12,12,12};

   fCacheId = cache_id[type];
   fLength = cache_len[type];
   fHandler = handler_id[type];
   UInt_t current_free = fFree[fCacheId];
   Double_t *location = fCache[fCacheId]+fLength*current_free;
   TGeoMatHandler *handler = fHandlers[fHandler];
   handler->AddMatrix(location, matrix);

   fBitsArray[fCacheId]->SetBitNumber(current_free);
   fFree[fCacheId] = fBitsArray[fCacheId]->FirstNullBit(current_free);
   if (fFree[fCacheId] >= fSize[fCacheId]-1) {
      IncreaseCache();
      location = fCache[fCacheId]+fLength*current_free;
      handler->SetLocation(location);
   }
   fMatrix = current_free;
   UChar_t *type_loc = (UChar_t*)&fMatrix+3;
   *type_loc = type;
   return fMatrix;
}

//_____________________________________________________________________________
void TGeoMatrixCache::cd(Int_t mindex)
{
// Make a matrix index the current one.
   fMatrix = mindex;
   if (!fMatrix) {
      fHandler = 13;
      return;
   }
//   printf("%i\n", mindex);
   const Int_t cache_id[32] = {
      0, 0, 0, 1, 0, 1, 1, 2, 4, 5, 5, 5, 5, 5, 5, 5,
      2, 3, 3, 3, 3, 3, 3, 3, 5, 6, 6, 6, 6, 6, 6, 6};
   const Int_t cache_len[32] = {
      0, 1, 1, 2, 1, 2, 2, 3, 9,12,12,12,12,12,12,12,
      3, 6, 6, 6, 6, 6, 6, 6,12,15,15,15,15,15,15,15};
   const Int_t handler_id[32] = {
      0, 0, 1, 3, 2, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8,
      9,10,10,10,10,10,10,10,11,12,12,12,12,12,12,12};
   UChar_t *type = (UChar_t*)&mindex+3;
   fHandler = handler_id[*type];
   fCacheId = cache_id[*type];
   fLength = cache_len[*type];
   fHandlers[fHandler]->SetLocation(fCache[fCacheId]+(mindex&0x00FFFFFF)*fLength);
}

//_____________________________________________________________________________
void TGeoMatrixCache::ClearMatrix(Int_t mindex)
{
// Release the space occupied by a matrix.
   if (!mindex) return;
   cd(mindex);
   Int_t offset = fMatrix&0x00FFFFFF;
   fBitsArray[fCacheId]->SetBitNumber(offset, kFALSE);
   if (UInt_t(offset)<fFree[fCacheId]) fFree[fCacheId] = offset;
}

//_____________________________________________________________________________
void TGeoMatrixCache::GetMatrix(TGeoHMatrix *matrix) const
{
// Get a matrix from cache.
   if (!fMatrix) {
      matrix->Clear();
      return;
   }
   Int_t matptr = fMatrix & 0x00FFFFFF;
   // clear the matrix if needed
   if (!matrix->IsIdentity()) matrix->Clear();
   Double_t *new_ptr = fCache[fCacheId] + matptr*fLength;
   // ask the handler to get the matrix from cache
   fHandlers[fHandler]->GetMatrix(new_ptr, matrix);
}

//_____________________________________________________________________________
void TGeoMatrixCache::IncreaseCache()
{
// Doubles the cache size.
   UInt_t new_size = 2*fSize[fCacheId];
   fBitsArray[fCacheId]->SetBitNumber(new_size-1, kFALSE);
   Double_t *new_cache = new Double_t[new_size*fLength];
   // copy old bits to new bits and old data to new data
   memcpy(new_cache, fCache[fCacheId], fSize[fCacheId]*fLength*sizeof(Double_t));
   delete fCache[fCacheId];
   fCache[fCacheId] = new_cache;
   fSize[fCacheId] = new_size;
}

//_____________________________________________________________________________
void TGeoMatrixCache::Status() const
{
// Print current status of matrix cache.
   Int_t ntot, ntotc,ntotused, nused, nfree, length;
   printf("Matrix cache status :   total    used    free\n");
   ntot = 0;
   ntotused = 0;
   for (Int_t i=0; i<7; i++) {
      length = 3*(i-1);
      if (length == 0) length=2;
      if (length < 0) length=1;
      ntotc = fSize[i];
      nused = fBitsArray[i]->CountBits();
      nfree = ntotc-nused;
      ntot += length*sizeof(Double_t)*ntotc;
      ntotused += length*sizeof(Double_t)*nused;
      printf(" - Cache %i :         %i       %i       %i\n",length, ntotc,nused,nfree);
   }
   printf("total size : %i  bytes   used : %i bytes\n", ntot, ntotused);
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
   fBranch = 0;
   fMatrices = 0;
   fPoint = 0;
}

//_____________________________________________________________________________
TGeoCacheState::TGeoCacheState(Int_t capacity)
{
// Ctor.
   fCapacity = capacity;
   fLevel = 0;
   fNmany = 0;
   fBranch = new Int_t[capacity];
   fMatrices = new Int_t[capacity];
   fPoint = new Double_t[3];
}

//_____________________________________________________________________________
TGeoCacheState::TGeoCacheState(const TGeoCacheState& gcs) : 
  TObject(gcs),
  fCapacity(gcs.fCapacity),
  fLevel(gcs.fLevel),
  fNmany(gcs.fNmany),
  fStart(gcs.fStart),
  fPoint(gcs.fPoint),
  fOverlapping(gcs.fOverlapping)
{
   //copy constructor
   for(Int_t i=0; i<30; i++) fIdBranch[i]=gcs.fIdBranch[i];
}

//_____________________________________________________________________________
TGeoCacheState& TGeoCacheState::operator=(const TGeoCacheState& gcs) 
{
   //assignment operator
   if(this!=&gcs) {
      TObject::operator=(gcs);
      fCapacity=gcs.fCapacity;
      fLevel=gcs.fLevel;
      fNmany=gcs.fNmany;
      fStart=gcs.fStart;
      for(Int_t i=0; i<30; i++) fIdBranch[i]=gcs.fIdBranch[i];
      fPoint=gcs.fPoint;
      fOverlapping=gcs.fOverlapping;
   } 
   return *this;
}

//_____________________________________________________________________________
TGeoCacheState::~TGeoCacheState()
{
// Dtor.
   if (fBranch) {
      delete [] fBranch;
      delete [] fMatrices;
      delete [] fPoint;
   }
}

//_____________________________________________________________________________
void TGeoCacheState::SetState(Int_t level, Int_t startlevel, Int_t nmany, Bool_t ovlp, Double_t *point)
{
// Fill current modeller state.
   fLevel = level;
   fStart = startlevel;
   fNmany = nmany;
   if (gGeoManager->IsOutside()) {
      fLevel = -1;
      return;
   }
   TGeoNodeCache *cache = gGeoManager->GetCache();
   Int_t *branch = (Int_t*)cache->GetBranch();
   Int_t *matrices = (Int_t*)cache->GetMatrices();
   if (cache->HasIdArray()) memcpy(fIdBranch, cache->GetIdBranch()+fStart, (level+1-fStart)*sizeof(Int_t));
   memcpy(fBranch, branch+fStart, (level+1-fStart)*sizeof(Int_t));
   memcpy(fMatrices, matrices+fStart, (level+1-fStart)*sizeof(Int_t));
   fOverlapping = ovlp;
   if (point) memcpy(fPoint, point, 3*sizeof(Double_t));
}

//_____________________________________________________________________________
Bool_t TGeoCacheState::GetState(Int_t &level, Int_t &nmany, Double_t *point) const
{
// Restore a modeller state.
   level = fLevel;
   nmany = fNmany;
   if (fLevel<0) {
      level = 0;
      return kFALSE;
   }
   TGeoNodeCache *cache = gGeoManager->GetCache();
   if (cache->HasIdArray()) cache->FillIdBranch(fIdBranch, fStart);
   Int_t *branch = (Int_t*)cache->GetBranch();
   Int_t *matrices = (Int_t*)cache->GetMatrices();
   memcpy(branch+fStart, fBranch, (level+1-fStart)*sizeof(Int_t));
   memcpy(matrices+fStart, fMatrices, (level+1-fStart)*sizeof(Int_t));
   if (point) memcpy(point, fPoint, 3*sizeof(Double_t));
   return fOverlapping;
}



ClassImp(TGeoCacheStateDummy)

/*************************************************************************
* TGeoCacheStateDummy - class storing the state of modeler at a given moment
*
*
*************************************************************************/

//_____________________________________________________________________________
TGeoCacheStateDummy::TGeoCacheStateDummy()
{
// Default ctor.
   fNodeBranch = 0;
   fMatrixBranch = 0;
   fMatPtr = 0;
}

//_____________________________________________________________________________
TGeoCacheStateDummy::TGeoCacheStateDummy(Int_t capacity)
{
// Ctor.
   fCapacity = capacity;
   fNodeBranch = new TGeoNode *[capacity];
   fMatrixBranch = new TGeoHMatrix *[capacity];
   fMatPtr = new TGeoHMatrix *[capacity];
   for (Int_t i=0; i<capacity; i++)
      fMatrixBranch[i] = new TGeoHMatrix("global");
   fPoint = new Double_t[3];
}

//_____________________________________________________________________________
TGeoCacheStateDummy::TGeoCacheStateDummy(const TGeoCacheStateDummy& csd) :
  TGeoCacheState(csd),
  fNodeBranch(csd.fNodeBranch),
  fMatrixBranch(csd.fMatrixBranch),
  fMatPtr(csd.fMatPtr)
{ 
   //copy constructor
}

//_____________________________________________________________________________
TGeoCacheStateDummy& TGeoCacheStateDummy::operator=(const TGeoCacheStateDummy& csd) 
{
   //assignment operator
   if(this!=&csd) {
      TGeoCacheState::operator=(csd);
      fNodeBranch=csd.fNodeBranch;
      fMatrixBranch=csd.fMatrixBranch;
      fMatPtr=csd.fMatPtr;
   } 
   return *this;
}

//_____________________________________________________________________________
TGeoCacheStateDummy::~TGeoCacheStateDummy()
{
// Dtor.
   if (fNodeBranch) {
      delete [] fNodeBranch;
      for (Int_t i=0; i<fCapacity; i++)
         delete fMatrixBranch[i];
      delete [] fMatrixBranch;
      delete [] fMatPtr;
      delete [] fPoint;
   }
}

//_____________________________________________________________________________
void TGeoCacheStateDummy::SetState(Int_t level, Int_t startlevel, Int_t nmany, Bool_t ovlp, Double_t *point)
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
Bool_t TGeoCacheStateDummy::GetState(Int_t &level, Int_t &nmany, Double_t *point) const
{
 // Restore a modeller state.
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


ClassImp(TGeoMatHandler)
ClassImp(TGeoMatHandlerId)

//_____________________________________________________________________________
TGeoMatHandler::TGeoMatHandler()
{
// Default ctor.
   fLocation = 0;
}

ClassImp(TGeoMatHandlerX)

//_____________________________________________________________________________
void TGeoMatHandlerX::GetMatrix(Double_t *from, TGeoHMatrix *matrix)
{
// Restore translation from cache.
   fLocation = from;
   Double_t *translation = matrix->GetTranslation();
   translation[0] = *from;
   matrix->SetBit(TGeoMatrix::kGeoTranslation);
}

//_____________________________________________________________________________
void TGeoMatHandlerX::AddMatrix(Double_t *to, TGeoMatrix *matrix)
{
// Add translation to cache.
   fLocation = to;
   *to = (matrix->GetTranslation())[0];
}

//_____________________________________________________________________________
void TGeoMatHandlerX::LocalToMaster(const Double_t *local, Double_t *master) const
{
// Local to master conversion.
   memcpy(master, local, 3*sizeof(Double_t));
   master[0] += fLocation[0];
}

//_____________________________________________________________________________
void TGeoMatHandlerX::MasterToLocal(const Double_t *master, Double_t *local) const
{
// Master to local conversion.
   memcpy(local, master, 3*sizeof(Double_t));
   local[0] -= fLocation[0];
}

//_____________________________________________________________________________
void TGeoMatHandlerX::LocalToMasterBomb(const Double_t *local, Double_t *master) const
{
// Local to master conversion within exploded view.
   Double_t tr[3], bombtr[3];
   memset(&tr[0], 0, 3*sizeof(Double_t));
   tr[0] = fLocation[0];
   gGeoManager->BombTranslation(&tr[0], &bombtr[0]);
   memcpy(master, local, 3*sizeof(Double_t));
   master[0] += bombtr[0];
}

//_____________________________________________________________________________
void TGeoMatHandlerX::MasterToLocalBomb(const Double_t *master, Double_t *local) const
{
// Master to local conversion within exploded view.
   Double_t tr[3], bombtr[3];
   memset(&tr[0], 0, 3*sizeof(Double_t));
   tr[0] = fLocation[0];
   gGeoManager->UnbombTranslation(&tr[0], &bombtr[0]);
   memcpy(local, master, 3*sizeof(Double_t));
   local[0] -= bombtr[0];
}

ClassImp(TGeoMatHandlerY)

//_____________________________________________________________________________
void TGeoMatHandlerY::GetMatrix(Double_t *from, TGeoHMatrix *matrix)
{
// Restore translation from cache.
   fLocation = from;
   Double_t *translation = matrix->GetTranslation();
   translation[1] = *from;
   matrix->SetBit(TGeoMatrix::kGeoTranslation);
}

//_____________________________________________________________________________
void TGeoMatHandlerY::AddMatrix(Double_t *to, TGeoMatrix *matrix)
{
// Add translation to cache.
   fLocation = to;
   *to = (matrix->GetTranslation())[1];
}

//_____________________________________________________________________________
void TGeoMatHandlerY::LocalToMaster(const Double_t *local, Double_t *master) const
{
// Local to master conversion.
   memcpy(master, local, 3*sizeof(Double_t));
   master[1] += fLocation[0];
}

//_____________________________________________________________________________
void TGeoMatHandlerY::MasterToLocal(const Double_t *master, Double_t *local) const
{
// Master to local conversion.
   memcpy(local, master, 3*sizeof(Double_t));
   local[1] -= fLocation[0];
}

//_____________________________________________________________________________
void TGeoMatHandlerY::LocalToMasterBomb(const Double_t *local, Double_t *master) const
{
// Local to master conversion within exploded view.
   Double_t tr[3], bombtr[3];
   memset(&tr[0], 0, 3*sizeof(Double_t));
   tr[1] = fLocation[0];
   gGeoManager->BombTranslation(&tr[0], &bombtr[0]);
   memcpy(master, local, 3*sizeof(Double_t));
   master[1] += bombtr[1];
}

//_____________________________________________________________________________
void TGeoMatHandlerY::MasterToLocalBomb(const Double_t *master, Double_t *local) const
{
// Master to local conversion within exploded view.
   Double_t tr[3], bombtr[3];
   memset(&tr[0], 0, 3*sizeof(Double_t));
   tr[1] = fLocation[0];
   gGeoManager->UnbombTranslation(&tr[0], &bombtr[0]);
   memcpy(local, master, 3*sizeof(Double_t));
   local[1] -= bombtr[1];
}

ClassImp(TGeoMatHandlerZ)

//_____________________________________________________________________________
void TGeoMatHandlerZ::GetMatrix(Double_t *from, TGeoHMatrix *matrix)
{
// Restore translation from cache.
   fLocation = from;
   Double_t *translation = matrix->GetTranslation();
   translation[2] = *from;
   matrix->SetBit(TGeoMatrix::kGeoTranslation);
}

//_____________________________________________________________________________
void TGeoMatHandlerZ::AddMatrix(Double_t *to, TGeoMatrix *matrix)
{
// Add translation to cache.
   fLocation = to;
   *to = (matrix->GetTranslation())[2];
}

//_____________________________________________________________________________
void TGeoMatHandlerZ::LocalToMaster(const Double_t *local, Double_t *master) const
{
// Local to master conversion.
   memcpy(master, local, 3*sizeof(Double_t));
   master[2] += fLocation[0];
}

//_____________________________________________________________________________
void TGeoMatHandlerZ::MasterToLocal(const Double_t *master, Double_t *local) const
{
// Master to local conversion.
   memcpy(local, master, 3*sizeof(Double_t));
   local[2] -= fLocation[0];
}

//_____________________________________________________________________________
void TGeoMatHandlerZ::LocalToMasterBomb(const Double_t *local, Double_t *master) const
{
// Local to master conversion within exploded view.
   Double_t tr[3], bombtr[3];
   memset(&tr[0], 0, 3*sizeof(Double_t));
   tr[2] = fLocation[0];
   gGeoManager->BombTranslation(&tr[0], &bombtr[0]);
   memcpy(master, local, 3*sizeof(Double_t));
   master[2] += bombtr[2];
}

//_____________________________________________________________________________
void TGeoMatHandlerZ::MasterToLocalBomb(const Double_t *master, Double_t *local) const
{
// Master to local conversion within exploded view.
   Double_t tr[3], bombtr[3];
   memset(&tr[0], 0, 3*sizeof(Double_t));
   tr[2] = fLocation[0];
   gGeoManager->UnbombTranslation(&tr[0], &bombtr[0]);
   memcpy(local, master, 3*sizeof(Double_t));
   local[2] -= bombtr[2];
}

ClassImp(TGeoMatHandlerXY)

//_____________________________________________________________________________
void TGeoMatHandlerXY::GetMatrix(Double_t *from, TGeoHMatrix *matrix)
{
// Restore translation from cache.
   fLocation = from;
   Double_t *translation = matrix->GetTranslation();
   translation[0] = from[0];
   translation[1] = from[1];
   matrix->SetBit(TGeoMatrix::kGeoTranslation);
}

//_____________________________________________________________________________
void TGeoMatHandlerXY::AddMatrix(Double_t *to, TGeoMatrix *matrix)
{
// Add translation to cache.
   fLocation = to;
   to[0] = (matrix->GetTranslation())[0];
   to[1] = (matrix->GetTranslation())[1];
}

//_____________________________________________________________________________
void TGeoMatHandlerXY::LocalToMaster(const Double_t *local, Double_t *master) const
{
// Local to master conversion.
   memcpy(master, local, 3*sizeof(Double_t));
   master[0] += fLocation[0];
   master[1] += fLocation[1];
}

//_____________________________________________________________________________
void TGeoMatHandlerXY::MasterToLocal(const Double_t *master, Double_t *local) const
{
// Master to local conversion.
   memcpy(local, master, 3*sizeof(Double_t));
   local[0] -= fLocation[0];
   local[1] -= fLocation[1];
}

//_____________________________________________________________________________
void TGeoMatHandlerXY::LocalToMasterBomb(const Double_t *local, Double_t *master) const
{
// Local to master conversion within exploded view.
   Double_t tr[3], bombtr[3];
   memset(&tr[0], 0, 3*sizeof(Double_t));
   tr[0] = fLocation[0];
   tr[1] = fLocation[1];
   gGeoManager->BombTranslation(&tr[0], &bombtr[0]);
   memcpy(master, local, 3*sizeof(Double_t));
   master[0] += bombtr[0];
   master[1] += bombtr[1];
}

//_____________________________________________________________________________
void TGeoMatHandlerXY::MasterToLocalBomb(const Double_t *master, Double_t *local) const
{
// Master to local conversion within exploded view.
   Double_t tr[3], bombtr[3];
   memset(&tr[0], 0, 3*sizeof(Double_t));
   tr[0] = fLocation[0];
   tr[1] = fLocation[1];
   gGeoManager->UnbombTranslation(&tr[0], &bombtr[0]);
   memcpy(local, master, 3*sizeof(Double_t));
   local[0] -= bombtr[0];
   local[1] -= bombtr[1];
}

ClassImp(TGeoMatHandlerXZ)

//_____________________________________________________________________________
void TGeoMatHandlerXZ::GetMatrix(Double_t *from, TGeoHMatrix *matrix)
{
// Restore translation from cache.
   fLocation = from;
   Double_t *translation = matrix->GetTranslation();
   translation[0] = from[0];
   translation[2] = from[1];
   matrix->SetBit(TGeoMatrix::kGeoTranslation);
}

//_____________________________________________________________________________
void TGeoMatHandlerXZ::AddMatrix(Double_t *to, TGeoMatrix *matrix)
{
// Add translation to cache.
   fLocation = to;
   to[0] = (matrix->GetTranslation())[0];
   to[1] = (matrix->GetTranslation())[2];
}

//_____________________________________________________________________________
void TGeoMatHandlerXZ::LocalToMaster(const Double_t *local, Double_t *master) const
{
// Local to master conversion.
   memcpy(master, local, 3*sizeof(Double_t));
   master[0] += fLocation[0];
   master[2] += fLocation[1];
}

//_____________________________________________________________________________
void TGeoMatHandlerXZ::MasterToLocal(const Double_t *master, Double_t *local) const
{
// Master to local conversion.
   memcpy(local, master, 3*sizeof(Double_t));
   local[0] -= fLocation[0];
   local[2] -= fLocation[1];
}

//_____________________________________________________________________________
void TGeoMatHandlerXZ::LocalToMasterBomb(const Double_t *local, Double_t *master) const
{
// Local to master conversion within exploded view.
   Double_t tr[3], bombtr[3];
   memset(&tr[0], 0, 3*sizeof(Double_t));
   tr[0] = fLocation[0];
   tr[2] = fLocation[1];
   gGeoManager->BombTranslation(&tr[0], &bombtr[0]);
   memcpy(master, local, 3*sizeof(Double_t));
   master[0] += bombtr[0];
   master[2] += bombtr[2];
}

//_____________________________________________________________________________
void TGeoMatHandlerXZ::MasterToLocalBomb(const Double_t *master, Double_t *local) const
{
// Master to local conversion within exploded view.
   Double_t tr[3], bombtr[3];
   memset(&tr[0], 0, 3*sizeof(Double_t));
   tr[0] = fLocation[0];
   tr[2] = fLocation[1];
   gGeoManager->UnbombTranslation(&tr[0], &bombtr[0]);
   memcpy(local, master, 3*sizeof(Double_t));
   local[0] -= bombtr[0];
   local[2] -= bombtr[2];
}

ClassImp(TGeoMatHandlerYZ)

//_____________________________________________________________________________
void TGeoMatHandlerYZ::GetMatrix(Double_t *from, TGeoHMatrix *matrix)
{
// Restore translation from cache.
   fLocation = from;
   Double_t *translation = matrix->GetTranslation();
   translation[1] = from[0];
   translation[2] = from[1];
   matrix->SetBit(TGeoMatrix::kGeoTranslation);
}

//_____________________________________________________________________________
void TGeoMatHandlerYZ::AddMatrix(Double_t *to, TGeoMatrix *matrix)
{
// Add translation to cache.
   fLocation = to;
   to[0] = (matrix->GetTranslation())[1];
   to[1] = (matrix->GetTranslation())[2];
}

//_____________________________________________________________________________
void TGeoMatHandlerYZ::LocalToMaster(const Double_t *local, Double_t *master) const
{
// Local to master conversion.
   memcpy(master, local, 3*sizeof(Double_t));
   master[1] += fLocation[0];
   master[2] += fLocation[1];

}

//_____________________________________________________________________________
void TGeoMatHandlerYZ::MasterToLocal(const Double_t *master, Double_t *local) const
{
// Master to local conversion.
   memcpy(local, master, 3*sizeof(Double_t));
   local[1] -= fLocation[0];
   local[2] -= fLocation[1];
}

//_____________________________________________________________________________
void TGeoMatHandlerYZ::LocalToMasterBomb(const Double_t *local, Double_t *master) const
{
// Local to master conversion within exploded view.
   Double_t tr[3], bombtr[3];
   memset(&tr[0], 0, 3*sizeof(Double_t));
   tr[1] = fLocation[0];
   tr[2] = fLocation[1];
   gGeoManager->BombTranslation(&tr[0], &bombtr[0]);
   memcpy(master, local, 3*sizeof(Double_t));
   master[1] += bombtr[1];
   master[2] += bombtr[2];
}

//_____________________________________________________________________________
void TGeoMatHandlerYZ::MasterToLocalBomb(const Double_t *master, Double_t *local) const
{
// Master to local conversion within exploded view.
   Double_t tr[3], bombtr[3];
   memset(&tr[0], 0, 3*sizeof(Double_t));
   tr[1] = fLocation[0];
   tr[2] = fLocation[1];
   gGeoManager->UnbombTranslation(&tr[0], &bombtr[0]);
   memcpy(local, master, 3*sizeof(Double_t));
   local[1] -= bombtr[1];
   local[2] -= bombtr[2];
}

ClassImp(TGeoMatHandlerXYZ)

//_____________________________________________________________________________
void TGeoMatHandlerXYZ::GetMatrix(Double_t *from, TGeoHMatrix *matrix)
{
// Restore translation from cache.
   fLocation = from;
   memcpy(matrix->GetTranslation(), from, 3*sizeof(Double_t));
   matrix->SetBit(TGeoMatrix::kGeoTranslation);
}

//_____________________________________________________________________________
void TGeoMatHandlerXYZ::AddMatrix(Double_t *to, TGeoMatrix *matrix)
{
// Add translation to cache.
   fLocation = to;
   memcpy(to, matrix->GetTranslation(), 3*sizeof(Double_t));
}

//_____________________________________________________________________________
void TGeoMatHandlerXYZ::LocalToMaster(const Double_t *local, Double_t *master) const
{
// Local to master conversion.
   memcpy(master, local, 3*sizeof(Double_t));
   master[0] += fLocation[0];
   master[1] += fLocation[1];
   master[2] += fLocation[2];
}

//_____________________________________________________________________________
void TGeoMatHandlerXYZ::MasterToLocal(const Double_t *master, Double_t *local) const
{
// Master to local conversion.
   memcpy(local, master, 3*sizeof(Double_t));
   local[0] -= fLocation[0];
   local[1] -= fLocation[1];
   local[2] -= fLocation[2];
}

//_____________________________________________________________________________
void TGeoMatHandlerXYZ::LocalToMasterBomb(const Double_t *local, Double_t *master) const
{
// Local to master conversion within exploded view.
   Double_t tr[3], bombtr[3];
   memset(&tr[0], 0, 3*sizeof(Double_t));
   tr[0] = fLocation[0];
   tr[1] = fLocation[1];
   tr[2] = fLocation[2];
   gGeoManager->BombTranslation(&tr[0], &bombtr[0]);
   memcpy(master, local, 3*sizeof(Double_t));
   master[0] += bombtr[0];
   master[1] += bombtr[1];
   master[2] += bombtr[2];
}

//_____________________________________________________________________________
void TGeoMatHandlerXYZ::MasterToLocalBomb(const Double_t *master, Double_t *local) const
{
// Master to local conversion within exploded view.
   Double_t tr[3], bombtr[3];
   memset(&tr[0], 0, 3*sizeof(Double_t));
   tr[0] = fLocation[0];
   tr[1] = fLocation[1];
   tr[2] = fLocation[2];
   gGeoManager->UnbombTranslation(&tr[0], &bombtr[0]);
   memcpy(local, master, 3*sizeof(Double_t));
   local[0] -= bombtr[0];
   local[1] -= bombtr[1];
   local[2] -= bombtr[2];
}

ClassImp(TGeoMatHandlerRot)

//_____________________________________________________________________________
void TGeoMatHandlerRot::GetMatrix(Double_t *from, TGeoHMatrix *matrix)
{
// Restore rotation from cache.
   fLocation = from;
   memcpy(matrix->GetRotationMatrix(), from, 9*sizeof(Double_t));
   matrix->SetBit(TGeoMatrix::kGeoRotation);
}

//_____________________________________________________________________________
void TGeoMatHandlerRot::AddMatrix(Double_t *to, TGeoMatrix *matrix)
{
// Add rotation to cache.
   fLocation = to;
   memcpy(to, matrix->GetRotationMatrix(), 9*sizeof(Double_t));
}

//_____________________________________________________________________________
void TGeoMatHandlerRot::LocalToMaster(const Double_t *local, Double_t *master) const
{
// Local to master conversion.
   master[0] = local[0]*fLocation[0]+local[1]*fLocation[1]+local[2]*fLocation[2];
   master[1] = local[0]*fLocation[3]+local[1]*fLocation[4]+local[2]*fLocation[5];
   master[2] = local[0]*fLocation[6]+local[1]*fLocation[7]+local[2]*fLocation[8];
}

//_____________________________________________________________________________
void TGeoMatHandlerRot::MasterToLocal(const Double_t *master, Double_t *local) const
{
 // Master to local conversion.
   local[0] = master[0]*fLocation[0]+master[1]*fLocation[3]+master[2]*fLocation[6];
   local[1] = master[0]*fLocation[1]+master[1]*fLocation[4]+master[2]*fLocation[7];
   local[2] = master[0]*fLocation[2]+master[1]*fLocation[5]+master[2]*fLocation[8];
}

ClassImp(TGeoMatHandlerRotTr)

//_____________________________________________________________________________
void TGeoMatHandlerRotTr::GetMatrix(Double_t *from, TGeoHMatrix *matrix)
{
// Restore translation/rotation from cache.
   fLocation = from;
   memcpy(matrix->GetRotationMatrix(), from, 9*sizeof(Double_t));
   memcpy(matrix->GetTranslation(), from+9, 3*sizeof(Double_t));
   matrix->SetBit(TGeoMatrix::kGeoTranslation);
   matrix->SetBit(TGeoMatrix::kGeoRotation);
}

//_____________________________________________________________________________
void TGeoMatHandlerRotTr::AddMatrix(Double_t *to, TGeoMatrix *matrix)
{
// Add translation/rotation to cache.
   fLocation = to;
   memcpy(to, matrix->GetRotationMatrix(), 9*sizeof(Double_t));
   memcpy(to+9, matrix->GetTranslation(), 3*sizeof(Double_t));
}

//_____________________________________________________________________________
void TGeoMatHandlerRotTr::LocalToMaster(const Double_t *local, Double_t *master) const
{
// Local to master conversion.
   master[0] = fLocation[9] +
               local[0]*fLocation[0]+local[1]*fLocation[1]+local[2]*fLocation[2];
   master[1] = fLocation[10]+
               local[0]*fLocation[3]+local[1]*fLocation[4]+local[2]*fLocation[5];
   master[2] = fLocation[11]+
               local[0]*fLocation[6]+local[1]*fLocation[7]+local[2]*fLocation[8];
}

//_____________________________________________________________________________
void TGeoMatHandlerRotTr::LocalToMasterVect(const Double_t *local, Double_t *master) const
{
// Local to master conversion for a vector.
   master[0] = local[0]*fLocation[0]+local[1]*fLocation[1]+local[2]*fLocation[2];
   master[1] = local[0]*fLocation[3]+local[1]*fLocation[4]+local[2]*fLocation[5];
   master[2] = local[0]*fLocation[6]+local[1]*fLocation[7]+local[2]*fLocation[8];
}

//_____________________________________________________________________________
void TGeoMatHandlerRotTr::MasterToLocal(const Double_t *master, Double_t *local) const
{
// Master to local conversion.
   local[0] = (master[0]-fLocation[9]) *fLocation[0]+
              (master[1]-fLocation[10])*fLocation[3]+
              (master[2]-fLocation[11])*fLocation[6];
   local[1] = (master[0]-fLocation[9]) *fLocation[1]+
              (master[1]-fLocation[10])*fLocation[4]+
              (master[2]-fLocation[11])*fLocation[7];
   local[2] = (master[0]-fLocation[9])*fLocation[2] +
              (master[1]-fLocation[10])*fLocation[5]+
              (master[2]-fLocation[11])*fLocation[8];
}

//_____________________________________________________________________________
void TGeoMatHandlerRotTr::MasterToLocalVect(const Double_t *master, Double_t *local) const
{
// Master to local conversion for a vector.
   local[0] = master[0]*fLocation[0]+master[1]*fLocation[3]+master[2]*fLocation[6];
   local[1] = master[0]*fLocation[1]+master[1]*fLocation[4]+master[2]*fLocation[7];
   local[2] = master[0]*fLocation[2]+master[1]*fLocation[5]+master[2]*fLocation[8];
}

//_____________________________________________________________________________
void TGeoMatHandlerRotTr::LocalToMasterBomb(const Double_t *local, Double_t *master) const
{
// Local to master conversion within exploded view.
   Double_t bombtr[3];
   gGeoManager->BombTranslation(&fLocation[9], &bombtr[0]);
   master[0] = bombtr[0] +
               local[0]*fLocation[0]+local[1]*fLocation[1]+local[2]*fLocation[2];
   master[1] = bombtr[1]+
               local[0]*fLocation[3]+local[1]*fLocation[4]+local[2]*fLocation[5];
   master[2] = bombtr[2]+
               local[0]*fLocation[6]+local[1]*fLocation[7]+local[2]*fLocation[8];
}

//_____________________________________________________________________________
void TGeoMatHandlerRotTr::MasterToLocalBomb(const Double_t *master, Double_t *local) const
{
// Master to local conversion within exploded view.
   Double_t bombtr[3];
   gGeoManager->UnbombTranslation(&fLocation[9], &bombtr[0]);
   local[0] = (master[0]-bombtr[0]) *fLocation[0]+
              (master[1]-bombtr[1])*fLocation[3]+
              (master[2]-bombtr[2])*fLocation[6];
   local[1] = (master[0]-bombtr[0]) *fLocation[1]+
              (master[1]-bombtr[1])*fLocation[4]+
              (master[2]-bombtr[2])*fLocation[7];
   local[2] = (master[0]-bombtr[0])*fLocation[2] +
              (master[1]-bombtr[1])*fLocation[5]+
              (master[2]-bombtr[2])*fLocation[8];
}

ClassImp(TGeoMatHandlerScl)

//_____________________________________________________________________________
void TGeoMatHandlerScl::GetMatrix(Double_t *from, TGeoHMatrix *matrix)
{
// Restore scale from cache.
   memcpy(matrix->GetScale(), from, 3*sizeof(Double_t));
   matrix->SetBit(TGeoMatrix::kGeoScale);
}

//_____________________________________________________________________________
void TGeoMatHandlerScl::AddMatrix(Double_t *to, TGeoMatrix *matrix)
{
// Add scale to cache.
   memcpy(to, matrix->GetScale(), 3*sizeof(Double_t));
}

ClassImp(TGeoMatHandlerTrScl)

//_____________________________________________________________________________
void TGeoMatHandlerTrScl::GetMatrix(Double_t *from, TGeoHMatrix *matrix)
{
// Restore translation/scale from cache.
   memcpy(matrix->GetTranslation(), from, 3*sizeof(Double_t));
   memcpy(matrix->GetScale(), from+3, 3*sizeof(Double_t));
   matrix->SetBit(TGeoMatrix::kGeoTranslation);
   matrix->SetBit(TGeoMatrix::kGeoScale);
}

//_____________________________________________________________________________
void TGeoMatHandlerTrScl::AddMatrix(Double_t *to, TGeoMatrix *matrix)
{
// Add translation/scale to cache.
   memcpy(to, matrix->GetTranslation(), 3*sizeof(Double_t));
   memcpy(to+3, matrix->GetScale(), 3*sizeof(Double_t));
}

ClassImp(TGeoMatHandlerRotScl)

//_____________________________________________________________________________
void TGeoMatHandlerRotScl::GetMatrix(Double_t *from, TGeoHMatrix *matrix)
{
// Restore rotation/scale from cache.
   memcpy(matrix->GetRotationMatrix(), from, 9*sizeof(Double_t));
   memcpy(matrix->GetScale(), from+9, 3*sizeof(Double_t));
   matrix->SetBit(TGeoMatrix::kGeoRotation);
   matrix->SetBit(TGeoMatrix::kGeoScale);
}

//_____________________________________________________________________________
void TGeoMatHandlerRotScl::AddMatrix(Double_t *to, TGeoMatrix *matrix)
{
// Add rotation/scale to cache.
   memcpy(to, matrix->GetRotationMatrix(), 9*sizeof(Double_t));
   memcpy(to+9, matrix->GetScale(), 3*sizeof(Double_t));
}

ClassImp(TGeoMatHandlerRotTrScl)

//_____________________________________________________________________________
void TGeoMatHandlerRotTrScl::GetMatrix(Double_t *from, TGeoHMatrix *matrix)
{
// Restore translation/rotation/scale from cache.
   memcpy(matrix->GetRotationMatrix(), from, 9*sizeof(Double_t));
   memcpy(matrix->GetTranslation(), from+9, 3*sizeof(Double_t));
   memcpy(matrix->GetScale(), from+12, 3*sizeof(Double_t));
   matrix->SetBit(TGeoMatrix::kGeoTranslation);
   matrix->SetBit(TGeoMatrix::kGeoRotation);
   matrix->SetBit(TGeoMatrix::kGeoScale);
}

//_____________________________________________________________________________
void TGeoMatHandlerRotTrScl::AddMatrix(Double_t *to, TGeoMatrix *matrix)
{
// Add translation/rotation/scale to cache.
   memcpy(to, matrix->GetRotationMatrix(), 9*sizeof(Double_t));
   memcpy(to+9, matrix->GetTranslation(), 3*sizeof(Double_t));
   memcpy(to+12, matrix->GetScale(), 3*sizeof(Double_t));
}
