// @(#)root/geom:$Name:$:$Id:$
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



ClassImp(TGeoNodeCache)
/*************************************************************************
 * TGeoNodeCache - special pool of reusable nodes
 *    
 *
 *************************************************************************/


const Int_t TGeoNodeCache::kGeoCacheMaxDaughters = 128;
const Int_t TGeoNodeCache::kGeoCacheMaxSize      = 1000000;
const Int_t TGeoNodeCache::kGeoCacheStackSize    = 1000;
const Int_t TGeoNodeCache::kGeoCacheDefaultLevel = 4;
const Int_t TGeoNodeCache::kGeoCacheMaxLevels    = 30;
const Int_t TGeoNodeCache::kGeoCacheObjArrayInd  = 0xFF;
const Double_t TGeoNodeCache::kGeoCacheUsageRatio = 0.01;


//-----------------------------------------------------------------------------
TGeoNodeCache::TGeoNodeCache()
{
// dummy constructor
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
   fBranch      = 0;
   fMatrices    = 0;
   fGlobalMatrix= 0;
   fMatrixPool  = 0;
}
//-----------------------------------------------------------------------------
TGeoNodeCache::TGeoNodeCache(Int_t size)
{
// constructor
   gGeoManager->SetCache(this);
   gGeoNodeCache = this;
//   Int_t no_nodes = gGeoManager->CountNodes(gGeoManager->GetTopVolume(), 
//                                            kGeoCacheMaxLevels);
//   if (no_nodes <= kGeoCacheMaxSize) fDefaultLevel = kGeoCacheMaxLevels;
   fDefaultLevel = kGeoCacheDefaultLevel;
   fSize = 0;
   fNused = 0;
   fLevel = 0;
   fStackLevel = 0;
   fCache = new TGeoNodeArray *[256];
   memset(fCache, 0, 0xFF*sizeof(TGeoNodeArray*));
   for (Int_t ic=0; ic<kGeoCacheMaxDaughters+1; ic++) {
      fCache[ic] = new TGeoNodeArray(ic);
      fSize += fCache[ic]->GetSize();
   }
   fCache[kGeoCacheObjArrayInd] = new TGeoNodeObjArray(0);
   fSize += fCache[kGeoCacheObjArrayInd]->GetSize();

   fPath = "";
   fPath.Resize(400);
   fCurrentCache = gGeoManager->GetTopNode()->GetNdaughters();
   if (fCurrentCache>kGeoCacheMaxDaughters) 
      fCurrentCache = kGeoCacheObjArrayInd;
   fBranch = new Int_t[kGeoCacheMaxLevels];
   memset(fBranch, 0, kGeoCacheMaxLevels*sizeof(Int_t));
   fMatrices = new Int_t[kGeoCacheMaxLevels];
   memset(fMatrices, 0, kGeoCacheMaxLevels*sizeof(Int_t));
   fGlobalMatrix = new TGeoHMatrix("current_global");
   fTopNode = AddNode(gGeoManager->GetTopNode());
   fCurrentNode = fTopNode;
   fBranch[0] = fTopNode;
   fCache[fCurrentCache]->SetPersistency();
   fStack = new TObjArray(kGeoCacheStackSize);
   for (Int_t ist=0; ist<kGeoCacheStackSize; ist++)
      fStack->Add(new TGeoCacheState(100)); // !obsolete 100
   printf("### nodes stored in cache %i ###\n", fSize);
   fMatrixPool = new TGeoMatrixCache(0);
   CdTop();
}
//-----------------------------------------------------------------------------
TGeoNodeCache::~TGeoNodeCache()
{
// dtor
   if (fCache) {
      DeleteCaches();
      delete fBranch;
      delete fMatrices;
      delete fGlobalMatrix;
      delete fMatrixPool;
   }
   if (fStack) {
      fStack->Delete();
      delete fStack;
   }   
   gGeoNodeCache = 0;
   gGeoMatrixCache = 0;
}
//-----------------------------------------------------------------------------
void TGeoNodeCache::Compact()
{
// compact arrays
   Int_t old_size, new_size;
   for (Int_t ic=0; ic<kGeoCacheMaxDaughters+1; ic++) {
      old_size = fCache[ic]->GetSize();
      fCache[ic]->Compact();
      new_size = fCache[ic]->GetSize();
      fSize -= (old_size-new_size);
   }
}
//-----------------------------------------------------------------------------
void TGeoNodeCache::DeleteCaches()
{
// delete all node caches
   if (!fCache) return;
   for (Int_t ic=0; ic<kGeoCacheMaxDaughters+1; ic++) {
      fCache[ic]->DeleteArray();
      delete fCache[ic]; 
   }
   delete fCache[kGeoCacheObjArrayInd];
   delete fCache;
}
//-----------------------------------------------------------------------------
Int_t TGeoNodeCache::AddNode(TGeoNode *node)
{
// add a logical node in the cache corresponding to ndaughters
   Int_t ic = node->GetNdaughters();
   if (ic > kGeoCacheMaxDaughters) ic = kGeoCacheObjArrayInd;
   return fCache[ic]->AddNode(node);
   //fNused++;
}
//-----------------------------------------------------------------------------
//Int_t TGeoNodeCache::CacheId(Int_t nindex) 
//{
//   Int_t id = (nindex>>24) & 0xFF;
//   return (id>kGeoCacheMaxDaughters)?kGeoCacheObjArrayInd:id;
//}
//-----------------------------------------------------------------------------
Bool_t TGeoNodeCache::CdDown(Int_t index, Bool_t make)
{
// make daughter 'index' of current node current
   // first make sure that current node is also current in its cache
   fCache[fCurrentCache]->cd(fCurrentIndex);
   Int_t nind_d = fCache[fCurrentCache]->GetDaughter(index);
//   Bool_t check_mat = kTRUE;
   Bool_t persistent = kFALSE;
//   printf("nind_d=%x\n", nind_d);
//   Int_t mat_ind = fCache[fCurrentCache]->GetMatrixInd();
   // if daughter is not stored, create it
   if (!nind_d) {
      if (!make) return kFALSE;
      TGeoNode *node = GetNode()->GetDaughter(index);
//      printf("adding daughter %s of %s\n", node->GetName(), GetNode()->GetName());
      nind_d = fCache[fCurrentCache]->AddDaughter(node, index);
//      printf("   nind_d=%x\n", nind_d);
      fNused++;
//      check_mat = kFALSE;
      if (fLevel < kGeoCacheDefaultLevel) persistent=kTRUE;
   }
   // make daughter current 
   fBranch[++fLevel] = nind_d;
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
      gGeoMatrixCache->cd(fMatrices[fLevel]); 
      return kTRUE;
   }
   // compute matrix and add it to cache
   // get the local matrix
   TGeoMatrix *local = GetNode()->GetMatrix();
//   printf("local matrix for %s : %x\n", GetNode()->GetName(),(UInt_t)local);
   if (local->IsIdentity()) {
   // just copy the matrix from fLevel-1
      fMatrices[fLevel] = fMatrices[fLevel-1];
   // bookkeep the matrix location
      fCache[fCurrentCache]->SetMatrix(fMatrices[fLevel]);
      return kTRUE;
   }
   gGeoMatrixCache->GetMatrix(fGlobalMatrix);
   fGlobalMatrix->Multiply(local);
   // store it in cache and bookkeep its location
   fMatrices[fLevel] = fCache[fCurrentCache]->AddMatrix(fGlobalMatrix);
   return kTRUE;
}  
//-----------------------------------------------------------------------------
void TGeoNodeCache::CdUp()
{
// change current path to mother.
   if (!fLevel) return;
   fLevel--;
   fCurrentNode = fBranch[fLevel];
   fCurrentCache = CacheId(fCurrentNode);
   fCurrentIndex = Index(fCurrentNode);
   fCache[fCurrentCache]->cd(fCurrentIndex);
   gGeoMatrixCache->cd(fMatrices[fLevel]);
}
//-----------------------------------------------------------------------------
//void TGeoNodeCache::CdTop()
//{
// change current path to top node.
//   fLevel = 1;
//   CdUp();
//}
//-----------------------------------------------------------------------------
void TGeoNodeCache::CleanCache()
{
// free nodes which are not persistent from cache
// except the current branch
   // first compute count limit for persistency
   printf("Cleaning cache...\n");
   fCountLimit = Int_t(kGeoCacheUsageRatio*(Double_t)fCount);
   // save level and node branch
   Int_t level = fLevel;
   Int_t branch[kGeoCacheMaxLevels];
   memcpy(&branch[0], fBranch, kGeoCacheMaxLevels*sizeof(Int_t));
   // mark all nodes in the current branch as not-dumpable
   Bool_t flags[kGeoCacheMaxLevels];
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
   memcpy(fBranch, &branch[0], kGeoCacheMaxLevels*sizeof(Int_t));
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
//   Compact();
   Status();
}
//-----------------------------------------------------------------------------
Bool_t TGeoNodeCache::DumpNodes()
{
// dump all non-persistent branches
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
//-----------------------------------------------------------------------------
//void TGeoNodeCache::ClearDaughter(Int_t index)
//{
// clear all the branch of a daughter node. If all is kFALSE,
// clear only non-persistent nodes
//   fCache[fCurrentCache]->cd(fCurrentIndex);
//   fCache[fCurrentCache]->ClearDaughter(index);
//}
//-----------------------------------------------------------------------------
void TGeoNodeCache::ClearNode(Int_t nindex)
{
// clear the only the node nindex
//   printf("clearing node %x\n", (UInt_t)nindex);
   Int_t ic = CacheId(nindex);
   Int_t index = Index(nindex);
   fCache[ic]->cd(index);
   fNused--;
   fCount-=GetUsageCount();
   fCache[ic]->ClearNode();
}
//-----------------------------------------------------------------------------
TGeoNode *TGeoNodeCache::GetMother(Int_t up) const
{
// get mother of current logical node, <up> levels up
   if (!fLevel || (up>fLevel)) return 0;
   Int_t inode = fBranch[fLevel-up];
   Int_t id = CacheId(inode);
   fCache[id]->cd(Index(inode));
   TGeoNode *mother = fCache[id]->GetNode();
   if (fCurrentCache == id) fCache[fCurrentCache]->cd(fCurrentIndex);
   return mother;
}
//-----------------------------------------------------------------------------
const char *TGeoNodeCache::GetPath()
{
// prints the current path
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
//-----------------------------------------------------------------------------
void TGeoNodeCache::PrintNode() const
{
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
//-----------------------------------------------------------------------------
Int_t TGeoNodeCache::PushState(Bool_t ovlp, Double_t *point)
{
   if (fStackLevel>=kGeoCacheStackSize) return 0; 
   ((TGeoCacheState*)fStack->At(fStackLevel))->SetState(fLevel,ovlp,point);
   return ++fStackLevel;   
}   
//-----------------------------------------------------------------------------
void TGeoNodeCache::Refresh() 
{
   fCurrentNode=fBranch[fLevel]; 
   fCurrentCache=CacheId(fCurrentNode);
   fCurrentIndex=Index(fCurrentNode); 
   fCache[fCurrentCache]->cd(fCurrentIndex);
   gGeoMatrixCache->cd(fMatrices[fLevel]);
}
//-----------------------------------------------------------------------------
Bool_t TGeoNodeCache::PopState(Double_t *point) 
{
   if (!fStackLevel) return 0;
   ((TGeoCacheState*)fStack->At(--fStackLevel))->GetState(fLevel,point);
   Refresh(); 
   return (fStackLevel+1);
}
//-----------------------------------------------------------------------------
Bool_t TGeoNodeCache::PopState(Int_t level, Double_t *point) 
{
   if (level<=0) return 0;
   ((TGeoCacheState*)fStack->At(level-1))->GetState(fLevel,point);
   Refresh(); 
   return level;
}
//-----------------------------------------------------------------------------
Bool_t TGeoNodeCache::SetPersistency()
{
   if (fCache[fCurrentCache]->IsPersistent()) return kTRUE;
   Int_t usage = GetUsageCount();
   if (usage>fCountLimit) {
      fCache[fCurrentCache]->SetPersistency();
      return kTRUE;
   }
   return kFALSE;
}
//-----------------------------------------------------------------------------
void TGeoNodeCache::Status() const
{
// print status of cache
   printf("Cache status : total %i   used %i   free %i nodes\n",
          fSize, fNused, fSize-fNused);
}
//-----------------------------------------------------------------------------


/*************************************************************************
 * TGeoCacheDummy - a dummy cache for physical nodes
 *    
 *
 *************************************************************************/
ClassImp(TGeoCacheDummy)

//-----------------------------------------------------------------------------
TGeoCacheDummy::TGeoCacheDummy()
{
   fTop = 0;
   fNode = 0;
   fNodeBranch = 0;
   fMatrixBranch = 0;
}   
//-----------------------------------------------------------------------------
TGeoCacheDummy::TGeoCacheDummy(TGeoNode *top)
{
   fTop = top;
   fNode = top;
   fNodeBranch = new TGeoNode *[kGeoCacheMaxLevels];
   fNodeBranch[0] = top;
   fMatrixBranch = new TGeoHMatrix *[kGeoCacheMaxLevels];
   for (Int_t i=0; i<kGeoCacheMaxLevels; i++)
      fMatrixBranch[i] = new TGeoHMatrix("global");
   fMatrix = fMatrixBranch[0];
   fStack = new TObjArray(kGeoCacheStackSize);
   for (Int_t ist=0; ist<kGeoCacheStackSize; ist++)
      fStack->Add(new TGeoCacheStateDummy(100)); // !obsolete 100
   gGeoNodeCache = this;
   gGeoMatrixCache = 0;
}   
//-----------------------------------------------------------------------------
TGeoCacheDummy::~TGeoCacheDummy()
{
   if (fNodeBranch) delete [] fNodeBranch;
   if (fMatrixBranch) {
      for (Int_t i=0; i<kGeoCacheMaxLevels; i++)
         delete fMatrixBranch[i];
      delete [] fMatrixBranch;
   }   
}   
//-----------------------------------------------------------------------------
Bool_t TGeoCacheDummy::CdDown(Int_t index, Bool_t make)
{
   TGeoNode *newnode = fNode->GetDaughter(index);
   if (!newnode) return kFALSE;
   TGeoHMatrix *newmat = fMatrixBranch[fLevel+1];
   TGeoMatrix  *local = newnode->GetMatrix();
   *newmat = fMatrix;
   newmat->Multiply(local);
   fLevel++;
   fMatrix = newmat;
   fNode = newnode;
   fNodeBranch[fLevel] = fNode;
   fMatrixBranch[fLevel] = fMatrix;
   return kTRUE;
}
//-----------------------------------------------------------------------------
void TGeoCacheDummy::CdUp()
{
   if (!fLevel) return;
   fLevel--;
   fNode = fNodeBranch[fLevel];
   fMatrix = fMatrixBranch[fLevel];
}
//-----------------------------------------------------------------------------
const char *TGeoCacheDummy::GetPath()
{
// prints the current path
   fPath = "";
   for (Int_t level=0;level<fLevel+1; level++) {
      fPath += "/";
      fPath += fNodeBranch[level]->GetName();
   }   
   return fPath.Data();
}
//-----------------------------------------------------------------------------
void TGeoCacheDummy::LocalToMaster(Double_t *local, Double_t *master) const
{
   fMatrix->LocalToMaster(local, master);
}
//-----------------------------------------------------------------------------
void TGeoCacheDummy::LocalToMasterVect(Double_t *local, Double_t *master) const
{
   fMatrix->LocalToMasterVect(local, master);
}
//-----------------------------------------------------------------------------
void TGeoCacheDummy::LocalToMasterBomb(Double_t *local, Double_t *master) const
{
   fMatrix->LocalToMasterBomb(local, master);
}
//-----------------------------------------------------------------------------
void TGeoCacheDummy::MasterToLocal(Double_t *master, Double_t *local) const
{
   fMatrix->MasterToLocal(master, local);
}
//-----------------------------------------------------------------------------
void TGeoCacheDummy::MasterToLocalVect(Double_t *master, Double_t *local) const
{
   fMatrix->MasterToLocalVect(master, local);
}
//-----------------------------------------------------------------------------
void TGeoCacheDummy::MasterToLocalBomb(Double_t *master, Double_t *local) const
{
   fMatrix->MasterToLocalBomb(master, local);
}


const Int_t TGeoNodeArray::kGeoArrayMaxSize  = 1000000;
const Int_t TGeoNodeArray::kGeoArrayInitSize = 1000;
const Int_t TGeoNodeArray::kGeoReleasedSpace = 1000;

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

//-----------------------------------------------------------------------------
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
   fBits = 0;
   fArray = 0;
}
//-----------------------------------------------------------------------------
TGeoNodeArray::TGeoNodeArray(Int_t ndaughters, Int_t size)
{
// default constructor
   if ((ndaughters<0) || (ndaughters>254)) return;
   fNdaughters = ndaughters;
   fSize = size;
   if ((size<kGeoArrayInitSize) || (size>kGeoArrayMaxSize))
      fSize = kGeoArrayInitSize;
   // number of integers stored in a node
   fNodeSize = 3+fNdaughters;
   fFirstFree = 0;
   fCurrent = 0;
   fNused = 0;
   fOffset = 0;
   fBits = new TBits(fSize);
   fArray = new Int_t[fSize*fNodeSize];
   memset(fArray, 0, fSize*fNodeSize*sizeof(Int_t));
   if (!fNdaughters) {
      // never use first location of array with no daughters
      fFirstFree = fCurrent = 1;
      fBits->SetBitNumber(0);
   }
}
//-----------------------------------------------------------------------------
TGeoNodeArray::~TGeoNodeArray()
{
// destructor
//   DeleteArray();
}
//-----------------------------------------------------------------------------
void TGeoNodeArray::Compact()
{
// compact the array
   fBits->Compact();
   Int_t new_size = fBits->GetNbits();
   Int_t *old_array = fArray;
   fArray = new Int_t[new_size*fNodeSize];
   memcpy(fArray, old_array, new_size*fNodeSize*sizeof(Int_t));
   delete old_array;
   fSize = new_size;
}
//-----------------------------------------------------------------------------
void TGeoNodeArray::DeleteArray()
{
   if (fArray) delete fArray;
   fArray = 0;
   if (fBits) delete fBits;
   fBits = 0;
}
//-----------------------------------------------------------------------------
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
   fBits->SetBitNumber(fFirstFree);
   fFirstFree = fBits->FirstNullBit(fFirstFree);
   fNused++;
   if (fFirstFree >= fSize-1) IncreaseArray();
   UChar_t *cache = (UChar_t*)&index;
   cache[3] = (UChar_t)fNdaughters;
   cd(current);
   return index;
}
//-----------------------------------------------------------------------------
void TGeoNodeArray::ClearDaughter(Int_t ind)
{
// clear the daughter ind from the list of the current node. Send the
// signal back to TGeoNodeCache, that proceeds with dispatching the
// clear signal for all the branch
   Int_t nindex_d = GetDaughter(ind);
   if (!nindex_d) return;
   fOffset[3+ind] = 0;
   gGeoNodeCache->ClearNode(nindex_d);
}
//-----------------------------------------------------------------------------
void TGeoNodeArray::ClearMatrix()
{
// clears the global matrix of this node from matrix cache
   Int_t ind_mat = fOffset[1];
   if (ind_mat && !(GetNode()->GetMatrix()->IsIdentity()))
      gGeoMatrixCache->ClearMatrix(ind_mat);
}  
//-----------------------------------------------------------------------------
void TGeoNodeArray::ClearNode()
{
// clear the current node. All branch from this point downwords
// will be deleted
   // remember the current node
   Int_t inode = fCurrent;
   // clear the daughters
   for (Int_t ind=0; ind<fNdaughters; ind++) ClearDaughter(ind);
   cd(inode);
   // clear the global matrix from matrix cache
   ClearMatrix();
   if (fCurrent<fFirstFree) fFirstFree = fCurrent;
   fBits->SetBitNumber(fCurrent, kFALSE);
   fNused--;
   // empty all locations of current node
   memset(fOffset, 0, fNodeSize*sizeof(Int_t));
}
//-----------------------------------------------------------------------------
Bool_t TGeoNodeArray::HasDaughters() const
{
   for (Int_t ind=0; ind<fNdaughters; ind++) {
      if (fOffset[3+ind]) return kTRUE;
   }
   return kFALSE;
}
//-----------------------------------------------------------------------------
void TGeoNodeArray::IncreaseArray()
{
// Doubles the array size unless maximum cache limit is reached or
// global cache limit is reached. In this case forces the cache 
// manager to do the garbage collection.
//   printf("Increasing array %i\n", fNdaughters);
   Int_t new_size = 2*fSize;
   Int_t free_space = gGeoNodeCache->GetFreeSpace();
   if (free_space<10) {
      gGeoNodeCache->CleanCache();
      return;
   }
   if (free_space<fSize) new_size = fSize+free_space;
//   new_size = (new_size>kGeoArrayMaxSize)?kGeoArrayMaxSize:new_size;
/*
   if ((gGeoNodeCache->GetSize()+new_size-fSize) > TGeoNodeCache::kGeoCacheMaxSize) {
      gGeoNodeCache->CleanCache();   
      IncreaseArray();
      return;
   }
*/
   // Increase the cache size and the TBits size
   fBits->SetBitNumber(new_size-1, kFALSE);
   Int_t *new_array = new Int_t[new_size*fNodeSize];
   memset(new_array, 0, new_size*fNodeSize*sizeof(Int_t));
   memcpy(new_array, fArray, fSize*fNodeSize*sizeof(Int_t));
//   printf("array %i fSize=%i newsize=%i\n", fNdaughters, fSize, new_size);
   delete fArray;
   fArray = new_array;
   gGeoNodeCache->IncreasePool(new_size-fSize);
   fSize = new_size;
}
//-----------------------------------------------------------------------------
Bool_t TGeoNodeArray::IsPersistent() const
{
// returns persistency flag of the node
   return ((fOffset[2] & 0x80000000)==0)?kFALSE:kTRUE;
}
//-----------------------------------------------------------------------------
void TGeoNodeArray::SetPersistency(Bool_t flag)
{
   if (flag) fOffset[2] |= 0x80000000;
   else      fOffset[2] &= 0x7FFFFFFF;
}

/*************************************************************************
 * TGeoNodeObjArray - container class for nodes with more than 254
 *     daughters. 
 *
 *************************************************************************/

ClassImp(TGeoNodeObjArray)

//-----------------------------------------------------------------------------
TGeoNodeObjArray::TGeoNodeObjArray()
{
// dummy ctor
   fObjArray = 0;
   fCurrent  = 0;
   fIndex = 0;
}
//-----------------------------------------------------------------------------
TGeoNodeObjArray::TGeoNodeObjArray(Int_t size)
{
// default ctor
   fSize = size;
   fIndex = 0;
   if (size<TGeoNodeArray::kGeoArrayInitSize)
      fSize = TGeoNodeArray::kGeoArrayInitSize;
   fObjArray = new TObjArray(fSize);
   for (Int_t i=0; i<fSize; i++) fObjArray->AddAt(new TGeoNodePos(), i);
   fBits  = new TBits(fSize);
   fCurrent = 0;
}
//-----------------------------------------------------------------------------
TGeoNodeObjArray::~TGeoNodeObjArray()
{
// destructor
   fObjArray->Delete();
   delete fObjArray;
}   
//-----------------------------------------------------------------------------
Int_t TGeoNodeObjArray::AddDaughter(TGeoNode *node, Int_t i)
{
// node must be the i'th daughter of current node (inode, fOffset)
// This is called ONLY after GetDaughter(i) returns 0
   return fCurrent->AddDaughter(i, gGeoNodeCache->AddNode(node));
}
//-----------------------------------------------------------------------------
Int_t TGeoNodeObjArray::AddNode(TGeoNode *node)
{
// Add node in the node array. 
   // first map the node to the first free location which becomes current
   Int_t index = fFirstFree;
   Int_t oldindex = fIndex;
   cd(index);
   fCurrent->Map(node);
   // mark the location as used and compute first free
   fBits->SetBitNumber(fFirstFree);
   fFirstFree = fBits->FirstNullBit(fFirstFree);
   fNused++;
   if (fFirstFree >= fSize-1) IncreaseArray();
   UChar_t *cache = (UChar_t*)&index;
   cache[3] = (UChar_t)TGeoNodeCache::kGeoCacheObjArrayInd;
   cd(oldindex);
   return index;
}
//-----------------------------------------------------------------------------
Int_t TGeoNodeObjArray::AddMatrix(TGeoMatrix *global)
{
// store the global matrix for the current node
   return fCurrent->AddMatrix(global);
}
//-----------------------------------------------------------------------------
void TGeoNodeObjArray::cd(Int_t inode)
{
// make inode the current node
   fCurrent = (TGeoNodePos*)fObjArray->At(inode);
   fIndex = inode;
}
//-----------------------------------------------------------------------------
void TGeoNodeObjArray::ClearDaughter(Int_t ind)
{
// clear the daughter ind from the list of the current node. Send the
// signal back to TGeoNodeCache, that proceeds with dispatching the
// clear signal for all the branch
   Int_t nindex = fCurrent->GetDaughter(ind);
   if (!nindex) return;
   fCurrent->ClearDaughter(ind);
   gGeoNodeCache->ClearNode(nindex);
}
//-----------------------------------------------------------------------------
void TGeoNodeObjArray::ClearMatrix()
{
// clear the global matrix of this node from matrix cache
   Int_t ind_mat = fCurrent->GetMatrixInd();
   if (ind_mat && !fCurrent->GetNode()->GetMatrix()->IsIdentity())
      gGeoMatrixCache->ClearMatrix(ind_mat);
}  
//-----------------------------------------------------------------------------
void TGeoNodeObjArray::ClearNode()
{
// clear the current node. All branch from this point downwords
// will be deleted
   // remember the current node
   Int_t inode = fIndex;
   Int_t nd = GetNdaughters();
   // clear the daughters
   for (Int_t ind=0; ind<nd; ind++) ClearDaughter(ind);
   cd(inode);
   // clear the global matrix from matrix cache
   ClearMatrix();
   if (fIndex<fFirstFree) fFirstFree = fIndex;
   fBits->SetBitNumber(fIndex, kFALSE);
   fNused--;
   // mapping this node to a new logical node is the task of AddNode
}
//-----------------------------------------------------------------------------
void TGeoNodeObjArray::IncreaseArray()
{
// Doubles the array size unless maximum cache limit is reached or
// global cache limit is reached. In this case forces the cache 
// manager to do the garbage collection.
   
//   printf("Increasing ARRAY\n");
   Int_t new_size = 2*fSize;
   Int_t free_space = gGeoNodeCache->GetFreeSpace();
   if (free_space<10) {
      gGeoNodeCache->CleanCache();
      return;
   }
   if (free_space<fSize) new_size = fSize+free_space;

   // Increase the cache size and the TBits size
   fBits->SetBitNumber(new_size-1, kFALSE);
   fObjArray->Expand(new_size);
   for (Int_t i=fSize; i<new_size; i++) fObjArray->AddAt(new TGeoNodePos(), i);
   gGeoNodeCache->IncreasePool(new_size-fSize);
   fSize = new_size;
}



/*************************************************************************
 * TGeoNodePos - the physical geometry node with links to mother and
 *   daughters. 
 *
 *************************************************************************/

const Int_t TGeoNodePos::kPersistentNodeMask   = 0x80000000;
const UChar_t TGeoNodePos::kPersistentMatrixMask = 64;
const UInt_t  TGeoNodePos::kNoMatrix = 1000000000;

ClassImp(TGeoNodePos)

//-----------------------------------------------------------------------------
TGeoNodePos::TGeoNodePos()
{
// dummy ctor
   fNdaughters = 0;
   fDaughters = 0;
   fMatrix = 0;
   fCount = 0;  
   fNode = 0;
}
//-----------------------------------------------------------------------------
TGeoNodePos::TGeoNodePos(Int_t ndaughters)
{
// default constructor.
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
//-----------------------------------------------------------------------------
TGeoNodePos::~TGeoNodePos()
{
// destructor. It deletes the daughters also. 
   // delete daughters 
   if (fDaughters) delete [] fDaughters;
}
//-----------------------------------------------------------------------------
Int_t TGeoNodePos::AddMatrix(TGeoMatrix *global)
{
// cache the global matrix
   return (fMatrix=gGeoMatrixCache->AddMatrix(global));
}   
//-----------------------------------------------------------------------------
void TGeoNodePos::ClearMatrix()
{
// clear the matrix if not used by other nodes
   if (fMatrix && !fNode->GetMatrix()->IsIdentity()) {
      gGeoMatrixCache->ClearMatrix(fMatrix);
      fMatrix = 0;
   }
}
//-----------------------------------------------------------------------------
Int_t TGeoNodePos::GetDaughter(Int_t ind) const
{
// get the i-th daughter.
   if (fDaughters) return fDaughters[ind];
   return 0;
}
//-----------------------------------------------------------------------------
//void TGeoNodePos::GetMatrix(TGeoHMatrix *matrix)
//{
// count the total number of nodes in this branch
//   if (!fMatrix) return;
//   gGeoMatrixCache->GetMatrix(fMatrix, matrix);
//}
//-----------------------------------------------------------------------------
Bool_t TGeoNodePos::HasDaughters() const
{
   for (Int_t i=0; i<fNdaughters; i++) {
      if (fDaughters[i]!=0) return kTRUE;
   }
   return kFALSE;
}
//-----------------------------------------------------------------------------
void TGeoNodePos::Map(TGeoNode *node)
{
// map this nodepos to a physical node
   fNdaughters = node->GetNdaughters();
   if (fDaughters) delete [] fDaughters;
   fDaughters = new Int_t[fNdaughters];
   memset(fDaughters, 0, fNdaughters*sizeof(Int_t));
   fMatrix = 0;
   fCount = 0;
   fNode = node;
}
//-----------------------------------------------------------------------------
void TGeoNodePos::SetPersistency(Bool_t flag)
{
// set this node persistent in cache
   if (flag) fCount |= kPersistentNodeMask;
   else      fCount &= !kPersistentNodeMask;
}

/*************************************************************************
 * TGeoMatrixCache - cache of global matrices
 *    
 *
 *************************************************************************/

const Int_t TGeoMatrixCache::kGeoDefaultIncrease = 1000;
const Int_t TGeoMatrixCache::kGeoMinCacheSize    = 1000;
const UChar_t TGeoMatrixCache::kGeoMaskX         = 1; 
const UChar_t TGeoMatrixCache::kGeoMaskY         = 2; 
const UChar_t TGeoMatrixCache::kGeoMaskZ         = 4; 
const UChar_t TGeoMatrixCache::kGeoMaskXYZ       = 7; 
const UChar_t TGeoMatrixCache::kGeoMaskRot       = 8; 
const UChar_t TGeoMatrixCache::kGeoMaskScale     = 16; 

TGeoMatrixCache *gGeoMatrixCache = 0;

ClassImp(TGeoMatrixCache)

//-----------------------------------------------------------------------------
TGeoMatrixCache::TGeoMatrixCache()
{
// dummy ctor
   for (Int_t i=0; i<7; i++) {
      fSize[i]  = 0;
      fCache[i] = 0;
      fFree[i]  = 0;
      fBits[i]  = 0;
   }
   fMatrix = 0;
   fHandler = 0;
   fCacheId = 0;
   fLength = 0;
   fHandlers = 0;
}
//-----------------------------------------------------------------------------
TGeoMatrixCache::TGeoMatrixCache(Int_t size)
{
// default constructor
   gGeoMatrixCache = this;
   Int_t length;
   for (Int_t i=0; i<7; i++) {
      if (size < kGeoMinCacheSize) {
         fSize[i] = kGeoMinCacheSize;
//         if (i==5) fSize[i]=100000;
      } else {
         fSize[i] = size;
      }
      length = 3*(i-1);
      if (length == 0) length=2;
      if (length < 0) length=1;
      fCache[i] = new Double_t[fSize[i]*length];
      fBits[i]  = new TBits(fSize[i]);
      fFree[i]  = 0;
      if (i==0) {
         fBits[i]->SetBitNumber(0);
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
//   Status();
}
//-----------------------------------------------------------------------------
TGeoMatrixCache::~TGeoMatrixCache()
{
// destructor
   if (fSize[0]) {
      for (Int_t i=0; i<7; i++) {
         delete fCache[i];
         delete fBits[i];
      }
      for (Int_t j=0; j<14; j++)
         delete fHandlers[j];
      delete [] fHandlers;
   }
   gGeoMatrixCache = 0;
}
//-----------------------------------------------------------------------------
Int_t TGeoMatrixCache::AddMatrix(TGeoMatrix *matrix)
{
// add a global matrix to the first free array of corresponding type
   if (matrix->IsIdentity()) {fHandler=13; return (fMatrix=0);}
   
   const Double_t *translation = matrix->GetTranslation();

   UChar_t type = 0;
   if (matrix->IsRotation()) type |= kGeoMaskRot;
   if (matrix->IsScale())    type |= kGeoMaskScale;
   if (matrix->IsTranslation()) {
      if (translation[0]!=0)  type |= kGeoMaskX;
      if (translation[1]!=0)  type |= kGeoMaskY;
      if (translation[2]!=0)  type |= kGeoMaskZ;
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
//   matrix->Print();
//   printf("type=%x cache_id=%i length=%i handler:%i\n", type, index, data_len, h);

   fBits[fCacheId]->SetBitNumber(current_free);
   fFree[fCacheId] = fBits[fCacheId]->FirstNullBit(current_free); 
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
//-----------------------------------------------------------------------------
void TGeoMatrixCache::cd(Int_t mindex)
{
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
//-----------------------------------------------------------------------------
void TGeoMatrixCache::ClearMatrix(Int_t mindex)
{
// release the space occupied by a matrix
   if (!mindex) return;
   cd(mindex);
   Int_t offset = fMatrix&0x00FFFFFF;
   fBits[fCacheId]->SetBitNumber(offset, kFALSE);
   if (UInt_t(offset)<fFree[fCacheId]) fFree[fCacheId] = offset;
}
//-----------------------------------------------------------------------------
void TGeoMatrixCache::GetMatrix(TGeoHMatrix *matrix) const
{
// get a matrix from cache
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
//-----------------------------------------------------------------------------
void TGeoMatrixCache::IncreaseCache()
{
// doubles the cache size
//   printf("Increasing matrix cache %i ...\n", fCacheId);
   UInt_t new_size = 2*fSize[fCacheId];
   fBits[fCacheId]->SetBitNumber(new_size-1, kFALSE);
   Double_t *new_cache = new Double_t[new_size*fLength];
   // copy old bits to new bits and old data to new data
   memcpy(new_cache, fCache[fCacheId], fSize[fCacheId]*fLength*sizeof(Double_t));
   delete fCache[fCacheId];
   fCache[fCacheId] = new_cache;
   fSize[fCacheId] = new_size;
//   Status();
}
//-----------------------------------------------------------------------------
void TGeoMatrixCache::Status() const
{
// print current status of matrix cache
   Int_t ntot, ntotc,ntotused, nused, nfree, length;
   printf("Matrix cache status :   total    used    free\n");
   ntot = 0; 
   ntotused = 0;
   for (Int_t i=0; i<7; i++) {
      length = 3*(i-1);
      if (length == 0) length=2;
      if (length < 0) length=1;
      ntotc = fSize[i];
      nused = fBits[i]->CountBits();
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

//-----------------------------------------------------------------------------
TGeoCacheState::TGeoCacheState()
{
//--- Default ctor
   fLevel = 0;
   fBranch = 0;
   fMatrices = 0;
   fPoint = 0;
}
//-----------------------------------------------------------------------------
TGeoCacheState::TGeoCacheState(Int_t capacity)
{
//--- ctor
   fLevel = 0;
   fBranch = new Int_t[TGeoNodeCache::kGeoCacheMaxLevels];
   fMatrices = new Int_t[TGeoNodeCache::kGeoCacheMaxLevels];
   fPoint = new Double_t[3];
}
//-----------------------------------------------------------------------------
TGeoCacheState::~TGeoCacheState()
{
//--- dtor
   if (fBranch) {
      delete [] fBranch;
      delete [] fMatrices;
      delete [] fPoint;
   }
}
//-----------------------------------------------------------------------------
void TGeoCacheState::SetState(Int_t level, Bool_t ovlp, Double_t *point)
{
   fLevel = level;
   memcpy(fBranch, (Int_t*)gGeoNodeCache->GetBranch(), (level+1)*sizeof(Int_t));
   memcpy(fMatrices, (Int_t*)gGeoNodeCache->GetMatrices(), (level+1)*sizeof(Int_t));
   fOverlapping = ovlp;
   if (point) memcpy(fPoint, point, 3*sizeof(Double_t));
}   
//-----------------------------------------------------------------------------
Bool_t TGeoCacheState::GetState(Int_t &level, Double_t *point) const
{
   level = fLevel;
   memcpy((Int_t*)gGeoNodeCache->GetBranch(), fBranch, (level+1)*sizeof(Int_t));
   memcpy((Int_t*)gGeoNodeCache->GetMatrices(), fMatrices, (level+1)*sizeof(Int_t));
   if (point) memcpy(point, fPoint, 3*sizeof(Double_t));
   return fOverlapping;
}   



ClassImp(TGeoCacheStateDummy)

/*************************************************************************
* TGeoCacheStateDummy - class storing the state of modeler at a given moment
*    
*
*************************************************************************/

//-----------------------------------------------------------------------------
TGeoCacheStateDummy::TGeoCacheStateDummy()
{
//--- Default ctor
   fNodeBranch = 0;
   fMatrixBranch = 0;
}
//-----------------------------------------------------------------------------
TGeoCacheStateDummy::TGeoCacheStateDummy(Int_t capacity)
{
//--- ctor
   fNodeBranch = new TGeoNode *[TGeoNodeCache::kGeoCacheMaxLevels];
   fMatrixBranch = new TGeoHMatrix *[TGeoNodeCache::kGeoCacheMaxLevels];
   for (Int_t i=0; i<TGeoNodeCache::kGeoCacheMaxLevels; i++)
      fMatrixBranch[i] = new TGeoHMatrix("global");
   fPoint = new Double_t[3];
}
//-----------------------------------------------------------------------------
TGeoCacheStateDummy::~TGeoCacheStateDummy()
{
//--- dtor
   if (fNodeBranch) {
      delete [] fNodeBranch;
      for (Int_t i=0; i<TGeoNodeCache::kGeoCacheMaxLevels; i++)
         delete fMatrixBranch[i];
      delete [] fMatrixBranch;
      delete [] fPoint;
   }
}
//-----------------------------------------------------------------------------
void TGeoCacheStateDummy::SetState(Int_t level, Bool_t ovlp, Double_t *point)
{
   fLevel = level;
   TGeoNode **node_branch = (TGeoNode **) gGeoNodeCache->GetBranch();
   TGeoHMatrix **mat_branch  = (TGeoHMatrix **) gGeoNodeCache->GetMatrices();

   memcpy(fNodeBranch, node_branch, (level+1)*sizeof(TGeoNode *));
   for (Int_t i=0; i<level+1; i++)
      *fMatrixBranch[i] = mat_branch[i];
   fOverlapping = ovlp;
   if (point) memcpy(fPoint, point, 3*sizeof(Double_t));
}   
//-----------------------------------------------------------------------------
Bool_t TGeoCacheStateDummy::GetState(Int_t &level, Double_t *point) const
{
   level = fLevel;
   TGeoNode **node_branch = (TGeoNode **) gGeoNodeCache->GetBranch();
   TGeoHMatrix **mat_branch  = (TGeoHMatrix **) gGeoNodeCache->GetMatrices();

   memcpy(node_branch, fNodeBranch, (level+1)*sizeof(TGeoNode *));
   for (Int_t i=0; i<level+1; i++)
      *mat_branch[i] = fMatrixBranch[i];
   if (point) memcpy(point, fPoint, 3*sizeof(Double_t));
   return fOverlapping;
}   


ClassImp(TGeoMatHandler)
ClassImp(TGeoMatHandlerId)

//-----------------------------------------------------------------------------
TGeoMatHandler::TGeoMatHandler()
{
   fLocation = 0;
}

ClassImp(TGeoMatHandlerX)

//-----------------------------------------------------------------------------
void TGeoMatHandlerX::GetMatrix(Double_t *from, TGeoHMatrix *matrix)
{
   fLocation = from;
   Double_t *translation = matrix->GetTranslation();
   translation[0] = *from;
   matrix->SetBit(kGeoTranslation);
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerX::AddMatrix(Double_t *to, TGeoMatrix *matrix)
{
   fLocation = to;
   *to = (matrix->GetTranslation())[0];
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerX::LocalToMaster(Double_t *local, Double_t *master) const
{
   memcpy(master, local, 3*sizeof(Double_t));
   master[0] += fLocation[0];
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerX::MasterToLocal(Double_t *master, Double_t *local) const
{
   memcpy(local, master, 3*sizeof(Double_t));
   local[0] -= fLocation[0];
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerX::LocalToMasterBomb(Double_t *local, Double_t *master) const
{
   Double_t tr[3], bombtr[3];
   memset(&tr[0], 0, 3*sizeof(Double_t));
   tr[0] = fLocation[0];
   gGeoManager->BombTranslation(&tr[0], &bombtr[0]);
   memcpy(master, local, 3*sizeof(Double_t));
   master[0] += bombtr[0];
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerX::MasterToLocalBomb(Double_t *master, Double_t *local) const
{
   Double_t tr[3], bombtr[3];
   memset(&tr[0], 0, 3*sizeof(Double_t));
   tr[0] = fLocation[0];
   gGeoManager->UnbombTranslation(&tr[0], &bombtr[0]);
   memcpy(local, master, 3*sizeof(Double_t));
   local[0] -= bombtr[0];
}

ClassImp(TGeoMatHandlerY)

//-----------------------------------------------------------------------------
void TGeoMatHandlerY::GetMatrix(Double_t *from, TGeoHMatrix *matrix)
{
   fLocation = from;
   Double_t *translation = matrix->GetTranslation();
   translation[1] = *from;
   matrix->SetBit(kGeoTranslation);
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerY::AddMatrix(Double_t *to, TGeoMatrix *matrix)
{
   fLocation = to;
   *to = (matrix->GetTranslation())[1];
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerY::LocalToMaster(Double_t *local, Double_t *master) const
{
   memcpy(master, local, 3*sizeof(Double_t));
   master[1] += fLocation[0];
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerY::MasterToLocal(Double_t *master, Double_t *local) const
{
   memcpy(local, master, 3*sizeof(Double_t));
   local[1] -= fLocation[0];
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerY::LocalToMasterBomb(Double_t *local, Double_t *master) const
{
   Double_t tr[3], bombtr[3];
   memset(&tr[0], 0, 3*sizeof(Double_t));
   tr[1] = fLocation[0];
   gGeoManager->BombTranslation(&tr[0], &bombtr[0]);
   memcpy(master, local, 3*sizeof(Double_t));
   master[1] += bombtr[1];
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerY::MasterToLocalBomb(Double_t *master, Double_t *local) const
{
   Double_t tr[3], bombtr[3];
   memset(&tr[0], 0, 3*sizeof(Double_t));
   tr[1] = fLocation[0];
   gGeoManager->UnbombTranslation(&tr[0], &bombtr[0]);
   memcpy(local, master, 3*sizeof(Double_t));
   local[1] -= bombtr[1];
}

ClassImp(TGeoMatHandlerZ)

//-----------------------------------------------------------------------------
void TGeoMatHandlerZ::GetMatrix(Double_t *from, TGeoHMatrix *matrix)
{
   fLocation = from;
   Double_t *translation = matrix->GetTranslation();
   translation[2] = *from;
   matrix->SetBit(kGeoTranslation);
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerZ::AddMatrix(Double_t *to, TGeoMatrix *matrix)
{
   fLocation = to;
   *to = (matrix->GetTranslation())[2];
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerZ::LocalToMaster(Double_t *local, Double_t *master) const
{
   memcpy(master, local, 3*sizeof(Double_t));
   master[2] += fLocation[0];
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerZ::MasterToLocal(Double_t *master, Double_t *local) const
{
   memcpy(local, master, 3*sizeof(Double_t));
   local[2] -= fLocation[0];
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerZ::LocalToMasterBomb(Double_t *local, Double_t *master) const
{
   Double_t tr[3], bombtr[3];
   memset(&tr[0], 0, 3*sizeof(Double_t));
   tr[2] = fLocation[0];
   gGeoManager->BombTranslation(&tr[0], &bombtr[0]);
   memcpy(master, local, 3*sizeof(Double_t));
   master[2] += bombtr[2];
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerZ::MasterToLocalBomb(Double_t *master, Double_t *local) const
{
   Double_t tr[3], bombtr[3];
   memset(&tr[0], 0, 3*sizeof(Double_t));
   tr[2] = fLocation[0];
   gGeoManager->UnbombTranslation(&tr[0], &bombtr[0]);
   memcpy(local, master, 3*sizeof(Double_t));
   local[2] -= bombtr[2];
}

ClassImp(TGeoMatHandlerXY)

//-----------------------------------------------------------------------------
void TGeoMatHandlerXY::GetMatrix(Double_t *from, TGeoHMatrix *matrix)
{
   fLocation = from;
   Double_t *translation = matrix->GetTranslation();
   translation[0] = from[0];
   translation[1] = from[1];
   matrix->SetBit(kGeoTranslation);
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerXY::AddMatrix(Double_t *to, TGeoMatrix *matrix)
{
   fLocation = to;
   to[0] = (matrix->GetTranslation())[0];
   to[1] = (matrix->GetTranslation())[1];
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerXY::LocalToMaster(Double_t *local, Double_t *master) const
{
   memcpy(master, local, 3*sizeof(Double_t));
   master[0] += fLocation[0];
   master[1] += fLocation[1];
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerXY::MasterToLocal(Double_t *master, Double_t *local) const
{
   memcpy(local, master, 3*sizeof(Double_t));
   local[0] -= fLocation[0];
   local[1] -= fLocation[1];
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerXY::LocalToMasterBomb(Double_t *local, Double_t *master) const
{
   Double_t tr[3], bombtr[3];
   memset(&tr[0], 0, 3*sizeof(Double_t));
   tr[0] = fLocation[0];
   tr[1] = fLocation[1];
   gGeoManager->BombTranslation(&tr[0], &bombtr[0]);
   memcpy(master, local, 3*sizeof(Double_t));
   master[0] += bombtr[0];
   master[1] += bombtr[1];
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerXY::MasterToLocalBomb(Double_t *master, Double_t *local) const
{
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

//-----------------------------------------------------------------------------
void TGeoMatHandlerXZ::GetMatrix(Double_t *from, TGeoHMatrix *matrix)
{
   fLocation = from;
   Double_t *translation = matrix->GetTranslation();
   translation[0] = from[0];
   translation[2] = from[1];
   matrix->SetBit(kGeoTranslation);
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerXZ::AddMatrix(Double_t *to, TGeoMatrix *matrix)
{
   fLocation = to;
   to[0] = (matrix->GetTranslation())[0];
   to[1] = (matrix->GetTranslation())[2];
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerXZ::LocalToMaster(Double_t *local, Double_t *master) const
{
   memcpy(master, local, 3*sizeof(Double_t));
   master[0] += fLocation[0];
   master[2] += fLocation[1];
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerXZ::MasterToLocal(Double_t *master, Double_t *local) const
{
   memcpy(local, master, 3*sizeof(Double_t));
   local[0] -= fLocation[0];
   local[2] -= fLocation[1];
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerXZ::LocalToMasterBomb(Double_t *local, Double_t *master) const
{
   Double_t tr[3], bombtr[3];
   memset(&tr[0], 0, 3*sizeof(Double_t));
   tr[0] = fLocation[0];
   tr[2] = fLocation[1];
   gGeoManager->BombTranslation(&tr[0], &bombtr[0]);
   memcpy(master, local, 3*sizeof(Double_t));
   master[0] += bombtr[0];
   master[2] += bombtr[2];
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerXZ::MasterToLocalBomb(Double_t *master, Double_t *local) const
{
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

//-----------------------------------------------------------------------------
void TGeoMatHandlerYZ::GetMatrix(Double_t *from, TGeoHMatrix *matrix)
{
   fLocation = from;
   Double_t *translation = matrix->GetTranslation();
   translation[1] = from[0];
   translation[2] = from[1];
   matrix->SetBit(kGeoTranslation);
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerYZ::AddMatrix(Double_t *to, TGeoMatrix *matrix)
{
   fLocation = to;
   to[0] = (matrix->GetTranslation())[1];
   to[1] = (matrix->GetTranslation())[2];
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerYZ::LocalToMaster(Double_t *local, Double_t *master) const
{
   memcpy(master, local, 3*sizeof(Double_t));
   master[1] += fLocation[0];
   master[2] += fLocation[1];

}
//-----------------------------------------------------------------------------
void TGeoMatHandlerYZ::MasterToLocal(Double_t *master, Double_t *local) const
{
   memcpy(local, master, 3*sizeof(Double_t));
   local[1] -= fLocation[0];
   local[2] -= fLocation[1];
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerYZ::LocalToMasterBomb(Double_t *local, Double_t *master) const
{
   Double_t tr[3], bombtr[3];
   memset(&tr[0], 0, 3*sizeof(Double_t));
   tr[1] = fLocation[0];
   tr[2] = fLocation[1];
   gGeoManager->BombTranslation(&tr[0], &bombtr[0]);
   memcpy(master, local, 3*sizeof(Double_t));
   master[1] += bombtr[1];
   master[2] += bombtr[2];
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerYZ::MasterToLocalBomb(Double_t *master, Double_t *local) const
{
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

//-----------------------------------------------------------------------------
void TGeoMatHandlerXYZ::GetMatrix(Double_t *from, TGeoHMatrix *matrix)
{
   fLocation = from;
   memcpy(matrix->GetTranslation(), from, 3*sizeof(Double_t));
   matrix->SetBit(kGeoTranslation);
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerXYZ::AddMatrix(Double_t *to, TGeoMatrix *matrix)
{
   fLocation = to;
   memcpy(to, matrix->GetTranslation(), 3*sizeof(Double_t));
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerXYZ::LocalToMaster(Double_t *local, Double_t *master) const
{
   memcpy(master, local, 3*sizeof(Double_t));
   master[0] += fLocation[0];
   master[1] += fLocation[1];
   master[2] += fLocation[2];
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerXYZ::MasterToLocal(Double_t *master, Double_t *local) const
{
   memcpy(local, master, 3*sizeof(Double_t));
   local[0] -= fLocation[0];
   local[1] -= fLocation[1];
   local[2] -= fLocation[2];
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerXYZ::LocalToMasterBomb(Double_t *local, Double_t *master) const
{
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
//-----------------------------------------------------------------------------
void TGeoMatHandlerXYZ::MasterToLocalBomb(Double_t *master, Double_t *local) const
{
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

//-----------------------------------------------------------------------------
void TGeoMatHandlerRot::GetMatrix(Double_t *from, TGeoHMatrix *matrix)
{
   fLocation = from;
   memcpy(matrix->GetRotationMatrix(), from, 9*sizeof(Double_t));
   matrix->SetBit(kGeoRotation);
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerRot::AddMatrix(Double_t *to, TGeoMatrix *matrix)
{
   fLocation = to;
   memcpy(to, matrix->GetRotationMatrix(), 9*sizeof(Double_t));
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerRot::LocalToMaster(Double_t *local, Double_t *master) const
{
   master[0] = local[0]*fLocation[0]+local[1]*fLocation[1]+local[2]*fLocation[2];
   master[1] = local[0]*fLocation[3]+local[1]*fLocation[4]+local[2]*fLocation[5];
   master[2] = local[0]*fLocation[6]+local[1]*fLocation[7]+local[2]*fLocation[8];
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerRot::MasterToLocal(Double_t *master, Double_t *local) const
{
   local[0] = master[0]*fLocation[0]+master[1]*fLocation[3]+master[2]*fLocation[6];
   local[1] = master[0]*fLocation[1]+master[1]*fLocation[4]+master[2]*fLocation[7];
   local[2] = master[0]*fLocation[2]+master[1]*fLocation[5]+master[2]*fLocation[8];   
}

ClassImp(TGeoMatHandlerRotTr)

//-----------------------------------------------------------------------------
void TGeoMatHandlerRotTr::GetMatrix(Double_t *from, TGeoHMatrix *matrix)
{
   fLocation = from;
   memcpy(matrix->GetRotationMatrix(), from, 9*sizeof(Double_t));
   memcpy(matrix->GetTranslation(), from+9, 3*sizeof(Double_t));
   matrix->SetBit(kGeoTranslation);
   matrix->SetBit(kGeoRotation);
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerRotTr::AddMatrix(Double_t *to, TGeoMatrix *matrix)
{
   fLocation = to;
   memcpy(to, matrix->GetRotationMatrix(), 9*sizeof(Double_t));
   memcpy(to+9, matrix->GetTranslation(), 3*sizeof(Double_t));
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerRotTr::LocalToMaster(Double_t *local, Double_t *master) const
{
   master[0] = fLocation[9] +
               local[0]*fLocation[0]+local[1]*fLocation[1]+local[2]*fLocation[2];
   master[1] = fLocation[10]+
               local[0]*fLocation[3]+local[1]*fLocation[4]+local[2]*fLocation[5];
   master[2] = fLocation[11]+
               local[0]*fLocation[6]+local[1]*fLocation[7]+local[2]*fLocation[8];
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerRotTr::LocalToMasterVect(Double_t *local, Double_t *master) const
{
   master[0] = local[0]*fLocation[0]+local[1]*fLocation[1]+local[2]*fLocation[2];
   master[1] = local[0]*fLocation[3]+local[1]*fLocation[4]+local[2]*fLocation[5];
   master[2] = local[0]*fLocation[6]+local[1]*fLocation[7]+local[2]*fLocation[8];
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerRotTr::MasterToLocal(Double_t *master, Double_t *local) const
{
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
//-----------------------------------------------------------------------------
void TGeoMatHandlerRotTr::MasterToLocalVect(Double_t *master, Double_t *local) const
{
   local[0] = master[0]*fLocation[0]+master[1]*fLocation[3]+master[2]*fLocation[6];
   local[1] = master[0]*fLocation[1]+master[1]*fLocation[4]+master[2]*fLocation[7];
   local[2] = master[0]*fLocation[2]+master[1]*fLocation[5]+master[2]*fLocation[8];   
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerRotTr::LocalToMasterBomb(Double_t *local, Double_t *master) const
{
   Double_t bombtr[3];
   gGeoManager->BombTranslation(&fLocation[9], &bombtr[0]);
   master[0] = bombtr[0] +
               local[0]*fLocation[0]+local[1]*fLocation[1]+local[2]*fLocation[2];
   master[1] = bombtr[1]+
               local[0]*fLocation[3]+local[1]*fLocation[4]+local[2]*fLocation[5];
   master[2] = bombtr[2]+
               local[0]*fLocation[6]+local[1]*fLocation[7]+local[2]*fLocation[8];
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerRotTr::MasterToLocalBomb(Double_t *master, Double_t *local) const
{
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

//-----------------------------------------------------------------------------
void TGeoMatHandlerScl::GetMatrix(Double_t *from, TGeoHMatrix *matrix)
{
   memcpy(matrix->GetScale(), from, 3*sizeof(Double_t));
   matrix->SetBit(kGeoScale);
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerScl::AddMatrix(Double_t *to, TGeoMatrix *matrix)
{
   memcpy(to, matrix->GetScale(), 3*sizeof(Double_t));
}

ClassImp(TGeoMatHandlerTrScl)

//-----------------------------------------------------------------------------
void TGeoMatHandlerTrScl::GetMatrix(Double_t *from, TGeoHMatrix *matrix)
{
   memcpy(matrix->GetTranslation(), from, 3*sizeof(Double_t));
   memcpy(matrix->GetScale(), from+3, 3*sizeof(Double_t));
   matrix->SetBit(kGeoTranslation);
   matrix->SetBit(kGeoScale);
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerTrScl::AddMatrix(Double_t *to, TGeoMatrix *matrix)
{
   memcpy(to, matrix->GetTranslation(), 3*sizeof(Double_t));
   memcpy(to+3, matrix->GetScale(), 3*sizeof(Double_t));
}

ClassImp(TGeoMatHandlerRotScl)

//-----------------------------------------------------------------------------
void TGeoMatHandlerRotScl::GetMatrix(Double_t *from, TGeoHMatrix *matrix)
{
   memcpy(matrix->GetRotationMatrix(), from, 9*sizeof(Double_t));
   memcpy(matrix->GetScale(), from+9, 3*sizeof(Double_t));
   matrix->SetBit(kGeoRotation);
   matrix->SetBit(kGeoScale);
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerRotScl::AddMatrix(Double_t *to, TGeoMatrix *matrix)
{
   memcpy(to, matrix->GetRotationMatrix(), 9*sizeof(Double_t));
   memcpy(to+9, matrix->GetScale(), 3*sizeof(Double_t));
}

ClassImp(TGeoMatHandlerRotTrScl)

//-----------------------------------------------------------------------------
void TGeoMatHandlerRotTrScl::GetMatrix(Double_t *from, TGeoHMatrix *matrix)
{
   memcpy(matrix->GetRotationMatrix(), from, 9*sizeof(Double_t));
   memcpy(matrix->GetTranslation(), from+9, 3*sizeof(Double_t));
   memcpy(matrix->GetScale(), from+12, 3*sizeof(Double_t));
   matrix->SetBit(kGeoTranslation);
   matrix->SetBit(kGeoRotation);
   matrix->SetBit(kGeoScale);
}
//-----------------------------------------------------------------------------
void TGeoMatHandlerRotTrScl::AddMatrix(Double_t *to, TGeoMatrix *matrix)
{
   memcpy(to, matrix->GetRotationMatrix(), 9*sizeof(Double_t));
   memcpy(to+9, matrix->GetTranslation(), 3*sizeof(Double_t));
   memcpy(to+12, matrix->GetScale(), 3*sizeof(Double_t));
}
