// @(#)root/tree:$Name:  $:$Id: TTreeFilePrefetch.cxx,v 1.3 2006/06/07 18:52:26 brun Exp $
// Author: Rene Brun   04/06/2006

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTreeFilePrefetch                                                    //
//                                                                      //
//  A specialized TFilePrefetch object for a TTree                      //
//  This class acts as a file cache, registering automatically the      //
//  baskets from the branches being processed (TTree::Draw or           //
//  TTree::Process and TSelectors.                                      //
//  This cache speeds-up considerably the performance, in particular    //
//  when the Tree is accessed remotely via a high latency network.      //
//                                                                      //
//  The default cache size (10 Mbytes) may be changed via the function  //
//      TTreeFilePrefetch::SetCacheSize                                 //
//                                                                      //
//  Only the baskets for the requested entry range are put in the cache //
//                                                                      //
//  For each Tree being processed a TTreeFilePrefetch object is created.//
//  This object is automatically deleted when the Tree is deleted or    //
//  when the file is deleted.
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TTreeFilePrefetch.h"
#include "TTree.h"
#include "TBranch.h"
#include "TLeaf.h"

ClassImp(TTreeFilePrefetch)

//______________________________________________________________________________
TTreeFilePrefetch::TTreeFilePrefetch() : TFilePrefetch(),
   fTree(0),
   fEntryMax(1)
{
   // Default Constructor.
}

//______________________________________________________________________________
TTreeFilePrefetch::TTreeFilePrefetch(TTree *tree, Int_t buffersize) : TFilePrefetch(tree->GetCurrentFile(),buffersize),
   fTree(tree),
   fEntryMax(tree->GetEntries())
{
   // Constructor.
}

//______________________________________________________________________________
TTreeFilePrefetch::TTreeFilePrefetch(const TTreeFilePrefetch &pf) : TFilePrefetch(pf)
{
   // Copy Constructor.
}

//______________________________________________________________________________
TTreeFilePrefetch::~TTreeFilePrefetch()
{
   // destructor. (in general called by the TFile destructor
}

//______________________________________________________________________________
TTreeFilePrefetch& TTreeFilePrefetch::operator=(const TTreeFilePrefetch& pf)
{
   // Assignment.

   if (this != &pf) TFilePrefetch::operator=(pf);
   return *this;
}         

//_____________________________________________________________________________
Bool_t TTreeFilePrefetch::ReadBuffer(char *buf, Long64_t pos, Int_t len)
{
   // Read buffer at position pos.
   // If pos is in the list of prefetched blocks read from fBuffer,
   // otherwise normal read from file. Returns kTRUE in case of failure.
   //This function overloads TFilePrefetch::ReadBuffer.
   
   if (fNseek > 0 && !fIsSorted) {
      TFilePrefetch::Sort();
      if (fFile->ReadBuffers(fBuffer,fPos,fLen,fNb))
         return kTRUE;
   }
   Int_t loc = (Int_t)TMath::BinarySearch(fNseek,fSeekSort,pos);
   if (loc >= 0 && loc <fNseek && pos == fSeekSort[loc]) {
      memcpy(buf,&fBuffer[fSeekPos[loc]],len);
      fFile->Seek(pos+len);
      return kFALSE;
   }
   
   //not found in cache. Register this block
   if (!Register(pos)) return kTRUE;
   return ReadBuffer(buf,pos,len);
}

//_____________________________________________________________________________
Bool_t TTreeFilePrefetch::Register(Long64_t offset)
{ 
   // Register branch owning basket at starting position offset
   // return kTRUE if the registration succeeds
   
   //reset cache when reaching the maximum cache size
   if (fNtot+30000 > fBufferSize) TFilePrefetch::Prefetch(0,0);
   Int_t nleaves = fTree->GetListOfLeaves()->GetEntriesFast();
   //loop on all the branches to find the branch with a buffer starting at offset
   Bool_t status = kFALSE;
   TBranch *branch = 0;
   for (Int_t i=0;i<nleaves;i++) {
      TLeaf *leaf = (TLeaf*)fTree->GetListOfLeaves()->At(i);
      branch = leaf->GetBranch();
      //if (branch->GetListOfBranches()->GetEntriesFast() > 0) continue;
      Int_t nb = branch->GetMaxBaskets();
      Int_t *lbaskets   = branch->GetBasketBytes();
      Long64_t *entries = branch->GetBasketEntry();
      //we have found the branch. We now register all its baskets
      //from the requested offset to the basket below fEntrymax
      for (Int_t j=0;j<nb;j++) {
         if (branch->GetBasketSeek(j) == offset) {
            for (Int_t k=j;k<nb;k++) {
               Long64_t pos = branch->GetBasketSeek(k);
               Int_t len = lbaskets[k];
               if (pos <= 0) continue;
               if (fNtot+len > fBufferSize || entries[k] > fEntryMax) {
                  return status;
               }
               TFilePrefetch::Prefetch(pos,len);
               status = kTRUE;
            }
            if (gDebug > 0) printf("registering branch %s offset=%lld\n",branch->GetName(),offset);
            return status;
         }
      }
   }
   if (status && branch && gDebug > 0) printf("registering branch %s offset=%lld\n",branch->GetName(),offset);
   return status;
}

//_____________________________________________________________________________
void TTreeFilePrefetch::SetEntryMax(Long64_t emax)
{
   // Set the maximum entry number to be processed
   // this information helps to optimize the number of baskets to read
   // when prefetching the branch buffers.
   
   fEntryMax = emax;
}

//_____________________________________________________________________________
void TTreeFilePrefetch::SetTree(TTree *tree)
{
   //change current tree
   
   fTree = tree;
}
