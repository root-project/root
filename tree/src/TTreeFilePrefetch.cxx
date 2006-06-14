// @(#)root/tree:$Name:  $:$Id: TTreeFilePrefetch.cxx,v 1.6 2006/06/12 09:02:03 brun Exp $
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
//  TTree::Process and TSelectors) when in the learning phase.          //
//  The learning phase is by default the first 1 per cent of entries.   //
//  It can be changed via TTreeFileFrefetch::SetLearnRatio.             //
//                                                                      //
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

Double_t TTreeFilePrefetch::fgLearnRatio = 0.01;

ClassImp(TTreeFilePrefetch)

//______________________________________________________________________________
TTreeFilePrefetch::TTreeFilePrefetch() : TFilePrefetch(),
   fEntryMin(0),
   fEntryMax(1),
   fEntryNext(1),
   fNbranches(0),
   fBranches(0),
   fIsLearning(kTRUE)
{
   // Default Constructor.
}

//______________________________________________________________________________
TTreeFilePrefetch::TTreeFilePrefetch(TTree *tree, Int_t buffersize) : TFilePrefetch(tree->GetCurrentFile(),buffersize),
   fEntryMin(0),
   fEntryMax(tree->GetEntries()),
   fEntryNext(0),
   fNbranches(0),
   fBranches(0),
   fIsLearning(kTRUE)
{
   // Constructor.
   fEntryNext = Long64_t(fgLearnRatio*(fEntryMax-fEntryMin));
   if (fEntryNext == fEntryMin) fEntryNext++;
   Int_t nleaves = tree->GetListOfLeaves()->GetEntries();
   fBranches = new TBranch*[nleaves];
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
   
   delete [] fBranches;
}

//______________________________________________________________________________
TTreeFilePrefetch& TTreeFilePrefetch::operator=(const TTreeFilePrefetch& pf)
{
   // Assignment.

   if (this != &pf) TFilePrefetch::operator=(pf);
   return *this;
}         

//_____________________________________________________________________________
void TTreeFilePrefetch::AddBranch(TBranch *b)
{
   //add a branch to the list of branches to be stored in the cache
   //this function is called by TBranch::GetBasket
      
   if (!fIsLearning) return;

   //Is branch already in the cache?
   Bool_t isNew = kTRUE;
   for (int i=0;i<fNbranches;i++) {
      if (fBranches[i] == b) {isNew = kFALSE; break;}
   }
   if (isNew) {
      fBranches[fNbranches] = b;
      fNbranches++;
      if (gDebug > 0) printf("Entry: %lld, registering branch: %s\n",b->GetTree()->GetReadEntry(),b->GetName());
   }
}

//_____________________________________________________________________________
void TTreeFilePrefetch::Clear(Option_t *)
{
   //clear the cache (called by TChain::LoadTree)
   
   Prefetch(0,0);
   fNbranches = 0;
   fIsLearning = kTRUE;
}
   

//_____________________________________________________________________________
Bool_t TTreeFilePrefetch::FillBuffer()
{
   //Fill the cache buffer with the branchse in the cache
   
   if (fNbranches <= 0) return kFALSE;
   TTree *tree = fBranches[0]->GetTree();
   Long64_t entry = tree->GetReadEntry();
   if (fIsLearning && entry < fEntryNext) return kFALSE;
   //compute total size of the branches stored in cache
   Long64_t totbytes = 0;
   Int_t i;
   for (i=0;i<fNbranches;i++) {
      totbytes += fBranches[i]->GetZipBytes();
   }
   //estimate number of entries that can fit in the cache
   Long64_t oldEntryNext = fEntryNext;
   fEntryNext = entry + tree->GetEntries()*fBufferSize/totbytes;
   if (fEntryNext > fEntryMax) fEntryNext = fEntryMax+1;
         
   //clear cache buffer
   TFilePrefetch::Prefetch(0,0);
   //store baskets
   for (i=0;i<fNbranches;i++) {
      TBranch *b = fBranches[i];
      Int_t nb = b->GetMaxBaskets();
      Int_t *lbaskets   = b->GetBasketBytes();
      Long64_t *entries = b->GetBasketEntry();
      if (!lbaskets || !entries) continue;
      //we have found the branch. We now register all its baskets
      //from the requested offset to the basket below fEntrymax
      for (Int_t j=0;j<nb;j++) {
         Long64_t pos = b->GetBasketSeek(j);
         Int_t len = lbaskets[j];
         if (pos <= 0 || len <= 0) continue;
         if (entries[j] >= oldEntryNext && entries[j] < fEntryNext) {
            TFilePrefetch::Prefetch(pos,len);
         }
      }
      if (gDebug > 0) printf("Entry: %lld, registering baskets branch %s\n",entry,fBranches[i]->GetName());
   }
   fIsLearning = kFALSE;
   return kTRUE;
}


//_____________________________________________________________________________
Double_t TTreeFilePrefetch::GetLearnRatio()
{
   //static function returning fgLearnRatio
   //see SetLearnRatio
   
   return fgLearnRatio;
}

//_____________________________________________________________________________
TTree *TTreeFilePrefetch::GetTree() const
{
   //return Tree in the cache
   if (fNbranches <= 0) return 0;
   return fBranches[0]->GetTree();
}

//_____________________________________________________________________________
Bool_t TTreeFilePrefetch::ReadBuffer(char *buf, Long64_t pos, Int_t len)
{
   // Read buffer at position pos.
   // If pos is in the list of prefetched blocks read from fBuffer,
   // then try to fill the cache from the list of selected branches,
   // otherwise normal read from file. Returns kTRUE in case of failure.
   // This function overloads TFilePrefetch::ReadBuffer.
   // It returns kFALSE if the requested block is in the cache
   
   //Is request already in the cache?
   Bool_t inCache = !TFilePrefetch::ReadBuffer(buf,pos,len);
   if (inCache) return kFALSE;
   
   //not found in cache. Do we need to fill the cache?
   Bool_t bufferFilled = FillBuffer();
   if (bufferFilled) return TFilePrefetch::ReadBuffer(buf,pos,len);
   return kTRUE;
}

//_____________________________________________________________________________
void TTreeFilePrefetch::SetEntryRange(Long64_t emin, Long64_t emax)
{
   // Set the minimum and maximum entry number to be processed
   // this information helps to optimize the number of baskets to read
   // when prefetching the branch buffers.
   
   fEntryMin  = emin;
   fEntryMax  = emax;
   Long64_t learn = Long64_t(fgLearnRatio*(fEntryMax-fEntryMin));
   if (learn < 2) learn = 2;
   fEntryNext = emin + learn;
   fIsLearning = kTRUE;
   fNbranches = 0;
   if (gDebug > 0) printf("SetEntryRange: fEntryMin=%lld, fEntryMax=%lld, fEntryNext=%lld\n",fEntryMin,fEntryMax,fEntryNext);
   
}

//_____________________________________________________________________________
void TTreeFilePrefetch::SetLearnRatio(Double_t ratio)
{
   // Static function to set the fraction of entries to be used in learning mode
   // The default value for ratio is 0.01 (1 per cent).
   // In case the ratio specified is such that less than 2 entries
   // participate to the learning, a minimum of 2 entries are used.
   
   fgLearnRatio = ratio;
}
