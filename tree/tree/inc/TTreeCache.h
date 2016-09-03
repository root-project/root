// @(#)root/tree:$Id$
// Author: Rene Brun   04/06/2006

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTreeCache
#define ROOT_TTreeCache


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTreeCache                                                           //
//                                                                      //
// Specialization of TFileCacheRead for a TTree                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TFileCacheRead.h"
#include "TObjArray.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

class TTree;
class TBranch;

class TTreeCache : public TFileCacheRead {

public:
   enum EPrefillType { kNoPrefill, kAllBranches };

protected:
   Long64_t        fEntryMin;         ///<! first entry in the cache
   Long64_t        fEntryMax;         ///<! last entry in the cache
   Long64_t        fEntryCurrent;     ///<! current lowest entry number in the cache
   Long64_t        fEntryNext;        ///<! next entry number where cache must be filled
   Int_t           fNbranches;        ///<! Number of branches in the cache
   Int_t           fNReadOk;          ///<  Number of blocks read and found in the cache
   Int_t           fNMissReadOk{0};   ///<  Number of blocks read, not found in the primary cache, and found in the secondary cache.
   Int_t           fNReadMiss;        ///<  Number of blocks read and not found in the cache
   Int_t           fNMissReadMiss{0}; ///<  Number of blocks read and not found in either cache.
   Int_t           fNReadPref;        ///<  Number of blocks that were prefetched
   Int_t           fNMissReadPref{0}; ///<  Number of blocks read into the secondary ("miss") cache.
   TObjArray      *fBranches;         ///<! List of branches to be stored in the cache
   TList          *fBrNames;          ///<! list of branch names in the cache
   TTree          *fTree;             ///<! pointer to the current Tree
   Bool_t          fIsLearning;       ///<! true if cache is in learning mode
   Bool_t          fIsManual;         ///<! true if cache is StopLearningPhase was used
   Bool_t          fFirstBuffer;      ///<! true if first buffer is used for prefetching
   Bool_t          fOneTime;          ///<! used in the learning phase
   Bool_t          fReverseRead;      ///<! reading in reverse mode
   Int_t           fFillTimes;        ///<! how many times we can fill the current buffer
   Bool_t          fFirstTime;        ///<! save the fact that we processes the first entry
   Long64_t        fFirstEntry;       ///<! save the value of the first entry
   Bool_t          fReadDirectionSet; ///<! read direction established
   Bool_t          fEnabled;          ///<! cache enabled for cached reading
   EPrefillType    fPrefillType;      ///<  Whether a pre-filling is enabled (and if applicable which type)
   static  Int_t   fgLearnEntries;    ///<  number of entries used for learning mode
   Bool_t          fAutoCreated;      ///<! true if cache was automatically created

   // These members hold cached data for missed branches when miss optimization
   // is enabled.  Pointers are only initialized if the miss cache is enabled.
   bool            fOptimizeMisses {false}; //! true if we should optimize cache misses.
   Long64_t        fFirstMiss {-1}; //! set to the event # of the first miss.
   Long64_t        fLastMiss  {-1}; //! set to the event # of the last miss.
   std::unique_ptr<std::vector<TBranch*>> fMissBranches; //! list of branches that we read on misses.
   std::unique_ptr<std::vector<char>> fMissCache; //! Cache contents for misses
   // TODO: there's a 1-1 correspondence between an element in fEntries and an element in fEntryOffsets
   // Instead of munging std::pairs, put this into a simple struct.
   std::unique_ptr<std::vector<std::pair<ULong64_t, UInt_t>>> fEntries;  //! Buffers in the miss cache.
   std::unique_ptr<std::vector<size_t>> fEntryOffsets;  //! Map from (offset, pos) in fEntries to memory location in fMissCache

private:
   TTreeCache(const TTreeCache &);            //this class cannot be copied
   TTreeCache& operator=(const TTreeCache &);

   // These are all functions related to the "miss cache": this attempts to
   // optimize ROOT's behavior when the TTreeCache has a cache miss.  In this
   // case, we try to read several branches for the event with the miss.
   //
   // The miss cache is more CPU-intensive than the rest of the TTreeCache code;
   // for local work (i.e., laptop with SSD), this CPU cost may outweight the
   // benefit.
   bool CheckMissCache(char *buf, Long64_t pos, int len);  // Check the miss cache for a particular buffer, fetching if deemed necessary.
   bool FillMissCache();  // Fill the miss cache from the current set of active branches.
   bool CalculateMissCache();  // Calculate the appropriate miss cache to fetch; helper function for FillMissCache
   std::pair<ULong64_t, UInt_t> FindBranchBasket(TBranch &);  // Given a branch, determine the location of its basket for the current entry.
   TBranch* CalculateMissEntries(Long64_t, int, bool);   // Given an file read, try to determine the corresponding branch.
   bool ProcessMiss(Long64_t pos, int len);   // Given a file read not in the miss cache, handle (possibly) loading the data.

public:

   TTreeCache();
   TTreeCache(TTree *tree, Int_t buffersize=0);
   virtual ~TTreeCache();
   virtual Int_t        AddBranch(TBranch *b, Bool_t subgbranches = kFALSE);
   virtual Int_t        AddBranch(const char *branch, Bool_t subbranches = kFALSE);
   virtual Int_t        DropBranch(TBranch *b, Bool_t subbranches = kFALSE);
   virtual Int_t        DropBranch(const char *branch, Bool_t subbranches = kFALSE);
   virtual void         Disable() {fEnabled = kFALSE;}
   virtual void         Enable() {fEnabled = kTRUE;}
   void                 SetOptimizeMisses(bool opt);
   bool                 GetOptimizeMisses() const {return fOptimizeMisses;}
   const TObjArray     *GetCachedBranches() const { return fBranches; }
   EPrefillType         GetConfiguredPrefillType() const;
   Double_t             GetEfficiency() const;
   Double_t             GetEfficiencyRel() const;
   double               GetMissEfficiency() const;
   double               GetMissEfficiencyRel() const;
   virtual Int_t        GetEntryMin() const {return fEntryMin;}
   virtual Int_t        GetEntryMax() const {return fEntryMax;}
   static Int_t         GetLearnEntries();
   virtual EPrefillType GetLearnPrefill() const {return fPrefillType;}
   TTree               *GetTree() const {return fTree;}
   Bool_t               IsAutoCreated() const {return fAutoCreated;}
   virtual Bool_t       IsEnabled() const {return fEnabled;}
   virtual Bool_t       IsLearning() const {return fIsLearning;}

   virtual Bool_t       FillBuffer();
   virtual void         LearnPrefill();

   virtual void         Print(Option_t *option="") const;
   virtual Int_t        ReadBuffer(char *buf, Long64_t pos, Int_t len);
   virtual Int_t        ReadBufferNormal(char *buf, Long64_t pos, Int_t len);
   virtual Int_t        ReadBufferPrefetch(char *buf, Long64_t pos, Int_t len);
   virtual void         ResetCache();
   void                 ResetMissCache();  // Reset the miss cache.
   void                 SetAutoCreated(Bool_t val) {fAutoCreated = val;}
   virtual Int_t        SetBufferSize(Int_t buffersize);
   virtual void         SetEntryRange(Long64_t emin,   Long64_t emax);
   virtual void         SetFile(TFile *file, TFile::ECacheAction action=TFile::kDisconnect);
   virtual void         SetLearnPrefill(EPrefillType type = kNoPrefill);
   static void          SetLearnEntries(Int_t n = 10);
   void                 StartLearningPhase();
   virtual void         StopLearningPhase();
   virtual void         UpdateBranches(TTree *tree);

   ClassDef(TTreeCache,2)  //Specialization of TFileCacheRead for a TTree
};

#endif
