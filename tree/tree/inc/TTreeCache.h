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

#include <vector>

class TTree;
class TBranch;
class TObjArray;

class TTreeCache : public TFileCacheRead {

public:
   enum EPrefillType { kNoPrefill, kAllBranches };

protected:
   Long64_t     fEntryMin{0};         ///<! first entry in the cache
   Long64_t     fEntryMax{1};         ///<! last entry in the cache
   Long64_t     fEntryCurrent{-1};    ///<! current lowest entry number in the cache
   Long64_t     fEntryNext{-1};       ///<! next entry number where cache must be filled
   Long64_t     fCurrentClusterStart{-1}; ///<! Start of the cluster(s) where the current content was picked out
   Long64_t     fNextClusterStart{-1};    ///<! End+1 of the cluster(s) where the current content was picked out
   Int_t        fNbranches{0};        ///<! Number of branches in the cache
   Int_t        fNReadOk{0};          ///<  Number of blocks read and found in the cache
   Int_t        fNMissReadOk{0};      ///<  Number of blocks read, not found in the primary cache, and found in the secondary cache.
   Int_t        fNReadMiss{0};        ///<  Number of blocks read and not found in the cache
   Int_t        fNMissReadMiss{0};    ///<  Number of blocks read and not found in either cache.
   Int_t        fNReadPref{0};        ///<  Number of blocks that were prefetched
   Int_t        fNMissReadPref{0};    ///<  Number of blocks read into the secondary ("miss") cache.
   TObjArray   *fBranches{nullptr};   ///<! List of branches to be stored in the cache
   TList       *fBrNames{nullptr};    ///<! list of branch names in the cache
   TTree       *fTree{nullptr};       ///<! pointer to the current Tree
   bool         fIsLearning{true};   ///<! true if cache is in learning mode
   bool         fIsManual{false};    ///<! true if cache is StopLearningPhase was used
   bool         fFirstBuffer{true};  ///<! true if first buffer is used for prefetching
   bool         fOneTime{false};     ///<! used in the learning phase
   bool         fReverseRead{false}; ///<! reading in reverse mode
   Int_t        fFillTimes{0};        ///<! how many times we can fill the current buffer
   bool         fFirstTime{true};    ///<! save the fact that we processes the first entry
   Long64_t     fFirstEntry{-1};      ///<! save the value of the first entry
   bool         fReadDirectionSet{false}; ///<! read direction established
   bool         fEnabled{true};      ///<! cache enabled for cached reading
   EPrefillType fPrefillType;         ///<  Whether a pre-filling is enabled (and if applicable which type)
   static Int_t fgLearnEntries;       ///<  number of entries used for learning mode
   bool         fAutoCreated{false}; ///<! true if cache was automatically created

   bool         fLearnPrefilling{false}; ///<! true if we are in the process of executing LearnPrefill

   // These members hold cached data for missed branches when miss optimization
   // is enabled.  Pointers are only initialized if the miss cache is enabled.
   bool     fOptimizeMisses{false}; ///<! true if we should optimize cache misses.
   Long64_t fFirstMiss{-1};          ///<! set to the event # of the first miss.
   Long64_t fLastMiss{-1};           ///<! set to the event # of the last miss.

   // Representation of a positioned buffer IO.
   // {0,0} designates the uninitialized - or invalid - buffer.
   struct IOPos {
      IOPos(Long64_t pos, Int_t len) : fPos(pos), fLen(len) {}

      Long64_t fPos{0}; //! Position in file of cache entry.
      Int_t fLen{0};    //! Length of cache entry.
   };

   struct MissCache {
      struct Entry {
         Entry(IOPos io) : fIO(io) {}

         IOPos fIO;
         ULong64_t fIndex{0}; ///<! Location in fData corresponding to this entry.
         friend bool operator<(const Entry &a, const Entry &b) { return a.fIO.fPos < b.fIO.fPos; }
      };
      std::vector<Entry> fEntries;      ///<! Description of buffers in the miss cache.
      std::vector<TBranch *> fBranches; ///<! list of branches that we read on misses.
      std::vector<char> fData;          ///<! Actual data in the cache.

      void clear()
      {
         fEntries.clear();
         fBranches.clear();
         fData.clear();
      }
   };

   std::unique_ptr<MissCache> fMissCache; ///<! Cache contents for misses

private:
   TTreeCache(const TTreeCache &) = delete; ///< this class cannot be copied
   TTreeCache &operator=(const TTreeCache &) = delete;

   // These are all functions related to the "miss cache": this attempts to
   // optimize ROOT's behavior when the TTreeCache has a cache miss.  In this
   // case, we try to read several branches for the event with the miss.
   //
   // The miss cache is more CPU-intensive than the rest of the TTreeCache code;
   // for local work (i.e., laptop with SSD), this CPU cost may outweight the
   // benefit.
   bool CheckMissCache(char *buf, Long64_t pos,
                         int len); ///< Check the miss cache for a particular buffer, fetching if deemed necessary.
   bool FillMissCache();         ///< Fill the miss cache from the current set of active branches.
   bool CalculateMissCache();    ///< Calculate the appropriate miss cache to fetch; helper function for FillMissCache
   IOPos  FindBranchBasketPos(TBranch &, Long64_t entry); ///< Given a branch and an entry, determine the file location
                                                          ///< (offset / size) of the corresponding basket.
   TBranch *CalculateMissEntries(Long64_t, int, bool);    ///< Given an file read, try to determine the corresponding branch.
   bool     ProcessMiss(Long64_t pos, int len); ///<! Given a file read not in the miss cache, handle (possibly) loading the data.

public:

   TTreeCache();
   TTreeCache(TTree *tree, Int_t bufsize=0);
   ~TTreeCache() override;
   Int_t                AddBranch(TBranch *b, bool subgbranches = false) override;
   Int_t                AddBranch(const char *branch, bool subbranches = false) override;
   virtual Int_t        DropBranch(TBranch *b, bool subbranches = false);
   virtual Int_t        DropBranch(const char *branch, bool subbranches = false);
   virtual void         Disable() {fEnabled = false;}
   virtual void         Enable() {fEnabled = true;}
   bool                 GetOptimizeMisses() const { return fOptimizeMisses; }
   const TObjArray     *GetCachedBranches() const { return fBranches; }
   EPrefillType         GetConfiguredPrefillType() const;
   Double_t             GetEfficiency() const;
   Double_t             GetEfficiencyRel() const;
   virtual Int_t        GetEntryMin() const {return fEntryMin;}
   virtual Int_t        GetEntryMax() const {return fEntryMax;}
   static Int_t         GetLearnEntries();
   virtual EPrefillType GetLearnPrefill() const {return fPrefillType;}
   Double_t             GetMissEfficiency() const;
   Double_t             GetMissEfficiencyRel() const;
   TTree               *GetTree() const {return fTree;}
   bool                 IsAutoCreated() const {return fAutoCreated;}
   virtual bool         IsEnabled() const {return fEnabled;}
   bool                 IsLearning() const override {return fIsLearning;}

   virtual bool         FillBuffer();
   Int_t                LearnBranch(TBranch *b, bool subgbranches = false) override;
   virtual void         LearnPrefill();

   void                 Print(Option_t *option="") const override;
   Int_t                ReadBuffer(char *buf, Long64_t pos, Int_t len) override;
   virtual Int_t        ReadBufferNormal(char *buf, Long64_t pos, Int_t len);
   virtual Int_t        ReadBufferPrefetch(char *buf, Long64_t pos, Int_t len);
   virtual void         ResetCache();
   void                 ResetMissCache(); // Reset the miss cache.
   void                 SetAutoCreated(bool val) {fAutoCreated = val;}
   Int_t                SetBufferSize(Long64_t buffersize) override;
   virtual void         SetEntryRange(Long64_t emin,   Long64_t emax);
   void                 SetFile(TFile *file, TFile::ECacheAction action=TFile::kDisconnect) override;
   virtual void         SetLearnPrefill(EPrefillType type = kNoPrefill);
   static void          SetLearnEntries(Int_t n = 10);
   void                 SetOptimizeMisses(bool opt);
   void                 StartLearningPhase();
   virtual void         StopLearningPhase();
   virtual void         UpdateBranches(TTree *tree);

   ClassDefOverride(TTreeCache,3)  //Specialization of TFileCacheRead for a TTree
};

#endif
