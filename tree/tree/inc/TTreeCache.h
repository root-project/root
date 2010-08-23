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

#ifndef ROOT_TFileCacheRead
#include "TFileCacheRead.h"
#endif
#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif

class TTree;
class TBranch;

class TTreeCache : public TFileCacheRead {

protected:
   Long64_t        fEntryMin;    //! first entry in the cache
   Long64_t        fEntryMax;    //! last entry in the cache
   Long64_t        fEntryCurrent;//! current lowest entry number in the cache
   Long64_t        fEntryNext;   //! next entry number where cache must be filled
   Long64_t        fZipBytes;    //! Total compressed size of branches in cache
   Int_t           fNbranches;   //! Number of branches in the cache
   Int_t           fNReadOk;     //Number of blocks read and found in the cache
   Int_t           fNReadMiss;   //Number of blocks read and not found in the chache
   Int_t           fNReadPref;   //Number of blocks that were prefetched
   TObjArray      *fBranches;    //! List of branches to be stored in the cache
   TList          *fBrNames;     //! list of branch names in the cache
   TTree          *fOwner;       //! pointer to the owner Tree/chain
   TTree          *fTree;        //! pointer to the current Tree
   Bool_t          fIsLearning;  //! true if cache is in learning mode
   Bool_t          fIsManual;    //! true if cache is StopLearningPhase was used
   static  Int_t fgLearnEntries; //Number of entries used for learning mode

private:
   TTreeCache(const TTreeCache &);            //this class cannot be copied
   TTreeCache& operator=(const TTreeCache &);

public:
   TTreeCache();
   TTreeCache(TTree *tree, Int_t buffersize=0);
   virtual ~TTreeCache();
   virtual void        AddBranch(TBranch *b, Bool_t subbranches = kFALSE);
   virtual void        AddBranch(const char *branch, Bool_t subbranches = kFALSE);
   Double_t            GetEfficiency() const;
   Double_t            GetEfficiencyRel() const;
   static Int_t        GetLearnEntries();

   virtual Bool_t      FillBuffer();
   TTree              *GetOwner() const;
   TTree              *GetTree() const;
   virtual Bool_t      IsLearning() const {return fIsLearning;}
   virtual void        Print(Option_t *option="") const;
   virtual Int_t       ReadBuffer(char *buf, Long64_t pos, Int_t len);
   virtual void        ResetCache();
   virtual void        SetEntryRange(Long64_t emin,   Long64_t emax);
   static void         SetLearnEntries(Int_t n = 10);
   void                StartLearningPhase();
   virtual void        StopLearningPhase();
   virtual void        UpdateBranches(TTree *tree, Bool_t owner = kFALSE);

   ClassDef(TTreeCache,2)  //Specialization of TFileCacheRead for a TTree
};

#endif
