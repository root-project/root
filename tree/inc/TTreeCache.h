// @(#)root/tree:$Name:  $:$Id: TTreeCache.h,v 1.7 2006/06/16 11:01:16 brun Exp $
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

class TTree;
class TBranch;

class TTreeCache : public TFileCacheRead {

protected:
   Long64_t        fEntryMin;    //! first entry in the cache
   Long64_t        fEntryMax;    //! last entry in the cache
   Long64_t        fEntryNext;   //! next entry number where cache must be filled
   Long64_t        fZipBytes;    //! Total compressed size of branches in cache
   Int_t           fNbranches;   //! Number of branches in the cache
   TBranch       **fBranches;    //! [fNbranches] List of branches to be stored in the cache
   TList          *fBrNames;     //! list of branch names in the cache
   TTree          *fOwner;       //! pointer to the owner Tree/chain
   TTree          *fTree;        //! pointer to the current Tree
   Bool_t          fIsLearning;  //! true if cache is in learning mode
   static  Int_t fgLearnEntries; //Number of entries used for learning mode

protected:
   TTreeCache(const TTreeCache &);            //this class cannot be copied
   TTreeCache& operator=(const TTreeCache &);

public:
   TTreeCache();
   TTreeCache(TTree *tree, Int_t buffersize=0);
   virtual ~TTreeCache();
   void                AddBranch(TBranch *b);
   static Int_t        GetLearnEntries();
   Bool_t              FillBuffer();
   TTree              *GetTree() const;
   Bool_t              IsLearning() const {return fIsLearning;}
   virtual Bool_t      ReadBuffer(char *buf, Long64_t pos, Int_t len);
   void                SetEntryRange(Long64_t emin,   Long64_t emax);
   static void         SetLearnEntries(Int_t n = 100);
   void                UpdateBranches(TTree *tree);
           
   ClassDef(TTreeCache,1)  //Specialization of TFileCacheRead for a TTree 
};

#endif
