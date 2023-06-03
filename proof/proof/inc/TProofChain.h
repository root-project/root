// @(#)root/proof:$Id$
// Author: G. Ganis Nov 2006

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofChain
#define ROOT_TProofChain


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofChain                                                          //
//                                                                      //
// A TChain proxy on PROOF.                                             //
// Uses an internal TDSet to handle processing.                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TChain.h"

class TDSet;
class TList;
class TProof;

class TProofChain : public TChain {

public:
   // TProofChain constants
   enum { kOwnsChain   = BIT(19) };

private:
   void           AddAliases();
   void           FillDrawAttributes(TProof *p);

protected:
   TChain        *fChain;            // mother chain: needed for the browsing list
   TDSet         *fSet;              // TDSet

public:
   TProofChain();
   TProofChain(TChain *chain, Bool_t gettreeheader);
   TProofChain(TDSet *dset, Bool_t gettreeheader);
   ~TProofChain() override;

   void         Browse(TBrowser *b) override;
   Int_t                Debug() const {return fDebug;}
   Long64_t     Draw(const char *varexp, const TCut &selection, Option_t *option=""
                             ,Long64_t nentries=TTree::kMaxEntries, Long64_t firstentry=0) override;
   Long64_t     Draw(const char *varexp, const char *selection, Option_t *option=""
                             ,Long64_t nentries=TTree::kMaxEntries, Long64_t firstentry=0) override; // *MENU*
   void         Draw(Option_t *opt) override { Draw(opt, "", "", TTree::kMaxEntries, 0); }
   TBranch     *FindBranch(const char *name) override;
   TLeaf       *FindLeaf(const char *name) override;
   TBranch     *GetBranch(const char *name) override;
   Bool_t       GetBranchStatus(const char *branchname) const override;
   Long64_t     GetEntries() const override;
   Long64_t     GetEntries(const char *sel) override;
   TList       *GetListOfClones() override { return 0; }
   TObjArray   *GetListOfBranches() override {return (fTree ? fTree->GetListOfBranches() : (TObjArray *)0); }
   TObjArray   *GetListOfLeaves() override   {return (fTree ? fTree->GetListOfLeaves() : (TObjArray *)0);}
   TList       *GetListOfFriends()    const override {return 0;}
   TList       *GetListOfAliases() const override {return 0;}

    // GetMakeClass is left non-virtual for efficiency reason.
    // Making it virtual affects the performance of the I/O
           Int_t        GetMakeClass() const {return fMakeClass;}

   TVirtualTreePlayer  *GetPlayer();
   Long64_t     GetReadEntry()  const override;
   Bool_t               HasTreeHeader() const { return (fTree ? kTRUE : kFALSE); }
   Long64_t     Process(const char *filename, Option_t *option="",
                                Long64_t nentries=TTree::kMaxEntries, Long64_t firstentry=0) override; // *MENU*
   virtual void         Progress(Long64_t total, Long64_t processed);
   Long64_t     Process(TSelector *selector, Option_t *option="",
                                Long64_t nentries=TTree::kMaxEntries, Long64_t firstentry=0) override;
   void         SetDebug(Int_t level=1, Long64_t min=0, Long64_t max=9999999) override; // *MENU*
   void         SetEventList(TEventList *evlist) override { fEventList = evlist; }
   void         SetEntryList(TEntryList *enlist, const Option_t *) override { fEntryList = enlist; }
   void         SetName(const char *name) override; // *MENU*
   virtual void         ConnectProof();
   virtual void         ReleaseProof();

   ClassDefOverride(TProofChain,0)  //TChain proxy for running chains on PROOF
};

#endif
