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

#ifndef ROOT_TChain
#include "TChain.h"
#endif

class TDSet;
class TDrawFeedback;
class TList;
class TProof;

class TProofChain : public TChain {

public:
   // TProofChain constants
   enum { kOwnsChain   = BIT(19) };

private:
   void           FillDrawAttributes(TProof *p);

protected:
   TChain        *fChain;            // mother chain: needed for the browsing list
   TDSet         *fSet;              // TDSet
   TDrawFeedback *fDrawFeedback;     // feedback handler

public:
   TProofChain();
   TProofChain(TChain *chain, Bool_t gettreeheader);
   TProofChain(TDSet *dset, Bool_t gettreeheader);
   virtual ~TProofChain();

   virtual void         Browse(TBrowser *b);
   Int_t                Debug() const {return fDebug;}
   virtual Long64_t     Draw(const char *varexp, const TCut &selection, Option_t *option=""
                             ,Long64_t nentries=1000000000, Long64_t firstentry=0);
   virtual Long64_t     Draw(const char *varexp, const char *selection, Option_t *option=""
                             ,Long64_t nentries=1000000000, Long64_t firstentry=0); // *MENU*
   virtual void         Draw(Option_t *opt) { Draw(opt, "", "", 1000000000, 0); }
   virtual TBranch     *FindBranch(const char *name);
   virtual TLeaf       *FindLeaf(const char *name);
   virtual TBranch     *GetBranch(const char *name);
   virtual Bool_t       GetBranchStatus(const char *branchname) const;
   virtual Long64_t     GetEntries() const;
   virtual Long64_t     GetEntries(const char *sel);
   virtual TList       *GetListOfClones() { return 0; }
   virtual TObjArray   *GetListOfBranches() {return (fTree ? fTree->GetListOfBranches() : (TObjArray *)0); }
   virtual TObjArray   *GetListOfLeaves()   {return (fTree ? fTree->GetListOfLeaves() : (TObjArray *)0);}
   virtual TList       *GetListOfFriends()    const {return 0;}
   virtual TList       *GetListOfAliases() const {return 0;}

    // GetMakeClass is left non-virtual for efficiency reason.
    // Making it virtual affects the performance of the I/O
           Int_t        GetMakeClass() const {return fMakeClass;}

   TVirtualTreePlayer  *GetPlayer();
   virtual Long64_t     GetReadEntry()  const;
   Bool_t               HasTreeHeader() const { return (fTree ? kTRUE : kFALSE); }
   virtual Long64_t     Process(const char *filename, Option_t *option="",
                                Long64_t nentries=1000000000, Long64_t firstentry=0); // *MENU*
   virtual void         Progress(Long64_t total, Long64_t processed);
   virtual Long64_t     Process(TSelector *selector, Option_t *option="",
                                Long64_t nentries=1000000000, Long64_t firstentry=0);
   virtual void         SetDebug(Int_t level=1, Long64_t min=0, Long64_t max=9999999); // *MENU*
   virtual void         SetEventList(TEventList *evlist) { fEventList = evlist; }
   virtual void         SetEntryList(TEntryList *enlist, const Option_t *) { fEntryList = enlist; }
   virtual void         SetName(const char *name); // *MENU*
   virtual void         ConnectProof();
   virtual void         ReleaseProof();

   ClassDef(TProofChain,0)  //TChain proxy for running chains on PROOF
};

#endif
