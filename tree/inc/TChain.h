// @(#)root/tree:$Name:  $:$Id: TChain.h,v 1.37 2004/06/22 06:42:11 brun Exp $
// Author: Rene Brun   03/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TChain
#define ROOT_TChain


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TChain                                                               //
//                                                                      //
// A chain of TTrees.                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TTree
#include "TTree.h"
#endif
#ifndef ROOT_TCache
#include "TCache.h"
#endif

class TFile;
class TBrowser;
class TCut;


class TChain : public TTree {

protected:
    Int_t       fTreeOffsetLen;     //  Current size of fTreeOffset array
    Int_t       fNtrees;            //  Number of Trees
    Int_t       fTreeNumber;        //! Current Tree number in fTreeOffset table
    Long64_t    *fTreeOffset;       //[fTreeOffsetLen]Array of variables
    Int_t        fMaxCacheSize;     //! Max cache size passed to TFile's
    Int_t        fPageSize;         //! Cache page size passed to TFile's
    Bool_t       fCanDeleteRefs;    //! if true, TProcessIDs are deleted when closing a file
    TTree       *fTree;             //! Pointer to current tree
    TFile       *fFile;             //! Pointer to current file
    TObjArray   *fFiles;            //->  List of file names containing the Trees
    TList       *fStatus;           //->  List of active/inactive branches

public:
    // TChain constants
    enum {
       kGlobalWeight   = BIT(15),
       kAutoDelete     = BIT(16),
       kBigNumber      = 1234567890
    };
    TChain();
    TChain(const char *name, const char *title="");
    virtual ~TChain();

    virtual Int_t     Add(TChain *chain);
    virtual Int_t     Add(const char *name, Long64_t nentries=kBigNumber);
    virtual Int_t     AddFile(const char *name, Long64_t nentries=kBigNumber);
    virtual TFriendElement *AddFriend(const char *chainname, const char *dummy="");
    virtual TFriendElement *AddFriend(const char *chainname, TFile *dummy);
    virtual TFriendElement *AddFriend(TTree *chain, const char *alias, Bool_t warn = kFALSE);
    virtual void      Browse(TBrowser *b);
    virtual void      CanDeleteRefs(Bool_t flag=kTRUE);
    virtual void      CreatePackets();
    virtual void      Draw(Option_t *opt);
    virtual Long64_t  Draw(const char *varexp, const TCut &selection, Option_t *option=""
                       ,Long64_t nentries=kBigNumber, Long64_t firstentry=0);
    virtual Long64_t  Draw(const char *varexp, const char *selection, Option_t *option=""
                     ,Long64_t nentries=kBigNumber, Long64_t firstentry=0); // *MENU*
    virtual Int_t     Fill() {MayNotUse("Fill()"); return -1;}
    virtual TBranch  *GetBranch(const char *name);
    virtual Long64_t  GetChainEntryNumber(Long64_t entry) const;
            Int_t     GetNtrees() const {return fNtrees;}
    virtual Long64_t  GetEntries() const;
    virtual Int_t     GetEntry(Long64_t entry=0, Int_t getall=0);
    virtual Int_t     GetEntryWithIndex(Int_t major, Int_t minor=0);
    TFile            *GetFile() const;
    TLeaf            *GetLeaf(const char *name);
    TObjArray        *GetListOfBranches();
    TObjArray        *GetListOfFiles() const {return fFiles;}
    TObjArray        *GetListOfLeaves();
    const char       *GetAlias(const char *aliasName) const;
    virtual Double_t  GetMaximum(const char *columname);
    virtual Double_t  GetMinimum(const char *columname);
    virtual Int_t     GetNbranches();
    TList            *GetStatus() const {return fStatus;}
    TTree            *GetTree() const {return fTree;}
            Int_t     GetTreeNumber() const {return fTreeNumber;}
            Long64_t *GetTreeOffset() const {return fTreeOffset;}
            Int_t     GetTreeOffsetLen() const {return fTreeOffsetLen;}
    virtual Double_t  GetWeight() const;
            Long64_t  LoadTree(Long64_t entry);
    virtual void      Loop(Option_t *option="",Long64_t nentries=kBigNumber, Long64_t firstentry=0); // *MENU*
    virtual void      ls(Option_t *option="") const;
    virtual Long64_t  Merge(const char *name);
    virtual Long64_t  Merge(TCollection *list);
    virtual Long64_t  Merge(TFile *file, Int_t basketsize, Option_t *option="");
    virtual void      Print(Option_t *option="") const;
    virtual Long64_t  Process(const char *filename,Option_t *option="", Long64_t nentries=kBigNumber, Long64_t firstentry=0); // *MENU*
    virtual Long64_t  Process(TSelector *selector,Option_t *option="",  Long64_t nentries=kBigNumber, Long64_t firstentry=0);
    virtual void      Reset(Option_t *option="");
    virtual void      SetAutoDelete(Bool_t autodel=kTRUE);
    virtual void      SetBranchAddress(const char *bname,void *add);
    virtual void      SetBranchStatus(const char *bname,Bool_t status=1, UInt_t *found=0);
    virtual void      SetDirectory(TDirectory *dir);
    virtual void      SetMakeClass(Int_t make) { TTree::SetMakeClass(make); if (fTree) fTree->SetMakeClass(make);}
    virtual void      SetPacketSize(Int_t size = 100);
    virtual void      SetWeight(Double_t w=1, Option_t *option="");
    virtual void      UseCache(Int_t maxCacheSize = 10, Int_t pageSize = TCache::kDfltPageSize);

    ClassDef(TChain,5)  //A chain of TTrees
};

inline void TChain::Draw(Option_t *opt)
{ Draw(opt, "", "", 1000000000, 0); }

#endif
