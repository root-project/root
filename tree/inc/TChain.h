// @(#)root/tree:$Name:  $:$Id: TChain.h,v 1.11 2000/12/26 14:22:45 brun Exp $
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

class TFile;
class TBrowser;

class TChain : public TTree {

protected:
    Int_t       fTreeOffsetLen;     //  Current size of fTreeOffset array
    Int_t       fNtrees;            //  Number of Trees
    Int_t       fTreeNumber;        //! Current Tree number in fTreeOffset table
    Int_t       *fTreeOffset;       //[fTreeOffsetLen]Array of variables
    TTree       *fTree;             //! Pointer to current tree
    TFile       *fFile;             //! Pointer to current file
    TObjArray   *fFiles;            //->  List of file names containing the Trees
    TList       *fStatus;           //->  List of active/inactive branches
    TObject     *fNotify;           //! Object to be notified when loading a new file

public:
    TChain();
    TChain(const char *name, const char *title="");
    virtual ~TChain();

    virtual Int_t     Add(TChain *chain);
    virtual Int_t     Add(const char *name, Int_t nentries=-1);
    virtual Int_t     AddFile(const char *name, Int_t nentries);
    virtual TFriendElement *AddFriend(const char *chainname, const char *dummy="");
    virtual void      Browse(TBrowser *b);
    virtual void      CreatePackets();
    virtual void      Draw(Option_t *opt);
    virtual Int_t     Draw(const char *varexp, TCut selection, Option_t *option=""
                       ,Int_t nentries=1000000000, Int_t firstentry=0);
    virtual Int_t     Draw(const char *varexp, const char *selection, Option_t *option=""
                     ,Int_t nentries=1000000000, Int_t firstentry=0); // *MENU*
    virtual Int_t     Fill() {MayNotUse("TChain::Fill()"); return -1;}
    virtual TBranch  *GetBranch(const char *name);
    virtual Int_t     GetChainEntryNumber(Int_t entry) const;
            Int_t     GetNtrees() const {return fNtrees;}
    virtual Int_t     GetEntry(Int_t entry=0, Int_t getall=0);
    TFile            *GetFile() const {return fFile;}
    TLeaf            *GetLeaf(const char *name);
    TObjArray        *GetListOfBranches();
    TObjArray        *GetListOfFiles() const {return fFiles;}
    TObjArray        *GetListOfLeaves();
    virtual Double_t  GetMaximum(const char *columname);
    virtual Double_t  GetMinimum(const char *columname);
    virtual Int_t     GetNbranches();
    TObject          *GetNotify() const {return fNotify;}
    TList            *GetStatus() const {return fStatus;}
    TTree            *GetTree() const {return fTree;}
            Int_t     GetTreeNumber() const {return fTreeNumber;}
            Int_t    *GetTreeOffset() const {return fTreeOffset;}
            Int_t     GetTreeOffsetLen() const {return fTreeOffsetLen;}
            Int_t     LoadTree(Int_t entry);
    virtual void      Loop(Option_t *option="",Int_t nentries=1000000000, Int_t firstentry=0); // *MENU*
    virtual void      ls(Option_t *option="") const;
    virtual void      Merge(const char *name);
    virtual void      Merge(TFile *file, Int_t basketsize, Option_t *option="");
    virtual void      Print(Option_t *option="") const;
    virtual Int_t     Process(const char *filename,Option_t *option="", Int_t nentries=1000000000, Int_t firstentry=0); // *MENU*
    virtual Int_t     Process(TSelector *selector,Option_t *option="",  Int_t nentries=1000000000, Int_t firstentry=0);
    virtual void      SetBranchAddress(const char *bname,void *add);
    virtual void      SetBranchStatus(const char *bname,Bool_t status=1);
    virtual void      SetPacketSize(Int_t size = 100);
    virtual void      SetNotify(TObject *obj) {fNotify = obj;}

    ClassDef(TChain,3)  //A chain of TTrees
};

inline void TChain::Draw(Option_t *opt)
{ Draw(opt, "", "", 1000000000, 0); }

#endif
