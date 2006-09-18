// @(#)root/tree:$Name:  $:$Id: TChainProof.h,v 1.7 2006/07/04 23:45:50 rdm Exp $
// Author: Marek Biskup   10/12/2004

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TChainProof
#define ROOT_TChainProof


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TChainProof                                                          //
//                                                                      //
// A wrapper for TDSet to behave as a Tree/Chain.                       //
// Uses an internal TDSet to handle processing and a TTree              //
// which holds the branch structure.                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TTree
#include "TTree.h"
#endif

class TDSet;
class TDrawFeedback;
class TVirtualProof;


class TChainProof : public TTree {

protected:
   TTree         *fTree;             // dummy tree
   TDSet         *fSet;              // TDSet
   TVirtualProof *fProof;            // PROOF
   TDrawFeedback *fDrawFeedback;     // feedback handler
   
protected:
   void             AddClone(TTree*);
   const   char    *GetNameByIndex(TString &varexp, Int_t *index,Int_t colindex) const;
   virtual void     KeepCircular();
   virtual void     MakeIndex(TString &varexp, Int_t *index);
   virtual TFile   *ChangeFile(TFile *file);

public:
   TChainProof(TDSet *set, TTree* tree, TVirtualProof* proof);
   virtual ~TChainProof();
   
   virtual TFriendElement *AddFriend(const char *treename, const char *filename="");
   virtual TFriendElement *AddFriend(const char *treename, TFile *file);
   virtual TFriendElement *AddFriend(TTree *tree, const char* alias="", Bool_t warn = kFALSE);
   virtual void         AddTotBytes(Int_t tot) {fTotBytes += tot;}
   virtual void         AddZipBytes(Int_t zip) {fZipBytes += zip;}
   virtual Long64_t     AutoSave(Option_t *option="");
   virtual Int_t        Branch(TCollection *list, Int_t bufsize=32000, Int_t splitlevel=99, const char *name="");
   virtual Int_t        Branch(TList *list, Int_t bufsize=32000, Int_t splitlevel=99);
   virtual Int_t        Branch(const char *folder, Int_t bufsize=32000, Int_t splitlevel=99);
   virtual TBranch     *Branch(const char *name, void *address, const char *leaflist, Int_t bufsize=32000);
   virtual TBranch     *Branch(const char *name, void *clonesaddress, Int_t bufsize=32000, Int_t splitlevel=1);
   virtual TBranch     *Branch(const char *name, TClonesArray **clonesaddress, Int_t bufsize=32000, Int_t splitlevel=1);
#if !defined(__CINT__)
   virtual TBranch     *Branch(const char *name, const char *classname, void *addobj, Int_t bufsize=32000, Int_t splitlevel=99);
#endif
   virtual TBranch     *Bronch(const char *name, const char *classname, void *addobj, Int_t bufsize=32000, Int_t splitlevel=99);
   virtual TBranch     *BranchOld(const char *name, const char *classname, void *addobj, Int_t bufsize=32000, Int_t splitlevel=1);
   virtual TBranch     *BranchRef();
   virtual void         Browse(TBrowser *b);
   virtual Int_t        BuildIndex(const char *majorname, const char *minorname="0");
   TStreamerInfo       *BuildStreamerInfo(TClass *cl, void *pointer=0);
   virtual TTree       *CloneTree(Long64_t nentries=-1, Option_t *option="");
   virtual void         CopyAddresses(TTree *tree);
   virtual Long64_t     CopyEntries(TTree *tree, Long64_t nentries=-1);
   virtual TTree       *CopyTree(const char *selection, Option_t *option=""
                                 ,Long64_t nentries=1000000000, Long64_t firstentry=0);
   Int_t                Debug() const {return fDebug;}
   virtual void         Delete(Option_t *option=""); // *MENU*
   virtual Long64_t     Draw(const char *varexp, const TCut &selection, Option_t *option=""
                             ,Long64_t nentries=1000000000, Long64_t firstentry=0);
   virtual Long64_t     Draw(const char *varexp, const char *selection, Option_t *option=""
                             ,Long64_t nentries=1000000000, Long64_t firstentry=0); // *MENU*
   virtual void         Draw(Option_t *opt) { Draw(opt, "", "", 1000000000, 0); }
   virtual void         DropBuffers(Int_t nbytes);
   virtual Int_t        Fill();
   virtual TBranch     *FindBranch(const char *name);
   virtual TLeaf       *FindLeaf(const char *name);
   virtual Long64_t     Fit(const char *funcname ,const char *varexp, const char *selection="",Option_t *option="" ,Option_t *goption=""
                            ,Long64_t nentries=1000000000, Long64_t firstentry=0); // *MENU*
   
   virtual const char  *GetAlias(const char *aliasName) const;
   virtual TBranch     *GetBranch(const char *name);
   virtual TBranchRef  *GetBranchRef() const {return fBranchRef;};
   virtual Bool_t       GetBranchStatus(const char *branchname) const;
   static  Int_t        GetBranchStyle();
   virtual Long64_t     GetChainEntryNumber(Long64_t entry) const {return entry;}
   virtual Long64_t     GetChainOffset() const { return fChainOffset; }
   TFile               *GetCurrentFile() const;
           Long64_t     GetDebugMax()  const {return fDebugMax;}
           Long64_t     GetDebugMin()  const {return fDebugMin;}
   TDirectory          *GetDirectory() const {return fDirectory;}
   virtual Long64_t     GetEntries() const;
   virtual Long64_t     GetEntries(const char *sel);
   virtual Long64_t     GetEntriesFast() const   {return fEntries;}
   virtual Long64_t     GetEntriesFriend() const;
   virtual Long64_t     GetEstimate() const { return fEstimate; }
   virtual Int_t        GetEntry(Long64_t entry=0, Int_t getall=0);
           Int_t        GetEvent(Long64_t entry=0, Int_t getall=0) {return GetEntry(entry,getall);}
   virtual Int_t        GetEntryWithIndex(Int_t major, Int_t minor=0);
   virtual Long64_t     GetEntryNumberWithBestIndex(Int_t major, Int_t minor=0) const;
   virtual Long64_t     GetEntryNumberWithIndex(Int_t major, Int_t minor=0) const;
//    TEventList          *GetEventList() const {return fEventList;}
   virtual Long64_t     GetEntryNumber(Long64_t entry) const;
   virtual Int_t        GetFileNumber() const {return fFileNumber;}
   virtual const char  *GetFriendAlias(TTree *) const;
   TH1                 *GetHistogram() {return GetPlayer()->GetHistogram();}
   virtual Int_t       *GetIndex() {return &fIndex.fArray[0];}
   virtual Double_t    *GetIndexValues() {return &fIndexValues.fArray[0];}
   virtual TIterator   *GetIteratorOnAllLeaves(Bool_t dir = kIterForward);
   virtual TLeaf       *GetLeaf(const char *name);
   virtual TList       *GetListOfClones() { return 0; }
   virtual TObjArray   *GetListOfBranches() {return (fTree ? fTree->GetListOfBranches() : (TObjArray *)0); }
   virtual TObjArray   *GetListOfLeaves()   {return (fTree ? fTree->GetListOfLeaves() : (TObjArray *)0);}
   virtual TList       *GetListOfFriends()    const {return 0;}
   virtual TSeqCollection *GetListOfAliases() const {return 0;}

    // GetMakeClass is left non-virtual for efficiency reason.
    // Making it virtual affects the performance of the I/O
           Int_t        GetMakeClass() const {return fMakeClass;}

   virtual Long64_t     GetMaxEntryLoop() const {return fMaxEntryLoop;}
   virtual Double_t     GetMaximum(const char *columname);
   static  Long64_t     GetMaxTreeSize();
   virtual Long64_t     GetMaxVirtualSize() const {return fMaxVirtualSize;}
   virtual Double_t     GetMinimum(const char *columname);
   virtual Int_t        GetNbranches() {return fBranches.GetEntriesFast();}
   TObject             *GetNotify() const {return fNotify;}
   TVirtualTreePlayer  *GetPlayer();
   virtual Int_t        GetPacketSize() const {return fPacketSize;}
   virtual TVirtualProof* GetProof() const {return fProof;}
   virtual Long64_t     GetReadEntry()  const;
   virtual Long64_t     GetReadEvent()  const {return fReadEntry;}
   virtual Int_t        GetScanField()  const {return fScanField;}
   TTreeFormula        *GetSelect()    {return GetPlayer()->GetSelect();}
   virtual Long64_t     GetSelectedRows() {return GetPlayer()->GetSelectedRows();}
   virtual Int_t        GetTimerInterval() const {return fTimerInterval;}
   virtual Long64_t     GetTotBytes() const {return fTotBytes;}
   virtual TTree       *GetTree() const {return (TTree*)this;}
   virtual TVirtualIndex  *GetTreeIndex() const {return fTreeIndex;}
   virtual Int_t        GetTreeNumber() const {return 0;}
   virtual Int_t        GetUpdate() const {return fUpdate;}
   virtual TList       *GetUserInfo();
   TTreeFormula        *GetVar1() {return GetPlayer()->GetVar1();}
   TTreeFormula        *GetVar2() {return GetPlayer()->GetVar2();}
   TTreeFormula        *GetVar3() {return GetPlayer()->GetVar3();}
   TTreeFormula        *GetVar4() {return GetPlayer()->GetVar4();}
   virtual Double_t    *GetV1()   {return GetPlayer()->GetV1();}
   virtual Double_t    *GetV2()   {return GetPlayer()->GetV2();}
   virtual Double_t    *GetV3()   {return GetPlayer()->GetV3();}
   virtual Double_t    *GetV4()   {return GetPlayer()->GetV4();}
   virtual Double_t    *GetW()    {return GetPlayer()->GetW();}
   virtual Double_t     GetWeight() const   {return fWeight;}
   virtual Long64_t     GetZipBytes() const {return fZipBytes;}
   Bool_t               HasTreeHeader() const { return (fTree ? kTRUE : kFALSE); }
   virtual void         IncrementTotalBuffers(Int_t nbytes) {fTotalBuffers += nbytes;}
   Bool_t               IsFolder() const {return kTRUE;}
   virtual Int_t        LoadBaskets(Long64_t maxmemory=2000000000);
   virtual Long64_t     LoadTree(Long64_t entry);
   virtual Long64_t     LoadTreeFriend(Long64_t entry, TTree *T);
   virtual Int_t        MakeClass(const char *classname=0,Option_t *option="");
   virtual Int_t        MakeCode(const char *filename=0);
   virtual Int_t        MakeProxy(const char *classname, const char *macrofilename = 0,
                                  const char *cutfilename = 0,
                                  const char *option = 0, Int_t maxUnrolling = 3);
   virtual Int_t        MakeSelector(const char *selector=0);
   Bool_t               MemoryFull(Int_t nbytes);
   virtual Long64_t     Merge(TCollection *list,Option_t *option ="");
   static  TTree       *MergeTrees(TList *list,Option_t *option ="");
   virtual Bool_t       Notify();
   TPrincipal          *Principal(const char *varexp="", const char *selection="", Option_t *option="np"
                                  ,Long64_t nentries=1000000000, Long64_t firstentry=0);
   virtual void         Print(Option_t *option="") const; // *MENU*
   virtual Long64_t     Process(const char *filename,Option_t *option="", Long64_t nentries=1000000000, Long64_t firstentry=0); // *MENU*
   virtual void         Progress(Long64_t total, Long64_t processed);
   virtual Long64_t     Process(TSelector *selector, Option_t *option="", Long64_t nentries=1000000000, Long64_t firstentry=0);
   virtual Long64_t     Project(const char *hname, const char *varexp, const char *selection="", Option_t *option=""
                                ,Long64_t nentries=1000000000, Long64_t firstentry=0);
   virtual TSQLResult  *Query(const char *varexp="", const char *selection="", Option_t *option=""
                              ,Long64_t nentries=1000000000, Long64_t firstentry=0);
   virtual Long64_t     ReadFile(const char *filename, const char *branchDescriptor="");
   virtual void         Refresh();
   virtual void         RemoveFriend(TTree*);
   virtual void         Reset(Option_t *option="");
   virtual void         ResetBranchAddresses();
   virtual Long64_t     Scan(const char *varexp="", const char *selection="", Option_t *option=""
                             ,Long64_t nentries=1000000000, Long64_t firstentry=0); // *MENU*
   virtual Bool_t       SetAlias(const char *aliasName, const char *aliasFormula);
   virtual void         SetAutoSave(Long64_t autos=10000000) {fAutoSave=autos;}
   virtual void         SetBasketSize(const char *bname,Int_t buffsize=16000);
#if !defined(__CINT__)
   virtual void         SetBranchAddress(const char *bname,void *add, TBranch **ptr);
   virtual void         SetBranchAddress(const char *bname,void *add, TBranch **ptr, TClass *realClass, EDataType datatype, Bool_t isptr);
#endif
   virtual void         SetBranchAddress(const char *bname,void *add, TClass *realClass, EDataType datatype, Bool_t ptr);
   virtual void         SetBranchStatus(const char *bname,Bool_t status=1,UInt_t *found=0);
   static  void         SetBranchStyle(Int_t style=1);  //style=0 for old branch, =1 for new branch style
   virtual void         SetChainOffset(Int_t offset=0) {fChainOffset=offset;}
   virtual void         SetCircular(Long64_t maxEntries);
   virtual void         SetDebug(Int_t level=1, Long64_t min=0, Long64_t max=9999999); // *MENU*
   virtual void         SetDirectory(TDirectory *dir);
   virtual Long64_t     SetEntries(Long64_t n);
   virtual void         SetEstimate(Long64_t nentries=10000);
   virtual void         SetFileNumber(Int_t number=0);
   //    virtual void         SetEventList(TEventList *list) {TTree::SetEventList(list);}
   virtual void         SetMakeClass(Int_t make) {fMakeClass = make;}
   virtual void         SetMaxEntryLoop(Long64_t maxev=1000000000) {fMaxEntryLoop = maxev;} // *MENU*
   static  void         SetMaxTreeSize(Long64_t maxsize=1900000000);
   virtual void         SetMaxVirtualSize(Long64_t size=0) {fMaxVirtualSize = size;} // *MENU*
   virtual void         SetName(const char *name); // *MENU*
   virtual void         SetNotify(TObject *obj) {fNotify = obj;}
   virtual void         SetObject(const char *name, const char *title);
   virtual void         SetScanField(Int_t n=50) {fScanField = n;} // *MENU*
   virtual void         SetTimerInterval(Int_t msec=333) {fTimerInterval=msec;}
   virtual void         SetTreeIndex(TVirtualIndex*index);
   virtual void         SetWeight(Double_t w=1, Option_t *option="");
   virtual void         SetUpdate(Int_t freq=0) {fUpdate = freq;}
   virtual void         Show(Long64_t entry=-1, Int_t lenmax=20);
   virtual void         StartViewer(); // *MENU*
   virtual Long64_t     UnbinnedFit(const char *funcname ,const char *varexp, const char *selection="",Option_t *option=""
                                    ,Long64_t nentries=1000000000, Long64_t firstentry=0);
   void                 UseCurrentStyle();
   virtual void         ConnectProof(TVirtualProof* proof);
   virtual void         ReleaseProof();
   
   static  TChainProof *MakeChainProof(TDSet *set, TVirtualProof *proof, Bool_t gettreeheader = kFALSE);
   
   ClassDef(TChainProof,0)  //TChain proxy for running chains on PROOF
};

#endif
