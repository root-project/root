// @(#)root/tree:$Id$
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

#include "TTree.h"

class TFile;
class TBrowser;
class TCut;
class TEntryList;
class TEventList;
class TCollection;

class TChain : public TTree {

protected:
   Int_t        fTreeOffsetLen;    ///<  Current size of fTreeOffset array
   Int_t        fNtrees;           ///<  Number of trees
   Int_t        fTreeNumber;       ///<! Current Tree number in fTreeOffset table
   Long64_t    *fTreeOffset;       ///<[fTreeOffsetLen] Array of variables
   bool         fCanDeleteRefs;    ///<! If true, TProcessIDs are deleted when closing a file
   TTree       *fTree;             ///<! Pointer to current tree (Note: We do *not* own this tree.)
   TFile       *fFile;             ///<! Pointer to current file (We own the file).
   TObjArray   *fFiles;            ///< -> List of file names containing the trees (TChainElement, owned)
   TList       *fStatus;           ///< -> List of active/inactive branches (TChainElement, owned)
   bool         fGlobalRegistration;  ///<! if true, bypass use of global lists

private:
   TChain(const TChain&);            // not implemented
   TChain& operator=(const TChain&); // not implemented
   void
   ParseTreeFilename(const char *name, TString &filename, TString &treename, TString &query, TString &suffix) const;
   Long64_t RefreshFriendAddresses();

protected:
   void InvalidateCurrentTree();

   // Called when setting the branch address of friends. In the case of TChain, the TChainElement for branch
   // 'bname' is created calling IsDelayed, to avoid missing branch errors when connecting it to the trees
   // of this chain
   Int_t SetBranchAddress(const char *bname, void *add, TBranch **ptr, TClass *realClass, EDataType datatype,
                          bool isptr, bool suppressMissingBranchError) override;

public:
   // TChain constants
   enum EStatusBits {
      kGlobalWeight   = BIT(15),
      kAutoDelete     = BIT(16)
   };

   // This used to be 1234567890, if user code hardcoded this number, the user code will need to change.
   static constexpr auto kBigNumber = TTree::kMaxEntries;

public:
   enum Mode { kWithoutGlobalRegistration, kWithGlobalRegistration };

   TChain(Mode mode = kWithGlobalRegistration);
   TChain(const char *name, const char *title = "", Mode mode = kWithGlobalRegistration);
   ~TChain() override;

   virtual Int_t     Add(TChain* chain);
   virtual Int_t     Add(const char* name, Long64_t nentries = TTree::kMaxEntries);
   virtual Int_t     AddFile(const char* name, Long64_t nentries = TTree::kMaxEntries, const char* tname = "");
   virtual Int_t     AddFileInfoList(TCollection* list, Long64_t nfiles = TTree::kMaxEntries);
   TFriendElement *AddFriend(const char* chainname, const char* dummy = "") override;
   TFriendElement *AddFriend(const char* chainname, TFile* dummy) override;
   TFriendElement *AddFriend(TTree* chain, const char* alias = "", bool warn = false) override;
   void      Browse(TBrowser*) override;
   virtual void      CanDeleteRefs(bool flag = true);
   virtual void      CreatePackets();
   void      DirectoryAutoAdd(TDirectory *) override;
   Long64_t  Draw(const char* varexp, const TCut& selection, Option_t* option = "", Long64_t nentries = kMaxEntries, Long64_t firstentry = 0) override;
   Long64_t  Draw(const char* varexp, const char* selection, Option_t* option = "", Long64_t nentries = kMaxEntries, Long64_t firstentry = 0) override; // *MENU*
   void      Draw(Option_t* opt) override { Draw(opt, "", "", kMaxEntries, 0); }
   Int_t     Fill() override { MayNotUse("Fill()"); return -1; }
   TBranch  *FindBranch(const char* name) override;
   TLeaf    *FindLeaf(const char* name) override;
   TBranch  *GetBranch(const char* name) override;
   bool      GetBranchStatus(const char* branchname) const override;
   Long64_t  GetCacheSize() const override { return fTree ? fTree->GetCacheSize() : fCacheSize; }
   Long64_t  GetChainEntryNumber(Long64_t entry) const override;
   TClusterIterator GetClusterIterator(Long64_t firstentry) override;
           Int_t     GetNtrees() const { return fNtrees; }
   Long64_t  GetEntries() const override;
   Long64_t  GetEntries(const char *sel) override { return TTree::GetEntries(sel); }
   Int_t     GetEntry(Long64_t entry=0, Int_t getall=0) override;
   Long64_t  GetEntryNumber(Long64_t entry) const override;
   Int_t     GetEntryWithIndex(Long64_t major, Long64_t minor=0) override;
   TFile            *GetFile() const;
   TLeaf    *GetLeaf(const char* branchname, const char* leafname) override;
   TLeaf    *GetLeaf(const char* name) override;
   TObjArray *GetListOfBranches() override;
   //                Warning, GetListOfFiles returns the list of TChainElements (not the list of files)
   //                see TChain::AddFile to see how to get the corresponding TFile objects
   TObjArray        *GetListOfFiles() const {return fFiles;}
   TObjArray *GetListOfLeaves() override;
   const char *GetAlias(const char *aliasName) const override;
   Double_t  GetMaximum(const char *columname) override;
   Double_t  GetMinimum(const char *columname) override;
   Int_t     GetNbranches() override;
   Long64_t  GetReadEntry() const override;
   TList            *GetStatus() const { return fStatus; }
   TTree    *GetTree() const override { return fTree; }
   Int_t     GetTreeNumber() const override { return fTreeNumber; }
           Long64_t *GetTreeOffset() const { return fTreeOffset; }
           Int_t     GetTreeOffsetLen() const { return fTreeOffsetLen; }
   Double_t  GetWeight() const override;
   bool      InPlaceClone(TDirectory *newdirectory, const char *options = "") override;
   Int_t     LoadBaskets(Long64_t maxmemory) override;
   Long64_t  LoadTree(Long64_t entry) override;
           void      Lookup(bool force = false);
   virtual void      Loop(Option_t *option="", Long64_t nentries=kMaxEntries, Long64_t firstentry=0); // *MENU*
   void      ls(Option_t *option="") const override;
   virtual Long64_t  Merge(const char *name, Option_t *option = "");
   Long64_t  Merge(TCollection *list, Option_t *option = "") override;
   Long64_t  Merge(TCollection *list, TFileMergeInfo *info) override;
   virtual Long64_t  Merge(TFile *file, Int_t basketsize, Option_t *option="");
   void      Print(Option_t *option="") const override;
   Long64_t  Process(const char *filename, Option_t *option="", Long64_t nentries=kMaxEntries, Long64_t firstentry=0) override; // *MENU*
   Long64_t  Process(TSelector* selector, Option_t* option = "", Long64_t nentries = kMaxEntries, Long64_t firstentry = 0) override;
   void      RecursiveRemove(TObject *obj) override;
   void      RemoveFriend(TTree*) override;
   void      Reset(Option_t *option="") override;
   void      ResetAfterMerge(TFileMergeInfo *) override;
   void      ResetBranchAddress(TBranch *) override;
   void      ResetBranchAddresses() override;
   void      SavePrimitive (std::ostream &out, Option_t *option="") override;
   Long64_t  Scan(const char *varexp="", const char *selection="", Option_t *option="", Long64_t nentries=kMaxEntries, Long64_t firstentry=0) override; // *MENU*
   virtual void      SetAutoDelete(bool autodel=true);
   Int_t     SetBranchAddress(const char *bname,void *add, TBranch **ptr = nullptr) override;
   Int_t     SetBranchAddress(const char *bname,void *add, TBranch **ptr, TClass *realClass, EDataType datatype, bool isptr) override;
   Int_t     SetBranchAddress(const char *bname,void *add, TClass *realClass, EDataType datatype, bool isptr) override;
   template <class T> Int_t SetBranchAddress(const char *bname, T **add, TBranch **ptr = nullptr) {
     return TTree::SetBranchAddress<T>(bname, add, ptr);
   }
#ifndef R__NO_CLASS_TEMPLATE_SPECIALIZATION
   // This can only be used when the template overload resolution can distinguish between
   // T* and T**
   template <class T> Int_t SetBranchAddress(const char *bname, T *add, TBranch **ptr = nullptr) {
     return TTree::SetBranchAddress<T>(bname, add, ptr);
   }
#endif

   void      SetBranchStatus(const char *bname, bool status = true, UInt_t *found = nullptr) override;
   Int_t     SetCacheSize(Long64_t cacheSize = -1) override;
   void      SetDirectory(TDirectory *dir) override;
   void      SetEntryList(TEntryList *elist, Option_t *opt="") override;
   virtual void      SetEntryListFile(const char *filename="", Option_t *opt="");
   void      SetEventList(TEventList *evlist) override;
   void      SetMakeClass(Int_t make) override { TTree::SetMakeClass(make); if (fTree) fTree->SetMakeClass(make);}
   void      SetName(const char *name) override;
   virtual void      SetPacketSize(Int_t size = 100);
   void      SetWeight(Double_t w=1, Option_t *option="") override;
   virtual void      UseCache(Int_t maxCacheSize = 10, Int_t pageSize = 0);

   ClassDefOverride(TChain,5)  //A chain of TTrees
};

#endif // ROOT_TChain
