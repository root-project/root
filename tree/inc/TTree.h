// @(#)root/tree:$Name:  $:$Id: TTree.h,v 1.1.1.1 2000/05/16 17:00:45 rdm Exp $
// Author: Rene Brun   12/01/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTree
#define ROOT_TTree


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTree                                                                //
//                                                                      //
// A TTree object is a list of TBranch.                                 //
//   To Create a TTree object one must:                                 //
//    - Create the TTree header via the TTree constructor               //
//    - Call the TBranch constructor for every branch.                  //
//                                                                      //
//   To Fill this object, use member function Fill with no parameters.  //
//     The Fill function loops on all defined TBranch.                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif

#ifndef ROOT_TClonesArray
#include "TClonesArray.h"
#endif

#ifndef ROOT_TAttLine
#include "TAttLine.h"
#endif

#ifndef ROOT_TAttFill
#include "TAttFill.h"
#endif

#ifndef ROOT_TAttMarker
#include "TAttMarker.h"
#endif

#ifndef ROOT_TBranch
#include "TBranch.h"
#endif

#ifndef ROOT_TCut
#include "TCut.h"
#endif

#ifndef ROOT_TArrayD
#include "TArrayD.h"
#endif

#ifndef ROOT_TArrayI
#include "TArrayI.h"
#endif

#ifndef ROOT_TVirtualTreePlayer
#include "TVirtualTreePlayer.h"
#endif

class TBrowser;
class TFile;
class TDirectory;
class TLeaf;
class TH1;
class TTreeFormula;
class TPolyMarker;
class TEventList;
class TSQLResult;

class TTree : public TNamed, public TAttLine, public TAttFill, public TAttMarker {

protected:
    Int_t         fScanField;         //Number of runs before prompting in Scan
    Int_t         fUpdate;            //Update frequency for EntryLoop
    Int_t         fMaxEntryLoop;      //Maximum number of entries to process
    Int_t         fMaxVirtualSize;    //Maximum total size of buffers kept in memory
    Int_t         fAutoSave;          //Autosave tree when fAutoSave bytes produced
    Stat_t        fEntries;           //Number of entries
    Stat_t        fTotBytes;          //Total number of bytes in all branches before compression
    Stat_t        fZipBytes;          //Total number of bytes in all branches after compression
    Stat_t        fSavedBytes;        //Number of autosaved bytes
    Int_t         fChainOffset;       //Offset of 1st entry of this Tree in a TChain
    Int_t         fReadEntry;         //Number of the entry being processed
    Int_t         fTotalBuffers;      //Total number of bytes in branch buffers
    Int_t         fEstimate;          //Number of entries to estimate histogram limits
    Int_t         fDimension;         //Dimension of the current expression
    Int_t         fPacketSize;        //Number of entries in one packet for parallel root
    TDirectory    *fDirectory;        //Pointer to directory holding this tree
    TObjArray     fBranches;          //List of Branches
    TObjArray     fLeaves;            //Direct pointers to individual branch leaves
    TEventList    *fEventList;        //Pointer to event selection list (if one)
    Int_t          fNfill;            //Local for EntryLoop
    Int_t          fTimerInterval;    //Timer interval in milliseconds
    TArrayD        fIndexValues;      //Sorted index values
    TArrayI        fIndex;            //Index of sorted values
    TList         *fStreamerInfoList; //list of StreamerInfo for all TBranchObjects
    TVirtualTreePlayer *fPlayer;      //Pointer to current Tree player

protected:
    const   char    *GetNameByIndex(TString &varexp, Int_t *index,Int_t colindex);
    virtual void     MakeIndex(TString &varexp, Int_t *index);

public:
    // TTree status bits
    enum {
       kForceRead   = BIT(11)
    };

    TTree();
    TTree(const char *name, const char *title, Int_t maxvirtualsize=0);
    virtual ~TTree();

    virtual void      AddTotBytes(Int_t tot) {fTotBytes += tot;}
    virtual void      AddZipBytes(Int_t zip) {fZipBytes += zip;}
    virtual void      AutoSave();
    virtual TBranch  *Branch(const char *name, void *clonesaddress, Int_t bufsize=32000, Int_t splitlevel=1);
    virtual TBranch  *Branch(const char *name, const char *classname, void *addobj, Int_t bufsize=32000, Int_t splitlevel=1);
    virtual TBranch  *Branch(const char *name, void *address, const char *leaflist, Int_t bufsize=32000);
    virtual Int_t     Branch(TList *list, Int_t bufsize=32000);
    virtual void      Browse(TBrowser *b);
    virtual void      BuildIndex(const char *majorname, const char *minorname);
    virtual TTree    *CloneTree(Int_t nentries=-1, Option_t *option="");
    virtual Int_t     CopyEntries(TTree *tree, Int_t nentries=-1);
    virtual TTree    *CopyTree(const char *selection, Option_t *option=""
                       ,Int_t nentries=1000000000, Int_t firstentry=0);
    virtual void      Delete(Option_t *option=""); // *MENU*
    virtual void      Draw(Option_t *opt);
    virtual void      Draw(TCut varexp, TCut selection, Option_t *option=""
                       ,Int_t nentries=1000000000, Int_t firstentry=0);
    virtual void      Draw(const char *varexp, const char *selection, Option_t *option=""
                       ,Int_t nentries=1000000000, Int_t firstentry=0); // *MENU*
    virtual void      DropBuffers(Int_t nbytes);
    virtual Int_t     Fill();
    virtual void      Fit(const char *formula ,const char *varexp, const char *selection="",Option_t *option="" ,Option_t *goption=""
                       ,Int_t nentries=1000000000, Int_t firstentry=0); // *MENU*

    virtual TBranch  *GetBranch(const char *name);
    virtual Int_t     GetChainOffset() const { return fChainOffset; }
    TFile            *GetCurrentFile();
    TList            *GetStreamerInfoList() {return fStreamerInfoList;}
    TDirectory       *GetDirectory() {return fDirectory;}
    virtual Stat_t    GetEntries()   {return fEntries;}
    virtual Int_t     GetEstimate() const { return fEstimate; }
    virtual Int_t     GetEntry(Int_t entry=0, Int_t getall=0);
            Int_t     GetEvent(Int_t entry=0, Int_t getall=0) {return GetEntry(entry,getall);}
    virtual Int_t     GetEntryWithIndex(Int_t major, Int_t minor);
    virtual Int_t     GetEntryNumberWithIndex(Int_t major, Int_t minor);
    TEventList       *GetEventList() {return fEventList;}
    virtual Int_t     GetEntryNumber(Int_t entry);
    TH1              *GetHistogram() {return GetPlayer()->GetHistogram();}
    virtual Int_t    *GetIndex() {return &fIndex.fArray[0];}
    virtual Double_t *GetIndexValues() {return &fIndexValues.fArray[0];}
    virtual TLeaf    *GetLeaf(const char *name);
    virtual TObjArray       *GetListOfBranches()  {return &fBranches;}
    virtual TObjArray       *GetListOfLeaves()    {return &fLeaves;}
    virtual Int_t     GetMaxEntryLoop() {return fMaxEntryLoop;}
    virtual Float_t   GetMaximum(const char *columname);
    virtual Float_t   GetMinimum(const char *columname);
    virtual Int_t     GetMaxVirtualSize() {return fMaxVirtualSize;}
    virtual Int_t     GetNbranches() {return fBranches.GetEntriesFast();}
    TVirtualTreePlayer  *GetPlayer();
    virtual Int_t     GetPacketSize() const {return fPacketSize;}
    virtual Int_t     GetReadEntry() {return fReadEntry;}
    virtual Int_t     GetReadEvent() {return fReadEntry;}
    virtual Int_t     GetScanField() {return fScanField;}
    TTreeFormula     *GetSelect()    {return GetPlayer()->GetSelect();}
    virtual Int_t     GetSelectedRows()  {return GetPlayer()->GetSelectedRows();}
    virtual Int_t     GetTimerInterval() {return fTimerInterval;}
    virtual TTree    *GetTree() {return this;}
    virtual Int_t     GetUpdate() {return fUpdate;}
    TTreeFormula     *GetVar1() {return GetPlayer()->GetVar1();}
    TTreeFormula     *GetVar2() {return GetPlayer()->GetVar2();}
    TTreeFormula     *GetVar3() {return GetPlayer()->GetVar3();}
    TTreeFormula     *GetVar4() {return GetPlayer()->GetVar4();}
    virtual Float_t  *GetV1()   {return GetPlayer()->GetV1();}
    virtual Float_t  *GetV2()   {return GetPlayer()->GetV2();}
    virtual Float_t  *GetV3()   {return GetPlayer()->GetV3();}
    virtual Double_t *GetW()    {return GetPlayer()->GetW();}
    virtual Stat_t    GetTotBytes() {return fTotBytes;}
    virtual Stat_t    GetZipBytes() {return fZipBytes;}
    virtual void      IncrementTotalBuffers(Int_t nbytes) {fTotalBuffers += nbytes;}
    Bool_t            IsFolder() {return kTRUE;}
    virtual Int_t     LoadTree(Int_t entry);
    virtual void      Loop(Option_t *option="",Int_t nentries=1000000000, Int_t firstentry=0); // *MENU*
    virtual Int_t     MakeClass(const char *classname=0);
    virtual Int_t     MakeCode(const char *filename=0);
    Bool_t            MemoryFull(Int_t nbytes);
    virtual void      Print(Option_t *option=""); // *MENU*
    virtual void      Project(const char *hname, const char *varexp, const char *selection="", Option_t *option=""
                       ,Int_t nentries=1000000000, Int_t firstentry=0);
    virtual void      Reset(Option_t *option="");
    virtual void      Scan(const char *varexp="", const char *selection="", Option_t *option=""
                       ,Int_t nentries=1000000000, Int_t firstentry=0); // *MENU*
    virtual TSQLResult  *Query(const char *varexp="", const char *selection="", Option_t *option=""
                          ,Int_t nentries=1000000000, Int_t firstentry=0);
    virtual void      SetAutoSave(Int_t autosave=10000000) {fAutoSave=autosave;}
    virtual void      SetBasketSize(const char *bname,Int_t buffsize=16000);
    virtual void      SetBranchAddress(const char *bname,void *add);
    virtual void      SetBranchStatus(const char *bname,Bool_t status=1);
    virtual void      SetChainOffset(Int_t offset=0) {fChainOffset=offset;}
    virtual void      SetDirectory(TDirectory *dir);
    virtual void      SetEstimate(Int_t nentries=10000);
    virtual void      SetEventList(TEventList *list) {fEventList = list;}
    virtual void      SetMaxEntryLoop(Int_t maxev=1000000000) {fMaxEntryLoop = maxev;} // *MENU*
    virtual void      SetMaxVirtualSize(Int_t size=0) {fMaxVirtualSize = size;} // *MENU*
    virtual void      SetName(const char *name); // *MENU*
    virtual void      SetObject(const char *name, const char *title);
    virtual void      SetScanField(Int_t n=50) {fScanField = n;} // *MENU*
    virtual void      SetTimerInterval(Int_t msec=333) {fTimerInterval=msec;}
    virtual void      SetUpdate(Int_t freq=0) {fUpdate = freq;}
    virtual void      Show(Int_t entry=-1);
    virtual void      StartViewer(Int_t ww=520, Int_t wh=400); // *MENU*
    void              UseCurrentStyle();

    ClassDef(TTree,4)  //Tree descriptor (the main ROOT I/O class)
};

inline void TTree::Draw(Option_t *opt)
{ Draw(opt, "", "", 1000000000, 0); }

#endif
