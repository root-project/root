// @(#)root/tree:$Name:  $:$Id: TBranch.h,v 1.13 2002/01/23 08:38:59 brun Exp $
// Author: Rene Brun   12/01/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBranch
#define ROOT_TBranch


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBranch                                                              //
//                                                                      //
// A TTree object is a list of TBranchs.                                //
//   A TBranch describes the branch data structure and supports :       //
//     the list of TBaskets (branch buffers) associated to this branch. //
//     the list of TLeaves (branch description)                         //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif

#ifndef ROOT_TStringLong
#include "TStringLong.h"
#endif

#ifndef ROOT_Htypes
#include "Htypes.h"
#endif

class TTree;
class TBasket;
class TLeaf;
class TBrowser;
class TDirectory;
class TFile;
class TClonesArray;

   const Int_t kDoNotProcess = BIT(10); // Active bit for branches
   const Int_t kIsClone      = BIT(11); // to indicate a TBranchClones
   const Int_t kBranchObject = BIT(12); // branch is a TObject*

class TBranch : public TNamed {

protected:
    // TBranch status bits
    enum { kAutoDelete = BIT(15) };

   static Int_t fgCount;          //! branch counter
    Int_t       fCompress;        //  (=1 branch is compressed, 0 otherwise)
    Int_t       fBasketSize;      //  Initial Size of  Basket Buffer
    Int_t       fEntryOffsetLen;  //  Initial Length of fEntryOffset table in the basket buffers
    Int_t       fWriteBasket;     //  Last basket number written
    Int_t       fEntryNumber;     //  Current entry number (last one filled in this branch)
    Int_t       fOffset;          //  Offset of this branch
    Int_t       fMaxBaskets;      //  Maximum number of Baskets so far
    Int_t       fSplitLevel;      //  Branch split level
    Int_t       fNleaves;         //! Number of leaves
    Int_t       fReadBasket;      //! Current basket number when reading
    Int_t       fReadEntry;       //! Current entry number when reading
    Stat_t      fEntries;         //  Number of entries
    Stat_t      fTotBytes;        //  Total number of bytes in all leaves before compression
    Stat_t      fZipBytes;        //  Total number of bytes in all leaves after compression
    TObjArray   fBranches;        //-> List of Branches of this branch
    TObjArray   fLeaves;          //-> List of leaves of this branch
    TObjArray   fBaskets;         //-> List of baskets of this branch
    Int_t       fNBasketRAM;      //! Number of baskets in fBasketRAM
    Int_t      *fBasketRAM;       //! [fNBasketRAM] table of basket numbers in memory
    Int_t      *fBasketBytes;     //[fMaxBaskets] Lenght of baskets on file
    Int_t      *fBasketEntry;     //[fMaxBaskets] Table of first entry in eack basket
    Seek_t     *fBasketSeek;      //[fMaxBaskets] Addresses of baskets on file
    TTree      *fTree;            //! Pointer to Tree header
    char       *fAddress;         //! Address of 1st leaf (variable or object)
    TDirectory *fDirectory;       //! Pointer to directory where this branch buffers are stored
    TString     fFileName;        //  Name of file where buffers are stored ("" if in same file as Tree header)
    TBuffer    *fEntryBuffer;     //! Buffer used to directly pass the content without streaming

public:
    TBranch();
    TBranch(const char *name, void *address, const char *leaflist, Int_t basketsize=32000, Int_t compress=-1);
    virtual ~TBranch();

    virtual void     Browse(TBrowser *b);
    virtual void     DropBaskets();
    virtual Int_t    Fill();
    virtual void     FillLeaves(TBuffer &b);
    virtual TBranch *FindBranch(const char *name);
    virtual TLeaf   *FindLeaf(const char *name);
    virtual char    *GetAddress() const {return fAddress;}
    virtual Int_t    GetBasketSize() const {return fBasketSize;}
    virtual const char* GetClassName() const { return ""; }
    virtual Int_t    GetCompressionLevel() const {return fCompress;}
    virtual Int_t    GetEntry(Int_t entry=0, Int_t getall = 0);
    virtual Int_t    GetEntryExport(Int_t entry, Int_t getall, TClonesArray *list, Int_t n);
            Int_t    GetEvent(Int_t entry=0) {return GetEntry(entry);}
            Int_t    GetEntryOffsetLen() const {return fEntryOffsetLen;}
    virtual TLeaf   *GetLeaf(const char *name) const;
            TBasket *GetBasket(Int_t basket);
            Int_t   *GetBasketBytes() const {return fBasketBytes;}
            Int_t   *GetBasketEntry() const {return fBasketEntry;}
    virtual Seek_t   GetBasketSeek(Int_t basket) const;
    TDirectory      *GetDirectory() const {return fDirectory;}
    virtual TFile   *GetFile(Int_t mode=0);
    const char      *GetFileName() const {return fFileName.Data();}
            Int_t    GetOffset() const {return fOffset;}
            Int_t    GetReadBasket() const {return fReadBasket;}
            Int_t    GetReadEntry() const {return fReadEntry;}
            Int_t    GetWriteBasket() const {return fWriteBasket;}
            Stat_t   GetTotalSize() const;
            Stat_t   GetTotBytes() const {return fTotBytes;}
            Stat_t   GetZipBytes() const {return fZipBytes;}
            Int_t    GetEntryNumber() const {return fEntryNumber;}
          TObjArray *GetListOfBaskets()  {return &fBaskets;}
          TObjArray *GetListOfBranches() {return &fBranches;}
          TObjArray *GetListOfLeaves()   {return &fLeaves;}
            Int_t    GetMaxBaskets() const  {return fMaxBaskets;}
            Int_t    GetNleaves() const {return fNleaves;}
            Int_t    GetSplitLevel() const {return fSplitLevel;}
            Stat_t   GetEntries() const {return fEntries;}
            TTree   *GetTree() const {return fTree;}
    virtual Int_t    GetRow(Int_t row);
    Bool_t           IsAutoDelete() const;
    Bool_t           IsFolder() const;
    virtual void     Print(Option_t *option="") const;
    virtual void     ReadBasket(TBuffer &b);
    virtual void     ReadLeaves(TBuffer &b);
    virtual void     Reset(Option_t *option="");
    virtual void     ResetReadEntry() {fReadEntry = -1;}
    virtual void     SetAddress(void *add);
    virtual void     SetAutoDelete(Bool_t autodel=kTRUE);
    virtual void     SetBasketSize(Int_t buffsize);
    virtual void     SetBufferAddress(TBuffer *entryBuffer);
    virtual void     SetCompressionLevel(Int_t level=1);
    virtual void     SetEntryOffsetLen(Int_t len) {fEntryOffsetLen = len;}
    virtual void     SetFile(TFile *file=0);
    virtual void     SetFile(const char *filename);
    virtual void     SetOffset(Int_t offset=0) {fOffset=offset;}
    virtual void     SetTree(TTree *tree) { fTree = tree;}
    virtual void     UpdateAddress() {;}
    
    static  void     ResetCount() {fgCount = 0;}

    ClassDef(TBranch,7)  //Branch descriptor
};

#endif
