// @(#)root/tree:$Id$
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

#ifndef ROOT_TAttFill
#include "TAttFill.h"
#endif

#ifndef ROOT_TDataType
#include "TDataType.h"
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
   const Int_t kBranchAny    = BIT(17); // branch is an object*
   const Int_t kMapObject    = kBranchObject | kBranchAny;

class TBranch : public TNamed , public TAttFill {

protected:
   // TBranch status bits
   enum EStatusBits {
      kAutoDelete = BIT(15),
      kDoNotUseBufferMap = BIT(22) // If set, at least one of the entry in the branch will use the buffer's map of classname and objects.
   };

   static Int_t fgCount;          //! branch counter
   Int_t       fCompress;        //  (=1 branch is compressed, 0 otherwise)
   Int_t       fBasketSize;      //  Initial Size of  Basket Buffer
   Int_t       fEntryOffsetLen;  //  Initial Length of fEntryOffset table in the basket buffers
   Int_t       fWriteBasket;     //  Last basket number written
   Long64_t    fEntryNumber;     //  Current entry number (last one filled in this branch)
   Int_t       fOffset;          //  Offset of this branch
   Int_t       fMaxBaskets;      //  Maximum number of Baskets so far
   Int_t       fNBaskets;        //! Number of baskets in memory
   Int_t       fSplitLevel;      //  Branch split level
   Int_t       fNleaves;         //! Number of leaves
   Int_t       fReadBasket;      //! Current basket number when reading
   Long64_t    fReadEntry;       //! Current entry number when reading
   Long64_t    fFirstBasketEntry;//! First entry in the current basket.
   Long64_t    fNextBasketEntry; //! Next entry that will requires us to go to the next basket
   TBasket    *fCurrentBasket;   //! Pointer to the current basket.
   Long64_t    fEntries;         //  Number of entries
   Long64_t    fFirstEntry;      //  Number of the first entry in this branch
   Long64_t    fTotBytes;        //  Total number of bytes in all leaves before compression
   Long64_t    fZipBytes;        //  Total number of bytes in all leaves after compression
   TObjArray   fBranches;        //-> List of Branches of this branch
   TObjArray   fLeaves;          //-> List of leaves of this branch
   TObjArray   fBaskets;         //-> List of baskets of this branch
   Int_t      *fBasketBytes;     //[fMaxBaskets] Lenght of baskets on file
   Long64_t   *fBasketEntry;     //[fMaxBaskets] Table of first entry in eack basket
   Long64_t   *fBasketSeek;      //[fMaxBaskets] Addresses of baskets on file
   TTree      *fTree;            //! Pointer to Tree header
   TBranch    *fMother;          //! Pointer to top-level parent branch in the tree.
   TBranch    *fParent;          //! Pointer to parent branch.
   char       *fAddress;         //! Address of 1st leaf (variable or object)
   TDirectory *fDirectory;       //! Pointer to directory where this branch buffers are stored
   TString     fFileName;        //  Name of file where buffers are stored ("" if in same file as Tree header)
   TBuffer    *fEntryBuffer;     //! Buffer used to directly pass the content without streaming
   TList      *fBrowsables;      //! List of TVirtualBranchBrowsables used for Browse()

   Bool_t      fSkipZip;         //! After being read, the buffer will not be unziped.

   typedef void (TBranch::*ReadLeaves_t)(TBuffer &b); 
   ReadLeaves_t fReadLeaves;     //! Pointer to the ReadLeaves implementation to use. 
   void     ReadLeavesImpl(TBuffer &b);
   void     ReadLeaves0Impl(TBuffer &b);
   void     ReadLeaves1Impl(TBuffer &b);
   void     ReadLeaves2Impl(TBuffer &b);
   
   void     SetSkipZip(Bool_t skip = kTRUE) { fSkipZip = skip; }
   void     Init(const char *name, const char *leaflist, Int_t compress);

   TBasket *GetFreshBasket();
   Int_t    WriteBasket(TBasket* basket, Int_t where);
   
   TString  GetRealFileName() const;

private:
   TBranch(const TBranch&);             // not implemented
   TBranch& operator=(const TBranch&);  // not implemented

public:
   TBranch();
   TBranch(TTree *tree, const char *name, void *address, const char *leaflist, Int_t basketsize=32000, Int_t compress=-1);
   TBranch(TBranch *parent, const char *name, void *address, const char *leaflist, Int_t basketsize=32000, Int_t compress=-1);
   virtual ~TBranch();

   virtual void      AddBasket(TBasket &b, Bool_t ondisk, Long64_t startEntry);
   virtual void      AddLastBasket(Long64_t startEntry);
   virtual void      Browse(TBrowser *b);
   virtual void      DeleteBaskets(Option_t* option="");
   virtual void      DropBaskets(Option_t *option = "");
           void      ExpandBasketArrays();
   virtual Int_t     Fill();
   virtual void      FillLeaves(TBuffer &b);
   virtual TBranch  *FindBranch(const char *name);
   virtual TLeaf    *FindLeaf(const char *name);
           Int_t     FlushBaskets();
           Int_t     FlushOneBasket(UInt_t which);

   virtual char     *GetAddress() const {return fAddress;}
           TBasket  *GetBasket(Int_t basket);
           Int_t    *GetBasketBytes() const {return fBasketBytes;}
           Long64_t *GetBasketEntry() const {return fBasketEntry;}
   virtual Long64_t  GetBasketSeek(Int_t basket) const;
   virtual Int_t     GetBasketSize() const {return fBasketSize;}
   virtual TList    *GetBrowsables();
   virtual const char* GetClassName() const;
   virtual Int_t     GetCompressionLevel() const {return fCompress;}
   TDirectory       *GetDirectory() const {return fDirectory;}
   virtual Int_t     GetEntry(Long64_t entry=0, Int_t getall = 0);
   virtual Int_t     GetEntryExport(Long64_t entry, Int_t getall, TClonesArray *list, Int_t n);
           Int_t     GetEntryOffsetLen() const { return fEntryOffsetLen; }
           Int_t     GetEvent(Long64_t entry=0) {return GetEntry(entry);}
   const char       *GetIconName() const;
   virtual Int_t     GetExpectedType(TClass *&clptr,EDataType &type);
   virtual TLeaf    *GetLeaf(const char *name) const;
   virtual TFile    *GetFile(Int_t mode=0);
   const char       *GetFileName()    const {return fFileName.Data();}
           Int_t     GetOffset()      const {return fOffset;}
           Int_t     GetReadBasket()  const {return fReadBasket;}
           Long64_t  GetReadEntry()   const {return fReadEntry;}
           Int_t     GetWriteBasket() const {return fWriteBasket;}
           Long64_t  GetTotalSize(Option_t *option="")   const;
           Long64_t  GetTotBytes(Option_t *option="")    const;
           Long64_t  GetZipBytes(Option_t *option="")    const;
           Long64_t  GetEntryNumber() const {return fEntryNumber;}
           Long64_t  GetFirstEntry()  const {return fFirstEntry; }
         TObjArray  *GetListOfBaskets()  {return &fBaskets;}
         TObjArray  *GetListOfBranches() {return &fBranches;}
         TObjArray  *GetListOfLeaves()   {return &fLeaves;}
           Int_t     GetMaxBaskets()  const  {return fMaxBaskets;}
           Int_t     GetNleaves()     const {return fNleaves;}
           Int_t     GetSplitLevel()  const {return fSplitLevel;}
           Long64_t  GetEntries()     const {return fEntries;}
           TTree    *GetTree()        const {return fTree;}
   virtual Int_t     GetRow(Int_t row);
   virtual Bool_t    GetMakeClass() const;
   TBranch          *GetMother() const;
   TBranch          *GetSubBranch(const TBranch *br) const;
   Bool_t            IsAutoDelete() const;
   Bool_t            IsFolder() const;
   virtual void      KeepCircular(Long64_t maxEntries);
   virtual Int_t     LoadBaskets();
   virtual void      Print(Option_t *option="") const;
   virtual void      ReadBasket(TBuffer &b);
   virtual void      Refresh(TBranch *b);
   virtual void      Reset(Option_t *option="");
   virtual void      ResetAfterMerge(TFileMergeInfo *);
   virtual void      ResetAddress();
   virtual void      ResetReadEntry() {fReadEntry = -1;}
   virtual void      SetAddress(void *add);
   virtual void      SetObject(void *objadd);
   virtual void      SetAutoDelete(Bool_t autodel=kTRUE);
   virtual void      SetBasketSize(Int_t buffsize);
   virtual void      SetBufferAddress(TBuffer *entryBuffer);
   virtual void      SetCompressionLevel(Int_t level=1);
   virtual void      SetEntries(Long64_t entries);
   virtual void      SetEntryOffsetLen(Int_t len, Bool_t updateSubBranches = kFALSE);
   virtual void      SetFirstEntry( Long64_t entry );
   virtual void      SetFile(TFile *file=0);
   virtual void      SetFile(const char *filename);
   virtual Bool_t    SetMakeClass(Bool_t decomposeObj = kTRUE);
   virtual void      SetOffset(Int_t offset=0) {fOffset=offset;}
   virtual void      SetStatus(Bool_t status=1);
   virtual void      SetTree(TTree *tree) { fTree = tree;}
   virtual void      SetupAddresses();
   virtual void      UpdateAddress() {;}
   virtual void      UpdateFile();

   static  void      ResetCount();

   ClassDef(TBranch,12);  //Branch descriptor
};

#endif
