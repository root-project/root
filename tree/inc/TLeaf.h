// @(#)root/tree:$Name:  $:$Id: TLeaf.h,v 1.5 2000/12/13 15:13:54 brun Exp $
// Author: Rene Brun   12/01/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TLeaf
#define ROOT_TLeaf


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TLeaf                                                                //
//                                                                      //
// A TTree object is a list of TBranch.                                 //
// A TBranch object is a list of TLeaf.                                 //
// A TLeaf describes the branch data types.                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TBranch
#include "TBranch.h"
#endif
#ifndef ROOT_TClonesArray
#include "TClonesArray.h"
#endif

class TBrowser;

class TLeaf : public TNamed {

protected:

    Int_t       fNdata;           //! Number of elements in fAddress data buffer
    Int_t       fLen;             //  Number of fixed length elements
    Int_t       fLenType;         //  Number of bytes for this data type
    Int_t       fOffset;          //  Offset in ClonesArray object (if one)
    Bool_t      fIsRange;         //  (=kTRUE if leaf has a range, kFALSE otherwise)
    Bool_t      fIsUnsigned;      //  (=kTRUE if unsigned, kFALSE otherwise)
    TLeaf      *fLeafCount;       //  Pointer to Leaf count if variable length
    TBranch    *fBranch;          //! Pointer to supporting branch

public:
    enum { kIndirectAddress = BIT(11), // Addresses passed via pointer
           kNewValue = BIT(12) };

    TLeaf();
    TLeaf(const char *name, const char *type);
    virtual ~TLeaf();

    virtual void     Browse(TBrowser *b);
    virtual void     Export(TClonesArray *list, Int_t n);
    virtual void     FillBasket(TBuffer &b);
    TBranch         *GetBranch() const { return fBranch; }
    virtual TLeaf   *GetLeafCount() const { return fLeafCount; }
    virtual TLeaf   *GetLeafCounter(Int_t &countval) const;
    virtual Int_t    GetLen() const;
    virtual Int_t    GetLenStatic() const { return fLen; }
    virtual Int_t    GetLenType() const { return fLenType; }
    virtual Int_t    GetMaximum() const { return 0; }
    virtual Int_t    GetMinimum() const { return 0; }
    virtual Int_t    GetNdata() const { return fNdata; }
    virtual Int_t    GetOffset() const { return fOffset; }
    virtual void    *GetValuePointer() const { return 0; }
    virtual const char *GetTypeName() const { return ""; }
    virtual Double_t GetValue(Int_t i=0) const;
    virtual void     Import(TClonesArray *list, Int_t n);
    virtual Bool_t   IsRange() const    { return fIsRange; }
    virtual Bool_t   IsUnsigned() const { return fIsUnsigned; }
    virtual void     PrintValue(Int_t i=0) const {;}
    virtual void     ReadBasket(TBuffer &) {;}
    virtual void     ReadBasketExport(TBuffer &, TClonesArray *, Int_t) {;}
            Int_t    ResetAddress(void *add, Bool_t destructor = kFALSE);
    virtual void     SetAddress(void *add=0);
    virtual void     SetBranch(TBranch *branch) { fBranch = branch; }
    virtual void     SetLeafCount(TLeaf *leaf) { fLeafCount=leaf; }
    virtual void     SetLen(Int_t len=1) { fLen=len; }
    virtual void     SetOffset(Int_t offset=0) { fOffset = offset; }
    virtual void     SetRange(Bool_t range=kTRUE) { fIsRange = range; }
    virtual void     SetUnsigned() { fIsUnsigned = kTRUE; }

    ClassDef(TLeaf,2)  //Leaf: description of a Branch data type
};

inline void     TLeaf::Export(TClonesArray *, Int_t) { }
inline Double_t TLeaf::GetValue(Int_t) const { return 0; }
inline void     TLeaf::Import(TClonesArray *, Int_t) { }
inline void     TLeaf::SetAddress(void *) { }

#endif
