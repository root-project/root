// @(#)root/tree:$Id$
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
#ifndef ROOT_Riosfwd
#include "Riosfwd.h"
#endif

class TClonesArray;
class TBrowser;

class TLeaf : public TNamed {

protected:

   Int_t       fNdata;           //! Number of elements in fAddress data buffer
   Int_t       fLen;             //  Number of fixed length elements
   Int_t       fLenType;         //  Number of bytes for this data type
   Int_t       fOffset;          //  Offset in ClonesArray object (if one)
   Bool_t      fIsRange;         //  (=kTRUE if leaf has a range, kFALSE otherwise)
   Bool_t      fIsUnsigned;      //  (=kTRUE if unsigned, kFALSE otherwise)
   TLeaf      *fLeafCount;       //  Pointer to Leaf count if variable length (we do not own the counter)
   TBranch    *fBranch;          //! Pointer to supporting branch (we do not own the branch)

   TLeaf(const TLeaf&);
   TLeaf& operator=(const TLeaf&);

public:
   enum {
      kIndirectAddress = BIT(11), // Data member is a pointer to an array of basic types.
      kNewValue = BIT(12)         // Set if we own the value buffer and so must delete it ourselves.
   };

   TLeaf();
   TLeaf(TBranch *parent, const char* name, const char* type);
   virtual ~TLeaf();

   virtual void     Browse(TBrowser* b);
   virtual void     Export(TClonesArray*, Int_t) {}
   virtual void     FillBasket(TBuffer& b);
   TBranch         *GetBranch() const { return fBranch; }
   virtual TLeaf   *GetLeafCount() const { return fLeafCount; }
   virtual TLeaf   *GetLeafCounter(Int_t& countval) const;
   virtual Int_t    GetLen() const;
   virtual Int_t    GetLenStatic() const { return fLen; }
   virtual Int_t    GetLenType() const { return fLenType; }
   virtual Int_t    GetMaximum() const { return 0; }
   virtual Int_t    GetMinimum() const { return 0; }
   virtual Int_t    GetNdata() const { return fNdata; }
   virtual Int_t    GetOffset() const { return fOffset; }
   virtual void    *GetValuePointer() const { return 0; }
   virtual const char *GetTypeName() const { return ""; }
   virtual Double_t GetValue(Int_t i = 0) const;
   virtual void     Import(TClonesArray*, Int_t) {}
   virtual Bool_t   IsOnTerminalBranch() const { return kTRUE; }
   virtual Bool_t   IsRange() const { return fIsRange; }
   virtual Bool_t   IsUnsigned() const { return fIsUnsigned; }
   virtual void     PrintValue(Int_t i = 0) const;
   virtual void     ReadBasket(TBuffer&) {}
   virtual void     ReadBasketExport(TBuffer&, TClonesArray*, Int_t) {}
   virtual void     ReadValue(istream& /*s*/) {}
           Int_t    ResetAddress(void* add, Bool_t destructor = kFALSE);
   virtual void     SetAddress(void* add = 0);
   virtual void     SetBranch(TBranch* branch) { fBranch = branch; }
   virtual void     SetLeafCount(TLeaf* leaf);
   virtual void     SetLen(Int_t len = 1) { fLen = len; }
   virtual void     SetOffset(Int_t offset = 0) { fOffset = offset; }
   virtual void     SetRange(Bool_t range = kTRUE) { fIsRange = range; }
   virtual void     SetUnsigned() { fIsUnsigned = kTRUE; }

   ClassDef(TLeaf,2);  //Leaf: description of a Branch data type
};

inline Double_t TLeaf::GetValue(Int_t /*i = 0*/) const { return 0.0; }
inline void     TLeaf::PrintValue(Int_t /* i = 0*/) const {}
inline void     TLeaf::SetAddress(void* /* add = 0 */) {}

#endif
