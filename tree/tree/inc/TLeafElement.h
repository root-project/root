// @(#)root/tree:$Id$
// Author: Rene Brun   14/01/2001

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TLeafElement
#define ROOT_TLeafElement


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TLeafElement                                                          //
//                                                                      //
// A TLeaf for a general object derived from TObject.                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <atomic>

#include "TDataType.h"
#include "TLeaf.h"
#include "TBranchElement.h"

class TMethodCall;

class TLeafElement : public TLeaf {

protected:
   char               *fAbsAddress;   ///<! Absolute leaf Address
   Int_t               fID;           ///<  element serial number in fInfo
   Int_t               fType;         ///<  leaf type
   mutable std::atomic<DeserializeType> fDeserializeTypeCache{ DeserializeType::kInvalid }; ///<! Cache of the type of deserialization.
   mutable std::atomic<EDataType> fDataTypeCache{EDataType::kOther_t}; ///<! Cache of the EDataType of deserialization.

private:
   Int_t            GetOffsetHeaderSize() const override {return 1;}

public:
   TLeafElement();
   TLeafElement(TBranch *parent, const char *name, Int_t id, Int_t type);
   virtual ~TLeafElement();

   Bool_t           CanGenerateOffsetArray() override { return fLeafCount && fLenType; }
   virtual Int_t   *GenerateOffsetArrayBase(Int_t /*base*/, Int_t /*events*/) { return nullptr; }
   DeserializeType  GetDeserializeType() const override;

   Int_t            GetID() const { return fID; }
   TString          GetFullName() const override;
   Int_t            GetLen() const override { return ((TBranchElement*)fBranch)->GetNdata()*fLen; }
   TMethodCall     *GetMethodCall(const char *name);
   Int_t            GetMaximum() const override { return ((TBranchElement*)fBranch)->GetMaximum(); }
   Int_t            GetNdata() const override { return ((TBranchElement*)fBranch)->GetNdata()*fLen; }
   const char      *GetTypeName() const override { return ((TBranchElement*)fBranch)->GetTypeName(); }

   Double_t         GetValue(Int_t i=0) const override { return ((TBranchElement*)fBranch)->GetValue(i, fLen, kFALSE);}
   Long64_t         GetValueLong64(Int_t i = 0) const override { return ((TBranchElement*)fBranch)->GetTypedValue<Long64_t>(i, fLen, kFALSE); }
   LongDouble_t     GetValueLongDouble(Int_t i = 0) const override { return ((TBranchElement*)fBranch)->GetTypedValue<LongDouble_t>(i, fLen, kFALSE); }
   template<typename T> T GetTypedValueSubArray(Int_t i=0, Int_t j=0) const {return ((TBranchElement*)fBranch)->GetTypedValue<T>(i, j, kTRUE);}

   bool             ReadBasketFast(TBuffer&, Long64_t) override;

   void            *GetValuePointer() const override { return ((TBranchElement*)fBranch)->GetValuePointer(); }
   Bool_t           IncludeRange(TLeaf *) override;
   Bool_t           IsOnTerminalBranch() const override;
   void             PrintValue(Int_t i=0) const override {((TBranchElement*)fBranch)->PrintValue(i);}
   void             SetLeafCount(TLeaf *leaf) override { fLeafCount = leaf; }

   ClassDefOverride(TLeafElement,1);  //A TLeaf for a general object derived from TObject.
};

#endif
