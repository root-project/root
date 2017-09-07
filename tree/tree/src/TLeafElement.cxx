// @(#)root/tree:$Id$
// Author: Rene Brun   14/01/2001

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TLeafElement
\ingroup tree

A TLeaf for the general case when using the branches created via
a TStreamerInfo (i.e. using TBranchElement).
*/

#include "TLeafElement.h"
//#include "TMethodCall.h"

#include "TVirtualStreamerInfo.h"

ClassImp(TLeafElement);

////////////////////////////////////////////////////////////////////////////////
/// Default constructor for LeafObject.

TLeafElement::TLeafElement(): TLeaf()
{
   fAbsAddress = 0;
   fID   = -1;
   fType = -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Create a LeafObject.

TLeafElement::TLeafElement(TBranch *parent, const char *name, Int_t id, Int_t type)
   :TLeaf(parent, name,name)
{
   fAbsAddress = 0;
   fID         = id;
   fType       = type;
   if (type < TVirtualStreamerInfo::kObject) {
      Int_t bareType = type;
      if (bareType > TVirtualStreamerInfo::kOffsetP)
         bareType -= TVirtualStreamerInfo::kOffsetP;
      else if (bareType > TVirtualStreamerInfo::kOffsetL)
         bareType -= TVirtualStreamerInfo::kOffsetL;

      if ((bareType >= TVirtualStreamerInfo::kUChar && bareType <= TVirtualStreamerInfo::kULong)
          || bareType == TVirtualStreamerInfo::kULong64)
      SetUnsigned();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Default destructor for a LeafObject.

TLeafElement::~TLeafElement()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Determine if this TLeafElement supports bulk IO
TLeaf::DeserializeType
TLeafElement::GetDeserializeType() const
{
   if (R__likely(fDeserializeTypeCache.load(std::memory_order_relaxed) != kInvalid))
      return fDeserializeTypeCache;

   EDataType type = EDataType::kOther_t;

   // Sometimes, a TLeafElement is identified as a class (with EDataType -1)
   // but nevertheless, it is represented by a primitive type (e.g. counters).
   // GetTypeName() catches those cases.
   // 
   // Don't worry about the string comparisons: this only happens once (fDeserializeTypeCache).

   std::string typeName(GetTypeName());
   if (typeName == std::string("Char_t"))
     type = EDataType::kChar_t;
   else if (typeName == std::string("UChar_t"))
     type = EDataType::kUChar_t;
   else if (typeName == std::string("Bool_t"))
     type = EDataType::kBool_t;
   else if (typeName == std::string("Float_t"))
     type = EDataType::kFloat_t;
   else if (typeName == std::string("Double_t"))
     type = EDataType::kDouble_t;
   else if (typeName == std::string("Short_t"))
     type = EDataType::kShort_t;
   else if (typeName == std::string("UShort_t"))
     type = EDataType::kUShort_t;
   else if (typeName == std::string("Int_t"))
     type = EDataType::kInt_t;
   else if (typeName == std::string("UInt_t"))
     type = EDataType::kUInt_t;
   else if (typeName == std::string("Long_t"))
     type = EDataType::kLong_t;
   else if (typeName == std::string("ULong_t"))
     type = EDataType::kULong_t;
   else if (typeName == std::string("Long64_t"))
     type = EDataType::kLong64_t;
   else if (typeName == std::string("ULong64_t"))
     type = EDataType::kULong64_t;

   if (type == EDataType::kOther_t) {
     TClass *clptr = nullptr;
     if (fBranch->GetExpectedType(clptr, type)) {  // Returns non-zero in case of failure
       fDeserializeTypeCache.store(kDestructive, std::memory_order_relaxed);
       return kDestructive;  // I don't know what it is, but we aren't going to use bulk IO.
     }
     fDataTypeCache.store(type, std::memory_order_release);

     if (clptr) {  // Something that requires a dictionary to read; skip.
       fDeserializeTypeCache.store(kDestructive, std::memory_order_relaxed);
       return kDestructive;
     }
   }

   if (type == EDataType::kChar_t || type == EDataType::kUChar_t || type == EDataType::kBool_t || type == EDataType::kchar) {
      fDeserializeTypeCache.store(kZeroCopy, std::memory_order_relaxed);
      return kZeroCopy;
   } else if ((type == EDataType::kFloat_t) || (type == EDataType::kDouble_t) ||
              (type == EDataType::kShort_t) || (type == EDataType::kUShort_t) ||
              (type == EDataType::kInt_t) || (type == EDataType::kUInt_t) ||
              (type == EDataType::kLong_t) || (type == EDataType::kULong_t) ||
              (type == EDataType::kLong64_t) || (type == EDataType::kULong64_t) ||
              (type == EDataType::kCounter)) {
      fDeserializeTypeCache.store(kInPlace, std::memory_order_relaxed);
      return kInPlace;
   }

   fDeserializeTypeCache.store(kDestructive, std::memory_order_relaxed);
   return kDestructive;
}

////////////////////////////////////////////////////////////////////////////////
/// Deserialize N events from an input buffer.
bool
TLeafElement::ReadBasketFast(TBuffer &input_buf, Long64_t N) {

   EDataType type = fDataTypeCache.load(std::memory_order_consume);

   if ((type == EDataType::kFloat_t) || (type == EDataType::kInt_t) || (type == EDataType::kUInt_t)) {
      Int_t *buf __attribute__((aligned(8)));
      buf = reinterpret_cast<Int_t*>(input_buf.GetCurrent());
      for (int idx=0; idx<fLen*N; idx++) {
         buf[idx] = __builtin_bswap32(buf[idx]);
      }
   } else if ((type == EDataType::kDouble_t) || (type == EDataType::kLong64_t) || (type == EDataType::kULong64_t)) {
      Long64_t *buf __attribute__((aligned(8)));
      buf = reinterpret_cast<Long64_t*>(input_buf.GetCurrent());
      for (int idx=0; idx<fLen*N; idx++) {
         buf[idx] = __builtin_bswap64(buf[idx]);
      }
   } else {
      return false;
   }

   return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns pointer to method corresponding to name name is a string
/// with the general form "method(list of params)" If list of params is
/// omitted, () is assumed;

TMethodCall *TLeafElement::GetMethodCall(const char * /*name*/)
{
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Return true if this leaf is does not have any sub-branch/leaf.

Bool_t TLeafElement::IsOnTerminalBranch() const
{
   if (fBranch->GetListOfBranches()->GetEntriesFast()) return kFALSE;
   return kTRUE;
}
