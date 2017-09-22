// Author: Brian Bockelman UNL 09/2017

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/TIOFeatures.hxx"
#include "TBranch.h"
#include "TEnum.h"
#include "TEnumConstant.h"
#include "TError.h"
#include "TTree.h"

#include <bitset>

using namespace ROOT::Experimental;

/**
 * \class TIOFeatures
 * \ingroup tree
 *
 * TIOFeatures provides the end-user with the ability to change the IO behavior
 * of data written via a TTree.  This class allows access to experimental and non-default
 * features.
 *
 * When one of these features are activated, forward compatibility breaks may occur.
 * That is, older versions of ROOT may not be able to read files written by this version
 * of ROOT that have enabled these non-default features.
 *
 */

void TIOFeatures::Clear(TBasket::EIOBits enum_bits)
{
   auto bits = static_cast<UChar_t>(enum_bits);
   if (R__unlikely((bits & static_cast<UChar_t>(TBasket::EIOBits::kSupported)) != bits)) {
      Error("TestFeature", "A feature is being cleared that is not supported.");
      return;
   }
   fIOBits &= ~bits;
}

static std::string GetUnsupportedName(TBasket::EUnsupportedIOBits enum_flag)
{
   UChar_t flag = static_cast<UChar_t>(enum_flag);

   std::string retval = "unknown";

   TClass *cl = TClass::GetClass("TBasket");
   if (cl == nullptr) {
      return retval;
   }

   TEnum *eUnsupportedIOBits = (TEnum *)cl->GetListOfEnums()->FindObject("EUnsupportedIOBits");
   if (eUnsupportedIOBits == nullptr) {
      return retval;
   }

   for (auto constant : ROOT::Detail::TRangeStaticCast<TEnumConstant>(eUnsupportedIOBits->GetConstants())) {
      if (constant->GetValue() == flag) {
         retval = constant->GetName();
         break;
      }
   }
   return retval;
}

bool TIOFeatures::Set(TBasket::EIOBits enum_bits)
{
   auto bits = static_cast<UChar_t>(enum_bits);
   if (R__unlikely((bits & static_cast<UChar_t>(TBasket::EIOBits::kSupported)) != bits)) {
      UChar_t unsupported = bits & static_cast<UChar_t>(TBasket::EUnsupportedIOBits::kUnsupported);
      if (unsupported) {
         Error("SetFeature", "A feature was request (%s) but this feature is no longer supported.",
               GetUnsupportedName(static_cast<TBasket::EUnsupportedIOBits>(unsupported)).c_str());
      } else {
         Error("SetFeature", "An unknown feature was requested (flag=%s); cannot enable it.",
               std::bitset<32>(unsupported).to_string().c_str());
      }
      return kFALSE;
   }
   fIOBits |= static_cast<UChar_t>(bits);
   return kTRUE;
}

bool TIOFeatures::Test(TBasket::EIOBits enum_bits) const
{
   auto bits = static_cast<UChar_t>(enum_bits);
   if (R__unlikely((bits & static_cast<UChar_t>(TBasket::EIOBits::kSupported)) != bits)) {
      Error("TestFeature", "A feature is being tested for that is not supported or known.");
      return kFALSE;
   }
   return (fIOBits & bits) == bits;
}

UChar_t TIOFeatures::GetFeatures() const
{
   return fIOBits;
}
