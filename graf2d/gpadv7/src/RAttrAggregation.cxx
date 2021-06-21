/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RAttrAggregation.hxx>

#include <ROOT/RLogger.hxx>

#include <utility>

#include "TROOT.h"
#include "TList.h"
#include "TClass.h"
#include "TDataMember.h"

using namespace ROOT::Experimental;


///////////////////////////////////////////////////////////////////////////////
/// Return default values for attributes, empty for base class

const RAttrMap &RAttrAggregation::GetDefaults() const
{
   static RAttrMap empty;
   return empty;
}

///////////////////////////////////////////////////////////////////////////////
/// Collect all attributes in derived class
/// Works only if such class has dictionary.
/// In special cases one has to provide special implementation directly

RAttrMap RAttrAggregation::CollectDefaults() const
{
   RAttrMap res;

   const std::type_info &info = typeid(*this);
   auto thisClass = TClass::GetClass(info);
   auto baseClass = TClass::GetClass<RAttrBase>();
   if (thisClass && baseClass) {
      for (auto data_member: TRangeDynCast<TDataMember>(thisClass->GetListOfDataMembers())) {
         TClass *cl = data_member && !data_member->IsBasic() && !data_member->IsEnum() ? gROOT->GetClass(data_member->GetFullTypeName()) : nullptr;
         if (cl && cl->InheritsFrom(baseClass) && (cl->GetBaseClassOffset(baseClass) == 0))
            res.AddDefaults(*((const RAttrBase *)((char*) this + data_member->GetOffset())));
      }
   } else {
      R__LOG_ERROR(GPadLog()) << "Missing dictionary for " << info.name() << " class";
   }

   return res;
}

///////////////////////////////////////////////////////////////////////////////
/// Copy attributes into target object

void RAttrAggregation::CopyTo(RAttrAggregation &tgt, bool use_style) const
{
   for (const auto &entry : GetDefaults()) {
      if (auto v = AccessValue(entry.first, use_style))
         tgt.CopyValue(entry.first, *v.value);
   }
}

///////////////////////////////////////////////////////////////////////////////
/// Copy attributes from other object

bool RAttrAggregation::CopyValue(const std::string &name, const RAttrMap::Value_t &value, bool check_type)
{
   if (check_type) {
      const auto *dvalue = GetDefaults().Find(name);
      if (!dvalue || !dvalue->CanConvertFrom(value.Kind()))
         return false;
   }

   if (auto access = EnsureAttr(name)) {
      access.attr->Add(access.fullname, value.Copy());
      return true;
   }

   return false;
}


///////////////////////////////////////////////////////////////////////////////
/// Check if provided value equal to attribute in the map

bool RAttrAggregation::IsValueEqual(const std::string &name, const RAttrMap::Value_t &value, bool use_style) const
{
   if (auto v = AccessValue(name, use_style))
      return v.value->CanConvertFrom(value.Kind()) && v.value->IsEqual(value);

   return value.Kind() == RAttrMap::kNoValue;
}


///////////////////////////////////////////////////////////////////////////////
/// Check if all values which are evaluated in this object are exactly the same as in tgt object

bool RAttrAggregation::IsSame(const RAttrAggregation &tgt, bool use_style) const
{
   for (const auto &entry : GetDefaults()) {
      // R__LOG_DEBUG(0, GPadLog()) << "Comparing entry " << entry.first;
      if (auto v = AccessValue(entry.first, use_style))
         if (!tgt.IsValueEqual(entry.first, *v.value, use_style))
            return false;
   }

   return true;
}

///////////////////////////////////////////////////////////////////////////////
/// Clear all respective values from drawable. Only defaults can be used

void RAttrAggregation::Clear()
{
   for (const auto &entry : GetDefaults())
      ClearValue(entry.first);
}

