/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RAttrBase.hxx>

#include <ROOT/RLogger.hxx>

#include <utility>

#include "TList.h"
#include "TClass.h"
#include "TDataMember.h"

using namespace ROOT::Experimental;

RLogChannel &ROOT::Experimental::GPadLog()
{
   static RLogChannel sLog("ROOT.GPad");
   return sLog;
}

///////////////////////////////////////////////////////////////////////////////
/// Clear internal data

void RAttrBase::ClearData()
{
   if ((fKind == kOwnAttr) && fD.ownattr) {
      delete fD.ownattr;
      fD.ownattr = nullptr;
   }
}

///////////////////////////////////////////////////////////////////////////////
/// Creates own attribute - only if no drawable and no parent are assigned

RAttrMap *RAttrBase::CreateOwnAttr()
{
   if (((fKind == kParent) && !fD.parent) || ((fKind == kDrawable) && !fD.drawable))
      fKind = kOwnAttr;

   if (fKind != kOwnAttr)
      return nullptr;

   if (!fD.ownattr)
      fD.ownattr = new RAttrMap();

   return fD.ownattr;
}


///////////////////////////////////////////////////////////////////////////////
/// Return default values for attributes, empty for base class

const RAttrMap &RAttrBase::GetDefaults() const
{
   static RAttrMap empty;
   return empty;
}

///////////////////////////////////////////////////////////////////////////////
/// Copy attributes from other object

bool RAttrBase::CopyValue(const std::string &name, const RAttrMap::Value_t &value, bool check_type)
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

bool RAttrBase::IsValueEqual(const std::string &name, const RAttrMap::Value_t &value, bool use_style) const
{
   if (auto v = AccessValue(name, use_style))
      return v.value->CanConvertFrom(value.Kind()) && v.value->IsEqual(value);

   return value.Kind() == RAttrMap::kNoValue;
}

///////////////////////////////////////////////////////////////////////////////
/// Copy attributes into target object

void RAttrBase::CopyTo(RAttrBase &tgt, bool use_style) const
{
   for (const auto &entry : GetDefaults()) {
      if (auto v = AccessValue(entry.first, use_style))
         tgt.CopyValue(entry.first, *v.value);
   }
}

///////////////////////////////////////////////////////////////////////////////
/// Move all fields into target object

void RAttrBase::MoveTo(RAttrBase &tgt)
{
   std::swap(fKind, tgt.fKind);
   std::swap(fD, tgt.fD);
   std::swap(fPrefix, tgt.fPrefix);
}

///////////////////////////////////////////////////////////////////////////////
/// Check if all values which are evaluated in this object are exactly the same as in tgt object

bool RAttrBase::IsSame(const RAttrBase &tgt, bool use_style) const
{
   for (const auto &entry : GetDefaults()) {
      if (auto v = AccessValue(entry.first, use_style))
         if (!tgt.IsValueEqual(entry.first, *v.value, use_style))
            return false;
   }

   return true;
}

///////////////////////////////////////////////////////////////////////////////
/// Assign drawable object for this RAttrBase

void RAttrBase::AssignDrawable(RDrawable *drawable, const std::string &prefix)
{
   ClearData();
   fKind = kDrawable;
   fD.drawable = drawable;

   fPrefix = prefix;
   if (!IsValue() && !fPrefix.empty()) fPrefix.append("_"); // naming convention
}

///////////////////////////////////////////////////////////////////////////////
/// Assign parent object for this RAttrBase

void RAttrBase::AssignParent(RAttrBase *parent, const std::string &prefix)
{
   ClearData();
   fKind = kParent;
   fD.parent = parent;

   fPrefix = prefix;
   if (!IsValue() && !fPrefix.empty()) fPrefix.append("_"); // naming convention
}

///////////////////////////////////////////////////////////////////////////////
/// Clear value if any with specified name

void RAttrBase::ClearValue(const std::string &name)
{
   if (auto access = AccessAttr(name))
       access.attr->Clear(access.fullname);
}

///////////////////////////////////////////////////////////////////////////////
/// Set <NoValue> for attribute. Ensure that value can not be configured via style - defaults will be used
/// Equivalent to css syntax { attrname:; }

void RAttrBase::SetNoValue(const std::string &name)
{
   if (auto access = AccessAttr(name))
       access.attr->AddNoValue(access.fullname);
}

///////////////////////////////////////////////////////////////////////////////
/// Set boolean value

void RAttrBase::SetValue(const std::string &name, bool value)
{
   if (auto access = EnsureAttr(name))
      access.attr->AddBool(access.fullname, value);
}

///////////////////////////////////////////////////////////////////////////////
/// Set integer value

void RAttrBase::SetValue(const std::string &name, int value)
{
   if (auto access = EnsureAttr(name))
      access.attr->AddInt(access.fullname, value);
}

///////////////////////////////////////////////////////////////////////////////
/// Set double value

void RAttrBase::SetValue(const std::string &name, double value)
{
   if (auto access = EnsureAttr(name))
      access.attr->AddDouble(access.fullname, value);
}

///////////////////////////////////////////////////////////////////////////////
/// Set string value

void RAttrBase::SetValue(const std::string &name, const std::string &value)
{
   if (auto access = EnsureAttr(name))
      access.attr->AddString(access.fullname, value);
}

///////////////////////////////////////////////////////////////////////////////
/// Set PadLength value

void RAttrBase::SetValue(const std::string &name, const RPadLength &value)
{
   if (value.Empty())
      ClearValue(name);
   else
      SetValue(name, value.AsString());
}

///////////////////////////////////////////////////////////////////////////////
/// Set RColor value

void RAttrBase::SetValue(const std::string &name, const RColor &value)
{
   if (value.IsEmpty())
      ClearValue(name);
   else
      SetValue(name, value.AsString());
}

///////////////////////////////////////////////////////////////////////////////
/// Clear all respective values from drawable. Only defaults can be used

void RAttrBase::Clear()
{
   for (const auto &entry : GetDefaults())
      ClearValue(entry.first);
}


///////////////////////////////////////////////////////////////////////////////
/// Collect all attributes in derived class
/// Works only if such class has dictionary.
/// In special cases one has to provide special implementation directly

RAttrMap RAttrBase::CollectDefaults() const
{
   RAttrMap res;

   const std::type_info &info = typeid(*this);
   auto thisClass = TClass::GetClass(info);
   auto baseClass = TClass::GetClass<RAttrBase>();
   if (thisClass && baseClass) {
      for (auto data_member: TRangeDynCast<TDataMember>(thisClass->GetListOfDataMembers())) {
         if (data_member && data_member->GetClass() && data_member->GetClass()->InheritsFrom(baseClass) &&
             (data_member->GetClass()->GetBaseClassOffset(baseClass) == 0)) {
               res.AddDefaults(*((const RAttrBase *)((char*) this + data_member->GetOffset())));
         }
      }
   } else {
      R__LOG_ERROR(GPadLog()) << "Missing dictionary for " << info.name() << " class";
   }

   return res;
}
