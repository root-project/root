/// \file ROOT/RAttrBase.cxx
/// \ingroup Gpad ROOT7
/// \author Sergey Linev <s.linev@gsi.de>
/// \date 2019-09-17
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RAttrBase.hxx>

#include "ROOT/RLogger.hxx"

///////////////////////////////////////////////////////////////////////////////
/// Returns prefix relative to parent
/// Normally prefix is relative to

std::string ROOT::Experimental::RAttrBase::GetPrefixToParent() const
{
   if (!fAttr || !fParent) return fPrefix;

   if (!fParent->GetAttr()) return fPrefix;

   if (fParent->fAttr != fAttr) {
      R__ERROR_HERE("Graf2d") << "Mismatch in parent/child attributes containers";
      return fPrefix;
   }

   if (fParent->fPrefix.empty())
      return fPrefix;

   return fPrefix.substr(fParent->fPrefix.length());
}


///////////////////////////////////////////////////////////////////////////////
/// Create own attributes

void ROOT::Experimental::RAttrBase::CreateOwnAttr()
{
   // create independent container
   fOwnAttr = std::make_unique<RAttrValues>();

   // set pointer on the container
   fAttr = fOwnAttr.get();
}


///////////////////////////////////////////////////////////////////////////////
/// Copy attributes from other object

bool ROOT::Experimental::RAttrBase::CopyValue(const std::string &name, const RAttrValues::Value_t *value, bool check_type)
{
   if (!value)
      return false;

   if (check_type) {
      const auto *dvalue = GetDefaults().Find(name);
      if (!dvalue || !dvalue->Compatible(value->Kind()))
         return false;
   }

   if (!EnsureAttr())
      return false;

   fAttr->map.Add(GetFullName(name), value->Copy());

   return true;
}

///////////////////////////////////////////////////////////////////////////////
/// Copy attributes into target object

bool ROOT::Experimental::RAttrBase::IsValueEqual(const std::string &name, const RAttrValues::Value_t *value, bool use_style) const
{
   if (!GetAttr() || !value)
      return false;

   auto fullname = GetFullName(name);

   auto value2 = fAttr->map.Find(fullname);
   if (value2) return value2->IsEqual(value);

   const auto *prnt = this;
   while (prnt && use_style) {
      if (auto observe = const_cast<RAttrBase*> (prnt)->fStyle.lock()) {
         value2 = observe->Eval(fAttr->type, fAttr->user_class, fullname);
         if (value2) return value2->IsEqual(value);
      }
      prnt = prnt->fParent;
   }

   return false;
}

///////////////////////////////////////////////////////////////////////////////
/// Copy attributes into target object

void ROOT::Experimental::RAttrBase::CopyTo(RAttrBase &tgt, bool use_style) const
{
   if (GetAttr())
      for (const auto &entry : GetDefaults()) {

         auto fullname = GetFullName(entry.first);

         auto rec = fAttr->map.Find(fullname);
         if (rec && tgt.CopyValue(entry.first,rec)) continue;

         const auto *prnt = this;
         while (prnt && use_style) {
            if (auto observe = prnt->fStyle.lock()) {
               rec = observe->Eval(fAttr->type, fAttr->user_class, fullname);
               if (rec && tgt.CopyValue(entry.first, rec)) break;
            }
            prnt = prnt->fParent;
         }
      }
}

///////////////////////////////////////////////////////////////////////////////
/// Check if all values which are evaluated in this object are exactly the same as in tgt object

bool ROOT::Experimental::RAttrBase::IsSame(const RAttrBase &tgt, bool use_style) const
{
   if (GetAttr())
      for (const auto &entry : GetDefaults()) {

         auto fullname = GetFullName(entry.first);
         auto rec = fAttr->map.Find(fullname);

         if (rec) {
            if (!tgt.IsValueEqual(entry.first, rec, use_style)) return false;
            continue;
         }

         const auto *prnt = this;
         while (prnt && use_style) {
            if (auto observe = const_cast<RAttrBase *>(prnt)->fStyle.lock()) {
               rec = observe->Eval(fAttr->type, fAttr->user_class, fullname);
               if (rec) {
                  if (!tgt.IsValueEqual(entry.first, rec, use_style)) return false;
                  break;
               }
            }
            prnt = prnt->fParent;
         }
      }
   return true;
}


///////////////////////////////////////////////////////////////////////////////
/// Semantic copy attributes from other object
/// Search in the container all attributes which match source prefix and copy them

void ROOT::Experimental::RAttrBase::SemanticCopy(const RAttrBase &src)
{
   if (!src.GetAttr()) return;

   for (const auto &pair : src.fAttr->map) {
      auto attrname = pair.first;

      if (!src.fPrefix.empty()) {
         if (!attrname.compare(9, src.fPrefix.length(), src.fPrefix)) continue;
         attrname.erase(0, src.fPrefix.length());
      }


      if (!attrname.empty())
         CopyValue(attrname, pair.second.get(), false);
   }
}

///////////////////////////////////////////////////////////////////////////////
/// Access attributes container
/// If pointer not yet assigned, try to find it in parents of just allocate if force flag is specified

bool ROOT::Experimental::RAttrBase::GetAttr() const
{
   if (fAttr)
      return true;

   const RAttrBase *prnt = fParent;
   auto prefix = fPrefix;
   while (prnt) {
      if (prnt->fAttr) {
         const_cast<RAttrBase*>(this)->fAttr = prnt->fAttr;
         const_cast<RAttrBase*>(this)->fPrefix = prnt->fPrefix + prefix;
         return true;
      }
      prefix = prnt->fPrefix + prefix;
      prnt = prnt->fParent;
   }

   return fAttr != nullptr;
}

///////////////////////////////////////////////////////////////////////////////
/// Ensure that attributes container exists
/// If not exists before, created for very most parent

bool ROOT::Experimental::RAttrBase::EnsureAttr()
{
   if (fAttr)
      return true;

   const RAttrBase *prnt = fParent;
   auto prefix = fPrefix;
   while (prnt) {
      if (!prnt->fParent && !prnt->fAttr)
         const_cast<RAttrBase *>(prnt)->CreateOwnAttr();
      if (prnt->fAttr) {
         fAttr = prnt->fAttr;
         fPrefix = prnt->fPrefix + prefix;
         return true;
      }
      prefix = prnt->fPrefix + prefix;
      prnt = prnt->fParent;
   }

   CreateOwnAttr();

   return true;
}


///////////////////////////////////////////////////////////////////////////////
/// Return value from attributes container - no style or defaults are used

void ROOT::Experimental::RAttrBase::ClearValue(const std::string &name)
{
   if (GetAttr())
      fAttr->map.Clear(GetFullName(name));
}

void ROOT::Experimental::RAttrBase::SetValue(const std::string &name, int value)
{
   if (EnsureAttr())
      fAttr->map.AddInt(GetFullName(name), value);
}

void ROOT::Experimental::RAttrBase::SetValue(const std::string &name, double value)
{
   if (EnsureAttr())
      fAttr->map.AddDouble(GetFullName(name), value);
}

double *ROOT::Experimental::RAttrBase::GetDoublePtr(const std::string &name) const
{
   return GetAttr() ? fAttr->map.GetDoublePtr(GetFullName(name)) : nullptr;
}

void ROOT::Experimental::RAttrBase::SetValue(const std::string &name, const std::string &value)
{
   if (EnsureAttr())
      fAttr->map.AddString(GetFullName(name), value);
}

/** Clear all respective values from drawable. Only defaults can be used */
void ROOT::Experimental::RAttrBase::Clear()
{
   if (GetAttr())
      for (const auto &entry : GetDefaults())
         fAttr->map.Clear(GetFullName(entry.first));
}
