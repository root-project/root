/// \file RDrawingAttrBase.cxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-09-26
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDrawingAttr.hxx"

#include "ROOT/RLogger.hxx"

#include <algorithm>
#include <iterator>

template<> bool ROOT::Experimental::RDrawableAttributes::Value_t::get<bool>() const { return GetBool(); }
template<> int ROOT::Experimental::RDrawableAttributes::Value_t::get<int>() const { return GetInt(); }
template<> double ROOT::Experimental::RDrawableAttributes::Value_t::get<double>() const { return GetDouble(); }
template<> std::string ROOT::Experimental::RDrawableAttributes::Value_t::get<std::string>() const { return GetString(); }

template<> bool ROOT::Experimental::RDrawableAttributes::Value_t::get_value<bool,void>(const Value_t *rec) { return rec ? rec->GetBool() : false; }
template<> int ROOT::Experimental::RDrawableAttributes::Value_t::get_value<int,void>(const Value_t *rec) { return rec ? rec->GetInt() : 0; }
template<> double ROOT::Experimental::RDrawableAttributes::Value_t::get_value<double,void>(const Value_t *rec) { return rec ? rec->GetDouble() : 0.; }
template<> std::string ROOT::Experimental::RDrawableAttributes::Value_t::get_value<std::string,void>(const Value_t *rec) { return rec ? rec->GetString() : ""; }

template<> const ROOT::Experimental::RDrawableAttributes::Value_t *ROOT::Experimental::RDrawableAttributes::Value_t::get_value<const ROOT::Experimental::RDrawableAttributes::Value_t *,void>(const Value_t *rec) { return rec; }
template<> const ROOT::Experimental::RDrawableAttributes::Value_t *ROOT::Experimental::RDrawableAttributes::Value_t::get_value<const ROOT::Experimental::RDrawableAttributes::Value_t *,bool>(const Value_t *rec) { return rec && rec->Kind() == RDrawableAttributes::kBool ? rec : nullptr; }
template<> const ROOT::Experimental::RDrawableAttributes::Value_t *ROOT::Experimental::RDrawableAttributes::Value_t::get_value<const ROOT::Experimental::RDrawableAttributes::Value_t *,int>(const Value_t *rec) { return rec && rec->Kind() == RDrawableAttributes::kInt ? rec : nullptr; }
template<> const ROOT::Experimental::RDrawableAttributes::Value_t *ROOT::Experimental::RDrawableAttributes::Value_t::get_value<const ROOT::Experimental::RDrawableAttributes::Value_t *,double>(const Value_t *rec) { return rec && rec->Kind() == RDrawableAttributes::kDouble ? rec : nullptr; }
template<> const ROOT::Experimental::RDrawableAttributes::Value_t *ROOT::Experimental::RDrawableAttributes::Value_t::get_value<const ROOT::Experimental::RDrawableAttributes::Value_t *,std::string>(const Value_t *rec) { return rec && rec->Kind() == RDrawableAttributes::kString ? rec : nullptr;  }


using namespace std::string_literals;

ROOT::Experimental::RDrawableAttributes::Map_t &ROOT::Experimental::RDrawableAttributes::Map_t::AddDefaults(const RAttributesVisitor &vis)
{
   auto prefix = vis.GetPrefixToParent();

   for (const auto &entry : vis.GetDefaults())
      m[prefix+entry.first] = std::unique_ptr<Value_t>(entry.second->Copy());

   return *this;
}

///////////////////////////////////////////////////////////////////////////////
/// Evaluate style

const ROOT::Experimental::RDrawableAttributes::Value_t *ROOT::Experimental::RStyle::Eval(const std::string &type, const std::string &user_class, const std::string &field) const
{
   for (const auto &block : fBlocks) {

      bool match = (block.selector == type) || (!user_class.empty() && (block.selector == "."s + user_class));

      if (match) {
         auto res = block.map.Find(field);
         if (res) return res;
      }
   }

   return nullptr;
}

///////////////////////////////////////////////////////////////////////////////
/// Returns prefix relative to parent
/// Normally prefix is relative to

std::string ROOT::Experimental::RAttributesVisitor::GetPrefixToParent() const
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


void ROOT::Experimental::RAttributesVisitor::CreateOwnAttr()
{
   // create independent container
   fOwnAttr = std::make_unique<RDrawableAttributes>();

   // set pointer on the container
   fAttr = fOwnAttr.get();
}


///////////////////////////////////////////////////////////////////////////////
/// Copy attributes from other object

bool ROOT::Experimental::RAttributesVisitor::CopyValue(const std::string &name, const RDrawableAttributes::Value_t *value, bool check_type)
{
   if (!value) return false;

   if (check_type) {
      const auto *dvalue = GetDefaults().Find(name);
      if (!dvalue || !dvalue->Compatible(value->Kind()))
         return false;
   }

   if (!GetAttr(true))
      return false;

   fAttr->map.Add(GetFullName(name), value->Copy());

   return true;
}


///////////////////////////////////////////////////////////////////////////////
/// Copy attributes into target object

void ROOT::Experimental::RAttributesVisitor::CopyTo(RAttributesVisitor &tgt, bool use_style) const
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
/// Semantic copy attributes from other object
/// Search in the container all attributes which match source prefix and copy them

void ROOT::Experimental::RAttributesVisitor::SemanticCopy(const RAttributesVisitor &src)
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

bool ROOT::Experimental::RAttributesVisitor::GetAttr(bool force) const
{
   if (fAttr)
      return true;

   auto prnt = fParent;
   auto prefix = fPrefix;
   while (prnt) {
      if (force && !prnt->fParent && !prnt->fAttr)
         const_cast<RAttributesVisitor*>(prnt)->CreateOwnAttr();
      if (prnt->fAttr) {
         const_cast<RAttributesVisitor*>(this)->fAttr = prnt->fAttr;
         const_cast<RAttributesVisitor*>(this)->fPrefix = prnt->fPrefix + prefix;
         return true;
      }
      prefix = prnt->fPrefix + prefix;
      prnt = prnt->fParent;
   }

   if (!fParent && force)
      const_cast<RAttributesVisitor*>(this)->CreateOwnAttr();

   return fAttr != nullptr;
}

///////////////////////////////////////////////////////////////////////////////
/// Return value from attributes container - no style or defaults are used

void ROOT::Experimental::RAttributesVisitor::ClearValue(const std::string &name)
{
   if (GetAttr())
      fAttr->map.Clear(GetFullName(name));
}

void ROOT::Experimental::RAttributesVisitor::SetValue(const std::string &name, int value)
{
   if (GetAttr(true))
      fAttr->map.AddInt(GetFullName(name), value);
}

void ROOT::Experimental::RAttributesVisitor::SetValue(const std::string &name, double value)
{
   if (GetAttr(true))
      fAttr->map.AddDouble(GetFullName(name), value);
}

double *ROOT::Experimental::RAttributesVisitor::GetDoublePtr(const std::string &name) const
{
   return GetAttr() ? fAttr->map.GetDoublePtr(GetFullName(name)) : nullptr;
}

void ROOT::Experimental::RAttributesVisitor::SetValue(const std::string &name, const std::string &value)
{
   if (GetAttr(true))
      fAttr->map.AddString(GetFullName(name), value);
}

/** Clear all respective values from drawable. Only defaults can be used */
void ROOT::Experimental::RAttributesVisitor::Clear()
{
   if (GetAttr())
      for (const auto &entry : GetDefaults())
         fAttr->map.Clear(GetFullName(entry.first));
}
