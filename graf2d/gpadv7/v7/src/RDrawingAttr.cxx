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

#include "ROOT/RDrawingOptsBase.hxx"
#include "ROOT/RLogger.hxx"

#include <algorithm>
#include <iterator>

ROOT::Experimental::RDrawingAttrBase::RDrawingAttrBase(const Name &name, const RDrawingAttrBase &parent) :
   fPath(parent.GetPath() + name), fHolder(parent.GetHolderPtr())
{
}

ROOT::Experimental::RDrawingAttrBase::RDrawingAttrBase(FromOption_t, const Name &name, RDrawingOptsBase &opts) :
   fPath{name.fStr}, fHolder(opts.GetHolder())
{
}

ROOT::Experimental::RDrawingAttrBase &ROOT::Experimental::RDrawingAttrBase::operator=(const RDrawingAttrBase& rhs)
{
   auto otherHolder = rhs.fHolder.lock();
   if (!otherHolder)
      return *this;

   auto thisHolder = fHolder.lock();
   if (!thisHolder)
      return *this;

   // First, remove all attributes in fPath; we will replace them with what's in rhs (if any).
   thisHolder->EraseAttributesInPath(fPath);
   thisHolder->CopyAttributesInPath(fPath, *otherHolder, rhs.fPath);
   return *this;
}

void ROOT::Experimental::RDrawingAttrBase::SetValueString(const Name &name, const std::string &strVal)
{
   if (auto holder = GetHolderPtr().lock())
      holder->At(GetPath() + name) = strVal;
}

std::string ROOT::Experimental::RDrawingAttrBase::GetValueString(const Path &path) const
{
   auto holder = GetHolderPtr().lock();
   if (!holder)
      return "";

   if (const std::string *pStr = holder->AtIf(path))
      return *pStr;
   return holder->GetAttrFromStyle(path);
}

bool ROOT::Experimental::RDrawingAttrBase::IsFromStyle(const Path& path) const
{
   auto holder = GetHolderPtr().lock();
   if (!holder)
      return "";

   return !holder->AtIf(path);
}

bool ROOT::Experimental::RDrawingAttrBase::IsFromStyle(const Name& name) const
{
   return IsFromStyle(GetPath() + name);
}

bool ROOT::Experimental::RDrawingAttrBase::operator==(const RDrawingAttrBase &other) const
{
   auto thisHolder = GetHolderPtr().lock();
   auto otherHolder = other.GetHolderPtr().lock();
   if (!thisHolder && !otherHolder)
      return true;
   if (!thisHolder != !otherHolder)
      return false;

   // We have valid holders for both.
   return thisHolder->Equal(*otherHolder.get(), GetPath(), other.GetPath());
}

float ROOT::Experimental::FromAttributeString(const std::string &val, const std::string &/*name*/, float *)
{
   return std::stof(val);
}

double ROOT::Experimental::FromAttributeString(const std::string &val, const std::string &/*name*/, double *)
{
   return std::stod(val);
}

int ROOT::Experimental::FromAttributeString(const std::string &val, const std::string &/*name*/, int*)
{
   return std::stoi(val);
}

std::string ROOT::Experimental::ToAttributeString(float val)
{
   return std::to_string(val);
}

std::string ROOT::Experimental::ToAttributeString(double val)
{
   return std::to_string(val);
}

std::string ROOT::Experimental::ToAttributeString(int val)
{
   return std::to_string(val);
}

const std::string *ROOT::Experimental::RDrawingAttrHolder::AtIf(const Path_t &path) const
{
   auto it = fAttrNameVals.find(path.fStr);
   if (it != fAttrNameVals.end())
      return &it->second;
   return nullptr;
}

std::string ROOT::Experimental::RDrawingAttrHolder::GetAttrFromStyle(const Path_t &path)
{
   R__WARNING_HERE("Graf2d") << "Failed to get attribute for "
      << path.fStr << ": not yet implemented!";
   return "";
}

bool ROOT::Experimental::RDrawingAttrHolder::Equal(const RDrawingAttrHolder &other, const Path_t &thisPath, const Path_t &otherPath)
{
   std::vector<Map_t::const_iterator> thisIters = GetAttributesInPath(thisPath);
   std::vector<Map_t::const_iterator> otherIters = other.GetAttributesInPath(otherPath);

   if (thisIters.size() != otherIters.size())
      return false;

   for (auto thisIter: thisIters) {
      // thisIters and otherIters have equal size. If any element in thisIters does not exist
      // in other.fAttrNameVals then they are not equal (if other.fAttrNameVals has an entry that
      // does not exist in this->fAttrNameVals, there must also be a key in this->fAttrNameVals
      // that does not exist in other.fAttrNameVals or the counts of thisIters and otherIters
      // would differ).
      // If all keys' values are equal, thisIters and otherIters are equal.
      auto otherIter = other.fAttrNameVals.find(thisIter->first);
      if (otherIter == other.fAttrNameVals.end())
         return false;
      if (thisIter->second != otherIter->second)
         return false;
   }
   return true;
}

std::vector<ROOT::Experimental::RDrawingAttrHolder::Map_t::const_iterator>
ROOT::Experimental::RDrawingAttrHolder::GetAttributesInPath(const Path_t &path) const
{
   std::vector<Map_t::const_iterator> ret;
   const std::string &stem = path.fStr;
   for (auto i = fAttrNameVals.begin(), e = fAttrNameVals.end(); i !=e; ++i)
      if (i->first.compare(0, stem.length(), stem) == 0) {
         // Require i->first to be complete stem, or more but then stem followed by ".":
         // stem "a.b", i->first can be "a.b" or "a.b.c.d"
         if (stem.length() == i->first.length()
             || i->first[stem.length()] == '.')
         ret.emplace_back(i);
      }
   return ret;
}

void ROOT::Experimental::RDrawingAttrHolder::EraseAttributesInPath(const Path_t &path)
{
   // Iterators are stable under erase()ing!
   auto iters = GetAttributesInPath(path);
   for (auto iter: iters)
      fAttrNameVals.erase(iter);
}


void ROOT::Experimental::RDrawingAttrHolder::CopyAttributesInPath(const Path_t &targetPath, const RDrawingAttrHolder &source, const Path_t &sourcePath)
{
   auto sourceIters = source.GetAttributesInPath(sourcePath);
   if (targetPath != sourcePath) {
      for (auto sourceIter: sourceIters)
         fAttrNameVals.emplace(sourceIter->first, sourceIter->second);
   } else {
      for (auto sourceIter: sourceIters) {
         std::string newPath = targetPath.fStr + sourceIter->first.substr(sourcePath.fStr.length());
         fAttrNameVals.emplace(newPath, sourceIter->second);
      }
   }
}

///////////////////////////////////////////////////////////////////////////////


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

const ROOT::Experimental::RDrawableAttributes::Value_t *ROOT::Experimental::RStyleNew::Eval(const std::string &type, const std::string &user_class, const std::string &field) const
{
   for (const auto &block : fBlocks) {

      bool match = (block.selector == type) || (!user_class.empty() && (block.selector == "."s + user_class));

      if (match) {
         auto res = block.map.Eval(field);
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

void ROOT::Experimental::RAttributesVisitor::Copy(const RAttributesVisitor &src, bool use_dflts)
{
   if (!GetAttr(true)) return;

   bool same_dflts = &GetDefaults() == &src.GetDefaults();

   for (const auto &entry : src.GetDefaults()) {
      const auto *value = src.Eval(entry.first, use_dflts);

      // check if element with given name exists at all
      if (!same_dflts && value) {
         const auto *dvalue = GetDefaults().Eval(entry.first);
         if (!dvalue || !dvalue->Compatible(value->Kind()))
            value = nullptr;
      }

      if (value)
         fAttr->map.Add(GetFullName(entry.first), value->Copy());
   }
}

///////////////////////////////////////////////////////////////////////////////
/// Semantic copy attributes from other object
/// Search in the container all attributes which match source prefix and copy them

void ROOT::Experimental::RAttributesVisitor::SemanticCopy(const RAttributesVisitor &src)
{
   if (!src.GetAttr() || !GetAttr(true)) return;

   for (const auto &pair : src.fAttr->map) {
      auto attrname = pair.first;

      if (!src.fPrefix.empty()) {
         if (!attrname.compare(9, src.fPrefix.length(), src.fPrefix)) continue;
         attrname.erase(0, src.fPrefix.length());
      }

      if (!attrname.empty())
         fAttr->map.Add(GetFullName(attrname), pair.second->Copy());
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

const ROOT::Experimental::RDrawableAttributes::Value_t *ROOT::Experimental::RAttributesVisitor::GetValue(const std::string &name) const
{
   return GetAttr() ? fAttr->map.Eval(GetFullName(name)) : nullptr;
}

///////////////////////////////////////////////////////////////////////////////
/// Evaluate attribute value

const ROOT::Experimental::RDrawableAttributes::Value_t *ROOT::Experimental::RAttributesVisitor::Eval(const std::string &name, bool use_dflts) const
{
   const RDrawableAttributes::Value_t *res = nullptr;

   if (GetAttr()) {
      auto fullname = GetFullName(name);

      res = fAttr->map.Eval(fullname);
      if (res) return res;

      const auto *prnt = this;

      while (prnt) {
         if (prnt->fStyle)
            res = prnt->fStyle->Eval(fAttr->type, fAttr->user_class, fullname);
         if (res) return res;
         prnt = prnt->fParent;
      }
   }

   if (use_dflts) {
      res = GetDefaults().Eval(name);
      if (res) return res;
   }

   if (use_dflts && fAttr && fAttr->defaults) {
      res = fAttr->defaults->Eval(GetFullName(name));
   }

   return res;
}

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

std::string ROOT::Experimental::RAttributesVisitor::GetString(const std::string &name) const
{
   auto res = Eval(name);
   if (!res || !res->Compatible(RDrawableAttributes::kString)) return ""s;
   return res->GetString();
}

int ROOT::Experimental::RAttributesVisitor::GetInt(const std::string &name) const
{
   auto res = Eval(name);
   if (!res || !res->Compatible(RDrawableAttributes::kInt)) return 0;
   return res->GetInt();
}

double ROOT::Experimental::RAttributesVisitor::GetDouble(const std::string &name) const
{
   auto res = Eval(name);
   if (!res || !res->Compatible(RDrawableAttributes::kDouble)) return 0.;
   return res->GetDouble();
}
