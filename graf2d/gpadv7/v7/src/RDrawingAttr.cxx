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
#include "ROOT/RStyle.hxx"
#include "ROOT/TLogger.hxx"

#include <algorithm>
#include <iterator>

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

void ROOT::Experimental::RDrawingAttrBase::InitializeFromStyle(const Path &path, const RDrawingAttrHolder &holder)
{
   for (const MemberAssociation &assoc: GetMembers()) {
      Path attrPath = path + assoc.fName;
      if (assoc.fNestedAttr)
         assoc.fNestedAttr->InitializeFromStyle(attrPath, holder);
      else
         assoc.fSetMemberFromString(holder.GetAttrValStringFromStyle(attrPath), attrPath.Str());
   }
}

void ROOT::Experimental::RDrawingAttrBase::InsertModifiedAttributeStrings(const Path &path, const RDrawingAttrHolder &holder,
                                                                          std::vector<std::pair<std::string, std::string>> &keyval)
{
   for (MemberAssociation &assoc: GetMembers()) {
      Path attrPath = path + assoc.fName;
      if (assoc.fNestedAttr)
         assoc.fNestedAttr->InsertModifiedAttributeStrings(attrPath, holder, keyval);
      else {
         std::string val = assoc.fMemberToString();
         if (val != holder.GetAttrValStringFromStyle(attrPath))
            keyval.emplace_back(attrPath.Str(), val);
      }
   }
}

ROOT::Experimental::RDrawingAttrBase *ROOT::Experimental::RDrawingAttrHolder::AtIf(const Name_t &name) const
{
   auto it = fAttrNameVals.find(name.fStr);
   if (it != fAttrNameVals.end())
      return it->second.get();
   return nullptr;
}

std::string ROOT::Experimental::RDrawingAttrHolder::GetAttrValStringFromStyle(const Path_t &path) const
{
   for (auto &&cls: GetStyleClasses()) {
      std::string val = RStyle::GetCurrent().GetAttribute(path.Str(), cls);
      if (!val.empty())
         return val;
   }
   return RStyle::GetCurrent().GetAttribute(path.Str());
}

std::vector<std::pair<std::string, std::string>> ROOT::Experimental::RDrawingAttrHolder::CustomizedValuesToString(const Name_t &option_name)
{
   std::vector<std::pair<std::string, std::string>> ret;
   Path_t path(option_name);
   for (auto &&name_attr: fAttrNameVals)
      name_attr.second->InsertModifiedAttributeStrings(path + name_attr.first, *this, ret);
   return ret;
}
