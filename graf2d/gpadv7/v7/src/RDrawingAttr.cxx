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
#include "ROOT/TLogger.hxx"

#include <algorithm>
#include <iterator>

ROOT::Experimental::RDrawingAttrBase::RDrawingAttrBase(const std::string &namePart, const RDrawingAttrBase &parent)
:
   fName(parent.GetName()), fHolder(parent.GetHolderPtr())
{
   fName.emplace_back(namePart);
}

ROOT::Experimental::RDrawingAttrBase::RDrawingAttrBase(AsOption_t, const std::string &namePart, RDrawingOptsBase &opts) :
   fName{namePart}, fHolder(opts.GetHolder())
{
}

std::string ROOT::Experimental::RDrawingAttrBase::NameToDottedDiagName(const Name_t &name)
{
   std::stringstream strm;
   std::copy(name.begin(), name.end(),
      std::ostream_iterator<Name_t::value_type>(strm, "."));
   return strm.str();
}

void ROOT::Experimental::RDrawingAttrBase::SetValueString(const std::string &name, const std::string &strVal)
{
   if (auto holder = GetHolderPtr().lock()) {
      Name_t fullName(GetName());
      fullName.emplace_back(name);
      holder->At(fullName) = strVal;
   }
}

std::string ROOT::Experimental::RDrawingAttrBase::GetValueString(const std::string &name) const
{
   auto holder = GetHolderPtr().lock();
   if (!holder)
      return {"", false};

   Name_t fullName(GetName());
   fullName.emplace_back(name);
   if (const std::string *pStr = holder->AtIf(fullName))
      return {*pStr, true};
   return {holder->GetAttrFromStyle(fullName), false};
}

bool ROOT::Experimental::RDrawingAttrBase::IsFromStyle(const std::string& name) const
{
   auto holder = GetHolderPtr().lock();
   if (!holder)
      return "";

   Name_t fullName(GetName());
   fullName.emplace_back(name);
   return !holder->AtIf(fullName);
}

float ROOT::Experimental::FromAttributeString(const std::string &val, const RDrawingAttrBase& /*attr*/, const std::string &/*name*/, float *)
{
   return std::stof(val);
}

double ROOT::Experimental::FromAttributeString(const std::string &val, const RDrawingAttrBase& /*attr*/, const std::string &/*name*/, double *)
{
   return std::stod(val);
}

int ROOT::Experimental::FromAttributeString(const std::string &val, const RDrawingAttrBase& /*attr*/, const std::string &/*name*/, int*)
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

namespace {
static void HashCombine(std::size_t& seed, const std::string& v)
{
   seed ^= std::hash<std::string>()(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}
}

std::size_t ROOT::Experimental::RDrawingAttrHolder::StringVecHash::operator()(const Name_t &vec) const {
   std::size_t hash = std::hash<std::size_t>()(vec.size());
   for (auto &&el: vec)
      HashCombine(hash, el);
   return hash;
}

const std::string *ROOT::Experimental::RDrawingAttrHolder::AtIf(const Name_t &attrName) const
{
   auto it = fAttrNameVals.find(attrName);
   if (it != fAttrNameVals.end())
      return &it->second;
   return nullptr;
}

std::string ROOT::Experimental::RDrawingAttrHolder::GetAttrFromStyle(const Name_t &attrName)
{
    
   R__WARNING_HERE("Graf2d") << "Failed to get attribute for "
      << RDrawingAttrBase::NameToDottedDiagName(attrName) << ": not yet implemented!";
   return "";
}
