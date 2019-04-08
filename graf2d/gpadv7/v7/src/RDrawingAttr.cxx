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

ROOT::Experimental::RDrawingAttrBase::RDrawingAttrBase(const char* namePart, RDrawingAttrHolderBase *holder,
   RDrawingAttrBase *parent):
   fNamePart(namePart), fHolder(holder), fParent(parent)
{
   parent->Register(*this);
}

ROOT::Experimental::RDrawingAttrBase::RDrawingAttrBase(const char* namePart, RDrawingAttrHolderBase *holder,
   RDrawingAttrBase *parent, const std::vector<NamePart_t> &valueNames):
   RDrawingAttrBase(namePart, holder, parent)
{
   fValueNames = std::make_unique<std::vector<NamePart_t>>(valueNames);
}

void ROOT::Experimental::RDrawingAttrBase::GetName(Name_t &name) const
{
   if (fParent)
      fParent->GetName(name);
   name.emplace_back(fNamePart);
}

std::string ROOT::Experimental::RDrawingAttrBase::NameToDottedDiagName(const Name_t &name)
{
   std::stringstream strm;
   std::copy(name.begin(), name.end(),
      std::ostream_iterator<Name_t::value_type>(strm, "."));
   return strm.str();
}

void ROOT::Experimental::RDrawingAttrBase::CollectChildNames(std::vector<Name_t> &names) const
{
   if (fValueNames)
      std::transform(fValueNames->begin(), fValueNames->end(), names.end(),
         [](NamePart_t name) { return Name_t{name}; } );
   if (fChildren) {
      for (auto &&ch: *fChildren)
         ch->CollectChildNames(names);
   }
   // Prepend our name to each child:
   for (auto &name: names)
      name.insert(name.begin(), fNamePart);
}

void ROOT::Experimental::RDrawingAttrBase::Register(const RDrawingAttrBase &subAttr)
{
   if (!fChildren)
      fChildren.reset(new decltype(fChildren)::element_type);
   fChildren->push_back(&subAttr);
}


ROOT::Experimental::RDrawingAttrBase::Name_t ROOT::Experimental::RDrawingAttrBase::BuildNameForVal(std::size_t valueIndex) const
{
   Name_t name;
   GetName(name);
   if (!fValueNames) {
      R__ERROR_HERE("Graf2d") << "attribute " << NameToDottedDiagName(name) << "has no attribute values";
      return {};
   }
   name.emplace_back((*fValueNames)[valueIndex]);
   return name;
}

void ROOT::Experimental::RDrawingAttrBase::Set(std::size_t valueIndex, const std::string &strVal)
{
   GetHolder()->At(BuildNameForVal(valueIndex)) = strVal;
}

std::pair<std::string, bool> ROOT::Experimental::RDrawingAttrBase::Get(std::size_t valueIndex) const
{
   Name_t name = BuildNameForVal(valueIndex);
   if (const std::string *pStr = GetHolder()->AtIf(name))
      return {*pStr, true};
   return {GetHolder()->GetAttrFromStyle(name), false};
}


// pin vtable.
ROOT::Experimental::RDrawingAttrHolderBase::~RDrawingAttrHolderBase() = default;

namespace {
static void HashCombine(std::size_t& seed, const std::string& v)
{
   seed ^= std::hash<std::string>()(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}
}

std::size_t ROOT::Experimental::RDrawingAttrHolderBase::StringVecHash::operator()(const Name_t &vec) const {
   std::size_t hash = std::hash<std::size_t>()(vec.size());
   for (auto &&el: vec)
      HashCombine(hash, el);
   return hash;
}

const std::string *ROOT::Experimental::RDrawingAttrHolderBase::AtIf(const Name_t &attrName) const
{
   auto it = fAttrNameVals.find(attrName);
   if (it != fAttrNameVals.end())
      return &it->second;
   return nullptr;
}
