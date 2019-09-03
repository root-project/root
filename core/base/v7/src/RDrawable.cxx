/// \file RDrawable.cxx
/// \ingroup Base ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-07-08
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDrawable.hxx"

#include <cassert>
#include <string>



ROOT::Experimental::RDrawableAttributesContainer gNoDefaults = {};



ROOT::Experimental::RDrawable::RDrawable() : fDefaults(gNoDefaults)
{
}


// pin vtable
ROOT::Experimental::RDrawable::~RDrawable() {}

void ROOT::Experimental::RDrawable::Execute(const std::string &)
{
   assert(false && "Did not expect a menu item to be invoked!");
}


void ROOT::Experimental::RDrawable::ClearAttributes()
{
   fNewAttributes.reset();
}


ROOT::Experimental::RDrawableAttributesNew::RDrawableAttributesNew(RDrawable &d, const std::string &prefix) :
    fDrawable(d), fPrefix(prefix), fDefaults(gNoDefaults)
{
}


const char *ROOT::Experimental::RDrawableAttributesNew::Eval(const std::string &name) const
{
   if (fDrawable.fNewAttributes) {
      auto entry = fDrawable.fNewAttributes->find(GetFullName(name));
      if (entry != fDrawable.fNewAttributes->end())
         return entry->second.c_str();
   }

   auto entry = fDefaults.find(name);
   if (entry != fDefaults.end())
     return entry->second.c_str();

   const auto centry = fDrawable.fDefaults.find(GetFullName(name));
   if (centry != fDrawable.fDefaults.end())
      return centry->second.c_str();

   return nullptr;
}

void ROOT::Experimental::RDrawableAttributesNew::SetValue(const std::string &name, const char *val)
{
   if (val) {

      if (!fDrawable.fNewAttributes)
         fDrawable.fNewAttributes = std::make_unique<RDrawableAttributesContainer>();

      fDrawable.fNewAttributes->at(GetFullName(name)) = val;

   } else if (fDrawable.fNewAttributes) {
      auto elem = fDrawable.fNewAttributes->find(GetFullName(name));
      if (elem != fDrawable.fNewAttributes->end()) {
         fDrawable.fNewAttributes->erase(elem);
         if (fDrawable.fNewAttributes->size() == 0)
            fDrawable.fNewAttributes.reset();
      }
   }
}

void ROOT::Experimental::RDrawableAttributesNew::SetValue(const std::string &name, const std::string &value)
{
   if (!fDrawable.fNewAttributes)
      fDrawable.fNewAttributes = std::make_unique<RDrawableAttributesContainer>();

   fDrawable.fNewAttributes->at(GetFullName(name)) = value;
}

/** Clear all respective values from drawable. Only defaults can be used */
void ROOT::Experimental::RDrawableAttributesNew::Clear()
{
   for (const auto &entry : fDefaults)
      ClearValue(entry.first);
}


int ROOT::Experimental::RDrawableAttributesNew::GetInt(const std::string &name) const
{
   auto res = Eval(name);
   return res ? std::stoi(res) : 0;
}

void ROOT::Experimental::RDrawableAttributesNew::SetInt(const std::string &name, const int value)
{
   SetValue(name, std::to_string(value));
}

float ROOT::Experimental::RDrawableAttributesNew::GetFloat(const std::string &name) const
{
   auto res = Eval(name);
   return res ? std::stof(res) : 0.;
}

void ROOT::Experimental::RDrawableAttributesNew::SetFloat(const std::string &name, const float value)
{
   SetValue(name, std::to_string(value));
}

