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


ROOT::Experimental::RDrawable::RDrawable()
{
}


// pin vtable
ROOT::Experimental::RDrawable::~RDrawable() {}

void ROOT::Experimental::RDrawable::Execute(const std::string &)
{
   assert(false && "Did not expect a menu item to be invoked!");
}


//////////////////////////////////////////////////////////////////////////////////////

const char *ROOT::Experimental::RAttributesContainer::Eval(const std::string &name) const
{
   auto cont = GetContainer();
   if (cont) {
      auto entry = cont->find(name);
      if (entry != cont->end())
         return entry->second.c_str();
   }

   return nullptr;
}


void ROOT::Experimental::RAttributesContainer::SetValue(const std::string &name, const char *val)
{
   if (val) {

      auto cont = MakeContainer();

      (*cont)[name] = val;

   } else if (GetContainer()) {

      auto elem = fCont->find(name);
      if (elem != fCont->end())
         fCont->erase(elem);
   }
}

void ROOT::Experimental::RAttributesContainer::SetValue(const std::string &name, const std::string &value)
{
   auto cont = MakeContainer();

   (*cont)[name] = value;
}

void ROOT::Experimental::RAttributesContainer::Clear()
{
   // special case when container was read by I/O but not yet assigned to
   if (fContIO && !fCont)
      delete fContIO;

   fContIO = nullptr;
   fCont.reset();
}


///////////////////////////////////////////////////////////////////////////////

const char *ROOT::Experimental::RAttributesVisitor::Eval(const std::string &name) const
{
   if (fFirstTime) {
      fFirstTime = false;
      if (!fCont)
         fCont = fWeak.lock();
   }

   if (fCont) {
      auto entry = fCont->find(GetFullName(name));
      if (entry != fCont->end())
         return entry->second.c_str();
   }

   if (fDefaults) {
      const auto centry = fDefaults->find(name);
      if (centry != fDefaults->end())
         return centry->second.c_str();
   }

/*   if (fDrawable.fDefaults) {
      const auto centry = fDrawable.fDefaults->find(GetFullName(name));
      if (centry != fDrawable.fDefaults->end())
         return centry->second.c_str();
   }
*/
   return nullptr;
}

void ROOT::Experimental::RAttributesVisitor::SetValue(const std::string &name, const char *val)
{
   if (fFirstTime) {
      fFirstTime = false;
      if (!fCont)
         fCont = fWeak.lock();
   }

   if (!fCont)
      return;

   if (val) {

      (*fCont)[GetFullName(name)] = val;

   } else {

      auto elem = fCont->find(GetFullName(name));
      if (elem != fCont->end())
         fCont->erase(elem);

   }
}

void ROOT::Experimental::RAttributesVisitor::SetValue(const std::string &name, const std::string &value)
{
   if (fFirstTime) {
      fFirstTime = false;
      if (!fCont)
         fCont = fWeak.lock();
   }

   if (fCont)
      (*fCont)[GetFullName(name)] = value;
}

/** Clear all respective values from drawable. Only defaults can be used */
void ROOT::Experimental::RAttributesVisitor::Clear()
{
   if (fDefaults)
      for (const auto &entry : *fDefaults)
         ClearValue(entry.first);
}


int ROOT::Experimental::RAttributesVisitor::GetInt(const std::string &name) const
{
   auto res = Eval(name);
   return res ? std::stoi(res) : 0;
}

void ROOT::Experimental::RAttributesVisitor::SetInt(const std::string &name, const int value)
{
   SetValue(name, std::to_string(value));
}

float ROOT::Experimental::RAttributesVisitor::GetFloat(const std::string &name) const
{
   auto res = Eval(name);
   return res ? std::stof(res) : 0.;
}

void ROOT::Experimental::RAttributesVisitor::SetFloat(const std::string &name, const float value)
{
   SetValue(name, std::to_string(value));
}
