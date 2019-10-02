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
#include "ROOT/RDisplayItem.hxx"


#include <cassert>
#include <string>


// pin vtable
ROOT::Experimental::RDrawable::~RDrawable() {}

void ROOT::Experimental::RDrawable::Execute(const std::string &)
{
   assert(false && "Did not expect a menu item to be invoked!");
}

/////////////////////////////////////////////////////////////////////////////
/// Preliminary method which checks if drawable matches with given selector
/// Following selector are allowed:
///   "type" or "#id" or ".class_name"
///  Here type is drawable kind like 'rect' or 'pad'
///       id is drawable identifier, specified with RDrawable::SetId() method
///       class_name is drawable class name, specified with RDrawable::SetCssClass() method

bool ROOT::Experimental::RDrawable::MatchSelector(const std::string &selector) const
{
   return (selector == fCssType) || (!fCssClass.empty() && (selector == std::string(".") + fCssClass)) || (!fId.empty() && (selector == std::string("#") + fId));
}

/////////////////////////////////////////////////////////////////////////////
/// Creates display item for drawable
/// By default item contains drawble data itself

std::unique_ptr<ROOT::Experimental::RDisplayItem> ROOT::Experimental::RDrawable::Display() const
{
   return std::make_unique<RDrawableDisplayItem>(*this);
}
