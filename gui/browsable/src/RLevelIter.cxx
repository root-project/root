/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/Browsable/RLevelIter.hxx>

#include <ROOT/Browsable/RElement.hxx>

#include <ROOT/RBrowserItem.hxx>

using namespace ROOT::Experimental;
using namespace ROOT::Experimental::Browsable;
using namespace std::string_literals;


/////////////////////////////////////////////////////////////////////
/// Find item with specified name
/// Default implementation, should work for all

bool RLevelIter::Find(const std::string &name)
{
   if (!Reset()) return false;

   while (Next()) {
      if (GetName() == name)
         return true;
   }

   return false;
}


/////////////////////////////////////////////////////////////////////
/// Create generic description item for RBrowser

std::unique_ptr<RBrowserItem> RLevelIter::CreateBrowserItem()
{
   return HasItem() ? std::make_unique<RBrowserItem>(GetName(), CanHaveChilds(), CanHaveChilds() > 0 ? "sap-icon://folder-blank" : "sap-icon://document") : nullptr;
}

