/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/Browsable/RLevelIter.hxx>

#include <ROOT/Browsable/RElement.hxx>
#include <ROOT/Browsable/RItem.hxx>

using namespace ROOT::Experimental::Browsable;

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

std::unique_ptr<RItem> RLevelIter::CreateItem()
{
   return HasItem() ? std::make_unique<RItem>(GetName(), CanHaveChilds(), CanHaveChilds() > 0 ? "sap-icon://folder-blank" : "sap-icon://document") : nullptr;
}

