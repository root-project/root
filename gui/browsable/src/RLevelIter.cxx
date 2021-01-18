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
/// Returns number of childs in current entry
/// 0 if there is no childs
/// >0 if there are really childs
/// -1 if entry may have child elements

int RLevelIter::GetNumItemChilds() const
{
   return 0;
}

/////////////////////////////////////////////////////////////////////
/// Find item with specified name
/// Default implementation, should work for all

bool RLevelIter::Find(const std::string &name)
{
   while (Next()) {
      if (GetItemName() == name)
         return true;
   }

   return false;
}


/////////////////////////////////////////////////////////////////////
/// Create generic description item for RBrowser

std::unique_ptr<RItem> RLevelIter::CreateItem()
{
   auto nchilds = GetNumItemChilds();

   return std::make_unique<RItem>(GetItemName(), nchilds, nchilds != 0 ? "sap-icon://folder-blank" : "sap-icon://document");
}

