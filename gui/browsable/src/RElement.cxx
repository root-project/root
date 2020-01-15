/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/Browsable/RElement.hxx>

#include <ROOT/Browsable/RLevelIter.hxx>

using namespace ROOT::Experimental::Browsable;
using namespace std::string_literals;


/////////////////////////////////////////////////////////////////////
/// Returns child iterator (if any)

std::unique_ptr<RLevelIter> RElement::GetChildsIter()
{
   return nullptr;
}

/////////////////////////////////////////////////////////////////////
/// Find item with specified name
/// Default implementation, should work for all

RElement::EContentKind RElement::GetContentKind(const std::string &kind)
{
    std::string lkind = kind;
    std::transform(lkind.begin(), lkind.end(), lkind.begin(), ::tolower);

   if (lkind == "text") return kText;
   if ((lkind == "image") || (lkind == "image64")) return kImage;
   if (lkind == "png") return kPng;
   if ((lkind == "jpg") || (lkind == "jpeg")) return kJpeg;
   if (lkind == "filename") return kFileName;
   return kNone;
}

/////////////////////////////////////////////////////////////////////
/// Returns sub element

std::shared_ptr<RElement> RElement::GetSubElement(std::shared_ptr<RElement> &elem, const RElementPath_t &path)
{
   auto curr = elem;

   for (auto &itemname : path) {
      if (!curr)
         return nullptr;

      auto iter = curr->GetChildsIter();
      if (!iter || !iter->Find(itemname))
         return nullptr;

      curr = iter->GetElement();
   }

   return curr;
}
