/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/Browsable/RElement.hxx>

#include <ROOT/Browsable/RLevelIter.hxx>
#include <ROOT/RLogger.hxx>
#include "TBufferJSON.h"

using namespace ROOT::Experimental::Browsable;
using namespace std::string_literals;

ROOT::Experimental::RLogChannel &ROOT::Experimental::BrowsableLog() {
   static RLogChannel sLog("ROOT.Browsable");
   return sLog;
}


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
   if (lkind == "json") return kJson;
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

/////////////////////////////////////////////////////////////////////
/// Returns string content like text file content or json representation

std::string RElement::GetContent(const std::string &kind)
{
   if (GetContentKind(kind) == kJson) {
      auto obj = GetObject();
      if (obj)
         return TBufferJSON::ConvertToJSON(obj->GetObject(), obj->GetClass()).Data();
   }

   return ""s;
}

/////////////////////////////////////////////////////////////////////
/// Compare two paths,
/// Returns number of elements matches in both paths

int RElement::ComparePaths(const RElementPath_t &path1, const RElementPath_t &path2)
{
   int sz = path1.size();
   if (sz > (int) path2.size()) sz = path2.size();

   for (int n = 0; n < sz; ++n)
      if (path1[n] != path2[n])
         return n;

   return sz;
}

/////////////////////////////////////////////////////////////////////
/// Converts element path back to string

std::string RElement::GetPathAsString(const RElementPath_t &path)
{
   std::string res;
   for (auto &elem : path) {
      res.append("/");
      res.append(elem);
   }

   return res;
}

