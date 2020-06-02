/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RBrowserData.hxx>

#include <ROOT/Browsable/RProvider.hxx>
#include <ROOT/Browsable/RLevelIter.hxx>
#include <ROOT/RLogger.hxx>

#include "TBufferJSON.h"

#include <algorithm>
#include <regex>

using namespace ROOT::Experimental;
using namespace std::string_literals;

/////////////////////////////////////////////////////////////////////
/// set top element for browsing

void RBrowserData::SetTopElement(std::shared_ptr<Browsable::RElement> elem)
{
   fTopElement = elem;

   SetWorkingDirectory("");
}

/////////////////////////////////////////////////////////////////////
/// set working directory relative to top element

void RBrowserData::SetWorkingDirectory(const std::string &strpath)
{
   auto path = DecomposePath(strpath);

   SetWorkingPath(path);
}

/////////////////////////////////////////////////////////////////////
/// set working directory relative to top element

void RBrowserData::SetWorkingPath(const Browsable::RElementPath_t &path)
{
   fWorkingPath = path;
   fWorkElement = Browsable::RElement::GetSubElement(fTopElement, path);

   ResetLastRequest();
}

/////////////////////////////////////////////////////////////////////
/// Reset all data correspondent to last request

void RBrowserData::ResetLastRequest()
{
   fLastAllChilds = false;
   fLastSortedItems.clear();
   fLastSortMethod.clear();
   fLastItems.clear();
   fLastPath.clear();
   fLastElement.reset();
}

/////////////////////////////////////////////////////////////////////////
/// Decompose path to elements
/// Returns array of names for each element in the path, first element either "/" or "."
/// If returned array empty - it is error

Browsable::RElementPath_t RBrowserData::DecomposePath(const std::string &strpath)
{
   Browsable::RElementPath_t arr;

   if (strpath.empty())
      return arr;

   std::string slash = "/";

   std::string::size_type previous = 0;
   if (strpath[0] == slash[0]) previous++;

   auto current = strpath.find(slash, previous);
   while (current != std::string::npos) {
      if (current > previous)
         arr.emplace_back(strpath.substr(previous, current - previous));
      previous = current + 1;
      current = strpath.find(slash, previous);
   }

   if (previous < strpath.length())
      arr.emplace_back(strpath.substr(previous));

   return arr;
}

/////////////////////////////////////////////////////////////////////////
/// Process browser request

bool RBrowserData::ProcessBrowserRequest(const RBrowserRequest &request, RBrowserReply &reply)
{
   if (gDebug > 0)
      printf("REQ: Do decompose path '%s'\n",request.path.c_str());

   auto path = DecomposePath(request.path);

   if ((path != fLastPath) || !fLastElement) {

      auto elem = Browsable::RElement::GetSubElement(fWorkElement, path);
      if (!elem) return false;

      ResetLastRequest();

      fLastPath = path;
      fLastElement = elem;
   }

   // when request childs, always try to make elements
   if (fLastItems.empty()) {
      auto iter = fLastElement->GetChildsIter();
      if (!iter) return false;
      int id = 0;
      fLastAllChilds = true;

      while (iter->Next() && fLastAllChilds) {
         fLastItems.emplace_back(iter->CreateItem());
         if (id++ > 10000)
            fLastAllChilds = false;
      }

      fLastSortedItems.clear();
      fLastSortMethod.clear();
   }

   // create sorted array
   if ((fLastSortedItems.size() != fLastItems.size()) || (fLastSortMethod != request.sort)) {
      fLastSortedItems.resize(fLastItems.size(), nullptr);
      int id = 0;
      if (request.sort.empty()) {
         // no sorting, just move all folders up
         for (auto &item : fLastItems)
            if (item->IsFolder())
               fLastSortedItems[id++] = item.get();
         for (auto &item : fLastItems)
            if (!item->IsFolder())
               fLastSortedItems[id++] = item.get();
      } else {
         // copy items
         for (auto &item : fLastItems)
            fLastSortedItems[id++] = item.get();

         if (request.sort != "unsorted")
            std::sort(fLastSortedItems.begin(), fLastSortedItems.end(),
                      [request](const Browsable::RItem *a, const Browsable::RItem *b) { return a->Compare(b, request.sort); });
      }
      fLastSortMethod = request.sort;
   }

   const std::regex expr(request.regex);

   int id = 0;
   for (auto &item : fLastSortedItems) {

      if (!request.regex.empty() && !item->IsFolder() && !std::regex_match(item->GetName(), expr))
         continue;

      if ((id >= request.first) && ((request.number == 0) || (id < request.first + request.number)))
         reply.nodes.emplace_back(item);

      id++;
   }

   reply.first = request.first;
   reply.nchilds = id; // total number of childs

   return true;
}

/////////////////////////////////////////////////////////////////////////
/// Process browser request, returns string with JSON of RBrowserReply data

std::string RBrowserData::ProcessRequest(const RBrowserRequest &request)
{
   RBrowserReply reply;

   reply.path = request.path;
   reply.first = 0;
   reply.nchilds = 0;

   ProcessBrowserRequest(request, reply);

   return TBufferJSON::ToJSON(&reply, TBufferJSON::kSkipTypeInfo + TBufferJSON::kNoSpaces).Data();
}

/////////////////////////////////////////////////////////////////////////
/// Returns element with path, specified as string

std::shared_ptr<Browsable::RElement> RBrowserData::GetElement(const std::string &str)
{
   auto path = DecomposePath(str);

   return Browsable::RElement::GetSubElement(fWorkElement, path);
}

/////////////////////////////////////////////////////////////////////////
/// Returns element with path, specified as Browsable::RElementPath_t

std::shared_ptr<Browsable::RElement> RBrowserData::GetElementFromTop(const Browsable::RElementPath_t &path)
{
   return Browsable::RElement::GetSubElement(fTopElement, path);
}

