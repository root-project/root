/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RBrowsable.hxx"

#include "ROOT/RLogger.hxx"

#include "TClass.h"
#include "TBaseClass.h"
#include "TList.h"

using namespace ROOT::Experimental;


RBrowsableProvider::Map_t &RBrowsableProvider::GetMap()
{
   static RBrowsableProvider::Map_t sMap;
   return sMap;
}


void RBrowsableProvider::Register(std::shared_ptr<RBrowsableProvider> provider)
{
    auto &map = GetMap();

    if (map.find(provider->GetSupportedClass()) != map.end())
       R__ERROR_HERE("Browserv7") << "FATAL Try to setup provider for class " << provider->GetSupportedClass() << " which already exists";
    else
       map[provider->GetSupportedClass()] = provider;
}

std::shared_ptr<RBrowsableProvider> RBrowsableProvider::GetProvider(const TClass *cl, bool check_base)
{
   if (!cl) return nullptr;

   auto &map = GetMap();

   auto iter = map.find(cl);
   if (iter != map.end())
      return iter->second;

   while (check_base) {
      auto lst = const_cast<TClass *>(cl)->GetListOfBases();
      // only first parent in list of parents is used
      cl = lst && (lst->GetSize() > 0) ? ((TBaseClass *) lst->First())->GetClassPointer() : nullptr;

      if (!cl) break;

      iter = map.find(cl);
      if (iter != map.end())
         return iter->second;
   }

   return nullptr;
}

/////////////////////////////////////////////////////////////////////
/// Find item with specified name
/// Default implementation, should work for all


bool RBrowsableLevelIter::Find(const std::string &name)
{
   if (!Reset()) return false;

   while (Next()) {
      if (GetName() == name)
         return true;
   }

   return false;
}


/////////////////////////////////////////////////////////////////////
/// Navigate to specified path

bool RBrowsable::Navigate(const std::vector<std::string> &paths)
{
   if (!fItem) return false;

   // TODO: reuse existing items if any

   fLevels.clear();

   auto *curr = fItem.get(); // use pointer instead of

   bool find = true;

   for (auto &subdir : paths) {

      fLevels.emplace_back(subdir);

      auto &level = fLevels.back();

      level.iter = curr->GetChildsIter();
      if (!level.iter || !level.iter->Find(subdir)) {
         find = false;
         break;
      }

      level.item = level.iter->GetElement();
      if (!level.item) {
         find = false;
         break;
      }

      curr = level.item.get();
   }

   if (!find) fLevels.clear();

   return find;
}


bool RBrowsable::DecomposePath(const std::string &path, std::vector<std::string> &arr)
{
   arr.clear();

   std::string slash = "/";

   if (path.empty() || (path == slash))
      return true;

   std::size_t previous = 0;
   if (path[0] == slash[0]) previous++;
   std::size_t current = path.find(slash, previous);
   while (current != std::string::npos) {
      if (current > previous)
         arr.emplace_back(path.substr(previous, current - previous));
      previous = current + 1;
      current = path.find(slash, previous);
   }

   if (previous < path.length())
      arr.emplace_back(path.substr(previous));
   return true;
}


bool RBrowsable::ProcessRequest(const RBrowserRequest &request, RBrowserReplyNew &reply)
{
   reply.path = request.path;
   reply.first = 0;
   reply.nchilds = 0;
   reply.nodes.clear();

   std::vector<std::string> arr;

   printf("REQ: Do decompose path '%s'\n",request.path.c_str());

   if (!DecomposePath(request.path, arr))
      return false;

   printf("REQ:Try to navigate %d\n", (int) arr.size());
   for (auto & subdir : arr) printf("   %s\n", subdir.c_str());

   if (!Navigate(arr))
      return false;

   auto iter = fLevels.empty() ? fItem->GetChildsIter() : fLevels.back().item->GetChildsIter();

   printf("REQ:Create iterator %p\n", iter.get());

   if (!iter) return false;

   int id = 0;

   while (iter->Next()) {

      if ((id >= request.first) && ((request.number == 0) || ((int) reply.nodes.size() < request.number))) {
         // access element
         auto elem = iter->GetElement();

         std::unique_ptr<RBrowserItem> item;

         if (elem)
            // produce item object, can include extra information
            item = elem->CreateBrowserItem();

         if (!item)
            item = std::make_unique<RBrowserItem>(iter->GetName(), -1);

         printf("REQ:    item %s\n", item->GetName().c_str());

         reply.nodes.emplace_back(std::move(item));
      }

      id++;
   }

   printf("REQ:  Done processing cnt %d\n", id);

   reply.first = request.first;
   reply.nchilds = id; // total number of childs

   return true;
}


std::unique_ptr<RBrowsableElement> RBrowsable::GetElement(const std::string &path)
{
   std::vector<std::string> arr;

   if (!DecomposePath(path, arr))
      return nullptr;

   if (arr.size() == 0) {
      R__ERROR_HERE("Browserv7") << "Cannot access top-level element via GetElement method";
      return nullptr;
   }

   if (!Navigate(arr))
      return nullptr;

   auto res = std::move(fLevels.back().item);

   fLevels.pop_back();

   return res;
}




