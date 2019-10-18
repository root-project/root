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

using namespace ROOT::Experimental;
using namespace ROOT::Experimental::Browsable;


RProvider::BrowseMap_t &RProvider::GetBrowseMap()
{
   static RProvider::BrowseMap_t sMap;
   return sMap;
}

RProvider::FileMap_t &RProvider::GetFileMap()
{
   static RProvider::FileMap_t sMap;
   return sMap;
}


RProvider::~RProvider()
{
   // here to remove all correspondent entries
   auto &fmap = GetFileMap();

   for (auto fiter = fmap.begin();fiter != fmap.end();) {
      //if (fiter->second == provider)
      //   fiter = fmap.erase(fiter);
      //else
         fiter++;
   }

   auto &bmap = GetBrowseMap();
   for (auto biter = bmap.begin(); biter != bmap.end();) {
      //if (biter->second == provider)
      //   biter = bmap.erase(biter);
      //else
         biter++;
   }

}


void RProvider::RegisterFile(const std::string &extension, FileFunc_t provider)
{
    auto &fmap = GetFileMap();

    if ((extension != "*") && (fmap.find(extension) != fmap.end()))
       R__ERROR_HERE("Browserv7") << "Provider for file extension  " << extension << " already exists";

    fmap.emplace(extension, provider);
}

void RProvider::RegisterBrowse(const TClass *cl, BrowseFunc_t provider)
{
    auto &bmap = GetBrowseMap();

    if (cl && (bmap.find(cl) != bmap.end()))
       R__ERROR_HERE("Browserv7") << "Browse provider for class " << cl->GetName() << " already exists";

    bmap.emplace(cl, provider);
}



//////////////////////////////////////////////////////////////////////////////////
// remove provider from all registered lists


std::shared_ptr<RElement> RProvider::OpenFile(const std::string &extension, const std::string &fullname)
{
   auto &fmap = GetFileMap();

   auto iter = fmap.find(extension);

   if (iter != fmap.end()) {
      auto res = iter->second(fullname);
      if (res) return res;
   }

   for (auto &pair : fmap)
      if ((pair.first == "*") || (pair.first == extension)) {
         auto res = pair.second(fullname);
         if (res) return res;
      }

   return nullptr;
}

std::shared_ptr<RElement> RProvider::Browse(const TClass *cl, const void *object)
{
   auto &bmap = GetBrowseMap();

   auto iter = bmap.find(cl);

   if (iter != bmap.end()) {
      auto res = iter->second(cl, object);
      if (res) return res;
   }

   for (auto &pair : bmap)
      if ((pair.first == nullptr) || (pair.first == cl)) {
         auto res = pair.second(cl, object);
         if (res) return res;
      }

   return nullptr;
}


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

   if (gDebug > 0)
      printf("REQ: Do decompose path '%s'\n",request.path.c_str());

   if (!DecomposePath(request.path, arr))
      return false;

   if (gDebug > 0) {
      printf("REQ:Try to navigate %d\n", (int) arr.size());
      for (auto & subdir : arr) printf("   %s\n", subdir.c_str());
   }

   if (!Navigate(arr))
      return false;

   auto iter = fLevels.empty() ? fItem->GetChildsIter() : fLevels.back().item->GetChildsIter();

   if (gDebug > 0)
      printf("REQ:Create iterator %p\n", iter.get());

   if (!iter) return false;

   int id = 0;

   while (iter->Next()) {

      if ((id >= request.first) && ((request.number == 0) || ((int) reply.nodes.size() < request.number))) {

         // access item
         auto item = iter->CreateBrowserItem();

         // actually error
         if (!item)
            item = std::make_unique<RBrowserItem>(iter->GetName(), -1);

         if (gDebug > 0)
            printf("REQ:    item %s icon %s\n", item->GetName().c_str(), item->GetIcon().c_str());

         reply.nodes.emplace_back(std::move(item));
      }

      id++;
   }

   if (gDebug > 0)
      printf("REQ:  Done processing cnt %d\n", id);

   reply.first = request.first;
   reply.nchilds = id; // total number of childs

   return true;
}


std::shared_ptr<RElement> RBrowsable::GetElement(const std::string &path)
{
   std::vector<std::string> arr;

   if (!DecomposePath(path, arr))
      return nullptr;

   if (arr.size() == 0)
      return fItem;

   if (!Navigate(arr))
      return nullptr;

   auto res = std::move(fLevels.back().item);

   fLevels.pop_back();

   return res;
}




