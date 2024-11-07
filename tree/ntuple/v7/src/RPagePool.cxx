/// \file RPagePool.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-04
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RPagePool.hxx>
#include <ROOT/RColumn.hxx>

#include <TError.h>

#include <algorithm>
#include <cstdlib>
#include <utility>

void ROOT::Experimental::Internal::RPagePool::AddPage(const RPage &page, const RKey &key, std::size_t index)
{
   assert(fLookupByBuffer.count(page.GetBuffer()) == 0);
   fLookupByBuffer[page.GetBuffer()] = index;
   auto itr = fLookupByKey.find(key);
   if (itr == fLookupByKey.end()) {
      fLookupByKey.emplace(key, std::vector<size_t>({index}));
   } else {
      itr->second.emplace_back(index);
   }
}

ROOT::Experimental::Internal::RPageRef ROOT::Experimental::Internal::RPagePool::RegisterPage(RPage page, RKey key)
{
   std::lock_guard<std::mutex> lockGuard(fLock);
   auto index = fEntries.size();
   auto &entry = fEntries.emplace_back(REntry{std::move(page), key, 1});
   AddPage(page, key, index);
   return RPageRef(entry.fPage, this);
}

void ROOT::Experimental::Internal::RPagePool::PreloadPage(RPage page, RKey key)
{
   std::lock_guard<std::mutex> lockGuard(fLock);
   auto index = fEntries.size();
   fEntries.emplace_back(REntry{std::move(page), key, 0});
   AddPage(page, key, index);
}

void ROOT::Experimental::Internal::RPagePool::ReleasePage(const RPage &page)
{
   if (page.IsNull()) return;
   std::lock_guard<std::mutex> lockGuard(fLock);

   const auto idx = fLookupByBuffer.at(page.GetBuffer());
   const auto N = fEntries.size();

   if (--fEntries[idx].fRefCounter == 0) {
      fLookupByBuffer.erase(page.GetBuffer());

      auto itrPageSet = fLookupByKey.find(fEntries[idx].fKey);
      assert(itrPageSet != fLookupByKey.end());
      itrPageSet->second.erase(std::find(itrPageSet->second.begin(), itrPageSet->second.end(), idx));
      if (itrPageSet->second.empty())
         fLookupByKey.erase(itrPageSet);

      if (idx != (N - 1)) {
         fLookupByBuffer[fEntries[N - 1].fPage.GetBuffer()] = idx;
         itrPageSet = fLookupByKey.find(fEntries[N - 1].fKey);
         assert(itrPageSet != fLookupByKey.end());
         auto itrPageIdx = std::find(itrPageSet->second.begin(), itrPageSet->second.end(), N - 1);
         assert(itrPageIdx != itrPageSet->second.end());
         *itrPageIdx = idx;
         fEntries[idx] = std::move(fEntries[N - 1]);
      }

      fEntries.resize(N - 1);
   }
}

ROOT::Experimental::Internal::RPageRef
ROOT::Experimental::Internal::RPagePool::GetPage(RKey key, NTupleSize_t globalIndex)
{
   std::lock_guard<std::mutex> lockGuard(fLock);
   auto itrPageSet = fLookupByKey.find(key);
   if (itrPageSet == fLookupByKey.end())
      return RPageRef();

   for (auto idx : itrPageSet->second) {
      if (fEntries[idx].fPage.Contains(globalIndex)) {
         fEntries[idx].fRefCounter++;
         return RPageRef(fEntries[idx].fPage, this);
      }
   }
   return RPageRef();
}

ROOT::Experimental::Internal::RPageRef
ROOT::Experimental::Internal::RPagePool::GetPage(RKey key, RClusterIndex clusterIndex)
{
   std::lock_guard<std::mutex> lockGuard(fLock);
   auto itrPageSet = fLookupByKey.find(key);
   if (itrPageSet == fLookupByKey.end())
      return RPageRef();

   for (auto idx : itrPageSet->second) {
      if (fEntries[idx].fPage.Contains(clusterIndex)) {
         fEntries[idx].fRefCounter++;
         return RPageRef(fEntries[idx].fPage, this);
      }
   }
   return RPageRef();
}
