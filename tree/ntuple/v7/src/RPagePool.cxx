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

ROOT::Experimental::Internal::RPagePool::REntry &
ROOT::Experimental::Internal::RPagePool::AddPage(RPage page, const RKey &key, std::int64_t initialRefCounter)
{
   assert(fLookupByBuffer.count(page.GetBuffer()) == 0);
   const auto index = fEntries.size();
   auto &entry = fEntries.emplace_back(REntry{std::move(page), key, initialRefCounter});
   fLookupByBuffer[page.GetBuffer()] = index;
   fLookupByKey[key].emplace_back(index);
   return entry;
}

ROOT::Experimental::Internal::RPageRef ROOT::Experimental::Internal::RPagePool::RegisterPage(RPage page, RKey key)
{
   std::lock_guard<std::mutex> lockGuard(fLock);
   return RPageRef(AddPage(std::move(page), key, 1).fPage, this);
}

void ROOT::Experimental::Internal::RPagePool::PreloadPage(RPage page, RKey key)
{
   std::lock_guard<std::mutex> lockGuard(fLock);
   AddPage(std::move(page), key, 0);
}

void ROOT::Experimental::Internal::RPagePool::ReleasePage(const RPage &page)
{
   if (page.IsNull()) return;
   std::lock_guard<std::mutex> lockGuard(fLock);

   auto itrLookup = fLookupByBuffer.find(page.GetBuffer());
   assert(itrLookup != fLookupByBuffer.end());
   const auto idx = itrLookup->second;
   const auto N = fEntries.size();

   assert(fEntries[idx].fRefCounter >= 1);
   if (--fEntries[idx].fRefCounter == 0) {
      fLookupByBuffer.erase(itrLookup);

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
