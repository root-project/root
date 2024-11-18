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

   const auto entryIndex = fEntries.size();

   auto itrPageSet = fLookupByKey.find(key);
   if (itrPageSet != fLookupByKey.end()) {
      auto [itrEntryIdx, isNew] = itrPageSet->second.emplace(RPagePosition(page), entryIndex);
      if (!isNew) {
         assert(itrEntryIdx->second < fEntries.size());
         // We require that pages cover pairwise distinct element ranges of the column
         assert(fEntries[itrEntryIdx->second].fPage.GetGlobalRangeLast() == page.GetGlobalRangeLast());
         fEntries[itrEntryIdx->second].fRefCounter += initialRefCounter;
         return fEntries[itrEntryIdx->second];
      }
   } else {
      fLookupByKey.emplace(key, std::map<RPagePosition, std::size_t>{{RPagePosition(page), entryIndex}});
   }

   fLookupByBuffer[page.GetBuffer()] = entryIndex;

   return fEntries.emplace_back(REntry{std::move(page), key, initialRefCounter});
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
   assert(idx < N);

   assert(fEntries[idx].fRefCounter >= 1);
   if (--fEntries[idx].fRefCounter == 0) {
      fLookupByBuffer.erase(itrLookup);

      auto itrPageSet = fLookupByKey.find(fEntries[idx].fKey);
      assert(itrPageSet != fLookupByKey.end());
      itrPageSet->second.erase(RPagePosition(page));
      if (itrPageSet->second.empty())
         fLookupByKey.erase(itrPageSet);

      if (idx != (N - 1)) {
         fLookupByBuffer[fEntries[N - 1].fPage.GetBuffer()] = idx;
         itrPageSet = fLookupByKey.find(fEntries[N - 1].fKey);
         assert(itrPageSet != fLookupByKey.end());
         auto itrEntryIdx = itrPageSet->second.find(RPagePosition(fEntries[N - 1].fPage));
         assert(itrEntryIdx != itrPageSet->second.end());
         itrEntryIdx->second = idx;
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
   assert(!itrPageSet->second.empty());

   auto itrEntryIdx = itrPageSet->second.upper_bound(RPagePosition(globalIndex));
   if (itrEntryIdx == itrPageSet->second.begin())
      return RPageRef();

   --itrEntryIdx;
   if (fEntries[itrEntryIdx->second].fPage.Contains(globalIndex)) {
      fEntries[itrEntryIdx->second].fRefCounter++;
      return RPageRef(fEntries[itrEntryIdx->second].fPage, this);
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
   assert(!itrPageSet->second.empty());

   auto itrEntryIdx = itrPageSet->second.upper_bound(RPagePosition(clusterIndex));
   if (itrEntryIdx == itrPageSet->second.begin())
      return RPageRef();

   --itrEntryIdx;
   if (fEntries[itrEntryIdx->second].fPage.Contains(clusterIndex)) {
      fEntries[itrEntryIdx->second].fRefCounter++;
      return RPageRef(fEntries[itrEntryIdx->second].fPage, this);
   }
   return RPageRef();
}
