/// \file RPagePool.cxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-04

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

ROOT::Internal::RPagePool::REntry &
ROOT::Internal::RPagePool::AddPage(RPage page, const RKey &key, std::int64_t initialRefCounter)
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

ROOT::Internal::RPageRef ROOT::Internal::RPagePool::RegisterPage(RPage page, RKey key)
{
   std::lock_guard<std::mutex> lockGuard(fLock);
   return RPageRef(AddPage(std::move(page), key, 1).fPage, this);
}

void ROOT::Internal::RPagePool::PreloadPage(RPage page, RKey key)
{
   std::lock_guard<std::mutex> lockGuard(fLock);
   const auto &entry = AddPage(std::move(page), key, 0);
   if (entry.fRefCounter == 0)
      fUnusedPages[entry.fPage.GetClusterInfo().GetId()].emplace(entry.fPage.GetBuffer());
}

void ROOT::Internal::RPagePool::ErasePage(std::size_t entryIdx, decltype(fLookupByBuffer)::iterator lookupByBufferItr)
{
   fLookupByBuffer.erase(lookupByBufferItr);

   auto itrPageSet = fLookupByKey.find(fEntries[entryIdx].fKey);
   assert(itrPageSet != fLookupByKey.end());
   itrPageSet->second.erase(RPagePosition(fEntries[entryIdx].fPage));
   if (itrPageSet->second.empty())
      fLookupByKey.erase(itrPageSet);

   const auto N = fEntries.size();
   assert(entryIdx < N);
   if (entryIdx != (N - 1)) {
      fLookupByBuffer[fEntries[N - 1].fPage.GetBuffer()] = entryIdx;
      itrPageSet = fLookupByKey.find(fEntries[N - 1].fKey);
      assert(itrPageSet != fLookupByKey.end());
      auto itrEntryIdx = itrPageSet->second.find(RPagePosition(fEntries[N - 1].fPage));
      assert(itrEntryIdx != itrPageSet->second.end());
      itrEntryIdx->second = entryIdx;
      fEntries[entryIdx] = std::move(fEntries[N - 1]);
   }

   fEntries.resize(N - 1);
}

void ROOT::Internal::RPagePool::ReleasePage(const RPage &page)
{
   if (page.IsNull()) return;
   std::lock_guard<std::mutex> lockGuard(fLock);

   auto itrLookup = fLookupByBuffer.find(page.GetBuffer());
   assert(itrLookup != fLookupByBuffer.end());
   const auto idx = itrLookup->second;

   assert(fEntries[idx].fRefCounter >= 1);
   if (--fEntries[idx].fRefCounter == 0) {
      ErasePage(idx, itrLookup);
   }
}

void ROOT::Internal::RPagePool::RemoveFromUnusedPages(const RPage &page)
{
   auto itr = fUnusedPages.find(page.GetClusterInfo().GetId());
   assert(itr != fUnusedPages.end());
   itr->second.erase(page.GetBuffer());
   if (itr->second.empty())
      fUnusedPages.erase(itr);
}

ROOT::Internal::RPageRef ROOT::Internal::RPagePool::GetPage(RKey key, ROOT::NTupleSize_t globalIndex)
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
      if (fEntries[itrEntryIdx->second].fRefCounter == 0)
         RemoveFromUnusedPages(fEntries[itrEntryIdx->second].fPage);
      fEntries[itrEntryIdx->second].fRefCounter++;
      return RPageRef(fEntries[itrEntryIdx->second].fPage, this);
   }
   return RPageRef();
}

ROOT::Internal::RPageRef ROOT::Internal::RPagePool::GetPage(RKey key, RNTupleLocalIndex localIndex)
{
   std::lock_guard<std::mutex> lockGuard(fLock);
   auto itrPageSet = fLookupByKey.find(key);
   if (itrPageSet == fLookupByKey.end())
      return RPageRef();
   assert(!itrPageSet->second.empty());

   auto itrEntryIdx = itrPageSet->second.upper_bound(RPagePosition(localIndex));
   if (itrEntryIdx == itrPageSet->second.begin())
      return RPageRef();

   --itrEntryIdx;
   if (fEntries[itrEntryIdx->second].fPage.Contains(localIndex)) {
      if (fEntries[itrEntryIdx->second].fRefCounter == 0)
         RemoveFromUnusedPages(fEntries[itrEntryIdx->second].fPage);
      fEntries[itrEntryIdx->second].fRefCounter++;
      return RPageRef(fEntries[itrEntryIdx->second].fPage, this);
   }
   return RPageRef();
}

void ROOT::Internal::RPagePool::Evict(ROOT::DescriptorId_t clusterId)
{
   std::lock_guard<std::mutex> lockGuard(fLock);
   auto itr = fUnusedPages.find(clusterId);
   if (itr == fUnusedPages.end())
      return;

   for (auto pageBuffer : itr->second) {
      const auto itrLookupByBuffer = fLookupByBuffer.find(pageBuffer);
      assert(itrLookupByBuffer != fLookupByBuffer.end());
      const auto entryIdx = itrLookupByBuffer->second;
      assert(fEntries[entryIdx].fRefCounter == 0);
      ErasePage(entryIdx, itrLookupByBuffer);
   }

   fUnusedPages.erase(itr);
}
