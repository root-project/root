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

#include <cstdlib>
#include <utility>

ROOT::Experimental::Internal::RPageRef ROOT::Experimental::Internal::RPagePool::RegisterPage(RPage page, RKey key)
{
   std::lock_guard<std::mutex> lockGuard(fLock);
   fLookupByBuffer[page.GetBuffer()] = fEntries.size();
   auto &entry = fEntries.emplace_back(REntry{std::move(page), key, 1});
   return RPageRef(entry.fPage, this);
}

void ROOT::Experimental::Internal::RPagePool::PreloadPage(RPage page, RKey key)
{
   std::lock_guard<std::mutex> lockGuard(fLock);
   fLookupByBuffer[page.GetBuffer()] = fEntries.size();
   fEntries.emplace_back(REntry{std::move(page), key, 0});
}

void ROOT::Experimental::Internal::RPagePool::ReleasePage(const RPage &page)
{
   if (page.IsNull()) return;
   std::lock_guard<std::mutex> lockGuard(fLock);

   const auto idx = fLookupByBuffer.at(page.GetBuffer());
   const auto N = fEntries.size();

   if (--fEntries[idx].fRefCounter == 0) {
      fLookupByBuffer.erase(page.GetBuffer());

      if (idx != (N - 1)) {
         fLookupByBuffer[fEntries[N - 1].fPage.GetBuffer()] = idx;
         fEntries[idx] = std::move(fEntries[N - 1]);
      }

      fEntries.resize(N - 1);
   }
}

ROOT::Experimental::Internal::RPageRef
ROOT::Experimental::Internal::RPagePool::GetPage(RKey key, NTupleSize_t globalIndex)
{
   std::lock_guard<std::mutex> lockGuard(fLock);
   unsigned int N = fEntries.size();
   for (unsigned int i = 0; i < N; ++i) {
      if (fEntries[i].fRefCounter < 0)
         continue;
      if (fEntries[i].fKey != key)
         continue;
      if (!fEntries[i].fPage.Contains(globalIndex))
         continue;
      fEntries[i].fRefCounter++;
      return RPageRef(fEntries[i].fPage, this);
   }
   return RPageRef();
}

ROOT::Experimental::Internal::RPageRef
ROOT::Experimental::Internal::RPagePool::GetPage(RKey key, RClusterIndex clusterIndex)
{
   std::lock_guard<std::mutex> lockGuard(fLock);
   unsigned int N = fEntries.size();
   for (unsigned int i = 0; i < N; ++i) {
      if (fEntries[i].fRefCounter < 0)
         continue;
      if (fEntries[i].fKey != key)
         continue;
      if (!fEntries[i].fPage.Contains(clusterIndex))
         continue;
      fEntries[i].fRefCounter++;
      return RPageRef(fEntries[i].fPage, this);
   }
   return RPageRef();
}
