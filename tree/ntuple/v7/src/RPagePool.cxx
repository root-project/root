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
   fLookupByBuffer[page.GetBuffer()] = fPages.size();
   fPages.emplace_back(std::move(page));
   RPageInfo pageInfo;
   pageInfo.fKey = key;
   pageInfo.fRefCounter = 1;
   fPageInfos.emplace_back(pageInfo);
   return RPageRef(page, this);
}

void ROOT::Experimental::Internal::RPagePool::PreloadPage(RPage page, RKey key)
{
   std::lock_guard<std::mutex> lockGuard(fLock);
   fLookupByBuffer[page.GetBuffer()] = fPages.size();
   fPages.emplace_back(std::move(page));
   RPageInfo pageInfo;
   pageInfo.fKey = key;
   pageInfo.fRefCounter = 0;
   fPageInfos.emplace_back(pageInfo);
}

void ROOT::Experimental::Internal::RPagePool::ReleasePage(const RPage &page)
{
   if (page.IsNull()) return;
   std::lock_guard<std::mutex> lockGuard(fLock);

   const auto idx = fLookupByBuffer.at(page.GetBuffer());
   const auto N = fPages.size();

   if (--fPageInfos[idx].fRefCounter == 0) {
      fLookupByBuffer.erase(page.GetBuffer());

      if (idx != (N - 1)) {
         fLookupByBuffer[fPages[N - 1].GetBuffer()] = idx;
         fPages[idx] = std::move(fPages[N - 1]);
         fPageInfos[idx] = fPageInfos[N - 1];
      }

      fPages.resize(N - 1);
      fPageInfos.resize(N - 1);
   }
}

ROOT::Experimental::Internal::RPageRef
ROOT::Experimental::Internal::RPagePool::GetPage(RKey key, NTupleSize_t globalIndex)
{
   std::lock_guard<std::mutex> lockGuard(fLock);
   unsigned int N = fPages.size();
   for (unsigned int i = 0; i < N; ++i) {
      if (fPageInfos[i].fRefCounter < 0)
         continue;
      if (fPageInfos[i].fKey != key)
         continue;
      if (!fPages[i].Contains(globalIndex)) continue;
      fPageInfos[i].fRefCounter++;
      return RPageRef(fPages[i], this);
   }
   return RPageRef();
}

ROOT::Experimental::Internal::RPageRef
ROOT::Experimental::Internal::RPagePool::GetPage(RKey key, RClusterIndex clusterIndex)
{
   std::lock_guard<std::mutex> lockGuard(fLock);
   unsigned int N = fPages.size();
   for (unsigned int i = 0; i < N; ++i) {
      if (fPageInfos[i].fRefCounter < 0)
         continue;
      if (fPageInfos[i].fKey != key)
         continue;
      if (!fPages[i].Contains(clusterIndex)) continue;
      fPageInfos[i].fRefCounter++;
      return RPageRef(fPages[i], this);
   }
   return RPageRef();
}
