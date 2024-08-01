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

ROOT::Experimental::Internal::RPagePool::~RPagePool()
{
   for (auto &p : fPages)
      fPageAllocator->DeletePage(p);
}

void ROOT::Experimental::Internal::RPagePool::RegisterPage(RPage &page)
{
   std::lock_guard<std::mutex> lockGuard(fLock);
   fPages.emplace_back(page);
   fReferences.emplace_back(1);
}

void ROOT::Experimental::Internal::RPagePool::PreloadPage(RPage &page)
{
   std::lock_guard<std::mutex> lockGuard(fLock);
   fPages.emplace_back(page);
   fReferences.emplace_back(0);
}

void ROOT::Experimental::Internal::RPagePool::ReturnPage(RPage &page)
{
   if (page.IsNull()) return;
   std::lock_guard<std::mutex> lockGuard(fLock);

   unsigned int N = fPages.size();
   for (unsigned i = 0; i < N; ++i) {
      if (fPages[i] != page) continue;

      if (--fReferences[i] == 0) {
         fPageAllocator->DeletePage(fPages[i]);
         fPages[i] = fPages[N-1];
         fReferences[i] = fReferences[N - 1];
         fPages.resize(N-1);
         fReferences.resize(N - 1);
      }
      return;
   }
   R__ASSERT(false);
}

ROOT::Experimental::Internal::RPage
ROOT::Experimental::Internal::RPagePool::GetPage(ColumnId_t columnId, NTupleSize_t globalIndex)
{
   std::lock_guard<std::mutex> lockGuard(fLock);
   unsigned int N = fPages.size();
   for (unsigned int i = 0; i < N; ++i) {
      if (fReferences[i] < 0) continue;
      if (fPages[i].GetColumnId() != columnId) continue;
      if (!fPages[i].Contains(globalIndex)) continue;
      fReferences[i]++;
      return fPages[i];
   }
   return RPage();
}

ROOT::Experimental::Internal::RPage
ROOT::Experimental::Internal::RPagePool::GetPage(ColumnId_t columnId, RClusterIndex clusterIndex)
{
   std::lock_guard<std::mutex> lockGuard(fLock);
   unsigned int N = fPages.size();
   for (unsigned int i = 0; i < N; ++i) {
      if (fReferences[i] < 0) continue;
      if (fPages[i].GetColumnId() != columnId) continue;
      if (!fPages[i].Contains(clusterIndex)) continue;
      fReferences[i]++;
      return fPages[i];
   }
   return RPage();
}
