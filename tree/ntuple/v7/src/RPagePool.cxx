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

ROOT::Experimental::Internal::RPageRef ROOT::Experimental::Internal::RPagePool::RegisterPage(RPage page)
{
   std::lock_guard<std::mutex> lockGuard(fLock);
   fPages.emplace_back(std::move(page));
   fReferences.emplace_back(1);
   return RPageRef(page, this);
}

void ROOT::Experimental::Internal::RPagePool::PreloadPage(RPage page)
{
   std::lock_guard<std::mutex> lockGuard(fLock);
   fPages.emplace_back(std::move(page));
   fReferences.emplace_back(0);
}

void ROOT::Experimental::Internal::RPagePool::ReleasePage(const RPage &page)
{
   if (page.IsNull()) return;
   std::lock_guard<std::mutex> lockGuard(fLock);

   unsigned int N = fPages.size();
   for (unsigned i = 0; i < N; ++i) {
      if (fPages[i] != page) continue;

      if (--fReferences[i] == 0) {
         fPages[i] = std::move(fPages[N - 1]);
         fReferences[i] = fReferences[N - 1];
         fPages.resize(N-1);
         fReferences.resize(N - 1);
      }
      return;
   }
   R__ASSERT(false);
}

ROOT::Experimental::Internal::RPageRef
ROOT::Experimental::Internal::RPagePool::GetPage(ColumnId_t columnId, NTupleSize_t globalIndex)
{
   std::lock_guard<std::mutex> lockGuard(fLock);
   unsigned int N = fPages.size();
   for (unsigned int i = 0; i < N; ++i) {
      if (fReferences[i] < 0) continue;
      if (fPages[i].GetColumnId() != columnId) continue;
      if (!fPages[i].Contains(globalIndex)) continue;
      fReferences[i]++;
      return RPageRef(fPages[i], this);
   }
   return RPageRef();
}

ROOT::Experimental::Internal::RPageRef
ROOT::Experimental::Internal::RPagePool::GetPage(ColumnId_t columnId, RClusterIndex clusterIndex)
{
   std::lock_guard<std::mutex> lockGuard(fLock);
   unsigned int N = fPages.size();
   for (unsigned int i = 0; i < N; ++i) {
      if (fReferences[i] < 0) continue;
      if (fPages[i].GetColumnId() != columnId) continue;
      if (!fPages[i].Contains(clusterIndex)) continue;
      fReferences[i]++;
      return RPageRef(fPages[i], this);
   }
   return RPageRef();
}
