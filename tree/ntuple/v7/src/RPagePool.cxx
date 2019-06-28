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

void ROOT::Experimental::Detail::RPagePool::RegisterPage(const RPage &page)
{
   fPages.emplace_back(page);
   fReferences.emplace_back(1);

}

bool ROOT::Experimental::Detail::RPagePool::ReturnPage(const RPage& page)
{
   if (page.IsNull()) return false;

   unsigned int N = fPages.size();
   for (unsigned i = 0; i < N; ++i) {
      if (fPages[i] == page) {
         if (--fReferences[i] == 0) {
            fPages[i] = fPages[N-1];
            fReferences[i] = fReferences[N-1];
            fPages.resize(N-1);
            fReferences.resize(N-1);
            return true;
         }
         return false;
      }
   }
   R__ASSERT(false);
   return false;
}

ROOT::Experimental::Detail::RPage ROOT::Experimental::Detail::RPagePool::GetPage(
   ColumnId_t columnId, NTupleSize_t index)
{
   unsigned int N = fPages.size();
   for (unsigned int i = 0; i < N; ++i) {
      if (fReferences[i] == 0) continue;
      if (fPages[i].GetColumnId() != columnId) continue;
      if (!fPages[i].Contains(index)) continue;
      fReferences[i]++;
      return fPages[i];
   }
   return RPage();
}
