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

ROOT::Experimental::Detail::RPagePool::RPagePool(std::size_t pageSize, std::size_t nPages)
   : fMemory(nullptr), fPageSize(pageSize), fNPages(nPages)
{
   if (nPages > 0) {
      fMemory = malloc(pageSize * nPages);
      R__ASSERT(fMemory != nullptr);
      fPages.resize(nPages);
      fReferences.resize(nPages, 0);
   }
}


ROOT::Experimental::Detail::RPagePool::~RPagePool()
{
   free(fMemory);
}


ROOT::Experimental::Detail::RPage ROOT::Experimental::Detail::RPagePool::ReservePage(RColumn* column)
{
   RPage result;
   for (std::size_t i = 0; i < fNPages; ++i) {
      if (fPages[i].IsNull()) {
         void* buffer = static_cast<unsigned char *>(fMemory) + (fPageSize * i);
         result = RPage(column->GetColumnIdSource(), buffer, fPageSize, column->GetModel().GetElementSize());
         fPages[i] = result;
         return result;
      }
   }
   /// No space left
   return result;
}


void ROOT::Experimental::Detail::RPagePool::CommitPage(const RPage& page)
{
   for (unsigned i = 0; i < fNPages; ++i) {
      if (fPages[i] == page) {
         fReferences[i] = 1;
         return;
      }
   }
   R__ASSERT(false);
}

void ROOT::Experimental::Detail::RPagePool::ReleasePage(const RPage& page)
{
   if (page.IsNull()) return;
   for (unsigned i = 0; i < fNPages; ++i) {
      if (fPages[i] == page) {
         if (--fReferences[i] == 0) {
            fPages[i] = RPage();
         }
         return;
      }
   }
   R__ASSERT(false);
}

ROOT::Experimental::Detail::RPage ROOT::Experimental::Detail::RPagePool::GetPage(RColumn* column, NTupleSize_t index)
{
   for (unsigned i = 0; i < fNPages; ++i) {
      if (fReferences[i] == 0) continue;
      if (fPages[i].GetColumnId() != column->GetColumnIdSource()) continue;
      if (!fPages[i].Contains(index)) continue;
      fReferences[i]++;
      return fPages[i];
   }
   RPage newPage = ReservePage(column);
   column->GetPageSource()->PopulatePage(column->GetHandleSource(), index, &newPage);
   CommitPage(newPage);
   return newPage;
}
