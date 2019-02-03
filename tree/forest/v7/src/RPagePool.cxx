/// \file RPagePool.cxx
/// \ingroup Forest ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-04
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
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
   fMemory = malloc(pageSize * nPages);
   R__ASSERT(fMemory != nullptr);
   fPages.resize(nPages);
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
         result = RPage(buffer, fPageSize, column->GetModel().GetElementSize());
         fPages[i] = result;
         return result;
      }
   }
   /// No space left
   return result;
}
