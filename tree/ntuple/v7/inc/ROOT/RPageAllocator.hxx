/// \file ROOT/RPageAllocator.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2019-06-25
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RPageAllocator
#define ROOT7_RPageAllocator

#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RPage.hxx>

#include <cstddef>

namespace ROOT {
namespace Experimental {
namespace Detail {

// clang-format off
/**
\class ROOT::Experimental::Detail::RPageAllocator
\ingroup NTuple
\brief Basic memory management of column pages: reserve and release

The page allocator acquires and releases memory for pages.  It does not populate the pages, the returned pages
are empty but guaranteed to have enough contiguous space for the given number of elements.  While a common
concrete implementation uses the heap, other implementations are possible, e.g. using arenas or mmap().
*/
// clang-format on
class RPageAllocator {
public:
   /// Reserves memory large enough to hold nElements of the given size; the page is immediately tagged with a column id
   virtual RPage AllocatePage(ColumnId_t columnId, std::size_t elementSize, std::size_t nElements) = 0;
   /// Frees the memory pointed to by page and resets the page's information
   virtual void ReleasePage(RPage &page) = 0;
};


// clang-format off
/**
\class ROOT::Experimental::Detail::RPageAllocatorHeap
\ingroup NTuple
\brief Uses standard C++ memory allocation for the pages
*/
// clang-format on
class RPageAllocatorHeap : public RPageAllocator {
public:
   RPage AllocatePage(ColumnId_t columnId, std::size_t elementSize, std::size_t nElements) final;
   void ReleasePage(RPage& page) final;
};

} // namespace Detail
} // namespace Experimental
} // namespace ROOT

#endif
