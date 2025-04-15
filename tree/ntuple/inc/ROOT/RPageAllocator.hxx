/// \file ROOT/RPageAllocator.hxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2019-06-25

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RPageAllocator
#define ROOT_RPageAllocator

#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RPage.hxx>

#include <cstddef>
#include <functional>

namespace ROOT {
namespace Internal {

// clang-format off
/**
\class ROOT::Internal::RPageAllocator
\ingroup NTuple
\brief Abstract interface to allocate and release pages

The page allocator acquires and releases memory for pages.  It does not load the page data, the returned pages
are empty but guaranteed to have enough contiguous space for the given number of elements.
The page allocator must be thread-safe.
*/
// clang-format on
class RPageAllocator {
   friend class RPage;

protected:
   /// Releases the memory pointed to by page and resets the page's information. Note that the memory of the
   /// zero page must not be deleted. Called by the RPage destructor.
   virtual void DeletePage(RPage &page) = 0;

public:
   virtual ~RPageAllocator() = default;

   /// Reserves memory large enough to hold nElements of the given size. The page is immediately tagged with
   /// a column id. Returns a default constructed page on out-of-memory condition.
   virtual RPage NewPage(std::size_t elementSize, std::size_t nElements) = 0;
};

// clang-format off
/**
\class ROOT::Internal::RPageAllocatorHeap
\ingroup NTuple
\brief Uses standard C++ memory allocation for the column data pages
*/
// clang-format on
class RPageAllocatorHeap : public RPageAllocator {
protected:
   void DeletePage(RPage &page) final;

public:
   RPage NewPage(std::size_t elementSize, std::size_t nElements) final;
};

} // namespace Internal
} // namespace ROOT

#endif
