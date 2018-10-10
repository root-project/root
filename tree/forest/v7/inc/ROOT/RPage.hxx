/// \file ROOT/RPage.hxx
/// \ingroup Forest ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-09
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RPage
#define ROOT7_RPage

#include <ROOT/RTreeUtil.hxx>

#include <cstddef>

namespace ROOT {
namespace Experimental {

namespace Detail {

class RColumn;

// clang-format off
/**
\class ROOT::Experimental::Detail::RPage
\ingroup Forest
\brief A page is a fixed size slice of an column that is mapped into memory

The page provides a fixed-size opaque memory buffer for uncompressed data. It does not know how to interpret
the contents but it does now about the size (and thus the number) of the elements inside as well as the element
number range within the backing column. The memory buffer is not managed by the page but normally by the page cache.
*/
// clang-format on
class RPage {
public:
   RPage(std::size_t capacity, RColumn* column, void* buffer);
   ~RPage();

   /// The total space available in the page
   std::size_t GetCapacity();
   /// The space taken by column elements in the buffer
   std::size_t GetSize();
   TreeIndex_t GetNElements();
   TreeIndex_t GetRangeStart();
   void* GetBuffer();
   /// Return a pointer after the last element that has space for nElements new elements. If there is not enough capacity,
   /// return nullptr
   void* Reserve(std::size_t nElements);
   /// Forget all currently stored elements (size == 0) and set a new starting index.
   void Reset(TreeIndex_t rangeStart);
};

} // namespace Detail

} // namespace Experimental
} // namespace ROOT

#endif
