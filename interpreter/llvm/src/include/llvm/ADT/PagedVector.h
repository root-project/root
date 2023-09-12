//===- llvm/ADT/PagedVector.h - 'Lazyly allocated' vectors --------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the PagedVector class.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_ADT_PAGEDVECTOR_H
#define LLVM_ADT_PAGEDVECTOR_H

#include <cassert>
#include <vector>

namespace llvm {
// A vector that allocates memory in pages.
// Order is kept, but memory is allocated only when one element of the page is
// accessed. This introduces a level of indirection, but it is useful when you
// have a sparsely initialised vector where the full size is allocated upfront
// with the default constructor and elements are initialised later, on first
// access.
//
// Notice that this does not have iterators, because if you
// have iterators it probably means you are going to touch
// all the memory in any case, so better use a std::vector in
// the first place.
template <typename T, int PAGE_SIZE = 1024 / sizeof(T)> class PagedVector {
  // The actual number of element in the vector which can be accessed.
  std::size_t Size = 0;
  // The position of the initial element of the page in the Data vector.
  // Pages are allocated contiguously in the Data vector.
  mutable std::vector<int> Lookup;
  // Actual page data. All the page elements are added to this vector on the
  // first access of any of the elements of the page. Elements default
  // constructed and elements of the page are stored contiguously. The oder of
  // the elements however depends on the order of access of the pages.
  mutable std::vector<T> Data;

public:
  // Lookup an element at position Index.
  T &operator[](std::size_t Index) const { return at(Index); }

  // Lookup an element at position i.
  // If the associated page is not filled, it will be filled with default
  // constructed elements. If the associated page is filled, return the element.
  T &at(std::size_t Index) const {
    assert(Index < Size);
    assert(Index / PAGE_SIZE < Lookup.size());
    auto &PageId = Lookup[Index / PAGE_SIZE];
    // If the range is not filled, fill it
    if (PageId == -1) {
      int OldSize = Data.size();
      PageId = OldSize / PAGE_SIZE;
      // Allocate the memory
      Data.resize(OldSize + PAGE_SIZE);
      // Fill the whole capacity with empty elements
      for (int I = 0; I < PAGE_SIZE; ++I) {
        Data[I + OldSize] = T();
      }
    }
    // Calculate the actual position in the Data vector
    // by taking the start of the page and adding the offset
    // in the page.
    std::size_t StoreIndex = Index % PAGE_SIZE + PAGE_SIZE * PageId;
    // Return the element
    assert(StoreIndex < Data.size());
    return Data[StoreIndex];
  }

  // Return the capacity of the vector. I.e. the maximum size it can be expanded
  // to with the expand method without allocating more pages.
  std::size_t capacity() const { return Lookup.size() * PAGE_SIZE; }

  // Return the size of the vector. I.e. the maximum index that can be
  // accessed, i.e. the maximum value which was used as argument of the
  // expand method.
  std::size_t size() const { return Size; }

  // Expands the vector to the given NewSize number of elements.
  // If the vector was smaller, allocates new pages as needed.
  // It should be called only with NewSize >= Size.
  void expand(std::size_t NewSize) {
    // You cannot shrink the vector, otherwise
    // one would have to invalidate contents which is expensive and
    // while giving the false hope that the resize is cheap.
    if (NewSize <= Size) {
      return;
    }
    // If the capacity is enough, just update the size and continue
    // with the currently allocated pages.
    if (NewSize <= capacity()) {
      Size = NewSize;
      return;
    }
    // The number of pages to allocate. The Remainder is calculated
    // for the case in which the NewSize is not a multiple of PAGE_SIZE.
    // In that case we need one more page.
    auto Pages = NewSize / PAGE_SIZE;
    auto Remainder = NewSize % PAGE_SIZE;
    if (Remainder) {
      Pages += 1;
    }
    assert(Pages > Lookup.size());
    // We use -1 to indicate that a page has not been allocated yet.
    // This cannot be 0, because 0 is a valid page id.
    // We use -1 instead of a separate bool to avoid wasting space.
    Lookup.resize(Pages, -1);
    Size = NewSize;
  }

  // Return true if the vector is empty
  bool empty() const { return Size == 0; }

  /// Clear the vector, i.e. clear the allocated pages, the whole page
  /// lookup index and reset the size.
  void clear() {
    Size = 0;
    Lookup.clear();
    Data.clear();
  }

  /// Return the materialised vector. This is useful if you want to iterate
  /// in an efficient way over the non default constructed elements.
  /// It's not called data() because that would be misleading, since only
  /// elements for pages which have been accessed are actually allocated.
  std::vector<T> const &materialised() const { return Data; }
};
} // namespace llvm
#endif // LLVM_ADT_PAGEDVECTOR_H
