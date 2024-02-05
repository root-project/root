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

#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Allocator.h"
#include <cassert>
#include <vector>

namespace llvm {
/// A vector that allocates memory in pages.
///
/// Order is kept, but memory is allocated only when one element of the page is
/// accessed. This introduces a level of indirection, but it is useful when you
/// have a sparsely initialised vector where the full size is allocated upfront.
///
/// As a side effect the elements are initialised later than in a normal vector.
/// On the first access to one of the elements of a given page all, the elements
/// of the page are initialised. This also means that the elements of the page
/// are initialised beyond the size of the vector.
///
/// Similarly on destruction the elements are destroyed only when the page is
/// not needed anymore, delaying invoking the destructor of the elements.
///
/// Notice that this does not have iterators, because if you have iterators it
/// probably means you are going to touch all the memory in any case, so better
/// use a std::vector in the first place.
template <typename T, size_t PageSize = 1024 / sizeof(T)> class PagedVector {
  static_assert(PageSize > 1, "PageSize must be greater than 0. Most likely "
                              "you want it to be greater than 16.");
  // The actual number of element in the vector which can be accessed.
  size_t Size = 0;

  // The position of the initial element of the page in the Data vector.
  // Pages are allocated contiguously in the Data vector.
  mutable SmallVector<T *, 0> PageToDataPtrs;
  // Actual page data. All the page elements are added to this vector on the
  // first access of any of the elements of the page. Elements default
  // constructed and elements of the page are stored contiguously. The order of
  // the elements however depends on the order of access of the pages.
  PointerIntPair<BumpPtrAllocator *, 1, bool> Allocator;

  constexpr static T *InvalidPage = nullptr;

public:
  using value_type = T;

  /// Default constructor. We build our own allocator and mark it as such with
  /// `true` in the second pair element.
  PagedVector() : Allocator(new BumpPtrAllocator, true) {}
  PagedVector(BumpPtrAllocator *A) : Allocator(A, false) {
    assert(A != nullptr && "Allocator cannot be null");
  }

  ~PagedVector() {
    clear();
    // If we own the allocator, delete it.
    if (Allocator.getInt())
      delete Allocator.getPointer();
  }

  /// Look up an element at position `Index`.
  /// If the associated page is not filled, it will be filled with default
  /// constructed elements. If the associated page is filled, return the
  /// element.
  T &operator[](size_t Index) const {
    assert(Index < Size);
    assert(Index / PageSize < PageToDataPtrs.size());
    T *&PagePtr = PageToDataPtrs[Index / PageSize];
    // If the page was not yet allocated, allocate it.
    if (PagePtr == InvalidPage) {
      T *NewPagePtr = Allocator.getPointer()->template Allocate<T>(PageSize);
      // We need to invoke the default constructor on all the elements of the
      // page.
      for (size_t I = 0; I < PageSize; ++I)
        new (NewPagePtr + I) T();

      PagePtr = NewPagePtr;
    }
    // Dereference the element in the page.
    return *((Index % PageSize) + PagePtr);
  }

  /// Return the capacity of the vector. I.e. the maximum size it can be
  /// expanded to with the resize method without allocating more pages.
  [[nodiscard]] size_t capacity() const {
    return PageToDataPtrs.size() * PageSize;
  }

  /// Return the size of the vector. I.e. the maximum index that can be
  /// accessed, i.e. the maximum value which was used as argument of the
  /// resize method.
  [[nodiscard]] size_t size() const { return Size; }

  /// Resize the vector. Notice that the constructor of the elements will not
  /// be invoked until an element of a given page is accessed, at which point
  /// all the elements of the page will be constructed.
  ///
  /// If the new size is smaller than the current size, the elements of the
  /// pages that are not needed anymore will be destroyed, however, elements of
  /// the last page will not be destroyed.
  ///
  /// For these reason the usage of this vector is discouraged if you rely
  /// on the construction / destructor of the elements to be invoked.
  void resize(size_t NewSize) {
    if (NewSize == 0) {
      clear();
      return;
    }
    // Handle shrink case: destroy the elements in the pages that are not
    // needed anymore and deallocate the pages.
    //
    // On the other hand, we do not destroy the extra elements in the last page,
    // because we might need them later and the logic is simpler if we do not
    // destroy them. This means that elements are only destroyed only when the
    // page they belong to is destroyed. This is similar to what happens on
    // access of the elements of a page, where all the elements of the page are
    // constructed not only the one effectively neeeded.
    if (NewSize < Size) {
      size_t NewLastPage = (NewSize - 1) / PageSize;
      for (size_t I = NewLastPage + 1, E = PageToDataPtrs.size(); I < E; ++I) {
        T *PagePtr = PageToDataPtrs[I];
        if (PagePtr == InvalidPage)
          continue;
        T *Page = PagePtr;
        // We need to invoke the destructor on all the elements of the page.
        for (size_t J = 0; J < PageSize; ++J)
          Page[J].~T();
        Allocator.getPointer()->Deallocate(Page);
        // We mark the page invalid, to avoid double deletion.
        PageToDataPtrs[I] = InvalidPage;
      }
      PageToDataPtrs.resize(NewLastPage + 1);
    }
    Size = NewSize;
    // If the capacity is enough, just update the size and continue
    // with the currently allocated pages.
    if (Size <= capacity())
      return;
    // The number of pages to allocate. The Remainder is calculated
    // for the case in which the NewSize is not a multiple of PageSize.
    // In that case we need one more page.
    size_t Pages = Size / PageSize;
    size_t Remainder = Size % PageSize;
    if (Remainder != 0)
      Pages += 1;
    assert(Pages > PageToDataPtrs.size());
    // We use InvalidPage to indicate that a page has not been allocated yet.
    // This cannot be 0, because 0 is a valid page id.
    // We use InvalidPage instead of a separate bool to avoid wasting space.
    PageToDataPtrs.resize(Pages, InvalidPage);
  }

  [[nodiscard]] bool empty() const { return Size == 0; }

  /// Clear the vector, i.e. clear the allocated pages, the whole page
  /// lookup index and reset the size.
  void clear() {
    Size = 0;
    for (T *Page : PageToDataPtrs) {
      if (Page == InvalidPage)
        continue;
      for (size_t I = 0; I < PageSize; ++I)
        Page[I].~T();
      // If we do not own the allocator, deallocate the pages one by one.
      if (!Allocator.getInt()) {
        Allocator.getPointer()->Deallocate(Page);
      }
    }
    // If we own the allocator, simply reset it.
    if (Allocator.getInt() == true) {
      Allocator.getPointer()->Reset();
    }
    PageToDataPtrs.clear();
  }

  /// Iterator on all the elements of the vector
  /// which have actually being constructed.
  class MaterialisedIterator {
    const PagedVector *PV;
    size_t ElementIdx;

  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T *;
    using reference = T &;

    MaterialisedIterator(PagedVector const *PV, size_t ElementIdx)
        : PV(PV), ElementIdx(ElementIdx) {}

    /// Pre-increment operator.
    ///
    /// When incrementing the iterator, we skip the elements which have not
    /// been materialised yet.
    MaterialisedIterator &operator++() {
      ++ElementIdx;
      if (ElementIdx % PageSize == 0) {
        while (ElementIdx < PV->Size &&
               PV->PageToDataPtrs[ElementIdx / PageSize] == InvalidPage)
          ElementIdx += PageSize;
        if (ElementIdx > PV->Size)
          ElementIdx = PV->Size;
      }

      return *this;
    }

    MaterialisedIterator operator++(int) {
      MaterialisedIterator Copy = *this;
      ++*this;
      return Copy;
    }

    /// Dereference operator.
    ///
    /// Notice this can materialise elements if needed so there might be
    /// a page allocation and additional construction of the elements of
    /// such page.
    T const &operator*() const {
      assert(ElementIdx < PV->Size);
      assert(PV->PageToDataPtrs[ElementIdx / PageSize] != InvalidPage);
      T *PagePtr = PV->PageToDataPtrs[ElementIdx / PageSize];
      return *((ElementIdx % PageSize) + PagePtr);
    }

    friend bool operator==(MaterialisedIterator const &LHS,
                           MaterialisedIterator const &RHS);
    friend bool operator!=(MaterialisedIterator const &LHS,
                           MaterialisedIterator const &RHS);

    [[nodiscard]] size_t getIndex() const { return ElementIdx; }
  };

  /// Equality operator.
  friend bool operator==(MaterialisedIterator const &LHS,
                         MaterialisedIterator const &RHS) {
    assert(LHS.PV == RHS.PV);
    auto *PV = LHS.PV;

    // Any iterator for an empty vector is equal to any other iterator.
    if (LHS.PV->empty())
      return true;
    // Get the pages of the two iterators. If between the two pages there
    // are no valid pages, we can condider the iterators equal.
    size_t PageMin = std::min(LHS.ElementIdx, RHS.ElementIdx) / PageSize;
    size_t PageMax = std::max(LHS.ElementIdx, RHS.ElementIdx) / PageSize;
    // If the two pages are past the end, the iterators are equal.
    if (PageMin >= PV->PageToDataPtrs.size())
      return true;
    // If only the last page is past the end, the iterators are equal if
    // all the pages up to the end are invalid.
    if (PageMax >= PV->PageToDataPtrs.size()) {
      for (size_t PageIdx = PageMin; PageIdx < PV->PageToDataPtrs.size();
           ++PageIdx)
        if (LHS.PV->PageToDataPtrs[PageIdx] != InvalidPage)
          return false;
      return true;
    }

    T *Page1 = PV->PageToDataPtrs[PageMin];
    T *Page2 = PV->PageToDataPtrs[PageMax];
    if (Page1 == InvalidPage && Page2 == InvalidPage)
      return true;

    // If the two pages are the same, the iterators are equal if they point
    // to the same element.
    if (PageMin == PageMax)
      return LHS.ElementIdx == RHS.ElementIdx;

    // If the two pages are different, the iterators are equal if all the
    // pages between them are invalid.
    return std::all_of(PV->PageToDataPtrs.begin() + PageMin,
                       PV->PageToDataPtrs.begin() + PageMax,
                       [](T *Page) { return Page == InvalidPage; });
  }

  friend bool operator!=(MaterialisedIterator const &LHS,
                         MaterialisedIterator const &RHS) {
    return !(LHS == RHS);
  }

  friend class MaterialisedIterator;

  /// Iterators over the materialised elements of the vector.
  ///
  /// This includes all the elements belonging to allocated pages,
  /// even if they have not been accessed yet. It's enough to access
  /// one element of a page to materialise all the elements of the page.
  MaterialisedIterator materialised_begin() const {
    // Look for the first valid page
    for (size_t ElementIdx = 0; ElementIdx < Size; ElementIdx += PageSize)
      if (PageToDataPtrs[ElementIdx / PageSize] != InvalidPage)
        return MaterialisedIterator(this, ElementIdx);

    return MaterialisedIterator(this, Size);
  }

  MaterialisedIterator materialised_end() const {
    return MaterialisedIterator(this, Size);
  }

  [[nodiscard]] llvm::iterator_range<MaterialisedIterator>
  materialised() const {
    return {materialised_begin(), materialised_end()};
  }
};
} // namespace llvm
#endif // LLVM_ADT_PAGEDVECTOR_H
