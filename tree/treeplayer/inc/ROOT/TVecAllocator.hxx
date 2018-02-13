// Author: Enrico Guiraud, Enric Tejedor, Danilo Piparo CERN  01/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVECALLOCATOR
#define ROOT_TVECALLOCATOR

#include <iostream>
#include <memory>

namespace ROOT {
namespace Detail {
namespace VecOps {

template <typename T>
class TVecAllocator {
public:
   using StdAlloc_t = std::allocator<T>;
   using value_type = typename StdAlloc_t::value_type;
   using pointer = typename StdAlloc_t::pointer;
   using const_pointer = typename StdAlloc_t::const_pointer;
   using reference = typename StdAlloc_t::reference;
   using const_reference = typename StdAlloc_t::const_reference;
   using size_type = typename StdAlloc_t::size_type;
   using difference_type = typename StdAlloc_t::difference_type;

private:
   enum class EAllocType : char { kRegular, kFromExternalPointer, kNoneYet };
   using StdAllocTraits_t = std::allocator_traits<StdAlloc_t>;
   pointer fInitialAddress = nullptr;
   size_type fInitialSize = 0;
   EAllocType fAllocType = EAllocType::kRegular;
   StdAlloc_t fStdAllocator;

public:
   TVecAllocator(pointer p, size_type n) : fInitialAddress(p), fInitialSize(n), fAllocType(EAllocType::kNoneYet){};
   TVecAllocator() = default;
   TVecAllocator(const TVecAllocator &) = default;

   void construct(pointer p, const_reference val)
   {
      // We refuse to do anything since we assume the memory is already initialised
      if (EAllocType::kFromExternalPointer == fAllocType)
         return;
      fStdAllocator.construct(p, val);
   }

   template <class U, class... Args>
   void construct(U *p, Args &&... args)
   {
      // We refuse to do anything since we assume the memory is already initialised
      if (EAllocType::kFromExternalPointer == fAllocType)
         return;
      fStdAllocator.construct(p, args...);
   }

   pointer allocate(std::size_t n)
   {
      if (n > std::size_t(-1) / sizeof(T))
         throw std::bad_alloc();
      if (EAllocType::kNoneYet == fAllocType) {
         fAllocType = EAllocType::kFromExternalPointer;
         return fInitialAddress;
      }
      fAllocType = EAllocType::kRegular;
      return StdAllocTraits_t::allocate(fStdAllocator, n);
   }

   void deallocate(pointer p, std::size_t n)
   {
      if (p != fInitialAddress)
         StdAllocTraits_t::deallocate(fStdAllocator, p, n);
   }

   bool operator==(const TVecAllocator<T> &other) { return fAllocType == other.fAllocType; }
   bool operator!=(const TVecAllocator<T> &other) { return fAllocType != other.fAllocType; }
};

} // End NS VecOps
} // End NS Internal
} // End NS ROOT

#endif
