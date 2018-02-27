// Author: Enrico Guiraud, Enric Tejedor, Danilo Piparo CERN  01/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TADOPTALLOCATOR
#define ROOT_TADOPTALLOCATOR

#include <iostream>
#include <memory>

namespace ROOT {
namespace Detail {
namespace VecOps {

/**
\class ROOT::Detail::VecOps::TAdoptAllocator
\ingroup vecops
\brief TAdoptAllocator allows to bind to an already initialised memory region as managed.

The TAdoptAllocator behaves like the standard allocator, and, as such, can be used to create
stl containers. In addition, it can pretend to have allocated a certain memory region which
is not managed by it, but rather adopted.
This is most useful to take advantage of widely adopted entities such as std::vector in a
novel way, namely offering nice interfaces around an arbitrary memory region.

If memory is adopted, the first allocation returns the address of this memory region. For
the subsequent allocations, the TAdoptAllocator behaves like a standard allocator.

For example:
~~~{.cpp}
std::vector<double> model {1, 2, 3};
unsigned int dummy;
TAdoptAllocator<double> alloc(model.data(), model.size());
std::vector<double, TAdoptAllocator<double>> v(model.size(), 0., alloc);
~~~
Now the vector *v* is ready to be used, de facto proxying the memory of the vector *model*.
Upon a second allocation, the vector *v* ceases to be a proxy
~~~{.cpp}
v.emplace_back(0.);
~~~
now the vector *v* owns its memory as a regular vector.
**/

template <typename T>
class TAdoptAllocator {
public:
   using propagate_on_container_move_assignment = std::true_type;
   using propagate_on_container_swap = std::true_type;
   using StdAlloc_t = std::allocator<T>;
   using value_type = typename StdAlloc_t::value_type;
   using pointer = typename StdAlloc_t::pointer;
   using const_pointer = typename StdAlloc_t::const_pointer;
   using reference = typename StdAlloc_t::reference;
   using const_reference = typename StdAlloc_t::const_reference;
   using size_type = typename StdAlloc_t::size_type;
   using difference_type = typename StdAlloc_t::difference_type;
   template<typename U>
   struct rebind { using other = TAdoptAllocator<U>; };
private:
   enum class EAllocType : char { kOwning, kAdopting, kAdoptingNoAllocYet };
   using StdAllocTraits_t = std::allocator_traits<StdAlloc_t>;
   pointer fInitialAddress = nullptr;
   EAllocType fAllocType = EAllocType::kOwning;
   StdAlloc_t fStdAllocator;

public:
   /// This is the constructor which allows the allocator to adopt a certain memory region.
   TAdoptAllocator(pointer p)
      : fInitialAddress(p), fAllocType(EAllocType::kAdoptingNoAllocYet){};
   TAdoptAllocator() = default;
   TAdoptAllocator(const TAdoptAllocator &) = default;
   TAdoptAllocator(TAdoptAllocator &&) = default;
   TAdoptAllocator &operator=(const TAdoptAllocator &) = default;
   TAdoptAllocator &operator=(TAdoptAllocator &&) = default;

   /// Construct a value at a certain memory address
   /// This method is a no op if memory has been adopted.
   void construct(pointer p, const_reference val)
   {
      // We refuse to do anything since we assume the memory is already initialised
      if (EAllocType::kAdopting == fAllocType)
         return;
      fStdAllocator.construct(p, val);
   }

   /// Construct an object at a certain memory address
   /// \tparam U The type of the memory address at which the object needs to be constructed
   /// \tparam Args The arguments' types necessary for the construction of the object
   /// \param[in] p The memory address at which the object needs to be constructed
   /// \param[in] args The arguments necessary for the construction of the object
   /// This method is a no op if memory has been adopted.
   template <class U, class... Args>
   void construct(U *p, Args &&... args)
   {
      // We refuse to do anything since we assume the memory is already initialised
      if (EAllocType::kAdopting == fAllocType)
         return;
      fStdAllocator.construct(p, args...);
   }

   /// \brief Allocate some memory
   /// If an address has been adopted, at the first call, that address is returned.
   /// Subsequent calls will make "decay" the allocator to a regular stl allocator.
   pointer allocate(std::size_t n)
   {
      if (n > std::size_t(-1) / sizeof(T))
         throw std::bad_alloc();
      if (EAllocType::kAdoptingNoAllocYet == fAllocType) {
         fAllocType = EAllocType::kAdopting;
         return fInitialAddress;
      }
      fAllocType = EAllocType::kOwning;
      return StdAllocTraits_t::allocate(fStdAllocator, n);
   }

   /// \brief Dellocate some memory if that had not been adopted.
   void deallocate(pointer p, std::size_t n)
   {
      if (p != fInitialAddress)
         StdAllocTraits_t::deallocate(fStdAllocator, p, n);
   }

   bool operator==(const TAdoptAllocator<T> &other)
   {
      return fInitialAddress == other.fInitialAddress &&
             fAllocType == other.fAllocType &&
             fStdAllocator == other.fStdAllocator;
   }
   bool operator!=(const TAdoptAllocator<T> &other)
   {
      return !(*this == other);
   }
};

} // End NS VecOps
} // End NS Internal
} // End NS ROOT

#endif
