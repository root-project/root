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
\class ROOT::Detail::VecOps::RAdoptAllocator
\ingroup vecops
\brief RAdoptAllocator can provide a view on already allocated memory.

The RAdoptAllocator behaves like the standard allocator, and, as such, can be used to create
stl containers. In addition, it behaves as if it allocated a certain memory region which
is indeed not managed by it, but rather is "adopted".
This is most useful to take advantage of widely adopted entities such as std::vector in a
novel way, namely offering nice interfaces around an arbitrary memory region.

If memory is adopted, the first allocation returns the address of this memory region. For
the subsequent allocations, the RAdoptAllocator behaves like a standard allocator.

For example:
~~~{.cpp}
std::vector<double> model {1, 2, 3};
unsigned int dummy;
RAdoptAllocator<double> alloc(model.data(), model.size());
std::vector<double, RAdoptAllocator<double>> v(model.size(), 0., alloc);
~~~
Now the vector *v* is ready to be used, de facto proxying the memory of the vector *model*.
Upon a second allocation, the vector *v* ceases to be a proxy
~~~{.cpp}
v.emplace_back(0.);
~~~
now the vector *v* owns its memory as a regular vector.
**/

template<typename T, bool IsCopyConstructible = std::is_copy_constructible<T>::value>
class RConstructHelper
{
   public:
      template <class... Args>
      static void Construct(std::allocator<T> &alloc, T *p, Args &&... args)
      {
         alloc.construct(p, args...);
      }
};

template <typename T>
class RConstructHelper<T, false> {
public:
   template <class... Args>
   static void Construct(std::allocator<T> &, T *, Args &&... ){}
};

template <typename T>
class RAdoptAllocator {
public:
   friend class RAdoptAllocator<bool>;

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
   template <typename U>
   struct rebind {
      using other = RAdoptAllocator<U>;
   };

private:
   enum class EAllocType : char { kOwning, kAdopting, kAdoptingNoAllocYet };
   using StdAllocTraits_t = std::allocator_traits<StdAlloc_t>;
   pointer fInitialAddress{nullptr};
   EAllocType fAllocType{EAllocType::kOwning};
   StdAlloc_t fStdAllocator;
   std::size_t fBufferSize{0};

public:
   /// This is the constructor which allows the allocator to adopt a certain memory region.
   RAdoptAllocator() = default;
   RAdoptAllocator(pointer p, std::size_t bufSize = 0)
      : fInitialAddress(p), fAllocType(p ? EAllocType::kAdoptingNoAllocYet : EAllocType::kOwning), fBufferSize(bufSize)
   {
   }
   RAdoptAllocator(const RAdoptAllocator &) = default;
   RAdoptAllocator(RAdoptAllocator &&) = default;
   RAdoptAllocator &operator=(const RAdoptAllocator &) = default;
   RAdoptAllocator &operator=(RAdoptAllocator &&) = default;
   RAdoptAllocator(const RAdoptAllocator<bool> &);

   std::size_t GetBufferSize() const { return fBufferSize;}
   bool IsAdoptingExternalMemory() const {
      return fBufferSize == 0 &&
             fInitialAddress != nullptr &&
             fAllocType != EAllocType::kOwning;
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
      if (fBufferSize == 0 && EAllocType::kAdopting == fAllocType)
         return;
      RConstructHelper<U>::Construct(fStdAllocator, p, args...);
      //fStdAllocator.construct(p, args...);
   }

   /// \brief Allocate some memory
   /// If an address has been adopted, at the first call, that address is returned.
   /// Subsequent calls will make "decay" the allocator to a regular stl allocator.
   pointer allocate(std::size_t n)
   {
      if (n > std::size_t(-1) / sizeof(T))
         throw std::bad_alloc();
      if ((EAllocType::kAdoptingNoAllocYet == fAllocType) &&
          (fBufferSize == 0 || (fBufferSize > 0 && n <= fBufferSize))) {
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

   template <class U>
   void destroy(U *p)
   {
      if (EAllocType::kAdopting != fAllocType) {
         fStdAllocator.destroy(p);
      }
   }

   bool operator==(const RAdoptAllocator<T> &other)
   {
      return fInitialAddress == other.fInitialAddress && fAllocType == other.fAllocType &&
             fStdAllocator == other.fStdAllocator;
   }

   bool operator!=(const RAdoptAllocator<T> &other) { return !(*this == other); }

   size_type max_size() const { return fStdAllocator.max_size(); };

};

// The different semantics of std::vector<bool> make  memory adoption through a
// custom allocator more complex -- namely, RAdoptAllocator<bool> must be rebindable
// to RAdoptAllocator<unsigned long>, but if adopted memory is really a buffer of
// bools reinterpretation of the buffer is not going to work. As a workaround,
// RAdoptAllocator<bool> is specialized to be a simple allocator that forwards calls
// to std::allocator and never adopts memory.
template <>
class RAdoptAllocator<bool> {
   std::allocator<bool> fStdAllocator;

public:
   template <typename U>
   struct rebind {
      using other = RAdoptAllocator<U>;
   };

   template <typename T>
   friend class RAdoptAllocator;

   using StdAlloc_t = std::allocator<bool>;
   using value_type = typename StdAlloc_t::value_type;
   using pointer = typename StdAlloc_t::pointer;
   using const_pointer = typename StdAlloc_t::const_pointer;
   using reference = typename StdAlloc_t::reference;
   using const_reference = typename StdAlloc_t::const_reference;
   using size_type = typename StdAlloc_t::size_type;
   using difference_type = typename StdAlloc_t::difference_type;

   RAdoptAllocator() = default;
   RAdoptAllocator(const RAdoptAllocator &) = default;

   template <typename U>
   RAdoptAllocator(const RAdoptAllocator<U> &o) : fStdAllocator(o.fStdAllocator)
   {
      if (o.fAllocType != RAdoptAllocator<U>::EAllocType::kOwning)
         throw std::runtime_error("Cannot rebind owning RAdoptAllocator");
   }

   bool *allocate(std::size_t n) { return fStdAllocator.allocate(n); }

   std::size_t GetBufferSize() const { return 0; }
   bool IsAdoptingExternalMemory() const { return false; }

   template <typename U, class... Args>
   void construct(U *p, Args &&... args)
   {
      fStdAllocator.construct(p, std::forward<Args>(args)...);
   }

   void deallocate(bool *p, std::size_t s) noexcept { fStdAllocator.deallocate(p, s); }

   template <class U>
   void destroy(U *p)
   {
      fStdAllocator.destroy(p);
   }

   bool operator==(const RAdoptAllocator &) { return true; }

   bool operator!=(const RAdoptAllocator &) { return false; }
};

// Helpers to initialise an allocator according to its value type
// if bool we initialise a stl allocator, if not an adopt allocator
template <typename ValueType>
RAdoptAllocator<ValueType> MakeAdoptAllocator(ValueType *buf, std::size_t n)
{
   return RAdoptAllocator<ValueType>(buf, n);
}

inline std::allocator<bool> MakeAdoptAllocator(bool *, std::size_t)
{
   return std::allocator<bool>();
}

template <typename ValueType>
RAdoptAllocator<ValueType> MakeAdoptAllocator(ValueType *p)
{
   return RAdoptAllocator<ValueType>(p);
}

inline std::allocator<bool> MakeAdoptAllocator(bool *)
{
   return std::allocator<bool>();
}

template <typename T>
RAdoptAllocator<T>::RAdoptAllocator(const RAdoptAllocator<bool> &o) : fStdAllocator(o.fStdAllocator)
{
}

template <typename Alloc_t>
std::size_t GetBufferSize(const Alloc_t &alloc)
{
   return alloc.GetBufferSize();
}

inline std::size_t GetBufferSize(const std::allocator<bool> &) { return 0;}

template <typename Alloc_t>
bool IsAdoptingExternalMemory(const Alloc_t &alloc)
{
   return alloc.IsAdoptingExternalMemory();
}

inline bool IsAdoptingExternalMemory(const std::allocator<bool> &)
{
   return false;
}

} // namespace VecOps
} // namespace Detail
} // namespace ROOT

#endif
