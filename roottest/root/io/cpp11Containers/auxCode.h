#pragma once

#include "commonUtils.h"

// All of this would be terrible if real, but luckily it is just a mock up!

//------------------------------------------------------------------------------
template<class T>
class myLess{
public:
   constexpr bool operator()(const T &lhs, const T &rhs) const {
    return true;
   }
};

//------------------------------------------------------------------------------

template<class T>
class myHash
{
public:
    std::size_t operator()(T const& s) const
    {
        return 0;
    }
};

namespace std{
//   template<>
//   class hash<TH1F>
//   {
//   public:
//      std::size_t operator()(TH1F const& s) const
//      {
//         return 0;
//      }
//   };
//   template<>
//   class hash<vector<TH1F>>
//   {
//   public:
//      std::size_t operator()(vector<TH1F> const& s) const
//      {
//         return 0;
//      }
//   };
}
//------------------------------------------------------------------------------

template <class T>
class myAlloc : public std::allocator<T> {
public:
   myAlloc() = default;

   template <class U>
   myAlloc(const allocator<U> &other) noexcept
   {
   }

   template <typename U>
   struct rebind {
      using other = myAlloc<U>;
   };
};

//------------------------------------------------------------------------------

template< class T >
struct myEqual_to{
   constexpr bool operator()(const T &lhs, const T &rhs) const
{
    return true;
}
};
