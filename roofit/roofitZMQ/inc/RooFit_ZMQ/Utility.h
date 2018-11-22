#ifndef SERIALIZE_UTILITY_H
#define SERIALIZE_UTILITY_H 1

#if defined(__clang__) && (__clang_major__ > 3 || (__clang_major__ == 3 && __clang_minor__ >= 10))
#define HAVE_TRIVIALLY_COPYABLE 1
#elif defined(__GNUC__) && __GNUC__ >= 5
#define HAVE_TRIVIALLY_COPYABLE 1
#else
#undef HAVE_TRIVIALLY_COPYABLE
#endif

#include <type_traits>

namespace ZMQ {
  namespace Detail {

    template<typename... Ts>
    struct is_container_helper {
    };

#if defined(HAVE_TRIVIALLY_COPYABLE)
    template<class T>
    using simple_object = std::is_trivially_copyable<T>;
#else
    template<class T>
    using simple_object = std::is_pod<T>;
#endif

// is trivial
    template<class T>
    struct is_trivial : std::conditional<
        simple_object<
            typename std::decay<T>::type
        >::value,
        std::true_type,
        std::false_type
    >::type {
    };

  }
}

// is_pair
template<class>
struct is_pair : std::false_type {};

template<class F, class S>
struct is_pair<std::pair<F, S>> : public std::true_type {};

// is_container
template<typename T, typename _ = void>
struct is_container : std::false_type {};

template<typename T>
struct is_container<
   T,
   typename std::conditional<
      false,
      ZMQ::Detail::is_container_helper<
         typename T::value_type,
         typename T::size_type,
         typename T::allocator_type,
         typename T::iterator,
         typename T::const_iterator,
         decltype(std::declval<T>().size()),
         decltype(std::declval<T>().begin()),
         decltype(std::declval<T>().end()),
         decltype(std::declval<T>().cbegin()),
         decltype(std::declval<T>().cend())
         >,
      void
      >::type
   > : public std::true_type {};

#endif // SERIALIZE_UTILITY_H
