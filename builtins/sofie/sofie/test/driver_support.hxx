// Serialization of values between the test drivers and the tests. A driver is a
// small program compiled by a test around a generated model header: it runs the
// model and writes the result with write(), the test reads it back with Reader.
//
// The format is a whitespace separated list of tokens: arithmetic values are
// written in decimal with enough digits to round trip, a vector is its size
// followed by its elements and a tuple is the concatenation of its elements.
//
// This header only depends on the C++ standard library, as it is also included
// by the drivers.

#ifndef SOFIE_TEST_DRIVER_SUPPORT
#define SOFIE_TEST_DRIVER_SUPPORT

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace sofie_test {

template <class T>
struct IsVector : std::false_type {
};
template <class T, class A>
struct IsVector<std::vector<T, A>> : std::true_type {
};

template <class T>
struct IsTuple : std::false_type {
};
template <class... Ts>
struct IsTuple<std::tuple<Ts...>> : std::true_type {
};

template <class T>
void write(std::ostream &os, T const &value)
{
   if constexpr (std::is_floating_point_v<T>) {
      os << std::setprecision(std::numeric_limits<T>::max_digits10) << value << '\n';
   } else if constexpr (std::is_integral_v<T>) {
      if constexpr (std::is_signed_v<T>)
         os << static_cast<long long>(value) << '\n';
      else
         os << static_cast<unsigned long long>(value) << '\n';
   } else if constexpr (IsVector<T>::value) {
      os << value.size() << '\n';
      for (auto const &element : value)
         write(os, element);
   } else if constexpr (IsTuple<T>::value) {
      std::apply([&os](auto const &...elements) { (write(os, elements), ...); }, value);
   } else {
      static_assert(!sizeof(T), "type not supported");
   }
}

/// Reads back what write() wrote
class Reader {
public:
   explicit Reader(std::istream &in) : fIn(in) {}

   template <class T>
   T read()
   {
      if constexpr (std::is_arithmetic_v<T>) {
         std::string token;
         if (!(fIn >> token))
            throw std::runtime_error("unexpected end of the output of the driver");
         // std::strto* also parse nan and inf, unlike the stream extraction operators
         if constexpr (std::is_floating_point_v<T>)
            return static_cast<T>(std::strtod(token.c_str(), nullptr));
         else if constexpr (std::is_signed_v<T>)
            return static_cast<T>(std::strtoll(token.c_str(), nullptr, 10));
         else
            return static_cast<T>(std::strtoull(token.c_str(), nullptr, 10));
      } else if constexpr (IsVector<T>::value) {
         T out;
         auto size = read<std::size_t>();
         out.reserve(size);
         for (std::size_t i = 0; i < size; ++i)
            out.push_back(read<typename T::value_type>());
         return out;
      } else if constexpr (IsTuple<T>::value) {
         return readTuple<T>(std::make_index_sequence<std::tuple_size_v<T>>{});
      } else {
         static_assert(!sizeof(T), "type not supported");
      }
   }

private:
   template <class Tuple, std::size_t... Is>
   Tuple readTuple(std::index_sequence<Is...>)
   {
      // braced initialization evaluates the arguments from left to right
      return Tuple{read<std::tuple_element_t<Is, Tuple>>()...};
   }

   std::istream &fIn;
};

} // namespace sofie_test

#endif
