//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Boris Perovic <boris.perovic@cern.ch>
// author:  Danilo Piparo <danilo.piparo@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------
#ifndef CLING_RUNTIME_PRINT_VALUE_H
#define CLING_RUNTIME_PRINT_VALUE_H

#if !defined(__CLING__)
#error "This file must not be included by compiled programs."
#endif

#include <string>
#include <tuple>

namespace cling {

  class Value;

  // General fallback - prints the address
  std::string printValue(const void *ptr, unsigned int);

  // void pointer
  std::string printValue(const void **ptr, unsigned int);

  // Bool
  std::string printValue(const bool *val, unsigned int);

  // Chars
  std::string printValue(const char *val, unsigned int);

  std::string printValue(const signed char *val, unsigned int);

  std::string printValue(const unsigned char *val, unsigned int);

  // Ints
  std::string printValue(const short *val, unsigned int);

  std::string printValue(const unsigned short *val, unsigned);

  std::string printValue(const int *val, unsigned int);

  std::string printValue(const unsigned int *val, unsigned int);

  std::string printValue(const long *val, unsigned int);

  std::string printValue(const unsigned long *val, unsigned int);

  std::string printValue(const long long *val, unsigned int);

  std::string printValue(const unsigned long long *val, unsigned int);

  // Reals
  std::string printValue(const float *val, unsigned int);

  std::string printValue(const double *val, unsigned int);

  std::string printValue(const long double *val, unsigned int);

  // Char pointers
  std::string printValue(const char *const *val, unsigned int);

  std::string printValue(const char **val, unsigned int);

  // std::string
  std::string printValue(const std::string *val, unsigned int);

  // cling::Value
  std::string printValue(const Value *value, unsigned int);

  // Collections internal declaration
  namespace collectionPrinterInternal {
    // Maps declaration
    template<typename CollectionType>
    auto printValue_impl(const CollectionType *obj, unsigned int recurseDepth, short)
      -> decltype(
      ++(obj->begin()), obj->end(),
        obj->begin()->first, obj->begin()->second,
        std::string());

    // Vector, set, deque etc. declaration
    template<typename CollectionType>
    auto printValue_impl(const CollectionType *obj, unsigned int recurseDepth, int)
      -> decltype(
      ++(obj->begin()), obj->end(),
        *(obj->begin()),
        std::string());

    // No general fallback anymore here, void* overload used for that now
  }

  // Collections
  template<typename CollectionType>
  auto printValue(const CollectionType *obj, unsigned int recurseDepth)
  -> decltype(collectionPrinterInternal::printValue_impl(obj, recurseDepth, 0), std::string())
  {
    return collectionPrinterInternal::printValue_impl(obj, recurseDepth, (short)0);  // short -> int -> long = priority order
  }

  // Arrays
  template<typename T, size_t N>
  std::string printValue(const T (*obj)[N], unsigned int recurseDepth) {
    std::string str = "{ ";

    for (int i = 0; i < N; ++i) {
      if (recurseDepth > 0) {
        str += printValue(*obj + i, recurseDepth - 1);
      } else {
        str += printValue((void *) (*obj + i), recurseDepth - 1);
      }
      if (i < N - 1) str += ", ";
    }

    return str + " }";
  }

  // Collections internal
  namespace collectionPrinterInternal {
    // Maps
    template<typename CollectionType>
    auto printValue_impl(const CollectionType *obj, unsigned int recurseDepth, short)
    -> decltype(
    ++(obj->begin()), obj->end(),
        obj->begin()->first, obj->begin()->second,
        std::string())
    {
      std::string str = "{ ";

      auto iter = obj->begin();
      auto iterEnd = obj->end();
      while (iter != iterEnd) {
        if (recurseDepth > 0) {
          str += printValue(&iter->first, recurseDepth-1);
          str += " => ";
          str += printValue(&iter->second, recurseDepth-1);
        } else {
          str += printValue((void*)&iter->first, recurseDepth-1);
          str += " => ";
          str += printValue((void*)&iter->second, recurseDepth-1);
        }
        ++iter;
        if (iter != iterEnd) {
          str += ", ";
        }
      }

      return str + " }";
    }

    // Vector, set, deque etc.
    template<typename CollectionType>
    auto printValue_impl(const CollectionType *obj, unsigned int recurseDepth, int)
    -> decltype(
    ++(obj->begin()), obj->end(),
        *(obj->begin()),  &(*(obj->begin())),
        std::string())
    {
      std::string str = "{ ";

      auto iter = obj->begin();
      auto iterEnd = obj->end();
      while (iter != iterEnd) {
        if (recurseDepth > 0) {
          str += printValue(&(*iter), recurseDepth-1);
        } else {
          str += printValue((void*)&(*iter), recurseDepth-1);
        }
        ++iter;
        if (iter != iterEnd) {
          str += ", ";
        }
      }

      return str + " }";
    }

    // Vector, set, deque etc.
    template<typename CollectionType>
    auto printValue_impl(const CollectionType *obj, int)
    -> decltype(
    ++(obj->begin()), obj->end(),
        *(obj->begin()),
        std::string())
     {
        std::string str = "{ ";

        auto iter = obj->begin();
        auto iterEnd = obj->end();
        while (iter != iterEnd) {
           const auto value = (*iter);
           str += printValue(&value);
           ++iter;
           if (iter != iterEnd) {
              str += ", ";
           }
        }

        return str + " }";
     }

  }

  // Tuples
  template <class... ARGS>
  std::string printValue(std::tuple<ARGS...> *);

  namespace collectionPrinterInternal {

    template <std::size_t N>
    const char *GetCommaOrEmpty()
    {
      static const auto comma = ", ";
      return comma;
    }

    template <>
    const char *GetCommaOrEmpty<0>()
    {
      static const auto empty = "";
      return empty;
    }

    // We loop at compile time from element 0 to element TUPLE_SIZE - 1
    // of the tuple. The running index is N which has as initial value
    // TUPLE_SIZE. We can therefore stop the iteration and account for the
    // empty tuple case with one single specialisation.
    template <class TUPLE,
              std::size_t N = std::tuple_size<TUPLE>(),
              std::size_t TUPLE_SIZE = std::tuple_size<TUPLE>()>
    struct tuplePrinter {
      static std::string print(TUPLE *t)
      {
        constexpr std::size_t elementNumber = TUPLE_SIZE - N;
        using Element_t = decltype(std::get<elementNumber>(*t));
        std::string ret;
        ret += GetCommaOrEmpty<elementNumber>();
        ret += cling::printValue(&std::get<elementNumber>(*t));
        // If N+1 is not smaller than the size of the tuple,
        // reroute the call to the printing function to the
        // no-op specialisation to stop recursion.
        constexpr std::size_t Nm1 = N - 1;
        ret += tuplePrinter<TUPLE, Nm1>::print((TUPLE *)t);
        return ret;
      }
    };

    // Special case: no op if last element reached or empty tuple
    template <class TUPLE, std::size_t TUPLE_SIZE>
    struct tuplePrinter<TUPLE, 0, TUPLE_SIZE>
    {
      static std::string print(TUPLE *t) {return "";}
    };

    template <class T>
    std::string tuplePairPrintValue(T *val)
    {
      std::string ret("{ ");
      ret += collectionPrinterInternal::tuplePrinter<T>::print(val);
      ret += " }";
      return ret;
    }
  }

  template <class... ARGS>
  std::string printValue(std::tuple<ARGS...> *val)
  {
    using T = std::tuple<ARGS...>;
    return collectionPrinterInternal::tuplePairPrintValue<T>(val);
  }

  template <class... ARGS>
  std::string printValue(std::pair<ARGS...> *val)
  {
    using T = std::pair<ARGS...>;
    return collectionPrinterInternal::tuplePairPrintValue<T>(val);
  }
}

#endif
