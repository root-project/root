// Author: Enrico Guiraud, Danilo Piparo CERN  12/2016

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDFUTILS
#define ROOT_RDFUTILS

#include "ROOT/RSpan.hxx"
#include <string_view>
#include "ROOT/RVec.hxx"
#include "ROOT/TypeTraits.hxx"
#include "Rtypes.h"

#include <array>
#include <deque>
#include <functional>
#include <memory>
#include <new> // std::hardware_destructive_interference_size
#include <unordered_set>
#include <shared_mutex>
#include <string>
#include <type_traits> // std::decay, std::false_type
#include <vector>

class TTree;
class TTreeReader;

namespace ROOT::RDF::Experimental {
class RDatasetSpec;
}
namespace ROOT {
namespace RDF {
using ColumnNames_t = std::vector<std::string>;
}

class RLogChannel;

namespace RDF {
class RDataSource;
}

namespace Detail {
namespace RDF {

using ROOT::RDF::ColumnNames_t;

ROOT::RLogChannel &RDFLogChannel();

// fwd decl for ColumnName2ColumnTypeName
class RDefineBase;

// type used for tag dispatching
struct RInferredType {
};

} // end ns Detail
} // end ns RDF

namespace Internal {
namespace RDF {

using namespace ROOT::TypeTraits;
using namespace ROOT::Detail::RDF;
using namespace ROOT::RDF;

/// Obtain or set the number of threads that will share a clone of a thread-safe 3D histogram.
/// Setting it to N will make N threads share a clone, setting it to 0 or 1 will use one clone
/// per thread.
/// Setting it to higher numbers reduces the RDF memory consumption, but might create contention
/// on TH3Ds. When the RDF computation graph consists mostly of filling TH3Ds, lower values are better.
/// \return A reference to the current divider.
unsigned int &NThreadPerTH3();

/// Check for container traits.
///
/// Note that for all uses in RDF we don't want to classify std::string as a container.
/// Template specializations of IsDataContainer make it return `true` for std::span<T>, std::vector<bool> and
/// RVec<bool>, which we do want to count as containers even though they do not satisfy all the traits tested by the
/// generic IsDataContainer<T>.
template <typename T>
struct IsDataContainer {
   using Test_t = std::decay_t<T>;

   template <typename A>
   static constexpr bool Test(A *pt, A const *cpt = nullptr, decltype(pt->begin()) * = nullptr,
                              decltype(pt->end()) * = nullptr, decltype(cpt->begin()) * = nullptr,
                              decltype(cpt->end()) * = nullptr, typename A::iterator *pi = nullptr,
                              typename A::const_iterator *pci = nullptr)
   {
      using It_t = typename A::iterator;
      using CIt_t = typename A::const_iterator;
      using V_t = typename A::value_type;
      return std::is_same<decltype(pt->begin()), It_t>::value && std::is_same<decltype(pt->end()), It_t>::value &&
             std::is_same<decltype(cpt->begin()), CIt_t>::value && std::is_same<decltype(cpt->end()), CIt_t>::value &&
             std::is_same<decltype(**pi), V_t &>::value && std::is_same<decltype(**pci), V_t const &>::value &&
             !std::is_same<T, std::string>::value;
   }

   template <typename A>
   static constexpr bool Test(...)
   {
      return false;
   }

   static constexpr bool value = Test<Test_t>(nullptr);
};

template<>
struct IsDataContainer<std::vector<bool>> {
   static constexpr bool value = true;
};

template<>
struct IsDataContainer<ROOT::VecOps::RVec<bool>> {
   static constexpr bool value = true;
};

template<typename T>
struct IsDataContainer<std::span<T>> {
   static constexpr bool value = true;
};

/// Detect whether a type is an instantiation of vector<T,A>
template <typename>
struct IsVector_t : public std::false_type {};

template <typename T, typename A>
struct IsVector_t<std::vector<T, A>> : public std::true_type {};

std::string GetBranchOrLeafTypeName(TTree &t, const std::string &colName);

const std::type_info &TypeName2TypeID(const std::string &name);

std::string TypeID2TypeName(const std::type_info &id);

std::string GetTypeNameWithOpts(const ROOT::RDF::RDataSource &df, std::string_view colName, bool vector2RVec);
std::string
ColumnName2ColumnTypeName(const std::string &colName, TTree *, RDataSource *, RDefineBase *, bool vector2RVec = true);

char TypeName2ROOTTypeName(const std::string &b);
char TypeID2ROOTTypeName(const std::type_info &tid);

unsigned int GetNSlots();

/// `type` is TypeList if MustRemove is false, otherwise it is a TypeList with the first type removed
template <bool MustRemove, typename TypeList>
struct RemoveFirstParameterIf {
   using type = TypeList;
};

template <typename TypeList>
struct RemoveFirstParameterIf<true, TypeList> {
   using type = RemoveFirstParameter_t<TypeList>;
};

template <bool MustRemove, typename TypeList>
using RemoveFirstParameterIf_t = typename RemoveFirstParameterIf<MustRemove, TypeList>::type;

template <bool MustRemove, typename TypeList>
struct RemoveFirstTwoParametersIf {
   using type = TypeList;
};

template <typename TypeList>
struct RemoveFirstTwoParametersIf<true, TypeList> {
   using typeTmp = typename RemoveFirstParameterIf<true, TypeList>::type;
   using type = typename RemoveFirstParameterIf<true, typeTmp>::type;
};

template <bool MustRemove, typename TypeList>
using RemoveFirstTwoParametersIf_t = typename RemoveFirstTwoParametersIf<MustRemove, TypeList>::type;

// Check the value_type type of a type with a SFINAE to allow compilation in presence
// fundamental types
template <typename T,
          bool IsDataContainer = IsDataContainer<std::decay_t<T>>::value || std::is_same<std::string, T>::value>
struct ValueType {
   using value_type = typename T::value_type;
};

template <typename T>
struct ValueType<T, false> {
   using value_type = T;
};

template <typename T>
struct ValueType<ROOT::VecOps::RVec<T>, false> {
   using value_type = T;
};

std::vector<std::string> ReplaceDotWithUnderscore(const std::vector<std::string> &columnNames);

/// Erase `that` element from vector `v`
template <typename T>
void Erase(const T &that, std::vector<T> &v)
{
   v.erase(std::remove(v.begin(), v.end(), that), v.end());
}

/// Declare code in the interpreter via the TInterpreter::Declare method, throw in case of errors
void InterpreterDeclare(const std::string &code);

/// Jit code in the interpreter with TInterpreter::Calc, throw in case of errors.
/// The optional `context` parameter, if present, is mentioned in the error message.
void InterpreterCalc(const std::string &code, const std::string &context = "");

/// Whether custom column with name colName is an "internal" column such as rdfentry_ or rdfslot_
bool IsInternalColumn(std::string_view colName);

/// Get optimal column width for printing a table given the names and the desired minimal space between columns
unsigned int GetColumnWidth(const std::vector<std::string>& names, const unsigned int minColumnSpace = 8u);

/// Stepping through CacheLineStep<T> values in a vector<T> brings you to a new cache line.
/// Useful to avoid false sharing.
template <typename T>
constexpr std::size_t CacheLineStep() {
   constexpr std::size_t cacheLineSize = R__HARDWARE_INTERFERENCE_SIZE;
   return (cacheLineSize + sizeof(T) - 1) / sizeof(T);
}

void CheckReaderTypeMatches(const std::type_info &colType, const std::type_info &requestedType,
                            const std::string &colName);

// TODO in C++17 this could be a lambda within FillHelper::Exec
template <typename T>
constexpr std::size_t FindIdxTrue(const T &arr)
{
   for (size_t i = 0; i < arr.size(); ++i) {
      if (arr[i])
         return i;
   }
   return arr.size();
}

// return type has to be decltype(auto) to preserve perfect forwarding
template <std::size_t N, typename... Ts>
decltype(auto) GetNthElement(Ts &&...args)
{
   auto tuple = std::forward_as_tuple(args...);
   return std::get<N>(tuple);
}

#if __cplusplus >= 201703L
template <class... Ts>
using Disjunction = std::disjunction<Ts...>;
#else
template <class...>
struct Disjunction : std::false_type {
};
template <class B1>
struct Disjunction<B1> : B1 {
};
template <class B1, class... Bn>
struct Disjunction<B1, Bn...> : std::conditional_t<bool(B1::value), B1, Disjunction<Bn...>> {
};
#endif

bool IsStrInVec(const std::string &str, const std::vector<std::string> &vec);

/// Return a vector with all elements of v1 and v2 and duplicates removed.
/// Precondition: each of v1 and v2 must not have duplicate elements.
template <typename T>
std::vector<T> Union(const std::vector<T> &v1, const std::vector<T> &v2)
{
   std::vector<T> res = v1;

   // Add the variations coming from the input columns
   for (const auto &e : v2)
      if (std::find(v1.begin(), v1.end(), e) == v1.end())
         res.emplace_back(e);

   return res;
}

/**
 * \brief A Thread-safe cache for strings.
 *
 * This is used to generically store strings that are created in the computation
 * graph machinery, for example when adding a new node.
 */
class RStringCache {
   std::unordered_set<std::string> fStrings{};
   std::shared_mutex fMutex{};

public:
   /**
    * \brief Inserts the input string in the cache and returns an iterator to the cached string.
    *
    * The function implements the following strategy for thread-safety:
    * 1. Take a shared lock and early return if the string is already in the cache.
    * 2. Release the shared lock and take an exclusive lock.
    * 3. Check again if another thread filled the cache meanwhile. If so, return the cached value.
    * 4. Insert the new value in the cache and return.
    */
   auto Insert(const std::string &string) -> decltype(fStrings)::const_iterator;
};

/**
 * \brief Struct to wrap the call to a function with a guaranteed order of
 *        execution of its arguments.
 * \tparam F Type of the callable.
 * \tparam Args Variadic types of the arguments to the callable.
 *
 * The execution order is guaranteed by calling the function in the constructor
 * thus enabling the exploitation of the list-initialization sequenced-before
 * feature (See rule 9 at https://en.cppreference.com/w/cpp/language/eval_order).
 */
struct CallGuaranteedOrder {
   template <typename F, typename... Args>
   CallGuaranteedOrder(F &&f, Args &&...args)
   {
      f(std::forward<Args>(args)...);
   }
};

template <typename T>
auto MakeAliasedSharedPtr(T *rawPtr)
{
   const static std::shared_ptr<T> fgRawPtrCtrlBlock;
   return std::shared_ptr<T>(fgRawPtrCtrlBlock, rawPtr);
}


/**
 * \brief Function to retrieve RDatasetSpec from JSON file provided
 * \param[in] jsonFile Path to the dataset specification JSON file.
 *
 * This function allows us to have access to an RDatasetSpec which needs to
 * be created when we use the FromSpec factory function.
 */
ROOT::RDF::Experimental::RDatasetSpec RetrieveSpecFromJson(const std::string &jsonFile);

/**
 * Tag to let data sources use the native data type when creating a column reader.
 *
 * See usage of this in RNTupleDS
 */
struct UseNativeDataType {};

} // end NS RDF
} // end NS Internal
} // end NS ROOT

#endif // RDFUTILS
