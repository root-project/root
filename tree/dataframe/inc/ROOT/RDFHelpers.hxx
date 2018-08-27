// Author: Enrico Guiraud, Danilo Piparo CERN  02/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// This header contains helper free functions that slim down RDataFrame's programming model

#ifndef ROOT_RDF_HELPERS
#define ROOT_RDF_HELPERS

#include <ROOT/TypeTraits.hxx>
#include <ROOT/RIntegerSequence.hxx>
#include <TSystemDirectory.h>

#include <functional>
#include <type_traits>

namespace ROOT {
namespace Internal {
namespace RDF {
template <typename... ArgTypes, typename F>
std::function<bool(ArgTypes...)> NotHelper(ROOT::TypeTraits::TypeList<ArgTypes...>, F &&f)
{
   return std::function<bool(ArgTypes...)>([=](ArgTypes... args) mutable { return !f(args...); });
}

template <typename... ArgTypes, typename Ret, typename... Args>
std::function<bool(ArgTypes...)> NotHelper(ROOT::TypeTraits::TypeList<ArgTypes...>, Ret (*f)(Args...))
{
   return std::function<bool(ArgTypes...)>([=](ArgTypes... args) mutable { return !f(args...); });
}

template <typename I, typename T, typename F>
class PassAsVecHelper;

template <std::size_t ... N, typename T, typename F>
class PassAsVecHelper<std::index_sequence<N...>, T, F>
{
    template<std::size_t Idx>
    using AlwaysT = T;
    F fFunc;
public:
    PassAsVecHelper(F&& f) : fFunc(std::forward<F>(f)) {}
    auto operator()(AlwaysT<N>...args) -> decltype(fFunc({args...})) { return fFunc({args...}); }
};

template <std::size_t N, typename T, typename F>
auto PassAsVec(F &&f) -> PassAsVecHelper<std::make_index_sequence<N>, T, F>
{
    return PassAsVecHelper<std::make_index_sequence<N>, T, F>(std::forward<F>(f));
}

} // namespace RDF
} // namespace Internal

namespace RDF {
namespace RDFInternal = ROOT::Internal::RDF;
// clag-format off
/// Given a callable with signature bool(T1, T2, ...) return a callable with same signature that returns the negated result
///
/// The callable must have one single non-template definition of operator(). This is a limitation with respect to
/// std::not_fn, required for interoperability with RDataFrame.
// clang-format on
template <typename F,
          typename Args = typename ROOT::TypeTraits::CallableTraits<typename std::decay<F>::type>::arg_types_nodecay,
          typename Ret = typename ROOT::TypeTraits::CallableTraits<typename std::decay<F>::type>::ret_type>
auto Not(F &&f) -> decltype(RDFInternal::NotHelper(Args(), std::forward<F>(f)))
{
   static_assert(std::is_same<Ret, bool>::value, "RDF::Not requires a callable that returns a bool.");
   return RDFInternal::NotHelper(Args(), std::forward<F>(f));
}

// clang-format off
/// PassAsVec is a callable generator that allows passing N variables of type T to a function as a single collection.
///
/// PassAsVec<N, T>(func) returns a callable that takes N arguments of type T, passes them down to function `func` as
/// an initializer list `{t1, t2, t3,..., tN}` and returns whatever f({t1, t2, t3, ..., tN}) returns.
///
/// Note that for this to work with RDataFrame the type of all columns that the callable is applied to must be exactly T.
/// Example usage together with RDataFrame ("varX" columns must all be `float` variables):
/// \code
/// bool myVecFunc(std::vector<float> args);
/// df.Filter(PassAsVec<3, float>(myVecFunc), {"var1", "var2", "var3"});
/// \endcode
// clang-format on
template <std::size_t N, typename T, typename F>
auto PassAsVec(F &&f) -> RDFInternal::PassAsVecHelper<std::make_index_sequence<N>, T, F>
{
    return RDFInternal::PassAsVecHelper<std::make_index_sequence<N>, T, F>(std::forward<F>(f));
}

// clang-format off
/// CreateFilelist returns a vector with the paths to the files in the given directory.
///
/// CreateFilelist takes a path to a directory and an optional hint to the desired files. The files found
/// in the directory are only added to the list if the hint is a substring of the filename. This helper
/// can be used to feed multiple files convenviently to an RDataFrame.
/// \code
/// auto files = CreateFilelist("my_input_directory", "my_hint");
/// RDataFrame df("tree_name", files);
/// \endcode
// clang-format on
std::vector<std::string> CreateFilelist(const std::string &path, const std::string &hint = "")
{
   std::vector<std::string> list;
   TSystemDirectory dir(path.c_str(), path.c_str());
   auto files = dir.GetListOfFiles();
   if (files) {
      TSystemFile *file;
      TString fname;
      TIter next(files);
      std::string sanitized_path = path;
      if (!(path[path.size() - 1] == '/')) {
         sanitized_path.push_back('/');
      }
      while ((file = (TSystemFile *)next())) {
         fname = file->GetName();
         if (!file->IsDirectory() && fname.Contains(hint.c_str())) {
            list.emplace(list.begin(), (TString(sanitized_path.c_str()) + fname).Data());
         }
      }
   }
   return list;
}

} // namespace RDF
} // namespace ROOT
#endif
