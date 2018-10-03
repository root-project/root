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

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RDF/GraphUtils.hxx>
#include <ROOT/RIntegerSequence.hxx>
#include <ROOT/TypeTraits.hxx>

#include <algorithm> // std::transform
#include <functional>
#include <type_traits>
#include <vector>
#include <memory>
#include <fstream>
#include <iostream>

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

template <std::size_t... N, typename T, typename F>
class PassAsVecHelper<std::index_sequence<N...>, T, F> {
   template <std::size_t Idx>
   using AlwaysT = T;
   F fFunc;

public:
   PassAsVecHelper(F &&f) : fFunc(std::forward<F>(f)) {}
   auto operator()(AlwaysT<N>... args) -> decltype(fFunc({args...})) { return fFunc({args...}); }
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
template <typename Proxied, typename DataSource>
class RInterface;

// clang-format off
/// Creates the dot representation of the graph.
/// Won't work if the event loop has been executed
/// \param[in] node any node of the graph. Called on the head (first) node, it prints the entire graph. Otherwise, only the branch the node belongs to.
/// \param[in] filePath where to save the representation. If not specified, will be printed on standard output.
// clang-format on
template <typename NodeType>
void SaveGraph(NodeType node, const std::string &dotFilePath = "")
{
   ROOT::Internal::RDF::GraphDrawing::GraphCreatorHelper helper;
   std::string dotGraph = helper(node);

   if (dotFilePath == "") {
      // No file specified, print on standard output
      std::cout << dotGraph << std::endl;
      return;
   }

   std::ofstream out(dotFilePath);
   if (!out.is_open()) {
      throw std::runtime_error("File path not valid");
   }

   out << dotGraph;
   out.close();
}

} // namespace RDF
} // namespace ROOT
#endif
