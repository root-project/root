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
#include <ROOT/RDF/RActionBase.hxx>
#include <ROOT/RDF/RResultMap.hxx>
#include <ROOT/RResultHandle.hxx>
#include <ROOT/TypeTraits.hxx>

#include <algorithm> // std::transform
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <type_traits>
#include <utility> // std::index_sequence
#include <vector>

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
   std::decay_t<F> fFunc;

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
          typename Args = typename ROOT::TypeTraits::CallableTraits<std::decay_t<F>>::arg_types_nodecay,
          typename Ret = typename ROOT::TypeTraits::CallableTraits<std::decay_t<F>>::ret_type>
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
/// Create a graphviz representation of the dataframe computation graph, return it as a string.
/// \param[in] node any node of the graph. Called on the head (first) node, it prints the entire graph. Otherwise, only the branch the node belongs to.
///
/// The output can be displayed with a command akin to `dot -Tpng output.dot > output.png && open output.png`.
///
/// Note that "hanging" Defines, i.e. Defines without downstream nodes, will not be displayed by SaveGraph as they are
/// effectively optimized away from the computation graph.
///
/// Note that SaveGraph is not thread-safe and must not be called concurrently from different threads.
// clang-format on
template <typename NodeType>
std::string SaveGraph(NodeType node)
{
   ROOT::Internal::RDF::GraphDrawing::GraphCreatorHelper helper;
   return helper(node);
}

// clang-format off
/// Create a graphviz representation of the dataframe computation graph, write it to the specified file.
/// \param[in] node any node of the graph. Called on the head (first) node, it prints the entire graph. Otherwise, only the branch the node belongs to.
/// \param[in] outputFile file where to save the representation.
///
/// The output can be displayed with a command akin to `dot -Tpng output.dot > output.png && open output.png`.
///
/// Note that "hanging" Defines, i.e. Defines without downstream nodes, will not be displayed by SaveGraph as they are
/// effectively optimized away from the computation graph.
///
/// Note that SaveGraph is not thread-safe and must not be called concurrently from different threads.
// clang-format on
template <typename NodeType>
void SaveGraph(NodeType node, const std::string &outputFile)
{
   ROOT::Internal::RDF::GraphDrawing::GraphCreatorHelper helper;
   std::string dotGraph = helper(node);

   std::ofstream out(outputFile);
   if (!out.is_open()) {
      throw std::runtime_error("Could not open output file \"" + outputFile  + "\"for reading");
   }

   out << dotGraph;
   out.close();
}

// clang-format off
/// Cast a RDataFrame node to the common type ROOT::RDF::RNode
/// \param[in] node Any node of a RDataFrame graph
// clang-format on
template <typename NodeType>
RNode AsRNode(NodeType node)
{
   return node;
}

// clang-format off
/// Trigger the event loop of multiple RDataFrames concurrently
/// \param[in] handles A vector of RResultHandles
///
/// This function triggers the event loop of all computation graphs which relate to the
/// given RResultHandles. The advantage compared to running the event loop implicitly by accessing the
/// RResultPtr is that the event loops will run concurrently. Therefore, the overall
/// computation of all results is generally more efficient.
/// It should be noted that user-defined operations (e.g., Filters and Defines) of the different RDataFrame graphs are assumed to be safe to call concurrently.
///
/// ~~~{.cpp}
/// ROOT::RDataFrame df1("tree1", "file1.root");
/// auto r1 = df1.Histo1D("var1");
///
/// ROOT::RDataFrame df2("tree2", "file2.root");
/// auto r2 = df2.Sum("var2");
///
/// // RResultPtr -> RResultHandle conversion is automatic
/// ROOT::RDF::RunGraphs({r1, r2});
/// ~~~
// clang-format on
void RunGraphs(std::vector<RResultHandle> handles);

namespace Experimental {

/// \brief Produce all required systematic variations for the given result.
/// \param[in] resPtr The result for which variations should be produced.
/// \return A \ref ROOT::RDF::Experimental::RResultMap "RResultMap" object with full variation names as strings
///         (e.g. "pt:down") and the corresponding varied results as values.
///
/// A given input RResultPtr<T> produces a corresponding RResultMap<T> with a "nominal"
/// key that will return a value identical to the one contained in the original RResultPtr.
/// Other keys correspond to the varied values of this result, one for each variation
/// that the result depends on.
/// VariationsFor does not trigger the event loop. The event loop is only triggered
/// upon first access to a valid key, similarly to what happens with RResultPtr.
///
/// If the result does not depend, directly or indirectly, from any registered systematic variation, the
/// returned RResultMap will contain only the "nominal" key.
///
/// See RDataFrame's \ref ROOT::RDF::RInterface::Vary() "Vary" method for more information and example usages.
///
/// \note Currently, producing variations for the results of \ref ROOT::RDF::RInterface::Display() "Display",
///       \ref ROOT::RDF::RInterface::Report() "Report" and \ref ROOT::RDF::RInterface::Snapshot() "Snapshot"
///       actions is not supported.
//
// An overview of how systematic variations work internally. Given N variations (including the nominal):
//
// RResultMap   owns    RVariedAction
//  N results            N action helpers
//                       N previous filters
//                       N*#input_cols column readers
//
// ...and each RFilter and RDefine knows for what universe it needs to construct column readers ("nominal" by default).
template <typename T>
RResultMap<T> VariationsFor(RResultPtr<T> resPtr)
{
   R__ASSERT(resPtr != nullptr && "Calling VariationsFor on an empty RResultPtr");

   // populate parts of the computation graph for which we only have "empty shells", e.g. RJittedActions and
   // RJittedFilters
   resPtr.fLoopManager->Jit();

   std::shared_ptr<RDFInternal::RActionBase> nominalAction = resPtr.fActionPtr;

   // clone the result once for each variation
   std::vector<std::string> variations = nominalAction->GetVariations();
   const auto nVariations = variations.size();
   std::vector<std::shared_ptr<T>> results;
   results.reserve(nVariations);
   for (auto i = 0u; i < nVariations; ++i)
      results.emplace_back(new T{*resPtr.fObjPtr}); // implicitly assuming that T is copiable: this should be the case
                                                    // for all result types in use, as they are copied for each slot

   std::vector<void *> typeErasedResults;
   typeErasedResults.reserve(results.size());
   for (auto &res : results)
      typeErasedResults.emplace_back(&res);

   // create the RVariedAction and inject it in the computation graph
   // this recursively creates all the required varied column readers and upstream nodes of the computation graph
   std::unique_ptr<RDFInternal::RActionBase> variedAction{
      resPtr.fActionPtr->MakeVariedAction(std::move(typeErasedResults))};
   resPtr.fLoopManager->Book(variedAction.get());

   return RDFInternal::MakeResultMap<T>(resPtr.fObjPtr, std::move(results), std::move(variations), *resPtr.fLoopManager,
                                        std::move(nominalAction), std::move(variedAction));
}

} // namespace Experimental
} // namespace RDF
} // namespace ROOT
#endif
