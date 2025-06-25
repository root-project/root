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

#include <ROOT/RDF/GraphUtils.hxx>
#include <ROOT/RDF/RActionBase.hxx>
#include <ROOT/RDF/RResultMap.hxx>
#include <ROOT/RResultHandle.hxx> // users of RunGraphs might rely on this transitive include
#include <ROOT/TypeTraits.hxx>

#include <array>
#include <chrono>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
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

// clang-format off
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
   return helper.RepresentGraph(node);
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
   std::string dotGraph = helper.RepresentGraph(node);

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
/// Run the event loops of multiple RDataFrames concurrently.
/// \param[in] handles A vector of RResultHandles whose event loops should be run.
/// \return The number of distinct computation graphs that have been processed.
///
/// This function triggers the event loop of all computation graphs which relate to the
/// given RResultHandles. The advantage compared to running the event loop implicitly by accessing the
/// RResultPtr is that the event loops will run concurrently. Therefore, the overall
/// computation of all results can be scheduled more efficiently.
/// It should be noted that user-defined operations (e.g., Filters and Defines) of the different RDataFrame graphs are assumed to be safe to call concurrently.
/// RDataFrame will pass slot numbers in the range [0, NThread-1] to all helpers used in nodes such as DefineSlot. NThread is the number of threads ROOT was
/// configured with in EnableImplicitMT().
/// Slot numbers are unique across all graphs, so no two tasks with the same slot number will run concurrently. Note that it is not guaranteed that each slot
/// number will be reached in every graph.
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
unsigned int RunGraphs(std::vector<RResultHandle> handles);

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

   std::unique_ptr<RDFInternal::RActionBase> variedAction;
   std::vector<std::shared_ptr<T>> variedResults;

   std::shared_ptr<RDFInternal::RActionBase> nominalAction = resPtr.fActionPtr;
   std::vector<std::string> variations = nominalAction->GetVariations();
   const auto nVariations = variations.size();

   if (nVariations > 0) {
      // clone the result once for each variation
      variedResults.reserve(nVariations);
      for (auto i = 0u; i < nVariations; ++i){
         // implicitly assuming that T is copiable: this should be the case
         // for all result types in use, as they are copied for each slot
         variedResults.emplace_back(new T{*resPtr.fObjPtr});

         // Check if the result's type T inherits from TNamed
         if constexpr (std::is_base_of<TNamed, T>::value) {
            // Get the current variation name
            std::string variationName = variations[i];
            // Replace the colon with an underscore
            std::replace(variationName.begin(), variationName.end(), ':', '_');
            // Get a pointer to the corresponding varied result
            auto &variedResult = variedResults.back();
            // Set the varied result's name to NOMINALNAME_VARIATIONAME
            variedResult->SetName((std::string(variedResult->GetName()) + "_" + variationName).c_str());
         }
      }

      std::vector<void *> typeErasedResults;
      typeErasedResults.reserve(variedResults.size());
      for (auto &res : variedResults)
         typeErasedResults.emplace_back(&res);

      // Create the RVariedAction and inject it in the computation graph.
      // This recursively creates all the required varied column readers and upstream nodes of the computation graph.
      variedAction = nominalAction->MakeVariedAction(std::move(typeErasedResults));
   }

   return RDFInternal::MakeResultMap<T>(resPtr.fObjPtr, std::move(variedResults), std::move(variations),
                                        *resPtr.fLoopManager, std::move(nominalAction), std::move(variedAction));
}

using SnapshotPtr_t = ROOT::RDF::RResultPtr<ROOT::RDF::RInterface<ROOT::Detail::RDF::RLoopManager, void>>;
SnapshotPtr_t VariationsFor(SnapshotPtr_t resPtr);

/// \brief Add ProgressBar to a ROOT::RDF::RNode
/// \param[in] df RDataFrame node at which ProgressBar is called.
///
/// The ProgressBar can be added not only at the RDataFrame head node, but also at any any computational node,
/// such as Filter or Define.
/// ###Example usage:
/// ~~~{.cpp}
/// ROOT::RDataFrame df("tree", "file.root");
/// auto df_1 = ROOT::RDF::RNode(df.Filter("x>1"));
/// ROOT::RDF::Experimental::AddProgressBar(df_1);
/// ~~~
void AddProgressBar(ROOT::RDF::RNode df);

/// \brief Add ProgressBar to an RDataFrame
/// \param[in] df RDataFrame for which ProgressBar is called.
///
/// This function adds a ProgressBar to display the event statistics in the terminal every
/// \b m events and every \b n seconds, including elapsed time, currently processed file,
/// currently processed events, the rate of event processing
/// and an estimated remaining time (per file being processed).
/// ProgressBar should be added after the dataframe object (df) is created first:
/// ~~~{.cpp}
/// ROOT::RDataFrame df("tree", "file.root");
/// ROOT::RDF::Experimental::AddProgressBar(df);
/// ~~~
/// For more details see ROOT::RDF::Experimental::ProgressHelper Class.
void AddProgressBar(ROOT::RDataFrame df);

/// @brief Set the number of threads sharing one TH3 in RDataFrame.
/// When RDF runs multi-threaded, each thread typically clones every histogram in the computation graph.
/// If this consumes too much memory, N threads can share one clone.
/// Higher values might slow down RDF because they lead to higher contention on the TH3Ds, but save memory.
/// Lower values run faster with less contention at the cost of higher memory usage.
/// @param nThread Number of threads that share a TH3D.
void ThreadsPerTH3(unsigned int nThread = 1);

class ProgressBarAction;

/// RDF progress helper.
/// This class provides callback functions to the RDataFrame. The event statistics
/// (including elapsed time, currently processed file, currently processed events, the rate of event processing
/// and an estimated remaining time (per file being processed))
/// are recorded and printed in the terminal every m events and every n seconds.
/// ProgressHelper::operator()(unsigned int, T&) is thread safe, and can be used as a callback in MT mode.
/// ProgressBar should be added after creating the dataframe object (df):
/// ~~~{.cpp}
/// ROOT::RDataFrame df("tree", "file.root");
/// ROOT::RDF::Experimental::AddProgressBar(df);
/// ~~~
/// alternatively RDataFrame can be cast to an RNode first giving it more flexibility.
/// For example, it can be called at any computational node, such as Filter or Define, not only the head node,
/// with no change to the ProgressBar function itself:
/// ~~~{.cpp}
/// ROOT::RDataFrame df("tree", "file.root");
/// auto df_1 = ROOT::RDF::RNode(df.Filter("x>1"));
/// ROOT::RDF::Experimental::AddProgressBar(df_1);
/// ~~~
class ProgressHelper {
private:
   double EvtPerSec() const;
   std::pair<std::size_t, std::chrono::seconds> RecordEvtCountAndTime();
   void PrintStats(std::ostream &stream, std::size_t currentEventCount, std::chrono::seconds totalElapsedSeconds) const;
   void PrintStatsFinal(std::ostream &stream, std::chrono::seconds totalElapsedSeconds) const;
   void PrintProgressBar(std::ostream &stream, std::size_t currentEventCount) const;

   std::chrono::time_point<std::chrono::system_clock> fBeginTime = std::chrono::system_clock::now();
   std::chrono::time_point<std::chrono::system_clock> fLastPrintTime = fBeginTime;
   std::chrono::seconds fPrintInterval{1};

   std::atomic<std::size_t> fProcessedEvents{0};
   std::size_t fLastProcessedEvents{0};
   std::size_t fIncrement;

   mutable std::mutex fSampleNameToEventEntriesMutex;
   std::map<std::string, ULong64_t> fSampleNameToEventEntries; // Filename, events in the file

   std::array<double, 20> fEventsPerSecondStatistics;
   std::size_t fEventsPerSecondStatisticsIndex{0};

   unsigned int fBarWidth;
   unsigned int fTotalFiles;

   std::mutex fPrintMutex;
   bool fIsTTY;
   bool fUseShellColours;

   std::shared_ptr<TTree> fTree{nullptr};

public:
   /// Create a progress helper.
   /// \param increment RDF callbacks are called every `n` events. Pass this `n` here.
   /// \param totalFiles read total number of files in the RDF.
   /// \param progressBarWidth Number of characters the progress bar will occupy.
   /// \param printInterval Update every stats every `n` seconds.
   /// \param useColors Use shell colour codes to colour the output. Automatically disabled when
   /// we are not writing to a tty.
   ProgressHelper(std::size_t increment, unsigned int totalFiles = 1, unsigned int progressBarWidth = 40,
                  unsigned int printInterval = 1, bool useColors = true);

   ~ProgressHelper() = default;

   friend class ProgressBarAction;

   /// Register a new sample for completion statistics.
   /// \see ROOT::RDF::RInterface::DefinePerSample().
   /// The *id.AsString()* refers to the name of the currently processed file.
   /// The idea is to populate the  event entries in the *fSampleNameToEventEntries* map
   /// by selecting the greater of the two values:
   /// *id.EntryRange().second* which is the upper event entry range of the processed sample
   /// and the current value of the event entries in the *fSampleNameToEventEntries* map.
   /// In the single threaded case, the two numbers are the same as the entry range corresponds
   /// to the number of events in an individual file (each sample is simply a single file).
   /// In the multithreaded case, the idea is to accumulate the higher event entry value until
   /// the total number of events in a given file is reached.
   void registerNewSample(unsigned int /*slot*/, const ROOT::RDF::RSampleInfo &id)
   {
      std::lock_guard<std::mutex> lock(fSampleNameToEventEntriesMutex);
      fSampleNameToEventEntries[id.AsString()] =
         std::max(id.EntryRange().second, fSampleNameToEventEntries[id.AsString()]);
   }

   /// Thread-safe callback for RDataFrame.
   /// It will record elapsed times and event statistics, and print a progress bar every n seconds (set by the
   /// fPrintInterval). \param slot Ignored. \param value Ignored.
   template <typename T>
   void operator()(unsigned int /*slot*/, T &value)
   {
      operator()(value);
   }
   // clang-format off
   /// Thread-safe callback for RDataFrame.
   /// It will record elapsed times and event statistics, and print a progress bar every n seconds (set by the fPrintInterval).
   /// \param value Ignored.
   // clang-format on
   template <typename T>
   void operator()(T & /*value*/)
   {
      using namespace std::chrono;
      // ***************************************************
      // Warning: Here, everything needs to be thread safe:
      // ***************************************************
      fProcessedEvents += fIncrement;

      // We only print every n seconds.
      if (duration_cast<seconds>(system_clock::now() - fLastPrintTime) < fPrintInterval) {
         return;
      }

      // ***************************************************
      // Protected by lock from here:
      // ***************************************************
      if (!fPrintMutex.try_lock())
         return;
      std::lock_guard<std::mutex> lockGuard(fPrintMutex, std::adopt_lock);

      std::size_t eventCount;
      seconds elapsedSeconds;
      std::tie(eventCount, elapsedSeconds) = RecordEvtCountAndTime();

      if (fIsTTY)
         std::cout << "\r";

      PrintProgressBar(std::cout, eventCount);
      PrintStats(std::cout, eventCount, elapsedSeconds);

      if (fIsTTY)
         std::cout << std::flush;
      else
         std::cout << std::endl;
   }

   std::size_t ComputeNEventsSoFar() const
   {
      std::unique_lock<std::mutex> lock(fSampleNameToEventEntriesMutex);
      std::size_t result = 0;
      for (const auto &item : fSampleNameToEventEntries)
         result += item.second;
      return result;
   }

   unsigned int ComputeCurrentFileIdx() const
   {
      std::unique_lock<std::mutex> lock(fSampleNameToEventEntriesMutex);
      return fSampleNameToEventEntries.size();
   }
};
} // namespace Experimental
} // namespace RDF
} // namespace ROOT
#endif
