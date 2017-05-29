// Author: Enrico Guiraud, Danilo Piparo CERN  03/2017

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDF_TINTERFACE
#define ROOT_TDF_TINTERFACE

#include "ROOT/TBufferMerger.hxx"
#include "ROOT/TResultProxy.hxx"
#include "ROOT/TDFNodes.hxx"
#include "ROOT/TDFActionHelpers.hxx"
#include "ROOT/TDFUtils.hxx"
#include "TChain.h"
#include "TFile.h"
#include "TH1.h" // For Histo actions
#include "TH2.h" // For Histo actions
#include "TH3.h" // For Histo actions
#include "TInterpreter.h"
#include "TProfile.h"   // For Histo actions
#include "TProfile2D.h" // For Histo actions
#include "TRegexp.h"
#include "TROOT.h" // IsImplicitMTEnabled

#include <initializer_list>
#include <memory>
#include <string>
#include <sstream>
#include <typeinfo>
#include <type_traits> // is_same, enable_if

namespace ROOT {

namespace Internal {
namespace TDF {
using namespace ROOT::Experimental::TDF;
using namespace ROOT::Detail::TDF;

using TmpBranchBasePtr_t = std::shared_ptr<TCustomColumnBase>;

template <typename TDFNode, typename ActionType, typename... BranchTypes, typename ActionResultType>
void CallBuildAndBook(TDFNode *node, const ColumnNames_t &bl, unsigned int nSlots,
                      const std::shared_ptr<ActionResultType> &r)
{
   node->template BuildAndBook<BranchTypes...>(bl, r, nSlots, (ActionType *)nullptr);
}

std::vector<std::string> GetUsedBranchesNames(const std::string, TObjArray *, const std::vector<std::string> &);

Long_t JitTransformation(void *thisPtr, const std::string &methodName, const std::string &nodeTypeName,
                         const std::string &name, const std::string &expression, TObjArray *branches,
                         const std::vector<std::string> &tmpBranches,
                         const std::map<std::string, TmpBranchBasePtr_t> &tmpBookedBranches, TTree *tree);

void JitBuildAndBook(const ColumnNames_t &bl, const std::string &nodeTypename, void *thisPtr, const std::type_info &art,
                     const std::type_info &at, const void *r, TTree &tree, unsigned int nSlots,
                     const std::map<std::string, TmpBranchBasePtr_t> &tmpBranches);

} // namespace TDF
} // namespace Internal

namespace Experimental {

// forward declarations
class TDataFrame;

} // namespace Experimental
} // namespace ROOT

namespace cling {
std::string printValue(ROOT::Experimental::TDataFrame *tdf); // For a nice printing at the promp
}

namespace ROOT {
namespace Experimental {
namespace TDF {
namespace TDFDetail = ROOT::Detail::TDF;
namespace TDFInternal = ROOT::Internal::TDF;

/**
* \class ROOT::Experimental::TDF::TInterface
* \ingroup dataframe
* \brief The public interface to the TDataFrame federation of classes
* \tparam T One of the "node" base types (e.g. TLoopManager, TFilterBase). The user never specifies this type manually.
*/
template <typename Proxied>
class TInterface {
   using ColumnNames_t = TDFDetail::ColumnNames_t;
   using TFilterBase = TDFDetail::TFilterBase;
   using TRangeBase = TDFDetail::TRangeBase;
   using TCustomColumnBase = TDFDetail::TCustomColumnBase;
   using TLoopManager = TDFDetail::TLoopManager;
   friend std::string cling::printValue(ROOT::Experimental::TDataFrame *tdf); // For a nice printing at the prompt
   template <typename T>
   friend class TInterface;
   template <typename TDFNode, typename ActionType, typename... BranchTypes, typename ActionResultType>
   friend void TDFInternal::CallBuildAndBook(TDFNode *, const TDFDetail::ColumnNames_t &, unsigned int nSlots,
                                             const std::shared_ptr<ActionResultType> &);

public:
   ////////////////////////////////////////////////////////////////////////////
   /// \brief Append a filter to the call graph.
   /// \param[in] f Function, lambda expression, functor class or any other callable object. It must return a `bool`
   /// signalling whether the event has passed the selection (true) or not (false).
   /// \param[in] bn Names of the branches in input to the filter function.
   /// \param[in] name Optional name of this filter. See `Report`.
   ///
   /// Append a filter node at the point of the call graph corresponding to the
   /// object this method is called on.
   /// The callable `f` should not have side-effects (e.g. modification of an
   /// external or static variable) to ensure correct results when implicit
   /// multi-threading is active.
   ///
   /// TDataFrame only evaluates filters when necessary: if multiple filters
   /// are chained one after another, they are executed in order and the first
   /// one returning false causes the event to be discarded.
   /// Even if multiple actions or transformations depend on the same filter,
   /// it is executed once per entry. If its result is requested more than
   /// once, the cached result is served.
   template <typename F, typename std::enable_if<!std::is_convertible<F, std::string>::value, int>::type = 0>
   TInterface<TFilterBase> Filter(F f, const ColumnNames_t &bn = {}, const std::string &name = "")
   {
      TDFInternal::CheckFilter(f);
      auto df = GetDataFrameChecked();
      const ColumnNames_t &defBl = df->GetDefaultBranches();
      auto nArgs = TDFInternal::TFunctionTraits<F>::Args_t::fgSize;
      const ColumnNames_t &actualBl = TDFInternal::PickBranchNames(nArgs, bn, defBl);
      using DFF_t = TDFDetail::TFilter<F, Proxied>;
      auto FilterPtr = std::make_shared<DFF_t>(std::move(f), actualBl, *fProxiedPtr, name);
      fProxiedPtr->IncrChildrenCount();
      df->Book(FilterPtr);
      TInterface<TFilterBase> tdf_f(FilterPtr, fImplWeakPtr);
      return tdf_f;
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Append a filter to the call graph.
   /// \param[in] f Function, lambda expression, functor class or any other callable object. It must return a `bool`
   /// signalling whether the event has passed the selection (true) or not (false).
   /// \param[in] name Optional name of this filter. See `Report`.
   ///
   /// Refer to the first overload of this method for the full documentation.
   template <typename F, typename std::enable_if<!std::is_convertible<F, std::string>::value, int>::type = 0>
   TInterface<TFilterBase> Filter(F f, const std::string &name)
   {
      // The sfinae is there in order to pick up the overloaded method which accepts two strings
      // rather than this template method.
      return Filter(f, {}, name);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Append a filter to the call graph.
   /// \param[in] f Function, lambda expression, functor class or any other callable object. It must return a `bool`
   /// signalling whether the event has passed the selection (true) or not (false).
   /// \param[in] bn Names of the branches in input to the filter function.
   ///
   /// Refer to the first overload of this method for the full documentation.
   template <typename F>
   TInterface<TFilterBase> Filter(F f, const std::initializer_list<std::string> &bn)
   {
      return Filter(f, ColumnNames_t{bn});
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Append a filter to the call graph.
   /// \param[in] expression The filter expression in C++
   /// \param[in] name Optional name of this filter. See `Report`.
   ///
   /// The expression is just in time compiled and used to filter entries. The
   /// variable names to be used inside are the names of the branches. Only
   /// valid C++ is accepted.
   /// Refer to the first overload of this method for the full documentation.
   TInterface<TFilterBase> Filter(const std::string &expression, const std::string &name = "")
   {
      auto df = GetDataFrameChecked();
      auto tree = df->GetTree();
      auto branches = tree->GetListOfBranches();
      auto tmpBranches = fProxiedPtr->GetTmpBranches();
      auto tmpBookedBranches = df->GetBookedBranches();
      auto retVal = TDFInternal::JitTransformation(this, "Filter", GetNodeTypeName(), name, expression, branches,
                                                   tmpBranches, tmpBookedBranches, tree);
      return *(TInterface<TFilterBase> *)retVal;
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Creates a temporary branch
   /// \param[in] name The name of the temporary branch.
   /// \param[in] expression Function, lambda expression, functor class or any other callable object producing the
   /// temporary value. Returns the value that will be assigned to the temporary branch.
   /// \param[in] bl Names of the branches in input to the producer function.
   ///
   /// Create a temporary branch that will be visible from all subsequent nodes
   /// of the functional chain. The `expression` is only evaluated for entries that pass
   /// all the preceding filters.
   /// A new variable is created called `name`, accessible as if it was contained
   /// in the dataset from subsequent transformations/actions.
   ///
   /// Use cases include:
   ///
   /// * caching the results of complex calculations for easy and efficient multiple access
   /// * extraction of quantities of interest from complex objects
   /// * branch aliasing, i.e. changing the name of a branch
   ///
   /// An exception is thrown if the name of the new branch is already in use
   /// for another branch in the TTree.
   template <typename F, typename std::enable_if<!std::is_convertible<F, std::string>::value, int>::type = 0>
   TInterface<TCustomColumnBase> Define(const std::string &name, F expression, const ColumnNames_t &bl = {})
   {
      auto df = GetDataFrameChecked();
      TDFInternal::CheckTmpBranch(name, df->GetTree());
      const ColumnNames_t &defBl = df->GetDefaultBranches();
      auto nArgs = TDFInternal::TFunctionTraits<F>::Args_t::fgSize;
      const ColumnNames_t &actualBl = TDFInternal::PickBranchNames(nArgs, bl, defBl);
      using DFB_t = TDFDetail::TCustomColumn<F, Proxied>;
      auto BranchPtr = std::make_shared<DFB_t>(name, std::move(expression), actualBl, *fProxiedPtr);
      fProxiedPtr->IncrChildrenCount();
      df->Book(BranchPtr);
      TInterface<TCustomColumnBase> tdf_b(BranchPtr, fImplWeakPtr);
      return tdf_b;
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Creates a temporary branch
   /// \param[in] name The name of the temporary branch.
   /// \param[in] expression An expression in C++ which represents the temporary value
   ///
   /// The expression is just in time compiled and used to produce new values. The
   /// variable names to be used inside are the names of the branches. Only
   /// valid C++ is accepted.
   /// Refer to the first overload of this method for the full documentation.
   TInterface<TCustomColumnBase> Define(const std::string &name, const std::string &expression)
   {
      auto df = GetDataFrameChecked();
      auto tree = df->GetTree();
      auto branches = tree->GetListOfBranches();
      auto tmpBranches = fProxiedPtr->GetTmpBranches();
      auto tmpBookedBranches = df->GetBookedBranches();
      auto retVal = TDFInternal::JitTransformation(this, "Define", GetNodeTypeName(), name, expression, branches,
                                                   tmpBranches, tmpBookedBranches, tree);
      return *(TInterface<TCustomColumnBase> *)retVal;
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Create a snapshot of the dataset on disk in the form of a TTree
   /// \tparam BranchTypes variadic list of branch/column types
   /// \param[in] treename The name of the output TTree
   /// \param[in] filename The name of the output TFile
   /// \param[in] bnames The list of names of the branches to be written
   /// \param[in] filecacheMB The cache size of each memory file in MB (default = 16)
   ///
   /// This function returns a `TDataFrame` built with the output tree as a source.
   template <typename... BranchTypes>
   TInterface<TLoopManager> Snapshot(const std::string &treename, const std::string &filename,
                                     const ColumnNames_t &bnames, Long_t filecacheMB = 16)
   {
      using TypeInd_t = typename TDFInternal::TGenStaticSeq<sizeof...(BranchTypes)>::Type_t;
      return SnapshotImpl<BranchTypes...>(treename, filename, bnames, filecacheMB, TypeInd_t());
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Create a snapshot of the dataset on disk in the form of a TTree
   /// \param[in] treename The name of the output TTree
   /// \param[in] filename The name of the output TFile
   /// \param[in] bnames The list of names of the branches to be written
   ///
   /// This function returns a `TDataFrame` built with the output tree as a source.
   /// The types of the branches are automatically inferred and do not need to be specified.
   TInterface<TLoopManager> Snapshot(const std::string &treename, const std::string &filename,
                                     const ColumnNames_t &bnames)
   {
      auto df = GetDataFrameChecked();
      auto tree = df->GetTree();
      std::stringstream snapCall;
      // build a string equivalent to
      // "reinterpret_cast</nodetype/*>(this)->Snapshot<Ts...>(treename,filename,*reinterpret_cast<ColumnNames_t*>(&bnames))"
      snapCall << "((" << GetNodeTypeName() << "*)" << this << ")->Snapshot<";
      bool first = true;
      for (auto &b : bnames) {
         if (!first) snapCall << ", ";
         snapCall << TDFInternal::ColumnName2ColumnTypeName(b, *tree, df->GetBookedBranch(b));
         first = false;
      };
      // TODO is there a way to use ColumnNames_t instead of std::vector<std::string> without parsing the whole header?
      snapCall << ">(\"" << treename << "\", \"" << filename << "\", "
               << "*reinterpret_cast<std::vector<std::string>*>(" << &bnames << ")"
               << ");";
      // jit snapCall, return result
      return *reinterpret_cast<TInterface<TLoopManager> *>(gInterpreter->ProcessLine(snapCall.str().c_str()));
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Create a snapshot of the dataset on disk in the form of a TTree
   /// \param[in] treename The name of the output TTree
   /// \param[in] filename The name of the output TFile
   /// \param[in] columnNameRegexp The regular expression to match the column names to be selected. Empty means all.
   ///
   /// This function returns a `TDataFrame` built with the output tree as a source.
   /// The types of the branches are automatically inferred and do not need to be specified.
   TInterface<TLoopManager> Snapshot(const std::string &treename, const std::string &filename,
                                     const std::string &columnNameRegexp = "")
   {
      ColumnNames_t selectedColumns;
      selectedColumns.reserve(32);
      const auto isEmptyRegex = "" == columnNameRegexp;

      const auto tmpBranches = fProxiedPtr->GetTmpBranches();
      // Since we support gcc48 and it does not provide in its stl std::regex,
      // we need to use TRegexp
      TRegexp regexp(columnNameRegexp);
      int dummy;
      for (auto &&branchName : tmpBranches) {
         if (isEmptyRegex || -1 != regexp.Index(branchName.c_str(), &dummy)) {
            selectedColumns.emplace_back(branchName);
         }
      }

      auto df = GetDataFrameChecked();
      const auto branches = df->GetTree()->GetListOfBranches();
      for (auto branch : *branches) {
         auto branchName = branch->GetName();
         if (isEmptyRegex || -1 != regexp.Index(branchName, &dummy)) {
            selectedColumns.emplace_back(branchName);
         }
      }

      return Snapshot(treename, filename, selectedColumns);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Creates a node that filters entries based on range
   /// \param[in] start How many entries to discard before resuming processing.
   /// \param[in] stop Total number of entries that will be processed before stopping. 0 means "never stop".
   /// \param[in] stride Process one entry every `stride` entries. Must be strictly greater than 0.
   ///
   /// Ranges are only available if EnableImplicitMT has _not_ been called. Multi-thread ranges are not supported.
   TInterface<TRangeBase> Range(unsigned int start, unsigned int stop, unsigned int stride = 1)
   {
      // check invariants
      if (stride == 0 || (stop != 0 && stop < start))
         throw std::runtime_error("Range: stride must be strictly greater than 0 and stop must be greater than start.");
      if (ROOT::IsImplicitMTEnabled())
         throw std::runtime_error("Range was called with ImplicitMT enabled. Multi-thread ranges are not supported.");

      auto df = GetDataFrameChecked();
      using Range_t = TDFDetail::TRange<Proxied>;
      auto RangePtr = std::make_shared<Range_t>(start, stop, stride, *fProxiedPtr);
      fProxiedPtr->IncrChildrenCount();
      df->Book(RangePtr);
      TInterface<TRangeBase> tdf_r(RangePtr, fImplWeakPtr);
      return tdf_r;
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Creates a node that filters entries based on range
   /// \param[in] stop Total number of entries that will be processed before stopping. 0 means "never stop".
   ///
   /// See the other Range overload for a detailed description.
   TInterface<TRangeBase> Range(unsigned int stop) { return Range(0, stop, 1); }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Execute a user-defined function on each entry (*instant action*)
   /// \param[in] f Function, lambda expression, functor class or any other callable object performing user defined
   /// calculations.
   /// \param[in] bl Names of the branches in input to the user function.
   ///
   /// The callable `f` is invoked once per entry. This is an *instant action*:
   /// upon invocation, an event loop as well as execution of all scheduled actions
   /// is triggered.
   /// Users are responsible for the thread-safety of this callable when executing
   /// with implicit multi-threading enabled (i.e. ROOT::EnableImplicitMT).
   template <typename F>
   void Foreach(F f, const ColumnNames_t &bl = {})
   {
      using Args_t = typename TDFInternal::TFunctionTraits<decltype(f)>::ArgsNoDecay_t;
      using Ret_t = typename TDFInternal::TFunctionTraits<decltype(f)>::Ret_t;
      ForeachSlot(TDFInternal::AddSlotParameter<Ret_t>(f, Args_t()), bl);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Execute a user-defined function requiring a processing slot index on each entry (*instant action*)
   /// \param[in] f Function, lambda expression, functor class or any other callable object performing user defined
   /// calculations.
   /// \param[in] bl Names of the branches in input to the user function.
   ///
   /// Same as `Foreach`, but the user-defined function takes an extra
   /// `unsigned int` as its first parameter, the *processing slot index*.
   /// This *slot index* will be assigned a different value, `0` to `poolSize - 1`,
   /// for each thread of execution.
   /// This is meant as a helper in writing thread-safe `Foreach`
   /// actions when using `TDataFrame` after `ROOT::EnableImplicitMT()`.
   /// The user-defined processing callable is able to follow different
   /// *streams of processing* indexed by the first parameter.
   /// `ForeachSlot` works just as well with single-thread execution: in that
   /// case `slot` will always be `0`.
   template <typename F>
   void ForeachSlot(F f, const ColumnNames_t &bl = {})
   {
      auto df = GetDataFrameChecked();
      const ColumnNames_t &defBl = df->GetDefaultBranches();
      auto nArgs = TDFInternal::TFunctionTraits<F>::Args_t::fgSize;
      const ColumnNames_t &actualBl = TDFInternal::PickBranchNames(nArgs - 1, bl, defBl);
      using Op_t = TDFInternal::ForeachSlotHelper<F>;
      using DFA_t = TDFInternal::TAction<Op_t, Proxied>;
      df->Book(std::make_shared<DFA_t>(Op_t(std::move(f)), actualBl, *fProxiedPtr));
      fProxiedPtr->IncrChildrenCount();
      df->Run();
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Execute a user-defined reduce operation on the values of a branch
   /// \tparam F The type of the reduce callable. Automatically deduced.
   /// \tparam T The type of the branch to apply the reduction to. Automatically deduced.
   /// \param[in] f A callable with signature `T(T,T)`
   /// \param[in] branchName The branch to be reduced. If omitted, the default branch is used instead.
   ///
   /// A reduction takes two values of a branch and merges them into one (e.g.
   /// by summing them, taking the maximum, etc). This action performs the
   /// specified reduction operation on all branch values, returning
   /// a single value of the same type. The callable f must satisfy the general
   /// requirements of a *processing function* besides having signature `T(T,T)`
   /// where `T` is the type of branch.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TResultProxy documentation.
   template <typename F, typename T = typename TDFInternal::TFunctionTraits<F>::Ret_t>
   TResultProxy<T> Reduce(F f, const std::string &branchName = {})
   {
      static_assert(std::is_default_constructible<T>::value,
                    "reduce object cannot be default-constructed. Please provide an initialisation value (initValue)");
      return Reduce(std::move(f), branchName, T());
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Execute a user-defined reduce operation on the values of a branch
   /// \tparam F The type of the reduce callable. Automatically deduced.
   /// \tparam T The type of the branch to apply the reduction to. Automatically deduced.
   /// \param[in] f A callable with signature `T(T,T)`
   /// \param[in] branchName The branch to be reduced. If omitted, the default branch is used instead.
   /// \param[in] initValue The reduced object is initialised to this value rather than being default-constructed
   ///
   /// See the description of the other Reduce overload for more information.
   template <typename F, typename T = typename TDFInternal::TFunctionTraits<F>::Ret_t>
   TResultProxy<T> Reduce(F f, const std::string &branchName, const T &initValue)
   {
      using Args_t = typename TDFInternal::TFunctionTraits<F>::Args_t;
      TDFInternal::CheckReduce(f, Args_t());
      auto df = GetDataFrameChecked();
      unsigned int nSlots = df->GetNSlots();
      auto bl = GetBranchNames<T>({branchName}, "reduce branch values");
      auto redObjPtr = std::make_shared<T>(initValue);
      using Op_t = TDFInternal::ReduceHelper<F, T>;
      using DFA_t = typename TDFInternal::TAction<Op_t, Proxied>;
      df->Book(std::make_shared<DFA_t>(Op_t(std::move(f), redObjPtr, nSlots), bl, *fProxiedPtr));
      fProxiedPtr->IncrChildrenCount();
      return MakeResultProxy(redObjPtr, df);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return the number of entries processed (*lazy action*)
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TResultProxy documentation.
   TResultProxy<unsigned int> Count()
   {
      auto df = GetDataFrameChecked();
      unsigned int nSlots = df->GetNSlots();
      auto cSPtr = std::make_shared<unsigned int>(0);
      using Op_t = TDFInternal::CountHelper;
      using DFA_t = TDFInternal::TAction<Op_t, Proxied>;
      df->Book(std::make_shared<DFA_t>(Op_t(cSPtr, nSlots), ColumnNames_t({}), *fProxiedPtr));
      fProxiedPtr->IncrChildrenCount();
      return MakeResultProxy(cSPtr, df);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return a collection of values of a branch (*lazy action*)
   /// \tparam T The type of the branch.
   /// \tparam COLL The type of collection used to store the values.
   /// \param[in] branchName The name of the branch of which the values are to be collected
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TResultProxy documentation.
   template <typename T, typename COLL = std::vector<T>>
   TResultProxy<COLL> Take(const std::string &branchName = "")
   {
      auto df = GetDataFrameChecked();
      unsigned int nSlots = df->GetNSlots();
      auto bl = GetBranchNames<T>({branchName}, "get the values of the branch");
      auto valuesPtr = std::make_shared<COLL>();
      using Op_t = TDFInternal::TakeHelper<T, COLL>;
      using DFA_t = TDFInternal::TAction<Op_t, Proxied>;
      df->Book(std::make_shared<DFA_t>(Op_t(valuesPtr, nSlots), bl, *fProxiedPtr));
      fProxiedPtr->IncrChildrenCount();
      return MakeResultProxy(valuesPtr, df);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a one-dimensional histogram with the values of a branch (*lazy action*)
   /// \tparam V The type of the branch used to fill the histogram.
   /// \param[in] model The returned histogram will be constructed using this as a model.
   /// \param[in] vName The name of the branch that will fill the histogram.
   ///
   /// The default branches, if available, will be used instead of branches whose names are left empty.
   /// Branches can be of a container type (e.g. std::vector<double>), in which case the histogram
   /// is filled with each one of the elements of the container. In case multiple branches of container type
   /// are provided (e.g. values and weights) they must have the same length for each one of the events (but
   /// possibly different lengths between events).
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TResultProxy documentation.
   /// The user gives up ownership of the model histogram.
   template <typename V = TDFDetail::TInferType>
   TResultProxy<::TH1F> Histo1D(::TH1F &&model = ::TH1F{"", "", 128u, 0., 0.}, const std::string &vName = "")
   {
      auto bl = GetBranchNames<V>({vName}, "fill the histogram");
      auto h = std::make_shared<::TH1F>(std::move(model));
      if (h->GetXaxis()->GetXmax() == h->GetXaxis()->GetXmin())
         TDFInternal::HistoUtils<::TH1F>::SetCanExtendAllAxes(*h);
      return CreateAction<TDFInternal::ActionTypes::Histo1D, V>(bl, h);
   }

   template <typename V = TDFDetail::TInferType>
   TResultProxy<::TH1F> Histo1D(const std::string &vName)
   {
      return Histo1D<V>(::TH1F{"", "", 128u, 0., 0.}, vName);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a one-dimensional histogram with the values of a branch (*lazy action*)
   /// \tparam V The type of the branch used to fill the histogram.
   /// \param[in] model The returned histogram will be constructed using this as a model.
   /// \param[in] vName The name of the branch that will fill the histogram.
   /// \param[in] wName The name of the branch that will provide the weights.
   ///
   /// The default branches, if available, will be used instead of branches whose names are left empty.
   /// Branches can be of a container type (e.g. std::vector<double>), in which case the histogram
   /// is filled with each one of the elements of the container. In case multiple branches of container type
   /// are provided (e.g. values and weights) they must have the same length for each one of the events (but
   /// possibly different lengths between events).
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TResultProxy documentation.
   /// The user gives up ownership of the model histogram.
   template <typename V = TDFDetail::TInferType, typename W = TDFDetail::TInferType>
   TResultProxy<::TH1F> Histo1D(::TH1F &&model, const std::string &vName, const std::string &wName)
   {
      auto bl = GetBranchNames<V, W>({vName, wName}, "fill the histogram");
      auto h = std::make_shared<::TH1F>(std::move(model));
      return CreateAction<TDFInternal::ActionTypes::Histo1D, V, W>(bl, h);
   }

   template <typename V = TDFDetail::TInferType, typename W = TDFDetail::TInferType>
   TResultProxy<::TH1F> Histo1D(const std::string &vName, const std::string &wName)
   {
      return Histo1D<V, W>(::TH1F{"", "", 128u, 0., 0.}, vName, wName);
   }

   template <typename V, typename W>
   TResultProxy<::TH1F> Histo1D(::TH1F &&model = ::TH1F{"", "", 128u, 0., 0.})
   {
      return Histo1D<V, W>(std::move(model), "", "");
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a two-dimensional histogram (*lazy action*)
   /// \tparam V1 The type of the branch used to fill the x axis of the histogram.
   /// \tparam V2 The type of the branch used to fill the y axis of the histogram.
   /// \param[in] model The returned histogram will be constructed using this as a model.
   /// \param[in] v1Name The name of the branch that will fill the x axis.
   /// \param[in] v2Name The name of the branch that will fill the y axis.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TResultProxy documentation.
   /// The user gives up ownership of the model histogram.
   template <typename V1 = TDFDetail::TInferType, typename V2 = TDFDetail::TInferType>
   TResultProxy<::TH2F> Histo2D(::TH2F &&model, const std::string &v1Name = "", const std::string &v2Name = "")
   {
      auto h = std::make_shared<::TH2F>(std::move(model));
      if (!TDFInternal::HistoUtils<::TH2F>::HasAxisLimits(*h)) {
         throw std::runtime_error("2D histograms with no axes limits are not supported yet.");
      }
      auto bl = GetBranchNames<V1, V2>({v1Name, v2Name}, "fill the histogram");
      return CreateAction<TDFInternal::ActionTypes::Histo2D, V1, V2>(bl, h);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a two-dimensional histogram (*lazy action*)
   /// \tparam V1 The type of the branch used to fill the x axis of the histogram.
   /// \tparam V2 The type of the branch used to fill the y axis of the histogram.
   /// \tparam W The type of the branch used for the weights of the histogram.
   /// \param[in] model The returned histogram will be constructed using this as a model.
   /// \param[in] v1Name The name of the branch that will fill the x axis.
   /// \param[in] v2Name The name of the branch that will fill the y axis.
   /// \param[in] wName The name of the branch that will provide the weights.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TResultProxy documentation.
   /// The user gives up ownership of the model histogram.
   template <typename V1 = TDFDetail::TInferType, typename V2 = TDFDetail::TInferType,
             typename W = TDFDetail::TInferType>
   TResultProxy<::TH2F> Histo2D(::TH2F &&model, const std::string &v1Name, const std::string &v2Name,
                                const std::string &wName)
   {
      auto h = std::make_shared<::TH2F>(std::move(model));
      if (!TDFInternal::HistoUtils<::TH2F>::HasAxisLimits(*h)) {
         throw std::runtime_error("2D histograms with no axes limits are not supported yet.");
      }
      auto bl = GetBranchNames<V1, V2, W>({v1Name, v2Name, wName}, "fill the histogram");
      return CreateAction<TDFInternal::ActionTypes::Histo2D, V1, V2, W>(bl, h);
   }

   template <typename V1, typename V2, typename W>
   TResultProxy<::TH2F> Histo2D(::TH2F &&model)
   {
      return Histo2D<V1, V2, W>(std::move(model), "", "", "");
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a three-dimensional histogram (*lazy action*)
   /// \tparam V1 The type of the branch used to fill the x axis of the histogram.
   /// \tparam V2 The type of the branch used to fill the y axis of the histogram.
   /// \tparam V3 The type of the branch used to fill the z axis of the histogram.
   /// \param[in] model The returned histogram will be constructed using this as a model.
   /// \param[in] v1Name The name of the branch that will fill the x axis.
   /// \param[in] v2Name The name of the branch that will fill the y axis.
   /// \param[in] v3Name The name of the branch that will fill the z axis.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TResultProxy documentation.
   /// The user gives up ownership of the model histogram.
   template <typename V1 = TDFDetail::TInferType, typename V2 = TDFDetail::TInferType,
             typename V3 = TDFDetail::TInferType>
   TResultProxy<::TH3F> Histo3D(::TH3F &&model, const std::string &v1Name = "", const std::string &v2Name = "",
                                const std::string &v3Name = "")
   {
      auto h = std::make_shared<::TH3F>(std::move(model));
      if (!TDFInternal::HistoUtils<::TH3F>::HasAxisLimits(*h)) {
         throw std::runtime_error("3D histograms with no axes limits are not supported yet.");
      }
      auto bl = GetBranchNames<V1, V2, V3>({v1Name, v2Name, v3Name}, "fill the histogram");
      return CreateAction<TDFInternal::ActionTypes::Histo3D, V1, V2, V3>(bl, h);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a three-dimensional histogram (*lazy action*)
   /// \tparam V1 The type of the branch used to fill the x axis of the histogram.
   /// \tparam V2 The type of the branch used to fill the y axis of the histogram.
   /// \tparam V3 The type of the branch used to fill the z axis of the histogram.
   /// \tparam W The type of the branch used for the weights of the histogram.
   /// \param[in] model The returned histogram will be constructed using this as a model.
   /// \param[in] v1Name The name of the branch that will fill the x axis.
   /// \param[in] v2Name The name of the branch that will fill the y axis.
   /// \param[in] v3Name The name of the branch that will fill the z axis.
   /// \param[in] wName The name of the branch that will provide the weights.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TResultProxy documentation.
   /// The user gives up ownership of the model histogram.
   template <typename V1 = TDFDetail::TInferType, typename V2 = TDFDetail::TInferType,
             typename V3 = TDFDetail::TInferType, typename W = TDFDetail::TInferType>
   TResultProxy<::TH3F> Histo3D(::TH3F &&model, const std::string &v1Name, const std::string &v2Name,
                                const std::string &v3Name, const std::string &wName)
   {
      auto h = std::make_shared<::TH3F>(std::move(model));
      if (!TDFInternal::HistoUtils<::TH3F>::HasAxisLimits(*h)) {
         throw std::runtime_error("3D histograms with no axes limits are not supported yet.");
      }
      auto bl = GetBranchNames<V1, V2, V3, W>({v1Name, v2Name, v3Name, wName}, "fill the histogram");
      return CreateAction<TDFInternal::ActionTypes::Histo3D, V1, V2, V3, W>(bl, h);
   }

   template <typename V1, typename V2, typename V3, typename W>
   TResultProxy<::TH3F> Histo3D(::TH3F &&model)
   {
      return Histo3D<V1, V2, V3, W>(std::move(model), "", "", "", "");
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a one-dimensional profile (*lazy action*)
   /// \tparam V1 The type of the branch the values of which are used to fill the profile.
   /// \tparam V2 The type of the branch the values of which are used to fill the profile.
   /// \param[in] model The model to be considered to build the new return value.
   /// \param[in] v1Name The name of the branch that will fill the x axis.
   /// \param[in] v2Name The name of the branch that will fill the y axis.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TResultProxy documentation.
   /// The user gives up ownership of the model profile object.
   template <typename V1 = TDFDetail::TInferType, typename V2 = TDFDetail::TInferType>
   TResultProxy<::TProfile> Profile1D(::TProfile &&model, const std::string &v1Name = "",
                                      const std::string &v2Name = "")
   {
      auto h = std::make_shared<::TProfile>(std::move(model));
      if (!TDFInternal::HistoUtils<::TProfile>::HasAxisLimits(*h)) {
         throw std::runtime_error("Profiles with no axes limits are not supported yet.");
      }
      auto bl = GetBranchNames<V1, V2>({v1Name, v2Name}, "fill the 1D Profile");
      return CreateAction<TDFInternal::ActionTypes::Profile1D, V1, V2>(bl, h);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a one-dimensional profile (*lazy action*)
   /// \tparam V1 The type of the branch the values of which are used to fill the profile.
   /// \tparam V2 The type of the branch the values of which are used to fill the profile.
   /// \tparam W The type of the branch the weights of which are used to fill the profile.
   /// \param[in] model The model to be considered to build the new return value.
   /// \param[in] v1Name The name of the branch that will fill the x axis.
   /// \param[in] v2Name The name of the branch that will fill the y axis.
   /// \param[in] wName The name of the branch that will provide the weights.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TResultProxy documentation.
   /// The user gives up ownership of the model profile object.
   template <typename V1 = TDFDetail::TInferType, typename V2 = TDFDetail::TInferType,
             typename W = TDFDetail::TInferType>
   TResultProxy<::TProfile> Profile1D(::TProfile &&model, const std::string &v1Name, const std::string &v2Name,
                                      const std::string &wName)
   {
      auto h = std::make_shared<::TProfile>(std::move(model));
      if (!TDFInternal::HistoUtils<::TProfile>::HasAxisLimits(*h)) {
         throw std::runtime_error("Profile histograms with no axes limits are not supported yet.");
      }
      auto bl = GetBranchNames<V1, V2, W>({v1Name, v2Name, wName}, "fill the 1D profile");
      return CreateAction<TDFInternal::ActionTypes::Profile1D, V1, V2, W>(bl, h);
   }

   template <typename V1, typename V2, typename W>
   TResultProxy<::TProfile> Profile1D(::TProfile &&model)
   {
      return Profile1D<V1, V2, W>(std::move(model), "", "", "");
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a two-dimensional profile (*lazy action*)
   /// \tparam V1 The type of the branch used to fill the x axis of the histogram.
   /// \tparam V2 The type of the branch used to fill the y axis of the histogram.
   /// \tparam V2 The type of the branch used to fill the z axis of the histogram.
   /// \param[in] model The returned profile will be constructed using this as a model.
   /// \param[in] v1Name The name of the branch that will fill the x axis.
   /// \param[in] v2Name The name of the branch that will fill the y axis.
   /// \param[in] v3Name The name of the branch that will fill the z axis.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TResultProxy documentation.
   /// The user gives up ownership of the model profile.
   template <typename V1 = TDFDetail::TInferType, typename V2 = TDFDetail::TInferType,
             typename V3 = TDFDetail::TInferType>
   TResultProxy<::TProfile2D> Profile2D(::TProfile2D &&model, const std::string &v1Name = "",
                                        const std::string &v2Name = "", const std::string &v3Name = "")
   {
      auto h = std::make_shared<::TProfile2D>(std::move(model));
      if (!TDFInternal::HistoUtils<::TProfile2D>::HasAxisLimits(*h)) {
         throw std::runtime_error("2D profiles with no axes limits are not supported yet.");
      }
      auto bl = GetBranchNames<V1, V2, V3>({v1Name, v2Name, v3Name}, "fill the 2D profile");
      return CreateAction<TDFInternal::ActionTypes::Profile2D, V1, V2, V3>(bl, h);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a two-dimensional profile (*lazy action*)
   /// \tparam V1 The type of the branch used to fill the x axis of the histogram.
   /// \tparam V2 The type of the branch used to fill the y axis of the histogram.
   /// \tparam V3 The type of the branch used to fill the z axis of the histogram.
   /// \tparam W The type of the branch used for the weights of the histogram.
   /// \param[in] model The returned histogram will be constructed using this as a model.
   /// \param[in] v1Name The name of the branch that will fill the x axis.
   /// \param[in] v2Name The name of the branch that will fill the y axis.
   /// \param[in] v3Name The name of the branch that will fill the z axis.
   /// \param[in] wName The name of the branch that will provide the weights.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TResultProxy documentation.
   /// The user gives up ownership of the model profile.
   template <typename V1 = TDFDetail::TInferType, typename V2 = TDFDetail::TInferType,
             typename V3 = TDFDetail::TInferType, typename W = TDFDetail::TInferType>
   TResultProxy<::TProfile2D> Profile2D(::TProfile2D &&model, const std::string &v1Name, const std::string &v2Name,
                                        const std::string &v3Name, const std::string &wName)
   {
      auto h = std::make_shared<::TProfile2D>(std::move(model));
      if (!TDFInternal::HistoUtils<::TProfile2D>::HasAxisLimits(*h)) {
         throw std::runtime_error("2D profiles with no axes limits are not supported yet.");
      }
      auto bl = GetBranchNames<V1, V2, V3, W>({v1Name, v2Name, v3Name, wName}, "fill the histogram");
      return CreateAction<TDFInternal::ActionTypes::Profile2D, V1, V2, V3, W>(bl, h);
   }

   template <typename V1, typename V2, typename V3, typename W>
   TResultProxy<::TProfile2D> Profile2D(::TProfile2D &&model)
   {
      return Profile2D<V1, V2, V3, W>(std::move(model), "", "", "", "");
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return any entity with a Fill method (*lazy action*)
   /// \tparam BranchTypes The types of the branches the values of which are used to fill the object.
   /// \param[in] model The model to be considered to build the new return value.
   /// \param[in] bl The name of the branches read to fill the object.
   ///
   /// The returned object is independent of the input one.
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TResultProxy documentation.
   /// The user gives up ownership of the model object.
   /// It is compulsory to express the branches to be considered.
   template <typename FirstBranch, typename... OtherBranches, typename T> // need FirstBranch to disambiguate overloads
   TResultProxy<T> Fill(T &&model, const ColumnNames_t &bl)
   {
      auto h = std::make_shared<T>(std::move(model));
      if (!TDFInternal::HistoUtils<T>::HasAxisLimits(*h)) {
         throw std::runtime_error("The absence of axes limits is not supported yet.");
      }
      return CreateAction<TDFInternal::ActionTypes::Fill, FirstBranch, OtherBranches...>(bl, h);
   }

   template <typename T>
   TResultProxy<T> Fill(T &&model, const ColumnNames_t &bl)
   {
      auto h = std::make_shared<T>(std::move(model));
      if (!TDFInternal::HistoUtils<T>::HasAxisLimits(*h)) {
         throw std::runtime_error("The absence of axes limits is not supported yet.");
      }
      return CreateAction<TDFInternal::ActionTypes::Fill, TDFDetail::TInferType>(bl, h);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return the minimum of processed branch values (*lazy action*)
   /// \tparam T The type of the branch.
   /// \param[in] branchName The name of the branch to be treated.
   ///
   /// If no branch type is specified, the implementation will try to guess one.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TResultProxy documentation.
   template <typename T = TDFDetail::TInferType>
   TResultProxy<double> Min(const std::string &branchName = "")
   {
      auto bl = GetBranchNames<T>({branchName}, "calculate the minimum");
      auto minV = std::make_shared<double>(std::numeric_limits<double>::max());
      return CreateAction<TDFInternal::ActionTypes::Min, T>(bl, minV);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return the maximum of processed branch values (*lazy action*)
   /// \tparam T The type of the branch.
   /// \param[in] branchName The name of the branch to be treated.
   ///
   /// If no branch type is specified, the implementation will try to guess one.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TResultProxy documentation.
   template <typename T = TDFDetail::TInferType>
   TResultProxy<double> Max(const std::string &branchName = "")
   {
      auto bl = GetBranchNames<T>({branchName}, "calculate the maximum");
      auto maxV = std::make_shared<double>(std::numeric_limits<double>::min());
      return CreateAction<TDFInternal::ActionTypes::Max, T>(bl, maxV);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return the mean of processed branch values (*lazy action*)
   /// \tparam T The type of the branch.
   /// \param[in] branchName The name of the branch to be treated.
   ///
   /// If no branch type is specified, the implementation will try to guess one.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TResultProxy documentation.
   template <typename T = TDFDetail::TInferType>
   TResultProxy<double> Mean(const std::string &branchName = "")
   {
      auto bl = GetBranchNames<T>({branchName}, "calculate the mean");
      auto meanV = std::make_shared<double>(0);
      return CreateAction<TDFInternal::ActionTypes::Mean, T>(bl, meanV);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Print filtering statistics on screen
   ///
   /// Calling `Report` on the main `TDataFrame` object prints stats for
   /// all named filters in the call graph. Calling this method on a
   /// stored chain state (i.e. a graph node different from the first) prints
   /// the stats for all named filters in the chain section between the original
   /// `TDataFrame` and that node (included). Stats are printed in the same
   /// order as the named filters have been added to the graph.
   void Report()
   {
      auto df = GetDataFrameChecked();
      if (!df->HasRunAtLeastOnce()) df->Run();
      fProxiedPtr->Report();
   }

private:
   inline const char *GetNodeTypeName() { return ""; };

   /// Returns the default branches if needed, takes care of the error handling.
   template <typename T1, typename T2 = void, typename T3 = void, typename T4 = void>
   ColumnNames_t GetBranchNames(ColumnNames_t bl, const std::string &actionNameForErr)
   {
      constexpr auto isT2Void = std::is_same<T2, void>::value;
      constexpr auto isT3Void = std::is_same<T3, void>::value;
      constexpr auto isT4Void = std::is_same<T4, void>::value;

      unsigned int neededBranches = 1 + !isT2Void + !isT3Void + !isT4Void;

      unsigned int providedBranches = 0;
      std::for_each(bl.begin(), bl.end(), [&providedBranches](const std::string &s) {
         if (!s.empty()) providedBranches++;
      });

      if (neededBranches == providedBranches) return bl;

      return GetDefaultBranchNames(neededBranches, actionNameForErr);
   }

   /// \cond HIDDEN_SYMBOLS

   /****** BuildAndBook overloads *******/
   // BuildAndBook builds a TAction with the right operation and book it with the TLoopManager

   // Generic filling (covers Histo2D, Histo3D, Profile1D and Profile2D actions, with and without weights)
   template <typename... BranchTypes, typename ActionType, typename ActionResultType>
   void BuildAndBook(const ColumnNames_t &bl, const std::shared_ptr<ActionResultType> &h, unsigned int nSlots,
                     ActionType *)
   {
      using Op_t = TDFInternal::FillTOHelper<ActionResultType>;
      using DFA_t = TDFInternal::TAction<Op_t, Proxied, TDFInternal::TTypeList<BranchTypes...>>;
      auto df = GetDataFrameChecked();
      df->Book(std::make_shared<DFA_t>(Op_t(h, nSlots), bl, *fProxiedPtr));
   }

   // Histo1D filling (must handle the special case of distinguishing FillTOHelper and FillHelper
   template <typename... BranchTypes>
   void BuildAndBook(const ColumnNames_t &bl, const std::shared_ptr<::TH1F> &h, unsigned int nSlots,
                     TDFInternal::ActionTypes::Histo1D *)
   {
      auto df = GetDataFrameChecked();
      auto hasAxisLimits = TDFInternal::HistoUtils<::TH1F>::HasAxisLimits(*h);

      if (hasAxisLimits) {
         using Op_t = TDFInternal::FillTOHelper<::TH1F>;
         using DFA_t = TDFInternal::TAction<Op_t, Proxied, TDFInternal::TTypeList<BranchTypes...>>;
         df->Book(std::make_shared<DFA_t>(Op_t(h, nSlots), bl, *fProxiedPtr));
      } else {
         using Op_t = TDFInternal::FillHelper;
         using DFA_t = TDFInternal::TAction<Op_t, Proxied, TDFInternal::TTypeList<BranchTypes...>>;
         df->Book(std::make_shared<DFA_t>(Op_t(h, nSlots), bl, *fProxiedPtr));
      }
   }

   // Min action
   template <typename BranchType>
   void BuildAndBook(const ColumnNames_t &bl, const std::shared_ptr<double> &minV, unsigned int nSlots,
                     TDFInternal::ActionTypes::Min *)
   {
      using Op_t = TDFInternal::MinHelper;
      using DFA_t = TDFInternal::TAction<Op_t, Proxied, TDFInternal::TTypeList<BranchType>>;
      auto df = GetDataFrameChecked();
      df->Book(std::make_shared<DFA_t>(Op_t(minV, nSlots), bl, *fProxiedPtr));
   }

   // Max action
   template <typename BranchType>
   void BuildAndBook(const ColumnNames_t &bl, const std::shared_ptr<double> &maxV, unsigned int nSlots,
                     TDFInternal::ActionTypes::Max *)
   {
      using Op_t = TDFInternal::MaxHelper;
      using DFA_t = TDFInternal::TAction<Op_t, Proxied, TDFInternal::TTypeList<BranchType>>;
      auto df = GetDataFrameChecked();
      df->Book(std::make_shared<DFA_t>(Op_t(maxV, nSlots), bl, *fProxiedPtr));
   }

   // Mean action
   template <typename BranchType>
   void BuildAndBook(const ColumnNames_t &bl, const std::shared_ptr<double> &meanV, unsigned int nSlots,
                     TDFInternal::ActionTypes::Mean *)
   {
      using Op_t = TDFInternal::MeanHelper;
      using DFA_t = TDFInternal::TAction<Op_t, Proxied, TDFInternal::TTypeList<BranchType>>;
      auto df = GetDataFrameChecked();
      df->Book(std::make_shared<DFA_t>(Op_t(meanV, nSlots), bl, *fProxiedPtr));
   }
   /****** end BuildAndBook ******/
   /// \endcond

   // Type was specified by the user, no need to infer it
   template <typename ActionType, typename... BranchTypes, typename ActionResultType,
             typename std::enable_if<!TDFInternal::TNeedJitting<BranchTypes...>::value, int>::type = 0>
   TResultProxy<ActionResultType> CreateAction(const ColumnNames_t &bl, const std::shared_ptr<ActionResultType> &r)
   {
      auto df = GetDataFrameChecked();
      unsigned int nSlots = df->GetNSlots();
      BuildAndBook<BranchTypes...>(bl, r, nSlots, (ActionType *)nullptr);
      fProxiedPtr->IncrChildrenCount();
      return MakeResultProxy(r, df);
   }

   // User did not specify type, do type inference
   template <typename ActionType, typename... BranchTypes, typename ActionResultType,
             typename std::enable_if<TDFInternal::TNeedJitting<BranchTypes...>::value, int>::type = 0>
   TResultProxy<ActionResultType> CreateAction(const ColumnNames_t &bl, const std::shared_ptr<ActionResultType> &r)
   {
      auto df = GetDataFrameChecked();
      unsigned int nSlots = df->GetNSlots();
      const auto &tmpBranches = df->GetBookedBranches();
      auto tree = df->GetTree();
      TDFInternal::JitBuildAndBook(bl, GetNodeTypeName(), this, typeid(std::shared_ptr<ActionResultType>),
                                   typeid(ActionType), &r, *tree, nSlots, tmpBranches);
      fProxiedPtr->IncrChildrenCount();
      return MakeResultProxy(r, df);
   }

protected:
   /// Get the TLoopManager if reachable. If not, throw.
   std::shared_ptr<TLoopManager> GetDataFrameChecked()
   {
      auto df = fImplWeakPtr.lock();
      if (!df) {
         throw std::runtime_error("The main TDataFrame is not reachable: did it go out of scope?");
      }
      return df;
   }

   const ColumnNames_t GetDefaultBranchNames(unsigned int nExpectedBranches, const std::string &actionNameForErr)
   {
      auto df = GetDataFrameChecked();
      const ColumnNames_t &defaultBranches = df->GetDefaultBranches();
      const auto dBSize = defaultBranches.size();
      if (nExpectedBranches > dBSize) {
         std::string msg("Trying to deduce the branches from the default list in order to ");
         msg += actionNameForErr;
         msg += ". A set of branches of size ";
         msg += std::to_string(dBSize);
         msg += " was found. ";
         msg += std::to_string(nExpectedBranches);
         msg += 1 != nExpectedBranches ? " are" : " is";
         msg += " needed. Please specify the branches explicitly.";
         throw std::runtime_error(msg);
      }
      auto bnBegin = defaultBranches.begin();
      return ColumnNames_t(bnBegin, bnBegin + nExpectedBranches);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Implementation of snapshot
   /// \param[in] treename The name of the TTree
   /// \param[in] filename The name of the TFile
   /// \param[in] bnames The list of names of the branches to be written
   /// \param[in] filecacheMB The cache size of each memory file in MB (default = 16)
   /// The implementation exploits Foreach. The association of the addresses to
   /// the branches takes place at the first event. This is possible because
   /// since there are no copies, the address of the value passed by reference
   /// is the address pointing to the storage of the read/created object in/by
   /// the TTreeReaderValue/TemporaryBranch
   template <typename... Args, int... S>
   TInterface<TLoopManager> SnapshotImpl(const std::string &treename, const std::string &filename,
                                         const ColumnNames_t &bnames, Long_t filecacheMB,
                                         TDFInternal::TStaticSeq<S...> /*dummy*/)
   {
      const auto templateParamsN = sizeof...(S);
      const auto bNamesN = bnames.size();
      if (templateParamsN != bNamesN) {
         std::string err_msg = "The number of template parameters specified for the snapshot is ";
         err_msg += std::to_string(templateParamsN);
         err_msg += " while ";
         err_msg += std::to_string(bNamesN);
         err_msg += " branches have been specified.";
         throw std::runtime_error(err_msg.c_str());
      }

      if (!ROOT::IsImplicitMTEnabled()) {
         std::unique_ptr<TFile> ofile(TFile::Open(filename.c_str(), "RECREATE"));
         TTree t(treename.c_str(), treename.c_str());

         bool FirstEvent = true;
         auto fillTree = [&t, &bnames, &FirstEvent](Args &... args) {
            if (FirstEvent) {
               // hack to call TTree::Branch on all variadic template arguments
               std::initializer_list<int> expander = {(t.Branch(bnames[S].c_str(), &args), 0)..., 0};
               (void)expander; // avoid unused variable warnings for older compilers such as gcc 4.9
               FirstEvent = false;
            }
            t.Fill();
         };

         Foreach(fillTree, {bnames[S]...});
         t.Write();
      } else {
         auto df = GetDataFrameChecked();
         unsigned int nSlots = df->GetNSlots();
         auto cachesize = filecacheMB * 1024L * 1024L;
         TBufferMerger merger(filename.c_str(), "RECREATE");
         std::vector<std::shared_ptr<TBufferMergerFile>> files(nSlots);
         std::vector<TTree *> trees(nSlots);

         auto fillTree = [&](unsigned int slot, Args &... args) {
            if (!trees[slot]) {
               files[slot] = merger.GetFile();
               trees[slot] = new TTree(treename.c_str(), treename.c_str());
               trees[slot]->ResetBit(kMustCleanup);
               // hack to call TTree::Branch on all variadic template arguments
               std::initializer_list<int> expander = {(trees[slot]->Branch(bnames[S].c_str(), &args), 0)..., 0};
               (void)expander; // avoid unused variable warnings for older compilers such as gcc 4.9
            }
            trees[slot]->Fill();
            if (files[slot]->GetBytesWritten() >= cachesize) files[slot]->Write();
         };

         ForeachSlot(fillTree, {bnames[S]...});
         for (auto &&file : files) file->Write();
      }

      ::TDirectory::TContext ctxt;
      // Now we mimic a constructor for the TDataFrame. We cannot invoke it here
      // since this would introduce a cyclic headers dependency.
      TInterface<TLoopManager> snapshotTDF(std::make_shared<TLoopManager>(nullptr, bnames));
      auto chain = new TChain(treename.c_str());
      chain->Add(filename.c_str());
      snapshotTDF.fProxiedPtr->SetTree(std::shared_ptr<TTree>(static_cast<TTree *>(chain)));

      return snapshotTDF;
   }

   TInterface(const std::shared_ptr<Proxied> &proxied, const std::weak_ptr<TLoopManager> &impl)
      : fProxiedPtr(proxied), fImplWeakPtr(impl)
   {
   }

   /// Only enabled when building a TInterface<TLoopManager>
   template <typename T = Proxied, typename std::enable_if<std::is_same<T, TLoopManager>::value, int>::type = 0>
   TInterface(const std::shared_ptr<Proxied> &proxied) : fProxiedPtr(proxied), fImplWeakPtr(proxied->GetSharedPtr())
   {
   }

   std::shared_ptr<Proxied> fProxiedPtr;
   std::weak_ptr<TLoopManager> fImplWeakPtr;
};

template <>
inline const char *TInterface<TDFDetail::TFilterBase>::GetNodeTypeName()
{
   return "ROOT::Experimental::TDF::TInterface<ROOT::Detail::TDF::TFilterBase>";
}

template <>
inline const char *TInterface<TDFDetail::TCustomColumnBase>::GetNodeTypeName()
{
   return "ROOT::Experimental::TDF::TInterface<ROOT::Detail::TDF::TCustomColumnBase>";
}

template <>
inline const char *TInterface<TDFDetail::TLoopManager>::GetNodeTypeName()
{
   return "ROOT::Experimental::TDF::TInterface<ROOT::Detail::TDF::TLoopManager>";
}

template <>
inline const char *TInterface<TDFDetail::TRangeBase>::GetNodeTypeName()
{
   return "ROOT::Experimental::TDF::TInterface<ROOT::Detail::TDF::TRangeBase>";
}

} // end NS TDF
} // end NS Experimental
} // end NS ROOT

#endif // ROOT_TDF_INTERFACE
