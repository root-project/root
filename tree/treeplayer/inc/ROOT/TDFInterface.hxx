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

#include "ROOT/TResultProxy.hxx"
#include "ROOT/TDataSource.hxx"
#include "ROOT/TDFNodes.hxx"
#include "ROOT/TDFActionHelpers.hxx"
#include "ROOT/TDFHistoModels.hxx"
#include "ROOT/TDFUtils.hxx"
#include "RtypesCore.h" // for ULong64_t
#include "TChain.h"
#include "TH1.h" // For Histo actions
#include "TH2.h" // For Histo actions
#include "TH3.h" // For Histo actions
#include "TInterpreter.h"
#include "TProfile.h"   // For Histo actions
#include "TProfile2D.h" // For Histo actions
#include "TRegexp.h"
#include "TROOT.h" // IsImplicitMTEnabled
#include "TTreeReader.h"

#include <initializer_list>
#include <memory>
#include <string>
#include <sstream>
#include <typeinfo>
#include <type_traits> // is_same, enable_if

namespace ROOT {

namespace Experimental {
namespace TDF {
template <typename T>
class TInterface;

} // end NS TDF
} // end NS Experimental

namespace Internal {
namespace TDF {
using namespace ROOT::Experimental::TDF;
using namespace ROOT::Detail::TDF;

// CACHE --------------

template <typename T>
class CacheColumnHolder {
public:
   using value_type = T; // A shortcut to the value_type of the vector
   std::vector<value_type> fContent;
   // This method returns a pointer since we treat these columns as if they come
   // from a data source, i.e. we forward an entry point to valid memory rather
   // than a value.
   value_type *operator()(unsigned int /*slot*/, ULong64_t iEvent) { return &fContent[iEvent]; };
};

// ENDCACHE------------

/****** BuildAndBook overloads *******/
// BuildAndBook builds a TAction with the right operation and books it with the TLoopManager

// Generic filling (covers Histo2D, Histo3D, Profile1D and Profile2D actions, with and without weights)
template <typename... BranchTypes, typename ActionType, typename ActionResultType, typename PrevNodeType>
TActionBase *BuildAndBook(const ColumnNames_t &bl, const std::shared_ptr<ActionResultType> &h,
                          const unsigned int nSlots, TLoopManager &loopManager, PrevNodeType &prevNode, ActionType *)
{
   using Helper_t = FillTOHelper<ActionResultType>;
   using Action_t = TAction<Helper_t, PrevNodeType, TTraits::TypeList<BranchTypes...>>;
   auto action = std::make_shared<Action_t>(Helper_t(h, nSlots), bl, prevNode);
   loopManager.Book(action);
   return action.get();
}

// Histo1D filling (must handle the special case of distinguishing FillTOHelper and FillHelper
template <typename... BranchTypes, typename PrevNodeType>
TActionBase *BuildAndBook(const ColumnNames_t &bl, const std::shared_ptr<::TH1D> &h, const unsigned int nSlots,
                          TLoopManager &loopManager, PrevNodeType &prevNode, ActionTypes::Histo1D *)
{
   auto hasAxisLimits = HistoUtils<::TH1D>::HasAxisLimits(*h);

   TActionBase *actionBase;
   if (hasAxisLimits) {
      using Helper_t = FillTOHelper<::TH1D>;
      using Action_t = TAction<Helper_t, PrevNodeType, TTraits::TypeList<BranchTypes...>>;
      auto action = std::make_shared<Action_t>(Helper_t(h, nSlots), bl, prevNode);
      loopManager.Book(action);
      actionBase = action.get();
   } else {
      using Helper_t = FillHelper;
      using Action_t = TAction<Helper_t, PrevNodeType, TTraits::TypeList<BranchTypes...>>;
      auto action = std::make_shared<Action_t>(Helper_t(h, nSlots), bl, prevNode);
      loopManager.Book(action);
      actionBase = action.get();
   }

   return actionBase;
}

// Min action
template <typename BranchType, typename PrevNodeType, typename ActionResultType>
TActionBase *
BuildAndBook(const ColumnNames_t &bl, const std::shared_ptr<ActionResultType> &minV, const unsigned int nSlots,
             TLoopManager &loopManager, PrevNodeType &prevNode, ActionTypes::Min *)
{
   using Helper_t = MinHelper<ActionResultType>;
   using Action_t = TAction<Helper_t, PrevNodeType, TTraits::TypeList<BranchType>>;
   auto action = std::make_shared<Action_t>(Helper_t(minV, nSlots), bl, prevNode);
   loopManager.Book(action);
   return action.get();
}

// Max action
template <typename BranchType, typename PrevNodeType, typename ActionResultType>
TActionBase *
BuildAndBook(const ColumnNames_t &bl, const std::shared_ptr<ActionResultType> &maxV, const unsigned int nSlots,
             TLoopManager &loopManager, PrevNodeType &prevNode, ActionTypes::Max *)
{
   using Helper_t = MaxHelper<ActionResultType>;
   using Action_t = TAction<Helper_t, PrevNodeType, TTraits::TypeList<BranchType>>;
   auto action = std::make_shared<Action_t>(Helper_t(maxV, nSlots), bl, prevNode);
   loopManager.Book(action);
   return action.get();
}

// Sum action
template <typename BranchType, typename PrevNodeType, typename ActionResultType>
TActionBase *
BuildAndBook(const ColumnNames_t &bl, const std::shared_ptr<ActionResultType> &sumV, const unsigned int nSlots,
             TLoopManager &loopManager, PrevNodeType &prevNode, ActionTypes::Sum *)
{
   using Helper_t = SumHelper<ActionResultType>;
   using Action_t = TAction<Helper_t, PrevNodeType, TTraits::TypeList<BranchType>>;
   auto action = std::make_shared<Action_t>(Helper_t(sumV, nSlots), bl, prevNode);
   loopManager.Book(action);
   return action.get();
}

// Mean action
template <typename BranchType, typename PrevNodeType>
TActionBase *BuildAndBook(const ColumnNames_t &bl, const std::shared_ptr<double> &meanV, const unsigned int nSlots,
                          TLoopManager &loopManager, PrevNodeType &prevNode, ActionTypes::Mean *)
{
   using Helper_t = MeanHelper;
   using Action_t = TAction<Helper_t, PrevNodeType, TTraits::TypeList<BranchType>>;
   auto action = std::make_shared<Action_t>(Helper_t(meanV, nSlots), bl, prevNode);
   loopManager.Book(action);
   return action.get();
}
/****** end BuildAndBook ******/

std::vector<std::string> FindUsedColumnNames(std::string_view, TObjArray *, const std::vector<std::string> &);

using TmpBranchBasePtr_t = std::shared_ptr<TCustomColumnBase>;

Long_t JitTransformation(void *thisPtr, std::string_view methodName, std::string_view interfaceTypeName,
                         std::string_view name, std::string_view expression,
                         const std::map<std::string, std::string> &aliasMap, const ColumnNames_t &branches,
                         const std::vector<std::string> &customColumns,
                         const std::map<std::string, TmpBranchBasePtr_t> &tmpBookedBranches, TTree *tree,
                         std::string_view returnTypeName, TDataSource *ds);

std::string JitBuildAndBook(const ColumnNames_t &bl, const std::string &prevNodeTypename, void *prevNode,
                            const std::type_info &art, const std::type_info &at, const void *r, TTree *tree,
                            const unsigned int nSlots, const std::map<std::string, TmpBranchBasePtr_t> &customColumns,
                            TDataSource *ds, const std::shared_ptr<TActionBase *> *const actionPtrPtr);

// allocate a shared_ptr on the heap, return a reference to it. the user is responsible of deleting the shared_ptr*.
// this function is meant to only be used by TInterface's action methods, and should be deprecated as soon as we find
// a better way to make jitting work: the problem it solves is that we need to pass the same shared_ptr to the Helper
// object of each action and to the TResultProxy returned by the action. While the former is only instantiated when
// the event loop is about to start, the latter has to be returned to the user as soon as the action is booked.
// a heap allocated shared_ptr will stay alive long enough that at jitting time its address is still valid.
template <typename T>
std::shared_ptr<T> *MakeSharedOnHeap(const std::shared_ptr<T> &shPtr)
{
   return new std::shared_ptr<T>(shPtr);
}

bool AtLeastOneEmptyString(const std::vector<std::string_view> strings);

/* The following functions upcast shared ptrs to TFilter, TCustomColumn, TRange to their parent class (***Base).
 * Shared ptrs to TLoopManager are just copied, as well as shared ptrs to ***Base classes. */
std::shared_ptr<TFilterBase> UpcastNode(const std::shared_ptr<TFilterBase> ptr);
std::shared_ptr<TCustomColumnBase> UpcastNode(const std::shared_ptr<TCustomColumnBase> ptr);
std::shared_ptr<TRangeBase> UpcastNode(const std::shared_ptr<TRangeBase> ptr);
std::shared_ptr<TLoopManager> UpcastNode(const std::shared_ptr<TLoopManager> ptr);

ColumnNames_t GetValidatedColumnNames(TLoopManager &lm, const unsigned int nColumns, const ColumnNames_t &columns,
                                      const ColumnNames_t &validCustomColumns, TDataSource *ds);

std::vector<bool> FindUndefinedDSColumns(const ColumnNames_t &requestedCols, const ColumnNames_t &definedDSCols);

/// Helper function to be used by `DefineDataSourceColumns`
template <typename T>
void DefineDSColumnHelper(std::string_view name, TLoopManager &lm, TDataSource &ds)
{
   auto readers = ds.GetColumnReaders<T>(name);
   auto getValue = [readers](unsigned int slot) { return *readers[slot]; };
   using NewCol_t = TCustomColumn<decltype(getValue), TCCHelperTypes::TSlot>;
   lm.Book(std::make_shared<NewCol_t>(name, std::move(getValue), ColumnNames_t{}, &lm, /*isDSColumn=*/true));
   lm.AddDataSourceColumn(name);
}

/// Take a list of data-source column names and define the ones that haven't been defined yet.
template <typename... ColumnTypes, int... S>
void DefineDataSourceColumns(const std::vector<std::string> &columns, TLoopManager &lm, StaticSeq<S...>,
                             TTraits::TypeList<ColumnTypes...>, TDataSource &ds)
{
   const auto mustBeDefined = FindUndefinedDSColumns(columns, lm.GetCustomColumnNames());
   if (std::none_of(mustBeDefined.begin(), mustBeDefined.end(), [](bool b) { return b; })) {
      // no need to define any column
      return;
   } else {
      // hack to expand a template parameter pack without c++17 fold expressions.
      std::initializer_list<int> expander{
         (mustBeDefined[S] ? DefineDSColumnHelper<ColumnTypes>(columns[S], lm, ds) : /*no-op*/ ((void)0), 0)...};
      (void)expander; // avoid unused variable warnings
   }
}

/// Convenience function invoked by jitted code to build action nodes at runtime
template <typename ActionType, typename... BranchTypes, typename PrevNodeType, typename ActionResultType>
void CallBuildAndBook(PrevNodeType &prevNode, const ColumnNames_t &bl, const unsigned int nSlots,
                      const std::shared_ptr<ActionResultType> *rOnHeap,
                      const std::shared_ptr<TActionBase *> *actionPtrPtrOnHeap)
{
   // if we are here it means we are jitting, if we are jitting the loop manager must be alive
   auto &loopManager = *prevNode.GetImplPtr();
   using ColTypes_t = TypeList<BranchTypes...>;
   constexpr auto nColumns = ColTypes_t::list_size;
   auto ds = loopManager.GetDataSource();
   if (ds)
      DefineDataSourceColumns(bl, loopManager, GenStaticSeq_t<nColumns>(), ColTypes_t(), *ds);
   TActionBase *actionPtr =
      BuildAndBook<BranchTypes...>(bl, *rOnHeap, nSlots, loopManager, prevNode, (ActionType *)nullptr);
   **actionPtrPtrOnHeap = actionPtr;
   delete rOnHeap;
   delete actionPtrPtrOnHeap;
}

/// The contained `type` alias is `double` if `T == TInferType`, `U` if `T == std::container<U>`, `T` otherwise.
template <typename T, bool Container = TTraits::IsContainer<T>::value>
struct TMinReturnType {
   using type = T;
};

template <>
struct TMinReturnType<TDFDetail::TInferType, false> {
   using type = double;
};

template <typename T>
struct TMinReturnType<T, true> {
   using type = TTraits::TakeFirstParameter_t<T>;
};

/// The aliased type is `double` if `T == TInferType`, `U` if `T == container<U>`, `T` otherwise.
template <typename T>
using MinReturnType_t = typename TMinReturnType<T>::type;

template <typename T>
using MaxReturnType_t = MinReturnType_t<T>;

template <typename T>
using SumReturnType_t = MinReturnType_t<T>;

} // namespace TDF
} // namespace Internal

namespace Detail {
namespace TDF {

template <typename T, typename COLL = std::vector<T>>
struct TakeRealTypes {
   // We cannot put in the output collection C arrays: the ownership is not defined.
   // We therefore proceed to check if T is an TArrayBranch
   // If yes, we check what type is the output collection and we rebuild it.
   // E.g. if a vector<V> was the selected collection, where V is TArrayBranch<T>,
   // the collection becomes vector<vector<T>>.
   static constexpr auto isAB = TDFInternal::IsTArrayBranch_t<T>::value;
   using RealT_t = typename TDFInternal::ValueType<T>::value_type;
   using VTColl_t = std::vector<RealT_t>;

   using NewC0_t =
      typename std::conditional<isAB && TDFInternal::IsVector_t<COLL>::value, std::vector<VTColl_t>, COLL>::type;
   using NewC1_t =
      typename std::conditional<isAB && TDFInternal::IsList_t<NewC0_t>::value, std::list<VTColl_t>, NewC0_t>::type;
   using NewC2_t =
      typename std::conditional<isAB && TDFInternal::IsDeque_t<NewC1_t>::value, std::deque<VTColl_t>, NewC1_t>::type;
   using RealColl_t = NewC2_t;
};

} // namespace TDF
} // namespace Detail

namespace Experimental {

// forward declarations
class TDataFrame;

} // namespace Experimental
} // namespace ROOT

namespace cling {
std::string printValue(ROOT::Experimental::TDataFrame *tdf); // For a nice printing at the prompt
}

namespace ROOT {
namespace Experimental {
namespace TDF {
namespace TDFDetail = ROOT::Detail::TDF;
namespace TDFInternal = ROOT::Internal::TDF;
namespace TTraits = ROOT::TypeTraits;

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
   friend std::string cling::printValue(::ROOT::Experimental::TDataFrame *tdf); // For a nice printing at the prompt
   template <typename T>
   friend class TInterface;

   const std::shared_ptr<Proxied> fProxiedPtr;     ///< Smart pointer to the graph node encapsulated by this TInterface.
   const std::weak_ptr<TLoopManager> fImplWeakPtr; ///< Weak pointer to the TLoopManager at the root of the graph.
   ColumnNames_t fValidCustomColumns; ///< Names of columns `Define`d for this branch of the functional graph.
   /// Non-owning pointer to a data-source object. Null if no data-source. TLoopManager has ownership of the object.
   TDataSource *const fDataSource = nullptr;

public:
   /// \cond HIDDEN_SYMBOLS
   // Template conversion operator, meant to use to convert TInterfaces of certain node types to TInterfaces of base
   // classes of those node types, e.g. TInterface<TFilter<F,P>> -> TInterface<TFilterBase>.
   // It is used implicitly when a call to Filter or Define is jitted: the jitted call must convert the
   // TInterface returned by the jitted transformations to a TInterface<***Base> before returning.
   // Must be public because it is cling that uses it.
   template <typename NewProxied>
   operator TInterface<NewProxied>()
   {
      static_assert(std::is_base_of<NewProxied, Proxied>::value,
                    "TInterface<T> can only be converted to TInterface<BaseOfT>");
      return TInterface<NewProxied>(fProxiedPtr, fImplWeakPtr, fValidCustomColumns, fDataSource);
   }
   /// \endcond

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Copy-assignment operator for TInterface.
   TInterface &operator=(const TInterface &) = default;

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Copy-ctor for TInterface.
   TInterface(const TInterface &) = default;

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Move-ctor for TInterface.
   TInterface(TInterface &&) = default;

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Append a filter to the call graph.
   /// \param[in] f Function, lambda expression, functor class or any other callable object. It must return a `bool`
   /// signalling whether the event has passed the selection (true) or not (false).
   /// \param[in] columns Names of the columns/branches in input to the filter function.
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
   TInterface<TDFDetail::TFilter<F, Proxied>> Filter(F f, const ColumnNames_t &columns = {}, std::string_view name = "")
   {
      TDFInternal::CheckFilter(f);
      auto loopManager = GetDataFrameChecked();
      using ColTypes_t = typename TTraits::CallableTraits<F>::arg_types;
      constexpr auto nColumns = ColTypes_t::list_size;
      const auto validColumnNames =
         TDFInternal::GetValidatedColumnNames(*loopManager, nColumns, columns, fValidCustomColumns, fDataSource);
      if (fDataSource)
         TDFInternal::DefineDataSourceColumns(validColumnNames, *loopManager, TDFInternal::GenStaticSeq_t<nColumns>(),
                                              ColTypes_t(), *fDataSource);
      using F_t = TDFDetail::TFilter<F, Proxied>;
      auto FilterPtr = std::make_shared<F_t>(std::move(f), validColumnNames, *fProxiedPtr, name);
      loopManager->Book(FilterPtr);
      return TInterface<F_t>(FilterPtr, fImplWeakPtr, fValidCustomColumns, fDataSource);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Append a filter to the call graph.
   /// \param[in] f Function, lambda expression, functor class or any other callable object. It must return a `bool`
   /// signalling whether the event has passed the selection (true) or not (false).
   /// \param[in] name Optional name of this filter. See `Report`.
   ///
   /// Refer to the first overload of this method for the full documentation.
   template <typename F, typename std::enable_if<!std::is_convertible<F, std::string>::value, int>::type = 0>
   TInterface<TDFDetail::TFilter<F, Proxied>> Filter(F f, std::string_view name)
   {
      // The sfinae is there in order to pick up the overloaded method which accepts two strings
      // rather than this template method.
      return Filter(f, {}, name);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Append a filter to the call graph.
   /// \param[in] f Function, lambda expression, functor class or any other callable object. It must return a `bool`
   /// signalling whether the event has passed the selection (true) or not (false).
   /// \param[in] columns Names of the columns/branches in input to the filter function.
   ///
   /// Refer to the first overload of this method for the full documentation.
   template <typename F>
   TInterface<TDFDetail::TFilter<F, Proxied>> Filter(F f, const std::initializer_list<std::string> &columns)
   {
      return Filter(f, ColumnNames_t{columns});
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Append a filter to the call graph.
   /// \param[in] expression The filter expression in C++
   /// \param[in] name Optional name of this filter. See `Report`.
   ///
   /// The expression is just-in-time compiled and used to filter entries. It must
   /// be valid C++ syntax in which variable names are substituted with the names
   /// of branches/columns.
   ///
   /// Refer to the first overload of this method for the full documentation.
   TInterface<TFilterBase> Filter(std::string_view expression, std::string_view name = "")
   {
      auto retVal = CallJitTransformation("Filter", name, expression, "ROOT::Detail::TDF::TFilterBase");
      return *(TInterface<TFilterBase> *)retVal;
   }

   // clang-format off
   ////////////////////////////////////////////////////////////////////////////
   /// \brief Creates a custom column
   /// \param[in] name The name of the custom column.
   /// \param[in] expression Function, lambda expression, functor class or any other callable object producing the temporary value. Returns the value that will be assigned to the custom column.
   /// \param[in] columns Names of the columns/branches in input to the producer function.
   ///
   /// Create a custom column that will be visible from all subsequent nodes
   /// of the functional chain. The `expression` is only evaluated for entries that pass
   /// all the preceding filters.
   /// A new variable is created called `name`, accessible as if it was contained
   /// in the dataset from subsequent transformations/actions.
   ///
   /// Use cases include:
   ///
   /// * caching the results of complex calculations for easy and efficient multiple access
   /// * extraction of quantities of interest from complex objects
   /// * column aliasing, i.e. changing the name of a branch/column
   ///
   /// An exception is thrown if the name of the new column is already in use.
   template <typename F, typename std::enable_if<!std::is_convertible<F, std::string>::value, int>::type = 0>
   TInterface<Proxied> Define(std::string_view name, F expression, const ColumnNames_t &columns = {})
   {
      return DefineImpl<F, TDFDetail::TCCHelperTypes::TNothing>(name, std::move(expression), columns);
   }
   // clang-format on

   // clang-format off
   ////////////////////////////////////////////////////////////////////////////
   /// \brief Creates a custom column with a value dependent on the processing slot.
   /// \param[in] name The name of the custom column.
   /// \param[in] expression Function, lambda expression, functor class or any other callable object producing the temporary value. Returns the value that will be assigned to the custom column.
   /// \param[in] columns Names of the columns/branches in input to the producer function (excluding the slot number).
   ///
   /// This alternative implementation of `Define` is meant as a helper in writing thread-safe custom columns.
   /// The expression must be a callable of signature R(unsigned int, T1, T2, ...) where `T1, T2...` are the types
   /// of the columns that the expression takes as input. The first parameter is reserved for an unsigned integer
   /// representing a "slot number". TDataFrame guarantees that different threads will invoke the expression with
   /// different slot numbers - slot numbers will range from zero to ROOT::GetImplicitMTPoolSize()-1.
   ///
   /// The following two calls are equivalent, although `DefineSlot` is slightly more performant:
   /// ~~~{.cpp}
   /// int function(unsigned int, double, double);
   /// Define("x", function, {"tdfslot_", "column1", "column2"})
   /// DefineSlot("x", function, {"column1", "column2"})
   /// ~~~
   ///
   /// See Define for more information.
   template <typename F>
   TInterface<Proxied> DefineSlot(std::string_view name, F expression, const ColumnNames_t &columns = {})
   {
      return DefineImpl<F, TDFDetail::TCCHelperTypes::TSlot>(name, std::move(expression), columns);
   }
   // clang-format on

   // clang-format off
   ////////////////////////////////////////////////////////////////////////////
   /// \brief Creates a custom column with a value dependent on the processing slot and the current entry.
   /// \param[in] name The name of the custom column.
   /// \param[in] expression Function, lambda expression, functor class or any other callable object producing the temporary value. Returns the value that will be assigned to the custom column.
   /// \param[in] columns Names of the columns/branches in input to the producer function (excluding slot and entry).
   ///
   /// This alternative implementation of `Define` is meant as a helper in writing entry-specific, thread-safe custom
   /// columns. The expression must be a callable of signature R(unsigned int, ULong64_t, T1, T2, ...) where `T1, T2...`
   /// are the types of the columns that the expression takes as input. The first parameter is reserved for an unsigned
   /// integer representing a "slot number". TDataFrame guarantees that different threads will invoke the expression with
   /// different slot numbers - slot numbers will range from zero to ROOT::GetImplicitMTPoolSize()-1. The second parameter
   /// is reserved for a `ULong64_t` representing the current entry being processed by the current thread.
   ///
   /// The following two `Define`s are equivalent, although `DefineSlotEntry` is slightly more performant:
   /// ~~~{.cpp}
   /// int function(unsigned int, ULong64_t, double, double);
   /// Define("x", function, {"tdfslot_", "tdfentry_", "column1", "column2"})
   /// DefineSlotEntry("x", function, {"column1", "column2"})
   /// ~~~
   ///
   /// See Define for more information.
   template <typename F>
   TInterface<Proxied> DefineSlotEntry(std::string_view name, F expression, const ColumnNames_t &columns = {})
   {
      return DefineImpl<F, TDFDetail::TCCHelperTypes::TSlotAndEntry>(name, std::move(expression), columns);
   }
   // clang-format on

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Creates a custom column
   /// \param[in] name The name of the custom column.
   /// \param[in] expression An expression in C++ which represents the temporary value
   ///
   /// The expression is just-in-time compiled and used to produce the column entries. The
   /// It must be valid C++ syntax in which variable names are substituted with the names
   /// of branches/columns.
   ///
   /// Refer to the first overload of this method for the full documentation.
   TInterface<TTraits::TakeFirstParameter_t<decltype(TDFInternal::UpcastNode(fProxiedPtr))>>
   Define(std::string_view name, std::string_view expression)
   {
      auto loopManager = GetDataFrameChecked();
      // this check must be done before jitting lest we throw exceptions in jitted code
      TDFInternal::CheckCustomColumn(name, loopManager->GetTree(), loopManager->GetCustomColumnNames(),
                                     fDataSource ? fDataSource->GetColumnNames() : ColumnNames_t{});
      using retType = TInterface<TTraits::TakeFirstParameter_t<decltype(TDFInternal::UpcastNode(fProxiedPtr))>>;
      auto retVal = CallJitTransformation("Define", name, expression, retType::GetNodeTypeName());
      auto retInterface = reinterpret_cast<retType *>(retVal);
      return *retInterface;
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Allow to refer to a column with a different name
   /// \param[in] alias name of the column alias
   /// \param[in] columnName of the column to be aliased
   /// Aliasing an alias is supported.
   TInterface<Proxied> Alias(std::string_view alias, std::string_view columnName)
   {
      // The symmetry with Define is clear. We want to:
      // - Create globally the alias and return this very node, unchanged
      // - Make aliases accessible based on chains and not globally
      auto loopManager = GetDataFrameChecked();

      // Helper to find out if a name is a column
      auto &dsColumnNames = fDataSource ? fDataSource->GetColumnNames() : ColumnNames_t{};

      // If the alias name is a column name, there is a problem
      TDFInternal::CheckCustomColumn(alias, loopManager->GetTree(), fValidCustomColumns, dsColumnNames);

      const auto validColumnName = TDFInternal::GetValidatedColumnNames(*loopManager, 1, {std::string(columnName)},
                                                                        fValidCustomColumns, fDataSource)[0];

      loopManager->AddColumnAlias(std::string(alias), validColumnName);
      TInterface<Proxied> newInterface(fProxiedPtr, fImplWeakPtr, fValidCustomColumns, fDataSource);
      newInterface.fValidCustomColumns.emplace_back(alias);
      return newInterface;
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Save selected columns to disk, in a new TTree `treename` in file `filename`.
   /// \tparam BranchTypes variadic list of branch/column types
   /// \param[in] treename The name of the output TTree
   /// \param[in] filename The name of the output TFile
   /// \param[in] columnList The list of names of the columns/branches to be written
   /// \param[in] options TSnapshotOptions struct with extra options to pass to TFile and TTree
   ///
   /// This function returns a `TDataFrame` built with the output tree as a source.
   template <typename... BranchTypes>
   TInterface<TLoopManager>
   Snapshot(std::string_view treename, std::string_view filename, const ColumnNames_t &columnList,
            const TSnapshotOptions &options = TSnapshotOptions())
   {
      return SnapshotImpl<BranchTypes...>(treename, filename, columnList, options);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Save selected columns to disk, in a new TTree `treename` in file `filename`.
   /// \param[in] treename The name of the output TTree
   /// \param[in] filename The name of the output TFile
   /// \param[in] columnList The list of names of the columns/branches to be written
   /// \param[in] options TSnapshotOptions struct with extra options to pass to TFile and TTree
   ///
   /// This function returns a `TDataFrame` built with the output tree as a source.
   /// The types of the columns are automatically inferred and do not need to be specified.
   TInterface<TLoopManager> Snapshot(std::string_view treename, std::string_view filename,
                                     const ColumnNames_t &columnList,
                                     const TSnapshotOptions &options = TSnapshotOptions())
   {
      // Early return: if the list of columns is empty, just return an empty TDF
      // If we proceed, the jitted call will not compile!
      if (columnList.empty()) {
         auto nEntries = *this->Count();
         TInterface<TLoopManager> emptyTDF(std::make_shared<TLoopManager>(nEntries));
         return emptyTDF;
      }
      auto df = GetDataFrameChecked();
      auto tree = df->GetTree();
      std::stringstream snapCall;
      auto upcastNode = TDFInternal::UpcastNode(fProxiedPtr);
      TInterface<TTraits::TakeFirstParameter_t<decltype(upcastNode)>> upcastInterface(fProxiedPtr, fImplWeakPtr,
                                                                                      fValidCustomColumns, fDataSource);
      // build a string equivalent to
      // "(TInterface<nodetype*>*)(this)->Snapshot<Ts...>(treename,filename,*(ColumnNames_t*)(&columnList), options)"
      snapCall << "reinterpret_cast<ROOT::Experimental::TDF::TInterface<" << upcastInterface.GetNodeTypeName() << ">*>("
               << &upcastInterface << ")->Snapshot<";
      bool first = true;
      for (auto &b : columnList) {
         if (!first)
            snapCall << ", ";
         snapCall << TDFInternal::ColumnName2ColumnTypeName(b, tree, df->GetBookedBranch(b), fDataSource);
         first = false;
      };
      snapCall << ">(\"" << treename << "\", \"" << filename << "\", "
               << "*reinterpret_cast<std::vector<std::string>*>(" // vector<string> should be ColumnNames_t
               << &columnList << "),"
               << "*reinterpret_cast<ROOT::Experimental::TDF::TSnapshotOptions*>(" << &options << "));";
      // jit snapCall, return result
      TInterpreter::EErrorCode errorCode;
      auto newTDFPtr = gInterpreter->Calc(snapCall.str().c_str(), &errorCode);
      if (TInterpreter::EErrorCode::kNoError != errorCode) {
         std::string msg = "Cannot jit Snapshot call. Interpreter error code is " + std::to_string(errorCode) + ".";
         throw std::runtime_error(msg);
      }
      return *reinterpret_cast<TInterface<TLoopManager> *>(newTDFPtr);
   }

   // clang-format off
   ////////////////////////////////////////////////////////////////////////////
   /// \brief Save selected columns to disk, in a new TTree `treename` in file `filename`.
   /// \param[in] treename The name of the output TTree
   /// \param[in] filename The name of the output TFile
   /// \param[in] columnNameRegexp The regular expression to match the column names to be selected. The presence of a '^' and a '$' at the end of the string is implicitly assumed if they are not specified. See the documentation of TRegexp for more details. An empty string signals the selection of all columns.
   /// \param[in] options TSnapshotOptions struct with extra options to pass to TFile and TTree
   ///
   /// This function returns a `TDataFrame` built with the output tree as a source.
   /// The types of the columns are automatically inferred and do not need to be specified.
   TInterface<TLoopManager> Snapshot(std::string_view treename, std::string_view filename,
                                     std::string_view columnNameRegexp = "",
                                     const TSnapshotOptions &options = TSnapshotOptions())
   {
      auto selectedColumns = ConvertRegexToColumns(columnNameRegexp);
      return Snapshot(treename, filename, selectedColumns, options);
   }
   // clang-format on

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Save selected columns in memory
   /// \param[in] columns to be cached in memory
   ///
   /// The content of the selected columns is saved in memory exploiting the functionality offered by
   /// the Take action. No extra copy is carried out when serving cached data to the actions and
   /// transformations requesting it.
   template <typename... BranchTypes>
   TInterface<TLoopManager> Cache(const ColumnNames_t &columnList)
   {
      auto staticSeq = TDFInternal::GenStaticSeq_t<sizeof...(BranchTypes)>();
      return CacheImpl<BranchTypes...>(columnList, staticSeq);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Save selected columns in memory
   /// \param[in] columns to be cached in memory
   ///
   /// The content of the selected columns is saved in memory exploiting the functionality offered by
   /// the Take action. No extra copy is carried out when serving cached data to the actions and
   /// transformations requesting it.
   TInterface<TLoopManager> Cache(const ColumnNames_t &columnList)
   {
      // Early return: if the list of columns is empty, just return an empty TDF
      // If we proceed, the jitted call will not compile!
      if (columnList.empty()) {
         auto nEntries = *this->Count();
         TInterface<TLoopManager> emptyTDF(std::make_shared<TLoopManager>(nEntries));
         return emptyTDF;
      }

      auto df = GetDataFrameChecked();
      auto tree = df->GetTree();
      std::stringstream snapCall;
      auto upcastNode = TDFInternal::UpcastNode(fProxiedPtr);
      TInterface<TTraits::TakeFirstParameter_t<decltype(upcastNode)>> upcastInterface(fProxiedPtr, fImplWeakPtr,
                                                                                      fValidCustomColumns, fDataSource);
      // build a string equivalent to
      // "(TInterface<nodetype*>*)(this)->Cache<Ts...>(*(ColumnNames_t*)(&columnList))"
      snapCall << "reinterpret_cast<ROOT::Experimental::TDF::TInterface<" << upcastInterface.GetNodeTypeName() << ">*>("
               << &upcastInterface << ")->Cache<";
      bool first = true;
      for (auto &b : columnList) {
         if (!first)
            snapCall << ", ";
         snapCall << TDFInternal::ColumnName2ColumnTypeName(b, tree, df->GetBookedBranch(b), fDataSource);
         first = false;
      };
      snapCall << ">(*reinterpret_cast<std::vector<std::string>*>(" // vector<string> should be ColumnNames_t
               << &columnList << "));";
      // jit snapCall, return result
      TInterpreter::EErrorCode errorCode;
      auto newTDFPtr = gInterpreter->Calc(snapCall.str().c_str(), &errorCode);
      if (TInterpreter::EErrorCode::kNoError != errorCode) {
         std::string msg = "Cannot jit Cache call. Interpreter error code is " + std::to_string(errorCode) + ".";
         throw std::runtime_error(msg);
      }
      return *reinterpret_cast<TInterface<TLoopManager> *>(newTDFPtr);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Save selected columns in memory
   /// \param[in] a regular expression to select the columns
   ///
   /// The existing columns are matched against the regeular expression. If the string provided
   /// is empty, all columns are selected.
   TInterface<TLoopManager> Cache(std::string_view columnNameRegexp = "")
   {
      auto selectedColumns = ConvertRegexToColumns(columnNameRegexp);
      return Cache(selectedColumns);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Creates a node that filters entries based on range
   /// \param[in] start How many entries to discard before resuming processing.
   /// \param[in] stop Total number of entries that will be processed before stopping. 0 means "never stop".
   /// \param[in] stride Process one entry every `stride` entries. Must be strictly greater than 0.
   ///
   /// Ranges are only available if EnableImplicitMT has _not_ been called. Multi-thread ranges are not supported.
   TInterface<TDFDetail::TRange<Proxied>> Range(unsigned int start, unsigned int stop, unsigned int stride = 1)
   {
      // check invariants
      if (stride == 0 || (stop != 0 && stop < start))
         throw std::runtime_error("Range: stride must be strictly greater than 0 and stop must be greater than start.");
      if (ROOT::IsImplicitMTEnabled())
         throw std::runtime_error("Range was called with ImplicitMT enabled. Multi-thread ranges are not supported.");

      auto df = GetDataFrameChecked();
      using Range_t = TDFDetail::TRange<Proxied>;
      auto RangePtr = std::make_shared<Range_t>(start, stop, stride, *fProxiedPtr);
      df->Book(RangePtr);
      TInterface<TDFDetail::TRange<Proxied>> tdf_r(RangePtr, fImplWeakPtr, fValidCustomColumns, fDataSource);
      return tdf_r;
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Creates a node that filters entries based on range
   /// \param[in] stop Total number of entries that will be processed before stopping. 0 means "never stop".
   ///
   /// See the other Range overload for a detailed description.
   TInterface<TDFDetail::TRange<Proxied>> Range(unsigned int stop) { return Range(0, stop, 1); }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Execute a user-defined function on each entry (*instant action*)
   /// \param[in] f Function, lambda expression, functor class or any other callable object performing user defined
   /// calculations.
   /// \param[in] columns Names of the columns/branches in input to the user function.
   ///
   /// The callable `f` is invoked once per entry. This is an *instant action*:
   /// upon invocation, an event loop as well as execution of all scheduled actions
   /// is triggered.
   /// Users are responsible for the thread-safety of this callable when executing
   /// with implicit multi-threading enabled (i.e. ROOT::EnableImplicitMT).
   template <typename F>
   void Foreach(F f, const ColumnNames_t &columns = {})
   {
      using arg_types = typename TTraits::CallableTraits<decltype(f)>::arg_types_nodecay;
      using ret_type = typename TTraits::CallableTraits<decltype(f)>::ret_type;
      ForeachSlot(TDFInternal::AddSlotParameter<ret_type>(f, arg_types()), columns);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Execute a user-defined function requiring a processing slot index on each entry (*instant action*)
   /// \param[in] f Function, lambda expression, functor class or any other callable object performing user defined
   /// calculations.
   /// \param[in] columns Names of the columns/branches in input to the user function.
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
   void ForeachSlot(F f, const ColumnNames_t &columns = {})
   {
      auto loopManager = GetDataFrameChecked();
      using ColTypes_t = TypeTraits::RemoveFirstParameter_t<typename TTraits::CallableTraits<F>::arg_types>;
      constexpr auto nColumns = ColTypes_t::list_size;
      const auto validColumnNames =
         TDFInternal::GetValidatedColumnNames(*loopManager, nColumns, columns, fValidCustomColumns, fDataSource);
      if (fDataSource)
         TDFInternal::DefineDataSourceColumns(validColumnNames, *loopManager, TDFInternal::GenStaticSeq_t<nColumns>(),
                                              ColTypes_t(), *fDataSource);
      using Helper_t = TDFInternal::ForeachSlotHelper<F>;
      using Action_t = TDFInternal::TAction<Helper_t, Proxied>;
      loopManager->Book(std::make_shared<Action_t>(Helper_t(std::move(f)), validColumnNames, *fProxiedPtr));
      loopManager->Run();
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Execute a user-defined reduce operation on the values of a column.
   /// \tparam F The type of the reduce callable. Automatically deduced.
   /// \tparam T The type of the column to apply the reduction to. Automatically deduced.
   /// \param[in] f A callable with signature `T(T,T)`
   /// \param[in] columnName The column to be reduced. If omitted, the first default column is used instead.
   ///
   /// A reduction takes two values of a column and merges them into one (e.g.
   /// by summing them, taking the maximum, etc). This action performs the
   /// specified reduction operation on all processed column values, returning
   /// a single value of the same type. The callable f must satisfy the general
   /// requirements of a *processing function* besides having signature `T(T,T)`
   /// where `T` is the type of column columnName.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TResultProxy documentation.
   template <typename F, typename T = typename TTraits::CallableTraits<F>::ret_type>
   TResultProxy<T> Reduce(F f, std::string_view columnName = "")
   {
      static_assert(std::is_default_constructible<T>::value,
                    "reduce object cannot be default-constructed. Please provide an initialisation value (initValue)");
      return Reduce(std::move(f), columnName, T());
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Execute a user-defined reduce operation on the values of a column.
   /// \tparam F The type of the reduce callable. Automatically deduced.
   /// \tparam T The type of the column to apply the reduction to. Automatically deduced.
   /// \param[in] f A callable with signature `T(T,T)`
   /// \param[in] columnName The column to be reduced. If omitted, the first default column is used instead.
   /// \param[in] initValue The reduced object is initialised to this value rather than being default-constructed.
   ///
   /// See the description of the first Reduce overload for more information.
   template <typename F, typename T = typename TTraits::CallableTraits<F>::ret_type>
   TResultProxy<T> Reduce(F f, std::string_view columnName, const T &initValue)
   {
      using arg_types = typename TTraits::CallableTraits<F>::arg_types;
      TDFInternal::CheckReduce(f, arg_types());
      auto loopManager = GetDataFrameChecked();
      const auto columns = columnName.empty() ? ColumnNames_t() : ColumnNames_t({std::string(columnName)});
      constexpr auto nColumns = arg_types::list_size;
      const auto validColumnNames =
         TDFInternal::GetValidatedColumnNames(*loopManager, 1, columns, fValidCustomColumns, fDataSource);
      if (fDataSource)
         TDFInternal::DefineDataSourceColumns(validColumnNames, *loopManager, TDFInternal::GenStaticSeq_t<nColumns>(),
                                              arg_types(), *fDataSource);
      auto redObjPtr = std::make_shared<T>(initValue);
      using Helper_t = TDFInternal::ReduceHelper<F, T>;
      using Action_t = typename TDFInternal::TAction<Helper_t, Proxied>;
      auto action = std::make_shared<Action_t>(Helper_t(std::move(f), redObjPtr, fProxiedPtr->GetNSlots()),
                                               validColumnNames, *fProxiedPtr);
      loopManager->Book(action);
      return MakeResultProxy(redObjPtr, loopManager, action.get());
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Execute a user-defined reduce operation on the values of the first default column.
   /// \tparam F The type of the reduce callable. Automatically deduced.
   /// \tparam T The type of the column to apply the reduction to. Automatically deduced.
   /// \param[in] f A callable with signature `T(T,T)`
   /// \param[in] initValue The reduced object is initialised to this value rather than being default-constructed
   ///
   /// See the description of the first Reduce overload for more information.
   template <typename F, typename T = typename TTraits::CallableTraits<F>::ret_type>
   TResultProxy<T> Reduce(F f, const T &initValue)
   {
      return Reduce(std::move(f), "", initValue);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return the number of entries processed (*lazy action*)
   ///
   /// Useful e.g. for counting the number of entries passing a certain filter (see also `Report`).
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TResultProxy documentation.
   TResultProxy<ULong64_t> Count()
   {
      auto df = GetDataFrameChecked();
      const auto nSlots = fProxiedPtr->GetNSlots();
      auto cSPtr = std::make_shared<ULong64_t>(0);
      using Helper_t = TDFInternal::CountHelper;
      using Action_t = TDFInternal::TAction<Helper_t, Proxied>;
      auto action = std::make_shared<Action_t>(Helper_t(cSPtr, nSlots), ColumnNames_t({}), *fProxiedPtr);
      df->Book(action);
      return MakeResultProxy(cSPtr, df, action.get());
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return a collection of values of a column (*lazy action*, returns a std::vector by default)
   /// \tparam T The type of the column.
   /// \tparam COLL The type of collection used to store the values.
   /// \param[in] column The name of the column to collect the values of.
   ///
   /// If the type of the column is TArrayBranch<T>, i.e. in the ROOT dataset this is
   /// a C-style array, the type stored in the return container is a std::vector<T> to
   /// guarantee the lifetime of the data involved.
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TResultProxy documentation.
   template <typename T, typename COLL = std::vector<T>>
   TResultProxy<typename TDFDetail::TakeRealTypes<T, COLL>::RealColl_t> Take(std::string_view column = "")
   {
      auto loopManager = GetDataFrameChecked();
      const auto columns = column.empty() ? ColumnNames_t() : ColumnNames_t({std::string(column)});
      const auto validColumnNames =
         TDFInternal::GetValidatedColumnNames(*loopManager, 1, columns, fValidCustomColumns, fDataSource);
      if (fDataSource)
         TDFInternal::DefineDataSourceColumns(validColumnNames, *loopManager, TDFInternal::GenStaticSeq_t<1>(),
                                              TTraits::TypeList<T>(), *fDataSource);

      using RealT_t = typename TDFDetail::TakeRealTypes<T, COLL>::RealT_t;
      using RealColl_t = typename TDFDetail::TakeRealTypes<T, COLL>::RealColl_t;

      using Helper_t = TDFInternal::TakeHelper<RealT_t, T, RealColl_t>;
      using Action_t = TDFInternal::TAction<Helper_t, Proxied>;
      auto valuesPtr = std::make_shared<RealColl_t>();
      const auto nSlots = fProxiedPtr->GetNSlots();
      auto action = std::make_shared<Action_t>(Helper_t(valuesPtr, nSlots), validColumnNames, *fProxiedPtr);
      loopManager->Book(action);
      return MakeResultProxy(valuesPtr, loopManager, action.get());
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a one-dimensional histogram with the values of a column (*lazy action*)
   /// \tparam V The type of the column used to fill the histogram.
   /// \param[in] model The returned histogram will be constructed using this as a model.
   /// \param[in] vName The name of the column that will fill the histogram.
   ///
   /// Columns can be of a container type (e.g. std::vector<double>), in which case the histogram
   /// is filled with each one of the elements of the container. In case multiple columns of container type
   /// are provided (e.g. values and weights) they must have the same length for each one of the events (but
   /// possibly different lengths between events).
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TResultProxy documentation.
   /// The user gives up ownership of the model histogram.
   template <typename V = TDFDetail::TInferType>
   TResultProxy<::TH1D> Histo1D(const TH1DModel &model = {"", "", 128u, 0., 0.}, std::string_view vName = "")
   {
      const auto userColumns = vName.empty() ? ColumnNames_t() : ColumnNames_t({std::string(vName)});
      std::shared_ptr<::TH1D> h(nullptr);
      {
         ROOT::Internal::TDF::TIgnoreErrorLevelRAII iel(kError);
         if (model.fBinXEdges.empty()) {
            h = std::make_shared<::TH1D>(model.fName, model.fTitle, model.fNbinsX, model.fXLow, model.fXUp);
         } else {
            h = std::make_shared<::TH1D>(model.fName, model.fTitle, model.fNbinsX, model.fBinXEdges.data());
         }
      }

      if (h->GetXaxis()->GetXmax() == h->GetXaxis()->GetXmin())
         TDFInternal::HistoUtils<::TH1D>::SetCanExtendAllAxes(*h);
      return CreateAction<TDFInternal::ActionTypes::Histo1D, V>(userColumns, h);
   }

   template <typename V = TDFDetail::TInferType>
   TResultProxy<::TH1D> Histo1D(std::string_view vName)
   {
      return Histo1D<V>({"", "", 128u, 0., 0.}, vName);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a one-dimensional histogram with the weighted values of a column (*lazy action*)
   /// \tparam V The type of the column used to fill the histogram.
   /// \tparam W The type of the column used as weights.
   /// \param[in] model The returned histogram will be constructed using this as a model.
   /// \param[in] vName The name of the column that will fill the histogram.
   /// \param[in] wName The name of the column that will provide the weights.
   ///
   /// See the description of the first Histo1D overload for more details.
   template <typename V = TDFDetail::TInferType, typename W = TDFDetail::TInferType>
   TResultProxy<::TH1D> Histo1D(const TH1DModel &model, std::string_view vName, std::string_view wName)
   {
      auto columnViews = {vName, wName};
      const auto userColumns = TDFInternal::AtLeastOneEmptyString(columnViews)
                                  ? ColumnNames_t()
                                  : ColumnNames_t(columnViews.begin(), columnViews.end());
      std::shared_ptr<::TH1D> h(nullptr);
      {
         ROOT::Internal::TDF::TIgnoreErrorLevelRAII iel(kError);
         if (model.fBinXEdges.empty()) {
            h = std::make_shared<::TH1D>(model.fName, model.fTitle, model.fNbinsX, model.fXLow, model.fXUp);
         } else {
            h = std::make_shared<::TH1D>(model.fName, model.fTitle, model.fNbinsX, model.fBinXEdges.data());
         }
      }
      return CreateAction<TDFInternal::ActionTypes::Histo1D, V, W>(userColumns, h);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a one-dimensional histogram with the weighted values of a column (*lazy action*)
   /// \tparam V The type of the column used to fill the histogram.
   /// \tparam W The type of the column used as weights.
   /// \param[in] vName The name of the column that will fill the histogram.
   /// \param[in] wName The name of the column that will provide the weights.
   ///
   /// This overload uses a default model histogram TH1D("", "", 128u, 0., 0.).
   /// See the description of the first Histo1D overload for more details.
   template <typename V = TDFDetail::TInferType, typename W = TDFDetail::TInferType>
   TResultProxy<::TH1D> Histo1D(std::string_view vName, std::string_view wName)
   {
      return Histo1D<V, W>({"", "", 128u, 0., 0.}, vName, wName);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a one-dimensional histogram with the weighted values of a column (*lazy action*)
   /// \tparam V The type of the column used to fill the histogram.
   /// \tparam W The type of the column used as weights.
   /// \param[in] model The returned histogram will be constructed using this as a model.
   ///
   /// This overload will use the first two default columns as column names.
   /// See the description of the first Histo1D overload for more details.
   template <typename V, typename W>
   TResultProxy<::TH1D> Histo1D(const TH1DModel &model = {"", "", 128u, 0., 0.})
   {
      return Histo1D<V, W>(model, "", "");
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a two-dimensional histogram (*lazy action*)
   /// \tparam V1 The type of the column used to fill the x axis of the histogram.
   /// \tparam V2 The type of the column used to fill the y axis of the histogram.
   /// \param[in] model The returned histogram will be constructed using this as a model.
   /// \param[in] v1Name The name of the column that will fill the x axis.
   /// \param[in] v2Name The name of the column that will fill the y axis.
   ///
   /// Columns can be of a container type (e.g. std::vector<double>), in which case the histogram
   /// is filled with each one of the elements of the container. In case multiple columns of container type
   /// are provided (e.g. values and weights) they must have the same length for each one of the events (but
   /// possibly different lengths between events).
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TResultProxy documentation.
   /// The user gives up ownership of the model histogram.
   template <typename V1 = TDFDetail::TInferType, typename V2 = TDFDetail::TInferType>
   TResultProxy<::TH2D> Histo2D(const TH2DModel &model, std::string_view v1Name = "", std::string_view v2Name = "")
   {
      std::shared_ptr<::TH2D> h(nullptr);
      {
         ROOT::Internal::TDF::TIgnoreErrorLevelRAII iel(kError);
         if (model.fBinXEdges.empty() && model.fBinYEdges.empty()) {
            h = std::make_shared<::TH2D>(model.fName, model.fTitle, model.fNbinsX, model.fXLow, model.fXUp,
                                         model.fNbinsY, model.fYLow, model.fYUp);
         } else if (!model.fBinXEdges.empty() && model.fBinYEdges.empty()) {
            h = std::make_shared<::TH2D>(model.fName, model.fTitle, model.fNbinsX, model.fBinXEdges.data(),
                                         model.fNbinsY, model.fYLow, model.fYUp);
         } else {
            h = std::make_shared<::TH2D>(model.fName, model.fTitle, model.fNbinsX, model.fBinXEdges.data(),
                                         model.fNbinsY, model.fBinYEdges.data());
         }
      }
      if (!TDFInternal::HistoUtils<::TH2D>::HasAxisLimits(*h)) {
         throw std::runtime_error("2D histograms with no axes limits are not supported yet.");
      }
      auto columnViews = {v1Name, v2Name};
      const auto userColumns = TDFInternal::AtLeastOneEmptyString(columnViews)
                                  ? ColumnNames_t()
                                  : ColumnNames_t(columnViews.begin(), columnViews.end());
      return CreateAction<TDFInternal::ActionTypes::Histo2D, V1, V2>(userColumns, h);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a weighted two-dimensional histogram (*lazy action*)
   /// \tparam V1 The type of the column used to fill the x axis of the histogram.
   /// \tparam V2 The type of the column used to fill the y axis of the histogram.
   /// \tparam W The type of the column used for the weights of the histogram.
   /// \param[in] model The returned histogram will be constructed using this as a model.
   /// \param[in] v1Name The name of the column that will fill the x axis.
   /// \param[in] v2Name The name of the column that will fill the y axis.
   /// \param[in] wName The name of the column that will provide the weights.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TResultProxy documentation.
   /// The user gives up ownership of the model histogram.
   template <typename V1 = TDFDetail::TInferType, typename V2 = TDFDetail::TInferType,
             typename W = TDFDetail::TInferType>
   TResultProxy<::TH2D>
   Histo2D(const TH2DModel &model, std::string_view v1Name, std::string_view v2Name, std::string_view wName)
   {
      std::shared_ptr<::TH2D> h(nullptr);
      {
         ROOT::Internal::TDF::TIgnoreErrorLevelRAII iel(kError);
         if (model.fBinXEdges.empty() && model.fBinYEdges.empty()) {
            h = std::make_shared<::TH2D>(model.fName, model.fTitle, model.fNbinsX, model.fXLow, model.fXUp,
                                         model.fNbinsY, model.fYLow, model.fYUp);
         } else if (!model.fBinXEdges.empty() && model.fBinYEdges.empty()) {
            h = std::make_shared<::TH2D>(model.fName, model.fTitle, model.fNbinsX, model.fBinXEdges.data(),
                                         model.fNbinsY, model.fYLow, model.fYUp);
         } else {
            h = std::make_shared<::TH2D>(model.fName, model.fTitle, model.fNbinsX, model.fBinXEdges.data(),
                                         model.fNbinsY, model.fBinYEdges.data());
         }
      }
      if (!TDFInternal::HistoUtils<::TH2D>::HasAxisLimits(*h)) {
         throw std::runtime_error("2D histograms with no axes limits are not supported yet.");
      }
      auto columnViews = {v1Name, v2Name, wName};
      const auto userColumns = TDFInternal::AtLeastOneEmptyString(columnViews)
                                  ? ColumnNames_t()
                                  : ColumnNames_t(columnViews.begin(), columnViews.end());
      return CreateAction<TDFInternal::ActionTypes::Histo2D, V1, V2, W>(userColumns, h);
   }

   template <typename V1, typename V2, typename W>
   TResultProxy<::TH2D> Histo2D(const TH2DModel &model)
   {
      return Histo2D<V1, V2, W>(model, "", "", "");
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a three-dimensional histogram (*lazy action*)
   /// \tparam V1 The type of the column used to fill the x axis of the histogram. Inferred if not present.
   /// \tparam V2 The type of the column used to fill the y axis of the histogram. Inferred if not present.
   /// \tparam V3 The type of the column used to fill the z axis of the histogram. Inferred if not present.
   /// \param[in] model The returned histogram will be constructed using this as a model.
   /// \param[in] v1Name The name of the column that will fill the x axis.
   /// \param[in] v2Name The name of the column that will fill the y axis.
   /// \param[in] v3Name The name of the column that will fill the z axis.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TResultProxy documentation.
   /// The user gives up ownership of the model histogram.
   template <typename V1 = TDFDetail::TInferType, typename V2 = TDFDetail::TInferType,
             typename V3 = TDFDetail::TInferType>
   TResultProxy<::TH3D> Histo3D(const TH3DModel &model, std::string_view v1Name = "", std::string_view v2Name = "",
                                std::string_view v3Name = "")
   {
      std::shared_ptr<::TH3D> h(nullptr);
      {
         ROOT::Internal::TDF::TIgnoreErrorLevelRAII iel(kError);
         if (model.fBinXEdges.empty() && model.fBinYEdges.empty() && model.fBinZEdges.empty()) {
            h =
               std::make_shared<::TH3D>(model.fName, model.fTitle, model.fNbinsX, model.fXLow, model.fXUp,
                                        model.fNbinsY, model.fYLow, model.fYUp, model.fNbinsZ, model.fZLow, model.fZUp);
         } else {
            h =
               std::make_shared<::TH3D>(model.fName, model.fTitle, model.fNbinsX, model.fBinXEdges.data(),
                                        model.fNbinsY, model.fBinYEdges.data(), model.fNbinsZ, model.fBinZEdges.data());
         }
      }
      if (!TDFInternal::HistoUtils<::TH3D>::HasAxisLimits(*h)) {
         throw std::runtime_error("3D histograms with no axes limits are not supported yet.");
      }
      auto columnViews = {v1Name, v2Name, v3Name};
      const auto userColumns = TDFInternal::AtLeastOneEmptyString(columnViews)
                                  ? ColumnNames_t()
                                  : ColumnNames_t(columnViews.begin(), columnViews.end());
      return CreateAction<TDFInternal::ActionTypes::Histo3D, V1, V2, V3>(userColumns, h);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a three-dimensional histogram (*lazy action*)
   /// \tparam V1 The type of the column used to fill the x axis of the histogram. Inferred if not present.
   /// \tparam V2 The type of the column used to fill the y axis of the histogram. Inferred if not present.
   /// \tparam V3 The type of the column used to fill the z axis of the histogram. Inferred if not present.
   /// \tparam W The type of the column used for the weights of the histogram. Inferred if not present.
   /// \param[in] model The returned histogram will be constructed using this as a model.
   /// \param[in] v1Name The name of the column that will fill the x axis.
   /// \param[in] v2Name The name of the column that will fill the y axis.
   /// \param[in] v3Name The name of the column that will fill the z axis.
   /// \param[in] wName The name of the column that will provide the weights.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TResultProxy documentation.
   /// The user gives up ownership of the model histogram.
   template <typename V1 = TDFDetail::TInferType, typename V2 = TDFDetail::TInferType,
             typename V3 = TDFDetail::TInferType, typename W = TDFDetail::TInferType>
   TResultProxy<::TH3D> Histo3D(const TH3DModel &model, std::string_view v1Name, std::string_view v2Name,
                                std::string_view v3Name, std::string_view wName)
   {
      std::shared_ptr<::TH3D> h(nullptr);
      {
         ROOT::Internal::TDF::TIgnoreErrorLevelRAII iel(kError);
         if (model.fBinXEdges.empty() && model.fBinYEdges.empty() && model.fBinZEdges.empty()) {
            h =
               std::make_shared<::TH3D>(model.fName, model.fTitle, model.fNbinsX, model.fXLow, model.fXUp,
                                        model.fNbinsY, model.fYLow, model.fYUp, model.fNbinsZ, model.fZLow, model.fZUp);
         } else {
            h =
               std::make_shared<::TH3D>(model.fName, model.fTitle, model.fNbinsX, model.fBinXEdges.data(),
                                        model.fNbinsY, model.fBinYEdges.data(), model.fNbinsZ, model.fBinZEdges.data());
         }
      }
      if (!TDFInternal::HistoUtils<::TH3D>::HasAxisLimits(*h)) {
         throw std::runtime_error("3D histograms with no axes limits are not supported yet.");
      }
      auto columnViews = {v1Name, v2Name, v3Name, wName};
      const auto userColumns = TDFInternal::AtLeastOneEmptyString(columnViews)
                                  ? ColumnNames_t()
                                  : ColumnNames_t(columnViews.begin(), columnViews.end());
      return CreateAction<TDFInternal::ActionTypes::Histo3D, V1, V2, V3, W>(userColumns, h);
   }

   template <typename V1, typename V2, typename V3, typename W>
   TResultProxy<::TH3D> Histo3D(const TH3DModel &model)
   {
      return Histo3D<V1, V2, V3, W>(model, "", "", "", "");
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a one-dimensional profile (*lazy action*)
   /// \tparam V1 The type of the column the values of which are used to fill the profile. Inferred if not present.
   /// \tparam V2 The type of the column the values of which are used to fill the profile. Inferred if not present.
   /// \param[in] model The model to be considered to build the new return value.
   /// \param[in] v1Name The name of the column that will fill the x axis.
   /// \param[in] v2Name The name of the column that will fill the y axis.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TResultProxy documentation.
   /// The user gives up ownership of the model profile object.
   template <typename V1 = TDFDetail::TInferType, typename V2 = TDFDetail::TInferType>
   TResultProxy<::TProfile>
   Profile1D(const TProfile1DModel &model, std::string_view v1Name = "", std::string_view v2Name = "")
   {
      std::shared_ptr<::TProfile> h(nullptr);
      {
         ROOT::Internal::TDF::TIgnoreErrorLevelRAII iel(kError);
         h = std::make_shared<::TProfile>(model.fName, model.fTitle, model.fNbinsX, model.fXLow, model.fXUp,
                                          model.fYLow, model.fYUp, model.fOption);
      }

      if (!TDFInternal::HistoUtils<::TProfile>::HasAxisLimits(*h)) {
         throw std::runtime_error("Profiles with no axes limits are not supported yet.");
      }
      auto columnViews = {v1Name, v2Name};
      const auto userColumns = TDFInternal::AtLeastOneEmptyString(columnViews)
                                  ? ColumnNames_t()
                                  : ColumnNames_t(columnViews.begin(), columnViews.end());
      return CreateAction<TDFInternal::ActionTypes::Profile1D, V1, V2>(userColumns, h);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a one-dimensional profile (*lazy action*)
   /// \tparam V1 The type of the column the values of which are used to fill the profile. Inferred if not present.
   /// \tparam V2 The type of the column the values of which are used to fill the profile. Inferred if not present.
   /// \tparam W The type of the column the weights of which are used to fill the profile. Inferred if not present.
   /// \param[in] model The model to be considered to build the new return value.
   /// \param[in] v1Name The name of the column that will fill the x axis.
   /// \param[in] v2Name The name of the column that will fill the y axis.
   /// \param[in] wName The name of the column that will provide the weights.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TResultProxy documentation.
   /// The user gives up ownership of the model profile object.
   template <typename V1 = TDFDetail::TInferType, typename V2 = TDFDetail::TInferType,
             typename W = TDFDetail::TInferType>
   TResultProxy<::TProfile>
   Profile1D(const TProfile1DModel &model, std::string_view v1Name, std::string_view v2Name, std::string_view wName)
   {
      std::shared_ptr<::TProfile> h(nullptr);
      {
         ROOT::Internal::TDF::TIgnoreErrorLevelRAII iel(kError);
         h = std::make_shared<::TProfile>(model.fName, model.fTitle, model.fNbinsX, model.fXLow, model.fXUp,
                                          model.fYLow, model.fYUp, model.fOption);
      }

      if (!TDFInternal::HistoUtils<::TProfile>::HasAxisLimits(*h)) {
         throw std::runtime_error("Profile histograms with no axes limits are not supported yet.");
      }
      auto columnViews = {v1Name, v2Name, wName};
      const auto userColumns = TDFInternal::AtLeastOneEmptyString(columnViews)
                                  ? ColumnNames_t()
                                  : ColumnNames_t(columnViews.begin(), columnViews.end());
      return CreateAction<TDFInternal::ActionTypes::Profile1D, V1, V2, W>(userColumns, h);
   }

   template <typename V1, typename V2, typename W>
   TResultProxy<::TProfile> Profile1D(const TProfile1DModel &model)
   {
      return Profile1D<V1, V2, W>(model, "", "", "");
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a two-dimensional profile (*lazy action*)
   /// \tparam V1 The type of the column used to fill the x axis of the histogram. Inferred if not present.
   /// \tparam V2 The type of the column used to fill the y axis of the histogram. Inferred if not present.
   /// \tparam V2 The type of the column used to fill the z axis of the histogram. Inferred if not present.
   /// \param[in] model The returned profile will be constructed using this as a model.
   /// \param[in] v1Name The name of the column that will fill the x axis.
   /// \param[in] v2Name The name of the column that will fill the y axis.
   /// \param[in] v3Name The name of the column that will fill the z axis.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TResultProxy documentation.
   /// The user gives up ownership of the model profile.
   template <typename V1 = TDFDetail::TInferType, typename V2 = TDFDetail::TInferType,
             typename V3 = TDFDetail::TInferType>
   TResultProxy<::TProfile2D> Profile2D(const TProfile2DModel &model, std::string_view v1Name = "",
                                        std::string_view v2Name = "", std::string_view v3Name = "")
   {
      std::shared_ptr<::TProfile2D> h(nullptr);
      {
         ROOT::Internal::TDF::TIgnoreErrorLevelRAII iel(kError);
         h = std::make_shared<::TProfile2D>(model.fName, model.fTitle, model.fNbinsX, model.fXLow, model.fXUp,
                                            model.fNbinsY, model.fYLow, model.fYUp, model.fZLow, model.fZUp,
                                            model.fOption);
      }

      if (!TDFInternal::HistoUtils<::TProfile2D>::HasAxisLimits(*h)) {
         throw std::runtime_error("2D profiles with no axes limits are not supported yet.");
      }
      auto columnViews = {v1Name, v2Name, v3Name};
      const auto userColumns = TDFInternal::AtLeastOneEmptyString(columnViews)
                                  ? ColumnNames_t()
                                  : ColumnNames_t(columnViews.begin(), columnViews.end());
      return CreateAction<TDFInternal::ActionTypes::Profile2D, V1, V2, V3>(userColumns, h);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a two-dimensional profile (*lazy action*)
   /// \tparam V1 The type of the column used to fill the x axis of the histogram. Inferred if not present.
   /// \tparam V2 The type of the column used to fill the y axis of the histogram. Inferred if not present.
   /// \tparam V3 The type of the column used to fill the z axis of the histogram. Inferred if not present.
   /// \tparam W The type of the column used for the weights of the histogram. Inferred if not present.
   /// \param[in] model The returned histogram will be constructed using this as a model.
   /// \param[in] v1Name The name of the column that will fill the x axis.
   /// \param[in] v2Name The name of the column that will fill the y axis.
   /// \param[in] v3Name The name of the column that will fill the z axis.
   /// \param[in] wName The name of the column that will provide the weights.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TResultProxy documentation.
   /// The user gives up ownership of the model profile.
   template <typename V1 = TDFDetail::TInferType, typename V2 = TDFDetail::TInferType,
             typename V3 = TDFDetail::TInferType, typename W = TDFDetail::TInferType>
   TResultProxy<::TProfile2D> Profile2D(const TProfile2DModel &model, std::string_view v1Name, std::string_view v2Name,
                                        std::string_view v3Name, std::string_view wName)
   {
      std::shared_ptr<::TProfile2D> h(nullptr);
      {
         ROOT::Internal::TDF::TIgnoreErrorLevelRAII iel(kError);
         h = std::make_shared<::TProfile2D>(model.fName, model.fTitle, model.fNbinsX, model.fXLow, model.fXUp,
                                            model.fNbinsY, model.fYLow, model.fYUp, model.fZLow, model.fZUp,
                                            model.fOption);
      }

      if (!TDFInternal::HistoUtils<::TProfile2D>::HasAxisLimits(*h)) {
         throw std::runtime_error("2D profiles with no axes limits are not supported yet.");
      }
      auto columnViews = {v1Name, v2Name, v3Name, wName};
      const auto userColumns = TDFInternal::AtLeastOneEmptyString(columnViews)
                                  ? ColumnNames_t()
                                  : ColumnNames_t(columnViews.begin(), columnViews.end());
      return CreateAction<TDFInternal::ActionTypes::Profile2D, V1, V2, V3, W>(userColumns, h);
   }

   template <typename V1, typename V2, typename V3, typename W>
   TResultProxy<::TProfile2D> Profile2D(const TProfile2DModel &model)
   {
      return Profile2D<V1, V2, V3, W>(model, "", "", "", "");
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return an object of type T on which `T::Fill` will be called once per event (*lazy action*)
   ///
   /// T must be a type that provides a copy- or move-constructor and a `T::Fill` method that takes as many arguments
   /// as the column names pass as columnList. The arguments of `T::Fill` must have type equal to the one of the
   /// specified columns (these types are passed as template parameters to this method).
   /// \tparam FirstColumn The first type of the column the values of which are used to fill the object.
   /// \tparam OtherColumns A list of the other types of the columns the values of which are used to fill the object.
   /// \tparam T The type of the object to fill. Automatically deduced.
   /// \param[in] model The model to be considered to build the new return value.
   /// \param[in] columnList A list containing the names of the columns that will be passed when calling `Fill`
   ///
   /// The user gives up ownership of the model object.
   /// The list of column names to be used for filling must always be specified.
   /// This action is *lazy*: upon invocation of this method the calculation is booked but not executed.
   /// See TResultProxy documentation.
   template <typename FirstColumn, typename... OtherColumns, typename T> // need FirstColumn to disambiguate overloads
   TResultProxy<T> Fill(T &&model, const ColumnNames_t &columnList)
   {
      auto h = std::make_shared<T>(std::move(model));
      if (!TDFInternal::HistoUtils<T>::HasAxisLimits(*h)) {
         throw std::runtime_error("The absence of axes limits is not supported yet.");
      }
      return CreateAction<TDFInternal::ActionTypes::Fill, FirstColumn, OtherColumns...>(columnList, h);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return an object of type T on which `T::Fill` will be called once per event (*lazy action*)
   ///
   /// This overload infers the types of the columns specified in columnList at runtime and just-in-time compiles the
   /// method with these types. See previous overload for more information.
   /// \tparam T The type of the object to fill. Automatically deduced.
   /// \param[in] model The model to be considered to build the new return value.
   /// \param[in] columnList The name of the columns read to fill the object.
   ///
   /// This overload of `Fill` infers the type of the specified columns at runtime and just-in-time compiles the
   /// previous overload. Check the previous overload for more details on `Fill`.
   template <typename T>
   TResultProxy<T> Fill(T &&model, const ColumnNames_t &bl)
   {
      auto h = std::make_shared<T>(std::move(model));
      if (!TDFInternal::HistoUtils<T>::HasAxisLimits(*h)) {
         throw std::runtime_error("The absence of axes limits is not supported yet.");
      }
      return CreateAction<TDFInternal::ActionTypes::Fill, TDFDetail::TInferType>(bl, h, bl.size());
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return the minimum of processed column values (*lazy action*)
   /// \tparam T The type of the branch/column.
   /// \param[in] columnName The name of the branch/column to be treated.
   ///
   /// If T is not specified, TDataFrame will infer it from the data and just-in-time compile the correct
   /// template specialization of this method.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TResultProxy documentation.
   template <typename T = TDFDetail::TInferType>
   TResultProxy<TDFInternal::MinReturnType_t<T>> Min(std::string_view columnName = "")
   {
      const auto userColumns = columnName.empty() ? ColumnNames_t() : ColumnNames_t({std::string(columnName)});
      using RetType_t = TDFInternal::MinReturnType_t<T>;
      auto minV = std::make_shared<RetType_t>(std::numeric_limits<RetType_t>::max());
      return CreateAction<TDFInternal::ActionTypes::Min, T>(userColumns, minV);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return the maximum of processed column values (*lazy action*)
   /// \tparam T The type of the branch/column.
   /// \param[in] columnName The name of the branch/column to be treated.
   ///
   /// If T is not specified, TDataFrame will infer it from the data and just-in-time compile the correct
   /// template specialization of this method.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TResultProxy documentation.
   template <typename T = TDFDetail::TInferType>
   TResultProxy<TDFInternal::MaxReturnType_t<T>> Max(std::string_view columnName = "")
   {
      const auto userColumns = columnName.empty() ? ColumnNames_t() : ColumnNames_t({std::string(columnName)});
      using RetType_t = TDFInternal::MaxReturnType_t<T>;
      auto maxV = std::make_shared<RetType_t>(std::numeric_limits<RetType_t>::lowest());
      return CreateAction<TDFInternal::ActionTypes::Max, T>(userColumns, maxV);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return the mean of processed column values (*lazy action*)
   /// \tparam T The type of the branch/column.
   /// \param[in] columnName The name of the branch/column to be treated.
   ///
   /// If T is not specified, TDataFrame will infer it from the data and just-in-time compile the correct
   /// template specialization of this method.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TResultProxy documentation.
   template <typename T = TDFDetail::TInferType>
   TResultProxy<double> Mean(std::string_view columnName = "")
   {
      const auto userColumns = columnName.empty() ? ColumnNames_t() : ColumnNames_t({std::string(columnName)});
      auto meanV = std::make_shared<double>(0);
      return CreateAction<TDFInternal::ActionTypes::Mean, T>(userColumns, meanV);
   }

   // clang-format off
   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return the sum of processed column values (*lazy action*)
   /// \tparam T The type of the branch/column.
   /// \param[in] columnName The name of the branch/column.
   /// \param[in] initValue Optional initial value for the sum. If not present, the column values must be default-constructible.
   ///
   /// If T is not specified, TDataFrame will infer it from the data and just-in-time compile the correct
   /// template specialization of this method.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See TResultProxy documentation.
   template <typename T = TDFDetail::TInferType>
   TResultProxy<TDFInternal::SumReturnType_t<T>>
   Sum(std::string_view columnName = "",
       const TDFInternal::SumReturnType_t<T> &initValue = TDFInternal::SumReturnType_t<T>{})
   {
      const auto userColumns = columnName.empty() ? ColumnNames_t() : ColumnNames_t({std::string(columnName)});
      auto sumV = std::make_shared<TDFInternal::SumReturnType_t<T>>(initValue);
      return CreateAction<TDFInternal::ActionTypes::Sum, T>(userColumns, sumV);
   }
   // clang-format on

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
      // if this is a TInterface<TLoopManager> on which `Define` has been called, users
      // are calling `Report` on a chain of the form LoopManager->Define->Define->..., which
      // certainly does not contain named filters.
      // The number 2 takes into account the implicit columns for entry and slot number
      if (std::is_same<Proxied, TLoopManager>::value && fValidCustomColumns.size() > 2)
         return;

      auto df = GetDataFrameChecked();
      // TODO: we could do better and check if the Report was asked for Filters that have already run
      // and in that case skip the event-loop
      if (df->MustRunNamedFilters())
         df->Run();
      fProxiedPtr->Report();
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Returns the names of the available columns
   ///
   /// This is not an action nor a transformation, just a simple utility to
   /// get columns names out of the TDataFrame nodes.
   ColumnNames_t GetColumnNames()
   {
      ColumnNames_t allColumns;

      auto addIfNotInternal = [&allColumns](std::string_view colName) {
         if (!TDFInternal::IsInternalColumn(colName))
            allColumns.emplace_back(colName);
      };

      std::for_each(fValidCustomColumns.begin(), fValidCustomColumns.end(), addIfNotInternal);

      auto df = GetDataFrameChecked();
      auto tree = df->GetTree();
      if (tree) {
         auto branchNames = TDFInternal::GetBranchNames(*tree);
         allColumns.insert(allColumns.end(), branchNames.begin(), branchNames.end());
      }

      if (fDataSource) {
         auto &dsColNames = fDataSource->GetColumnNames();
         allColumns.insert(allColumns.end(), dsColNames.begin(), dsColNames.end());
      }

      return allColumns;
   }

private:
   void AddDefaultColumns()
   {
      auto lm = GetDataFrameChecked();
      ColumnNames_t validColNames = {};

      // Entry number column
      auto entryColGen = [](unsigned int, ULong64_t entry) { return entry; };
      const auto entryColName = "tdfentry_";
      using EntryCol_t = TDFDetail::TCustomColumn<decltype(entryColGen), TDFDetail::TCCHelperTypes::TSlotAndEntry>;
      lm->Book(std::make_shared<EntryCol_t>(entryColName, std::move(entryColGen), validColNames, lm.get()));
      fValidCustomColumns.emplace_back(entryColName);

      // Slot number column
      auto slotColGen = [](unsigned int slot) { return slot; };
      const auto slotColName = "tdfslot_";
      using SlotCol_t = TDFDetail::TCustomColumn<decltype(slotColGen), TDFDetail::TCCHelperTypes::TSlot>;
      lm->Book(std::make_shared<SlotCol_t>(slotColName, std::move(slotColGen), validColNames, lm.get()));
      fValidCustomColumns.emplace_back(slotColName);
   }

   ColumnNames_t ConvertRegexToColumns(std::string_view columnNameRegexp)
   {
      const auto theRegexSize = columnNameRegexp.size();
      std::string theRegex(columnNameRegexp);

      const auto isEmptyRegex = 0 == theRegexSize;
      // This is to avoid cases where branches called b1, b2, b3 are all matched by expression "b"
      if (theRegexSize > 0 && theRegex[0] != '^')
         theRegex = "^" + theRegex;
      if (theRegexSize > 0 && theRegex[theRegexSize - 1] != '$')
         theRegex = theRegex + "$";

      ColumnNames_t selectedColumns;
      selectedColumns.reserve(32);

      // Since we support gcc48 and it does not provide in its stl std::regex,
      // we need to use TRegexp
      TRegexp regexp(theRegex);
      int dummy;
      for (auto &&branchName : fValidCustomColumns) {
         if ((isEmptyRegex || -1 != regexp.Index(branchName.c_str(), &dummy)) &&
             !TDFInternal::IsInternalColumn(branchName)) {
            selectedColumns.emplace_back(branchName);
         }
      }

      auto df = GetDataFrameChecked();
      auto tree = df->GetTree();
      if (tree) {
         auto branchNames = TDFInternal::GetBranchNames(*tree);
         for (auto &branchName : branchNames) {
            if (isEmptyRegex || -1 != regexp.Index(branchName, &dummy)) {
               selectedColumns.emplace_back(branchName);
            }
         }
      }

      if (fDataSource) {
         auto &dsColNames = fDataSource->GetColumnNames();
         for (auto &dsColName : dsColNames) {
            if (isEmptyRegex || -1 != regexp.Index(dsColName, &dummy)) {
               selectedColumns.emplace_back(dsColName);
            }
         }
      }

      return selectedColumns;
   }

   Long_t CallJitTransformation(std::string_view transformation, std::string_view nodeName, std::string_view expression,
                                std::string_view returnTypeName)
   {
      auto df = GetDataFrameChecked();
      auto &aliasMap = df->GetAliasMap();
      auto tree = df->GetTree();
      auto branches = tree ? TDFInternal::GetBranchNames(*tree) : ColumnNames_t();
      const auto &customColumns = df->GetCustomColumnNames();
      auto tmpBookedBranches = df->GetBookedColumns();
      auto upcastNode = TDFInternal::UpcastNode(fProxiedPtr);
      TInterface<TypeTraits::TakeFirstParameter_t<decltype(upcastNode)>> upcastInterface(
         upcastNode, fImplWeakPtr, fValidCustomColumns, fDataSource);
      const auto thisTypeName = "ROOT::Experimental::TDF::TInterface<" + upcastInterface.GetNodeTypeName() + ">";
      return TDFInternal::JitTransformation(&upcastInterface, transformation, thisTypeName, nodeName, expression,
                                            aliasMap, branches, customColumns, tmpBookedBranches, tree, returnTypeName,
                                            fDataSource);
   }

   /// Return string containing fully qualified type name of the node pointed by fProxied.
   /// The method is only defined for TInterface<{TFilterBase,TCustomColumnBase,TRangeBase,TLoopManager}> as it should
   /// only be called on "upcast" TInterfaces.
   inline static std::string GetNodeTypeName();

   // Type was specified by the user, no need to infer it
   template <typename ActionType, typename... BranchTypes, typename ActionResultType,
             typename std::enable_if<!TDFInternal::TNeedJitting<BranchTypes...>::value, int>::type = 0>
   TResultProxy<ActionResultType> CreateAction(const ColumnNames_t &columns, const std::shared_ptr<ActionResultType> &r)
   {
      auto lm = GetDataFrameChecked();
      constexpr auto nColumns = sizeof...(BranchTypes);
      const auto selectedCols =
         TDFInternal::GetValidatedColumnNames(*lm, nColumns, columns, fValidCustomColumns, fDataSource);
      if (fDataSource)
         TDFInternal::DefineDataSourceColumns(selectedCols, *lm, TDFInternal::GenStaticSeq_t<nColumns>(),
                                              TDFInternal::TypeList<BranchTypes...>(), *fDataSource);
      const auto nSlots = fProxiedPtr->GetNSlots();
      auto actionPtr =
         TDFInternal::BuildAndBook<BranchTypes...>(selectedCols, r, nSlots, *lm, *fProxiedPtr, (ActionType *)nullptr);
      return MakeResultProxy(r, lm, actionPtr);
   }

   // User did not specify type, do type inference
   // This version of CreateAction has a `nColumns` optional argument. If present, the number of required columns for
   // this action is taken equal to nColumns, otherwise it is assumed to be sizeof...(BranchTypes)
   template <typename ActionType, typename... BranchTypes, typename ActionResultType,
             typename std::enable_if<TDFInternal::TNeedJitting<BranchTypes...>::value, int>::type = 0>
   TResultProxy<ActionResultType>
   CreateAction(const ColumnNames_t &columns, const std::shared_ptr<ActionResultType> &r, const int nColumns = -1)
   {
      auto lm = GetDataFrameChecked();
      auto realNColumns = (nColumns > -1 ? nColumns : sizeof...(BranchTypes));
      const auto validColumnNames =
         TDFInternal::GetValidatedColumnNames(*lm, realNColumns, columns, fValidCustomColumns, fDataSource);
      const unsigned int nSlots = fProxiedPtr->GetNSlots();
      const auto &customColumns = lm->GetBookedColumns();
      auto tree = lm->GetTree();
      auto rOnHeap = TDFInternal::MakeSharedOnHeap(r);
      auto upcastNode = TDFInternal::UpcastNode(fProxiedPtr);
      TInterface<TypeTraits::TakeFirstParameter_t<decltype(upcastNode)>> upcastInterface(
         upcastNode, fImplWeakPtr, fValidCustomColumns, fDataSource);
      auto resultProxyAndActionPtrPtr = MakeResultProxy(r, lm);
      auto &resultProxy = resultProxyAndActionPtrPtr.first;
      auto actionPtrPtrOnHeap = TDFInternal::MakeSharedOnHeap(resultProxyAndActionPtrPtr.second);
      auto toJit = TDFInternal::JitBuildAndBook(validColumnNames, upcastInterface.GetNodeTypeName(), upcastNode.get(),
                                                typeid(std::shared_ptr<ActionResultType>), typeid(ActionType), rOnHeap,
                                                tree, nSlots, customColumns, fDataSource, actionPtrPtrOnHeap);
      lm->Jit(toJit);
      return resultProxy;
   }

   template <typename F, typename ExtraArgs>
   typename std::enable_if<std::is_default_constructible<typename TTraits::CallableTraits<F>::ret_type>::value,
                           TInterface<Proxied>>::type
   DefineImpl(std::string_view name, F &&expression, const ColumnNames_t &columns)
   {
      auto loopManager = GetDataFrameChecked();
      TDFInternal::CheckCustomColumn(name, loopManager->GetTree(), loopManager->GetCustomColumnNames(),
                                     fDataSource ? fDataSource->GetColumnNames() : ColumnNames_t{});

      using ArgTypes_t = typename TTraits::CallableTraits<F>::arg_types;
      using ColTypesTmp_t =
         typename TDFInternal::RemoveFirstParameterIf<std::is_same<ExtraArgs, TDFDetail::TCCHelperTypes::TSlot>::value,
                                                      ArgTypes_t>::type;
      using ColTypes_t = typename TDFInternal::RemoveFirstTwoParametersIf<
         std::is_same<ExtraArgs, TDFDetail::TCCHelperTypes::TSlotAndEntry>::value, ColTypesTmp_t>::type;

      constexpr auto nColumns = ColTypes_t::list_size;
      const auto validColumnNames =
         TDFInternal::GetValidatedColumnNames(*loopManager, nColumns, columns, fValidCustomColumns, fDataSource);
      if (fDataSource)
         TDFInternal::DefineDataSourceColumns(validColumnNames, *loopManager, TDFInternal::GenStaticSeq_t<nColumns>(),
                                              ColTypes_t(), *fDataSource);
      using NewCol_t = TDFDetail::TCustomColumn<F, ExtraArgs>;

      // Here we check if the return type is a pointer. In this case, we assume it points to valid memory
      // and we treat the column as if it came from a data source, i.e. it points to valid memory.
      loopManager->Book(std::make_shared<NewCol_t>(name, std::move(expression), validColumnNames, loopManager.get()));
      TInterface<Proxied> newInterface(fProxiedPtr, fImplWeakPtr, fValidCustomColumns, fDataSource);
      newInterface.fValidCustomColumns.emplace_back(name);
      return newInterface;
   }

   // This overload is chosen when the callable passed to Define or DefineSlot returns void.
   // It simply fires a compile-time error. This is preferable to a static_assert in the main `Define` overload because
   // this way compilation of `Define` has no way to continue after throwing the error.
   template <
      typename F, typename ExtraArgs,
      typename std::enable_if<!std::is_convertible<F, std::string>::value &&
                                 !std::is_default_constructible<typename TTraits::CallableTraits<F>::ret_type>::value,
                              int>::type = 0>
   TInterface<Proxied> DefineImpl(std::string_view, F, const ColumnNames_t & = {})
   {
      static_assert(std::is_default_constructible<typename TTraits::CallableTraits<F>::ret_type>::value,
                    "Error in `Define`: type returned by expression is not default-constructible");
      return *this; // never reached
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Implementation of snapshot
   /// \param[in] treename The name of the TTree
   /// \param[in] filename The name of the TFile
   /// \param[in] columnList The list of names of the branches to be written
   /// The implementation exploits Foreach. The association of the addresses to
   /// the branches takes place at the first event. This is possible because
   /// since there are no copies, the address of the value passed by reference
   /// is the address pointing to the storage of the read/created object in/by
   /// the TTreeReaderValue/TemporaryBranch
   template <typename... BranchTypes>
   TInterface<TLoopManager> SnapshotImpl(std::string_view treename, std::string_view filename,
                                         const ColumnNames_t &columnList, const TSnapshotOptions &options)
   {
      TDFInternal::CheckSnapshot(sizeof...(BranchTypes), columnList.size());
      auto df = GetDataFrameChecked();
      if (fDataSource)
         TDFInternal::DefineDataSourceColumns(columnList, *df, TDFInternal::GenStaticSeq_t<sizeof...(BranchTypes)>(),
                                              TTraits::TypeList<BranchTypes...>(), *fDataSource);

      const std::string fullTreename(treename);
      // split name into directory and treename if needed
      const auto lastSlash = treename.rfind('/');
      std::string_view dirname = "";
      if (std::string_view::npos != lastSlash) {
         dirname = treename.substr(0, lastSlash);
         treename = treename.substr(lastSlash + 1, treename.size());
      }

      // add action node to functional graph and run event loop
      std::shared_ptr<TDFInternal::TActionBase> actionPtr;
      if (!ROOT::IsImplicitMTEnabled()) {
         // single-thread snapshot
         using Helper_t = TDFInternal::SnapshotHelper<BranchTypes...>;
         using Action_t = TDFInternal::TAction<Helper_t, Proxied, TTraits::TypeList<BranchTypes...>>;
         actionPtr.reset(
            new Action_t(Helper_t(filename, dirname, treename, columnList, options), columnList, *fProxiedPtr));
      } else {
         // multi-thread snapshot
         using Helper_t = TDFInternal::SnapshotHelperMT<BranchTypes...>;
         using Action_t = TDFInternal::TAction<Helper_t, Proxied>;
         actionPtr.reset(
            new Action_t(Helper_t(fProxiedPtr->GetNSlots(), filename, dirname, treename, columnList, options),
                         columnList, *fProxiedPtr));
      }
      df->Book(std::move(actionPtr));
      df->Run();

      // create new TDF
      ::TDirectory::TContext ctxt;
      // Now we mimic a constructor for the TDataFrame. We cannot invoke it here
      // since this would introduce a cyclic headers dependency.
      TInterface<TLoopManager> snapshotTDF(std::make_shared<TLoopManager>(nullptr, columnList));
      auto chain = std::make_shared<TChain>(fullTreename.c_str());
      chain->Add(std::string(filename).c_str());
      snapshotTDF.fProxiedPtr->SetTree(chain);

      return snapshotTDF;
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Implementation of cache
   template <typename... BranchTypes, int... S>
   TInterface<TLoopManager> CacheImpl(const ColumnNames_t &columnList, TDFInternal::StaticSeq<S...> s)
   {

      // Check at compile time that the columns types are copy constructible
      constexpr bool areCopyConstructible =
         TDFInternal::TEvalAnd<std::is_copy_constructible<BranchTypes>::value...>::value;
      static_assert(areCopyConstructible, "Columns of a type which is not copy constructible cannot be cached yet.");

      // We share bits and pieces with snapshot. De facto this is a snapshot
      // in memory!
      TDFInternal::CheckSnapshot(sizeof...(BranchTypes), columnList.size());
      if (fDataSource) {
         auto lm = GetDataFrameChecked();
         TDFInternal::DefineDataSourceColumns(columnList, *lm, s, TTraits::TypeList<BranchTypes...>(), *fDataSource);
      }
      std::tuple<
         TDFInternal::CacheColumnHolder<typename TDFDetail::TakeRealTypes<BranchTypes>::RealColl_t::value_type>...>
         colHolders;

      // TODO: really fix the type of the Take....
      std::initializer_list<int> expander0{(
         // This gets expanded
         std::get<S>(colHolders).fContent = std::move(
            Take<typename std::decay<decltype(std::get<S>(colHolders))>::type::value_type>(columnList[S]).GetValue()),
         0)...};
      (void)expander0;

      auto nEntries = std::get<0>(colHolders).fContent.size();

      TInterface<TLoopManager> cachedTDF(std::make_shared<TLoopManager>(nEntries));
      const ColumnNames_t noCols = {};

      // Now we define the data columns. We add the name of the valid custom columns by hand later.
      auto lm = cachedTDF.GetDataFrameChecked();
      std::initializer_list<int> expander1{(
         // This gets expanded
         lm->Book(
            std::make_shared<TDFDetail::TCustomColumn<typename std::decay<decltype(std::get<S>(colHolders))>::type,
                                                      TDFDetail::TCCHelperTypes::TSlotAndEntry>>(
               columnList[S], std::move(std::get<S>(colHolders)), noCols, lm.get(), true)),
         0)...};
      (void)expander1;

      // Add the defined columns
      auto &vc = cachedTDF.fValidCustomColumns;
      vc.insert(vc.end(), columnList.begin(), columnList.end());
      return cachedTDF;
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

   TInterface(const std::shared_ptr<Proxied> &proxied, const std::weak_ptr<TLoopManager> &impl,
              const ColumnNames_t &validColumns, TDataSource *ds)
      : fProxiedPtr(proxied), fImplWeakPtr(impl), fValidCustomColumns(validColumns), fDataSource(ds)
   {
   }

   /// Only enabled when building a TInterface<TLoopManager>
   template <typename T = Proxied, typename std::enable_if<std::is_same<T, TLoopManager>::value, int>::type = 0>
   TInterface(const std::shared_ptr<Proxied> &proxied)
      : fProxiedPtr(proxied), fImplWeakPtr(proxied->GetSharedPtr()), fValidCustomColumns(),
        fDataSource(proxied->GetDataSource())
   {
      AddDefaultColumns();
   }

   const std::shared_ptr<Proxied> &GetProxiedPtr() const { return fProxiedPtr; }
};

template <>
inline std::string TInterface<TDFDetail::TFilterBase>::GetNodeTypeName()
{
   return "ROOT::Detail::TDF::TFilterBase";
}

template <>
inline std::string TInterface<TDFDetail::TLoopManager>::GetNodeTypeName()
{
   return "ROOT::Detail::TDF::TLoopManager";
}

template <>
inline std::string TInterface<TDFDetail::TRangeBase>::GetNodeTypeName()
{
   return "ROOT::Detail::TDF::TRangeBase";
}

} // end NS TDF
} // end NS Experimental
} // end NS ROOT

#endif // ROOT_TDF_INTERFACE
