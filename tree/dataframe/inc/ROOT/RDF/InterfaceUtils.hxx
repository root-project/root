// Author: Enrico Guiraud, Danilo Piparo CERN  02/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_TINTERFACE_UTILS
#define ROOT_RDF_TINTERFACE_UTILS

#include "RColumnRegister.hxx"
#include <ROOT/RDF/RAction.hxx>
#include <ROOT/RDF/ActionHelpers.hxx> // for BuildAction
#include <ROOT/RDF/RColumnRegister.hxx>
#include <ROOT/RDF/RDefine.hxx>
#include <ROOT/RDF/RDefinePerSample.hxx>
#include <ROOT/RDF/RFilter.hxx>
#include <ROOT/RDF/Utils.hxx>
#include <ROOT/RDF/RJittedAction.hxx>
#include <ROOT/RDF/RJittedDefine.hxx>
#include <ROOT/RDF/RJittedFilter.hxx>
#include <ROOT/RDF/RJittedVariation.hxx>
#include <ROOT/RDF/RLoopManager.hxx>
#include <ROOT/RStringView.hxx>
#include <ROOT/RDF/RVariation.hxx>
#include <ROOT/TypeTraits.hxx>
#include <TError.h> // gErrorIgnoreLevel
#include <TH1.h>
#include <TROOT.h> // IsImplicitMTEnabled

#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>
#include <unordered_map>

class TObjArray;
class TTree;
namespace ROOT {
namespace Detail {
namespace RDF {
class RNodeBase;
}
}
namespace RDF {
template <typename T>
class RResultPtr;
template<typename T, typename V>
class RInterface;
using RNode = RInterface<::ROOT::Detail::RDF::RNodeBase, void>;
class RDataSource;
} // namespace RDF

} // namespace ROOT

/// \cond HIDDEN_SYMBOLS

namespace ROOT {
namespace Internal {
namespace RDF {
using namespace ROOT::Detail::RDF;
using namespace ROOT::RDF;
namespace TTraits = ROOT::TypeTraits;

std::string DemangleTypeIdName(const std::type_info &typeInfo);

ColumnNames_t
ConvertRegexToColumns(const ColumnNames_t &colNames, std::string_view columnNameRegexp, std::string_view callerName);

/// An helper object that sets and resets gErrorIgnoreLevel via RAII.
class RIgnoreErrorLevelRAII {
private:
   int fCurIgnoreErrorLevel = gErrorIgnoreLevel;

public:
   RIgnoreErrorLevelRAII(int errorIgnoreLevel) { gErrorIgnoreLevel = errorIgnoreLevel; }
   ~RIgnoreErrorLevelRAII() { gErrorIgnoreLevel = fCurIgnoreErrorLevel; }
};

/****** BuildAction overloads *******/

// clang-format off
/// This namespace defines types to be used for tag dispatching in RInterface.
namespace ActionTags {
struct Histo1D{};
struct Histo2D{};
struct Histo3D{};
struct HistoND{};
struct Graph{};
struct GraphAsymmErrors{};
struct Profile1D{};
struct Profile2D{};
struct Min{};
struct Max{};
struct Sum{};
struct Mean{};
struct Fill{};
struct StdDev{};
struct Display{};
struct Snapshot{};
struct Book{};
}
// clang-format on

template <typename T, bool ISV6HISTO = std::is_base_of<TH1, std::decay_t<T>>::value>
struct HistoUtils {
   static void SetCanExtendAllAxes(T &h) { h.SetCanExtend(::TH1::kAllAxes); }
   static bool HasAxisLimits(T &h)
   {
      auto xaxis = h.GetXaxis();
      return !(xaxis->GetXmin() == 0. && xaxis->GetXmax() == 0.);
   }
};

template <typename T>
struct HistoUtils<T, false> {
   static void SetCanExtendAllAxes(T &) {}
   static bool HasAxisLimits(T &) { return true; }
};

// Generic filling (covers Histo2D, Histo3D, HistoND, Profile1D and Profile2D actions, with and without weights)
template <typename... ColTypes, typename ActionTag, typename ActionResultType, typename PrevNodeType>
std::unique_ptr<RActionBase>
BuildAction(const ColumnNames_t &bl, const std::shared_ptr<ActionResultType> &h, const unsigned int nSlots,
            std::shared_ptr<PrevNodeType> prevNode, ActionTag, const RColumnRegister &colRegister)
{
   using Helper_t = FillHelper<ActionResultType>;
   using Action_t = RAction<Helper_t, PrevNodeType, TTraits::TypeList<ColTypes...>>;
   return std::make_unique<Action_t>(Helper_t(h, nSlots), bl, std::move(prevNode), colRegister);
}

// Histo1D filling (must handle the special case of distinguishing FillHelper and BufferedFillHelper
template <typename... ColTypes, typename PrevNodeType>
std::unique_ptr<RActionBase>
BuildAction(const ColumnNames_t &bl, const std::shared_ptr<::TH1D> &h, const unsigned int nSlots,
            std::shared_ptr<PrevNodeType> prevNode, ActionTags::Histo1D, const RColumnRegister &colRegister)
{
   auto hasAxisLimits = HistoUtils<::TH1D>::HasAxisLimits(*h);

   if (hasAxisLimits) {
      using Helper_t = FillHelper<::TH1D>;
      using Action_t = RAction<Helper_t, PrevNodeType, TTraits::TypeList<ColTypes...>>;
      return std::make_unique<Action_t>(Helper_t(h, nSlots), bl, std::move(prevNode), colRegister);
   } else {
      using Helper_t = BufferedFillHelper;
      using Action_t = RAction<Helper_t, PrevNodeType, TTraits::TypeList<ColTypes...>>;
      return std::make_unique<Action_t>(Helper_t(h, nSlots), bl, std::move(prevNode), colRegister);
   }
}

template <typename... ColTypes, typename PrevNodeType>
std::unique_ptr<RActionBase>
BuildAction(const ColumnNames_t &bl, const std::shared_ptr<TGraph> &g, const unsigned int nSlots,
            std::shared_ptr<PrevNodeType> prevNode, ActionTags::Graph, const RColumnRegister &colRegister)
{
   using Helper_t = FillTGraphHelper;
   using Action_t = RAction<Helper_t, PrevNodeType, TTraits::TypeList<ColTypes...>>;
   return std::make_unique<Action_t>(Helper_t(g, nSlots), bl, std::move(prevNode), colRegister);
}

template <typename... ColTypes, typename PrevNodeType>
std::unique_ptr<RActionBase>
BuildAction(const ColumnNames_t &bl, const std::shared_ptr<TGraphAsymmErrors> &g, const unsigned int nSlots,
            std::shared_ptr<PrevNodeType> prevNode, ActionTags::GraphAsymmErrors, const RColumnRegister &colRegister)
{
   using Helper_t = FillTGraphAsymmErrorsHelper;
   using Action_t = RAction<Helper_t, PrevNodeType, TTraits::TypeList<ColTypes...>>;
   return std::make_unique<Action_t>(Helper_t(g, nSlots), bl, std::move(prevNode), colRegister);
}

// Min action
template <typename ColType, typename PrevNodeType, typename ActionResultType>
std::unique_ptr<RActionBase>
BuildAction(const ColumnNames_t &bl, const std::shared_ptr<ActionResultType> &minV, const unsigned int nSlots,
            std::shared_ptr<PrevNodeType> prevNode, ActionTags::Min, const RColumnRegister &colRegister)
{
   using Helper_t = MinHelper<ActionResultType>;
   using Action_t = RAction<Helper_t, PrevNodeType, TTraits::TypeList<ColType>>;
   return std::make_unique<Action_t>(Helper_t(minV, nSlots), bl, std::move(prevNode), colRegister);
}

// Max action
template <typename ColType, typename PrevNodeType, typename ActionResultType>
std::unique_ptr<RActionBase>
BuildAction(const ColumnNames_t &bl, const std::shared_ptr<ActionResultType> &maxV, const unsigned int nSlots,
            std::shared_ptr<PrevNodeType> prevNode, ActionTags::Max, const RColumnRegister &colRegister)
{
   using Helper_t = MaxHelper<ActionResultType>;
   using Action_t = RAction<Helper_t, PrevNodeType, TTraits::TypeList<ColType>>;
   return std::make_unique<Action_t>(Helper_t(maxV, nSlots), bl, std::move(prevNode), colRegister);
}

// Sum action
template <typename ColType, typename PrevNodeType, typename ActionResultType>
std::unique_ptr<RActionBase>
BuildAction(const ColumnNames_t &bl, const std::shared_ptr<ActionResultType> &sumV, const unsigned int nSlots,
            std::shared_ptr<PrevNodeType> prevNode, ActionTags::Sum, const RColumnRegister &colRegister)
{
   using Helper_t = SumHelper<ActionResultType>;
   using Action_t = RAction<Helper_t, PrevNodeType, TTraits::TypeList<ColType>>;
   return std::make_unique<Action_t>(Helper_t(sumV, nSlots), bl, std::move(prevNode), colRegister);
}

// Mean action
template <typename ColType, typename PrevNodeType>
std::unique_ptr<RActionBase>
BuildAction(const ColumnNames_t &bl, const std::shared_ptr<double> &meanV, const unsigned int nSlots,
            std::shared_ptr<PrevNodeType> prevNode, ActionTags::Mean, const RColumnRegister &colRegister)
{
   using Helper_t = MeanHelper;
   using Action_t = RAction<Helper_t, PrevNodeType, TTraits::TypeList<ColType>>;
   return std::make_unique<Action_t>(Helper_t(meanV, nSlots), bl, std::move(prevNode), colRegister);
}

// Standard Deviation action
template <typename ColType, typename PrevNodeType>
std::unique_ptr<RActionBase>
BuildAction(const ColumnNames_t &bl, const std::shared_ptr<double> &stdDeviationV, const unsigned int nSlots,
            std::shared_ptr<PrevNodeType> prevNode, ActionTags::StdDev, const RColumnRegister &colRegister)
{
   using Helper_t = StdDevHelper;
   using Action_t = RAction<Helper_t, PrevNodeType, TTraits::TypeList<ColType>>;
   return std::make_unique<Action_t>(Helper_t(stdDeviationV, nSlots), bl, prevNode, colRegister);
}

// Display action
template <typename... ColTypes, typename PrevNodeType>
std::unique_ptr<RActionBase> BuildAction(const ColumnNames_t &bl, const std::shared_ptr<RDisplay> &d,
                                         const unsigned int, std::shared_ptr<PrevNodeType> prevNode,
                                         ActionTags::Display, const RDFInternal::RColumnRegister &colRegister)
{
   using Helper_t = DisplayHelper<PrevNodeType>;
   using Action_t = RAction<Helper_t, PrevNodeType, TTraits::TypeList<ColTypes...>>;
   return std::make_unique<Action_t>(Helper_t(d, prevNode), bl, prevNode, colRegister);
}

struct SnapshotHelperArgs {
   std::string fFileName;
   std::string fDirName;
   std::string fTreeName;
   std::vector<std::string> fOutputColNames;
   ROOT::RDF::RSnapshotOptions fOptions;
};

// Snapshot action
template <typename... ColTypes, typename PrevNodeType>
std::unique_ptr<RActionBase>
BuildAction(const ColumnNames_t &colNames, const std::shared_ptr<SnapshotHelperArgs> &snapHelperArgs,
            const unsigned int nSlots, std::shared_ptr<PrevNodeType> prevNode, ActionTags::Snapshot,
            const RColumnRegister &colRegister)
{
   const auto &filename = snapHelperArgs->fFileName;
   const auto &dirname = snapHelperArgs->fDirName;
   const auto &treename = snapHelperArgs->fTreeName;
   const auto &outputColNames = snapHelperArgs->fOutputColNames;
   const auto &options = snapHelperArgs->fOptions;

   auto makeIsDefine = [&] {
      std::vector<bool> isDef;
      isDef.reserve(sizeof...(ColTypes));
      for (auto i = 0u; i < sizeof...(ColTypes); ++i)
         isDef[i] = colRegister.IsDefineOrAlias(colNames[i]);
      return isDef;
   };
   std::vector<bool> isDefine = makeIsDefine();

   std::unique_ptr<RActionBase> actionPtr;
   if (!ROOT::IsImplicitMTEnabled()) {
      // single-thread snapshot
      using Helper_t = SnapshotHelper<ColTypes...>;
      using Action_t = RAction<Helper_t, PrevNodeType>;
      actionPtr.reset(
         new Action_t(Helper_t(filename, dirname, treename, colNames, outputColNames, options, std::move(isDefine)),
                      colNames, prevNode, colRegister));
   } else {
      // multi-thread snapshot
      using Helper_t = SnapshotHelperMT<ColTypes...>;
      using Action_t = RAction<Helper_t, PrevNodeType>;
      actionPtr.reset(new Action_t(
         Helper_t(nSlots, filename, dirname, treename, colNames, outputColNames, options, std::move(isDefine)),
         colNames, prevNode, colRegister));
   }
   return actionPtr;
}

// Book with custom helper type
template <typename... ColTypes, typename PrevNodeType, typename Helper_t>
std::unique_ptr<RActionBase>
BuildAction(const ColumnNames_t &bl, const std::shared_ptr<Helper_t> &h, const unsigned int /*nSlots*/,
            std::shared_ptr<PrevNodeType> prevNode, ActionTags::Book, const RColumnRegister &colRegister)
{
   using Action_t = RAction<Helper_t, PrevNodeType, TTraits::TypeList<ColTypes...>>;
   return std::make_unique<Action_t>(Helper_t(std::move(*h)), bl, std::move(prevNode), colRegister);
}

/****** end BuildAndBook ******/

template <typename Filter>
void CheckFilter(Filter &)
{
   using FilterRet_t = typename RDF::CallableTraits<Filter>::ret_type;
   static_assert(std::is_convertible<FilterRet_t, bool>::value,
                 "filter expression returns a type that is not convertible to bool");
}

ColumnNames_t FilterArraySizeColNames(const ColumnNames_t &columnNames, const std::string &action);

void CheckValidCppVarName(std::string_view var, const std::string &where);

void CheckForRedefinition(const std::string &where, std::string_view definedCol, const RColumnRegister &colRegister,
                          const ColumnNames_t &treeColumns, const ColumnNames_t &dataSourceColumns);

void CheckForDefinition(const std::string &where, std::string_view definedColView, const RColumnRegister &colRegister,
                        const ColumnNames_t &treeColumns, const ColumnNames_t &dataSourceColumns);

void CheckForNoVariations(const std::string &where, std::string_view definedColView,
                          const RColumnRegister &colRegister);

std::string PrettyPrintAddr(const void *const addr);

std::shared_ptr<RJittedFilter> BookFilterJit(std::shared_ptr<RNodeBase> *prevNodeOnHeap, std::string_view name,
                                             std::string_view expression, const ColumnNames_t &branches,
                                             const RColumnRegister &colRegister, TTree *tree, RDataSource *ds);

std::shared_ptr<RJittedDefine> BookDefineJit(std::string_view name, std::string_view expression, RLoopManager &lm,
                                             RDataSource *ds, const RColumnRegister &colRegister,
                                             const ColumnNames_t &branches, std::shared_ptr<RNodeBase> *prevNodeOnHeap);

std::shared_ptr<RJittedDefine> BookDefinePerSampleJit(std::string_view name, std::string_view expression,
                                                      RLoopManager &lm, const RColumnRegister &colRegister,
                                                      std::shared_ptr<RNodeBase> *upcastNodeOnHeap);

std::shared_ptr<RJittedVariation>
BookVariationJit(const std::vector<std::string> &colNames, std::string_view variationName,
                 const std::vector<std::string> &variationTags, std::string_view expression, RLoopManager &lm,
                 RDataSource *ds, const RColumnRegister &colRegister, const ColumnNames_t &branches,
                 std::shared_ptr<RNodeBase> *upcastNodeOnHeap, bool isSingleColumn);

std::string JitBuildAction(const ColumnNames_t &bl, std::shared_ptr<RDFDetail::RNodeBase> *prevNode,
                           const std::type_info &art, const std::type_info &at, void *rOnHeap, TTree *tree,
                           const unsigned int nSlots, const RColumnRegister &colRegister, RDataSource *ds,
                           std::weak_ptr<RJittedAction> *jittedActionOnHeap);

// Allocate a weak_ptr on the heap, return a pointer to it. The user is responsible for deleting this weak_ptr.
// This function is meant to be used by RInterface's methods that book code for jitting.
// The problem it solves is that we generate code to be lazily jitted with the addresses of certain objects in them,
// and we need to check those objects are still alive when the generated code is finally jitted and executed.
// So we pass addresses to weak_ptrs allocated on the heap to the jitted code, which is then responsible for
// the deletion of the weak_ptr object.
template <typename T>
std::weak_ptr<T> *MakeWeakOnHeap(const std::shared_ptr<T> &shPtr)
{
   return new std::weak_ptr<T>(shPtr);
}

// Same as MakeWeakOnHeap, but create a shared_ptr that makes sure the object is definitely kept alive.
template <typename T>
std::shared_ptr<T> *MakeSharedOnHeap(const std::shared_ptr<T> &shPtr)
{
   return new std::shared_ptr<T>(shPtr);
}

bool AtLeastOneEmptyString(const std::vector<std::string_view> strings);

/// Take a shared_ptr<AnyNodeType> and return a shared_ptr<RNodeBase>.
/// This works for RLoopManager nodes as well as filters and ranges.
std::shared_ptr<RNodeBase> UpcastNode(std::shared_ptr<RNodeBase> ptr);

ColumnNames_t GetValidatedColumnNames(RLoopManager &lm, const unsigned int nColumns, const ColumnNames_t &columns,
                                      const RColumnRegister &validDefines, RDataSource *ds);

std::vector<std::string> GetValidatedArgTypes(const ColumnNames_t &colNames, const RColumnRegister &colRegister,
                                              TTree *tree, RDataSource *ds, const std::string &context,
                                              bool vector2rvec);

std::vector<bool> FindUndefinedDSColumns(const ColumnNames_t &requestedCols, const ColumnNames_t &definedDSCols);

template <typename T>
void AddDSColumnsHelper(const std::string &colName, RLoopManager &lm, RDataSource &ds, RColumnRegister &colRegister)
{
   if (colRegister.IsDefineOrAlias(colName) || !ds.HasColumn(colName) ||
       lm.HasDataSourceColumnReaders(colName, typeid(T)))
      return;

   const auto nSlots = lm.GetNSlots();
   std::vector<std::unique_ptr<RColumnReaderBase>> colReaders;
   colReaders.reserve(nSlots);

   const auto valuePtrs = ds.GetColumnReaders<T>(colName);
   if (!valuePtrs.empty()) { // we are using the old GetColumnReaders mechanism in this RDataSource
      for (auto *ptr : valuePtrs)
         colReaders.emplace_back(new RDSColumnReader<T>(ptr));

   } else { // using the new GetColumnReaders mechanism
      // TODO consider changing the interface so we return all of these for all slots in one go
      for (auto slot = 0u; slot < lm.GetNSlots(); ++slot)
         colReaders.emplace_back(ds.GetColumnReaders(slot, colName, typeid(T)));
   }

   lm.AddDataSourceColumnReaders(colName, std::move(colReaders), typeid(T));
}

/// Take list of column names that must be defined, current map of custom columns, current list of defined column names,
/// and return a new map of custom columns (with the new datasource columns added to it)
template <typename... ColumnTypes>
void AddDSColumns(const std::vector<std::string> &requiredCols, RLoopManager &lm, RDataSource &ds,
                  TTraits::TypeList<ColumnTypes...>, RColumnRegister &colRegister)
{
   // hack to expand a template parameter pack without c++17 fold expressions.
   using expander = int[];
   int i = 0;
   (void)expander{(AddDSColumnsHelper<ColumnTypes>(requiredCols[i], lm, ds, colRegister), ++i)..., 0};
}

// this function is meant to be called by the jitted code generated by BookFilterJit
template <typename F, typename PrevNode>
void JitFilterHelper(F &&f, const char **colsPtr, std::size_t colsSize, std::string_view name,
                     std::weak_ptr<RJittedFilter> *wkJittedFilter, std::shared_ptr<PrevNode> *prevNodeOnHeap,
                     RColumnRegister *colRegister) noexcept
{
   if (wkJittedFilter->expired()) {
      // The branch of the computation graph that needed this jitted code went out of scope between the type
      // jitting was booked and the time jitting actually happened. Nothing to do other than cleaning up.
      delete wkJittedFilter;
      delete colRegister;
      delete prevNodeOnHeap;
      return;
   }

   const ColumnNames_t cols(colsPtr, colsPtr + colsSize);
   delete[] colsPtr;

   const auto jittedFilter = wkJittedFilter->lock();

   // mock Filter logic -- validity checks and Define-ition of RDataSource columns
   using Callable_t = std::decay_t<F>;
   using F_t = RFilter<Callable_t, PrevNode>;
   using ColTypes_t = typename TTraits::CallableTraits<Callable_t>::arg_types;
   constexpr auto nColumns = ColTypes_t::list_size;
   CheckFilter(f);

   auto &lm = *jittedFilter->GetLoopManagerUnchecked(); // RLoopManager must exist at this time
   auto ds = lm.GetDataSource();

   if (ds != nullptr)
      AddDSColumns(cols, lm, *ds, ColTypes_t(), *colRegister);

   jittedFilter->SetFilter(
      std::unique_ptr<RFilterBase>(new F_t(std::forward<F>(f), cols, *prevNodeOnHeap, *colRegister, name)));
   // colRegister points to the columns structure in the heap, created before the jitted call so that the jitter can
   // share data after it has lazily compiled the code. Here the data has been used and the memory can be freed.
   delete colRegister;
   delete prevNodeOnHeap;
   delete wkJittedFilter;
}

namespace DefineTypes {
struct RDefineTag {};
struct RDefinePerSampleTag {};
}

template <typename F>
auto MakeDefineNode(DefineTypes::RDefineTag, std::string_view name, std::string_view dummyType, F &&f,
                    const ColumnNames_t &cols, RColumnRegister &colRegister, RLoopManager &lm)
{
   return std::unique_ptr<RDefineBase>(new RDefine<std::decay_t<F>, ExtraArgsForDefine::None>(
      name, dummyType, std::forward<F>(f), cols, colRegister, lm));
}

template <typename F>
auto MakeDefineNode(DefineTypes::RDefinePerSampleTag, std::string_view name, std::string_view dummyType, F &&f,
                    const ColumnNames_t &, RColumnRegister &, RLoopManager &lm)
{
   return std::unique_ptr<RDefineBase>(
      new RDefinePerSample<std::decay_t<F>>(name, dummyType, std::forward<F>(f), lm));
}

// Build a RDefine or a RDefinePerSample object and attach it to an existing RJittedDefine
// This function is meant to be called by jitted code right before starting the event loop.
// If colsPtr is null, build a RDefinePerSample (it has no input columns), otherwise a RDefine.
template <typename RDefineTypeTag, typename F>
void JitDefineHelper(F &&f, const char **colsPtr, std::size_t colsSize, std::string_view name, RLoopManager *lm,
                     std::weak_ptr<RJittedDefine> *wkJittedDefine, RColumnRegister *colRegister,
                     std::shared_ptr<RNodeBase> *prevNodeOnHeap) noexcept
{
   // a helper to delete objects allocated before jitting, so that the jitter can share data with lazily jitted code
   auto doDeletes = [&] {
      delete wkJittedDefine;
      delete colRegister;
      delete prevNodeOnHeap;
      delete[] colsPtr;
   };

   if (wkJittedDefine->expired()) {
      // The branch of the computation graph that needed this jitted code went out of scope between the type
      // jitting was booked and the time jitting actually happened. Nothing to do other than cleaning up.
      doDeletes();
      return;
   }

   const ColumnNames_t cols(colsPtr, colsPtr + colsSize);

   auto jittedDefine = wkJittedDefine->lock();

   using Callable_t = std::decay_t<F>;
   using ColTypes_t = typename TTraits::CallableTraits<Callable_t>::arg_types;

   auto ds = lm->GetDataSource();
   if (ds != nullptr)
      AddDSColumns(cols, *lm, *ds, ColTypes_t(), *colRegister);

   // will never actually be used (trumped by jittedDefine->GetTypeName()), but we set it to something meaningful
   // to help devs debugging
   const auto dummyType = "jittedCol_t";
   // use unique_ptr<RDefineBase> instead of make_unique<NewCol_t> to reduce jit/compile-times
   std::unique_ptr<RDefineBase> newCol{
      MakeDefineNode(RDefineTypeTag{}, name, dummyType, std::forward<F>(f), cols, *colRegister, *lm)};
   jittedDefine->SetDefine(std::move(newCol));

   doDeletes();
}

template <bool IsSingleColumn, typename F>
void JitVariationHelper(F &&f, const char **colsPtr, std::size_t colsSize, const char **variedCols,
                        std::size_t variedColsSize, const char **variationTags, std::size_t variationTagsSize,
                        std::string_view variationName, RLoopManager *lm,
                        std::weak_ptr<RJittedVariation> *wkJittedVariation, RColumnRegister *colRegister,
                        std::shared_ptr<RNodeBase> *prevNodeOnHeap) noexcept
{
   // a helper to delete objects allocated before jitting, so that the jitter can share data with lazily jitted code
   auto doDeletes = [&] {
      delete[] colsPtr;
      delete[] variedCols;
      delete[] variationTags;

      delete wkJittedVariation;
      delete colRegister;
      delete prevNodeOnHeap;
   };

   if (wkJittedVariation->expired()) {
      // The branch of the computation graph that needed this jitted variation went out of scope between the type
      // jitting was booked and the time jitting actually happened. Nothing to do other than cleaning up.
      doDeletes();
      return;
   }

   const ColumnNames_t inputColNames(colsPtr, colsPtr + colsSize);
   std::vector<std::string> variedColNames(variedCols, variedCols + variedColsSize);
   std::vector<std::string> tags(variationTags, variationTags + variationTagsSize);

   auto jittedVariation = wkJittedVariation->lock();

   using Callable_t = std::decay_t<F>;
   using ColTypes_t = typename TTraits::CallableTraits<Callable_t>::arg_types;

   auto ds = lm->GetDataSource();
   if (ds != nullptr)
      AddDSColumns(inputColNames, *lm, *ds, ColTypes_t(), *colRegister);

   // use unique_ptr<RDefineBase> instead of make_unique<NewCol_t> to reduce jit/compile-times
   std::unique_ptr<RVariationBase> newVariation{new RVariation<std::decay_t<F>, IsSingleColumn>(
      std::move(variedColNames), variationName, std::forward<F>(f), std::move(tags), jittedVariation->GetTypeName(),
      *colRegister, *lm, std::move(inputColNames))};
   jittedVariation->SetVariation(std::move(newVariation));

   doDeletes();
}

/// Convenience function invoked by jitted code to build action nodes at runtime
template <typename ActionTag, typename... ColTypes, typename PrevNodeType, typename HelperArgType>
void CallBuildAction(std::shared_ptr<PrevNodeType> *prevNodeOnHeap, const char **colsPtr, std::size_t colsSize,
                     const unsigned int nSlots, std::shared_ptr<HelperArgType> *helperArgOnHeap,
                     std::weak_ptr<RJittedAction> *wkJittedActionOnHeap, RColumnRegister *colRegister) noexcept
{
   // a helper to delete objects allocated before jitting, so that the jitter can share data with lazily jitted code
   auto doDeletes = [&] {
      delete[] colsPtr;
      delete helperArgOnHeap;
      delete wkJittedActionOnHeap;
      // colRegister must be deleted before prevNodeOnHeap because their dtor needs the RLoopManager to be alive
      // and prevNodeOnHeap is what keeps it alive if the rest of the computation graph is already out of scope
      delete colRegister;
      delete prevNodeOnHeap;
   };

   if (wkJittedActionOnHeap->expired()) {
      // The branch of the computation graph that needed this jitted variation went out of scope between the type
      // jitting was booked and the time jitting actually happened. Nothing to do other than cleaning up.
      doDeletes();
      return;
   }

   const ColumnNames_t cols(colsPtr, colsPtr + colsSize);

   auto jittedActionOnHeap = wkJittedActionOnHeap->lock();

   // if we are here it means we are jitting, if we are jitting the loop manager must be alive
   auto &prevNodePtr = *prevNodeOnHeap;
   auto &loopManager = *prevNodePtr->GetLoopManagerUnchecked();
   using ColTypes_t = TypeList<ColTypes...>;
   constexpr auto nColumns = ColTypes_t::list_size;
   auto ds = loopManager.GetDataSource();
   if (ds != nullptr)
      AddDSColumns(cols, loopManager, *ds, ColTypes_t(), *colRegister);

   auto actionPtr = BuildAction<ColTypes...>(cols, std::move(*helperArgOnHeap), nSlots, std::move(prevNodePtr),
                                             ActionTag{}, *colRegister);
   jittedActionOnHeap->SetAction(std::move(actionPtr));

   doDeletes();
}

/// The contained `type` alias is `double` if `T == RInferredType`, `U` if `T == std::container<U>`, `T` otherwise.
template <typename T, bool Container = IsDataContainer<T>::value && !std::is_same<T, std::string>::value>
struct RMinReturnType {
   using type = T;
};

template <>
struct RMinReturnType<RInferredType, false> {
   using type = double;
};

template <typename T>
struct RMinReturnType<T, true> {
   using type = TTraits::TakeFirstParameter_t<T>;
};

// return wrapper around f that prepends an `unsigned int slot` parameter
template <typename R, typename F, typename... Args>
std::function<R(unsigned int, Args...)> AddSlotParameter(F &f, TypeList<Args...>)
{
   return [f](unsigned int, Args... a) mutable -> R { return f(a...); };
}

template <typename ColType, typename... Rest>
struct RNeedJittingHelper {
   static constexpr bool value = RNeedJittingHelper<Rest...>::value;
};

template <typename... Rest>
struct RNeedJittingHelper<RInferredType, Rest...> {
   static constexpr bool value = true;
};

template <typename T>
struct RNeedJittingHelper<T> {
   static constexpr bool value = false;
};

template <>
struct RNeedJittingHelper<RInferredType> {
   static constexpr bool value = true;
};

template <typename ...ColTypes>
struct RNeedJitting {
   static constexpr bool value = RNeedJittingHelper<ColTypes...>::value;
};

template <>
struct RNeedJitting<> {
   static constexpr bool value = false;
};

///////////////////////////////////////////////////////////////////////////////
/// Check preconditions for RInterface::Aggregate:
/// - the aggregator callable must have signature `U(U,T)` or `void(U&,T)`.
/// - the merge callable must have signature `U(U,U)` or `void(std::vector<U>&)`
template <typename R, typename Merge, typename U, typename T, typename decayedU = std::decay_t<U>,
          typename mergeArgsNoDecay_t = typename CallableTraits<Merge>::arg_types_nodecay,
          typename mergeArgs_t = typename CallableTraits<Merge>::arg_types,
          typename mergeRet_t = typename CallableTraits<Merge>::ret_type>
void CheckAggregate(TypeList<U, T>)
{
   constexpr bool isAggregatorOk =
      (std::is_same<R, decayedU>::value) || (std::is_same<R, void>::value && std::is_lvalue_reference<U>::value);
   static_assert(isAggregatorOk, "aggregator function must have signature `U(U,T)` or `void(U&,T)`");
   constexpr bool isMergeOk =
      (std::is_same<TypeList<decayedU, decayedU>, mergeArgs_t>::value && std::is_same<decayedU, mergeRet_t>::value) ||
      (std::is_same<TypeList<std::vector<decayedU> &>, mergeArgsNoDecay_t>::value &&
       std::is_same<void, mergeRet_t>::value);
   static_assert(isMergeOk, "merge function must have signature `U(U,U)` or `void(std::vector<U>&)`");
}

///////////////////////////////////////////////////////////////////////////////
/// This overload of CheckAggregate is called when the aggregator takes more than two arguments
template <typename R, typename T>
void CheckAggregate(T)
{
   static_assert(sizeof(T) == 0, "aggregator function must take exactly two arguments");
}

///////////////////////////////////////////////////////////////////////////////
/// Check as many template parameters were passed as the number of column names, throw if this is not the case.
void CheckTypesAndPars(unsigned int nTemplateParams, unsigned int nColumnNames);

/// Return local BranchNames or default BranchNames according to which one should be used
const ColumnNames_t SelectColumns(unsigned int nArgs, const ColumnNames_t &bl, const ColumnNames_t &defBl);

/// Check whether column names refer to a valid branch of a TTree or have been `Define`d. Return invalid column names.
ColumnNames_t FindUnknownColumns(const ColumnNames_t &requiredCols, const ColumnNames_t &datasetColumns,
                                 const RColumnRegister &definedCols, const ColumnNames_t &dataSourceColumns);

/// Returns the list of Filters defined in the whole graph
std::vector<std::string> GetFilterNames(const std::shared_ptr<RLoopManager> &loopManager);

/// Returns the list of Filters defined in the branch
template <typename NodeType>
std::vector<std::string> GetFilterNames(const std::shared_ptr<NodeType> &node)
{
   std::vector<std::string> filterNames;
   node->AddFilterName(filterNames);
   return filterNames;
}

struct ParsedTreePath {
   std::string fTreeName;
   std::string fDirName;
};

ParsedTreePath ParseTreePath(std::string_view fullTreeName);

// Check if a condition is true for all types
template <bool...>
struct TBoolPack;

template <bool... bs>
using IsTrueForAllImpl_t = typename std::is_same<TBoolPack<bs..., true>, TBoolPack<true, bs...>>;

template <bool... Conditions>
struct TEvalAnd {
   static constexpr bool value = IsTrueForAllImpl_t<Conditions...>::value;
};

// Check if a class is a specialisation of stl containers templates
// clang-format off

template <typename>
struct IsList_t : std::false_type {};

template <typename T>
struct IsList_t<std::list<T>> : std::true_type {};

template <typename>
struct IsDeque_t : std::false_type {};

template <typename T>
struct IsDeque_t<std::deque<T>> : std::true_type {};
// clang-format on

void CheckForDuplicateSnapshotColumns(const ColumnNames_t &cols);

void TriggerRun(ROOT::RDF::RNode &node);

template <typename T>
struct InnerValueType {
   using type = T; // fallback for when T is not a nested RVec
};

template <typename Elem>
struct InnerValueType<ROOT::VecOps::RVec<ROOT::VecOps::RVec<Elem>>> {
   using type = Elem;
};

template <typename T>
using InnerValueType_t = typename InnerValueType<T>::type;

std::pair<std::vector<std::string>, std::vector<std::string>>
AddSizeBranches(const std::vector<std::string> &branches, TTree *tree, std::vector<std::string> &&colsWithoutAliases,
                std::vector<std::string> &&colsWithAliases);

void RemoveDuplicates(ColumnNames_t &columnNames);

} // namespace RDF
} // namespace Internal

namespace Detail {
namespace RDF {

/// The aliased type is `double` if `T == RInferredType`, `U` if `T == container<U>`, `T` otherwise.
template <typename T>
using MinReturnType_t = typename RDFInternal::RMinReturnType<T>::type;

template <typename T>
using MaxReturnType_t = MinReturnType_t<T>;

template <typename T>
using SumReturnType_t = MinReturnType_t<T>;

} // namespace RDF
} // namespace Detail
} // namespace ROOT

/// \endcond

#endif
