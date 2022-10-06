// Author: Enrico Guiraud, Danilo Piparo CERN  03/2017

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_TINTERFACE
#define ROOT_RDF_TINTERFACE

#include "ROOT/InternalTreeUtils.hxx" // for GetFileNamesFromTree and GetFriendInfo
#include "ROOT/RDataSource.hxx"
#include "ROOT/RDF/ActionHelpers.hxx"
#include "ROOT/RDF/HistoModels.hxx"
#include "ROOT/RDF/InterfaceUtils.hxx"
#include "ROOT/RDF/RColumnRegister.hxx"
#include "ROOT/RDF/RDefine.hxx"
#include "ROOT/RDF/RDefinePerSample.hxx"
#include "ROOT/RDF/RFilter.hxx"
#include "ROOT/RDF/RVariation.hxx"
#include "ROOT/RDF/RLazyDSImpl.hxx"
#include "ROOT/RDF/RLoopManager.hxx"
#include "ROOT/RDF/RRange.hxx"
#include "ROOT/RDF/Utils.hxx"
#include "ROOT/RDF/RDFDescription.hxx"
#include "ROOT/RDF/RVariationsDescription.hxx"
#include "ROOT/RResultPtr.hxx"
#include "ROOT/RSnapshotOptions.hxx"
#include "ROOT/RStringView.hxx"
#include "ROOT/RVec.hxx"
#include "ROOT/TypeTraits.hxx"
#include "RtypesCore.h"               // for ULong64_t
#include "TChain.h" // for checking fLoopManger->GetTree() return type
#include "TDirectory.h"
#include "TH1.h" // For Histo actions
#include "TH2.h" // For Histo actions
#include "TH3.h" // For Histo actions
#include "THn.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "TStatistic.h"

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <iterator> // std::back_insterter
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits> // is_same, enable_if
#include <typeinfo>
#include <unordered_set>
#include <utility> // std::index_sequence
#include <vector>

class TGraph;

// Windows requires a forward decl of printValue to accept it as a valid friend function in RInterface
namespace ROOT {
void DisableImplicitMT();
bool IsImplicitMTEnabled();
void EnableImplicitMT(UInt_t numthreads);
class RDataFrame;
namespace Internal {
namespace RDF {
class GraphCreatorHelper;
}
} // namespace Internal
} // namespace ROOT
namespace cling {
std::string printValue(ROOT::RDataFrame *tdf);
}

namespace ROOT {
namespace RDF {
namespace RDFDetail = ROOT::Detail::RDF;
namespace RDFInternal = ROOT::Internal::RDF;
namespace TTraits = ROOT::TypeTraits;

template <typename Proxied, typename DataSource>
class RInterface;

using RNode = RInterface<::ROOT::Detail::RDF::RNodeBase, void>;

// clang-format off
/**
 * \class ROOT::RDF::RInterface
 * \ingroup dataframe
 * \brief The public interface to the RDataFrame federation of classes.
 * \tparam Proxied One of the "node" base types (e.g. RLoopManager, RFilterBase). The user never specifies this type manually.
 * \tparam DataSource The type of the RDataSource which is providing the data to the data frame. There is no source by default.
 *
 * The documentation of each method features a one liner illustrating how to use the method, for example showing how
 * the majority of the template parameters are automatically deduced requiring no or very little effort by the user.
 */
// clang-format on
template <typename Proxied, typename DataSource = void>
class RInterface {
   using DS_t = DataSource;
   using RFilterBase = RDFDetail::RFilterBase;
   using RRangeBase = RDFDetail::RRangeBase;
   using RLoopManager = RDFDetail::RLoopManager;
   friend std::string cling::printValue(::ROOT::RDataFrame *tdf); // For a nice printing at the prompt
   friend class RDFInternal::GraphDrawing::GraphCreatorHelper;

   template <typename T, typename W>
   friend class RInterface;

   friend void RDFInternal::TriggerRun(RNode &node);

   std::shared_ptr<Proxied> fProxiedPtr; ///< Smart pointer to the graph node encapsulated by this RInterface.
   ///< The RLoopManager at the root of this computation graph. Never null.
   RLoopManager *fLoopManager;
   /// Non-owning pointer to a data-source object. Null if no data-source. RLoopManager has ownership of the object.
   RDataSource *fDataSource = nullptr;

   /// Contains the columns defined up to this node.
   RDFInternal::RColumnRegister fColRegister;

   std::string DescribeDataset() const
   {
      // TTree/TChain as input
      const auto tree = fLoopManager->GetTree();
      if (tree) {
         const auto treeName = tree->GetName();
         const auto isTChain = dynamic_cast<TChain *>(tree) ? true : false;
         const auto treeType = isTChain ? "TChain" : "TTree";
         const auto isInMemory = !isTChain && !tree->GetCurrentFile() ? true : false;
         const auto friendInfo = ROOT::Internal::TreeUtils::GetFriendInfo(*tree);
         const auto hasFriends = friendInfo.fFriendNames.empty() ? false : true;
         std::stringstream ss;
         ss << "Dataframe from " << treeType << " " << treeName;
         if (isInMemory) {
            ss << " (in-memory)";
         } else {
            const auto files = ROOT::Internal::TreeUtils::GetFileNamesFromTree(*tree);
            const auto numFiles = files.size();
            if (numFiles == 1) {
               ss << " in file " << files[0];
            } else {
               ss << " in files\n";
               for (auto i = 0u; i < numFiles; i++) {
                  ss << "  " << files[i];
                  if (i < numFiles - 1)
                     ss << '\n';
               }
            }
         }
         if (hasFriends) {
            const auto numFriends = friendInfo.fFriendNames.size();
            if (numFriends == 1) {
               ss << "\nwith friend\n";
            } else {
               ss << "\nwith friends\n";
            }
            for (auto i = 0u; i < numFriends; i++) {
               const auto nameAlias = friendInfo.fFriendNames[i];
               const auto files = friendInfo.fFriendFileNames[i];
               const auto numFiles = files.size();
               const auto subnames = friendInfo.fFriendChainSubNames[i];
               ss << "  " << nameAlias.first;
               if (nameAlias.first != nameAlias.second)
                  ss << " (" << nameAlias.second << ")";
               // case: TTree as friend
               if (numFiles == 1) {
                  ss << " " << files[0];
               }
               // case: TChain as friend
               else {
                  ss << '\n';
                  for (auto j = 0u; j < numFiles; j++) {
                     ss << "    " << subnames[j] << " " << files[j];
                     if (j < numFiles - 1)
                        ss << '\n';
                  }
               }
               if (i < numFriends - 1)
                  ss << '\n';
            }
         }
         return ss.str();
      }
      // Datasource as input
      else if (fDataSource) {
         const auto datasourceLabel = fDataSource->GetLabel();
         return "Dataframe from datasource " + datasourceLabel;
      }
      // Trivial/empty datasource
      else {
         const auto n = fLoopManager->GetNEmptyEntries();
         if (n == 1) {
            return "Empty dataframe filling 1 row";
         } else {
            return "Empty dataframe filling " + std::to_string(n) + " rows";
         }
      }
   }

public:
   ////////////////////////////////////////////////////////////////////////////
   /// \brief Copy-assignment operator for RInterface.
   RInterface &operator=(const RInterface &) = default;

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Copy-ctor for RInterface.
   RInterface(const RInterface &) = default;

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Move-ctor for RInterface.
   RInterface(RInterface &&) = default;

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Move-assignment operator for RInterface.
   RInterface &operator=(RInterface &&) = default;

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Build a RInterface from a RLoopManager.
   /// This constructor is only available for RInterface<RLoopManager>.
   template <typename T = Proxied, typename = std::enable_if_t<std::is_same<T, RLoopManager>::value, int>>
   RInterface(const std::shared_ptr<RLoopManager> &proxied)
      : fProxiedPtr(proxied), fLoopManager(proxied.get()), fDataSource(proxied->GetDataSource()), fColRegister(proxied)
   {
      AddDefaultColumns();
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Cast any RDataFrame node to a common type ROOT::RDF::RNode.
   /// Different RDataFrame methods return different C++ types. All nodes, however,
   /// can be cast to this common type at the cost of a small performance penalty.
   /// This allows, for example, storing RDataFrame nodes in a vector, or passing them
   /// around via (non-template, C++11) helper functions.
   /// Example usage:
   /// ~~~{.cpp}
   /// // a function that conditionally adds a Range to a RDataFrame node.
   /// RNode MaybeAddRange(RNode df, bool mustAddRange)
   /// {
   ///    return mustAddRange ? df.Range(1) : df;
   /// }
   /// // use as :
   /// ROOT::RDataFrame df(10);
   /// auto maybeRanged = MaybeAddRange(df, true);
   /// ~~~
   /// Note that it is not a problem to pass RNode's by value.
   operator RNode() const
   {
      return RNode(std::static_pointer_cast<::ROOT::Detail::RDF::RNodeBase>(fProxiedPtr), *fLoopManager, fColRegister,
                   fDataSource);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Append a filter to the call graph.
   /// \param[in] f Function, lambda expression, functor class or any other callable object. It must return a `bool`
   /// signalling whether the event has passed the selection (true) or not (false).
   /// \param[in] columns Names of the columns/branches in input to the filter function.
   /// \param[in] name Optional name of this filter. See `Report`.
   /// \return the filter node of the computation graph.
   ///
   /// Append a filter node at the point of the call graph corresponding to the
   /// object this method is called on.
   /// The callable `f` should not have side-effects (e.g. modification of an
   /// external or static variable) to ensure correct results when implicit
   /// multi-threading is active.
   ///
   /// RDataFrame only evaluates filters when necessary: if multiple filters
   /// are chained one after another, they are executed in order and the first
   /// one returning false causes the event to be discarded.
   /// Even if multiple actions or transformations depend on the same filter,
   /// it is executed once per entry. If its result is requested more than
   /// once, the cached result is served.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// // C++ callable (function, functor class, lambda...) that takes two parameters of the types of "x" and "y"
   /// auto filtered = df.Filter(myCut, {"x", "y"});
   ///
   /// // String: it must contain valid C++ except that column names can be used instead of variable names
   /// auto filtered = df.Filter("x*y > 0");
   /// ~~~
   template <typename F, std::enable_if_t<!std::is_convertible<F, std::string>::value, int> = 0>
   RInterface<RDFDetail::RFilter<F, Proxied>, DS_t>
   Filter(F f, const ColumnNames_t &columns = {}, std::string_view name = "")
   {
      RDFInternal::CheckFilter(f);
      using ColTypes_t = typename TTraits::CallableTraits<F>::arg_types;
      constexpr auto nColumns = ColTypes_t::list_size;
      const auto validColumnNames = GetValidatedColumnNames(nColumns, columns);
      CheckAndFillDSColumns(validColumnNames, ColTypes_t());

      using F_t = RDFDetail::RFilter<F, Proxied>;

      auto filterPtr = std::make_shared<F_t>(std::move(f), validColumnNames, fProxiedPtr, fColRegister, name);
      return RInterface<F_t, DS_t>(std::move(filterPtr), *fLoopManager, fColRegister, fDataSource);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Append a filter to the call graph.
   /// \param[in] f Function, lambda expression, functor class or any other callable object. It must return a `bool`
   /// signalling whether the event has passed the selection (true) or not (false).
   /// \param[in] name Optional name of this filter. See `Report`.
   /// \return the filter node of the computation graph.
   ///
   /// Refer to the first overload of this method for the full documentation.
   template <typename F, std::enable_if_t<!std::is_convertible<F, std::string>::value, int> = 0>
   RInterface<RDFDetail::RFilter<F, Proxied>, DS_t> Filter(F f, std::string_view name)
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
   /// \return the filter node of the computation graph.
   ///
   /// Refer to the first overload of this method for the full documentation.
   template <typename F>
   RInterface<RDFDetail::RFilter<F, Proxied>, DS_t> Filter(F f, const std::initializer_list<std::string> &columns)
   {
      return Filter(f, ColumnNames_t{columns});
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Append a filter to the call graph.
   /// \param[in] expression The filter expression in C++
   /// \param[in] name Optional name of this filter. See `Report`.
   /// \return the filter node of the computation graph.
   ///
   /// The expression is just-in-time compiled and used to filter entries. It must
   /// be valid C++ syntax in which variable names are substituted with the names
   /// of branches/columns.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// auto filtered_df = df.Filter("myCollection.size() > 3");
   /// auto filtered_name_df = df.Filter("myCollection.size() > 3", "Minumum collection size");
   /// ~~~
   RInterface<RDFDetail::RJittedFilter, DS_t> Filter(std::string_view expression, std::string_view name = "")
   {
      // deleted by the jitted call to JitFilterHelper
      auto upcastNodeOnHeap = RDFInternal::MakeSharedOnHeap(RDFInternal::UpcastNode(fProxiedPtr));
      using BaseNodeType_t = typename std::remove_pointer_t<decltype(upcastNodeOnHeap)>::element_type;
      RInterface<BaseNodeType_t> upcastInterface(*upcastNodeOnHeap, *fLoopManager, fColRegister, fDataSource);
      const auto jittedFilter =
         RDFInternal::BookFilterJit(upcastNodeOnHeap, name, expression, fLoopManager->GetBranchNames(), fColRegister,
                                    fLoopManager->GetTree(), fDataSource);

      return RInterface<RDFDetail::RJittedFilter, DS_t>(std::move(jittedFilter), *fLoopManager, fColRegister,
                                                        fDataSource);
   }

   // clang-format off
   ////////////////////////////////////////////////////////////////////////////
   /// \brief Define a new column.
   /// \param[in] name The name of the defined column.
   /// \param[in] expression Function, lambda expression, functor class or any other callable object producing the defined value. Returns the value that will be assigned to the defined column.
   /// \param[in] columns Names of the columns/branches in input to the producer function.
   /// \return the first node of the computation graph for which the new quantity is defined.
   ///
   /// Define a column that will be visible from all subsequent nodes
   /// of the functional chain. The `expression` is only evaluated for entries that pass
   /// all the preceding filters.
   /// A new variable is created called `name`, accessible as if it was contained
   /// in the dataset from subsequent transformations/actions.
   ///
   /// Use cases include:
   /// * caching the results of complex calculations for easy and efficient multiple access
   /// * extraction of quantities of interest from complex objects
   ///
   /// An exception is thrown if the name of the new column is already in use in this branch of the computation graph.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// // assuming a function with signature:
   /// double myComplexCalculation(const RVec<float> &muon_pts);
   /// // we can pass it directly to Define
   /// auto df_with_define = df.Define("newColumn", myComplexCalculation, {"muon_pts"});
   /// // alternatively, we can pass the body of the function as a string, as in Filter:
   /// auto df_with_define = df.Define("newColumn", "x*x + y*y");
   /// ~~~
   template <typename F, typename std::enable_if_t<!std::is_convertible<F, std::string>::value, int> = 0>
   RInterface<Proxied, DS_t> Define(std::string_view name, F expression, const ColumnNames_t &columns = {})
   {
      return DefineImpl<F, RDFDetail::CustomColExtraArgs::None>(name, std::move(expression), columns, "Define");
   }
   // clang-format on

   // clang-format off
   ////////////////////////////////////////////////////////////////////////////
   /// \brief Define a new column with a value dependent on the processing slot.
   /// \param[in] name The name of the defined column.
   /// \param[in] expression Function, lambda expression, functor class or any other callable object producing the defined value. Returns the value that will be assigned to the defined column.
   /// \param[in] columns Names of the columns/branches in input to the producer function (excluding the slot number).
   /// \return the first node of the computation graph for which the new quantity is defined.
   ///
   /// This alternative implementation of `Define` is meant as a helper to evaluate new column values in a thread-safe manner.
   /// The expression must be a callable of signature R(unsigned int, T1, T2, ...) where `T1, T2...` are the types
   /// of the columns that the expression takes as input. The first parameter is reserved for an unsigned integer
   /// representing a "slot number". RDataFrame guarantees that different threads will invoke the expression with
   /// different slot numbers - slot numbers will range from zero to ROOT::GetThreadPoolSize()-1.
   ///
   /// The following two calls are equivalent, although `DefineSlot` is slightly more performant:
   /// ~~~{.cpp}
   /// int function(unsigned int, double, double);
   /// df.Define("x", function, {"rdfslot_", "column1", "column2"})
   /// df.DefineSlot("x", function, {"column1", "column2"})
   /// ~~~
   ///
   /// See Define for more information.
   template <typename F>
   RInterface<Proxied, DS_t> DefineSlot(std::string_view name, F expression, const ColumnNames_t &columns = {})
   {
      return DefineImpl<F, RDFDetail::CustomColExtraArgs::Slot>(name, std::move(expression), columns, "DefineSlot");
   }
   // clang-format on

   // clang-format off
   ////////////////////////////////////////////////////////////////////////////
   /// \brief Define a new column with a value dependent on the processing slot and the current entry.
   /// \param[in] name The name of the defined column.
   /// \param[in] expression Function, lambda expression, functor class or any other callable object producing the defined value. Returns the value that will be assigned to the defined column.
   /// \param[in] columns Names of the columns/branches in input to the producer function (excluding slot and entry).
   /// \return the first node of the computation graph for which the new quantity is defined.
   ///
   /// This alternative implementation of `Define` is meant as a helper in writing entry-specific, thread-safe custom
   /// columns. The expression must be a callable of signature R(unsigned int, ULong64_t, T1, T2, ...) where `T1, T2...`
   /// are the types of the columns that the expression takes as input. The first parameter is reserved for an unsigned
   /// integer representing a "slot number". RDataFrame guarantees that different threads will invoke the expression with
   /// different slot numbers - slot numbers will range from zero to ROOT::GetThreadPoolSize()-1. The second parameter
   /// is reserved for a `ULong64_t` representing the current entry being processed by the current thread.
   ///
   /// The following two `Define`s are equivalent, although `DefineSlotEntry` is slightly more performant:
   /// ~~~{.cpp}
   /// int function(unsigned int, ULong64_t, double, double);
   /// Define("x", function, {"rdfslot_", "rdfentry_", "column1", "column2"})
   /// DefineSlotEntry("x", function, {"column1", "column2"})
   /// ~~~
   ///
   /// See Define for more information.
   template <typename F>
   RInterface<Proxied, DS_t> DefineSlotEntry(std::string_view name, F expression, const ColumnNames_t &columns = {})
   {
      return DefineImpl<F, RDFDetail::CustomColExtraArgs::SlotAndEntry>(name, std::move(expression), columns,
                                                                        "DefineSlotEntry");
   }
   // clang-format on

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Define a new column.
   /// \param[in] name The name of the defined column.
   /// \param[in] expression An expression in C++ which represents the defined value
   /// \return the first node of the computation graph for which the new quantity is defined.
   ///
   /// The expression is just-in-time compiled and used to produce the column entries.
   /// It must be valid C++ syntax in which variable names are substituted with the names
   /// of branches/columns.
   ///
   /// Refer to the first overload of this method for the full documentation.
   RInterface<Proxied, DS_t> Define(std::string_view name, std::string_view expression)
   {
      constexpr auto where = "Define";
      RDFInternal::CheckValidCppVarName(name, where);
      // these checks must be done before jitting lest we throw exceptions in jitted code
      RDFInternal::CheckForRedefinition(where, name, fColRegister, fLoopManager->GetBranchNames(),
                                        fDataSource ? fDataSource->GetColumnNames() : ColumnNames_t{});

      auto upcastNodeOnHeap = RDFInternal::MakeSharedOnHeap(RDFInternal::UpcastNode(fProxiedPtr));
      auto jittedDefine = RDFInternal::BookDefineJit(name, expression, *fLoopManager, fDataSource, fColRegister,
                                                     fLoopManager->GetBranchNames(), upcastNodeOnHeap);

      RDFInternal::RColumnRegister newCols(fColRegister);
      newCols.AddColumn(jittedDefine);

      RInterface<Proxied, DS_t> newInterface(fProxiedPtr, *fLoopManager, std::move(newCols), fDataSource);

      return newInterface;
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Overwrite the value and/or type of an existing column.
   /// \param[in] name The name of the column to redefine.
   /// \param[in] expression Function, lambda expression, functor class or any other callable object producing the defined value. Returns the value that will be assigned to the defined column.
   /// \param[in] columns Names of the columns/branches in input to the expression.
   /// \return the first node of the computation graph for which the quantity is redefined.
   ///
   /// The old value of the column can be used as an input for the expression.
   ///
   /// An exception is thrown in case the column to redefine does not already exist.
   /// See Define() for more information.
   template <typename F, std::enable_if_t<!std::is_convertible<F, std::string>::value, int> = 0>
   RInterface<Proxied, DS_t> Redefine(std::string_view name, F expression, const ColumnNames_t &columns = {})
   {
      return DefineImpl<F, RDFDetail::CustomColExtraArgs::None>(name, std::move(expression), columns, "Redefine");
   }

   // clang-format off
   ////////////////////////////////////////////////////////////////////////////
   /// \brief Overwrite the value and/or type of an existing column.
   /// \param[in] name The name of the column to redefine.
   /// \param[in] expression Function, lambda expression, functor class or any other callable object producing the defined value. Returns the value that will be assigned to the defined column.
   /// \param[in] columns Names of the columns/branches in input to the producer function (excluding slot).
   /// \return the first node of the computation graph for which the new quantity is defined.
   ///
   /// The old value of the column can be used as an input for the expression.
   /// An exception is thrown in case the column to redefine does not already exist.
   ///
   /// See DefineSlot() for more information.
   // clang-format on
   template <typename F>
   RInterface<Proxied, DS_t> RedefineSlot(std::string_view name, F expression, const ColumnNames_t &columns = {})
   {
      return DefineImpl<F, RDFDetail::CustomColExtraArgs::Slot>(name, std::move(expression), columns, "RedefineSlot");
   }

   // clang-format off
   ////////////////////////////////////////////////////////////////////////////
   /// \brief Overwrite the value and/or type of an existing column.
   /// \param[in] name The name of the column to redefine.
   /// \param[in] expression Function, lambda expression, functor class or any other callable object producing the defined value. Returns the value that will be assigned to the defined column.
   /// \param[in] columns Names of the columns/branches in input to the producer function (excluding slot and entry).
   /// \return the first node of the computation graph for which the new quantity is defined.
   ///
   /// The old value of the column can be used as an input for the expression.
   /// An exception is thrown in case the column to re-define does not already exist.
   ///
   /// See DefineSlotEntry() for more information.
   // clang-format on
   template <typename F>
   RInterface<Proxied, DS_t> RedefineSlotEntry(std::string_view name, F expression, const ColumnNames_t &columns = {})
   {
      return DefineImpl<F, RDFDetail::CustomColExtraArgs::SlotAndEntry>(name, std::move(expression), columns,
                                                                        "RedefineSlotEntry");
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Overwrite the value and/or type of an existing column.
   /// \param[in] name The name of the column to redefine.
   /// \param[in] expression An expression in C++ which represents the defined value
   /// \return the first node of the computation graph for which the new quantity is defined.
   ///
   /// The expression is just-in-time compiled and used to produce the column entries.
   /// It must be valid C++ syntax in which variable names are substituted with the names
   /// of branches/columns.
   ///
   /// The old value of the column can be used as an input for the expression.
   /// An exception is thrown in case the column to re-define does not already exist.
   ///
   /// Aliases cannot be overridden. See the corresponding Define() overload for more information.
   RInterface<Proxied, DS_t> Redefine(std::string_view name, std::string_view expression)
   {
      constexpr auto where = "Redefine";
      RDFInternal::CheckValidCppVarName(name, where);
      RDFInternal::CheckForDefinition(where, name, fColRegister, fLoopManager->GetBranchNames(),
                                      fDataSource ? fDataSource->GetColumnNames() : ColumnNames_t{});
      RDFInternal::CheckForNoVariations(where, name, fColRegister);

      auto upcastNodeOnHeap = RDFInternal::MakeSharedOnHeap(RDFInternal::UpcastNode(fProxiedPtr));
      auto jittedDefine = RDFInternal::BookDefineJit(name, expression, *fLoopManager, fDataSource, fColRegister,
                                                     fLoopManager->GetBranchNames(), upcastNodeOnHeap);

      RDFInternal::RColumnRegister newCols(fColRegister);
      newCols.AddColumn(jittedDefine);

      RInterface<Proxied, DS_t> newInterface(fProxiedPtr, *fLoopManager, std::move(newCols), fDataSource);

      return newInterface;
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Define a new column that is updated when the input sample changes.
   /// \param[in] name The name of the defined column.
   /// \param[in] expression A C++ callable that computes the new value of the defined column.
   /// \return the first node of the computation graph for which the new quantity is defined.
   ///
   /// The signature of the callable passed as second argument should be `T(unsigned int slot, const ROOT::RDF::RSampleInfo &id)`
   /// where:
   /// - `T` is the type of the defined column
   /// - `slot` is a number in the range [0, nThreads) that is different for each processing thread. This can simplify
   ///   the definition of thread-safe callables if you are interested in using parallel capabilities of RDataFrame.
   /// - `id` is an instance of a ROOT::RDF::RSampleInfo object which contains information about the sample which is
   ///   being processed (see the class docs for more information).
   ///
   /// DefinePerSample() is useful to e.g. define a quantity that depends on which TTree in which TFile is being
   /// processed or to inject a callback into the event loop that is only called when the processing of a new sample
   /// starts rather than at every entry.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// ROOT::RDataFrame df{"mytree", {"sample1.root","sample2.root"}};
   /// df.DefinePerSample("weightbysample",
   ///                    [](unsigned int slot, const ROOT::RDF::RSampleInfo &id)
   ///                    { return id.Contains("sample1") ? 1.0f : 2.0f; });
   /// ~~~
   // TODO we could SFINAE on F's signature to provide friendlier compilation errors in case of signature mismatch
   template <typename F, typename RetType_t = typename TTraits::CallableTraits<F>::ret_type>
   RInterface<Proxied, DS_t> DefinePerSample(std::string_view name, F expression)
   {
      RDFInternal::CheckValidCppVarName(name, "DefinePerSample");
      RDFInternal::CheckForRedefinition("DefinePerSample", name, fColRegister, fLoopManager->GetBranchNames(),
                                        fDataSource ? fDataSource->GetColumnNames() : ColumnNames_t{});

      auto retTypeName = RDFInternal::TypeID2TypeName(typeid(RetType_t));
      if (retTypeName.empty()) {
         // The type is not known to the interpreter.
         // We must not error out here, but if/when this column is used in jitted code
         const auto demangledType = RDFInternal::DemangleTypeIdName(typeid(RetType_t));
         retTypeName = "CLING_UNKNOWN_TYPE_" + demangledType;
      }

      auto newColumn =
         std::make_shared<RDFDetail::RDefinePerSample<F>>(name, retTypeName, std::move(expression), *fLoopManager);

      RDFInternal::RColumnRegister newCols(fColRegister);
      newCols.AddColumn(std::move(newColumn));
      RInterface<Proxied> newInterface(fProxiedPtr, *fLoopManager, std::move(newCols), fDataSource);
      return newInterface;
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Define a new column that is updated when the input sample changes.
   /// \param[in] name The name of the defined column.
   /// \param[in] expression A valid C++ expression as a string, which will be used to compute the defined value.
   /// \return the first node of the computation graph for which the new quantity is defined.
   ///
   /// The expression is just-in-time compiled and used to produce the column entries.
   /// It must be valid C++ syntax and the usage of the special variable names `rdfslot_` and `rdfsampleinfo_` is
   /// permitted, where these variables will take the same values as the `slot` and `id` parameters described at the
   /// DefinePerSample(std::string_view name, F expression) overload. See the documentation of that overload for more information.
   ///
   /// ### Example usage:
   /// ~~~{.py}
   /// df = ROOT.RDataFrame("mytree", ["sample1.root","sample2.root"])
   /// df.DefinePerSample("weightbysample", "rdfsampleinfo_.Contains('sample1') ? 1.0f : 2.0f")
   /// ~~~
   ///
   /// \note
   /// If you have declared some C++ function to the interpreter, the correct syntax to call that function with this
   /// overload of DefinePerSample is by calling it explicitly with the special names `rdfslot_` and `rdfsampleinfo_` as
   /// input parameters. This is for example the correct way to call this overload when working in PyROOT:
   /// ~~~{.py}
   /// ROOT.gInterpreter.Declare(
   /// """
   /// float weights(unsigned int slot, const ROOT::RDF::RSampleInfo &id){
   ///    return id.Contains("sample1") ? 1.0f : 2.0f;
   /// }
   /// """)
   /// df = ROOT.RDataFrame("mytree", ["sample1.root","sample2.root"])
   /// df.DefinePerSample("weightsbysample", "weights(rdfslot_, rdfsampleinfo_)")
   /// ~~~
   RInterface<Proxied, DS_t> DefinePerSample(std::string_view name, std::string_view expression)
   {
      RDFInternal::CheckValidCppVarName(name, "DefinePerSample");
      // these checks must be done before jitting lest we throw exceptions in jitted code
      RDFInternal::CheckForRedefinition("DefinePerSample", name, fColRegister, fLoopManager->GetBranchNames(),
                                        fDataSource ? fDataSource->GetColumnNames() : ColumnNames_t{});

      auto upcastNodeOnHeap = RDFInternal::MakeSharedOnHeap(RDFInternal::UpcastNode(fProxiedPtr));
      auto jittedDefine =
         RDFInternal::BookDefinePerSampleJit(name, expression, *fLoopManager, fColRegister, upcastNodeOnHeap);

      RDFInternal::RColumnRegister newCols(fColRegister);
      newCols.AddColumn(jittedDefine);

      RInterface<Proxied, DS_t> newInterface(fProxiedPtr, *fLoopManager, std::move(newCols), fDataSource);

      return newInterface;
   }

   /// \brief Register systematic variations for an existing column.
   /// \param[in] colNames names of the columns for which varied values are provided.
   /// \param[in] expression a callable that evaluates the varied values for the specified columns. The callable can
   ///            take any column values as input, similarly to what happens with Filter and Define calls. It must
   ///            return an RVec of varied values, one for each variation tag, in the same order as the tags.
   /// \param[in] inputColumns the names of the columns to be passed to the callable.
   /// \param[in] variationTags names for each of the varied values, e.g. "up" and "down".
   /// \param[in] variationName a generic name for this set of varied values, e.g. "ptvariation".
   ///
   /// Vary provides a natural and flexible syntax to define systematic variations that automatically propagate to
   /// Filters, Defines and results. RDataFrame usage of columns with attached variations does not change, but for
   /// results that depend on any varied quantity a map/dictionary of varied results can be produced with
   /// ROOT::RDF::Experimental::VariationsFor (see the example below).
   ///
   /// The dictionary will contain a "nominal" value (accessed with the "nominal" key) for the unchanged result, and
   /// values for each of the systematic variations that affected the result (via upstream Filters or via direct or
   /// indirect dependencies of the column values on some registered variations). The keys will be a composition of
   /// variation names and tags, e.g. "pt:up" and "pt:down" for the example below.
   ///
   /// In the following example we add up/down variations of pt and fill a histogram with a quantity that depends on pt.
   /// We automatically obtain three histograms in output ("nominal", "pt:up" and "pt:down"):
   /// ~~~{.cpp}
   /// auto nominal_hx =
   ///     df.Vary("pt", [] (double pt) { return RVecD{pt*0.9, pt*1.1}; }, {"down", "up"})
   ///       .Filter("pt > k")
   ///       .Define("x", someFunc, {"pt"})
   ///       .Histo1D("x");
   ///
   /// auto hx = ROOT::RDF::VariationsFor(nominal_hx);
   /// hx["nominal"].Draw();
   /// hx["pt:down"].Draw("SAME");
   /// ~~~
   template <typename F>
   RInterface<Proxied, DS_t> Vary(std::string_view colName, F &&expression, const ColumnNames_t &inputColumns,
                                  std::size_t nVariations, std::string_view variationName = "")
   {
      R__ASSERT(nVariations > 0 && "Must have at least one variation.");

      std::vector<std::string> variationTags;
      variationTags.reserve(nVariations);
      for (std::size_t i = 0u; i < nVariations; ++i)
         variationTags.emplace_back(std::to_string(i));

      const std::string theVariationName{variationName.empty() ? colName : variationName};

      return Vary(colName, std::forward<F>(expression), inputColumns, std::move(variationTags), theVariationName);
   }

   /// \brief Register systematic variations for an existing columns using auto-generated variation tags.
   /// This overload of Vary takes a nVariations parameter instead of a list of tag names. Tag names
   /// will be auto-generated as the sequence 0...nVariations-1.
   /// See the documentation of the previous overload for more information.
   template <typename F>
   RInterface<Proxied, DS_t> Vary(std::string_view colName, F &&expression, const ColumnNames_t &inputColumns,
                                  const std::vector<std::string> &variationTags, std::string_view variationName = "")
   {
      std::vector<std::string> colNames{{std::string(colName)}};
      const std::string theVariationName{variationName.empty() ? colName : variationName};

      return Vary(std::move(colNames), std::forward<F>(expression), inputColumns, variationTags, theVariationName);
   }

   /// \brief Register a systematic variation that affects multiple columns simultaneously.
   /// This overload of Vary takes a list of column names as first argument rather than a single name and
   /// requires that the expression returns an RVec of RVecs of values: one inner RVec for the variations of each
   /// affected column.
   /// See the documentation of the first Vary overload for more information.
   ///
   /// Example usage:
   /// ~~~{.cpp}
   /// // produce variations "ptAndEta:down" and "ptAndEta:up"
   /// df.Vary({"pt", "eta"},
   ///         [](double pt, double eta) { return RVec<RVecF>{{pt*0.9, pt*1.1}, {eta*0.9, eta*1.1}}; },
   ///         {"down", "up"},
   ///         "ptAndEta");
   /// ~~~
   template <typename F>
   RInterface<Proxied, DS_t>
   Vary(const std::vector<std::string> &colNames, F &&expression, const ColumnNames_t &inputColumns,
        const std::vector<std::string> &variationTags, std::string_view variationName)
   {
      /* SANITY CHECKS BEGIN */
      R__ASSERT(variationTags.size() > 0 && "Must have at least one variation.");
      R__ASSERT(colNames.size() > 0 && "Must have at least one varied column.");
      R__ASSERT(!variationName.empty() && "Must provide a variation name.");

      for (auto &colName : colNames) {
         RDFInternal::CheckValidCppVarName(colName, "Vary");
         RDFInternal::CheckForDefinition("Vary", colName, fColRegister, fLoopManager->GetBranchNames(),
                                         fDataSource ? fDataSource->GetColumnNames() : ColumnNames_t{});
      }
      RDFInternal::CheckValidCppVarName(variationName, "Vary");

      using F_t = std::decay_t<F>;
      using ColTypes_t = typename TTraits::CallableTraits<F_t>::arg_types;
      constexpr auto nColumns = ColTypes_t::list_size;
      const auto validColumnNames = GetValidatedColumnNames(nColumns, inputColumns);
      CheckAndFillDSColumns(validColumnNames, ColTypes_t{});

      using RetType = typename TTraits::CallableTraits<F_t>::ret_type;
      static_assert(RDFInternal::IsRVec<RetType>::value, "Vary expressions must return an RVec.");

      if (colNames.size() > 1) {
         constexpr bool hasInnerRVec = RDFInternal::IsRVec<typename RetType::value_type>::value;
         if (!hasInnerRVec)
            throw std::runtime_error("This Vary call is varying multiple columns simultaneously but the expression "
                                     "does not return an RVec of RVecs.");

         auto colTypes = GetColumnTypeNamesList(colNames);
         auto allColTypesEqual =
            std::all_of(colTypes.begin() + 1, colTypes.end(), [&](const std::string &t) { return t == colTypes[0]; });
         if (!allColTypesEqual)
            throw std::runtime_error("Cannot simultaneously vary multiple columns of different types.");

         const auto &innerTypeID = typeid(RDFInternal::InnerValueType_t<RetType>);

         const auto &definesMap = fColRegister.GetColumns();
         for (auto i = 0u; i < colTypes.size(); ++i) {
            const auto it = definesMap.find(colNames[i]);
            const auto &expectedTypeID =
               it == definesMap.end() ? RDFInternal::TypeName2TypeID(colTypes[i]) : it->second->GetTypeId();
            if (innerTypeID != expectedTypeID)
               throw std::runtime_error("Varied values for column \"" + colNames[i] + "\" have a different type (" +
                                        RDFInternal::TypeID2TypeName(innerTypeID) + ") than the nominal value (" +
                                        colTypes[i] + ").");
         }
      } else {
         const auto &retTypeID = typeid(typename RetType::value_type);
         const auto &colName = colNames[0]; // we have only one element in there
         const auto &definesMap = fColRegister.GetColumns();
         const auto it = definesMap.find(colName);
         const auto &expectedTypeID =
            it == definesMap.end() ? RDFInternal::TypeName2TypeID(GetColumnType(colName)) : it->second->GetTypeId();
         if (retTypeID != expectedTypeID)
            throw std::runtime_error("Varied values for column \"" + colName + "\" have a different type (" +
                                     RDFInternal::TypeID2TypeName(retTypeID) + ") than the nominal value (" +
                                     GetColumnType(colName) + ").");
      }

      auto retTypeName = RDFInternal::TypeID2TypeName(typeid(RetType));
      if (retTypeName.empty()) {
         // The type is not known to the interpreter, but we don't want to error out
         // here, rather if/when this column is used in jitted code, so we inject a broken but telling type name.
         const auto demangledType = RDFInternal::DemangleTypeIdName(typeid(RetType));
         retTypeName = "CLING_UNKNOWN_TYPE_" + demangledType;
      }

      // when varying multiple columns, they must be different columns
      if (colNames.size() > 1) {
         std::set<std::string> uniqueCols(colNames.begin(), colNames.end());
         if (uniqueCols.size() != colNames.size())
            throw std::logic_error("The same column was passed multiple times the same column twice: ");
      }
      /* SANITY CHECKS END */

      auto variation = std::make_shared<RDFInternal::RVariation<F_t>>(
         colNames, variationName, std::forward<F>(expression), variationTags, retTypeName, fColRegister, *fLoopManager,
         validColumnNames);

      RDFInternal::RColumnRegister newCols(fColRegister);
      newCols.AddVariation(variation);

      RInterface<Proxied> newInterface(fProxiedPtr, *fLoopManager, std::move(newCols), fDataSource);

      return newInterface;
   }

   /// \brief Register systematic variations for one or more existing columns using auto-generated tags.
   /// This overload of Vary takes a nVariations parameter instead of a list of tag names. Tag names
   /// will be auto-generated as the sequence 0...nVariations-1.
   /// See the documentation of the previous overload for more information.
   template <typename F>
   RInterface<Proxied, DS_t>
   Vary(const std::vector<std::string> &colNames, F &&expression, const ColumnNames_t &inputColumns,
        std::size_t nVariations, std::string_view variationName)
   {
      R__ASSERT(nVariations > 0 && "Must have at least one variation.");

      std::vector<std::string> variationTags;
      variationTags.reserve(nVariations);
      for (std::size_t i = 0u; i < nVariations; ++i)
         variationTags.emplace_back(std::to_string(i));

      return Vary(colNames, std::forward<F>(expression), inputColumns, std::move(variationTags), variationName);
   }

   /// \brief Register systematic variations for an existing column.
   /// \param[in] colName name of the column for which varied values are provided.
   /// \param[in] expression a string containing valid C++ code that evaluates to an RVec containing the varied
   ///            values for the specified column.
   /// \param[in] variationTags names for each of the varied values, e.g. "up" and "down".
   /// \param[in] variationName a generic name for this set of varied values, e.g. "ptvariation".
   ///            colName is used if none is provided.
   ///
   /// ~~~{.cpp}
   /// auto nominal_hx =
   ///     df.Vary("pt", "ROOT::RVecD{pt*0.9, pt*1.1}", {"down", "up"})
   ///       .Filter("pt > k")
   ///       .Define("x", someFunc, {"pt"})
   ///       .Histo1D("x");
   ///
   /// auto hx = ROOT::RDF::VariationsFor(nominal_hx);
   /// hx["nominal"].Draw();
   /// hx["pt:down"].Draw("SAME");
   /// ~~~
   RInterface<Proxied, DS_t> Vary(std::string_view colName, std::string_view expression,
                                  const std::vector<std::string> &variationTags, std::string_view variationName = "")
   {
      std::vector<std::string> colNames{{std::string(colName)}};
      const std::string theVariationName{variationName.empty() ? colName : variationName};

      return Vary(std::move(colNames), expression, variationTags, theVariationName);
   }

   /// \brief Register systematic variations for an existing column.
   /// \param[in] colName name of the column for which varied values are provided.
   /// \param[in] expression a string containing valid C++ code that evaluates to an RVec containing the varied
   ///            values for the specified column.
   /// \param[in] nVariations number of variations returned by the expression. The corresponding tags will be "0", "1", etc.
   /// \param[in] variationName a generic name for this set of varied values, e.g. "ptvariation".
   ///            colName is used if none is provided.
   ///
   /// See the documentation for the previous overload for more information.
   RInterface<Proxied, DS_t> Vary(std::string_view colName, std::string_view expression, std::size_t nVariations,
                                  std::string_view variationName = "")
   {
      std::vector<std::string> colNames{{std::string(colName)}};
      const std::string theVariationName{variationName.empty() ? colName : variationName};

      return Vary(std::move(colNames), expression, nVariations, theVariationName);
   }

   /// \brief Register systematic variations for one or more existing columns.
   /// \param[in] colNames names of the columns for which varied values are provided.
   /// \param[in] expression a string containing valid C++ code that evaluates to an RVec or RVecs containing the varied
   ///            values for the specified columns.
   /// \param[in] nVariations number of variations returned by the expression. The corresponding tags will be "0", "1", etc.
   /// \param[in] variationName a generic name for this set of varied values, e.g. "ptvariation".
   ///
   /// ~~~{.cpp}
   /// auto nominal_hx =
   ///     df.Vary({"x", "y"}, "ROOT::RVec<ROOT::RVecD>{{x*0.9, x*1.1}, {y*0.9, y*1.1}}", 2, "xy")
   ///       .Histo1D("x", "y");
   ///
   /// auto hx = ROOT::RDF::VariationsFor(nominal_hx);
   /// hx["nominal"].Draw();
   /// hx["xy:0"].Draw("SAME");
   /// hx["xy:1"].Draw("SAME");
   /// ~~~
   RInterface<Proxied, DS_t> Vary(const std::vector<std::string> &colNames, std::string_view expression,
                                  std::size_t nVariations, std::string_view variationName)
   {
      std::vector<std::string> variationTags;
      variationTags.reserve(nVariations);
      for (std::size_t i = 0u; i < nVariations; ++i)
         variationTags.emplace_back(std::to_string(i));

      return Vary(colNames, expression, std::move(variationTags), variationName);
   }

   /// \brief Register systematic variations for one or more existing columns.
   /// \param[in] colNames names of the columns for which varied values are provided.
   /// \param[in] expression a string containing valid C++ code that evaluates to an RVec or RVecs containing the varied
   ///            values for the specified columns.
   /// \param[in] variationTags names for each of the varied values, e.g. "up" and "down".
   /// \param[in] variationName a generic name for this set of varied values, e.g. "ptvariation".
   ///
   /// ~~~{.cpp}
   /// auto nominal_hx =
   ///     df.Vary({"x", "y"}, "ROOT::RVec<ROOT::RVecD>{{x*0.9, x*1.1}, {y*0.9, y*1.1}}", {"down", "up"}, "xy")
   ///       .Histo1D("x", "y");
   ///
   /// auto hx = ROOT::RDF::VariationsFor(nominal_hx);
   /// hx["nominal"].Draw();
   /// hx["xy:down"].Draw("SAME");
   /// hx["xy:up"].Draw("SAME");
   /// ~~~
   RInterface<Proxied, DS_t> Vary(const std::vector<std::string> &colNames, std::string_view expression,
                                  const std::vector<std::string> &variationTags, std::string_view variationName)
   {
      R__ASSERT(variationTags.size() > 0 && "Must have at least one variation.");
      R__ASSERT(colNames.size() > 0 && "Must have at least one varied column.");
      R__ASSERT(!variationName.empty() && "Must provide a variation name.");

      for (auto &colName : colNames) {
         RDFInternal::CheckValidCppVarName(colName, "Vary");
         RDFInternal::CheckForDefinition("Vary", colName, fColRegister, fLoopManager->GetBranchNames(),
                                         fDataSource ? fDataSource->GetColumnNames() : ColumnNames_t{});
      }
      RDFInternal::CheckValidCppVarName(variationName, "Vary");

      // when varying multiple columns, they must be different columns
      if (colNames.size() > 1) {
         std::set<std::string> uniqueCols(colNames.begin(), colNames.end());
         if (uniqueCols.size() != colNames.size())
            throw std::logic_error("The same column was passed multiple times the same column twice: ");
      }

      auto upcastNodeOnHeap = RDFInternal::MakeSharedOnHeap(RDFInternal::UpcastNode(fProxiedPtr));
      auto jittedVariation =
         RDFInternal::BookVariationJit(colNames, variationName, variationTags, expression, *fLoopManager, fDataSource,
                                       fColRegister, fLoopManager->GetBranchNames(), upcastNodeOnHeap);

      RDFInternal::RColumnRegister newColRegister(fColRegister);
      newColRegister.AddVariation(std::move(jittedVariation));

      RInterface<Proxied, DS_t> newInterface(fProxiedPtr, *fLoopManager, std::move(newColRegister), fDataSource);

      return newInterface;
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Allow to refer to a column with a different name.
   /// \param[in] alias name of the column alias
   /// \param[in] columnName of the column to be aliased
   /// \return the first node of the computation graph for which the alias is available.
   ///
   /// Aliasing an alias is supported.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// auto df_with_alias = df.Alias("simple_name", "very_long&complex_name!!!");
   /// ~~~
   RInterface<Proxied, DS_t> Alias(std::string_view alias, std::string_view columnName)
   {
      // The symmetry with Define is clear. We want to:
      // - Create globally the alias and return this very node, unchanged
      // - Make aliases accessible based on chains and not globally

      // Helper to find out if a name is a column
      auto &dsColumnNames = fDataSource ? fDataSource->GetColumnNames() : ColumnNames_t{};

      constexpr auto where = "Alias";
      RDFInternal::CheckValidCppVarName(alias, where);
      // If the alias name is a column name, there is a problem
      RDFInternal::CheckForRedefinition(where, alias, fColRegister, fLoopManager->GetBranchNames(), dsColumnNames);

      const auto validColumnName = GetValidatedColumnNames(1, {std::string(columnName)})[0];

      RDFInternal::RColumnRegister newCols(fColRegister);
      newCols.AddAlias(alias, validColumnName);

      RInterface<Proxied, DS_t> newInterface(fProxiedPtr, *fLoopManager, std::move(newCols), fDataSource);

      return newInterface;
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Save selected columns to disk, in a new TTree `treename` in file `filename`.
   /// \tparam ColumnTypes variadic list of branch/column types.
   /// \param[in] treename The name of the output TTree.
   /// \param[in] filename The name of the output TFile.
   /// \param[in] columnList The list of names of the columns/branches to be written.
   /// \param[in] options RSnapshotOptions struct with extra options to pass to TFile and TTree.
   /// \return a `RDataFrame` that wraps the snapshotted dataset.
   ///
   /// Support for writing of nested branches is limited (although RDataFrame is able to read them) and dot ('.')
   /// characters in input column names will be replaced by underscores ('_') in the branches produced by Snapshot.
   /// When writing a variable size array through Snapshot, it is required that the column indicating its size is also
   /// written out and it appears before the array in the columnList.
   ///
   /// By default, in case of TTree or TChain inputs, Snapshot will try to write out all top-level branches. For other
   /// types of inputs, all columns returned by GetColumnNames() will be written out. If friend trees or chains are
   /// present, by default all friend top-level branches that have names that do not collide with
   /// names of branches in the main TTree/TChain will be written out. Since v6.24, Snapshot will also write out
   /// friend branches with the same names of branches in the main TTree/TChain with names of the form
   /// '<friendname>_<branchname>' in order to differentiate them from the branches in the main tree/chain.
   ///
   /// ### Writing to a sub-directory
   ///
   /// Snapshot supports writing the TTree in a sub-directory inside the TFile. It is sufficient to specify the path to
   /// the TTree as part of the TTree name, e.g. `df.Snapshot("subdir/t", "f.root")` write TTree `t` in the
   /// sub-directory `subdir` of file `f.root` (creating file and sub-directory as needed).
   ///
   /// \attention In multi-thread runs (i.e. when EnableImplicitMT() has been called) threads will loop over clusters of
   /// entries in an undefined order, so Snapshot will produce outputs in which (clusters of) entries will be shuffled with
   /// respect to the input TTree. Using such "shuffled" TTrees as friends of the original trees would result in wrong
   /// associations between entries in the main TTree and entries in the "shuffled" friend. Since v6.22, ROOT will
   /// error out if such a "shuffled" TTree is used in a friendship.
   ///
   /// \note In case no events are written out (e.g. because no event passes all filters) the behavior of Snapshot in
   /// single-thread and multi-thread runs is different: in single-thread runs, Snapshot will write out a TTree with
   /// the specified name and zero entries; in multi-thread runs, no TTree object will be written out to disk.
   ///
   /// \note Snapshot will refuse to process columns with names of the form `#columnname`. These are special columns
   /// made available by some data sources (e.g. RNTupleDS) that represent the size of column `columnname`, and are
   /// not meant to be written out with that name (which is not a valid C++ variable name). Instead, go through an
   /// Alias(): `df.Alias("nbar", "#bar").Snapshot(..., {"nbar"})`.
   ///
   /// ### Example invocations:
   ///
   /// ~~~{.cpp}
   /// // without specifying template parameters (column types automatically deduced)
   /// df.Snapshot("outputTree", "outputFile.root", {"x", "y"});
   ///
   /// // specifying template parameters ("x" is `int`, "y" is `float`)
   /// df.Snapshot<int, float>("outputTree", "outputFile.root", {"x", "y"});
   /// ~~~
   ///
   /// To book a Snapshot without triggering the event loop, one needs to set the appropriate flag in
   /// `RSnapshotOptions`:
   /// ~~~{.cpp}
   /// RSnapshotOptions opts;
   /// opts.fLazy = true;
   /// df.Snapshot("outputTree", "outputFile.root", {"x"}, opts);
   /// ~~~
   template <typename... ColumnTypes>
   RResultPtr<RInterface<RLoopManager>>
   Snapshot(std::string_view treename, std::string_view filename, const ColumnNames_t &columnList,
            const RSnapshotOptions &options = RSnapshotOptions())
   {
      return SnapshotImpl<ColumnTypes...>(treename, filename, columnList, options);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Save selected columns to disk, in a new TTree `treename` in file `filename`.
   /// \param[in] treename The name of the output TTree.
   /// \param[in] filename The name of the output TFile.
   /// \param[in] columnList The list of names of the columns/branches to be written.
   /// \param[in] options RSnapshotOptions struct with extra options to pass to TFile and TTree.
   /// \return a `RDataFrame` that wraps the snapshotted dataset.
   ///
   /// This function returns a `RDataFrame` built with the output tree as a source.
   /// The types of the columns are automatically inferred and do not need to be specified.
   ///
   /// See above for a more complete description and example usages.
   RResultPtr<RInterface<RLoopManager>> Snapshot(std::string_view treename, std::string_view filename,
                                                 const ColumnNames_t &columnList,
                                                 const RSnapshotOptions &options = RSnapshotOptions())
   {
      // like columnList but with `#var` columns removed
      auto colListNoPoundSizes = RDFInternal::FilterArraySizeColNames(columnList, "Snapshot");
      // like columnListWithoutSizeColumns but with aliases resolved
      auto colListNoAliases = GetValidatedColumnNames(colListNoPoundSizes.size(), colListNoPoundSizes);
      RDFInternal::CheckForDuplicateSnapshotColumns(colListNoAliases);
      // like validCols but with missing size branches required by array branches added in the right positions
      const auto pairOfColumnLists =
         RDFInternal::AddSizeBranches(fLoopManager->GetBranchNames(), fLoopManager->GetTree(),
                                      std::move(colListNoAliases), std::move(colListNoPoundSizes));
      const auto &colListNoAliasesWithSizeBranches = pairOfColumnLists.first;
      const auto &colListWithAliasesAndSizeBranches = pairOfColumnLists.second;


      const auto fullTreeName = treename;
      const auto parsedTreePath = RDFInternal::ParseTreePath(fullTreeName);
      treename = parsedTreePath.fTreeName;
      const auto &dirname = parsedTreePath.fDirName;

      auto snapHelperArgs = std::make_shared<RDFInternal::SnapshotHelperArgs>(
         RDFInternal::SnapshotHelperArgs{std::string(filename), std::string(dirname), std::string(treename),
                                         colListWithAliasesAndSizeBranches, options});

      ::TDirectory::TContext ctxt;
      auto newRDF = std::make_shared<ROOT::RDataFrame>(fullTreeName, filename, colListNoAliasesWithSizeBranches);

      auto resPtr = CreateAction<RDFInternal::ActionTags::Snapshot, RDFDetail::RInferredType>(
         colListNoAliasesWithSizeBranches, newRDF, snapHelperArgs, colListNoAliasesWithSizeBranches.size());

      if (!options.fLazy)
         *resPtr;
      return resPtr;
   }

   // clang-format off
   ////////////////////////////////////////////////////////////////////////////
   /// \brief Save selected columns to disk, in a new TTree `treename` in file `filename`.
   /// \param[in] treename The name of the output TTree.
   /// \param[in] filename The name of the output TFile.
   /// \param[in] columnNameRegexp The regular expression to match the column names to be selected. The presence of a '^' and a '$' at the end of the string is implicitly assumed if they are not specified. The dialect supported is PCRE via the TPRegexp class. An empty string signals the selection of all columns.
   /// \param[in] options RSnapshotOptions struct with extra options to pass to TFile and TTree
   /// \return a `RDataFrame` that wraps the snapshotted dataset.
   ///
   /// This function returns a `RDataFrame` built with the output tree as a source.
   /// The types of the columns are automatically inferred and do not need to be specified.
   ///
   /// See above for a more complete description and example usages.
   RResultPtr<RInterface<RLoopManager>> Snapshot(std::string_view treename, std::string_view filename,
                                                 std::string_view columnNameRegexp = "",
                                                 const RSnapshotOptions &options = RSnapshotOptions())
   {
      const auto definedColumns = fColRegister.GetNames();
      auto *tree = fLoopManager->GetTree();
      const auto treeBranchNames = tree != nullptr ? RDFInternal::GetTopLevelBranchNames(*tree) : ColumnNames_t{};
      const auto dsColumns = fDataSource ? fDataSource->GetColumnNames() : ColumnNames_t{};
      // Ignore R_rdf_sizeof_* columns coming from datasources: we don't want to Snapshot those
      ColumnNames_t dsColumnsWithoutSizeColumns;
      std::copy_if(dsColumns.begin(), dsColumns.end(), std::back_inserter(dsColumnsWithoutSizeColumns),
                   [](const std::string &name) { return name.size() < 13 || name.substr(0, 13) != "R_rdf_sizeof_"; });
      ColumnNames_t columnNames;
      columnNames.reserve(definedColumns.size() + treeBranchNames.size() + dsColumnsWithoutSizeColumns.size());
      columnNames.insert(columnNames.end(), definedColumns.begin(), definedColumns.end());
      columnNames.insert(columnNames.end(), treeBranchNames.begin(), treeBranchNames.end());
      columnNames.insert(columnNames.end(), dsColumnsWithoutSizeColumns.begin(), dsColumnsWithoutSizeColumns.end());

      // The only way we can get duplicate entries is if a column coming from a tree or data-source is Redefine'd.
      // RemoveDuplicates should preserve ordering of the columns: it might be meaningful.
      RDFInternal::RemoveDuplicates(columnNames);

      const auto selectedColumns = RDFInternal::ConvertRegexToColumns(columnNames, columnNameRegexp, "Snapshot");
      return Snapshot(treename, filename, selectedColumns, options);
   }
   // clang-format on

   // clang-format off
   ////////////////////////////////////////////////////////////////////////////
   /// \brief Save selected columns to disk, in a new TTree `treename` in file `filename`.
   /// \param[in] treename The name of the output TTree.
   /// \param[in] filename The name of the output TFile.
   /// \param[in] columnList The list of names of the columns/branches to be written.
   /// \param[in] options RSnapshotOptions struct with extra options to pass to TFile and TTree.
   /// \return a `RDataFrame` that wraps the snapshotted dataset.
   ///
   /// This function returns a `RDataFrame` built with the output tree as a source.
   /// The types of the columns are automatically inferred and do not need to be specified.
   ///
   /// See above for a more complete description and example usages.
   RResultPtr<RInterface<RLoopManager>> Snapshot(std::string_view treename, std::string_view filename,
                                                 std::initializer_list<std::string> columnList,
                                                 const RSnapshotOptions &options = RSnapshotOptions())
   {
      ColumnNames_t selectedColumns(columnList);
      return Snapshot(treename, filename, selectedColumns, options);
   }
   // clang-format on

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Save selected columns in memory.
   /// \tparam ColumnTypes variadic list of branch/column types.
   /// \param[in] columnList columns to be cached in memory.
   /// \return a `RDataFrame` that wraps the cached dataset.
   ///
   /// This action returns a new `RDataFrame` object, completely detached from
   /// the originating `RDataFrame`. The new dataframe only contains the cached
   /// columns and stores their content in memory for fast, zero-copy subsequent access.
   ///
   /// Use `Cache` if you know you will only need a subset of the (`Filter`ed) data that
   /// fits in memory and that will be accessed many times.
   ///
   /// \note Cache will refuse to process columns with names of the form `#columnname`. These are special columns
   /// made available by some data sources (e.g. RNTupleDS) that represent the size of column `columnname`, and are
   /// not meant to be written out with that name (which is not a valid C++ variable name). Instead, go through an
   /// Alias(): `df.Alias("nbar", "#bar").Cache<std::size_t>(..., {"nbar"})`.
   ///
   /// ### Example usage:
   ///
   /// **Types and columns specified:**
   /// ~~~{.cpp}
   /// auto cache_some_cols_df = df.Cache<double, MyClass, int>({"col0", "col1", "col2"});
   /// ~~~
   ///
   /// **Types inferred and columns specified (this invocation relies on jitting):**
   /// ~~~{.cpp}
   /// auto cache_some_cols_df = df.Cache({"col0", "col1", "col2"});
   /// ~~~
   ///
   /// **Types inferred and columns selected with a regexp (this invocation relies on jitting):**
   /// ~~~{.cpp}
   /// auto cache_all_cols_df = df.Cache(myRegexp);
   /// ~~~
   template <typename... ColumnTypes>
   RInterface<RLoopManager> Cache(const ColumnNames_t &columnList)
   {
      auto staticSeq = std::make_index_sequence<sizeof...(ColumnTypes)>();
      return CacheImpl<ColumnTypes...>(columnList, staticSeq);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Save selected columns in memory.
   /// \param[in] columnList columns to be cached in memory
   /// \return a `RDataFrame` that wraps the cached dataset.
   ///
   /// See the previous overloads for more information.
   RInterface<RLoopManager> Cache(const ColumnNames_t &columnList)
   {
      // Early return: if the list of columns is empty, just return an empty RDF
      // If we proceed, the jitted call will not compile!
      if (columnList.empty()) {
         auto nEntries = *this->Count();
         RInterface<RLoopManager> emptyRDF(std::make_shared<RLoopManager>(nEntries));
         return emptyRDF;
      }

      std::stringstream cacheCall;
      auto upcastNode = RDFInternal::UpcastNode(fProxiedPtr);
      RInterface<TTraits::TakeFirstParameter_t<decltype(upcastNode)>> upcastInterface(fProxiedPtr, *fLoopManager,
                                                                                      fColRegister, fDataSource);
      // build a string equivalent to
      // "(RInterface<nodetype*>*)(this)->Cache<Ts...>(*(ColumnNames_t*)(&columnList))"
      RInterface<RLoopManager> resRDF(std::make_shared<ROOT::Detail::RDF::RLoopManager>(0));
      cacheCall << "*reinterpret_cast<ROOT::RDF::RInterface<ROOT::Detail::RDF::RLoopManager>*>("
                << RDFInternal::PrettyPrintAddr(&resRDF)
                << ") = reinterpret_cast<ROOT::RDF::RInterface<ROOT::Detail::RDF::RNodeBase>*>("
                << RDFInternal::PrettyPrintAddr(&upcastInterface) << ")->Cache<";

      const auto columnListWithoutSizeColumns = RDFInternal::FilterArraySizeColNames(columnList, "Cache");

      const auto validColumnNames =
         GetValidatedColumnNames(columnListWithoutSizeColumns.size(), columnListWithoutSizeColumns);
      const auto colTypes = GetValidatedArgTypes(validColumnNames, fColRegister, fLoopManager->GetTree(), fDataSource,
                                                 "Cache", /*vector2rvec=*/false);
      for (const auto &colType : colTypes)
         cacheCall << colType << ", ";
      if (!columnListWithoutSizeColumns.empty())
         cacheCall.seekp(-2, cacheCall.cur);                         // remove the last ",
      cacheCall << ">(*reinterpret_cast<std::vector<std::string>*>(" // vector<string> should be ColumnNames_t
                << RDFInternal::PrettyPrintAddr(&columnListWithoutSizeColumns) << "));";

      // book the code to jit with the RLoopManager and trigger the event loop
      fLoopManager->ToJitExec(cacheCall.str());
      fLoopManager->Jit();

      return resRDF;
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Save selected columns in memory.
   /// \param[in] columnNameRegexp The regular expression to match the column names to be selected. The presence of a '^' and a '$' at the end of the string is implicitly assumed if they are not specified. The dialect supported is PCRE via the TPRegexp class. An empty string signals the selection of all columns.
   /// \return a `RDataFrame` that wraps the cached dataset.
   ///
   /// The existing columns are matched against the regular expression. If the string provided
   /// is empty, all columns are selected. See the previous overloads for more information.
   RInterface<RLoopManager> Cache(std::string_view columnNameRegexp = "")
   {
      const auto definedColumns = fColRegister.GetNames();
      auto *tree = fLoopManager->GetTree();
      const auto treeBranchNames = tree != nullptr ? RDFInternal::GetTopLevelBranchNames(*tree) : ColumnNames_t{};
      const auto dsColumns = fDataSource ? fDataSource->GetColumnNames() : ColumnNames_t{};
      // Ignore R_rdf_sizeof_* columns coming from datasources: we don't want to Snapshot those
      ColumnNames_t dsColumnsWithoutSizeColumns;
      std::copy_if(dsColumns.begin(), dsColumns.end(), std::back_inserter(dsColumnsWithoutSizeColumns),
                   [](const std::string &name) { return name.size() < 13 || name.substr(0, 13) != "R_rdf_sizeof_"; });
      ColumnNames_t columnNames;
      columnNames.reserve(definedColumns.size() + treeBranchNames.size() + dsColumns.size());
      columnNames.insert(columnNames.end(), definedColumns.begin(), definedColumns.end());
      columnNames.insert(columnNames.end(), treeBranchNames.begin(), treeBranchNames.end());
      columnNames.insert(columnNames.end(), dsColumns.begin(), dsColumns.end());
      const auto selectedColumns = RDFInternal::ConvertRegexToColumns(columnNames, columnNameRegexp, "Cache");
      return Cache(selectedColumns);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Save selected columns in memory.
   /// \param[in] columnList columns to be cached in memory.
   /// \return a `RDataFrame` that wraps the cached dataset.
   ///
   /// See the previous overloads for more information.
   RInterface<RLoopManager> Cache(std::initializer_list<std::string> columnList)
   {
      ColumnNames_t selectedColumns(columnList);
      return Cache(selectedColumns);
   }

   // clang-format off
   ////////////////////////////////////////////////////////////////////////////
   /// \brief Creates a node that filters entries based on range: [begin, end).
   /// \param[in] begin Initial entry number considered for this range.
   /// \param[in] end Final entry number (excluded) considered for this range. 0 means that the range goes until the end of the dataset.
   /// \param[in] stride Process one entry of the [begin, end) range every `stride` entries. Must be strictly greater than 0.
   /// \return the first node of the computation graph for which the event loop is limited to a certain range of entries.
   ///
   /// Note that in case of previous Ranges and Filters the selected range refers to the transformed dataset.
   /// Ranges are only available if EnableImplicitMT has _not_ been called. Multi-thread ranges are not supported.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// auto d_0_30 = d.Range(0, 30); // Pick the first 30 entries
   /// auto d_15_end = d.Range(15, 0); // Pick all entries from 15 onwards
   /// auto d_15_end_3 = d.Range(15, 0, 3); // Stride: from event 15, pick an event every 3
   /// ~~~
   // clang-format on
   RInterface<RDFDetail::RRange<Proxied>, DS_t> Range(unsigned int begin, unsigned int end, unsigned int stride = 1)
   {
      // check invariants
      if (stride == 0 || (end != 0 && end < begin))
         throw std::runtime_error("Range: stride must be strictly greater than 0 and end must be greater than begin.");
      CheckIMTDisabled("Range");

      using Range_t = RDFDetail::RRange<Proxied>;
      auto rangePtr = std::make_shared<Range_t>(begin, end, stride, fProxiedPtr);
      RInterface<RDFDetail::RRange<Proxied>, DS_t> tdf_r(std::move(rangePtr), *fLoopManager, fColRegister, fDataSource);
      return tdf_r;
   }

   // clang-format off
   ////////////////////////////////////////////////////////////////////////////
   /// \brief Creates a node that filters entries based on range.
   /// \param[in] end Final entry number (excluded) considered for this range. 0 means that the range goes until the end of the dataset.
   /// \return a node of the computation graph for which the range is defined.
   ///
   /// See the other Range overload for a detailed description.
   // clang-format on
   RInterface<RDFDetail::RRange<Proxied>, DS_t> Range(unsigned int end) { return Range(0, end, 1); }

   // clang-format off
   ////////////////////////////////////////////////////////////////////////////
   /// \brief Execute a user-defined function on each entry (*instant action*).
   /// \param[in] f Function, lambda expression, functor class or any other callable object performing user defined calculations.
   /// \param[in] columns Names of the columns/branches in input to the user function.
   ///
   /// The callable `f` is invoked once per entry. This is an *instant action*:
   /// upon invocation, an event loop as well as execution of all scheduled actions
   /// is triggered.
   /// Users are responsible for the thread-safety of this callable when executing
   /// with implicit multi-threading enabled (i.e. ROOT::EnableImplicitMT).
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// myDf.Foreach([](int i){ std::cout << i << std::endl;}, {"myIntColumn"});
   /// ~~~
   // clang-format on
   template <typename F>
   void Foreach(F f, const ColumnNames_t &columns = {})
   {
      using arg_types = typename TTraits::CallableTraits<decltype(f)>::arg_types_nodecay;
      using ret_type = typename TTraits::CallableTraits<decltype(f)>::ret_type;
      ForeachSlot(RDFInternal::AddSlotParameter<ret_type>(f, arg_types()), columns);
   }

   // clang-format off
   ////////////////////////////////////////////////////////////////////////////
   /// \brief Execute a user-defined function requiring a processing slot index on each entry (*instant action*).
   /// \param[in] f Function, lambda expression, functor class or any other callable object performing user defined calculations.
   /// \param[in] columns Names of the columns/branches in input to the user function.
   ///
   /// Same as `Foreach`, but the user-defined function takes an extra
   /// `unsigned int` as its first parameter, the *processing slot index*.
   /// This *slot index* will be assigned a different value, `0` to `poolSize - 1`,
   /// for each thread of execution.
   /// This is meant as a helper in writing thread-safe `Foreach`
   /// actions when using `RDataFrame` after `ROOT::EnableImplicitMT()`.
   /// The user-defined processing callable is able to follow different
   /// *streams of processing* indexed by the first parameter.
   /// `ForeachSlot` works just as well with single-thread execution: in that
   /// case `slot` will always be `0`.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// myDf.ForeachSlot([](unsigned int s, int i){ std::cout << "Slot " << s << ": "<< i << std::endl;}, {"myIntColumn"});
   /// ~~~
   // clang-format on
   template <typename F>
   void ForeachSlot(F f, const ColumnNames_t &columns = {})
   {
      using ColTypes_t = TypeTraits::RemoveFirstParameter_t<typename TTraits::CallableTraits<F>::arg_types>;
      constexpr auto nColumns = ColTypes_t::list_size;

      const auto validColumnNames = GetValidatedColumnNames(nColumns, columns);
      CheckAndFillDSColumns(validColumnNames, ColTypes_t());

      using Helper_t = RDFInternal::ForeachSlotHelper<F>;
      using Action_t = RDFInternal::RAction<Helper_t, Proxied>;

      auto action = std::make_unique<Action_t>(Helper_t(std::move(f)), validColumnNames, fProxiedPtr, fColRegister);

      fLoopManager->Run();
   }

   // clang-format off
   ////////////////////////////////////////////////////////////////////////////
   /// \brief Execute a user-defined reduce operation on the values of a column.
   /// \tparam F The type of the reduce callable. Automatically deduced.
   /// \tparam T The type of the column to apply the reduction to. Automatically deduced.
   /// \param[in] f A callable with signature `T(T,T)`
   /// \param[in] columnName The column to be reduced. If omitted, the first default column is used instead.
   /// \return the reduced quantity wrapped in a ROOT::RDF:RResultPtr.
   ///
   /// A reduction takes two values of a column and merges them into one (e.g.
   /// by summing them, taking the maximum, etc). This action performs the
   /// specified reduction operation on all processed column values, returning
   /// a single value of the same type. The callable f must satisfy the general
   /// requirements of a *processing function* besides having signature `T(T,T)`
   /// where `T` is the type of column columnName.
   ///
   /// The returned reduced value of each thread (e.g. the initial value of a sum) is initialized to a
   /// default-constructed T object. This is commonly expected to be the neutral/identity element for the specific
   /// reduction operation `f` (e.g. 0 for a sum, 1 for a product). If a default-constructed T does not satisfy this
   /// requirement, users should explicitly specify an initialization value for T by calling the appropriate `Reduce`
   /// overload.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// auto sumOfIntCol = d.Reduce([](int x, int y) { return x + y; }, "intCol");
   /// ~~~
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. Also see RResultPtr.
   // clang-format on
   template <typename F, typename T = typename TTraits::CallableTraits<F>::ret_type>
   RResultPtr<T> Reduce(F f, std::string_view columnName = "")
   {
      static_assert(
         std::is_default_constructible<T>::value,
         "reduce object cannot be default-constructed. Please provide an initialisation value (redIdentity)");
      return Reduce(std::move(f), columnName, T());
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Execute a user-defined reduce operation on the values of a column.
   /// \tparam F The type of the reduce callable. Automatically deduced.
   /// \tparam T The type of the column to apply the reduction to. Automatically deduced.
   /// \param[in] f A callable with signature `T(T,T)`
   /// \param[in] columnName The column to be reduced. If omitted, the first default column is used instead.
   /// \param[in] redIdentity The reduced object of each thread is initialised to this value.
   /// \return the reduced quantity wrapped in a RResultPtr.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// auto sumOfIntColWithOffset = d.Reduce([](int x, int y) { return x + y; }, "intCol", 42);
   /// ~~~
   /// See the description of the first Reduce overload for more information.
   template <typename F, typename T = typename TTraits::CallableTraits<F>::ret_type>
   RResultPtr<T> Reduce(F f, std::string_view columnName, const T &redIdentity)
   {
      return Aggregate(f, f, columnName, redIdentity);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return the number of entries processed (*lazy action*).
   /// \return the number of entries wrapped in a RResultPtr.
   ///
   /// Useful e.g. for counting the number of entries passing a certain filter (see also `Report`).
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. Also see RResultPtr.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// auto nEntriesAfterCuts = myFilteredDf.Count();
   /// ~~~
   ///
   RResultPtr<ULong64_t> Count()
   {
      const auto nSlots = fLoopManager->GetNSlots();
      auto cSPtr = std::make_shared<ULong64_t>(0);
      using Helper_t = RDFInternal::CountHelper;
      using Action_t = RDFInternal::RAction<Helper_t, Proxied>;
      auto action = std::make_unique<Action_t>(Helper_t(cSPtr, nSlots), ColumnNames_t({}), fProxiedPtr,
                                               RDFInternal::RColumnRegister(fColRegister));
      return MakeResultPtr(cSPtr, *fLoopManager, std::move(action));
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return a collection of values of a column (*lazy action*, returns a std::vector by default).
   /// \tparam T The type of the column.
   /// \tparam COLL The type of collection used to store the values.
   /// \param[in] column The name of the column to collect the values of.
   /// \return the content of the selected column wrapped in a RResultPtr.
   ///
   /// The collection type to be specified for C-style array columns is `RVec<T>`:
   /// in this case the returned collection is a `std::vector<RVec<T>>`.
   /// ### Example usage:
   /// ~~~{.cpp}
   /// // In this case intCol is a std::vector<int>
   /// auto intCol = rdf.Take<int>("integerColumn");
   /// // Same content as above but in this case taken as a RVec<int>
   /// auto intColAsRVec = rdf.Take<int, RVec<int>>("integerColumn");
   /// // In this case intCol is a std::vector<RVec<int>>, a collection of collections
   /// auto cArrayIntCol = rdf.Take<RVec<int>>("cArrayInt");
   /// ~~~
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. Also see RResultPtr.
   template <typename T, typename COLL = std::vector<T>>
   RResultPtr<COLL> Take(std::string_view column = "")
   {
      const auto columns = column.empty() ? ColumnNames_t() : ColumnNames_t({std::string(column)});

      const auto validColumnNames = GetValidatedColumnNames(1, columns);
      CheckAndFillDSColumns(validColumnNames, TTraits::TypeList<T>());

      using Helper_t = RDFInternal::TakeHelper<T, T, COLL>;
      using Action_t = RDFInternal::RAction<Helper_t, Proxied>;
      auto valuesPtr = std::make_shared<COLL>();
      const auto nSlots = fLoopManager->GetNSlots();

      auto action =
         std::make_unique<Action_t>(Helper_t(valuesPtr, nSlots), validColumnNames, fProxiedPtr, fColRegister);
      return MakeResultPtr(valuesPtr, *fLoopManager, std::move(action));
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a one-dimensional histogram with the values of a column (*lazy action*).
   /// \tparam V The type of the column used to fill the histogram.
   /// \param[in] model The returned histogram will be constructed using this as a model.
   /// \param[in] vName The name of the column that will fill the histogram.
   /// \return the monodimensional histogram wrapped in a RResultPtr.
   ///
   /// Columns can be of a container type (e.g. `std::vector<double>`), in which case the histogram
   /// is filled with each one of the elements of the container. In case multiple columns of container type
   /// are provided (e.g. values and weights) they must have the same length for each one of the events (but
   /// possibly different lengths between events).
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. Also see RResultPtr.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// // Deduce column type (this invocation needs jitting internally)
   /// auto myHist1 = myDf.Histo1D({"histName", "histTitle", 64u, 0., 128.}, "myColumn");
   /// // Explicit column type
   /// auto myHist2 = myDf.Histo1D<float>({"histName", "histTitle", 64u, 0., 128.}, "myColumn");
   /// ~~~
   ///
   /// \note Differently from other ROOT interfaces, the returned histogram is not associated to gDirectory
   /// and the caller is responsible for its lifetime (in particular, a typical source of confusion is that
   /// if result histograms go out of scope before the end of the program, ROOT might display a blank canvas).
   template <typename V = RDFDetail::RInferredType>
   RResultPtr<::TH1D> Histo1D(const TH1DModel &model = {"", "", 128u, 0., 0.}, std::string_view vName = "")
   {
      const auto userColumns = vName.empty() ? ColumnNames_t() : ColumnNames_t({std::string(vName)});

      const auto validatedColumns = GetValidatedColumnNames(1, userColumns);

      std::shared_ptr<::TH1D> h(nullptr);
      {
         ROOT::Internal::RDF::RIgnoreErrorLevelRAII iel(kError);
         h = model.GetHistogram();
         h->SetDirectory(nullptr);
      }

      if (h->GetXaxis()->GetXmax() == h->GetXaxis()->GetXmin())
         RDFInternal::HistoUtils<::TH1D>::SetCanExtendAllAxes(*h);
      return CreateAction<RDFInternal::ActionTags::Histo1D, V>(validatedColumns, h, h);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a one-dimensional histogram with the values of a column (*lazy action*).
   /// \tparam V The type of the column used to fill the histogram.
   /// \param[in] vName The name of the column that will fill the histogram.
   /// \return the monodimensional histogram wrapped in a RResultPtr.
   ///
   /// This overload uses a default model histogram TH1D(name, title, 128u, 0., 0.).
   /// The "name" and "title" strings are built starting from the input column name.
   /// See the description of the first Histo1D() overload for more details.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// // Deduce column type (this invocation needs jitting internally)
   /// auto myHist1 = myDf.Histo1D("myColumn");
   /// // Explicit column type
   /// auto myHist2 = myDf.Histo1D<float>("myColumn");
   /// ~~~
   template <typename V = RDFDetail::RInferredType>
   RResultPtr<::TH1D> Histo1D(std::string_view vName)
   {
      const auto h_name = std::string(vName);
      const auto h_title = h_name + ";" + h_name + ";count";
      return Histo1D<V>({h_name.c_str(), h_title.c_str(), 128u, 0., 0.}, vName);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a one-dimensional histogram with the weighted values of a column (*lazy action*).
   /// \tparam V The type of the column used to fill the histogram.
   /// \tparam W The type of the column used as weights.
   /// \param[in] model The returned histogram will be constructed using this as a model.
   /// \param[in] vName The name of the column that will fill the histogram.
   /// \param[in] wName The name of the column that will provide the weights.
   /// \return the monodimensional histogram wrapped in a RResultPtr.
   ///
   /// See the description of the first Histo1D() overload for more details.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// // Deduce column type (this invocation needs jitting internally)
   /// auto myHist1 = myDf.Histo1D({"histName", "histTitle", 64u, 0., 128.}, "myValue", "myweight");
   /// // Explicit column type
   /// auto myHist2 = myDf.Histo1D<float, int>({"histName", "histTitle", 64u, 0., 128.}, "myValue", "myweight");
   /// ~~~
   template <typename V = RDFDetail::RInferredType, typename W = RDFDetail::RInferredType>
   RResultPtr<::TH1D> Histo1D(const TH1DModel &model, std::string_view vName, std::string_view wName)
   {
      const std::vector<std::string_view> columnViews = {vName, wName};
      const auto userColumns = RDFInternal::AtLeastOneEmptyString(columnViews)
                                  ? ColumnNames_t()
                                  : ColumnNames_t(columnViews.begin(), columnViews.end());
      std::shared_ptr<::TH1D> h(nullptr);
      {
         ROOT::Internal::RDF::RIgnoreErrorLevelRAII iel(kError);
         h = model.GetHistogram();
      }
      return CreateAction<RDFInternal::ActionTags::Histo1D, V, W>(userColumns, h, h);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a one-dimensional histogram with the weighted values of a column (*lazy action*).
   /// \tparam V The type of the column used to fill the histogram.
   /// \tparam W The type of the column used as weights.
   /// \param[in] vName The name of the column that will fill the histogram.
   /// \param[in] wName The name of the column that will provide the weights.
   /// \return the monodimensional histogram wrapped in a RResultPtr.
   ///
   /// This overload uses a default model histogram TH1D(name, title, 128u, 0., 0.).
   /// The "name" and "title" strings are built starting from the input column names.
   /// See the description of the first Histo1D() overload for more details.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// // Deduce column types (this invocation needs jitting internally)
   /// auto myHist1 = myDf.Histo1D("myValue", "myweight");
   /// // Explicit column types
   /// auto myHist2 = myDf.Histo1D<float, int>("myValue", "myweight");
   /// ~~~
   template <typename V = RDFDetail::RInferredType, typename W = RDFDetail::RInferredType>
   RResultPtr<::TH1D> Histo1D(std::string_view vName, std::string_view wName)
   {
      // We build name and title based on the value and weight column names
      std::string str_vName{vName};
      std::string str_wName{wName};
      const auto h_name = str_vName + "_weighted_" + str_wName;
      const auto h_title = str_vName + ", weights: " + str_wName + ";" + str_vName + ";count * " + str_wName;
      return Histo1D<V, W>({h_name.c_str(), h_title.c_str(), 128u, 0., 0.}, vName, wName);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a one-dimensional histogram with the weighted values of a column (*lazy action*).
   /// \tparam V The type of the column used to fill the histogram.
   /// \tparam W The type of the column used as weights.
   /// \param[in] model The returned histogram will be constructed using this as a model.
   /// \return the monodimensional histogram wrapped in a RResultPtr.
   ///
   /// This overload will use the first two default columns as column names.
   /// See the description of the first Histo1D() overload for more details.
   template <typename V, typename W>
   RResultPtr<::TH1D> Histo1D(const TH1DModel &model = {"", "", 128u, 0., 0.})
   {
      return Histo1D<V, W>(model, "", "");
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a two-dimensional histogram (*lazy action*).
   /// \tparam V1 The type of the column used to fill the x axis of the histogram.
   /// \tparam V2 The type of the column used to fill the y axis of the histogram.
   /// \param[in] model The returned histogram will be constructed using this as a model.
   /// \param[in] v1Name The name of the column that will fill the x axis.
   /// \param[in] v2Name The name of the column that will fill the y axis.
   /// \return the bidimensional histogram wrapped in a RResultPtr.
   ///
   /// Columns can be of a container type (e.g. std::vector<double>), in which case the histogram
   /// is filled with each one of the elements of the container. In case multiple columns of container type
   /// are provided (e.g. values and weights) they must have the same length for each one of the events (but
   /// possibly different lengths between events).
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. Also see RResultPtr.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// // Deduce column types (this invocation needs jitting internally)
   /// auto myHist1 = myDf.Histo2D({"histName", "histTitle", 64u, 0., 128., 32u, -4., 4.}, "myValueX", "myValueY");
   /// // Explicit column types
   /// auto myHist2 = myDf.Histo2D<float, float>({"histName", "histTitle", 64u, 0., 128., 32u, -4., 4.}, "myValueX", "myValueY");
   /// ~~~
   ///
   ///
   /// \note Differently from other ROOT interfaces, the returned histogram is not associated to gDirectory
   /// and the caller is responsible for its lifetime (in particular, a typical source of confusion is that
   /// if result histograms go out of scope before the end of the program, ROOT might display a blank canvas).
   template <typename V1 = RDFDetail::RInferredType, typename V2 = RDFDetail::RInferredType>
   RResultPtr<::TH2D> Histo2D(const TH2DModel &model, std::string_view v1Name = "", std::string_view v2Name = "")
   {
      std::shared_ptr<::TH2D> h(nullptr);
      {
         ROOT::Internal::RDF::RIgnoreErrorLevelRAII iel(kError);
         h = model.GetHistogram();
      }
      if (!RDFInternal::HistoUtils<::TH2D>::HasAxisLimits(*h)) {
         throw std::runtime_error("2D histograms with no axes limits are not supported yet.");
      }
      const std::vector<std::string_view> columnViews = {v1Name, v2Name};
      const auto userColumns = RDFInternal::AtLeastOneEmptyString(columnViews)
                                  ? ColumnNames_t()
                                  : ColumnNames_t(columnViews.begin(), columnViews.end());
      return CreateAction<RDFInternal::ActionTags::Histo2D, V1, V2>(userColumns, h, h);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a weighted two-dimensional histogram (*lazy action*).
   /// \tparam V1 The type of the column used to fill the x axis of the histogram.
   /// \tparam V2 The type of the column used to fill the y axis of the histogram.
   /// \tparam W The type of the column used for the weights of the histogram.
   /// \param[in] model The returned histogram will be constructed using this as a model.
   /// \param[in] v1Name The name of the column that will fill the x axis.
   /// \param[in] v2Name The name of the column that will fill the y axis.
   /// \param[in] wName The name of the column that will provide the weights.
   /// \return the bidimensional histogram wrapped in a RResultPtr.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. Also see RResultPtr.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// // Deduce column types (this invocation needs jitting internally)
   /// auto myHist1 = myDf.Histo2D({"histName", "histTitle", 64u, 0., 128., 32u, -4., 4.}, "myValueX", "myValueY", "myWeight");
   /// // Explicit column types
   /// auto myHist2 = myDf.Histo2D<float, float, double>({"histName", "histTitle", 64u, 0., 128., 32u, -4., 4.}, "myValueX", "myValueY", "myWeight");
   /// ~~~
   ///
   /// See the documentation of the first Histo2D() overload for more details.
   template <typename V1 = RDFDetail::RInferredType, typename V2 = RDFDetail::RInferredType,
             typename W = RDFDetail::RInferredType>
   RResultPtr<::TH2D>
   Histo2D(const TH2DModel &model, std::string_view v1Name, std::string_view v2Name, std::string_view wName)
   {
      std::shared_ptr<::TH2D> h(nullptr);
      {
         ROOT::Internal::RDF::RIgnoreErrorLevelRAII iel(kError);
         h = model.GetHistogram();
      }
      if (!RDFInternal::HistoUtils<::TH2D>::HasAxisLimits(*h)) {
         throw std::runtime_error("2D histograms with no axes limits are not supported yet.");
      }
      const std::vector<std::string_view> columnViews = {v1Name, v2Name, wName};
      const auto userColumns = RDFInternal::AtLeastOneEmptyString(columnViews)
                                  ? ColumnNames_t()
                                  : ColumnNames_t(columnViews.begin(), columnViews.end());
      return CreateAction<RDFInternal::ActionTags::Histo2D, V1, V2, W>(userColumns, h, h);
   }

   template <typename V1, typename V2, typename W>
   RResultPtr<::TH2D> Histo2D(const TH2DModel &model)
   {
      return Histo2D<V1, V2, W>(model, "", "", "");
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a three-dimensional histogram (*lazy action*).
   /// \tparam V1 The type of the column used to fill the x axis of the histogram. Inferred if not present.
   /// \tparam V2 The type of the column used to fill the y axis of the histogram. Inferred if not present.
   /// \tparam V3 The type of the column used to fill the z axis of the histogram. Inferred if not present.
   /// \param[in] model The returned histogram will be constructed using this as a model.
   /// \param[in] v1Name The name of the column that will fill the x axis.
   /// \param[in] v2Name The name of the column that will fill the y axis.
   /// \param[in] v3Name The name of the column that will fill the z axis.
   /// \return the tridimensional histogram wrapped in a RResultPtr.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. Also see RResultPtr.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// // Deduce column types (this invocation needs jitting internally)
   /// auto myHist1 = myDf.Histo3D({"name", "title", 64u, 0., 128., 32u, -4., 4., 8u, -2., 2.},
   ///                             "myValueX", "myValueY", "myValueZ");
   /// // Explicit column types
   /// auto myHist2 = myDf.Histo3D<double, double, float>({"name", "title", 64u, 0., 128., 32u, -4., 4., 8u, -2., 2.},
   ///                                                    "myValueX", "myValueY", "myValueZ");
   /// ~~~
   ///
   /// \note Differently from other ROOT interfaces, the returned histogram is not associated to gDirectory
   /// and the caller is responsible for its lifetime (in particular, a typical source of confusion is that
   /// if result histograms go out of scope before the end of the program, ROOT might display a blank canvas).
   template <typename V1 = RDFDetail::RInferredType, typename V2 = RDFDetail::RInferredType,
             typename V3 = RDFDetail::RInferredType>
   RResultPtr<::TH3D> Histo3D(const TH3DModel &model, std::string_view v1Name = "", std::string_view v2Name = "",
                              std::string_view v3Name = "")
   {
      std::shared_ptr<::TH3D> h(nullptr);
      {
         ROOT::Internal::RDF::RIgnoreErrorLevelRAII iel(kError);
         h = model.GetHistogram();
      }
      if (!RDFInternal::HistoUtils<::TH3D>::HasAxisLimits(*h)) {
         throw std::runtime_error("3D histograms with no axes limits are not supported yet.");
      }
      const std::vector<std::string_view> columnViews = {v1Name, v2Name, v3Name};
      const auto userColumns = RDFInternal::AtLeastOneEmptyString(columnViews)
                                  ? ColumnNames_t()
                                  : ColumnNames_t(columnViews.begin(), columnViews.end());
      return CreateAction<RDFInternal::ActionTags::Histo3D, V1, V2, V3>(userColumns, h, h);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a three-dimensional histogram (*lazy action*).
   /// \tparam V1 The type of the column used to fill the x axis of the histogram. Inferred if not present.
   /// \tparam V2 The type of the column used to fill the y axis of the histogram. Inferred if not present.
   /// \tparam V3 The type of the column used to fill the z axis of the histogram. Inferred if not present.
   /// \tparam W The type of the column used for the weights of the histogram. Inferred if not present.
   /// \param[in] model The returned histogram will be constructed using this as a model.
   /// \param[in] v1Name The name of the column that will fill the x axis.
   /// \param[in] v2Name The name of the column that will fill the y axis.
   /// \param[in] v3Name The name of the column that will fill the z axis.
   /// \param[in] wName The name of the column that will provide the weights.
   /// \return the tridimensional histogram wrapped in a RResultPtr.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. Also see RResultPtr.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// // Deduce column types (this invocation needs jitting internally)
   /// auto myHist1 = myDf.Histo3D({"name", "title", 64u, 0., 128., 32u, -4., 4., 8u, -2., 2.},
   ///                             "myValueX", "myValueY", "myValueZ", "myWeight");
   /// // Explicit column types
   /// using d_t = double;
   /// auto myHist2 = myDf.Histo3D<d_t, d_t, float, d_t>({"name", "title", 64u, 0., 128., 32u, -4., 4., 8u, -2., 2.},
   ///                                                    "myValueX", "myValueY", "myValueZ", "myWeight");
   /// ~~~
   ///
   ///
   /// See the documentation of the first Histo2D() overload for more details.
   template <typename V1 = RDFDetail::RInferredType, typename V2 = RDFDetail::RInferredType,
             typename V3 = RDFDetail::RInferredType, typename W = RDFDetail::RInferredType>
   RResultPtr<::TH3D> Histo3D(const TH3DModel &model, std::string_view v1Name, std::string_view v2Name,
                              std::string_view v3Name, std::string_view wName)
   {
      std::shared_ptr<::TH3D> h(nullptr);
      {
         ROOT::Internal::RDF::RIgnoreErrorLevelRAII iel(kError);
         h = model.GetHistogram();
      }
      if (!RDFInternal::HistoUtils<::TH3D>::HasAxisLimits(*h)) {
         throw std::runtime_error("3D histograms with no axes limits are not supported yet.");
      }
      const std::vector<std::string_view> columnViews = {v1Name, v2Name, v3Name, wName};
      const auto userColumns = RDFInternal::AtLeastOneEmptyString(columnViews)
                                  ? ColumnNames_t()
                                  : ColumnNames_t(columnViews.begin(), columnViews.end());
      return CreateAction<RDFInternal::ActionTags::Histo3D, V1, V2, V3, W>(userColumns, h, h);
   }

   template <typename V1, typename V2, typename V3, typename W>
   RResultPtr<::TH3D> Histo3D(const TH3DModel &model)
   {
      return Histo3D<V1, V2, V3, W>(model, "", "", "", "");
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return an N-dimensional histogram (*lazy action*).
   /// \tparam FirstColumn The first type of the column the values of which are used to fill the object. Inferred if not
   /// present.
   /// \tparam OtherColumns A list of the other types of the columns the values of which are used to fill the
   /// object.
   /// \param[in] model The returned histogram will be constructed using this as a model.
   /// \param[in] columnList
   /// A list containing the names of the columns that will be passed when calling `Fill`.
   ///  (N columns for unweighted filling, or N+1 columns for weighted filling)
   /// \return the N-dimensional histogram wrapped in a RResultPtr.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. See RResultPtr documentation.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// auto myFilledObj = myDf.HistoND<float, float, float, float>({"name","title", 4,
   ///                                                {40,40,40,40}, {20.,20.,20.,20.}, {60.,60.,60.,60.}},
   ///                                               {"col0", "col1", "col2", "col3"});
   /// ~~~
   ///
   template <typename FirstColumn, typename... OtherColumns> // need FirstColumn to disambiguate overloads
   RResultPtr<::THnD> HistoND(const THnDModel &model, const ColumnNames_t &columnList)
   {
      std::shared_ptr<::THnD> h(nullptr);
      {
         ROOT::Internal::RDF::RIgnoreErrorLevelRAII iel(kError);
         h = model.GetHistogram();

         if (int(columnList.size()) == (h->GetNdimensions() + 1)) {
            h->Sumw2();
         } else if (int(columnList.size()) != h->GetNdimensions()) {
            throw std::runtime_error("Wrong number of columns for the specified number of histogram axes.");
         }
      }
      return CreateAction<RDFInternal::ActionTags::HistoND, FirstColumn, OtherColumns...>(columnList, h, h);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return an N-dimensional histogram (*lazy action*).
   /// \param[in] model The returned histogram will be constructed using this as a model.
   /// \param[in] columnList A list containing the names of the columns that will be passed when calling `Fill`
   ///  (N columns for unweighted filling, or N+1 columns for weighted filling)
   /// \return the N-dimensional histogram wrapped in a RResultPtr.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. Also see RResultPtr.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// auto myFilledObj = myDf.HistoND({"name","title", 4,
   ///                                                {40,40,40,40}, {20.,20.,20.,20.}, {60.,60.,60.,60.}},
   ///                                               {"col0", "col1", "col2", "col3"});
   /// ~~~
   ///
   RResultPtr<::THnD> HistoND(const THnDModel &model, const ColumnNames_t &columnList)
   {
      std::shared_ptr<::THnD> h(nullptr);
      {
         ROOT::Internal::RDF::RIgnoreErrorLevelRAII iel(kError);
         h = model.GetHistogram();

         if (int(columnList.size()) == (h->GetNdimensions() + 1)) {
            h->Sumw2();
         } else if (int(columnList.size()) != h->GetNdimensions()) {
            throw std::runtime_error("Wrong number of columns for the specified number of histogram axes.");
         }
      }
      return CreateAction<RDFInternal::ActionTags::HistoND, RDFDetail::RInferredType>(columnList, h, h,
                                                                                      columnList.size());
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a graph (*lazy action*).
   /// \tparam V1 The type of the column used to fill the x axis of the graph.
   /// \tparam V2 The type of the column used to fill the y axis of the graph.
   /// \param[in] v1Name The name of the column that will fill the x axis.
   /// \param[in] v2Name The name of the column that will fill the y axis.
   /// \return the graph wrapped in a RResultPtr.
   ///
   /// Columns can be of a container type (e.g. std::vector<double>), in which case the graph
   /// is filled with each one of the elements of the container.
   /// If Multithreading is enabled, the order in which points are inserted is undefined.
   /// If the Graph has to be drawn, it is suggested to the user to sort it on the x before printing.
   /// A name and a title to the graph is given based on the input column names.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. Also see RResultPtr.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// // Deduce column types (this invocation needs jitting internally)
   /// auto myGraph1 = myDf.Graph("xValues", "yValues");
   /// // Explicit column types
   /// auto myGraph2 = myDf.Graph<int, float>("xValues", "yValues");
   /// ~~~
   ///
   /// \note Differently from other ROOT interfaces, the returned graph is not associated to gDirectory
   /// and the caller is responsible for its lifetime (in particular, a typical source of confusion is that
   /// if result histograms go out of scope before the end of the program, ROOT might display a blank canvas).
   template <typename V1 = RDFDetail::RInferredType, typename V2 = RDFDetail::RInferredType>
   RResultPtr<::TGraph> Graph(std::string_view v1Name = "", std::string_view v2Name = "")
   {
      auto graph = std::make_shared<::TGraph>();
      const std::vector<std::string_view> columnViews = {v1Name, v2Name};
      const auto userColumns = RDFInternal::AtLeastOneEmptyString(columnViews)
                                  ? ColumnNames_t()
                                  : ColumnNames_t(columnViews.begin(), columnViews.end());

      const auto validatedColumns = GetValidatedColumnNames(2, userColumns);

      // We build a default name and title based on the input columns
      if (!(validatedColumns[0].empty() && validatedColumns[1].empty())) {
         const auto g_name = std::string(v1Name) + "_vs_" + std::string(v2Name);
         const auto g_title = std::string(v1Name) + " vs " + std::string(v2Name);
         graph->SetNameTitle(g_name.c_str(), g_title.c_str());
         graph->GetXaxis()->SetTitle(std::string(v1Name).c_str());
         graph->GetYaxis()->SetTitle(std::string(v2Name).c_str());
      }

      return CreateAction<RDFInternal::ActionTags::Graph, V1, V2>(validatedColumns, graph, graph);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a one-dimensional profile (*lazy action*).
   /// \tparam V1 The type of the column the values of which are used to fill the profile. Inferred if not present.
   /// \tparam V2 The type of the column the values of which are used to fill the profile. Inferred if not present.
   /// \param[in] model The model to be considered to build the new return value.
   /// \param[in] v1Name The name of the column that will fill the x axis.
   /// \param[in] v2Name The name of the column that will fill the y axis.
   /// \return the monodimensional profile wrapped in a RResultPtr.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. Also see RResultPtr.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// // Deduce column types (this invocation needs jitting internally)
   /// auto myProf1 = myDf.Profile1D({"profName", "profTitle", 64u, -4., 4.}, "xValues", "yValues");
   /// // Explicit column types
   /// auto myProf2 = myDf.Graph<int, float>({"profName", "profTitle", 64u, -4., 4.}, "xValues", "yValues");
   /// ~~~
   ///
   /// \note Differently from other ROOT interfaces, the returned profile is not associated to gDirectory
   /// and the caller is responsible for its lifetime (in particular, a typical source of confusion is that
   /// if result histograms go out of scope before the end of the program, ROOT might display a blank canvas).
   template <typename V1 = RDFDetail::RInferredType, typename V2 = RDFDetail::RInferredType>
   RResultPtr<::TProfile>
   Profile1D(const TProfile1DModel &model, std::string_view v1Name = "", std::string_view v2Name = "")
   {
      std::shared_ptr<::TProfile> h(nullptr);
      {
         ROOT::Internal::RDF::RIgnoreErrorLevelRAII iel(kError);
         h = model.GetProfile();
      }

      if (!RDFInternal::HistoUtils<::TProfile>::HasAxisLimits(*h)) {
         throw std::runtime_error("Profiles with no axes limits are not supported yet.");
      }
      const std::vector<std::string_view> columnViews = {v1Name, v2Name};
      const auto userColumns = RDFInternal::AtLeastOneEmptyString(columnViews)
                                  ? ColumnNames_t()
                                  : ColumnNames_t(columnViews.begin(), columnViews.end());
      return CreateAction<RDFInternal::ActionTags::Profile1D, V1, V2>(userColumns, h, h);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a one-dimensional profile (*lazy action*).
   /// \tparam V1 The type of the column the values of which are used to fill the profile. Inferred if not present.
   /// \tparam V2 The type of the column the values of which are used to fill the profile. Inferred if not present.
   /// \tparam W The type of the column the weights of which are used to fill the profile. Inferred if not present.
   /// \param[in] model The model to be considered to build the new return value.
   /// \param[in] v1Name The name of the column that will fill the x axis.
   /// \param[in] v2Name The name of the column that will fill the y axis.
   /// \param[in] wName The name of the column that will provide the weights.
   /// \return the monodimensional profile wrapped in a RResultPtr.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. Also see RResultPtr.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// // Deduce column types (this invocation needs jitting internally)
   /// auto myProf1 = myDf.Profile1D({"profName", "profTitle", 64u, -4., 4.}, "xValues", "yValues", "weight");
   /// // Explicit column types
   /// auto myProf2 = myDf.Profile1D<int, float, double>({"profName", "profTitle", 64u, -4., 4.},
   ///                                                   "xValues", "yValues", "weight");
   /// ~~~
   ///
   /// See the first Profile1D() overload for more details.
   template <typename V1 = RDFDetail::RInferredType, typename V2 = RDFDetail::RInferredType,
             typename W = RDFDetail::RInferredType>
   RResultPtr<::TProfile>
   Profile1D(const TProfile1DModel &model, std::string_view v1Name, std::string_view v2Name, std::string_view wName)
   {
      std::shared_ptr<::TProfile> h(nullptr);
      {
         ROOT::Internal::RDF::RIgnoreErrorLevelRAII iel(kError);
         h = model.GetProfile();
      }

      if (!RDFInternal::HistoUtils<::TProfile>::HasAxisLimits(*h)) {
         throw std::runtime_error("Profile histograms with no axes limits are not supported yet.");
      }
      const std::vector<std::string_view> columnViews = {v1Name, v2Name, wName};
      const auto userColumns = RDFInternal::AtLeastOneEmptyString(columnViews)
                                  ? ColumnNames_t()
                                  : ColumnNames_t(columnViews.begin(), columnViews.end());
      return CreateAction<RDFInternal::ActionTags::Profile1D, V1, V2, W>(userColumns, h, h);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a one-dimensional profile (*lazy action*).
   /// See the first Profile1D() overload for more details.
   template <typename V1, typename V2, typename W>
   RResultPtr<::TProfile> Profile1D(const TProfile1DModel &model)
   {
      return Profile1D<V1, V2, W>(model, "", "", "");
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a two-dimensional profile (*lazy action*).
   /// \tparam V1 The type of the column used to fill the x axis of the histogram. Inferred if not present.
   /// \tparam V2 The type of the column used to fill the y axis of the histogram. Inferred if not present.
   /// \tparam V2 The type of the column used to fill the z axis of the histogram. Inferred if not present.
   /// \param[in] model The returned profile will be constructed using this as a model.
   /// \param[in] v1Name The name of the column that will fill the x axis.
   /// \param[in] v2Name The name of the column that will fill the y axis.
   /// \param[in] v3Name The name of the column that will fill the z axis.
   /// \return the bidimensional profile wrapped in a RResultPtr.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. Also see RResultPtr.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// // Deduce column types (this invocation needs jitting internally)
   /// auto myProf1 = myDf.Profile2D({"profName", "profTitle", 40, -4, 4, 40, -4, 4, 0, 20},
   ///                               "xValues", "yValues", "zValues");
   /// // Explicit column types
   /// auto myProf2 = myDf.Profile2D<int, float, double>({"profName", "profTitle", 40, -4, 4, 40, -4, 4, 0, 20},
   ///                                                   "xValues", "yValues", "zValues");
   /// ~~~
   ///
   /// \note Differently from other ROOT interfaces, the returned profile is not associated to gDirectory
   /// and the caller is responsible for its lifetime (in particular, a typical source of confusion is that
   /// if result histograms go out of scope before the end of the program, ROOT might display a blank canvas).
   template <typename V1 = RDFDetail::RInferredType, typename V2 = RDFDetail::RInferredType,
             typename V3 = RDFDetail::RInferredType>
   RResultPtr<::TProfile2D> Profile2D(const TProfile2DModel &model, std::string_view v1Name = "",
                                      std::string_view v2Name = "", std::string_view v3Name = "")
   {
      std::shared_ptr<::TProfile2D> h(nullptr);
      {
         ROOT::Internal::RDF::RIgnoreErrorLevelRAII iel(kError);
         h = model.GetProfile();
      }

      if (!RDFInternal::HistoUtils<::TProfile2D>::HasAxisLimits(*h)) {
         throw std::runtime_error("2D profiles with no axes limits are not supported yet.");
      }
      const std::vector<std::string_view> columnViews = {v1Name, v2Name, v3Name};
      const auto userColumns = RDFInternal::AtLeastOneEmptyString(columnViews)
                                  ? ColumnNames_t()
                                  : ColumnNames_t(columnViews.begin(), columnViews.end());
      return CreateAction<RDFInternal::ActionTags::Profile2D, V1, V2, V3>(userColumns, h, h);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Fill and return a two-dimensional profile (*lazy action*).
   /// \tparam V1 The type of the column used to fill the x axis of the histogram. Inferred if not present.
   /// \tparam V2 The type of the column used to fill the y axis of the histogram. Inferred if not present.
   /// \tparam V3 The type of the column used to fill the z axis of the histogram. Inferred if not present.
   /// \tparam W The type of the column used for the weights of the histogram. Inferred if not present.
   /// \param[in] model The returned histogram will be constructed using this as a model.
   /// \param[in] v1Name The name of the column that will fill the x axis.
   /// \param[in] v2Name The name of the column that will fill the y axis.
   /// \param[in] v3Name The name of the column that will fill the z axis.
   /// \param[in] wName The name of the column that will provide the weights.
   /// \return the bidimensional profile wrapped in a RResultPtr.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. Also see RResultPtr.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// // Deduce column types (this invocation needs jitting internally)
   /// auto myProf1 = myDf.Profile2D({"profName", "profTitle", 40, -4, 4, 40, -4, 4, 0, 20},
   ///                               "xValues", "yValues", "zValues", "weight");
   /// // Explicit column types
   /// auto myProf2 = myDf.Profile2D<int, float, double, int>({"profName", "profTitle", 40, -4, 4, 40, -4, 4, 0, 20},
   ///                                                        "xValues", "yValues", "zValues", "weight");
   /// ~~~
   ///
   /// See the first Profile2D() overload for more details.
   template <typename V1 = RDFDetail::RInferredType, typename V2 = RDFDetail::RInferredType,
             typename V3 = RDFDetail::RInferredType, typename W = RDFDetail::RInferredType>
   RResultPtr<::TProfile2D> Profile2D(const TProfile2DModel &model, std::string_view v1Name, std::string_view v2Name,
                                      std::string_view v3Name, std::string_view wName)
   {
      std::shared_ptr<::TProfile2D> h(nullptr);
      {
         ROOT::Internal::RDF::RIgnoreErrorLevelRAII iel(kError);
         h = model.GetProfile();
      }

      if (!RDFInternal::HistoUtils<::TProfile2D>::HasAxisLimits(*h)) {
         throw std::runtime_error("2D profiles with no axes limits are not supported yet.");
      }
      const std::vector<std::string_view> columnViews = {v1Name, v2Name, v3Name, wName};
      const auto userColumns = RDFInternal::AtLeastOneEmptyString(columnViews)
                                  ? ColumnNames_t()
                                  : ColumnNames_t(columnViews.begin(), columnViews.end());
      return CreateAction<RDFInternal::ActionTags::Profile2D, V1, V2, V3, W>(userColumns, h, h);
   }

   /// \brief Fill and return a two-dimensional profile (*lazy action*).
   /// See the first Profile2D() overload for more details.
   template <typename V1, typename V2, typename V3, typename W>
   RResultPtr<::TProfile2D> Profile2D(const TProfile2DModel &model)
   {
      return Profile2D<V1, V2, V3, W>(model, "", "", "", "");
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return an object of type T on which `T::Fill` will be called once per event (*lazy action*).
   ///
   /// Type T must provide at least:
   /// - a copy-constructor
   /// - a `Fill` method that accepts as many arguments and with same types as the column names passed as columnList
   ///   (these types can also be passed as template parameters to this method)
   /// - a `Merge` method with signature `Merge(TCollection *)` or `Merge(const std::vector<T *>&)` that merges the
   ///   objects passed as argument into the object on which `Merge` was called (an analogous of TH1::Merge). Note that
   ///   if the signature that takes a `TCollection*` is used, then T must inherit from TObject (to allow insertion in
   ///   the TCollection*).
   ///
   /// \tparam FirstColumn The first type of the column the values of which are used to fill the object. Inferred together with OtherColumns if not present.
   /// \tparam OtherColumns A list of the other types of the columns the values of which are used to fill the object.
   /// \tparam T The type of the object to fill. Automatically deduced.
   /// \param[in] model The model to be considered to build the new return value.
   /// \param[in] columnList A list containing the names of the columns that will be passed when calling `Fill`
   /// \return the filled object wrapped in a RResultPtr.
   ///
   /// The user gives up ownership of the model object.
   /// The list of column names to be used for filling must always be specified.
   /// This action is *lazy*: upon invocation of this method the calculation is booked but not executed.
   /// Also see RResultPtr.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// MyClass obj;
   /// // Deduce column types (this invocation needs jitting internally, and in this case
   /// // MyClass needs to be known to the interpreter)
   /// auto myFilledObj = myDf.Fill(obj, {"col0", "col1"});
   /// // explicit column types
   /// auto myFilledObj = myDf.Fill<float, float>(obj, {"col0", "col1"});
   /// ~~~
   ///
   template <typename FirstColumn = RDFDetail::RInferredType, typename... OtherColumns, typename T>
   RResultPtr<std::decay_t<T>> Fill(T &&model, const ColumnNames_t &columnList)
   {
      auto h = std::make_shared<std::decay_t<T>>(std::forward<T>(model));
      if (!RDFInternal::HistoUtils<T>::HasAxisLimits(*h)) {
         throw std::runtime_error("The absence of axes limits is not supported yet.");
      }
      return CreateAction<RDFInternal::ActionTags::Fill, FirstColumn, OtherColumns...>(columnList, h, h,
                                                                                       columnList.size());
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return a TStatistic object, filled once per event (*lazy action*).
   ///
   /// \tparam V The type of the value column
   /// \param[in] value The name of the column with the values to fill the statistics with.
   /// \return the filled TStatistic object wrapped in a RResultPtr.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// // Deduce column type (this invocation needs jitting internally)
   /// auto stats0 = myDf.Stats("values");
   /// // Explicit column type
   /// auto stats1 = myDf.Stats<float>("values");
   /// ~~~
   ///
   template <typename V = RDFDetail::RInferredType>
   RResultPtr<TStatistic> Stats(std::string_view value = "")
   {
      ColumnNames_t columns;
      if (!value.empty()) {
         columns.emplace_back(std::string(value));
      }
      const auto validColumnNames = GetValidatedColumnNames(1, columns);
      if (std::is_same<V, RDFDetail::RInferredType>::value) {
         return Fill(TStatistic(), validColumnNames);
      } else {
         return Fill<V>(TStatistic(), validColumnNames);
      }
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return a TStatistic object, filled once per event (*lazy action*).
   ///
   /// \tparam V The type of the value column
   /// \tparam W The type of the weight column
   /// \param[in] value The name of the column with the values to fill the statistics with.
   /// \param[in] weight The name of the column with the weights to fill the statistics with.
   /// \return the filled TStatistic object wrapped in a RResultPtr.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// // Deduce column types (this invocation needs jitting internally)
   /// auto stats0 = myDf.Stats("values", "weights");
   /// // Explicit column types
   /// auto stats1 = myDf.Stats<int, float>("values", "weights");
   /// ~~~
   ///
   template <typename V = RDFDetail::RInferredType, typename W = RDFDetail::RInferredType>
   RResultPtr<TStatistic> Stats(std::string_view value, std::string_view weight)
   {
      ColumnNames_t columns{std::string(value), std::string(weight)};
      constexpr auto vIsInferred = std::is_same<V, RDFDetail::RInferredType>::value;
      constexpr auto wIsInferred = std::is_same<W, RDFDetail::RInferredType>::value;
      const auto validColumnNames = GetValidatedColumnNames(2, columns);
      // We have 3 cases:
      // 1. Both types are inferred: we use Fill and let the jit kick in.
      // 2. One of the two types is explicit and the other one is inferred: the case is not supported.
      // 3. Both types are explicit: we invoke the fully compiled Fill method.
      if (vIsInferred && wIsInferred) {
         return Fill(TStatistic(), validColumnNames);
      } else if (vIsInferred != wIsInferred) {
         std::string error("The ");
         error += vIsInferred ? "value " : "weight ";
         error += "column type is explicit, while the ";
         error += vIsInferred ? "weight " : "value ";
         error += " is specified to be inferred. This case is not supported: please specify both types or none.";
         throw std::runtime_error(error);
      } else {
         return Fill<V, W>(TStatistic(), validColumnNames);
      }
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return the minimum of processed column values (*lazy action*).
   /// \tparam T The type of the branch/column.
   /// \param[in] columnName The name of the branch/column to be treated.
   /// \return the minimum value of the selected column wrapped in a RResultPtr.
   ///
   /// If T is not specified, RDataFrame will infer it from the data and just-in-time compile the correct
   /// template specialization of this method.
   /// If the type of the column is inferred, the return type is `double`, the type of the column otherwise.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. Also see RResultPtr.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// // Deduce column type (this invocation needs jitting internally)
   /// auto minVal0 = myDf.Min("values");
   /// // Explicit column type
   /// auto minVal1 = myDf.Min<double>("values");
   /// ~~~
   ///
   template <typename T = RDFDetail::RInferredType>
   RResultPtr<RDFDetail::MinReturnType_t<T>> Min(std::string_view columnName = "")
   {
      const auto userColumns = columnName.empty() ? ColumnNames_t() : ColumnNames_t({std::string(columnName)});
      using RetType_t = RDFDetail::MinReturnType_t<T>;
      auto minV = std::make_shared<RetType_t>(std::numeric_limits<RetType_t>::max());
      return CreateAction<RDFInternal::ActionTags::Min, T>(userColumns, minV, minV);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return the maximum of processed column values (*lazy action*).
   /// \tparam T The type of the branch/column.
   /// \param[in] columnName The name of the branch/column to be treated.
   /// \return the maximum value of the selected column wrapped in a RResultPtr.
   ///
   /// If T is not specified, RDataFrame will infer it from the data and just-in-time compile the correct
   /// template specialization of this method.
   /// If the type of the column is inferred, the return type is `double`, the type of the column otherwise.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. Also see RResultPtr.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// // Deduce column type (this invocation needs jitting internally)
   /// auto maxVal0 = myDf.Max("values");
   /// // Explicit column type
   /// auto maxVal1 = myDf.Max<double>("values");
   /// ~~~
   ///
   template <typename T = RDFDetail::RInferredType>
   RResultPtr<RDFDetail::MaxReturnType_t<T>> Max(std::string_view columnName = "")
   {
      const auto userColumns = columnName.empty() ? ColumnNames_t() : ColumnNames_t({std::string(columnName)});
      using RetType_t = RDFDetail::MaxReturnType_t<T>;
      auto maxV = std::make_shared<RetType_t>(std::numeric_limits<RetType_t>::lowest());
      return CreateAction<RDFInternal::ActionTags::Max, T>(userColumns, maxV, maxV);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return the mean of processed column values (*lazy action*).
   /// \tparam T The type of the branch/column.
   /// \param[in] columnName The name of the branch/column to be treated.
   /// \return the mean value of the selected column wrapped in a RResultPtr.
   ///
   /// If T is not specified, RDataFrame will infer it from the data and just-in-time compile the correct
   /// template specialization of this method.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. Also see RResultPtr.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// // Deduce column type (this invocation needs jitting internally)
   /// auto meanVal0 = myDf.Mean("values");
   /// // Explicit column type
   /// auto meanVal1 = myDf.Mean<double>("values");
   /// ~~~
   ///
   template <typename T = RDFDetail::RInferredType>
   RResultPtr<double> Mean(std::string_view columnName = "")
   {
      const auto userColumns = columnName.empty() ? ColumnNames_t() : ColumnNames_t({std::string(columnName)});
      auto meanV = std::make_shared<double>(0);
      return CreateAction<RDFInternal::ActionTags::Mean, T>(userColumns, meanV, meanV);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return the unbiased standard deviation of processed column values (*lazy action*).
   /// \tparam T The type of the branch/column.
   /// \param[in] columnName The name of the branch/column to be treated.
   /// \return the standard deviation value of the selected column wrapped in a RResultPtr.
   ///
   /// If T is not specified, RDataFrame will infer it from the data and just-in-time compile the correct
   /// template specialization of this method.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. Also see RResultPtr.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// // Deduce column type (this invocation needs jitting internally)
   /// auto stdDev0 = myDf.StdDev("values");
   /// // Explicit column type
   /// auto stdDev1 = myDf.StdDev<double>("values");
   /// ~~~
   ///
   template <typename T = RDFDetail::RInferredType>
   RResultPtr<double> StdDev(std::string_view columnName = "")
   {
      const auto userColumns = columnName.empty() ? ColumnNames_t() : ColumnNames_t({std::string(columnName)});
      auto stdDeviationV = std::make_shared<double>(0);
      return CreateAction<RDFInternal::ActionTags::StdDev, T>(userColumns, stdDeviationV, stdDeviationV);
   }

   // clang-format off
   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return the sum of processed column values (*lazy action*).
   /// \tparam T The type of the branch/column.
   /// \param[in] columnName The name of the branch/column.
   /// \param[in] initValue Optional initial value for the sum. If not present, the column values must be default-constructible.
   /// \return the sum of the selected column wrapped in a RResultPtr.
   ///
   /// If T is not specified, RDataFrame will infer it from the data and just-in-time compile the correct
   /// template specialization of this method.
   /// If the type of the column is inferred, the return type is `double`, the type of the column otherwise.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is
   /// booked but not executed. Also see RResultPtr.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// // Deduce column type (this invocation needs jitting internally)
   /// auto sum0 = myDf.Sum("values");
   /// // Explicit column type
   /// auto sum1 = myDf.Sum<double>("values");
   /// ~~~
   ///
   template <typename T = RDFDetail::RInferredType>
   RResultPtr<RDFDetail::SumReturnType_t<T>>
   Sum(std::string_view columnName = "",
       const RDFDetail::SumReturnType_t<T> &initValue = RDFDetail::SumReturnType_t<T>{})
   {
      const auto userColumns = columnName.empty() ? ColumnNames_t() : ColumnNames_t({std::string(columnName)});
      auto sumV = std::make_shared<RDFDetail::SumReturnType_t<T>>(initValue);
      return CreateAction<RDFInternal::ActionTags::Sum, T>(userColumns, sumV, sumV);
   }
   // clang-format on

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Gather filtering statistics.
   /// \return the resulting `RCutFlowReport` instance wrapped in a RResultPtr.
   ///
   /// Calling `Report` on the main `RDataFrame` object gathers stats for
   /// all named filters in the call graph. Calling this method on a
   /// stored chain state (i.e. a graph node different from the first) gathers
   /// the stats for all named filters in the chain section between the original
   /// `RDataFrame` and that node (included). Stats are gathered in the same
   /// order as the named filters have been added to the graph.
   /// A RResultPtr<RCutFlowReport> is returned to allow inspection of the
   /// effects cuts had.
   ///
   /// This action is *lazy*: upon invocation of
   /// this method the calculation is booked but not executed. See RResultPtr
   /// documentation.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// auto filtered = d.Filter(cut1, {"b1"}, "Cut1").Filter(cut2, {"b2"}, "Cut2");
   /// auto cutReport = filtered3.Report();
   /// cutReport->Print();
   /// ~~~
   ///
   RResultPtr<RCutFlowReport> Report()
   {
      bool returnEmptyReport = false;
      // if this is a RInterface<RLoopManager> on which `Define` has been called, users
      // are calling `Report` on a chain of the form LoopManager->Define->Define->..., which
      // certainly does not contain named filters.
      // The number 4 takes into account the implicit columns for entry and slot number
      // and their aliases (2 + 2, i.e. {r,t}dfentry_ and {r,t}dfslot_)
      if (std::is_same<Proxied, RLoopManager>::value && fColRegister.GetNames().size() > 4)
         returnEmptyReport = true;

      auto rep = std::make_shared<RCutFlowReport>();
      using Helper_t = RDFInternal::ReportHelper<Proxied>;
      using Action_t = RDFInternal::RAction<Helper_t, Proxied>;

      auto action = std::make_unique<Action_t>(Helper_t(rep, fProxiedPtr, returnEmptyReport), ColumnNames_t({}),
                                               fProxiedPtr, RDFInternal::RColumnRegister(fColRegister));

      return MakeResultPtr(rep, *fLoopManager, std::move(action));
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Returns the names of the available columns.
   /// \return the container of column names.
   ///
   /// This is not an action nor a transformation, just a query to the RDataFrame object.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// auto colNames = d.GetColumnNames();
   /// // Print columns' names
   /// for (auto &&colName : colNames) std::cout << colName << std::endl;
   /// ~~~
   ///
   ColumnNames_t GetColumnNames()
   {
      // there could be duplicates between Redefined columns and columns in the data source
      std::unordered_set<std::string> allColumns;

      auto addIfNotInternal = [&allColumns](std::string_view colName) {
         if (!RDFInternal::IsInternalColumn(colName))
            allColumns.emplace(colName);
      };

      auto definedColumns = fColRegister.GetNames();

      std::for_each(definedColumns.begin(), definedColumns.end(), addIfNotInternal);

      auto tree = fLoopManager->GetTree();
      if (tree) {
         for (const auto &bName : RDFInternal::GetBranchNames(*tree, /*allowDuplicates=*/false))
            allColumns.emplace(bName);
      }

      if (fDataSource) {
         for (const auto &s : fDataSource->GetColumnNames()) {
            if (s.rfind("R_rdf_sizeof", 0) != 0)
               allColumns.emplace(s);
         }
      }

      ColumnNames_t ret(allColumns.begin(), allColumns.end());
      std::sort(ret.begin(), ret.end());
      return ret;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Return the type of a given column as a string.
   /// \return the type of the required column.
   ///
   /// This is not an action nor a transformation, just a query to the RDataFrame object.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// auto colType = d.GetColumnType("columnName");
   /// // Print column type
   /// std::cout << "Column " << colType << " has type " << colType << std::endl;
   /// ~~~
   ///
   std::string GetColumnType(std::string_view column)
   {
      const auto col = fColRegister.ResolveAlias(std::string(column));

      RDFDetail::RDefineBase *define = fColRegister.HasName(col) ? fColRegister.GetColumns().at(col).get() : nullptr;

      const bool convertVector2RVec = true;
      return RDFInternal::ColumnName2ColumnTypeName(col, fLoopManager->GetTree(), fLoopManager->GetDataSource(), define,
                                                    convertVector2RVec);
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Return information about the dataframe.
   /// \return information about the dataframe as RDFDescription object
   ///
   /// This convenience function describes the dataframe and combines the following information:
   /// - Number of event loops run, see GetNRuns()
   /// - Number of total and defined columns, see GetColumnNames() and GetDefinedColumnNames()
   /// - Column names, see GetColumnNames()
   /// - Column types, see GetColumnType()
   /// - Number of processing slots, see GetNSlots()
   ///
   /// This is not an action nor a transformation, just a query to the RDataFrame object.
   /// The result is dependent on the node from which this method is called, e.g. the list of
   /// defined columns returned by GetDefinedColumnNames().
   ///
   /// Please note that this is a convenience feature and the layout of the output can be subject
   /// to change and should be parsed via RDFDescription methods.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// RDataFrame df(10);
   /// auto df2 = df.Define("x", "1.f").Define("s", "\"myStr\"");
   /// // Describe the dataframe
   /// df2.Describe().Print()
   /// df2.Describe().Print(/*shortFormat=*/true)
   /// std::cout << df2.Describe().AsString() << std::endl;
   /// std::cout << df2.Describe().AsString(/*shortFormat=*/true) << std::endl;
   /// ~~~
   ///
   RDFDescription Describe()
   {
      // Build set of defined column names to find later in all column names
      // the defined columns more efficiently
      const auto columnNames = GetColumnNames();
      std::set<std::string> definedColumnNamesSet;
      for (const auto &name : GetDefinedColumnNames())
         definedColumnNamesSet.insert(name);

      // Get information for the metadata table
      const std::vector<std::string> metadataProperties = {"Columns in total", "Columns from defines",
                                                           "Event loops run", "Processing slots"};
      const std::vector<std::string> metadataValues = {std::to_string(columnNames.size()),
                                                       std::to_string(definedColumnNamesSet.size()),
                                                       std::to_string(GetNRuns()), std::to_string(GetNSlots())};

      // Set header for metadata table
      const auto columnWidthProperties = RDFInternal::GetColumnWidth(metadataProperties);
      // The column width of the values is required to make right-bound numbers and is equal
      // to the maximum of the string "Value" and all values to be put in this column.
      const auto columnWidthValues =
         std::max(std::max_element(metadataValues.begin(), metadataValues.end())->size(), static_cast<std::size_t>(5u));
      std::stringstream ss;
      ss << std::left << std::setw(columnWidthProperties) << "Property" << std::setw(columnWidthValues) << "Value\n"
         << std::setw(columnWidthProperties) << "--------" << std::setw(columnWidthValues) << "-----\n";

      // Build metadata table
      // All numbers should be bound to the right and strings bound to the left.
      for (auto i = 0u; i < metadataProperties.size(); i++) {
         ss << std::left << std::setw(columnWidthProperties) << metadataProperties[i] << std::right
            << std::setw(columnWidthValues) << metadataValues[i] << '\n';
      }
      ss << '\n'; // put space between this and the next table

      // Set header for columns table
      const auto columnWidthNames = RDFInternal::GetColumnWidth(columnNames);
      const auto columnTypes = GetColumnTypeNamesList(columnNames);
      const auto columnWidthTypes = RDFInternal::GetColumnWidth(columnTypes);
      ss << std::left << std::setw(columnWidthNames) << "Column" << std::setw(columnWidthTypes) << "Type"
         << "Origin\n"
         << std::setw(columnWidthNames) << "------" << std::setw(columnWidthTypes) << "----"
         << "------\n";

      // Build columns table
      const auto nCols = columnNames.size();
      for (auto i = 0u; i < nCols; i++) {
         auto origin = "Dataset";
         if (definedColumnNamesSet.find(columnNames[i]) != definedColumnNamesSet.end())
            origin = "Define";
         ss << std::left << std::setw(columnWidthNames) << columnNames[i] << std::setw(columnWidthTypes)
            << columnTypes[i] << origin;
         if (i < nCols - 1)
            ss << '\n';
      }
      // Use the string returned from DescribeDataset() as the 'brief' description
      // Use the converted to string stringstream ss as the 'full' description
      return RDFDescription(DescribeDataset(), ss.str());
   }

   /// \brief Returns the names of the filters created.
   /// \return the container of filters names.
   ///
   /// If called on a root node, all the filters in the computation graph will
   /// be printed. For any other node, only the filters upstream of that node.
   /// Filters without a name are printed as "Unnamed Filter"
   /// This is not an action nor a transformation, just a query to the RDataFrame object.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// auto filtNames = d.GetFilterNames();
   /// for (auto &&filtName : filtNames) std::cout << filtName << std::endl;
   /// ~~~
   ///
   std::vector<std::string> GetFilterNames() { return RDFInternal::GetFilterNames(fProxiedPtr); }

   /// \brief Returns the names of the defined columns.
   /// \return the container of the defined column names.
   ///
   /// This is not an action nor a transformation, just a simple utility to
   /// get the columns names that have been defined up to the node.
   /// If no column has been defined, e.g. on a root node, it returns an
   /// empty collection.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// auto defColNames = d.GetDefinedColumnNames();
   /// // Print defined columns' names
   /// for (auto &&defColName : defColNames) std::cout << defColName << std::endl;
   /// ~~~
   ///
   ColumnNames_t GetDefinedColumnNames()
   {
      ColumnNames_t definedColumns;

      auto columns = fColRegister.GetColumns();

      for (const auto &column : columns) {
         if (!RDFInternal::IsInternalColumn(column.first))
            definedColumns.emplace_back(column.first);
      }

      return definedColumns;
   }

   /// \brief Return a descriptor for the systematic variations registered in this branch of the computation graph.
   ///
   /// This is not an action nor a transformation, just a simple utility to
   /// inspect the systematic variations that have been registered with Vary() up to this node.
   /// When called on the root node, it returns an empty descriptor.
   ///
   /// ### Example usage:
   /// ~~~{.cpp}
   /// auto variations = d.GetVariations();
   /// variations.Print();
   /// ~~~
   ///
   RVariationsDescription GetVariations()
   {
      return {fColRegister.GetVariations()};
   }

   /// \brief Checks if a column is present in the dataset.
   /// \return true if the column is available, false otherwise
   ///
   /// This method checks if a column is part of the input ROOT dataset, has
   /// been defined or can be provided by the data source.
   ///
   /// Example usage:
   /// ~~~{.cpp}
   /// ROOT::RDataFrame base(1);
   /// auto rdf = base.Define("definedColumn", [](){return 0;});
   /// rdf.HasColumn("definedColumn"); // true: we defined it
   /// rdf.HasColumn("rdfentry_"); // true: it's always there
   /// rdf.HasColumn("foo"); // false: it is not there
   /// ~~~
   bool HasColumn(std::string_view columnName)
   {
      if (fColRegister.HasName(columnName))
         return true;

      if (auto tree = fLoopManager->GetTree()) {
         const auto &branchNames = fLoopManager->GetBranchNames();
         const auto branchNamesEnd = branchNames.end();
         if (branchNamesEnd != std::find(branchNames.begin(), branchNamesEnd, columnName))
            return true;
      }

      if (fDataSource && fDataSource->HasColumn(columnName))
         return true;

      return false;
   }

   /// \brief Gets the number of data processing slots.
   /// \return The number of data processing slots used by this RDataFrame instance
   ///
   /// This method returns the number of data processing slots used by this RDataFrame
   /// instance. This number is influenced by the global switch ROOT::EnableImplicitMT().
   ///
   /// Example usage:
   /// ~~~{.cpp}
   /// ROOT::EnableImplicitMT(6)
   /// ROOT::RDataFrame df(1);
   /// std::cout << df.GetNSlots() << std::endl; // prints "6"
   /// ~~~
   unsigned int GetNSlots() const { return fLoopManager->GetNSlots(); }

   /// \brief Gets the number of event loops run.
   /// \return The number of event loops run by this RDataFrame instance
   ///
   /// This method returns the number of events loops run so far by this RDataFrame instance.
   ///
   /// Example usage:
   /// ~~~{.cpp}
   /// ROOT::RDataFrame df(1);
   /// std::cout << df.GetNRuns() << std::endl; // prints "0"
   /// df.Sum("rdfentry_").GetValue(); // trigger the event loop
   /// std::cout << df.GetNRuns() << std::endl; // prints "1"
   /// df.Sum("rdfentry_").GetValue(); // trigger another event loop
   /// std::cout << df.GetNRuns() << std::endl; // prints "2"
   /// ~~~
   unsigned int GetNRuns() const { return fLoopManager->GetNRuns(); }

   // clang-format off
   ////////////////////////////////////////////////////////////////////////////
   /// \brief Execute a user-defined accumulation operation on the processed column values in each processing slot.
   /// \tparam F The type of the aggregator callable. Automatically deduced.
   /// \tparam U The type of the aggregator variable. Must be default-constructible, copy-constructible and copy-assignable. Automatically deduced.
   /// \tparam T The type of the column to apply the reduction to. Automatically deduced.
   /// \param[in] aggregator A callable with signature `U(U,T)` or `void(U&,T)`, where T is the type of the column, U is the type of the aggregator variable
   /// \param[in] merger A callable with signature `U(U,U)` or `void(std::vector<U>&)` used to merge the results of the accumulations of each thread
   /// \param[in] columnName The column to be aggregated. If omitted, the first default column is used instead.
   /// \param[in] aggIdentity The aggregator variable of each thread is initialised to this value (or is default-constructed if the parameter is omitted)
   /// \return the result of the aggregation wrapped in a RResultPtr.
   ///
   /// An aggregator callable takes two values, an aggregator variable and a column value. The aggregator variable is
   /// initialized to aggIdentity or default-constructed if aggIdentity is omitted.
   /// This action calls the aggregator callable for each processed entry, passing in the aggregator variable and
   /// the value of the column columnName.
   /// If the signature is `U(U,T)` the aggregator variable is then copy-assigned the result of the execution of the callable.
   /// Otherwise the signature of aggregator must be `void(U&,T)`.
   ///
   /// The merger callable is used to merge the partial accumulation results of each processing thread. It is only called in multi-thread executions.
   /// If its signature is `U(U,U)` the aggregator variables of each thread are merged two by two.
   /// If its signature is `void(std::vector<U>& a)` it is assumed that it merges all aggregators in a[0].
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is booked but not executed. Also see RResultPtr.
   ///
   /// Example usage:
   /// ~~~{.cpp}
   /// auto aggregator = [](double acc, double x) { return acc * x; };
   /// ROOT::EnableImplicitMT();
   /// // If multithread is enabled, the aggregator function will be called by more threads
   /// // and will produce a vector of partial accumulators.
   /// // The merger function performs the final aggregation of these partial results.
   /// auto merger = [](std::vector<double> &accumulators) {
   ///    for (auto i : ROOT::TSeqU(1u, accumulators.size())) {
   ///       accumulators[0] *= accumulators[i];
   ///    }
   /// };
   ///
   /// // The accumulator is initialized at this value by every thread.
   /// double initValue = 1.;
   ///
   /// // Multiplies all elements of the column "x"
   /// auto result = d.Aggregate(aggregator, merger, columnName, initValue);
   /// ~~~
   // clang-format on
   template <typename AccFun, typename MergeFun, typename R = typename TTraits::CallableTraits<AccFun>::ret_type,
             typename ArgTypes = typename TTraits::CallableTraits<AccFun>::arg_types,
             typename ArgTypesNoDecay = typename TTraits::CallableTraits<AccFun>::arg_types_nodecay,
             typename U = TTraits::TakeFirstParameter_t<ArgTypes>,
             typename T = TTraits::TakeFirstParameter_t<TTraits::RemoveFirstParameter_t<ArgTypes>>>
   RResultPtr<U> Aggregate(AccFun aggregator, MergeFun merger, std::string_view columnName, const U &aggIdentity)
   {
      RDFInternal::CheckAggregate<R, MergeFun>(ArgTypesNoDecay());
      const auto columns = columnName.empty() ? ColumnNames_t() : ColumnNames_t({std::string(columnName)});

      const auto validColumnNames = GetValidatedColumnNames(1, columns);
      CheckAndFillDSColumns(validColumnNames, TTraits::TypeList<T>());

      auto accObjPtr = std::make_shared<U>(aggIdentity);
      using Helper_t = RDFInternal::AggregateHelper<AccFun, MergeFun, R, T, U>;
      using Action_t = RDFInternal::RAction<Helper_t, Proxied>;
      auto action = std::make_unique<Action_t>(
         Helper_t(std::move(aggregator), std::move(merger), accObjPtr, fLoopManager->GetNSlots()), validColumnNames,
         fProxiedPtr, fColRegister);
      return MakeResultPtr(accObjPtr, *fLoopManager, std::move(action));
   }

   // clang-format off
   ////////////////////////////////////////////////////////////////////////////
   /// \brief Execute a user-defined accumulation operation on the processed column values in each processing slot.
   /// \tparam F The type of the aggregator callable. Automatically deduced.
   /// \tparam U The type of the aggregator variable. Must be default-constructible, copy-constructible and copy-assignable. Automatically deduced.
   /// \tparam T The type of the column to apply the reduction to. Automatically deduced.
   /// \param[in] aggregator A callable with signature `U(U,T)` or `void(U,T)`, where T is the type of the column, U is the type of the aggregator variable
   /// \param[in] merger A callable with signature `U(U,U)` or `void(std::vector<U>&)` used to merge the results of the accumulations of each thread
   /// \param[in] columnName The column to be aggregated. If omitted, the first default column is used instead.
   /// \return the result of the aggregation wrapped in a RResultPtr.
   ///
   /// See previous Aggregate overload for more information.
   // clang-format on
   template <typename AccFun, typename MergeFun, typename R = typename TTraits::CallableTraits<AccFun>::ret_type,
             typename ArgTypes = typename TTraits::CallableTraits<AccFun>::arg_types,
             typename U = TTraits::TakeFirstParameter_t<ArgTypes>,
             typename T = TTraits::TakeFirstParameter_t<TTraits::RemoveFirstParameter_t<ArgTypes>>>
   RResultPtr<U> Aggregate(AccFun aggregator, MergeFun merger, std::string_view columnName = "")
   {
      static_assert(
         std::is_default_constructible<U>::value,
         "aggregated object cannot be default-constructed. Please provide an initialisation value (aggIdentity)");
      return Aggregate(std::move(aggregator), std::move(merger), columnName, U());
   }

   // clang-format off
   ////////////////////////////////////////////////////////////////////////////
   /// \brief Book execution of a custom action using a user-defined helper object.
   /// \tparam FirstColumn The type of the first column used by this action.  Inferred together with OtherColumns if not present.
   /// \tparam OtherColumns A list of the types of the other columns used by this action
   /// \tparam Helper The type of the user-defined helper. See below for the required interface it should expose.
   /// \param[in] helper The Action Helper to be scheduled.
   /// \param[in] columns The names of the columns on which the helper acts.
   /// \return the result of the helper wrapped in a RResultPtr.
   ///
   /// This method books a custom action for execution. The behavior of the action is completely dependent on the
   /// Helper object provided by the caller. The minimum required interface for the helper is the following (more
   /// methods can be present, e.g. a constructor that takes the number of worker threads is usually useful):
   ///
   /// * Helper must publicly inherit from ROOT::Detail::RDF::RActionImpl<Helper>
   /// * Helper(Helper &&): a move-constructor is required. Copy-constructors are discouraged.
   /// * Result_t: alias for the type of the result of this action helper. Must be default-constructible.
   /// * void Exec(unsigned int slot, ColumnTypes...columnValues): each working thread shall call this method
   ///   during the event-loop, possibly concurrently. No two threads will ever call Exec with the same 'slot' value:
   ///   this parameter is there to facilitate writing thread-safe helpers. The other arguments will be the values of
   ///   the requested columns for the particular entry being processed.
   /// * void InitTask(TTreeReader *, unsigned int slot): each working thread shall call this method during the event
   ///   loop, before processing a batch of entries (possibly read from the TTreeReader passed as argument, if not null).
   ///   This method can be used e.g. to prepare the helper to process a batch of entries in a given thread. Can be no-op.
   /// * void Initialize(): this method is called once before starting the event-loop. Useful for setup operations.
   ///   It must reset the state of the helper to the expected state at the beginning of the event loop: the same helper,
   ///   or copies of it, might be used for multiple event loops (e.g. in the presence of systematic variations).
   /// * void Finalize(): this method is called at the end of the event loop. Commonly used to finalize the contents of the result.
   /// * Result_t &PartialUpdate(unsigned int slot): this method is optional, i.e. can be omitted. If present, it should
   ///   return the value of the partial result of this action for the given 'slot'. Different threads might call this
   ///   method concurrently, but will always pass different 'slot' numbers.
   /// * std::shared_ptr<Result_t> GetResultPtr() const: return a shared_ptr to the result of this action (of type
   ///   Result_t). The RResultPtr returned by Book will point to this object. Note that this method can be called
   ///   before Initialize(), because the RResultPtr is constructed before the event loop is started.
   ///
   /// In case this is called without specifying column types, jitting is used,
   /// and the Helper class needs to be known to the interpreter.
   ///
   /// See ActionHelpers.hxx for the helpers used by standard RDF actions.
   /// This action is *lazy*: upon invocation of this method the calculation is booked but not executed. Also see RResultPtr.
   // clang-format on

   template <typename FirstColumn = RDFDetail::RInferredType, typename... OtherColumns, typename Helper>
   RResultPtr<typename std::decay_t<Helper>::Result_t> Book(Helper &&helper, const ColumnNames_t &columns = {})
   {
      using HelperT = std::decay_t<Helper>;
      // TODO add more static sanity checks on Helper
      using AH = RDFDetail::RActionImpl<HelperT>;
      static_assert(std::is_base_of<AH, HelperT>::value && std::is_convertible<HelperT *, AH *>::value,
                    "Action helper of type T must publicly inherit from ROOT::Detail::RDF::RActionImpl<T>");

      auto hPtr = std::make_shared<HelperT>(std::forward<Helper>(helper));
      auto resPtr = hPtr->GetResultPtr();

      if (std::is_same<FirstColumn, RDFDetail::RInferredType>::value && columns.empty()) {
         return CallCreateActionWithoutColsIfPossible<HelperT>(resPtr, hPtr, TTraits::TypeList<FirstColumn>{});
      } else {
         return CreateAction<RDFInternal::ActionTags::Book, FirstColumn, OtherColumns...>(columns, resPtr, hPtr,
                                                                                          columns.size());
      }
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Provides a representation of the columns in the dataset.
   /// \tparam ColumnTypes variadic list of branch/column types.
   /// \param[in] columnList Names of the columns to be displayed.
   /// \param[in] nRows Number of events for each column to be displayed.
   /// \param[in] nMaxCollectionElements Maximum number of collection elements to display per row.
   /// \return the `RDisplay` instance wrapped in a RResultPtr.
   ///
   /// This function returns a RResultPtr<RDisplay>` containing all the entries to be displayed, organized in a tabular
   /// form. RDisplay will either print on the standard output a summarized version through `Print()` or will return a
   /// complete version through `AsString()`.
   ///
   /// This action is *lazy*: upon invocation of this method the calculation is booked but not executed. Also see
   /// RResultPtr.
   ///
   /// Example usage:
   /// ~~~{.cpp}
   /// // Preparing the RResultPtr<RDisplay> object with all columns and default number of entries
   /// auto d1 = rdf.Display("");
   /// // Preparing the RResultPtr<RDisplay> object with two columns and 128 entries
   /// auto d2 = d.Display({"x", "y"}, 128);
   /// // Printing the short representations, the event loop will run
   /// d1->Print();
   /// d2->Print();
   /// ~~~
   template <typename... ColumnTypes>
   RResultPtr<RDisplay> Display(const ColumnNames_t &columnList, size_t nRows = 5, size_t nMaxCollectionElements = 10)
   {
      CheckIMTDisabled("Display");  
      auto newCols = columnList;
      newCols.insert(newCols.begin(), "rdfentry_"); // Artificially insert first column
      auto displayer = std::make_shared<RDisplay>(newCols, GetColumnTypeNamesList(newCols), nMaxCollectionElements);
      using displayHelperArgs_t = std::pair<size_t, std::shared_ptr<RDisplay>>;
      // Need to add ULong64_t type corresponding to the first column rdfentry_
      return CreateAction<RDFInternal::ActionTags::Display, ULong64_t, ColumnTypes...>(
         std::move(newCols), displayer, std::make_shared<displayHelperArgs_t>(nRows, displayer));
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Provides a representation of the columns in the dataset.
   /// \param[in] columnList Names of the columns to be displayed.
   /// \param[in] nRows Number of events for each column to be displayed.
   /// \param[in] nMaxCollectionElements  Maximum number of collection elements to display per row.
   /// \return the `RDisplay` instance wrapped in a RResultPtr.
   ///
   /// This overload automatically infers the column types.
   /// See the previous overloads for further details.
   ///
   /// Invoked when no types are specified to Display
   RResultPtr<RDisplay> Display(const ColumnNames_t &columnList, size_t nRows = 5, size_t nMaxCollectionElements = 10)
   {
      CheckIMTDisabled("Display");
      auto newCols = columnList;
      newCols.insert(newCols.begin(), "rdfentry_"); // Artificially insert first column
      auto displayer = std::make_shared<RDisplay>(newCols, GetColumnTypeNamesList(newCols), nMaxCollectionElements);
      using displayHelperArgs_t = std::pair<size_t, std::shared_ptr<RDisplay>>;
      return CreateAction<RDFInternal::ActionTags::Display, RDFDetail::RInferredType>(
         std::move(newCols), displayer, std::make_shared<displayHelperArgs_t>(nRows, displayer), columnList.size() + 1);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Provides a representation of the columns in the dataset.
   /// \param[in] columnNameRegexp A regular expression to select the columns.
   /// \param[in] nRows Number of events for each column to be displayed.
   /// \param[in] nMaxCollectionElements Maximum number of collection elements to display per row.
   /// \return the `RDisplay` instance wrapped in a RResultPtr.
   ///
   /// The existing columns are matched against the regular expression. If the string provided
   /// is empty, all columns are selected.
   /// See the previous overloads for further details.
   RResultPtr<RDisplay>
   Display(std::string_view columnNameRegexp = "", size_t nRows = 5, size_t nMaxCollectionElements = 10)
   {
      const auto columnNames = GetColumnNames();
      const auto selectedColumns = RDFInternal::ConvertRegexToColumns(columnNames, columnNameRegexp, "Display");
      return Display(selectedColumns, nRows, nMaxCollectionElements);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Provides a representation of the columns in the dataset.
   /// \param[in] columnList Names of the columns to be displayed.
   /// \param[in] nRows Number of events for each column to be displayed.
   /// \return the `RDisplay` instance wrapped in a RResultPtr.
   ///
   /// See the previous overloads for further details.
   RResultPtr<RDisplay>
   Display(std::initializer_list<std::string> columnList, size_t nRows = 5, size_t nMaxCollectionElements = 10)
   {
      ColumnNames_t selectedColumns(columnList);
      return Display(selectedColumns, nRows, nMaxCollectionElements);
   }

private:
   void AddDefaultColumns()
   {
      // Entry number column
      const std::string entryColName = "rdfentry_";
      const std::string entryColType = "ULong64_t";
      auto entryColGen = [](unsigned int, ULong64_t entry) { return entry; };
      using NewColEntry_t = RDFDetail::RDefine<decltype(entryColGen), RDFDetail::CustomColExtraArgs::SlotAndEntry>;

      auto entryColumn = std::make_shared<NewColEntry_t>(entryColName, entryColType, std::move(entryColGen),
                                                         ColumnNames_t{}, fColRegister, *fLoopManager);
      fColRegister.AddColumn(entryColumn);

      // Slot number column
      const std::string slotColName = "rdfslot_";
      const std::string slotColType = "unsigned int";
      auto slotColGen = [](unsigned int slot) { return slot; };
      using NewColSlot_t = RDFDetail::RDefine<decltype(slotColGen), RDFDetail::CustomColExtraArgs::Slot>;

      auto slotColumn = std::make_shared<NewColSlot_t>(slotColName, slotColType, std::move(slotColGen), ColumnNames_t{},
                                                       fColRegister, *fLoopManager);
      fColRegister.AddColumn(slotColumn);

      fColRegister.AddAlias("tdfentry_", entryColName);
      fColRegister.AddAlias("tdfslot_", slotColName);
   }

   std::vector<std::string> GetColumnTypeNamesList(const ColumnNames_t &columnList)
   {
      std::vector<std::string> types;

      for (auto column : columnList) {
         types.push_back(GetColumnType(column));
      }
      return types;
   }

   void CheckIMTDisabled(std::string_view callerName)
   {
      if (ROOT::IsImplicitMTEnabled()) {
         std::string error(callerName);
         error += " was called with ImplicitMT enabled, but multi-thread is not supported.";
         throw std::runtime_error(error);
      }
   }

   /// Create RAction object, return RResultPtr for the action
   /// Overload for the case in which all column types were specified (no jitting).
   /// For most actions, `r` and `helperArg` will refer to the same object, because the only argument to forward to
   /// the action helper is the result value itself. We need the distinction for actions such as Snapshot or Cache,
   /// for which the constructor arguments of the action helper are different from the returned value.
   template <typename ActionTag, typename... ColTypes, typename ActionResultType,
             typename HelperArgType = ActionResultType,
             std::enable_if_t<!RDFInternal::RNeedJitting<ColTypes...>::value, int> = 0>
   RResultPtr<ActionResultType>
   CreateAction(const ColumnNames_t &columns, const std::shared_ptr<ActionResultType> &r,
                const std::shared_ptr<HelperArgType> &helperArg, const int /*nColumns*/ = -1)
   {
      constexpr auto nColumns = sizeof...(ColTypes);

      const auto validColumnNames = GetValidatedColumnNames(nColumns, columns);
      CheckAndFillDSColumns(validColumnNames, RDFInternal::TypeList<ColTypes...>());

      const auto nSlots = fLoopManager->GetNSlots();

      auto action = RDFInternal::BuildAction<ColTypes...>(validColumnNames, helperArg, nSlots, fProxiedPtr, ActionTag{},
                                                          fColRegister);
      return MakeResultPtr(r, *fLoopManager, std::move(action));
   }

   /// Create RAction object, return RResultPtr for the action
   /// Overload for the case in which one or more column types were not specified (RTTI + jitting).
   /// This overload has a `nColumns` optional argument. If present, the number of required columns for
   /// this action is taken equal to nColumns, otherwise it is assumed to be sizeof...(ColTypes).
   template <typename ActionTag, typename... ColTypes, typename ActionResultType,
             typename HelperArgType = ActionResultType,
             std::enable_if_t<RDFInternal::RNeedJitting<ColTypes...>::value, int> = 0>
   RResultPtr<ActionResultType> CreateAction(const ColumnNames_t &columns, const std::shared_ptr<ActionResultType> &r,
                                             const std::shared_ptr<HelperArgType> &helperArg, const int nColumns = -1)
   {
      auto realNColumns = (nColumns > -1 ? nColumns : sizeof...(ColTypes));

      const auto validColumnNames = GetValidatedColumnNames(realNColumns, columns);
      const unsigned int nSlots = fLoopManager->GetNSlots();

      auto *tree = fLoopManager->GetTree();
      auto *helperArgOnHeap = RDFInternal::MakeSharedOnHeap(helperArg);

      auto upcastNodeOnHeap = RDFInternal::MakeSharedOnHeap(RDFInternal::UpcastNode(fProxiedPtr));
      using BaseNodeType_t = typename std::remove_pointer_t<decltype(upcastNodeOnHeap)>::element_type;
      RInterface<BaseNodeType_t> upcastInterface(*upcastNodeOnHeap, *fLoopManager, fColRegister, fDataSource);

      const auto jittedAction = std::make_shared<RDFInternal::RJittedAction>(
         *fLoopManager, validColumnNames, fColRegister, fProxiedPtr->GetVariations());
      auto jittedActionOnHeap = RDFInternal::MakeWeakOnHeap(jittedAction);

      auto toJit = RDFInternal::JitBuildAction(
         validColumnNames, upcastNodeOnHeap, typeid(std::shared_ptr<HelperArgType>), typeid(ActionTag), helperArgOnHeap,
         tree, nSlots, fColRegister, fDataSource, jittedActionOnHeap);
      fLoopManager->ToJitExec(toJit);
      return MakeResultPtr(r, *fLoopManager, std::move(jittedAction));
   }

   template <typename F, typename DefineType, typename RetType = typename TTraits::CallableTraits<F>::ret_type>
   std::enable_if_t<std::is_default_constructible<RetType>::value, RInterface<Proxied, DS_t>>
   DefineImpl(std::string_view name, F &&expression, const ColumnNames_t &columns, const std::string &where)
   {
      RDFInternal::CheckValidCppVarName(name, where);
      if (where.compare(0, 8, "Redefine") != 0) { // not a Redefine
         RDFInternal::CheckForRedefinition(where, name, fColRegister, fLoopManager->GetBranchNames(),
                                           fDataSource ? fDataSource->GetColumnNames() : ColumnNames_t{});
      } else {
         RDFInternal::CheckForDefinition(where, name, fColRegister, fLoopManager->GetBranchNames(),
                                         fDataSource ? fDataSource->GetColumnNames() : ColumnNames_t{});
         RDFInternal::CheckForNoVariations(where, name, fColRegister);
      }

      using ArgTypes_t = typename TTraits::CallableTraits<F>::arg_types;
      using ColTypesTmp_t = typename RDFInternal::RemoveFirstParameterIf<
         std::is_same<DefineType, RDFDetail::CustomColExtraArgs::Slot>::value, ArgTypes_t>::type;
      using ColTypes_t = typename RDFInternal::RemoveFirstTwoParametersIf<
         std::is_same<DefineType, RDFDetail::CustomColExtraArgs::SlotAndEntry>::value, ColTypesTmp_t>::type;

      constexpr auto nColumns = ColTypes_t::list_size;

      const auto validColumnNames = GetValidatedColumnNames(nColumns, columns);
      CheckAndFillDSColumns(validColumnNames, ColTypes_t());

      // Declare return type to the interpreter, for future use by jitted actions
      auto retTypeName = RDFInternal::TypeID2TypeName(typeid(RetType));
      if (retTypeName.empty()) {
         // The type is not known to the interpreter.
         // We must not error out here, but if/when this column is used in jitted code
         const auto demangledType = RDFInternal::DemangleTypeIdName(typeid(RetType));
         retTypeName = "CLING_UNKNOWN_TYPE_" + demangledType;
      }

      using NewCol_t = RDFDetail::RDefine<F, DefineType>;
      auto newColumn = std::make_shared<NewCol_t>(name, retTypeName, std::forward<F>(expression), validColumnNames,
                                                  fColRegister, *fLoopManager);

      RDFInternal::RColumnRegister newCols(fColRegister);
      newCols.AddColumn(newColumn);

      RInterface<Proxied> newInterface(fProxiedPtr, *fLoopManager, std::move(newCols), fDataSource);

      return newInterface;
   }

   // This overload is chosen when the callable passed to Define or DefineSlot returns void.
   // It simply fires a compile-time error. This is preferable to a static_assert in the main `Define` overload because
   // this way compilation of `Define` has no way to continue after throwing the error.
   template <typename F, typename DefineType, typename RetType = typename TTraits::CallableTraits<F>::ret_type,
             bool IsFStringConv = std::is_convertible<F, std::string>::value,
             bool IsRetTypeDefConstr = std::is_default_constructible<RetType>::value>
   std::enable_if_t<!IsFStringConv && !IsRetTypeDefConstr, RInterface<Proxied, DS_t>>
   DefineImpl(std::string_view, F, const ColumnNames_t &)
   {
      static_assert(std::is_default_constructible<typename TTraits::CallableTraits<F>::ret_type>::value,
                    "Error in `Define`: type returned by expression is not default-constructible");
      return *this; // never reached
   }

   template <typename... ColumnTypes>
   RResultPtr<RInterface<RLoopManager>> SnapshotImpl(std::string_view fullTreeName, std::string_view filename,
                                                     const ColumnNames_t &columnList, const RSnapshotOptions &options)
   {
      const auto columnListWithoutSizeColumns = RDFInternal::FilterArraySizeColNames(columnList, "Snapshot");

      RDFInternal::CheckTypesAndPars(sizeof...(ColumnTypes), columnListWithoutSizeColumns.size());
      const auto validCols = GetValidatedColumnNames(columnListWithoutSizeColumns.size(), columnListWithoutSizeColumns);
      RDFInternal::CheckForDuplicateSnapshotColumns(validCols);
      CheckAndFillDSColumns(validCols, TTraits::TypeList<ColumnTypes...>());

      const auto parsedTreePath = RDFInternal::ParseTreePath(fullTreeName);
      const auto &treename = parsedTreePath.fTreeName;
      const auto &dirname = parsedTreePath.fDirName;

      auto snapHelperArgs = std::make_shared<RDFInternal::SnapshotHelperArgs>(RDFInternal::SnapshotHelperArgs{
         std::string(filename), std::string(dirname), std::string(treename), columnListWithoutSizeColumns, options});

      ::TDirectory::TContext ctxt;
      auto newRDF = std::make_shared<ROOT::RDataFrame>(fullTreeName, filename, validCols);

      auto resPtr = CreateAction<RDFInternal::ActionTags::Snapshot, ColumnTypes...>(validCols, newRDF, snapHelperArgs);

      if (!options.fLazy)
         *resPtr;
      return resPtr;
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Implementation of cache.
   template <typename... ColTypes, std::size_t... S>
   RInterface<RLoopManager> CacheImpl(const ColumnNames_t &columnList, std::index_sequence<S...>)
   {
      const auto columnListWithoutSizeColumns = RDFInternal::FilterArraySizeColNames(columnList, "Snapshot");

      // Check at compile time that the columns types are copy constructible
      constexpr bool areCopyConstructible =
         RDFInternal::TEvalAnd<std::is_copy_constructible<ColTypes>::value...>::value;
      static_assert(areCopyConstructible, "Columns of a type which is not copy constructible cannot be cached yet.");

      RDFInternal::CheckTypesAndPars(sizeof...(ColTypes), columnListWithoutSizeColumns.size());

      auto colHolders = std::make_tuple(Take<ColTypes>(columnListWithoutSizeColumns[S])...);
      auto ds = std::make_unique<RLazyDS<ColTypes...>>(
         std::make_pair(columnListWithoutSizeColumns[S], std::get<S>(colHolders))...);

      RInterface<RLoopManager> cachedRDF(std::make_shared<RLoopManager>(std::move(ds), columnListWithoutSizeColumns));

      return cachedRDF;
   }

   template <typename Helper, typename ActionResultType>
   auto CallCreateActionWithoutColsIfPossible(const std::shared_ptr<ActionResultType> &resPtr,
                                              const std::shared_ptr<Helper> &hPtr,
                                              TTraits::TypeList<RDFDetail::RInferredType>)
      -> decltype(hPtr->Exec(0u), RResultPtr<ActionResultType>{})
   {
      return CreateAction<RDFInternal::ActionTags::Book>(/*columns=*/{}, resPtr, hPtr, 0u);
   }

   template <typename Helper, typename ActionResultType, typename... Others>
   RResultPtr<ActionResultType>
   CallCreateActionWithoutColsIfPossible(const std::shared_ptr<ActionResultType> &,
                                         const std::shared_ptr<Helper>& /*hPtr*/,
                                         Others...)
   {
      throw std::logic_error(std::string("An action was booked with no input columns, but the action requires "
                                         "columns! The action helper type was ") +
                             typeid(Helper).name());
      return {};
   }

protected:
   RInterface(const std::shared_ptr<Proxied> &proxied, RLoopManager &lm, const RDFInternal::RColumnRegister &columns,
              RDataSource *ds)
      : fProxiedPtr(proxied), fLoopManager(&lm), fDataSource(ds), fColRegister(columns)
   {
   }

   RLoopManager *GetLoopManager() const { return fLoopManager; }

   const std::shared_ptr<Proxied> &GetProxiedPtr() const { return fProxiedPtr; }

   ColumnNames_t GetValidatedColumnNames(const unsigned int nColumns, const ColumnNames_t &columns)
   {
      return RDFInternal::GetValidatedColumnNames(*fLoopManager, nColumns, columns, fColRegister, fDataSource);
   }

   template <typename... ColumnTypes>
   void CheckAndFillDSColumns(ColumnNames_t validCols, TTraits::TypeList<ColumnTypes...> typeList)
   {
      if (fDataSource != nullptr)
         RDFInternal::AddDSColumns(validCols, *fLoopManager, *fDataSource, typeList, fColRegister);
   }
};

} // namespace RDF

} // namespace ROOT

#endif // ROOT_RDF_INTERFACE
