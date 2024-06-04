// Author: Enrico Guiraud CERN 08/2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_RINTERFACEBASE
#define ROOT_RDF_RINTERFACEBASE

#include "ROOT/RVec.hxx"
#include <ROOT/RDF/InterfaceUtils.hxx>
#include <ROOT/RDF/RColumnRegister.hxx>
#include <ROOT/RDF/RDisplay.hxx>
#include <ROOT/RDF/RLoopManager.hxx>
#include <ROOT/RDataSource.hxx>
#include <ROOT/RResultPtr.hxx>
#include <string_view>
#include <TError.h> // R__ASSERT

#include <memory>
#include <set>
#include <string>
#include <vector>

namespace ROOT {
namespace RDF {

class RDFDescription;
class RVariationsDescription;

using ColumnNames_t = std::vector<std::string>;

namespace RDFDetail = ROOT::Detail::RDF;
namespace RDFInternal = ROOT::Internal::RDF;

// clang-format off
/**
 * \class ROOT::Internal::RDF::RInterfaceBase
 * \ingroup dataframe
 * \brief The public interface to the RDataFrame federation of classes.
 * \tparam Proxied One of the "node" base types (e.g. RLoopManager, RFilterBase). The user never specifies this type manually.
 * \tparam DataSource The type of the RDataSource which is providing the data to the data frame. There is no source by default.
 *
 * The documentation of each method features a one liner illustrating how to use the method, for example showing how
 * the majority of the template parameters are automatically deduced requiring no or very little effort by the user.
 */
// clang-format on
class RInterfaceBase {
protected:
   ///< The RLoopManager at the root of this computation graph. Never null.
   std::shared_ptr<ROOT::Detail::RDF::RLoopManager> fLoopManager;

   /// Contains the columns defined up to this node.
   RDFInternal::RColumnRegister fColRegister;

   std::string DescribeDataset() const;

   ColumnNames_t GetColumnTypeNamesList(const ColumnNames_t &columnList);

   void CheckIMTDisabled(std::string_view callerName);

   void AddDefaultColumns();

   template <typename RetType>
   void SanityChecksForVary(const std::vector<std::string> &colNames, const std::vector<std::string> &variationTags,
                            std::string_view variationName)
   {
      R__ASSERT(!variationTags.empty() && "Must have at least one variation.");
      R__ASSERT(!colNames.empty() && "Must have at least one varied column.");
      R__ASSERT(!variationName.empty() && "Must provide a variation name.");

      for (auto &colName : colNames) {
         RDFInternal::CheckForDefinition("Vary", colName, fColRegister, fLoopManager->GetBranchNames(),
                                         GetDataSource() ? GetDataSource()->GetColumnNames() : ColumnNames_t{});
      }
      RDFInternal::CheckValidCppVarName(variationName, "Vary");

      static_assert(ROOT::Internal::VecOps::IsRVec<RetType>::value, "Vary expressions must return an RVec.");

      if (colNames.size() > 1) { // we are varying multiple columns simultaneously, RetType is RVec<RVec<T>>
         constexpr bool hasInnerRVec = ROOT::Internal::VecOps::IsRVec<typename RetType::value_type>::value;
         if (!hasInnerRVec)
            throw std::runtime_error("This Vary call is varying multiple columns simultaneously but the expression "
                                     "does not return an RVec of RVecs.");

         // Check for type mismatches. We are interested in two cases:
         // - All columns that are going to be varied must be of the same type
         // - The return type of the expression must match the type of the nominal column
         auto colTypes = GetColumnTypeNamesList(colNames);
         auto &&nColTypes = colTypes.size();
         // Cache type_info when requested
         std::vector<const std::type_info *> colTypeIDs(nColTypes);
         const auto &innerTypeID = typeid(RDFInternal::InnerValueType_t<RetType>);
         for (decltype(nColTypes) i{}; i < nColTypes; ++i) {
            // Need to retrieve the type_info for each column. We start with
            // checking if the column comes from a Define, in which case the
            // type_info is cached already. Otherwise, we need to retrieve it
            // via TypeName2TypeID, which eventually might call the interpreter.
            const auto *define = fColRegister.GetDefine(colNames[i]);
            colTypeIDs[i] = define ? &define->GetTypeId() : &RDFInternal::TypeName2TypeID(colTypes[i]);
            // First check: whether the current column type is the same as the first one.
            if (*colTypeIDs[i] != *colTypeIDs[0]) {
               throw std::runtime_error("Cannot simultaneously vary multiple columns of different types.");
            }
            // Second check: mismatch between varied type and nominal type
            if (innerTypeID != *colTypeIDs[i])
               throw std::runtime_error("Varied values for column \"" + colNames[i] + "\" have a different type (" +
                                        RDFInternal::TypeID2TypeName(innerTypeID) + ") than the nominal value (" +
                                        colTypes[i] + ").");
         }

      } else { // we are varying a single column, RetType is RVec<T>
         const auto &retTypeID = typeid(typename RetType::value_type);
         const auto &colName = colNames[0]; // we have only one element in there
         const auto *define = fColRegister.GetDefine(colName);
         const auto *expectedTypeID =
            define ? &define->GetTypeId() : &RDFInternal::TypeName2TypeID(GetColumnType(colName));
         if (retTypeID != *expectedTypeID)
            throw std::runtime_error("Varied values for column \"" + colName + "\" have a different type (" +
                                     RDFInternal::TypeID2TypeName(retTypeID) + ") than the nominal value (" +
                                     GetColumnType(colName) + ").");
      }

      // when varying multiple columns, they must be different columns
      if (colNames.size() > 1) {
         std::set<std::string> uniqueCols(colNames.begin(), colNames.end());
         if (uniqueCols.size() != colNames.size())
            throw std::logic_error("A column name was passed to the same Vary invocation multiple times.");
      }
   }

   RDFDetail::RLoopManager *GetLoopManager() const { return fLoopManager.get(); }
   RDataSource *GetDataSource() const { return fLoopManager->GetDataSource(); }

   ColumnNames_t GetValidatedColumnNames(const unsigned int nColumns, const ColumnNames_t &columns)
   {
      return RDFInternal::GetValidatedColumnNames(*fLoopManager, nColumns, columns, fColRegister, GetDataSource());
   }

   template <typename... ColumnTypes>
   void CheckAndFillDSColumns(ColumnNames_t validCols, TTraits::TypeList<ColumnTypes...> typeList)
   {
      if (auto dataSource = GetDataSource())
         RDFInternal::AddDSColumns(validCols, *fLoopManager, *dataSource, typeList, fColRegister);
   }

   /// Create RAction object, return RResultPtr for the action
   /// Overload for the case in which all column types were specified (no jitting).
   /// For most actions, `r` and `helperArg` will refer to the same object, because the only argument to forward to
   /// the action helper is the result value itself. We need the distinction for actions such as Snapshot or Cache,
   /// for which the constructor arguments of the action helper are different from the returned value.
   template <typename ActionTag, typename... ColTypes, typename ActionResultType, typename RDFNode,
             typename HelperArgType = ActionResultType,
             std::enable_if_t<!RDFInternal::RNeedJitting<ColTypes...>::value, int> = 0>
   RResultPtr<ActionResultType> CreateAction(const ColumnNames_t &columns, const std::shared_ptr<ActionResultType> &r,
                                             const std::shared_ptr<HelperArgType> &helperArg,
                                             const std::shared_ptr<RDFNode> &proxiedPtr, const int /*nColumns*/ = -1)
   {
      constexpr auto nColumns = sizeof...(ColTypes);

      const auto validColumnNames = GetValidatedColumnNames(nColumns, columns);
      CheckAndFillDSColumns(validColumnNames, RDFInternal::TypeList<ColTypes...>());

      const auto nSlots = fLoopManager->GetNSlots();

      auto action = RDFInternal::BuildAction<ColTypes...>(validColumnNames, helperArg, nSlots, proxiedPtr, ActionTag{},
                                                          fColRegister);
      return MakeResultPtr(r, *fLoopManager, std::move(action));
   }

   /// Create RAction object, return RResultPtr for the action
   /// Overload for the case in which one or more column types were not specified (RTTI + jitting).
   /// This overload has a `nColumns` optional argument. If present, the number of required columns for
   /// this action is taken equal to nColumns, otherwise it is assumed to be sizeof...(ColTypes).
   template <typename ActionTag, typename... ColTypes, typename ActionResultType, typename RDFNode,
             typename HelperArgType = ActionResultType,
             std::enable_if_t<RDFInternal::RNeedJitting<ColTypes...>::value, int> = 0>
   RResultPtr<ActionResultType>
   CreateAction(const ColumnNames_t &columns, const std::shared_ptr<ActionResultType> &r,
                const std::shared_ptr<HelperArgType> &helperArg, const std::shared_ptr<RDFNode> &proxiedPtr,
                const int nColumns = -1, const bool vector2RVec = true)
   {
      auto realNColumns = (nColumns > -1 ? nColumns : sizeof...(ColTypes));

      const auto validColumnNames = GetValidatedColumnNames(realNColumns, columns);
      const unsigned int nSlots = fLoopManager->GetNSlots();

      auto *tree = fLoopManager->GetTree();
      auto *helperArgOnHeap = RDFInternal::MakeSharedOnHeap(helperArg);

      auto upcastNodeOnHeap = RDFInternal::MakeSharedOnHeap(RDFInternal::UpcastNode(proxiedPtr));

      const auto jittedAction = std::make_shared<RDFInternal::RJittedAction>(*fLoopManager, validColumnNames,
                                                                             fColRegister, proxiedPtr->GetVariations());
      auto jittedActionOnHeap = RDFInternal::MakeWeakOnHeap(jittedAction);

      auto toJit = RDFInternal::JitBuildAction(validColumnNames, upcastNodeOnHeap, typeid(HelperArgType),
                                               typeid(ActionTag), helperArgOnHeap, tree, nSlots, fColRegister,
                                               GetDataSource(), jittedActionOnHeap, vector2RVec);
      fLoopManager->ToJitExec(toJit);
      return MakeResultPtr(r, *fLoopManager, std::move(jittedAction));
   }

public:
   RInterfaceBase(std::shared_ptr<RDFDetail::RLoopManager> lm);
   RInterfaceBase(RDFDetail::RLoopManager &lm, const RDFInternal::RColumnRegister &colRegister);

   ColumnNames_t GetColumnNames();

   std::string GetColumnType(std::string_view column);

   RDFDescription Describe();

   RVariationsDescription GetVariations() const;
   bool HasColumn(std::string_view columnName);
   ColumnNames_t GetDefinedColumnNames();
   unsigned int GetNSlots() const;
   unsigned int GetNRuns() const;
   unsigned int GetNFiles();
};
} // namespace RDF
} // namespace ROOT

#endif
