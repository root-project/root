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

#include <ROOT/RDF/InterfaceUtils.hxx>
#include <ROOT/RDF/RColumnRegister.hxx>
#include <ROOT/RDF/RDisplay.hxx>
#include <ROOT/RDF/RLoopManager.hxx>
#include <ROOT/RDataSource.hxx>
#include <ROOT/RResultPtr.hxx>
#include <ROOT/RStringView.hxx>
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
   RDFDetail::RLoopManager *fLoopManager;
   /// Non-owning pointer to a data-source object. Null if no data-source. RLoopManager has ownership of the object.
   RDataSource *fDataSource = nullptr;

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
      R__ASSERT(variationTags.size() > 0 && "Must have at least one variation.");
      R__ASSERT(colNames.size() > 0 && "Must have at least one varied column.");
      R__ASSERT(!variationName.empty() && "Must provide a variation name.");

      for (auto &colName : colNames) {
         RDFInternal::CheckForDefinition("Vary", colName, fColRegister, fLoopManager->GetBranchNames(),
                                         fDataSource ? fDataSource->GetColumnNames() : ColumnNames_t{});
      }
      RDFInternal::CheckValidCppVarName(variationName, "Vary");

      static_assert(RDFInternal::IsRVec<RetType>::value, "Vary expressions must return an RVec.");

      if (colNames.size() > 1) { // we are varying multiple columns simultaneously, RetType is RVec<RVec<T>>
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

         for (auto i = 0u; i < colTypes.size(); ++i) {
            const auto *define = fColRegister.GetDefine(colNames[i]);
            const auto &expectedTypeID = define ? define->GetTypeId() : RDFInternal::TypeName2TypeID(colTypes[i]);
            if (innerTypeID != expectedTypeID)
               throw std::runtime_error("Varied values for column \"" + colNames[i] + "\" have a different type (" +
                                        RDFInternal::TypeID2TypeName(innerTypeID) + ") than the nominal value (" +
                                        colTypes[i] + ").");
         }
      } else { // we are varying a single column, RetType is RVec<T>
         const auto &retTypeID = typeid(typename RetType::value_type);
         const auto &colName = colNames[0]; // we have only one element in there
         const auto *define = fColRegister.GetDefine(colName);
         const auto &expectedTypeID =
            define ? define->GetTypeId() : RDFInternal::TypeName2TypeID(GetColumnType(colName));
         if (retTypeID != expectedTypeID)
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

   RDFDetail::RLoopManager *GetLoopManager() const { return fLoopManager; }

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
};
} // namespace RDF
} // namespace ROOT

#endif
