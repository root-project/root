// Author: Enrico Guiraud, Danilo Piparo CERN  02/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RLAZYDSIMPL
#define ROOT_RLAZYDSIMPL

#include "ROOT/RDataSource.hxx"
#include "ROOT/RResultPtr.hxx"
#include "ROOT/TSeq.hxx"

#include <algorithm>
#include <map>
#include <memory>
#include <tuple>
#include <string>
#include <typeinfo>
#include <utility> // std::index_sequence
#include <vector>

namespace ROOT {

namespace RDF {
////////////////////////////////////////////////////////////////////////////////////////////////
/// \brief A RDataSource implementation which is built on top of result proxies
///
/// This component allows to create a data source on a set of columns coming from
/// one or multiple data frames. The processing of the parent data frames starts
/// only when the event loop is triggered in the data frame initialized with a
/// RLazyDS.
///
/// The implementation takes care of matching compile time information with runtime
/// information, e.g. expanding in a smart way the template parameters packs.
template <typename... ColumnTypes>
class RLazyDS final : public ROOT::RDF::RDataSource {
   using PointerHolderPtrs_t = std::vector<ROOT::Internal::TDS::TPointerHolder *>;

   std::tuple<RResultPtr<std::vector<ColumnTypes>>...> fColumns;
   const std::vector<std::string> fColNames;
   const std::map<std::string, std::string> fColTypesMap;
   // The role of the fPointerHoldersModels is to be initialized with the pack
   // of arguments in the constrcutor signature at construction time
   // Once the number of slots is known, the fPointerHolders are initialized
   // according to the models.
   const PointerHolderPtrs_t fPointerHoldersModels;
   std::vector<PointerHolderPtrs_t> fPointerHolders;
   std::vector<std::pair<ULong64_t, ULong64_t>> fEntryRanges{};
   unsigned int fNSlots{0};

   Record_t GetColumnReadersImpl(std::string_view colName, const std::type_info &id)
   {
      auto colNameStr = std::string(colName);
      // This could be optimised and done statically
      const auto idName = ROOT::Internal::RDF::TypeID2TypeName(id);
      auto it = fColTypesMap.find(colNameStr);
      if (fColTypesMap.end() == it) {
         std::string err = "The specified column name, \"" + colNameStr + "\" is not known to the data source.";
         throw std::runtime_error(err);
      }

      const auto colIdName = it->second;
      if (colIdName != idName) {
         std::string err = "Column " + colNameStr + " has type " + colIdName +
                           " while the id specified is associated to type " + idName;
         throw std::runtime_error(err);
      }

      const auto colBegin = fColNames.begin();
      const auto colEnd = fColNames.end();
      const auto namesIt = std::find(colBegin, colEnd, colName);
      const auto index = std::distance(colBegin, namesIt);

      Record_t ret(fNSlots);
      for (auto slot : ROOT::TSeqU(fNSlots)) {
         ret[slot] = fPointerHolders[index][slot]->GetPointerAddr();
      }
      return ret;
   }

   size_t GetEntriesNumber() { return std::get<0>(fColumns)->size(); }
   template <std::size_t... S>
   void SetEntryHelper(unsigned int slot, ULong64_t entry, std::index_sequence<S...>)
   {
      std::initializer_list<int> expander{
         (*static_cast<ColumnTypes *>(fPointerHolders[S][slot]->GetPointer()) = (*std::get<S>(fColumns))[entry], 0)...};
      (void)expander; // avoid unused variable warnings
   }

   template <std::size_t... S>
   void ColLenghtChecker(std::index_sequence<S...>)
   {
      if (sizeof...(S) < 2)
         return;

      const std::vector<size_t> colLengths{std::get<S>(fColumns)->size()...};
      const auto expectedLen = colLengths[0];
      std::string err;
      for (auto i : TSeqI(1, colLengths.size())) {
         if (expectedLen != colLengths[i]) {
            err += "Column \"" + fColNames[i] + "\" and column \"" + fColNames[0] +
                   "\" have different lengths: " + std::to_string(expectedLen) + " and " +
                   std::to_string(colLengths[i]);
         }
      }
      if (!err.empty()) {
         throw std::runtime_error(err);
      }
   }

protected:
   std::string AsString() { return "lazy data source"; };

public:
   RLazyDS(std::pair<std::string, RResultPtr<std::vector<ColumnTypes>>>... colsNameVals)
      : fColumns(std::tuple<RResultPtr<std::vector<ColumnTypes>>...>(colsNameVals.second...)),
        fColNames({colsNameVals.first...}),
        fColTypesMap({{colsNameVals.first, ROOT::Internal::RDF::TypeID2TypeName(typeid(ColumnTypes))}...}),
        fPointerHoldersModels({new ROOT::Internal::TDS::TTypedPointerHolder<ColumnTypes>(new ColumnTypes())...})
   {
   }

   ~RLazyDS()
   {
      for (auto &&ptrHolderv : fPointerHolders) {
         for (auto &&ptrHolder : ptrHolderv) {
            delete ptrHolder;
         }
      }
   }

   const std::vector<std::string> &GetColumnNames() const { return fColNames; }

   std::vector<std::pair<ULong64_t, ULong64_t>> GetEntryRanges()
   {
      auto entryRanges(std::move(fEntryRanges)); // empty fEntryRanges
      return entryRanges;
   }

   std::string GetTypeName(std::string_view colName) const
   {
      const auto key = std::string(colName);
      return fColTypesMap.at(key);
   }

   bool HasColumn(std::string_view colName) const
   {
      const auto key = std::string(colName);
      const auto endIt = fColTypesMap.end();
      return endIt != fColTypesMap.find(key);
   }

   bool SetEntry(unsigned int slot, ULong64_t entry)
   {
      SetEntryHelper(slot, entry, std::index_sequence_for<ColumnTypes...>());
      return true;
   }

   void SetNSlots(unsigned int nSlots)
   {
      fNSlots = nSlots;
      const auto nCols = fColNames.size();
      fPointerHolders.resize(nCols); // now we need to fill it with the slots, all of the same type
      auto colIndex = 0U;
      for (auto &&ptrHolderv : fPointerHolders) {
         for (auto slot : ROOT::TSeqI(fNSlots)) {
            auto ptrHolder = fPointerHoldersModels[colIndex]->GetDeepCopy();
            ptrHolderv.emplace_back(ptrHolder);
            (void)slot;
         }
         colIndex++;
      }
      for (auto &&ptrHolder : fPointerHoldersModels)
         delete ptrHolder;
   }

   void Initialize()
   {
      ColLenghtChecker(std::index_sequence_for<ColumnTypes...>());
      const auto nEntries = GetEntriesNumber();
      const auto nEntriesInRange = nEntries / fNSlots; // between integers. Should make smaller?
      auto reminder = 1U == fNSlots ? 0 : nEntries % fNSlots;
      fEntryRanges.resize(fNSlots);
      auto init = 0ULL;
      auto end = 0ULL;
      for (auto &&range : fEntryRanges) {
         end = init + nEntriesInRange;
         if (0 != reminder) { // Distribute the reminder among the first chunks
            reminder--;
            end += 1;
         }
         range.first = init;
         range.second = end;
         init = end;
      }
   }

   std::string GetLabel() { return "LazyDS"; }
};

} // ns RDF

} // ns ROOT

#endif
