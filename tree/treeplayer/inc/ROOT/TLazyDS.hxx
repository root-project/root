#ifndef ROOT_TLAZYDS
#define ROOT_TLAZYDS

#include "ROOT/TDataSource.hxx"
#include "ROOT/TDataFrame.hxx"
#include "ROOT/TResultProxy.hxx"
#include "ROOT/RMakeUnique.hxx"
#include "ROOT/TSeq.hxx"

#include <algorithm>
#include <map>
#include <tuple>
#include <string>
#include <typeinfo>
#include <vector>

namespace ROOT {
namespace Experimental {
namespace TDF {
////////////////////////////////////////////////////////////////////////////////////////////////
/// \brief A TDataSource implementation which is built on top of result proxies
///
/// This component allows to create a data source on a set of columns coming from
/// one or multiple data frames. The processing of the parent data frames starts
/// only when the event loop is triggered in the data frame initialised with a
/// TLazyDS.
///
/// The implementation takes care of matching compile time information with runtime
/// information, e.g. expanding in a smart way the template parameters packs.
template <typename... ColumnTypes>
class TLazyDS final : public ROOT::Experimental::TDF::TDataSource {
   using PointerHolderPtrs_t = std::vector<ROOT::Internal::TDS::TPointerHolder *>;

   std::tuple<TResultProxy<std::vector<ColumnTypes>>...> fColumns;
   const std::vector<std::string> fColNames;
   const std::map<std::string, std::string> fColTypesMap;
   const PointerHolderPtrs_t fPointerHoldersModels;
   std::vector<PointerHolderPtrs_t> fPointerHolders;
   std::vector<std::pair<ULong64_t, ULong64_t>> fEntryRanges







   {};
   unsigned int fNSlots{0};

   Record_t GetColumnReadersImpl(std::string_view colName, const std::type_info &id)
   {
      auto colNameStr = std::string(colName);
      // This could be optimised and done statically
      const auto idName = ROOT::Internal::TDF::TypeID2TypeName(id);
      auto it = fColTypesMap.find(colNameStr);
      if (fColTypesMap.end() == it) {
         std::string err = "The specified column name, \"" + colNameStr
                         + "\" is not known to the data source.";
         throw std::runtime_error(err);
      }

      const auto colIdName = it->second;
      if (colIdName != idName) {
         std::string err = "Column " + colNameStr + " has type " + colIdName
                         + " while the id specified is associated to type " + idName;
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
   template <int... S>
   void SetEntryHelper(unsigned int slot, ULong64_t entry, ROOT::Internal::TDF::StaticSeq<S...>)
   {
      std::initializer_list<int> expander{
         (*((ColumnTypes *)fPointerHolders[S][slot]->GetPointer()) = (*std::get<S>(fColumns))[entry], 0)...};
      (void)expander; // avoid unused variable warnings
   }

   template <int... S>
   void ColLenghtChecker(ROOT::Internal::TDF::StaticSeq<S...>)
   {
      const std::vector<size_t> colLenghts{std::get<S>(fColumns)->size()...};
      const auto colLength = colLenghts[0];
      std::string err;
      for (auto i : TSeqI(colLenghts.size())) {
         if (colLength != colLenghts[i]) {
            err += "Column \"" + fColNames[i] + "\" and column \"" + fColNames[0] + "\" have different lengths: " +
                   colLenghts[0] + " and " + colLenghts[1];
         }
      }
      if (!err.empty()) {
         throw std::runtime_error(err);
      }
   }

public:
   TLazyDS(std::pair<std::string, TResultProxy<std::vector<ColumnTypes>>>... colsNameVals)
      : fColumns(std::tuple<TResultProxy<std::vector<ColumnTypes>>...>(colsNameVals.second...)),
        fColNames{colsNameVals.first...},
        fColTypesMap({{colsNameVals.first, ROOT::Internal::TDF::TypeID2TypeName(typeid(ColumnTypes))}...}),
        fPointerHoldersModels{new ROOT::Internal::TDS::TTypedPointerHolder<ColumnTypes>(new ColumnTypes())...}
   {
   }

   ~TLazyDS()
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

   void SetEntry(unsigned int slot, ULong64_t entry)
   {
      SetEntryHelper(slot, entry, ROOT::Internal::TDF::GenStaticSeq_t<sizeof...(ColumnTypes)>());
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

   void Initialise()
   {
      ColLenghtChecker(ROOT::Internal::TDF::GenStaticSeq_t<sizeof...(ColumnTypes)>());
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
};

// clang-format off
////////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Factory method to create a Lazy TDataFrame.
/// \param[in] colNameProxyPairs the series of pairs to describe the columns of the data source, first element of the pair is the name of the column and the second is the TResultProxy to the column in the parent data frame.
// clang-format on
template<typename... ColumnTypes>
TDataFrame MakeLazyDataFrame(std::pair<std::string, TResultProxy<std::vector<ColumnTypes>>>&&... colNameProxyPairs)
{
   TDataFrame tdf(std::make_unique<TLazyDS<ColumnTypes...>>(std::forward<std::pair<std::string, TResultProxy<std::vector<ColumnTypes>>>>(colNameProxyPairs)...));
   return tdf;
}


} // ns TDF
} // ns Experimental
} // ns ROOT

#endif
