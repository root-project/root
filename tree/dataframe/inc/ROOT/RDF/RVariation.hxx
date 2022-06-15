// Author: Enrico Guiraud, CERN 10/2021

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_RVARIATION
#define ROOT_RDF_RVARIATION

#include "Utils.hxx" // IsRVec
#include "ColumnReaderUtils.hxx"
#include "RColumnReaderBase.hxx"
#include "RLoopManager.hxx"
#include "RVariationBase.hxx"

#include <ROOT/RStringView.hxx>
#include <ROOT/TypeTraits.hxx>
#include <Rtypes.h> // R__CLING_PTRCHECK, Long64_t

#include <array>
#include <deque>
#include <map>
#include <string>
#include <type_traits> // std::is_same, std::conditional_t
#include <utility>     // std::index_sequence
#include <vector>

class TTreeReader;

namespace ROOT {

namespace RDF {
class RDataSource;
}

namespace Internal {
namespace RDF {

using namespace ROOT::TypeTraits;

template <typename T>
bool ResultsSizeEq(const RVec<RVec<T>> &results, std::size_t expected, std::size_t nColumns)
{
   if (nColumns == 1)
      return results.size() == expected;

   return std::all_of(results.begin(), results.end(), [&expected](const RVec<T> &v) { return v.size() == expected; });
}

template <typename T>
bool ResultsSizeEq(const RVec<T> &results, std::size_t expected, std::size_t nColumns)
{
   assert(nColumns == 1);
   (void)nColumns;

   return results.size() == expected;
}

template <typename T>
std::size_t GetNVariations(const RVec<RVec<T>> &results)
{
   assert(!results.empty());
   return results[0].size();
}

template <typename T>
std::size_t GetNVariations(const RVec<T> &results)
{
   return results.size();
}

template <typename RVec_t, typename Value_t = typename RVec_t::value_type>
std::enable_if_t<!IsRVec<Value_t>::value, const std::type_info &> GetInnerValueType(std::size_t)
{
   return typeid(Value_t);
}

template <typename RVec_t, typename Value_t = typename RVec_t::value_type>
std::enable_if_t<IsRVec<Value_t>::value, const std::type_info &> GetInnerValueType(std::size_t nCols)
{
   if (nCols == 1) // we are varying one column that is an RVec
      return typeid(Value_t);
   else // we are varying multiple columns whose type is the inner type of this RVec
      return typeid(typename Value_t::value_type);
}

// This overload is for the case of a single column and ret_type != RVec<RVec<...>>
template <typename T>
void ResizeResults(ROOT::RVec<T> &results, std::size_t /*nCols*/, std::size_t nVariations)
{
   results.resize(nVariations);
}

// This overload is for the case of ret_type == RVec<RVec<...>>
template <typename T>
void ResizeResults(ROOT::RVec<ROOT::RVec<T>> &results, std::size_t nCols, std::size_t nVariations)
{
   if (nCols == 1) {
      results.resize(nVariations);
   } else {
      results.resize(nCols);
      for (auto &rvecOverVariations : results) {
         rvecOverVariations.resize(nVariations);
      }
   }
}

// Assign into fLastResults[slot] without changing the addresses of its elements (we gave those addresses away in
// GetValuePtr)
// This overload is for the case of a single column and ret_type != RVec<RVec<...>>
template <typename T>
void AssignResults(ROOT::RVec<T> &resStorage, ROOT::RVec<T> &&tmpResults, std::size_t /*nCols*/)
{
   const auto nVariations = resStorage.size(); // we have already checked that tmpResults has the same size

   for (auto i = 0u; i < nVariations; ++i)
      resStorage[i] = std::move(tmpResults[i]);
}

// This overload is for the case of ret_type == RVec<RVec<...>>
template <typename T>
void AssignResults(ROOT::RVec<ROOT::RVec<T>> &resStorage, ROOT::RVec<ROOT::RVec<T>> &&tmpResults, std::size_t nCols)
{
   // we have already checked that tmpResults has the same inner size
   const auto nVariations = nCols == 1 ? resStorage.size() : resStorage[0].size();

   if (nCols == 1) {
      for (auto varIdx = 0u; varIdx < nVariations; ++varIdx)
         resStorage[varIdx] = std::move(tmpResults[varIdx]);
   } else {
      for (auto colIdx = 0u; colIdx < nCols; ++colIdx)
         for (auto varIdx = 0u; varIdx < nVariations; ++varIdx)
            resStorage[colIdx][varIdx] = std::move(tmpResults[colIdx][varIdx]);
   }
}

template <typename F>
class R__CLING_PTRCHECK(off) RVariation final : public RVariationBase {
   using ColumnTypes_t = typename CallableTraits<F>::arg_types;
   using TypeInd_t = std::make_index_sequence<ColumnTypes_t::list_size>;
   using ret_type = typename CallableTraits<F>::ret_type;

   F fExpression;
   std::vector<ret_type> fLastResults;

   /// Column readers per slot and per input column
   std::vector<std::array<std::shared_ptr<RColumnReaderBase>, ColumnTypes_t::list_size>> fValues;

   template <typename... ColTypes, std::size_t... S>
   void UpdateHelper(unsigned int slot, Long64_t entry, TypeList<ColTypes...>, std::index_sequence<S...>)
   {
      // fExpression must return an RVec<T>
      auto &&results = fExpression(fValues[slot][S]->template Get<ColTypes>(entry)...);

      if (!ResultsSizeEq(results, fVariationNames.size(), fColNames.size())) {
         std::string variationName = fVariationNames[0].substr(0, fVariationNames[0].find_first_of(':'));
         throw std::runtime_error("The evaluation of the expression for variation \"" + variationName +
                                  "\" resulted in " + std::to_string(GetNVariations(results)) + " values, but " +
                                  std::to_string(fVariationNames.size()) + " were expected.");
      }

      AssignResults(fLastResults[slot * CacheLineStep<ret_type>()], std::move(results), fColNames.size());

      // silence "unused parameter" warnings in gcc
      (void)slot;
      (void)entry;
   }

   // This overload is for the case of a single column and ret_type != RVec<RVec<...>> -- the colIdx is ignored.
   template <typename U = typename ret_type::value_type>
   std::enable_if_t<!IsRVec<U>::value, void *>
   GetValuePtr(unsigned int slot, std::size_t /*colIdx*/, std::size_t varIdx)
   {
      auto &value = fLastResults[slot * CacheLineStep<ret_type>()][varIdx];
      return static_cast<void *>(&value);
   }

   // This overload is for the case of ret_type == RVec<RVec<...>>
   template <typename U = typename ret_type::value_type>
   std::enable_if_t<IsRVec<U>::value, void *> GetValuePtr(unsigned int slot, std::size_t colIdx, std::size_t varIdx)
   {
      if (fColNames.size() == 1) {
         auto &value = fLastResults[slot * CacheLineStep<ret_type>()][varIdx];
         return static_cast<void *>(&value);
      }

      auto &value = fLastResults[slot * CacheLineStep<ret_type>()][colIdx][varIdx];
      return static_cast<void *>(&value);
   }

public:
   RVariation(const std::vector<std::string> &colNames, std::string_view variationName, F expression,
              const std::vector<std::string> &variationTags, std::string_view type, const RColumnRegister &defines,
              RLoopManager &lm, const ColumnNames_t &inputColNames)
      : RVariationBase(colNames, variationName, variationTags, type, defines, lm, inputColNames),
        fExpression(std::move(expression)), fLastResults(lm.GetNSlots() * RDFInternal::CacheLineStep<ret_type>()),
        fValues(lm.GetNSlots())
   {
      fLoopManager->Register(this);

      for (auto i = 0u; i < lm.GetNSlots(); ++i)
         ResizeResults(fLastResults[i * RDFInternal::CacheLineStep<ret_type>()], colNames.size(), variationTags.size());
   }

   RVariation(const RVariation &) = delete;
   RVariation &operator=(const RVariation &) = delete;
   ~RVariation() { fLoopManager->Deregister(this); }

   void InitSlot(TTreeReader *r, unsigned int slot) final
   {
      RColumnReadersInfo info{fInputColumns, fColumnRegister, fIsDefine.data(), *fLoopManager};
      fValues[slot] = MakeColumnReaders(slot, r, ColumnTypes_t{}, info);
      fLastCheckedEntry[slot * CacheLineStep<Long64_t>()] = -1;
   }

   /// Return the (type-erased) address of the value for the given processing slot.
   void *GetValuePtr(unsigned int slot, const std::string &column, const std::string &variation) final
   {
      const auto colIt = std::find(fColNames.begin(), fColNames.end(), column);
      assert(colIt != fColNames.end());
      const auto colIdx = std::distance(fColNames.begin(), colIt);

      const auto varIt = std::find(fVariationNames.begin(), fVariationNames.end(), variation);
      assert(varIt != fVariationNames.end());
      const auto varIdx = std::distance(fVariationNames.begin(), varIt);

      return GetValuePtr(slot, colIdx, varIdx);
   }

   /// Update the value at the address returned by GetValuePtr with the content corresponding to the given entry
   void Update(unsigned int slot, Long64_t entry) final
   {
      if (entry != fLastCheckedEntry[slot * CacheLineStep<Long64_t>()]) {
         // evaluate this filter, cache the result
         UpdateHelper(slot, entry, ColumnTypes_t{}, TypeInd_t{});
         fLastCheckedEntry[slot * CacheLineStep<Long64_t>()] = entry;
      }
   }

   const std::type_info &GetTypeId() const { return GetInnerValueType<ret_type>(fColNames.size()); }

   /// Clean-up operations to be performed at the end of a task.
   void FinalizeSlot(unsigned int slot) final
   {
      for (auto &v : fValues[slot])
         v.reset();
   }
};

} // namespace RDF
} // namespace Internal
} // namespace ROOT

#endif // ROOT_RDF_RVARIATION
