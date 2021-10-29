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

template <typename F>
class R__CLING_PTRCHECK(off) RVariation final : public RVariationBase {
   using ColumnTypes_t = typename CallableTraits<F>::arg_types;
   using TypeInd_t = std::make_index_sequence<ColumnTypes_t::list_size>;
   using ret_type = typename CallableTraits<F>::ret_type;
   // Avoid instantiating vector<bool> as `operator[]` returns temporaries in that case. Use std::deque instead.
   using ValuesPerSlot_t =
      std::conditional_t<std::is_same<ret_type, bool>::value, std::deque<ret_type>, std::vector<ret_type>>;

   F fExpression;
   ValuesPerSlot_t fLastResults;

   /// Column readers per slot and per input column
   std::vector<std::array<std::unique_ptr<RColumnReaderBase>, ColumnTypes_t::list_size>> fValues;

   template <typename... ColTypes, std::size_t... S>
   void UpdateHelper(unsigned int slot, Long64_t entry, TypeList<ColTypes...>, std::index_sequence<S...>)
   {
      // fExpression must return an RVec<T>
      const auto &results = fExpression(fValues[slot][S]->template Get<ColTypes>(entry)...);
      R__ASSERT(results.size() == fLastResults[slot * CacheLineStep<ret_type>()].size() &&
                "Variation expression has wrong size.");
      // Assign into fLastResults without changing the addresses of its elements (we gave those addresses away in
      // GetValuePtr)
      fLastResults[slot * CacheLineStep<ret_type>()].assign(results.begin(), results.end());

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
      for (auto i = 0u; i < lm.GetNSlots(); ++i)
         fLastResults[i * RDFInternal::CacheLineStep<ret_type>()].resize(fVariationNames.size());
   }

   RVariation(const RVariation &) = delete;
   RVariation &operator=(const RVariation &) = delete;

   void InitSlot(TTreeReader *r, unsigned int slot) final
   {
      for (auto &define : fColumnRegister.GetColumns())
         define.second->InitSlot(r, slot);
      RColumnReadersInfo info{fInputColumns, fColumnRegister, fIsDefine.data(), fLoopManager->GetDSValuePtrs(),
                              fLoopManager->GetDataSource()};
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

   const std::type_info &GetTypeId() const { return typeid(ret_type); }

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
