// Author: Enrico Guiraud, Danilo Piparo CERN  09/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_RDEFINE
#define ROOT_RDF_RDEFINE

#include "ROOT/RDF/ColumnReaderUtils.hxx"
#include "ROOT/RDF/RColumnReaderBase.hxx"
#include "ROOT/RDF/RDefineBase.hxx"
#include "ROOT/RDF/RLoopManager.hxx"
#include "ROOT/RDF/Utils.hxx"
#include <string_view>
#include "ROOT/TypeTraits.hxx"
#include "RtypesCore.h"

#include <array>
#include <deque>
#include <type_traits>
#include <utility> // std::index_sequence
#include <vector>

class TTreeReader;

namespace ROOT {
namespace Detail {
namespace RDF {

using namespace ROOT::TypeTraits;

// clang-format off
namespace ExtraArgsForDefine {
struct None{};
struct Slot{};
struct SlotAndEntry{};
}
// clang-format on

template <typename F, typename ExtraArgsTag = ExtraArgsForDefine::None>
class R__CLING_PTRCHECK(off) RDefine final : public RDefineBase {
   // shortcuts
   using NoneTag = ExtraArgsForDefine::None;
   using SlotTag = ExtraArgsForDefine::Slot;
   using SlotAndEntryTag = ExtraArgsForDefine::SlotAndEntry;
   // other types
   using FunParamTypes_t = typename CallableTraits<F>::arg_types;
   using ColumnTypesTmp_t =
      RDFInternal::RemoveFirstParameterIf_t<std::is_same<ExtraArgsTag, SlotTag>::value, FunParamTypes_t>;
   using ColumnTypes_t =
      RDFInternal::RemoveFirstTwoParametersIf_t<std::is_same<ExtraArgsTag, SlotAndEntryTag>::value, ColumnTypesTmp_t>;
   using TypeInd_t = std::make_index_sequence<ColumnTypes_t::list_size>;
   using ret_type = typename CallableTraits<F>::ret_type;
   using ValuesPerSlot_t = std::vector<ROOT::RVec<ret_type>>;

   F fExpression;
   // Each slot accesses a cache of values for the current bulk
   ValuesPerSlot_t fCachedResultsPerSlot;

   /// Column readers per slot and per input column
   std::vector<std::array<RColumnReaderBase *, ColumnTypes_t::list_size>> fValues;

   /// Define objects corresponding to systematic variations other than nominal for this defined column.
   /// The map key is the full variation name, e.g. "pt:up".
   std::unordered_map<std::string, std::unique_ptr<RDefineBase>> fVariedDefines;

   template <typename ColType>
   auto GetValueChecked(unsigned int slot, std::size_t readerIdx, std::size_t idx) -> ColType &
   {
      if (auto *val = fValues[slot][readerIdx]->template TryGet<ColType>(idx))
         return *val;

      throw std::out_of_range{"RDataFrame: Define could not retrieve value for column '" + fColumnNames[readerIdx] +
                              "' for entry " + std::to_string(idx) +
                              ". You can use the DefaultValueFor operation to provide a default value, or "
                              "FilterAvailable/FilterMissing to discard/keep entries with missing values instead."};
   }

   template <typename... ColTypes, std::size_t... S>
   auto UpdateHelper(unsigned int slot, std::size_t idx, Long64_t /*entry*/, TypeList<ColTypes...>,
                     std::index_sequence<S...>, NoneTag)
   {
      return fExpression(GetValueChecked<ColTypes>(slot, S, idx)...);
      (void)slot; // avoid unused parameter warning
      (void)idx;  // avoid unused parameter warning
   }

   template <typename... ColTypes, std::size_t... S>
   auto UpdateHelper(unsigned int slot, std::size_t idx, Long64_t /*entry*/, TypeList<ColTypes...>,
                     std::index_sequence<S...>, SlotTag)
   {
      return fExpression(slot, GetValueChecked<ColTypes>(slot, S, idx)...);
      (void)slot; // avoid unused parameter warning
      (void)idx;  // avoid unused parameter warning
   }

   template <typename... ColTypes, std::size_t... S>
   auto UpdateHelper(unsigned int slot, std::size_t idx, Long64_t entryInBatch, TypeList<ColTypes...>,
                     std::index_sequence<S...>, SlotAndEntryTag)
   {
      return fExpression(slot, entryInBatch, GetValueChecked<ColTypes>(slot, S, idx)...);
      (void)slot;         // avoid unused parameter warning
      (void)idx;          // avoid unused parameter warning
      (void)entryInBatch; // avoid unused parameter warning
   }

public:
   RDefine(std::string_view name, std::string_view type, F expression, const ROOT::RDF::ColumnNames_t &columns,
           const RDFInternal::RColumnRegister &colRegister, RLoopManager &lm,
           const std::string &variationName = "nominal")
      : RDefineBase(name, type, colRegister, lm, columns, variationName),
        fExpression(std::move(expression)),
        fCachedResultsPerSlot(lm.GetNSlots() * RDFInternal::CacheLineStep<ROOT::RVec<ret_type>>()),
        fValues(lm.GetNSlots())
   {
      fLoopManager->Register(this);
      // Assume 1-size bulk for now
      for (decltype(lm.GetNSlots()) i = 0; i < lm.GetNSlots(); ++i) {
         fCachedResultsPerSlot[i * RDFInternal::CacheLineStep<ROOT::RVec<ret_type>>()].resize(1ul);
      }
   }

   RDefine(const RDefine &) = delete;
   RDefine &operator=(const RDefine &) = delete;
   ~RDefine() { fLoopManager->Deregister(this); }

   void InitSlot(TTreeReader *r, unsigned int slot) final
   {
      RDFInternal::RColumnReadersInfo info{fColumnNames, fColRegister, fIsDefine.data(), *fLoopManager};
      fValues[slot] = RDFInternal::GetColumnReaders(slot, r, ColumnTypes_t{}, info, fVariation);
      fLastCheckedEntry[slot * RDFInternal::CacheLineStep<Long64_t>()] = -1;
   }

   /// Return the beginning of the cached results of the current bulk for the input processing slot
   void *GetValuePtr(unsigned int slot) final
   {
      return static_cast<void *>(
         fCachedResultsPerSlot[slot * RDFInternal::CacheLineStep<ROOT::RVec<ret_type>>()].data());
   }

   /// Update the values at the array returned by GetValuePtr with the content corresponding to the given mask
   void Update(unsigned int slot, const ROOT::Internal::RDF::RMaskedEntryRange &mask) final
   {
      if (static_cast<Long64_t>(mask.GetFirstEntry()) ==
          fLastCheckedEntry[slot * RDFInternal::CacheLineStep<Long64_t>()])
         return;

      std::for_each(fValues[slot].begin(), fValues[slot].end(), [&mask](auto *v) { v->Load(mask); });
      // Assume 1-size bulk for now
      const std::size_t bulkSize = 1;
      auto &result = fCachedResultsPerSlot[slot * RDFInternal::CacheLineStep<ROOT::RVec<ret_type>>()];
      result.clear();
      result.resize(bulkSize);
      for (std::size_t i = 0; i < bulkSize; ++i) {
         if (mask[i]) {
            result[i] = UpdateHelper(slot, i, mask.GetFirstEntry() + i, ColumnTypes_t{}, TypeInd_t{}, ExtraArgsTag{});
            fLastCheckedEntry[slot * RDFInternal::CacheLineStep<Long64_t>()] = mask.GetFirstEntry();
         }
      }
   }

   void Update(unsigned int /*slot*/, const ROOT::RDF::RSampleInfo &/*id*/) final {}

   const std::type_info &GetTypeId() const final { return typeid(ret_type); }

   /// Clean-up operations to be performed at the end of a task.
   void FinalizeSlot(unsigned int slot) final
   {
      fValues[slot].fill(nullptr);

      for (auto &e : fVariedDefines)
         e.second->FinalizeSlot(slot);
   }

   /// Create clones of this Define that work with values in varied "universes".
   void MakeVariations(const std::vector<std::string> &variations) final
   {
      for (const auto &variation : variations) {
         if (std::find(fVariationDeps.begin(), fVariationDeps.end(), variation) == fVariationDeps.end()) {
            // this Defined quantity does not depend on this variation, so no need to create a varied RDefine
            continue;
         }
         if (fVariedDefines.find(variation) != fVariedDefines.end())
            continue; // we already have this variation stored

         // the varied defines get a copy of the callable object.
         // TODO document this
         auto variedDefine = std::unique_ptr<RDefineBase>(
            new RDefine(fName, fType, fExpression, fColumnNames, fColRegister, *fLoopManager, variation));
         // TODO switch to fVariedDefines.insert({variationName, std::move(variedDefine)}) when we drop gcc 5
         fVariedDefines[variation] = std::move(variedDefine);
      }
   }

   /// Return a clone of this Define that works with values in the variationName "universe".
   RDefineBase &GetVariedDefine(const std::string &variationName) final
   {
      auto it = fVariedDefines.find(variationName);
      if (it == fVariedDefines.end()) {
         // We don't have a varied RDefine for this variation.
         // This means we don't depend on it and we can return ourselves, i.e. the RDefine for the nominal universe.
         assert(std::find(fVariationDeps.begin(), fVariationDeps.end(), variationName) == fVariationDeps.end());
         return *this;
      }

      return *(it->second);
   }
};

} // ns RDF
} // ns Detail
} // ns ROOT

#endif // ROOT_RDF_RDEFINE
