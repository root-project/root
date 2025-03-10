// Author: Vincenzo Eduardo Padulano, CERN 09/2024

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_RDefaultValueFor
#define ROOT_RDF_RDefaultValueFor

#include "ROOT/RDF/ColumnReaderUtils.hxx"
#include "ROOT/RDF/RColumnReaderBase.hxx"
#include "ROOT/RDF/RDefineBase.hxx"
#include "ROOT/RDF/RLoopManager.hxx"
#include "ROOT/RDF/Utils.hxx"
#include <string_view>
#include "ROOT/TypeTraits.hxx"
#include "RtypesCore.h"

#include <deque>
#include <type_traits>
#include <utility> // std::index_sequence
#include <vector>

class TTreeReader;

namespace ROOT {
namespace Detail {
namespace RDF {

using namespace ROOT::TypeTraits;

/**
 * \brief The implementation of the DefaultValueFor transformation.
 *
 * The class takes in the default value provided by the user to fill-in missing
 * values of the input column. During the Update step, the class checks for the
 * presence of the value of the column at the current event. If that value is
 * missing, it will return the default value to requesting nodes of the graph.
 */
template <typename T>
class R__CLING_PTRCHECK(off) RDefaultValueFor final : public RDefineBase {
   using ColumnTypes_t = ROOT::TypeTraits::TypeList<T>;
   using TypeInd_t = std::make_index_sequence<ColumnTypes_t::list_size>;
   // Avoid instantiating vector<bool> as `operator[]` returns temporaries in that case. Use std::deque instead.
   using ValuesPerSlot_t = std::conditional_t<std::is_same<T, bool>::value, std::deque<T>, std::vector<T>>;

   T fDefaultValue;
   ValuesPerSlot_t fLastResults;
   // One column reader per slot
   std::vector<RColumnReaderBase *> fValues;

   /// Define objects corresponding to systematic variations other than nominal for this defined column.
   /// The map key is the full variation name, e.g. "pt:up".
   std::unordered_map<std::string, std::unique_ptr<RDefineBase>> fVariedDefines;

   T &GetValueOrDefault(unsigned int slot, Long64_t entry)
   {
      if (auto *value = fValues[slot]->template TryGet<T>(entry))
         return *value;
      else
         return fDefaultValue;
   };

public:
   RDefaultValueFor(std::string_view name, std::string_view type, const T &defaultValue,
                    const ROOT::RDF::ColumnNames_t &columns, const RDFInternal::RColumnRegister &colRegister,
                    RLoopManager &lm, const std::string &variationName = "nominal")
      : RDefineBase(name, type, colRegister, lm, columns, variationName),
        fDefaultValue(defaultValue),
        fLastResults(lm.GetNSlots() * RDFInternal::CacheLineStep<T>()),
        fValues(lm.GetNSlots())
   {
      fLoopManager->Register(this);
      // We suppress errors that TTreeReader prints regarding the missing branch
      fLoopManager->InsertSuppressErrorsForMissingBranch(fColumnNames[0]);
   }

   RDefaultValueFor(const RDefaultValueFor &) = delete;
   RDefaultValueFor &operator=(const RDefaultValueFor &) = delete;
   RDefaultValueFor(RDefaultValueFor &&) = delete;
   RDefaultValueFor &operator=(RDefaultValueFor &&) = delete;
   ~RDefaultValueFor()
   {
      fLoopManager->Deregister(this);
      fLoopManager->EraseSuppressErrorsForMissingBranch(fColumnNames[0]);
   }

   void InitSlot(TTreeReader *r, unsigned int slot) final
   {
      fValues[slot] =
         RDFInternal::GetColumnReader(slot, fColRegister.GetReader(slot, fColumnNames[0], fVariation, typeid(T)),
                                      *fLoopManager, r, fColumnNames[0], typeid(T));
      fLastCheckedEntry[slot * RDFInternal::CacheLineStep<Long64_t>()] = -1;
   }

   /// Return the (type-erased) address of the Define'd value for the given processing slot.
   void *GetValuePtr(unsigned int slot) final
   {
      return static_cast<void *>(&fLastResults[slot * RDFInternal::CacheLineStep<T>()]);
   }

   /// Update the value at the address returned by GetValuePtr with the content corresponding to the given entry
   void Update(unsigned int slot, Long64_t entry) final
   {
      if (entry != fLastCheckedEntry[slot * RDFInternal::CacheLineStep<Long64_t>()]) {
         // evaluate this define expression, cache the result
         fLastResults[slot * RDFInternal::CacheLineStep<T>()] = GetValueOrDefault(slot, entry);
         fLastCheckedEntry[slot * RDFInternal::CacheLineStep<Long64_t>()] = entry;
      }
   }

   void Update(unsigned int /*slot*/, const ROOT::RDF::RSampleInfo & /*id*/) final {}

   const std::type_info &GetTypeId() const final { return typeid(T); }

   /// Clean-up operations to be performed at the end of a task.
   void FinalizeSlot(unsigned int slot) final
   {
      fValues[slot] = nullptr;

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
            new RDefaultValueFor(fName, fType, fDefaultValue, fColumnNames, fColRegister, *fLoopManager, variation));
         fVariedDefines.insert({variation, std::move(variedDefine)});
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

} // namespace RDF
} // namespace Detail
} // namespace ROOT

#endif // ROOT_RDF_RDefaultValueFor
