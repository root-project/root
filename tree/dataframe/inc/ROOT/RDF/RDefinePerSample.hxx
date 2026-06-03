// Author: Enrico Guiraud, CERN  08/2021

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_RDEFINEPERSAMPLE
#define ROOT_RDF_RDEFINEPERSAMPLE

#include "ROOT/RDF/RLoopManager.hxx"
#include "ROOT/RDF/RSampleInfo.hxx"
#include "ROOT/RDF/Utils.hxx"
#include <ROOT/RDF/RDefineBase.hxx>
#include <ROOT/TypeTraits.hxx>

#include <deque>
#include <vector>

namespace ROOT {
namespace Detail {
namespace RDF {

using namespace ROOT::TypeTraits;

template <typename F>
class R__CLING_PTRCHECK(off) RDefinePerSample final : public RDefineBase {
   using RetType_t = typename CallableTraits<F>::ret_type;

   using ValuesPerSlot_t = std::vector<ROOT::RVec<RetType_t>>;

   F fExpression;
   // Each slot accesses a cache of values for the current bulk
   ValuesPerSlot_t fCachedResultsPerSlot;

public:
   RDefinePerSample(std::string_view name, std::string_view type, F expression, RLoopManager &lm)
      : RDefineBase(name, type, RDFInternal::RColumnRegister{&lm}, lm, /*columnNames*/ {}),
        fExpression(std::move(expression)),
        fCachedResultsPerSlot(lm.GetNSlots() * RDFInternal::CacheLineStep<ROOT::RVec<RetType_t>>())
   {
      fLoopManager->Register(this);
      auto callUpdate = [this](unsigned int slot, const ROOT::RDF::RSampleInfo &id) { this->Update(slot, id); };
      fLoopManager->AddSampleCallback(this, std::move(callUpdate));
      // Assume 1-size bulk for now
      for (decltype(lm.GetNSlots()) i = 0; i < lm.GetNSlots(); ++i) {
         fCachedResultsPerSlot[i * RDFInternal::CacheLineStep<ROOT::RVec<RetType_t>>()].resize(1ul);
      }
   }

   RDefinePerSample(const RDefinePerSample &) = delete;
   RDefinePerSample &operator=(const RDefinePerSample &) = delete;

   ~RDefinePerSample() { fLoopManager->Deregister(this); }

   /// Return the beginning of the cached results of the current bulk for the input processing slot
   void *GetValuePtr(unsigned int slot) final
   {
      return static_cast<void *>(
         fCachedResultsPerSlot[slot * RDFInternal::CacheLineStep<ROOT::RVec<RetType_t>>()].data());
   }

   void Update(unsigned int, const ROOT::Internal::RDF::RMaskedEntryRange &) final
   {
      // no-op
   }

   /// Update the value at the address returned by GetValuePtr with the content corresponding to the given entry
   void Update(unsigned int slot, const ROOT::RDF::RSampleInfo &id) final
   {
      // Assume 1-size bulk for now
      fCachedResultsPerSlot[slot * RDFInternal::CacheLineStep<ROOT::RVec<RetType_t>>()][0] = fExpression(slot, id);
   }

   const std::type_info &GetTypeId() const final { return typeid(RetType_t); }

   void InitSlot(TTreeReader *, unsigned int) final {}

   void FinalizeSlot(unsigned int) final {}

   // No-op for RDefinePerSample: it never depends on systematic variations
   void MakeVariations(const std::vector<std::string> &) final {}

   RDefineBase &GetVariedDefine(const std::string &) final
   {
      // RDefinePerSample cannot depend on varied columns, so we return itself.
      // This supports the use case of a downstream defined variable that depends on variations and also on a column
      // created via DefinePerSample. The request for an action depending on that defined variable will end up here when
      // looking for a variation of the dependant column.
      return *this;
   }
};

} // namespace RDF
} // namespace Detail
} // namespace ROOT

#endif // ROOT_RDF_RDEFINEPERSAMPLE
