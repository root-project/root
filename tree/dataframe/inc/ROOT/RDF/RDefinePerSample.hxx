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

   // Avoid instantiating vector<bool> as `operator[]` returns temporaries in that case. Use std::deque instead.
   using ValuesPerSlot_t =
      std::conditional_t<std::is_same<RetType_t, bool>::value, std::deque<RetType_t>, std::vector<RetType_t>>;

   F fExpression;
   ValuesPerSlot_t fLastResults;

public:
   RDefinePerSample(std::string_view name, std::string_view type, F expression, RLoopManager &lm)
      : RDefineBase(name, type, RDFInternal::RColumnRegister{nullptr}, lm, /*columnNames*/ {}),
        fExpression(std::move(expression)), fLastResults(lm.GetNSlots() * RDFInternal::CacheLineStep<RetType_t>())
   {
   }

   RDefinePerSample(const RDefinePerSample &) = delete;
   RDefinePerSample &operator=(const RDefinePerSample &) = delete;

   /// Return the (type-erased) address of the Define'd value for the given processing slot.
   void *GetValuePtr(unsigned int slot) final
   {
      return static_cast<void *>(&fLastResults[slot * RDFInternal::CacheLineStep<RetType_t>()]);
   }

   void Update(unsigned int, Long64_t) final
   {
      // no-op
   }

   /// Update the value at the address returned by GetValuePtr with the content corresponding to the given entry
   void Update(unsigned int slot, const ROOT::RDF::RSampleInfo &id) final
   {
      fLastResults[slot * RDFInternal::CacheLineStep<RetType_t>()] = fExpression(slot, id);
   }

   const std::type_info &GetTypeId() const { return typeid(RetType_t); }

   void InitSlot(TTreeReader *, unsigned int) final {}

   void FinalizeSlot(unsigned int) final {}

   // No-op for RDefinePerSample: it never depends on systematic variations
   void MakeVariations(const std::vector<std::string> &) final {}

   RDefineBase &GetVariedDefine(const std::string &) final
   {
      R__ASSERT(false && "This should never be called");
      return *this;
   }
};

} // namespace RDF
} // namespace Detail
} // namespace ROOT

#endif // ROOT_RDF_RDEFINEPERSAMPLE
