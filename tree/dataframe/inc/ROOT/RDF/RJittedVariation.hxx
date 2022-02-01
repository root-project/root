// Author: Enrico Guiraud, CERN  02/2021

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RJITTEDVARIATION
#define ROOT_RJITTEDVARIATION

#include "ROOT/RDF/RVariationBase.hxx"
#include "ROOT/RStringView.hxx"

#include <memory>

class TTreeReader;

namespace ROOT {
namespace Internal {
namespace RDF {

/// A wrapper around a concrete RVariation, which forwards all calls to it
/// RJittedVariation is a placeholder that is inserted in the computation graph in place of a RVariation
/// that will be just-in-time compiled. Jitted code will assign the concrete RVariation to this RJittedVariation
/// before the event-loop starts.
class RJittedVariation : public RVariationBase {
   std::unique_ptr<RVariationBase> fConcreteVariation = nullptr;

public:
   RJittedVariation(const std::vector<std::string> &colNames, std::string_view variationName,
                    const std::vector<std::string> &variationTags, std::string_view type,
                    const RColumnRegister &colRegister, RLoopManager &lm, const ColumnNames_t &inputColNames)
      : RVariationBase(colNames, variationName, variationTags, type, colRegister, lm, inputColNames)
   {
   }
   ~RJittedVariation();

   void SetVariation(std::unique_ptr<RVariationBase> c) { fConcreteVariation = std::move(c); }

   void InitSlot(TTreeReader *r, unsigned int slot) final;
   void *GetValuePtr(unsigned int slot, const std::string &column, const std::string &variation) final;
   const std::type_info &GetTypeId() const final;
   void Update(unsigned int slot, Long64_t entry) final;
   void FinalizeSlot(unsigned int slot) final;
};

} // namespace RDF
} // namespace Internal
} // namespace ROOT

#endif // ROOT_RJITTEDVARIATION
