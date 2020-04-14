// Author: Enrico Guiraud, Danilo Piparo CERN  09/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RJITTEDCUSTOMCOLUMN
#define ROOT_RJITTEDCUSTOMCOLUMN

#include "ROOT/RDF/RCustomColumnBase.hxx"
#include "ROOT/RStringView.hxx"
#include "RtypesCore.h"

#include <memory>
#include <type_traits>

class TTreeReader;

namespace ROOT {
namespace Detail {
namespace RDF {

class RLoopManager;

/// A wrapper around a concrete RCustomColumn, which forwards all calls to it
/// RJittedCustomColumn is a placeholder that is put in the collection of custom columns in place of a RCustomColumn
/// that will be just-in-time compiled. Jitted code will assign the concrete RCustomColumn to this RJittedCustomColumn
/// before the event-loop starts.
class RJittedCustomColumn : public RCustomColumnBase {
   std::unique_ptr<RCustomColumnBase> fConcreteCustomColumn = nullptr;

public:
   RJittedCustomColumn(RLoopManager *lm, std::string_view name, std::string_view type, unsigned int nSlots)
      : RCustomColumnBase(lm, name, type, nSlots, /*isDSColumn=*/false, RDFInternal::RBookedCustomColumns())
   {
   }

   void SetCustomColumn(std::unique_ptr<RCustomColumnBase> c) { fConcreteCustomColumn = std::move(c); }

   void InitSlot(TTreeReader *r, unsigned int slot) final;
   void *GetValuePtr(unsigned int slot) final;
   const std::type_info &GetTypeId() const final;
   void Update(unsigned int slot, Long64_t entry) final;
   void ClearValueReaders(unsigned int slot) final;
   void InitNode() final;
};

} // ns RDF
} // ns Detail
} // ns ROOT

#endif // ROOT_RJITTEDCUSTOMCOLUMN
