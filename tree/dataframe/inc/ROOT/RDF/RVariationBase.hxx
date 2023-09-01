// Author: Enrico Guiraud, CERN 10/2021

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RVARIATIONBASE
#define ROOT_RVARIATIONBASE

#include <ROOT/RDF/RColumnRegister.hxx>
#include <ROOT/RDF/Utils.hxx> // ColumnNames_t
#include <ROOT/RVec.hxx>

#include <array>
#include <deque>
#include <memory>
#include <string>
#include <vector>

class TTreeReader;

namespace ROOT {
namespace RDF {
class RDataSource;
}
namespace Detail {
namespace RDF {
class RLoopManager;
}
} // namespace Detail
namespace Internal {
namespace RDF {

/// This type includes all parts of RVariation that do not depend on the callable signature.
class RVariationBase {
protected:
   std::vector<std::string> fColNames;       ///< The names of the varied columns.
   std::vector<std::string> fVariationNames; ///< The names of the systematic variation.
   std::string fType;                        ///< The type of the custom column as a text string.
   std::vector<Long64_t> fLastCheckedEntry;
   RColumnRegister fColumnRegister;
   RLoopManager *fLoopManager;
   ColumnNames_t fInputColumns;
   /// The nth flag signals whether the nth input column is a custom column or not.
   ROOT::RVecB fIsDefine;

public:
   RVariationBase(const std::vector<std::string> &colNames, std::string_view variationName,
                  const std::vector<std::string> &variationTags, std::string_view type,
                  const RColumnRegister &colRegister, RLoopManager &lm, const ColumnNames_t &inputColNames);

   RVariationBase(const RVariationBase &) = delete;
   RVariationBase(RVariationBase &&) = default;
   RVariationBase &operator=(const RVariationBase &) = delete;
   RVariationBase &operator=(RVariationBase &&) = default;
   virtual ~RVariationBase();

   virtual void InitSlot(TTreeReader *r, unsigned int slot) = 0;

   /// Return the (type-erased) address of the value of one variation of one column (can be safely cast back to a T*).
   virtual void *GetValuePtr(unsigned int slot, const std::string &column, const std::string &variation) = 0;
   virtual const std::type_info &GetTypeId() const = 0;
   const std::vector<std::string> &GetColumnNames() const;
   const std::vector<std::string> &GetVariationNames() const;
   std::string GetTypeName() const;
   /// Update the value at the address returned by GetValuePtr with the content corresponding to the given entry
   virtual void Update(unsigned int slot, Long64_t entry) = 0;
   /// Clean-up operations to be performed at the end of a task.
   virtual void FinalizeSlot(unsigned int slot) = 0;
};

} // namespace RDF
} // namespace Internal
} // namespace ROOT

#endif // ROOT_RVARIATIONBASE
