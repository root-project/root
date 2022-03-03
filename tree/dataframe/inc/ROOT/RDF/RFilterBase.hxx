// Author: Enrico Guiraud, Danilo Piparo CERN  09/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RFILTERBASE
#define ROOT_RFILTERBASE

#include "ROOT/RDF/RColumnRegister.hxx"
#include "ROOT/RDF/RNodeBase.hxx"
#include "ROOT/RDF/Utils.hxx" // ColumnNames_t
#include "ROOT/RVec.hxx"
#include "RtypesCore.h"

#include <cassert>
#include <string>
#include <vector>

class TTreeReader;

namespace ROOT {

namespace RDF {
class RCutFlowReport;
} // ns RDF

namespace Detail {
namespace RDF {
namespace RDFInternal = ROOT::Internal::RDF;

class RLoopManager;

class RFilterBase : public RNodeBase {
protected:
   std::vector<Long64_t> fLastCheckedEntry;
   std::vector<int> fLastResult = {true}; // std::vector<bool> cannot be used in a MT context safely
   std::vector<ULong64_t> fAccepted = {0};
   std::vector<ULong64_t> fRejected = {0};
   const std::string fName;
   const ROOT::RDF::ColumnNames_t fColumnNames;
   RDFInternal::RColumnRegister fColRegister;
   /// The nth flag signals whether the nth input column is a custom column or not.
   ROOT::RVecB fIsDefine;
   std::string fVariation; ///< This indicates for what variation this filter evaluates values.
   std::unordered_map<std::string, std::shared_ptr<RFilterBase>> fVariedFilters;

public:
   RFilterBase(RLoopManager *df, std::string_view name, const unsigned int nSlots,
               const RDFInternal::RColumnRegister &colRegister, const ColumnNames_t &columns,
               const std::vector<std::string> &prevVariations, const std::string &variation = "nominal");
   RFilterBase &operator=(const RFilterBase &) = delete;

   virtual ~RFilterBase();

   virtual void InitSlot(TTreeReader *r, unsigned int slot) = 0;
   bool HasName() const;
   std::string GetName() const;
   virtual void FillReport(ROOT::RDF::RCutFlowReport &) const;
   virtual void TriggerChildrenCount() = 0;
   virtual void ResetReportCount()
   {
      assert(!fName.empty()); // this method is to only be called on named filters
      // fAccepted and fRejected could be different than 0 if this is not the first event-loop run using this filter
      std::fill(fAccepted.begin(), fAccepted.end(), 0);
      std::fill(fRejected.begin(), fRejected.end(), 0);
   }
   /// Clean-up operations to be performed at the end of a task.
   virtual void FinaliseSlot(unsigned int slot) = 0;
   virtual void InitNode();
   virtual void AddFilterName(std::vector<std::string> &filters) = 0;
};

} // ns RDF
} // ns Detail
} // ns ROOT

#endif // ROOT_RFILTERBASE
