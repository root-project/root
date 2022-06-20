// Author: Enrico Guiraud, Danilo Piparo CERN  09/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDF/RCutFlowReport.hxx"
#include "ROOT/RDF/RDefineBase.hxx"
#include "ROOT/RDF/RFilterBase.hxx"
#include "ROOT/RDF/Utils.hxx"
#include <numeric> // std::accumulate

using namespace ROOT::Detail::RDF;

RFilterBase::RFilterBase(RLoopManager *implPtr, std::string_view name, const unsigned int nSlots,
                         const RDFInternal::RColumnRegister &colRegister, const ColumnNames_t &columns,
                         const std::vector<std::string> &prevVariations, const std::string &variation)
   : RNodeBase(ROOT::Internal::RDF::Union(colRegister.GetVariationDeps(columns), prevVariations), implPtr),
     fLastCheckedEntry(std::vector<Long64_t>(nSlots * RDFInternal::CacheLineStep<Long64_t>(), -1)),
     fLastResult(nSlots * RDFInternal::CacheLineStep<int>()),
     fAccepted(nSlots * RDFInternal::CacheLineStep<ULong64_t>()),
     fRejected(nSlots * RDFInternal::CacheLineStep<ULong64_t>()), fName(name), fColumnNames(columns),
     fColRegister(colRegister), fIsDefine(columns.size()), fVariation(variation)
{
   const auto nColumns = fColumnNames.size();
   for (auto i = 0u; i < nColumns; ++i) {
      fIsDefine[i] = fColRegister.IsDefineOrAlias(fColumnNames[i]);
      if (fVariation != "nominal" && fIsDefine[i])
         fColRegister.GetColumns().at(fColumnNames[i])->MakeVariations({fVariation});
   }
}

// outlined to pin virtual table
RFilterBase::~RFilterBase() {}

bool RFilterBase::HasName() const
{
   return !fName.empty();
}

std::string RFilterBase::GetName() const
{
   return fName;
}

void RFilterBase::FillReport(ROOT::RDF::RCutFlowReport &rep) const
{
   if (fName.empty()) // FillReport is no-op for unnamed filters
      return;
   const auto accepted = std::accumulate(fAccepted.begin(), fAccepted.end(), 0ULL);
   const auto all = accepted + std::accumulate(fRejected.begin(), fRejected.end(), 0ULL);
   rep.AddCut({fName, accepted, all});
}

void RFilterBase::InitNode()
{
   if (!fName.empty()) // if this is a named filter we care about its report count
      ResetReportCount();
}
