// Author: Enrico Guiraud, CERN 10/2021

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RDF/RLoopManager.hxx>
#include <ROOT/RDF/RVariationBase.hxx>
#include <ROOT/RDF/Utils.hxx> // CacheLineStep

namespace ROOT {
namespace Internal {
namespace RDF {

RVariationBase::RVariationBase(const std::vector<std::string> &colNames, std::string_view variationName,
                               const std::vector<std::string> &variationTags, std::string_view type,
                               const RColumnRegister &colRegister, RLoopManager &lm, const ColumnNames_t &inputColNames)
   : fColNames(colNames), fVariationNames(variationTags), fType(type),
     fLastCheckedEntry(lm.GetNSlots() * CacheLineStep<Long64_t>(), -1), fColumnRegister(colRegister), fLoopManager(&lm),
     fInputColumns(inputColNames), fIsDefine(inputColNames.size())
{
   // prepend the variation name to each tag
   for (auto &tag : fVariationNames)
      tag = std::string(variationName) + ':' + tag;

   const auto nColumns = fInputColumns.size();
   for (auto i = 0u; i < nColumns; ++i)
      fIsDefine[i] = fColumnRegister.HasName(fInputColumns[i]);
}

RVariationBase::~RVariationBase()
{
   fLoopManager->Deregister(this);
}

const std::vector<std::string> &RVariationBase::GetColumnNames() const
{
   return fColNames;
}

const std::vector<std::string> &RVariationBase::GetVariationNames() const
{
   return fVariationNames;
}

std::string RVariationBase::GetTypeName() const
{
   return fType;
}

} // namespace RDF
} // namespace Internal
} // namespace ROOT
