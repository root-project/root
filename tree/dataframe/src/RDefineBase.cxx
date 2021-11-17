// Author: Enrico Guiraud, Danilo Piparo CERN  09/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDF/RDefineBase.hxx"
#include "ROOT/RDF/Utils.hxx"
#include "ROOT/RStringView.hxx"
#include "RtypesCore.h" // Long64_t

#include <string>
#include <vector>
#include <atomic>

using ROOT::Detail::RDF::RDefineBase;
namespace RDFInternal = ROOT::Internal::RDF; // redundant (already present in the header), but Windows needs it

RDefineBase::RDefineBase(std::string_view name, std::string_view type, unsigned int nSlots,
                         const RDFInternal::RBookedDefines &defines,
                         const std::map<std::string, std::vector<void *>> &DSValuePtrs, ROOT::RDF::RDataSource *ds,
                         const ROOT::RDF::ColumnNames_t &columnNames)
   : fName(name), fType(type), fLastCheckedEntry(nSlots * RDFInternal::CacheLineStep<Long64_t>(), -1),
     fDefines(defines), fIsInitialized(nSlots, false), fDSValuePtrs(DSValuePtrs), fDataSource(ds),
     fColumnNames(columnNames), fIsDefine(columnNames.size())
{
   const auto nColumns = fColumnNames.size();
   for (auto i = 0u; i < nColumns; ++i)
      fIsDefine[i] = fDefines.HasName(fColumnNames[i]);
}

// pin vtable. Work around cling JIT issue.
RDefineBase::~RDefineBase() {}

std::string RDefineBase::GetName() const
{
   return fName;
}

std::string RDefineBase::GetTypeName() const
{
   return fType;
}
