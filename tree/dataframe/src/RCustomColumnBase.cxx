// Author: Enrico Guiraud, Danilo Piparo CERN  09/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDF/RCustomColumnBase.hxx"
#include "ROOT/RStringView.hxx"
#include "RtypesCore.h" // Long64_t

#include <string>
#include <vector>

using ROOT::Detail::RDF::RCustomColumnBase;
namespace RDFInternal = ROOT::Internal::RDF;

unsigned int RCustomColumnBase::GetNextID()
{
   static unsigned int id = 0U;
   ++id;
   return id;
}

RCustomColumnBase::RCustomColumnBase(std::string_view name, std::string_view type, unsigned int nSlots, bool isDSColumn,
                                     const RDFInternal::RBookedCustomColumns &customColumns,
                                     const std::map<std::string, std::vector<void *>> &DSValuePtrs)
   : fName(name), fType(type), fNSlots(nSlots), fIsDataSourceColumn(isDSColumn), fLastCheckedEntry(fNSlots, -1),
     fCustomColumns(customColumns), fIsInitialized(nSlots, false), fDSValuePtrs(DSValuePtrs)
{
}

// pin vtable. Work around cling JIT issue.
RCustomColumnBase::~RCustomColumnBase() {}

std::string RCustomColumnBase::GetName() const
{
   return fName;
}

std::string RCustomColumnBase::GetTypeName() const
{
   return fType;
}
