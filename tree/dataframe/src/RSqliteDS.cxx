// Author: Jakob Blomer CERN  07/2018

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// clang-format off
/** \class ROOT::RDF::RSqliteDS
    \ingroup dataframe
    \brief RDataFrame data source class for reading SQlite files.
*/
// clang-format on

#include <ROOT/RSqliteDS.hxx>
#include <ROOT/RDFUtils.hxx>
#include <ROOT/RMakeUnique.hxx>
#include <TError.h>

namespace ROOT {

namespace RDF {

RSqliteDS::RSqliteDS(std::string_view fileName, std::string_view query)
{
}


RSqliteDS::~RSqliteDS()
{
}


const std::vector<std::string> &RSqliteDS::GetColumnNames() const
{
}


RDataSource::Record_t RSqliteDS::GetColumnReadersImpl(std::string_view name, const std::type_info &)
{
}


std::vector<std::pair<ULong64_t, ULong64_t>> RSqliteDS::GetEntryRanges()
{
}


std::string RSqliteDS::GetTypeName(std::string_view) const
{
}


bool RSqliteDS::HasColumn(std::string_view) const
{
}


RDataFrame MakeSqliteDataFrame(std::string_view fileName, std::string_view query)
{
   ROOT::RDataFrame tdf(std::make_unique<RSqliteDS>(fileName, query));
   return tdf;
}


bool RSqliteDS::SetEntry(unsigned int slot, ULong64_t entry)
{
}


void RSqliteDS::SetNSlots(unsigned int nSlots)
{
}

} // ns RDF

} // ns ROOT
