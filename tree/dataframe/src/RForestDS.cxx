/// \file RForestDS.cxx
/// \ingroup Forest ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-04
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RForest.hxx>
#include <ROOT/RForestDS.hxx>
#include <ROOT/RStringView.hxx>

#include <string>
#include <vector>
#include <typeinfo>
#include <utility>

namespace ROOT {

namespace RDF {

RForestDS::RForestDS(ROOT::Experimental::RInputForest* /*forest*/)
{
}


RForestDS::~RForestDS()
{
}


const std::vector<std::string>& RForestDS::GetColumnNames() const
{
   return std::vector<std::string>();
}


RDataSource::Record_t RForestDS::GetColumnReadersImpl(std::string_view /*name*/, const std::type_info& /* ti */)
{
}

bool RForestDS::SetEntry(unsigned int /*slot*/, ULong64_t /*entry*/) {
   return false;
}

std::vector<std::pair<ULong64_t, ULong64_t>> RForestDS::GetEntryRanges()
{
   return std::vector<std::pair<ULong64_t, ULong64_t>>();
}


std::string RForestDS::GetTypeName(std::string_view /*colName*/) const
{
   return "";
}


bool RForestDS::HasColumn(std::string_view /*colName*/) const
{
   return false;
}


void RForestDS::Initialise() {
}


void RForestDS::SetNSlots(unsigned int /*nSlots*/)
{
}


} // ns RDF

} // ns ROOT
