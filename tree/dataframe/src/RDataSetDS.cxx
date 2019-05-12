/// \file RDataSetDS.cxx
/// \ingroup DataSet ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-04
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RDataSet.hxx>
#include <ROOT/RDataSetDS.hxx>
#include <ROOT/RStringView.hxx>

#include <TError.h>

#include <string>
#include <vector>
#include <typeinfo>
#include <utility>

namespace ROOT {
namespace Experimental {

RDataSetDS::RDataSetDS(std::unique_ptr<ROOT::Experimental::RInputForest> forest)
  : fForest(std::move(forest)), fEntry(fForest->GetModel()->CreateEntry()), fNSlots(1), fHasSeenAllRanges(false)
{
   auto rootField = fForest->GetModel()->GetRootField();
   for (auto& f : *rootField) {
      if (f.GetParent() != rootField)
         continue;
      fColumnNames.push_back(f.GetName());
      fColumnTypes.push_back(f.GetType());
      fValuePtrs.push_back(fEntry->GetValue(f.GetName()).GetRawPtr());
   }
}


RDataSetDS::~RDataSetDS()
{
}


const std::vector<std::string>& RDataSetDS::GetColumnNames() const
{
   return fColumnNames;
}


RDF::RDataSource::Record_t RDataSetDS::GetColumnReadersImpl(std::string_view name, const std::type_info& /* ti */)
{
   const auto index = std::distance(
      fColumnNames.begin(), std::find(fColumnNames.begin(), fColumnNames.end(), name));
   // TODO(jblomer): check expected type info like in, e.g., RRootDS.cxx
   // There is a problem extracting the type info for std::int32_t and company though

   std::vector<void*> ptrs;
   R__ASSERT(fNSlots == 1);
   ptrs.push_back(&fValuePtrs[index]);

   return ptrs;
}

bool RDataSetDS::SetEntry(unsigned int /*slot*/, ULong64_t entryIndex) {
   fForest->LoadEntry(entryIndex, fEntry.get());
   return true;
}

std::vector<std::pair<ULong64_t, ULong64_t>> RDataSetDS::GetEntryRanges()
{
   std::vector<std::pair<ULong64_t, ULong64_t>> ranges;
   if (fHasSeenAllRanges) return ranges;

   auto nEntries = fForest->GetNEntries();
   const auto chunkSize = nEntries / fNSlots;
   const auto reminder = 1U == fNSlots ? 0 : nEntries % fNSlots;
   auto start = 0UL;
   auto end = 0UL;
   for (auto i : ROOT::TSeqU(fNSlots)) {
      start = end;
      end += chunkSize;
      ranges.emplace_back(start, end);
      (void)i;
   }
   ranges.back().second += reminder;
   fHasSeenAllRanges = true;
   return ranges;
}


std::string RDataSetDS::GetTypeName(std::string_view colName) const
{
   const auto index = std::distance(
      fColumnNames.begin(), std::find(fColumnNames.begin(), fColumnNames.end(), colName));
   return fColumnTypes[index];
}


bool RDataSetDS::HasColumn(std::string_view colName) const
{
   return std::find(fColumnNames.begin(), fColumnNames.end(), colName) !=
          fColumnNames.end();
}


void RDataSetDS::Initialise()
{
   fHasSeenAllRanges = false;
}


void RDataSetDS::SetNSlots(unsigned int nSlots)
{
   fNSlots = nSlots;
}


RDataFrame MakeDataSetDataFrame(std::string_view forestName, std::string_view fileName) {
   auto forest = RInputForest::Open(forestName, fileName);
   ROOT::RDataFrame rdf(std::make_unique<RDataSetDS>(std::move(forest)));
   return rdf;
}

} // ns Experimental
} // ns ROOT
