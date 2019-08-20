/// \file RNTupleDS.cxx
/// \ingroup NTuple ROOT7
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

#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleDS.hxx>
#include <ROOT/RStringView.hxx>

#include <TError.h>

#include <string>
#include <vector>
#include <typeinfo>
#include <utility>

namespace ROOT {
namespace Experimental {

ROOT::Experimental::RNTupleDS::RNTupleDS(std::unique_ptr<ROOT::Experimental::RNTupleReader> ntuple)
{
   fReaders.emplace_back(std::move(ntuple));
   auto rootField = fReaders[0]->GetModel()->GetRootField();
   for (auto &f : *rootField) {
      if (f.GetParent() != rootField)
         continue;
      fColumnNames.push_back(f.GetName());
      fColumnTypes.push_back(f.GetType());
   }
}

const std::vector<std::string>& RNTupleDS::GetColumnNames() const
{
   return fColumnNames;
}


RDF::RDataSource::Record_t RNTupleDS::GetColumnReadersImpl(std::string_view name, const std::type_info& /* ti */)
{
   const auto index = std::distance(
      fColumnNames.begin(), std::find(fColumnNames.begin(), fColumnNames.end(), name));
   // TODO(jblomer): check expected type info like in, e.g., RRootDS.cxx
   // There is a problem extracting the type info for std::int32_t and company though

   std::vector<void*> ptrs;
   for (unsigned i = 0; i < fNSlots; ++i)
      ptrs.push_back(&fValuePtrs[i][index]);

   return ptrs;
}

bool RNTupleDS::SetEntry(unsigned int slot, ULong64_t entryIndex)
{
   fReaders[slot]->LoadEntry(entryIndex, fEntries[slot].get());
   return true;
}

std::vector<std::pair<ULong64_t, ULong64_t>> RNTupleDS::GetEntryRanges()
{
   // TODO(jblomer): use cluster boundaries for the entry ranges
   std::vector<std::pair<ULong64_t, ULong64_t>> ranges;
   if (fHasSeenAllRanges) return ranges;

   auto nEntries = fReaders[0]->GetNEntries();
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


std::string RNTupleDS::GetTypeName(std::string_view colName) const
{
   const auto index = std::distance(
      fColumnNames.begin(), std::find(fColumnNames.begin(), fColumnNames.end(), colName));
   return fColumnTypes[index];
}


bool RNTupleDS::HasColumn(std::string_view colName) const
{
   return std::find(fColumnNames.begin(), fColumnNames.end(), colName) !=
          fColumnNames.end();
}


void RNTupleDS::Initialise()
{
   fHasSeenAllRanges = false;
}


void RNTupleDS::SetNSlots(unsigned int nSlots)
{
   R__ASSERT(fNSlots == 0);
   R__ASSERT(nSlots > 0);
   fNSlots = nSlots;

   for (unsigned int i = 1; i < fNSlots; ++i) {
      fReaders.emplace_back(fReaders[0]->Clone());
   }

   for (unsigned int i = 0; i < fNSlots; ++i) {
      auto entry = fReaders[i]->GetModel()->CreateEntry();
      fValuePtrs.emplace_back(std::vector<void*>());
      for (unsigned j = 0; j < fColumnNames.size(); ++j) {
         fValuePtrs[i].emplace_back(entry->GetValue(fColumnNames[j]).GetRawPtr());
      }
      fEntries.emplace_back(std::move(entry));
   }
}


RDataFrame MakeNTupleDataFrame(std::string_view ntupleName, std::string_view fileName)
{
   auto ntuple = RNTupleReader::Open(ntupleName, fileName);
   ROOT::RDataFrame rdf(std::make_unique<RNTupleDS>(std::move(ntuple)));
   return rdf;
}

} // ns Experimental
} // ns ROOT
