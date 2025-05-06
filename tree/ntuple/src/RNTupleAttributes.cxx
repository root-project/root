/// \file RNTupleAttributes.cxx
/// \ingroup NTuple ROOT7
/// \author Giacomo Parolini <giacomo.parolini@cern.ch>
/// \date 2025-02-25
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RNTupleAttributes.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleFillContext.hxx>
#include <ROOT/RPageStorageFile.hxx>
#include <ROOT/RNTupleReader.hxx>

using namespace ROOT::Experimental::Internal::RNTupleAttributes;

static ROOT::RResult<void> ValidateAttributeModel(const ROOT::RNTupleModel &model)
{
   const auto &projFields = ROOT::Internal::GetProjectedFieldsOfModel(model);
   if (!projFields.IsEmpty())
      return R__FAIL("The Model used to create an AttributeSet cannot contain projected fields.");

   for (const auto &field : model.GetConstFieldZero()) {
      if (field.GetStructure() == ROOT::ENTupleStructure::kStreamer)
         return R__FAIL(std::string("The Model used to create an AttributeSet cannot contain Streamer field '") +
                        field.GetQualifiedFieldName() + "'");
   }
   return ROOT::RResult<void>::Success();
}

//
//  RNTupleAttributeSetWriter
//
ROOT::RResult<ROOT::Experimental::RNTupleAttributeSetWriter>
ROOT::Experimental::RNTupleAttributeSetWriter::Create(std::string_view name, std::unique_ptr<RNTupleModel> model,
                                                      const RNTupleFillContext *mainFillContext, TDirectory &dir)

{
   // Validate model
   if (auto modelValid = ValidateAttributeModel(*model); !modelValid)
      return R__FORWARD_ERROR(modelValid);

   // Add an internal EntryRange field to the model.
   // TODO: the entry range field name is not guaranteed to not be already taken!
   // TODO: do we need a bespoke field type?
   model->MakeField<REntryRange>(kEntryRangeFieldName);
   model->Freeze();

   // Create a sink that points to the same TDirectory as the main RNTuple
   auto opts = ROOT::RNTupleWriteOptions{};
   auto sink = std::make_unique<ROOT::Internal::RPageSinkFile>(name, dir, opts);
   RNTupleFillContext fillContext{std::move(model), std::move(sink)};
   return RNTupleAttributeSetWriter(mainFillContext, std::move(fillContext));
}

ROOT::Experimental::RNTupleAttributeSetWriter::RNTupleAttributeSetWriter(const RNTupleFillContext *mainFillContext,
                                                                         RNTupleFillContext fillContext)
   : fFillContext(std::move(fillContext)), fMainFillContext(mainFillContext)
{
}

const std::string &ROOT::Experimental::RNTupleAttributeSetWriter::GetName() const
{
   const auto &name = fFillContext.fSink->GetNTupleName();
   return name;
}

ROOT::Experimental::RNTupleAttributeRangeHandle ROOT::Experimental::RNTupleAttributeSetWriter::BeginRange()
{
   if (fOpenRange)
      throw ROOT::RException(R__FAIL("Called BeginRange() without having closed the currently open range!"));

   auto entry = fFillContext.GetModel().CreateEntry();
   const auto start = fMainFillContext->GetNEntries();
   fOpenRange = RNTupleAttributeRange(std::move(entry), start);
   auto handle = RNTupleAttributeRangeHandle{*fOpenRange};
   return handle;
}

void ROOT::Experimental::RNTupleAttributeSetWriter::EndRange(
   ROOT::Experimental::RNTupleAttributeRangeHandle rangeHandle)
{
   if (R__unlikely(!fOpenRange || &rangeHandle.fRange != &*fOpenRange))
      throw ROOT::RException(
         R__FAIL(std::string("Handle passed to EndRange() of Attribute Set \"") + GetName() +
                 "\" is invalid (it is not the Handle returned by the latest call to BeginRange())"));

   EndRangeInternal();
}

void ROOT::Experimental::RNTupleAttributeSetWriter::EndRangeInternal()
{
   // Get current entry number from the writer and use it as end of entry range
   const auto end = fMainFillContext->GetNEntries();
   auto &range = *fOpenRange;
   auto pRange = range.fEntry->GetPtr<REntryRange>(kEntryRangeFieldName);
   R__ASSERT(pRange);
   *pRange = {range.fStart, end};
   fFillContext.Fill(*range.fEntry);

   fOpenRange = std::nullopt;
}

void ROOT::Experimental::RNTupleAttributeSetWriter::Commit()
{
   if (fOpenRange)
      EndRangeInternal();
   fFillContext.FlushCluster();
   fFillContext.fSink->CommitClusterGroup();
   fFillContext.fSink->CommitDataset();
}

//
//  RNTupleAttributeSetReader
//
ROOT::Experimental::RNTupleAttributeSetReader::RNTupleAttributeSetReader(std::unique_ptr<RNTupleReader> reader)
   : fReader(std::move(reader))
{
   // Collect all entry ranges
   auto entryRangeView = fReader->GetView<REntryRange>(kEntryRangeFieldName);
   fEntryRanges.reserve(fReader->GetNEntries());
   for (auto i : fReader->GetEntryRange()) {
      auto range = entryRangeView(i);
      fEntryRanges.push_back({range, i});
   }

   assert(EntryRangesAreSorted(fEntryRanges));

   R__LOG_INFO(ROOT::Internal::NTupleLog()) << "Loaded " << fEntryRanges.size() << " attribute entries.";
}

const std::string &ROOT::Experimental::RNTupleAttributeSetReader::GetName() const
{
   const auto &name = fReader->GetDescriptor().GetName();
   return name;
}

// Entry ranges should be sorted with respect to Start by construction.
// TODO: make sure merging attributes preserves the sorting.
bool ROOT::Experimental::RNTupleAttributeSetReader::EntryRangesAreSorted(const decltype(fEntryRanges) &ranges)
{
   ROOT::NTupleSize_t prevEnd = 0;
   for (const auto &[range, _] : ranges) {
      const auto &[start, end] = range;
      if (start < prevEnd)
         return false;
      prevEnd = end;
   }
   return true;
};

std::vector<ROOT::Experimental::RNTupleAttributeRange>
ROOT::Experimental::RNTupleAttributeSetReader::GetAttributesRangeInternal(NTupleSize_t startEntry,
                                                                          NTupleSize_t endEntry, bool rangeIsContained)
{
   std::vector<RNTupleAttributeRange> result;

   if (endEntry < startEntry) {
      R__LOG_WARNING(ROOT::Internal::NTupleLog())
         << "end < start when getting attributes from Attribute Set '" << GetName() << "' (range given: [" << startEntry
         << ", " << endEntry << "].";
      return result;
   }

   assert(EntryRangesAreSorted(fEntryRanges));

   const auto FullyContained = [rangeIsContained](auto startEntry, auto endEntry, auto start, auto end) {
      if (rangeIsContained) {
         std::swap(start, startEntry);
         std::swap(end, endEntry);
      }
      return startEntry >= start && endEntry <= end;
   };

   // TODO: consider using binary search, since fEntryRanges is sorted
   // (maybe it should be done only if the size of the list is bigger than a threshold).
   for (std::uint32_t i = 0; i < fEntryRanges.size(); ++i) {
      const auto &[range, index] = fEntryRanges[i];
      const auto &[start, end] = range;
      if (start > endEntry)
         break; // We can break here because fEntryRanges is sorted.

      if (FullyContained(startEntry, endEntry, start, end)) {
         auto entry = fReader->CreateEntry();
         fReader->LoadEntry(index, *entry);
         result.push_back(RNTupleAttributeRange{std::move(entry), start, end});
      }
   }

   return result;
}

std::vector<ROOT::Experimental::RNTupleAttributeRange>
ROOT::Experimental::RNTupleAttributeSetReader::GetAttributesContainingRange(NTupleSize_t startEntry,
                                                                            NTupleSize_t endEntry)
{
   return GetAttributesRangeInternal(startEntry, endEntry, false);
}

std::vector<ROOT::Experimental::RNTupleAttributeRange>
ROOT::Experimental::RNTupleAttributeSetReader::GetAttributesInRange(NTupleSize_t startEntry, NTupleSize_t endEntry)
{
   return GetAttributesRangeInternal(startEntry, endEntry, true);
}

std::vector<ROOT::Experimental::RNTupleAttributeRange>
ROOT::Experimental::RNTupleAttributeSetReader::GetAttributes(NTupleSize_t entryIndex)
{
   return GetAttributesContainingRange(entryIndex, entryIndex);
}

std::vector<ROOT::Experimental::RNTupleAttributeRange> ROOT::Experimental::RNTupleAttributeSetReader::GetAttributes()
{
   std::vector<RNTupleAttributeRange> result;
   result.reserve(fEntryRanges.size());

   for (std::uint32_t i = 0; i < fEntryRanges.size(); ++i) {
      const auto &[range, index] = fEntryRanges[i];
      const auto &[entryStart, entryEnd] = range;
      auto entry = fReader->CreateEntry();
      fReader->LoadEntry(index, *entry);
      result.push_back(RNTupleAttributeRange{std::move(entry), entryStart, entryEnd});
   }

   return result;
}
