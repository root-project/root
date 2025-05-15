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

namespace {
class RAttributeRangeField : public ROOT::RRecordField {
public:
   RAttributeRangeField(std::string_view fieldName)
      // TODO: we could use an untyped record to avoid storing the StreamerInfo of RNTupleAttributeRange in the TFile;
      // however this requires using some unsafe API (e.g. GetView<void>) and it's probably not worth it.
      // Investigate if there are better solutions that achieve the same goal without forcing the use of the untyped
      // unsafe APIs.
      : RRecordField(fieldName, "ROOT::Experimental::RNTupleAttributeRange")
   {
      std::vector<std::unique_ptr<RFieldBase>> fields;
      fields.emplace_back(new ROOT::RField<ROOT::NTupleSize_t>("_0"));
      fields.emplace_back(new ROOT::RField<ROOT::NTupleSize_t>("_1"));
      AttachItemFields(std::move(fields));
      fOffsets = {0, 8};
   }
};
} // namespace

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

   // Add an internal range field to the model.
   // TODO: the entry range field name is not guaranteed to not be already taken!
   // XXX: using a bespoke Field to avoid the overhead of RClassField. Worth it or unneeded extra complexity?
   model->MakeField<RNTupleAttributeRange>(kEntryRangeFieldName);
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

ROOT::Experimental::RNTupleAttributeEntryHandle ROOT::Experimental::RNTupleAttributeSetWriter::BeginRange()
{
   if (fOpenEntry)
      throw ROOT::RException(R__FAIL("Called BeginRange() without having closed the currently open range!"));

   auto entry = fFillContext.GetModel().CreateEntry();
   const auto start = fMainFillContext->GetNEntries();
   fOpenEntry = RNTupleAttributeEntry(std::move(entry), start);
   auto handle = RNTupleAttributeEntryHandle{*fOpenEntry};
   return handle;
}

void ROOT::Experimental::RNTupleAttributeSetWriter::EndRange(
   ROOT::Experimental::RNTupleAttributeEntryHandle rangeHandle)
{
   if (R__unlikely(!fOpenEntry || rangeHandle.fRange != &*fOpenEntry))
      throw ROOT::RException(
         R__FAIL(std::string("Handle passed to EndRange() of Attribute Set \"") + GetName() +
                 "\" is invalid (it is not the Handle returned by the latest call to BeginRange())"));

   EndRangeInternal();
}

void ROOT::Experimental::RNTupleAttributeSetWriter::EndRangeInternal()
{
   // Get current entry number from the writer and use it as end of entry range
   const auto end = fMainFillContext->GetNEntries();
   fOpenEntry->fRange = RNTupleAttributeRange::FromStartEnd(fOpenEntry->fRange.Start(), end);
   auto pRange = fOpenEntry->fEntry->GetPtr<RNTupleAttributeRange>(kEntryRangeFieldName);
   R__ASSERT(pRange);
   *pRange = fOpenEntry->GetRange();
   fFillContext.Fill(*fOpenEntry->fEntry);

   fOpenEntry = std::nullopt;
}

void ROOT::Experimental::RNTupleAttributeSetWriter::Commit()
{
   if (fOpenEntry)
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
   auto entryRangeView = fReader->GetView<RNTupleAttributeRange>(kEntryRangeFieldName);
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
      const auto &[start, end] = range.GetStartEnd();
      if (start < prevEnd)
         return false;
      prevEnd = end;
   }
   return true;
};

std::vector<ROOT::Experimental::RNTupleAttributeEntry>
ROOT::Experimental::RNTupleAttributeSetReader::GetAttributesRangeInternal(NTupleSize_t startEntry,
                                                                          NTupleSize_t endEntry, bool rangeIsContained)
{
   std::vector<RNTupleAttributeEntry> result;

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
   for (const auto &[range, index] : fEntryRanges) {
      const auto &[start, end] = range.GetStartEnd();
      if (start > endEntry)
         break; // We can break here because fEntryRanges is sorted.

      if (FullyContained(startEntry, endEntry, start, end)) {
         auto entry = fReader->CreateEntry();
         fReader->LoadEntry(index, *entry);
         result.push_back(RNTupleAttributeEntry{std::move(entry), range});
      }
   }

   return result;
}

std::vector<ROOT::Experimental::RNTupleAttributeEntry>
ROOT::Experimental::RNTupleAttributeSetReader::GetAttributesContainingRange(NTupleSize_t startEntry,
                                                                            NTupleSize_t endEntry)
{
   return GetAttributesRangeInternal(startEntry, endEntry, false);
}

std::vector<ROOT::Experimental::RNTupleAttributeEntry>
ROOT::Experimental::RNTupleAttributeSetReader::GetAttributesInRange(NTupleSize_t startEntry, NTupleSize_t endEntry)
{
   return GetAttributesRangeInternal(startEntry, endEntry, true);
}

std::vector<ROOT::Experimental::RNTupleAttributeEntry>
ROOT::Experimental::RNTupleAttributeSetReader::GetAttributes(NTupleSize_t entryIndex)
{
   return GetAttributesContainingRange(entryIndex, entryIndex);
}

std::vector<ROOT::Experimental::RNTupleAttributeEntry> ROOT::Experimental::RNTupleAttributeSetReader::GetAttributes()
{
   std::vector<RNTupleAttributeEntry> result;
   result.reserve(fEntryRanges.size());

   for (const auto &[range, index] : fEntryRanges) {
      auto entry = fReader->CreateEntry();
      fReader->LoadEntry(index, *entry);
      result.push_back(RNTupleAttributeEntry{std::move(entry), range});
   }

   return result;
}
