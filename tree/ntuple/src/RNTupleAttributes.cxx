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
ROOT::RResult<std::unique_ptr<ROOT::Experimental::RNTupleAttributeSetWriter>>
ROOT::Experimental::RNTupleAttributeSetWriter::Create(std::string_view name, std::unique_ptr<RNTupleModel> model,
                                                      const RNTupleFillContext *mainFillContext, TDirectory &dir)

{
   // Validate model
   if (auto modelValid = ValidateAttributeModel(*model); !modelValid)
      return R__FORWARD_ERROR(modelValid);

   model->Unfreeze();

   // Add all fields of `model` as subfields of a single top-level untyped record field which has the same
   // name as the attribute set. This way we "namespace" all user-defined attribute fields and we are free to use
   // whichever name we want for our internal fields.
   // TODO: avoid cloning the fields
   std::vector<std::unique_ptr<RFieldBase>> fields;
   fields.reserve(model->GetFieldNames().size());
   for (const auto &fieldName : model->GetFieldNames()) {
      auto &field = model->GetMutableField(fieldName);
      fields.push_back(field.Clone(fieldName));
   }
   // TODO: evaluate if this is really needed
   auto userRootField = std::make_unique<ROOT::RRecordField>(name, std::move(fields));

   // TODO: avoid creating a new model
   auto newModel = RNTupleModel::CreateBare();
   newModel->MakeField<ROOT::NTupleSize_t>("__rangeStart");
   newModel->MakeField<ROOT::NTupleSize_t>("__rangeLen");
   newModel->AddField(std::move(userRootField));
   newModel->Freeze();

   // Create a sink that points to the same TDirectory as the main RNTuple
   auto opts = ROOT::RNTupleWriteOptions{};
   opts.SetCompression(mainFillContext->fSink->GetWriteOptions().GetCompression());
   auto sink = std::make_unique<ROOT::Internal::RPageSinkFile>(name, dir, opts);
   RNTupleFillContext fillContext{std::move(newModel), std::move(sink)};
   return std::unique_ptr<RNTupleAttributeSetWriter>(
      new RNTupleAttributeSetWriter(mainFillContext, std::move(fillContext)));
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

   const auto start = fMainFillContext->GetNEntries();
   auto &model = const_cast<RNTupleModel &>(fFillContext.GetModel());
   auto [entry, scopedEntry] = RNTupleAttributeEntry::CreateInternalEntries(model);
   fOpenEntry = RNTupleAttributeEntry{std::move(entry), std::move(scopedEntry), start};
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
   auto pRangeStart = fOpenEntry->fEntry->GetPtr<ROOT::NTupleSize_t>("__rangeStart");
   auto pRangeLen = fOpenEntry->fEntry->GetPtr<ROOT::NTupleSize_t>("__rangeLen");
   R__ASSERT(pRangeStart);
   R__ASSERT(pRangeLen);
   *pRangeStart = fOpenEntry->GetRange().Start();
   *pRangeLen = fOpenEntry->GetRange().Length();
   fFillContext.FillImpl(*fOpenEntry);

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
   auto entryRangeStartView = fReader->GetView<ROOT::NTupleSize_t>("__rangeStart");
   auto entryRangeLenView = fReader->GetView<ROOT::NTupleSize_t>("__rangeLen");
   fEntryRanges.reserve(fReader->GetNEntries());
   for (auto i : fReader->GetEntryRange()) {
      auto start = entryRangeStartView(i);
      auto len = entryRangeLenView(i);
      fEntryRanges.push_back({RNTupleAttributeRange::FromStartLength(start, len), i});
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
   ROOT::NTupleSize_t prevStart = 0;
   for (const auto &[range, _] : ranges) {
      if (range.Start() < prevStart)
         return false;
      prevStart = range.Start();
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

   auto &model = const_cast<RNTupleModel &>(fReader->GetModel());
   // TODO: consider using binary search, since fEntryRanges is sorted
   // (maybe it should be done only if the size of the list is bigger than a threshold).
   for (const auto &[range, index] : fEntryRanges) {
      const auto &firstLast = range.GetFirstLast();
      if (!firstLast)
         continue;

      const auto &[first, last] = *firstLast;
      if (first >= endEntry)
         break; // We can break here because fEntryRanges is sorted.

      if (FullyContained(startEntry, endEntry, first, last + 1)) {
         auto scopedEntry = RNTupleAttributeEntry::CreateScopedEntry(model);
         auto attrEntry = RNTupleAttributeEntry(nullptr, std::move(scopedEntry), range);
         fReader->LoadEntry(index, *attrEntry.fScopedEntry);
         result.push_back(std::move(attrEntry));
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
   return GetAttributesContainingRange(entryIndex, entryIndex + 1);
}

std::vector<ROOT::Experimental::RNTupleAttributeEntry> ROOT::Experimental::RNTupleAttributeSetReader::GetAttributes()
{
   std::vector<RNTupleAttributeEntry> result;
   result.reserve(fEntryRanges.size());

   auto &model = const_cast<RNTupleModel &>(fReader->GetModel());
   for (const auto &[range, index] : fEntryRanges) {
      auto scopedEntry = RNTupleAttributeEntry::CreateScopedEntry(model);
      auto attrEntry = RNTupleAttributeEntry(nullptr, std::move(scopedEntry), range);
      fReader->LoadEntry(index, *attrEntry.fScopedEntry);
      result.push_back(std::move(attrEntry));
   }

   return result;
}
