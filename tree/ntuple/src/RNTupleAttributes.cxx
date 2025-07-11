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

bool ROOT::Experimental::IsReservedRNTupleAttrSetName(std::string_view name)
{
   return name == "ROOT" || (name.length() > 4 && strncmp(name.data(), "ROOT.", 5) == 0);
}

//
//  RNTupleAttrSetWriter
//
std::unique_ptr<ROOT::Experimental::RNTupleAttrSetWriter>
ROOT::Experimental::RNTupleAttrSetWriter::Create(std::string_view name, std::unique_ptr<RNTupleModel> model,
                                                 const RNTupleFillContext *mainFillContext, TDirectory &dir)

{
   ValidateAttributeModel(*model).ThrowOnError();

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
   newModel->SetDescription(model->GetDescription());
   newModel->MakeField<ROOT::NTupleSize_t>(kRangeStartName);
   newModel->MakeField<ROOT::NTupleSize_t>(kRangeLenName);
   newModel->AddField(std::move(userRootField));
   newModel->Freeze();

   // Create a sink that points to the same TDirectory as the main RNTuple
   auto opts = ROOT::RNTupleWriteOptions{};
   opts.SetCompression(mainFillContext->fSink->GetWriteOptions().GetCompression());
   auto sink = std::make_unique<ROOT::Internal::RPageSinkFile>(name, dir, opts);
   RNTupleFillContext fillContext{std::move(newModel), std::move(sink)};
   return std::unique_ptr<RNTupleAttrSetWriter>(new RNTupleAttrSetWriter(mainFillContext, std::move(fillContext)));
}

ROOT::Experimental::RNTupleAttrSetWriter::RNTupleAttrSetWriter(const RNTupleFillContext *mainFillContext,
                                                               RNTupleFillContext fillContext)
   : fFillContext(std::move(fillContext)), fMainFillContext(mainFillContext)
{
}

const ROOT::RNTupleDescriptor &ROOT::Experimental::RNTupleAttrSetWriter::GetDescriptor() const
{
   return fFillContext.fSink->GetDescriptor();
}

ROOT::Experimental::RNTupleAttrEntry ROOT::Experimental::RNTupleAttrSetWriter::BeginRange()
{
   const auto start = fMainFillContext->GetNEntries();
   auto &model = const_cast<RNTupleModel &>(fFillContext.GetModel());
   auto [metaEntry, scopedEntry] = RNTupleAttrEntry::CreateInternalEntries(model);
   auto entry = RNTupleAttrEntry{std::move(metaEntry), std::move(scopedEntry), start};
   return entry;
}

void ROOT::Experimental::RNTupleAttrSetWriter::CommitRange(ROOT::Experimental::RNTupleAttrEntry entry)
{
   if (!entry)
      throw ROOT::RException(R__FAIL("Passed an invalid Attribute Entry to CommitRange()"));

   // Get current entry number from the writer and use it as end of entry range
   const auto end = fMainFillContext->GetNEntries();
   entry.fRange = RNTupleAttrRange::FromStartEnd(entry.fRange.Start(), end);
   auto pRangeStart = entry.fMetaEntry->GetPtr<ROOT::NTupleSize_t>(kRangeStartName);
   auto pRangeLen = entry.fMetaEntry->GetPtr<ROOT::NTupleSize_t>(kRangeLenName);
   R__ASSERT(pRangeStart);
   R__ASSERT(pRangeLen);
   *pRangeStart = entry.GetRange().Start();
   *pRangeLen = entry.GetRange().Length();
   fFillContext.FillImpl(entry);
}

void ROOT::Experimental::RNTupleAttrSetWriter::Commit()
{
   fFillContext.FlushCluster();
   fFillContext.fSink->CommitClusterGroup();
   fFillContext.fSink->CommitDataset();
}

//
//  RNTupleAttrSetReader
//
ROOT::Experimental::RNTupleAttrSetReader::RNTupleAttrSetReader(std::unique_ptr<RNTupleReader> reader)
   : fReader(std::move(reader))
{
   // Collect all entry ranges
   auto entryRangeStartView = fReader->GetView<ROOT::NTupleSize_t>(kRangeStartName);
   auto entryRangeLenView = fReader->GetView<ROOT::NTupleSize_t>(kRangeLenName);
   fEntryRanges.reserve(fReader->GetNEntries());
   for (auto i : fReader->GetEntryRange()) {
      auto start = entryRangeStartView(i);
      auto len = entryRangeLenView(i);
      fEntryRanges.push_back({RNTupleAttrRange::FromStartLength(start, len), i});
   }

   std::sort(fEntryRanges.begin(), fEntryRanges.end(),
             [](const auto &a, const auto &b) { return a.first.Start() < b.first.Start(); });

   R__LOG_INFO(ROOT::Internal::NTupleLog()) << "Loaded " << fEntryRanges.size() << " attribute entries.";
}

const ROOT::RNTupleDescriptor &ROOT::Experimental::RNTupleAttrSetReader::GetDescriptor() const
{
   return fReader->GetDescriptor();
}

// Entry ranges should be sorted with respect to Start by construction.
// TODO: make sure merging attributes preserves the sorting.
bool ROOT::Experimental::RNTupleAttrSetReader::EntryRangesAreSorted(const decltype(fEntryRanges) &ranges)
{
   ROOT::NTupleSize_t prevStart = 0;
   for (const auto &[range, _] : ranges) {
      if (range.Start() < prevStart)
         return false;
      prevStart = range.Start();
   }
   return true;
};

std::vector<ROOT::NTupleSize_t>
ROOT::Experimental::RNTupleAttrSetReader::GetAttributesRangeInternal(NTupleSize_t startEntry, NTupleSize_t endEntry,
                                                                     bool rangeIsContained)
{
   std::vector<ROOT::NTupleSize_t> result;

   if (endEntry < startEntry) {
      R__LOG_WARNING(ROOT::Internal::NTupleLog())
         << "end < start when getting attributes from Attribute Set '" << GetDescriptor().GetName()
         << "' (range given: [" << startEntry << ", " << endEntry << "].";
      return result;
   }

   assert(EntryRangesAreSorted(fEntryRanges));

   const auto FullyContained = [rangeIsContained](auto startInner, auto endInner, auto startOuter, auto endOuter) {
      if (rangeIsContained) {
         std::swap(startOuter, startInner);
         std::swap(endOuter, endInner);
      }
      return startOuter <= startInner && endInner <= endOuter;
   };

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
         result.push_back(index);
      }
   }

   return result;
}

ROOT::Experimental::RNTupleAttrEntryIterable
ROOT::Experimental::RNTupleAttrSetReader::GetAttributesContainingRange(NTupleSize_t startEntry, NTupleSize_t endEntry)
{
   RNTupleAttrRange range;
   if (endEntry <= startEntry) {
      R__LOG_WARNING(ROOT::Internal::NTupleLog())
         << "empty range given when getting attributes from Attribute Set '" << GetDescriptor().GetName()
         << "' (range given: [" << startEntry << ", " << endEntry << ")).";
      // Make sure we find 0 entries
      range = RNTupleAttrRange::FromStartLength(startEntry, 0);
   } else {
      range = RNTupleAttrRange::FromStartEnd(startEntry, endEntry);
   }
   RNTupleAttrEntryIterable::RFilter filter{range, false};
   return RNTupleAttrEntryIterable{*this, filter};
}

ROOT::Experimental::RNTupleAttrEntryIterable
ROOT::Experimental::RNTupleAttrSetReader::GetAttributesInRange(NTupleSize_t startEntry, NTupleSize_t endEntry)
{
   RNTupleAttrRange range;
   if (endEntry <= startEntry) {
      R__LOG_WARNING(ROOT::Internal::NTupleLog())
         << "empty range given when getting attributes from Attribute Set '" << GetDescriptor().GetName()
         << "' (range given: [" << startEntry << ", " << endEntry << ")).";
      // Make sure we find 0 entries
      range = RNTupleAttrRange::FromStartLength(startEntry, 0);
   } else {
      range = RNTupleAttrRange::FromStartEnd(startEntry, endEntry);
   }
   RNTupleAttrEntryIterable::RFilter filter{range, true};
   return RNTupleAttrEntryIterable{*this, filter};
}

ROOT::Experimental::RNTupleAttrEntryIterable
ROOT::Experimental::RNTupleAttrSetReader::GetAttributes(NTupleSize_t entryIndex)
{
   RNTupleAttrEntryIterable::RFilter filter{RNTupleAttrRange::FromStartEnd(entryIndex, entryIndex + 1), false};
   return RNTupleAttrEntryIterable{*this, filter};
}

ROOT::Experimental::RNTupleAttrEntryIterable ROOT::Experimental::RNTupleAttrSetReader::GetAttributes()
{
   return RNTupleAttrEntryIterable{*this};
}

ROOT::Experimental::RNTupleAttrEntry ROOT::Experimental::RNTupleAttrSetReader::CreateAttrEntry()
{
   auto &model = const_cast<RNTupleModel &>(fReader->GetModel());
   auto [metaEntry, scopedEntry] = RNTupleAttrEntry::CreateInternalEntries(model);
   auto attrEntry = RNTupleAttrEntry(std::move(metaEntry), std::move(scopedEntry), RNTupleAttrRange{});
   return attrEntry;
}

void ROOT::Experimental::RNTupleAttrSetReader::LoadAttrEntry(ROOT::NTupleSize_t index, RNTupleAttrEntry &entry)
{
   auto pStart = entry.fMetaEntry->GetPtr<NTupleSize_t>(kRangeStartName);
   auto pLen = entry.fMetaEntry->GetPtr<NTupleSize_t>(kRangeLenName);
   fReader->LoadEntry(index, *entry.fMetaEntry);
   fReader->LoadEntry(index, *entry.fScopedEntry);
   entry.fRange = RNTupleAttrRange::FromStartLength(*pStart, *pLen);
}

bool ROOT::Experimental::RNTupleAttrEntryIterable::RIterator::FullyContained(RNTupleAttrRange range) const
{
   assert(fFilter);
   if (fFilter->fIsContained) {
      return fFilter->fRange.Start() <= range.Start() && range.End() <= fFilter->fRange.End();
   } else {
      return range.Start() <= fFilter->fRange.Start() && fFilter->fRange.End() <= range.End();
   }
}

ROOT::Experimental::RNTupleAttrEntryIterable::RIterator::Iter_t
ROOT::Experimental::RNTupleAttrEntryIterable::RIterator::Next() const
{
   // TODO: consider using binary search, since fEntryRanges is sorted
   // (maybe it should be done only if the size of the list is bigger than a threshold).
   for (auto it = fCur; it != fEnd; ++it) {
      const auto &[range, index] = *it;
      // If we have no filter, every entry is valid.
      if (!fFilter)
         return it;

      const auto &firstLast = range.GetFirstLast();
      // If this is nullopt it means this is a zero-length entry: we always skip those except
      // for the "catch-all" GetAttributes() (which is when fFilter is also nullopt).
      if (!firstLast)
         continue;

      const auto &[first, last] = *firstLast;
      if (first >= fFilter->fRange.End()) {
         // Since fEntryRanges is sorted we know we are at the end of the iteration
         // TODO: tweak fEnd to directly pass the last entry?
         return fEnd;
      }

      if (FullyContained(RNTupleAttrRange::FromStartEnd(first, last + 1)))
         return it;
   }
   return fEnd;
}
