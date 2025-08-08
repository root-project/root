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
//  RNTupleAttrEntryPair
//
std::size_t ROOT::Experimental::Internal::RNTupleAttrEntryPair::Append()
{
   std::size_t bytesWritten = 0;
   // Write the meta entry values
   bytesWritten += fMetaEntry.fValues[0].Append(); // XXX: hardcoded
   bytesWritten += fMetaEntry.fValues[1].Append(); // XXX: hardcoded

   // Bind the user model's memory to the meta model's subfields
   const auto &userFields =
      ROOT::Internal::GetFieldZeroOfModel(fMetaModel).GetMutableSubfields()[2]->GetMutableSubfields(); // XXX: hardcoded
   assert(userFields.size() == fScopedEntry.fValues.size());
   for (std::size_t i = 0; i < fScopedEntry.fValues.size(); ++i) {
      void *userPtr = fScopedEntry.fValues[i].GetPtr<void>().get();
      auto value = userFields[i]->CreateValue();
      value.BindRawPtr(userPtr);
      bytesWritten += value.Append();
   }
   return bytesWritten;
}

//
//  RNTupleAttrSetWriter
//
std::unique_ptr<ROOT::Experimental::RNTupleAttrSetWriter>
ROOT::Experimental::RNTupleAttrSetWriter::Create(std::string_view name, std::unique_ptr<RNTupleModel> userModel,
                                                 const RNTupleFillContext &mainFillContext, TDirectory &dir)

{
   ValidateAttributeModel(*userModel).ThrowOnError();

   // We create a "meta model" that's what we'll use to write data to storage. This meta model has 3 fields:
   // the "meta fields" _rangeStart / _rangeLen and an untyped Record field which contains all the top-level fields
   // from the user model as its children. This is done to "namespace" all user-defined attribute fields so that we
   // are free to use whichever name we want for our meta fields.
   // Note that the user model is preserved as-is to allow the user to create entries from it or use its default
   // entry. When we actually write data to storage, we do some pointer trickery to correctly read the values from
   // the user model and store them under the meta model's fields (see RNTupleAttrEntryPair::Append())
   auto metaModel = RNTupleModel::Create();
   metaModel->SetDescription(userModel->GetDescription());
   metaModel->MakeField<ROOT::NTupleSize_t>(kRangeStartName);
   metaModel->MakeField<ROOT::NTupleSize_t>(kRangeLenName);
   std::vector<std::unique_ptr<RFieldBase>> fields;
   fields.reserve(userModel->GetConstFieldZero().GetConstSubfields().size());
   for (const auto *field : userModel->GetConstFieldZero().GetConstSubfields()) {
      fields.push_back(field->Clone(field->GetFieldName()));
   }
   auto userRootField = std::make_unique<ROOT::RRecordField>(kUserModelName, std::move(fields));
   metaModel->AddField(std::move(userRootField));

   metaModel->Freeze();
   userModel->Freeze();

   // Create a sink that points to the same TDirectory as the main RNTuple
   auto opts = ROOT::RNTupleWriteOptions{};
   opts.SetCompression(mainFillContext.fSink->GetWriteOptions().GetCompression());
   auto sink = std::make_unique<ROOT::Internal::RPageSinkFile>(name, dir, opts);
   RNTupleFillContext fillContext{std::move(metaModel), std::move(sink)};
   return std::unique_ptr<RNTupleAttrSetWriter>(
      new RNTupleAttrSetWriter(mainFillContext, std::move(fillContext), std::move(userModel)));
}

ROOT::Experimental::RNTupleAttrSetWriter::RNTupleAttrSetWriter(const RNTupleFillContext &mainFillContext,
                                                               RNTupleFillContext fillContext,
                                                               std::unique_ptr<RNTupleModel> userModel)
   : fFillContext(std::move(fillContext)), fMainFillContext(&mainFillContext), fUserModel(std::move(userModel))
{
}

ROOT::Experimental::RNTupleAttrPendingRange ROOT::Experimental::RNTupleAttrSetWriter::BeginRange()
{
   const auto start = fMainFillContext->GetNEntries();
   return RNTupleAttrPendingRange{start, fFillContext.GetModel().GetModelId()};
}

void ROOT::Experimental::RNTupleAttrSetWriter::CommitRange(ROOT::Experimental::RNTupleAttrPendingRange pendingRange,
                                                           REntry &entry)
{
   if (pendingRange.GetModelId() != fFillContext.GetModel().GetModelId())
      throw ROOT::RException(R__FAIL("Range passed to CommitRange() of AttributeSet '" + GetDescriptor().GetName() +
                                     "' was not created by it or was already committed."));

   // Get current entry number from the writer and use it as end of entry range
   const auto end = fMainFillContext->GetNEntries();
   auto &metaEntry = fFillContext.fModel->GetDefaultEntry();
   auto pRangeStart = metaEntry.GetPtr<ROOT::NTupleSize_t>(kRangeStartName);
   auto pRangeLen = metaEntry.GetPtr<ROOT::NTupleSize_t>(kRangeLenName);
   R__ASSERT(pRangeStart);
   R__ASSERT(pRangeLen);
   R__ASSERT(end >= pendingRange.Start());
   *pRangeStart = pendingRange.Start();
   *pRangeLen = end - pendingRange.Start();
   Internal::RNTupleAttrEntryPair pair{metaEntry, entry, *fFillContext.fModel};
   fFillContext.FillImpl(pair);
}

void ROOT::Experimental::RNTupleAttrSetWriter::CommitRange(ROOT::Experimental::RNTupleAttrPendingRange pendingRange)
{
   CommitRange(std::move(pendingRange), fUserModel->GetDefaultEntry());
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
   // Initialize user model
   fUserModel = RNTupleModel::Create();
   const auto *userFieldRoot = fReader->GetModel().GetConstFieldZero().GetConstSubfields()[2]; // XXX: hardcoded
   for (const auto *field : userFieldRoot->GetConstSubfields()) {
      fUserModel->AddField(field->Clone(field->GetFieldName()));
   }
   fUserModel->Freeze();

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

ROOT::Experimental::RNTupleAttrRange
ROOT::Experimental::RNTupleAttrSetReader::LoadAttrEntry(ROOT::NTupleSize_t index, REntry &entry)
{
   auto &metaModel = const_cast<ROOT::RNTupleModel &>(fReader->GetModel());
   auto &metaEntry = metaModel.GetDefaultEntry();

   if (R__unlikely(entry.GetModelId() != fUserModel->GetModelId()))
      throw RException(R__FAIL("mismatch between entry and model"));

   // Load the meta fields
   metaEntry.fValues[0].Read(index); // XXX: hardcoded
   metaEntry.fValues[1].Read(index); // XXX: hardcoded

   // Load the user fields into `entry`
   auto *userRootField = ROOT::Internal::GetFieldZeroOfModel(metaModel).GetMutableSubfields()[2]; // XXX: hardcoded
   const auto userFields = userRootField->GetMutableSubfields();
   assert(entry.fValues.size() == userFields.size());
   for (std::size_t i = 0; i < userFields.size(); ++i) {
      auto *field = userFields[i];
      field->Read(index, entry.fValues[i].GetPtr<void>().get());
   }

   auto pStart = metaEntry.GetPtr<NTupleSize_t>(kRangeStartName);
   auto pLen = metaEntry.GetPtr<NTupleSize_t>(kRangeLenName);

   return RNTupleAttrRange::FromStartLength(*pStart, *pLen);
}

ROOT::Experimental::RNTupleAttrRange ROOT::Experimental::RNTupleAttrSetReader::LoadAttrEntry(ROOT::NTupleSize_t index)
{
   auto &entry = fUserModel->GetDefaultEntry();
   return LoadAttrEntry(index, entry);
}

//
//  RNTupleAttrEntryIterable
//
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
