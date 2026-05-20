/// \file RNTupleAttrReading.cxx
/// \ingroup NTuple
/// \author Giacomo Parolini <giacomo.parolini@cern.ch>
/// \date 2026-04-01
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

#include <ROOT/RNTupleAttrReading.hxx>
#include <ROOT/RNTupleAttrUtils.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RLogger.hxx>

#include <algorithm>
#include <cstddef>
#include <utility>

using namespace ROOT::Experimental::Internal::RNTupleAttributes;

ROOT::Experimental::RNTupleAttrSetReader::RNTupleAttrSetReader(std::unique_ptr<RNTupleReader> reader,
                                                               std::uint16_t vSchemaMajor)
   : fReader(std::move(reader))
{
   // Changes in major version imply forward incompatibility
   if (vSchemaMajor != kSchemaVersionMajor)
      throw ROOT::RException(R__FAIL("unsupported attribute schema version: " + std::to_string(vSchemaMajor)));

   // Initialize user model
   fUserModel = RNTupleModel::Create();

   // Validate meta model format
   const auto &metaDesc = fReader->GetDescriptor();
   const auto &metaFieldIds = metaDesc.GetFieldZero().GetLinkIds();
   if (metaFieldIds.size() != kMetaFieldIndex_Count) {
      throw ROOT::RException(R__FAIL("invalid number of attribute meta-fields: expected " +
                                     std::to_string(kMetaFieldIndex_Count) + ", got " +
                                     std::to_string(metaFieldIds.size())));
   }
   for (std::size_t i = 0; i < kMetaFieldIndex_Count; ++i) {
      const auto fieldId = metaFieldIds[i];
      const auto &metaField = metaDesc.GetFieldDescriptor(fieldId);
      if (metaField.GetFieldName() != kMetaFieldNames[i]) {
         throw ROOT::RException(R__FAIL(std::string("invalid attribute meta-field name: expected '") +
                                        kMetaFieldNames[i] + "', got '" + metaField.GetFieldName() + "'"));
      }
   }

   const auto &userFieldRoot = metaDesc.GetFieldDescriptor(metaFieldIds[kMetaFieldIndex_UserData]);
   for (const auto fieldId : userFieldRoot.GetLinkIds()) {
      const auto &fdesc = metaDesc.GetFieldDescriptor(fieldId);
      auto userField = RFieldBase::Create(fdesc.GetFieldName(), fdesc.GetTypeName()).Unwrap();
      fUserModel->AddField(std::move(userField));
   }
   fUserModel->Freeze();

   // Collect all entry ranges
   auto entryRangeStartView = fReader->GetView<ROOT::NTupleSize_t>(kMetaFieldNames[kMetaFieldIndex_RangeStart]);
   auto entryRangeLenView = fReader->GetView<ROOT::NTupleSize_t>(kMetaFieldNames[kMetaFieldIndex_RangeLen]);
   fEntryRanges.reserve(fReader->GetNEntries());
   for (auto i : fReader->GetEntryRange()) {
      auto start = entryRangeStartView(i);
      auto len = entryRangeLenView(i);
      fEntryRanges.push_back({RNTupleAttrRange::FromStartLength(start, len), i});
   }

   std::stable_sort(fEntryRanges.begin(), fEntryRanges.end(),
                    [](const auto &a, const auto &b) { return a.first.GetStart() < b.first.GetStart(); });

   R__LOG_INFO(ROOT::Internal::NTupleLog()) << "Loaded " << fEntryRanges.size() << " attribute entries.";
}

const ROOT::RNTupleDescriptor &ROOT::Experimental::RNTupleAttrSetReader::GetDescriptor() const
{
   return fReader->GetDescriptor();
}

ROOT::Experimental::RNTupleAttrRange
ROOT::Experimental::RNTupleAttrSetReader::LoadEntry(ROOT::NTupleSize_t index, REntry &entry)
{
   // TODO(gparolini): find a way to avoid this const_cast
   auto &metaModel = const_cast<ROOT::RNTupleModel &>(fReader->GetModel());
   auto &metaEntry = metaModel.GetDefaultEntry();

   if (R__unlikely(entry.GetModelId() != fUserModel->GetModelId()))
      throw RException(R__FAIL("mismatch between entry and model"));

   // Load the meta fields
   metaEntry.fValues[kMetaFieldIndex_RangeStart].Read(index);
   metaEntry.fValues[kMetaFieldIndex_RangeLen].Read(index);

   // Load the user fields into `entry`
   auto *userRootField = ROOT::Internal::GetFieldZeroOfModel(metaModel).GetMutableSubfields()[kMetaFieldIndex_UserData];
   const auto userFields = userRootField->GetMutableSubfields();
   assert(entry.fValues.size() == userFields.size());
   for (std::size_t i = 0; i < userFields.size(); ++i) {
      auto *field = userFields[i];
      field->Read(index, entry.fValues[i].GetPtr<void>().get());
   }

   auto pStart = metaEntry.fValues[kMetaFieldIndex_RangeStart].GetPtr<NTupleSize_t>();
   auto pLen = metaEntry.fValues[kMetaFieldIndex_RangeLen].GetPtr<NTupleSize_t>();

   return RNTupleAttrRange::FromStartLength(*pStart, *pLen);
}

ROOT::Experimental::RNTupleAttrRange ROOT::Experimental::RNTupleAttrSetReader::LoadEntry(ROOT::NTupleSize_t index)
{
   auto &entry = fUserModel->GetDefaultEntry();
   return LoadEntry(index, entry);
}

std::unique_ptr<ROOT::REntry> ROOT::Experimental::RNTupleAttrSetReader::CreateEntry()
{
   return fUserModel->CreateEntry();
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
   RNTupleAttrEntryIterable::RFilter filter{range, /*fIsContained=*/false};
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
   RNTupleAttrEntryIterable::RFilter filter{range, /*fIsContained=*/true};
   return RNTupleAttrEntryIterable{*this, filter};
}

ROOT::Experimental::RNTupleAttrEntryIterable
ROOT::Experimental::RNTupleAttrSetReader::GetAttributes(NTupleSize_t entryIndex)
{
   RNTupleAttrEntryIterable::RFilter filter{RNTupleAttrRange::FromStartEnd(entryIndex, entryIndex + 1),
                                            /*fIsContained=*/false};
   return RNTupleAttrEntryIterable{*this, filter};
}

ROOT::Experimental::RNTupleAttrEntryIterable ROOT::Experimental::RNTupleAttrSetReader::GetAttributes()
{
   return RNTupleAttrEntryIterable{*this};
}

//
//  RNTupleAttrEntryIterable
//
bool ROOT::Experimental::RNTupleAttrEntryIterable::RIterator::FullyContained(RNTupleAttrRange range) const
{
   assert(fFilter);
   if (fFilter->fIsContained) {
      return fFilter->fRange.GetStart() <= range.GetStart() && range.GetEnd() <= fFilter->fRange.GetEnd();
   } else {
      return range.GetStart() <= fFilter->fRange.GetStart() && fFilter->fRange.GetEnd() <= range.GetEnd();
   }
}

ROOT::Experimental::RNTupleAttrEntryIterable::RIterator::Iter_t
ROOT::Experimental::RNTupleAttrEntryIterable::RIterator::SkipFiltered() const
{
   // If we have no filter, every entry is valid.
   if (!fFilter)
      return fCur;

   // TODO: consider using binary search, since fEntryRanges is sorted
   // (maybe it should be done only if the size of the list is bigger than a threshold).
   for (auto it = fCur; it != fEnd; ++it) {
      const auto &[range, index] = *it;
      const auto &firstLast = range.GetFirstLast();
      // If this is nullopt it means this is a zero-length entry: we always skip those except
      // for the "catch-all" GetAttributes() (which is when fFilter is also nullopt).
      if (!firstLast)
         continue;

      const auto &[first, last] = *firstLast;
      if (first >= fFilter->fRange.GetEnd()) {
         // Since fEntryRanges is sorted we know we are at the end of the iteration
         // TODO: tweak fEnd to directly pass the last entry?
         return fEnd;
      }

      if (FullyContained(RNTupleAttrRange::FromStartEnd(first, last + 1)))
         return it;
   }
   return fEnd;
}
