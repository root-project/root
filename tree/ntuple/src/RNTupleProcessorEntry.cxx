/// \file RNTupleProcessor.cxx
/// \ingroup NTuple
/// \author Florine de Geus <florine.de.geus@cern.ch>
/// \date 2026-05-11
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RNTupleProcessorEntry.hxx>

const std::string &ROOT::Experimental::Internal::RNTupleProcessorEntry::FindFieldName(FieldIndex_t fieldIdx) const
{
   assert(fieldIdx < fProcessorValues.size());

   for (const auto &[fieldName, index] : fFieldName2Index) {
      if (std::find(index.begin(), index.end(), fieldIdx) != index.end()) {
         return fieldName;
      }
   }
   // Should never happen, but avoid compiler warning about "returning reference to local temporary object".
   R__ASSERT(false);
   static const std::string empty = "";
   return empty;
}

std::optional<ROOT::Experimental::Internal::RNTupleProcessorEntry::FieldIndex_t>
ROOT::Experimental::Internal::RNTupleProcessorEntry::FindFieldIndex(std::string_view canonicalFieldName,
                                                                    std::string_view typeName) const
{
   auto it = fFieldName2Index.find(std::string(canonicalFieldName));
   if (it == fFieldName2Index.end()) {
      return std::nullopt;
   }

   const auto &fieldIdxs = it->second;
   assert(!fieldIdxs.empty());

   for (auto idx : fieldIdxs) {
      if (fProcessorValues[idx].fField->GetTypeName() == typeName || typeName == "") {
         return idx;
      }
   }

   return std::nullopt;
}

ROOT::Experimental::Internal::RNTupleProcessorEntry::FieldIndex_t
ROOT::Experimental::Internal::RNTupleProcessorEntry::AddField(const std::string &qualifiedFieldName,
                                                              std::unique_ptr<ROOT::RFieldBase> field, void *valuePtr,
                                                              const RNTupleProcessorProvenance &provenance,
                                                              bool isJoinField)
{
   auto fieldNameWithProcessorPrefix = qualifiedFieldName;
   if (const auto &processorPrefix = provenance.Get(); !processorPrefix.empty())
      fieldNameWithProcessorPrefix = processorPrefix + "." + qualifiedFieldName;

   if (FindFieldIndex(fieldNameWithProcessorPrefix, field->GetTypeName()))
      throw ROOT::RException(R__FAIL("field \"" + fieldNameWithProcessorPrefix + "\" is already present in the entry"));

   auto fieldIdx = fProcessorValues.size();
   fFieldName2Index[fieldNameWithProcessorPrefix].push_back(fieldIdx);

   assert(field);
   auto value = field->CreateValue();
   if (valuePtr)
      value.BindRawPtr(valuePtr);
   fProcessorValues.emplace_back(RProcessorValue(std::move(field), qualifiedFieldName, std::move(value), provenance,
                                                 isJoinField, /*isValid=*/true));

   return fieldIdx;
}

const ROOT::RFieldBase &ROOT::Experimental::Internal::RNTupleProcessorEntry::GetField(FieldIndex_t fieldIdx) const
{
   assert(fieldIdx < fProcessorValues.size());
   return *fProcessorValues[fieldIdx].fField;
}

void ROOT::Experimental::Internal::RNTupleProcessorEntry::UpdateField(FieldIndex_t fieldIdx,
                                                                      std::unique_ptr<ROOT::RFieldBase> field)
{
   assert(fieldIdx < fProcessorValues.size());

   auto &fieldInfo = fProcessorValues[fieldIdx];

   if (field) {
      auto newValue = field->CreateValue();
      auto currValuePtr = fieldInfo.fValue.GetPtr<void>();
      newValue.Bind(currValuePtr);
      fieldInfo.fField = std::move(field);
      fieldInfo.fValue = std::move(newValue);
      fieldInfo.fIsValid = true;
   } else {
      fieldInfo.fIsValid = false;
   }
}

void ROOT::Experimental::Internal::RNTupleProcessorEntry::BindRawPtr(FieldIndex_t fieldIdx, void *valuePtr)
{
   assert(fieldIdx < fProcessorValues.size());
   fProcessorValues[fieldIdx].fValue.BindRawPtr(valuePtr);
}

void ROOT::Experimental::Internal::RNTupleProcessorEntry::ReadValue(FieldIndex_t fieldIdx, ROOT::NTupleSize_t entryIdx)
{
   assert(fieldIdx < fProcessorValues.size());

   if (fProcessorValues[fieldIdx].fIsValid) {
      fProcessorValues[fieldIdx].fValue.Read(entryIdx);
   }
}

std::unordered_set<ROOT::Experimental::Internal::RNTupleProcessorEntry::FieldIndex_t>
ROOT::Experimental::Internal::RNTupleProcessorEntry::GetFieldIndices() const
{
   // Field indices are sequentially assigned, and the entry (currently) offers no way to remove fields, so we can just
   // generate and return a set {0, ..., |fProcessorValues| - 1}.
   std::unordered_set<FieldIndex_t> fieldIdxs(fProcessorValues.size());
   std::generate_n(std::inserter(fieldIdxs, fieldIdxs.begin()), fProcessorValues.size(),
                   [i = 0]() mutable { return i++; });
   return fieldIdxs;
}
