/// \file ROOT/RNTupleProcessor.hxx
/// \ingroup NTuple
/// \author Florine de Geus <florine.de.geus@cern.ch>
/// \date 2025-06-25
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RNTupleProcessorEntry
#define ROOT_RNTupleProcessorEntry

#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RFieldBase.hxx>

#include <cassert>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ROOT {
namespace Experimental {
namespace Internal {
// clang-format off
/**
\class ROOT::Experimental::Internal::RNTupleProcessorEntry
\ingroup NTuple
\brief Collection of values in an RNTupleProcessor, analogous to REntry, with checks and support for missing values.
*/
// clang-format on
class RNTupleProcessorEntry {
public:
   // We don't use RFieldTokens here, because it (semantically) does not make sense for the entry to be fixed to the
   // schema ID of a particular model.
   using FieldIndex_t = std::uint64_t;

private:
   /// Corresponds to the fields stored in the processor's proto-model
   std::vector<ROOT::RFieldBase::RValue> fValues;
   /// Marks whether fields are valid for reading
   std::vector<bool> fFieldValidities;
   /// For fast lookup of field indices given a (sub)field name present in the entry
   std::unordered_map<std::string, FieldIndex_t> fFieldName2Index;
   std::unordered_set<FieldIndex_t> fAuxiliaryFields;

public:
   /////////////////////////////////////////////////////////////////////////////
   /// \brief Set the validity of a field, i.e. whether it is possible to read its value in the current entry.
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   /// \param[in] isValid The new validity of the field.
   void SetFieldValidity(FieldIndex_t fieldIdx, bool isValid)
   {
      assert(fieldIdx < fValues.size());
      fFieldValidities[fieldIdx] = isValid;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Check whether a field is valid for reading.
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   bool IsValidField(FieldIndex_t fieldIdx) const
   {
      assert(fieldIdx < fValues.size());
      return fFieldValidities[fieldIdx];
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Check whether a field is auxiliary.
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   bool FieldIsAuxiliary(FieldIndex_t fieldIdx) const
   {
      return fAuxiliaryFields.find(fieldIdx) != fAuxiliaryFields.end();
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Find the name of a field from its field index.
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   ///
   /// \warning This function has linear complexity, only use it for more helpful error messages!
   const std::string &FindFieldName(FieldIndex_t fieldIdx) const
   {
      assert(fieldIdx < fValues.size());

      for (const auto &[fieldName, index] : fFieldName2Index) {
         if (index == fieldIdx) {
            return fieldName;
         }
      }
      // Should never happen, but avoid compiler warning about "returning reference to local temporary object".
      static const std::string empty = "";
      return empty;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Find the field index of the provided field in the entry.
   ///
   /// \param[in] fieldName The name of the field in the entry.
   ///
   /// \return A `std::optional` containing the field index if it was found.
   std::optional<FieldIndex_t> FindFieldIndex(std::string_view fieldName) const
   {
      auto it = fFieldName2Index.find(std::string(fieldName));
      if (it == fFieldName2Index.end()) {
         return std::nullopt;
      }
      return it->second;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Add a new field to the entry.
   ///
   /// \param[in] fieldName Name of the field to add.
   /// \param[in] field Reference to the field to add, used to to create its corresponding RValue.
   /// \param[in] valuePtr Pointer to an object corresponding to the field's type to bind to its value. If this is a
   /// `nullptr`, a pointer will be created.
   /// \param[in] isAuxiliary Whether the field is auxiliary.
   ///
   /// \return The field index of the newly added field.
   FieldIndex_t AddField(std::string_view fieldName, ROOT::RFieldBase &field, void *valuePtr, bool isAuxiliary = false)
   {
      if (FindFieldIndex(fieldName))
         throw ROOT::RException(
            R__FAIL("field \"" + field.GetQualifiedFieldName() + "\" is already present in the entry"));

      auto value = field.CreateValue();
      if (valuePtr)
         value.BindRawPtr(valuePtr);
      auto fieldIdx = fValues.size();
      fFieldName2Index[std::string(fieldName)] = fieldIdx;
      fValues.emplace_back(std::move(value));
      fFieldValidities.push_back(true);
      if (isAuxiliary)
         fAuxiliaryFields.insert(fieldIdx);

      return fieldIdx;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Update a field in the entry, preserving the value pointer.
   ///
   /// \param[in] fieldIdx Index of the field to update.
   /// \param[in] field The new field to use in the entry.
   void UpdateField(FieldIndex_t fieldIdx, ROOT::RFieldBase &field)
   {
      assert(fieldIdx < fValues.size());

      auto currValuePtr = fValues[fieldIdx].GetPtr<void>();
      auto value = field.CreateValue();
      value.Bind(currValuePtr);
      fValues[fieldIdx] = value;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Bind a new value pointer to a field in the entry.
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   /// \param[in] valuePtr Pointer to the value to bind to the field.
   void BindRawPtr(FieldIndex_t fieldIdx, void *valuePtr)
   {
      assert(fieldIdx < fValues.size());
      fValues[fieldIdx].BindRawPtr(valuePtr);
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Read the field value corresponding to the given field index for the provided entry index.
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   /// \param[in] entryIdx The entry number to read.
   void ReadValue(FieldIndex_t fieldIdx, ROOT::NTupleSize_t entryIdx)
   {
      assert(fieldIdx < fValues.size());

      if (fFieldValidities[fieldIdx]) {
         fValues[fieldIdx].Read(entryIdx);
      }
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get a pointer to the value for the field represented by the provided field index.
   ///
   /// \tparam T The type of the pointer.
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   ///
   /// \return A shared pointer of type `T` with the field's value.
   template <typename T>
   std::shared_ptr<T> GetPtr(FieldIndex_t fieldIdx) const
   {
      assert(fieldIdx < fValues.size());

      if (fFieldValidities[fieldIdx])
         return fValues[fieldIdx].GetPtr<T>();

      return nullptr;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get a reference to a field in the entry.
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   const ROOT::RFieldBase &GetField(FieldIndex_t fieldIdx) const
   {
      assert(fieldIdx < fValues.size());
      return fValues[fieldIdx].GetField();
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get all field indices of this entry.
   std::unordered_set<FieldIndex_t> GetFieldIndices() const
   {
      std::unordered_set<FieldIndex_t> fieldIdxs;
      for (auto &[_, fieldIdx] : fFieldName2Index) {
         fieldIdxs.insert(fieldIdx);
      }
      return fieldIdxs;
   }
};
} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif // ROOT_RNTupleProcessorEntry
