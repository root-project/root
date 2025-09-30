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
/**
\class ROOT::Experimental::RNTupleProcessorTag
\ingroup NTuple
\brief Identifies how a processor is composed.

The processor tag is used in RNTupleProcessorEntry to identify how a field in a composed processor can be accessed.
*/
// clang-format on
class RNTupleProcessorTag {
private:
   std::vector<std::string> fTag{};

   RNTupleProcessorTag(std::vector<std::string> &&tag) : fTag(tag) {}

public:
   RNTupleProcessorTag() = default;
   RNTupleProcessorTag(const std::string &tag) : fTag(std::vector<std::string>{tag}) {}

   const std::vector<std::string> &GetTag() const { return fTag; }
   bool IsEmpty() const { return fTag.empty(); }
   std::string GetFieldNamePrefix() const
   {
      if (fTag.empty())
         return std::string{};

      return std::accumulate(fTag.cbegin() + 1, fTag.cend(), fTag[0],
                             [](std::string acc, const std::string &t) { return std::move(acc) + "." + t; });
   }
   const std::string &GetOutermostProcessorName() const { return fTag.back(); }

   RNTupleProcessorTag Evolve(std::string_view processorName) const
   {
      std::vector<std::string> newTag = fTag;
      newTag.emplace_back(processorName);
      return RNTupleProcessorTag(std::move(newTag));
   }
};

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
   struct RFieldInfo {
      ROOT::RFieldBase::RValue fValue;
      bool fIsValid;
      RNTupleProcessorTag fProcessorTag;

      RFieldInfo(ROOT::RFieldBase::RValue &&value, bool isValid, RNTupleProcessorTag processorTag)
         : fValue(std::move(value)), fIsValid(isValid), fProcessorTag(processorTag)
      {
      }
   };

   std::vector<RFieldInfo> fFields;

   /// Corresponds to the fields of the proto model
   // std::vector<ROOT::RFieldBase::RValue> fValues;
   /// Marks whether fields are valid for reading
   // std::vector<bool> fFieldValidities;
   /// For fast lookup of field indices given a (sub)field name present in the entry
   std::unordered_map<std::string, FieldIndex_t> fFieldName2Index;

public:
   /////////////////////////////////////////////////////////////////////////////
   /// \brief Set the validity of a field, i.e. whether it is possible to read its value in the current entry.
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   /// \param[in] isValid The new validity of the field.
   void SetFieldValidity(FieldIndex_t fieldIdx, bool isValid)
   {
      assert(fieldIdx < fFields.size());
      fFields[fieldIdx].fIsValid = isValid;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Check whether a field is valid for reading.
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   bool IsValidField(FieldIndex_t fieldIdx) const
   {
      assert(fieldIdx < fFields.size());
      return fFields[fieldIdx].fIsValid;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Find the name of a field from its field index.
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   ///
   /// \warning This function has linear complexity, only use it for more helpful error messages!
   const std::string &FindFieldName(FieldIndex_t fieldIdx) const
   {
      assert(fieldIdx < fFields.size());

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
   FieldIndex_t
   AddField(std::string_view fieldName, ROOT::RFieldBase &field, void *valuePtr, const RNTupleProcessorTag &tag)
   {
      if (FindFieldIndex(fieldName))
         throw ROOT::RException(
            R__FAIL("field \"" + field.GetQualifiedFieldName() + "\" is already present in the entry"));

      auto value = field.CreateValue();
      if (valuePtr)
         value.BindRawPtr(valuePtr);
      auto fieldIdx = fFields.size();
      fFieldName2Index[std::string(fieldName)] = fieldIdx;
      // fValues.emplace_back(std::move(value));
      // fFieldValidities.push_back(true);
      fFields.emplace_back(RFieldInfo(std::move(value), true, tag));

      return fieldIdx;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Update a field in the entry, preserving the value pointer.
   ///
   ///
   /// \param[in] fieldIdx Index of the field to update.
   /// \param[in] field The new field to use in the entry.
   void UpdateField(FieldIndex_t fieldIdx, ROOT::RFieldBase &field)
   {
      assert(fieldIdx < fFields.size());

      auto currValuePtr = fFields[fieldIdx].fValue.GetPtr<void>();
      auto value = field.CreateValue();
      value.Bind(currValuePtr);
      fFields[fieldIdx].fValue = value;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Bind a new value pointer to a field in the entry.
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   /// \param[in] valuePtr Pointer to the value to bind to the field.
   void BindRawPtr(FieldIndex_t fieldIdx, void *valuePtr)
   {
      assert(fieldIdx < fFields.size());
      fFields[fieldIdx].fValue.BindRawPtr(valuePtr);
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Read the field values for the entry corresponding to the provided index.
   ///
   /// \param[in] entryIdx The entry number to read.
   void Read(ROOT::NTupleSize_t entryIdx)
   {
      for (unsigned i = 0; i < fFields.size(); i++) {
         if (fFields[i].fIsValid)
            fFields[i].fValue.Read(entryIdx);
      }
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Read the field value corresponding to the given field index for the provided entry index.
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   /// \param[in] entryIdx The entry number to read.
   void ReadValue(FieldIndex_t fieldIdx, ROOT::NTupleSize_t entryIdx)
   {
      assert(fieldIdx < fFields.size());

      if (fFields[fieldIdx].fIsValid) {
         fFields[fieldIdx].fValue.Read(entryIdx);
      }
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get a pointer to the value for the field represented by the provided field index.
   ///
   /// \tparam T The type of the pointer
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   template <typename T>
   std::shared_ptr<T> GetPtr(FieldIndex_t fieldIdx) const
   {
      assert(fieldIdx < fFields.size());

      if (fFields[fieldIdx].fIsValid)
         return fFields[fieldIdx].fValue.GetPtr<T>();

      return nullptr;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get a reference to a field in the entry.
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   const ROOT::RFieldBase &GetField(FieldIndex_t fieldIdx)
   {
      assert(fieldIdx < fFields.size());
      return fFields[fieldIdx].fValue.GetField();
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the name of a field in the entry, including processor name prefixes in the case of auxiliary fields.
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   std::string GetFieldName(FieldIndex_t fieldIdx)
   {
      assert(fieldIdx < fFields.size());
      return fFields[fieldIdx].fProcessorTag.GetFieldNamePrefix() + "." +
             fFields[fieldIdx].fValue.GetField().GetQualifiedFieldName();
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get all field indices of this entry.
   std::unordered_set<FieldIndex_t> GetFieldIndices()
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
