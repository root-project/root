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
\class ROOT::Experimental::RNTupleProcessorProvenance
\ingroup NTuple
\brief Identifies how a processor is composed.

The processor provenance is used in RNTupleProcessorEntry to identify how an (auxiliary) field in a composed processor
can be accessed.
*/
// clang-format on
class RNTupleProcessorProvenance {
private:
   std::string fProvenance{};

public:
   RNTupleProcessorProvenance() = default;
   RNTupleProcessorProvenance(const std::string &provenance) : fProvenance(provenance) {}

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the full processor provenance, in the form of "x.y.z".
   std::string Get() const { return fProvenance; }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Add a new processor to the provenance.
   ///
   /// \param[in] processorName Name of the processor to add.
   ///
   /// \return The updated provenance.
   RNTupleProcessorProvenance Evolve(const std::string &processorName) const
   {
      if (fProvenance.empty())
         return RNTupleProcessorProvenance(processorName);

      return RNTupleProcessorProvenance(fProvenance + "." + processorName);
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Check whether the provenance subsumes the provenance in `other`.
   ///
   /// \param[in] other The other provenance
   bool Contains(const RNTupleProcessorProvenance &other) const
   {
      return fProvenance.rfind(other.fProvenance) != std::string::npos;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Check whether the provided field name contains this provenance.
   ///
   /// \param[in] fieldName Field name to check.
   bool IsPresentInFieldName(std::string_view fieldName) const
   {
      return !fProvenance.empty() && fieldName.find(fProvenance + ".") == 0;
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
   struct RProcessorValue {
      ROOT::RFieldBase::RValue fValue;
      bool fIsValid;
      RNTupleProcessorProvenance fProcessorProvenance;

      RProcessorValue(ROOT::RFieldBase::RValue &&value, bool isValid, RNTupleProcessorProvenance provenance)
         : fValue(std::move(value)), fIsValid(isValid), fProcessorProvenance(provenance)
      {
      }
   };

   std::vector<RProcessorValue> fProcessorValues;
   std::unordered_map<std::string, FieldIndex_t> fFieldName2Index;

public:
   /////////////////////////////////////////////////////////////////////////////
   /// \brief Set the validity of a field, i.e. whether it is possible to read its value in the current entry.
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   /// \param[in] isValid The new validity of the field.
   void SetFieldValidity(FieldIndex_t fieldIdx, bool isValid)
   {
      assert(fieldIdx < fProcessorValues.size());
      fProcessorValues[fieldIdx].fIsValid = isValid;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Check whether a field is valid for reading.
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   bool IsValidField(FieldIndex_t fieldIdx) const
   {
      assert(fieldIdx < fProcessorValues.size());
      return fProcessorValues[fieldIdx].fIsValid;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Find the name of a field from its field index.
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   ///
   /// \warning This function has linear complexity, only use it for more helpful error messages!
   const std::string &FindFieldName(FieldIndex_t fieldIdx) const
   {
      assert(fieldIdx < fProcessorValues.size());

      for (const auto &[fieldName, index] : fFieldName2Index) {
         if (index == fieldIdx) {
            return fieldName;
         }
      }
      // Should never happen, but avoid compiler warning about "returning reference to local temporary object".
      R__ASSERT(false);
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
   /// \param[in] provenance Processor provenance of the field.
   ///
   /// \return The field index of the newly added field.
   FieldIndex_t AddField(std::string_view fieldName, ROOT::RFieldBase &field, void *valuePtr,
                         const RNTupleProcessorProvenance &provenance)
   {
      if (FindFieldIndex(fieldName))
         throw ROOT::RException(
            R__FAIL("field \"" + field.GetQualifiedFieldName() + "\" is already present in the entry"));

      auto value = field.CreateValue();
      if (valuePtr)
         value.BindRawPtr(valuePtr);
      auto fieldIdx = fProcessorValues.size();
      fFieldName2Index[std::string(fieldName)] = fieldIdx;
      fProcessorValues.emplace_back(RProcessorValue(std::move(value), true, provenance));

      return fieldIdx;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Update a field in the entry, preserving the value pointer.
   ///
   /// \param[in] fieldIdx Index of the field to update.
   /// \param[in] field The new field to use in the entry.
   void UpdateField(FieldIndex_t fieldIdx, ROOT::RFieldBase &field)
   {
      assert(fieldIdx < fProcessorValues.size());

      auto currValuePtr = fProcessorValues[fieldIdx].fValue.GetPtr<void>();
      auto value = field.CreateValue();
      value.Bind(currValuePtr);
      fProcessorValues[fieldIdx].fValue = value;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Bind a new value pointer to a field in the entry.
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   /// \param[in] valuePtr Pointer to the value to bind to the field.
   void BindRawPtr(FieldIndex_t fieldIdx, void *valuePtr)
   {
      assert(fieldIdx < fProcessorValues.size());
      fProcessorValues[fieldIdx].fValue.BindRawPtr(valuePtr);
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Read the field value corresponding to the given field index for the provided entry index.
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   /// \param[in] entryIdx The entry number to read.
   void ReadValue(FieldIndex_t fieldIdx, ROOT::NTupleSize_t entryIdx)
   {
      assert(fieldIdx < fProcessorValues.size());

      if (fProcessorValues[fieldIdx].fIsValid) {
         fProcessorValues[fieldIdx].fValue.Read(entryIdx);
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
      assert(fieldIdx < fProcessorValues.size());

      if (fProcessorValues[fieldIdx].fIsValid)
         return fProcessorValues[fieldIdx].fValue.GetPtr<T>();

      return nullptr;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get a reference to a field in the entry.
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   const ROOT::RFieldBase &GetField(FieldIndex_t fieldIdx) const
   {
      assert(fieldIdx < fProcessorValues.size());
      return fProcessorValues[fieldIdx].fValue.GetField();
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the processor provenance of a field in the entry.
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   const RNTupleProcessorProvenance &GetFieldProvenance(FieldIndex_t fieldIdx) const
   {
      assert(fieldIdx < fProcessorValues.size());
      return fProcessorValues[fieldIdx].fProcessorProvenance;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the name of a field in the entry, including processor name prefixes in the case of auxiliary fields.
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   std::string GetFieldName(FieldIndex_t fieldIdx) const
   {
      assert(fieldIdx < fProcessorValues.size());
      return fProcessorValues[fieldIdx].fProcessorProvenance.Get() + "." +
             fProcessorValues[fieldIdx].fValue.GetField().GetQualifiedFieldName();
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get all field indices of this entry.
   std::unordered_set<FieldIndex_t> GetFieldIndices() const
   {
      // Field indices are sequentially assigned, and the entry (currently) offers no way to remove fields, so we can
      // just generate and return a set {0, ..., |fProcessorValues| - 1}.
      std::unordered_set<FieldIndex_t> fieldIdxs(fProcessorValues.size());
      std::generate_n(std::inserter(fieldIdxs, fieldIdxs.begin()), fProcessorValues.size(),
                      [i = 0]() mutable { return i++; });
      return fieldIdxs;
   }
};
} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif // ROOT_RNTupleProcessorEntry
