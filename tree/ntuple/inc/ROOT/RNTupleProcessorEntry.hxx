/// \file ROOT/RNTupleProcessor.hxx
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
      return !fProvenance.empty() && fieldName.find(fProvenance) == 0;
   }

   bool Empty() const { return fProvenance.empty(); }
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
      std::unique_ptr<ROOT::RFieldBase> fField;
      std::string fQualifiedFieldName;
      ROOT::RFieldBase::RValue fValue;
      RNTupleProcessorProvenance fProcessorProvenance;
      const bool fIsJoinField;
      bool fIsActive;
      bool fIsValid;

      RProcessorValue(std::unique_ptr<ROOT::RFieldBase> field, std::string_view qualifiedFieldName,
                      ROOT::RFieldBase::RValue &&value, RNTupleProcessorProvenance provenance, bool isJoinField,
                      bool isActive, bool isValid)
         : fField(std::move(field)),
           fQualifiedFieldName(qualifiedFieldName),
           fValue(std::move(value)),
           fProcessorProvenance(provenance),
           fIsJoinField(isJoinField),
           fIsActive(isActive),
           fIsValid(isValid)
      {
      }
   };

   std::vector<RProcessorValue> fProcessorValues;
   // Maps from the field name to all type alternatives for that field that have been added to the entry.
   std::unordered_map<std::string, std::vector<FieldIndex_t>> fFieldName2Index;

public:
   /////////////////////////////////////////////////////////////////////////////
   /// \brief Clear all fields from the entry.
   void Clear()
   {
      fProcessorValues.clear();
      fFieldName2Index.clear();
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief (De)activate reading of a field.
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   /// \param[in] isActive The new status of the field.
   void SetFieldIsActive(FieldIndex_t fieldIdx, bool isActive)
   {
      assert(fieldIdx < fProcessorValues.size());
      fProcessorValues[fieldIdx].fIsActive = isActive;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Check whether a field is activated for reading.
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   bool IsActiveField(FieldIndex_t fieldIdx) const
   {
      assert(fieldIdx < fProcessorValues.size());
      return fProcessorValues[fieldIdx].fIsActive;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Check whether a field is a join field
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   bool IsJoinField(FieldIndex_t fieldIdx) const
   {
      assert(fieldIdx < fProcessorValues.size());
      return fProcessorValues[fieldIdx].fIsJoinField;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Set the validity of a field, i.e. whether it is possible to read its value in the current entry.
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   /// \param[in] isValid The new validity of the field.
   void SetFieldIsValid(FieldIndex_t fieldIdx, bool isValid)
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
   const std::string &FindFieldName(FieldIndex_t fieldIdx) const;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Find the field index of the provided field in the entry.
   ///
   /// \param[in] canonicalFieldName The name of the field in the entry, including its processor name prefixes and
   /// parent field names, if applicable.
   /// \param[in] typeName Type of the field, if relevant. If no type name is provided, the first field corresponding to
   /// the provided name is returned.
   ///
   /// \return A `std::optional` containing the field index if it was found.
   std::optional<FieldIndex_t>
   FindFieldIndex(std::string_view canonicalFieldName, std::string_view typeName = "") const;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Add a new field to the entry.
   ///
   /// \param[in] qualifiedFieldName Name of the field to add, including its parent field if applicable.
   /// \param[in] field Reference to the field to add, used to to create its corresponding RValue.
   /// \param[in] valuePtr Pointer to an object corresponding to the field's type to bind to its value. If this is a
   /// `nullptr`, a pointer will be created.
   /// \param[in] provenance Processor provenance of the field.
   /// \param[in] isJoinField Whether the field is a join field.
   ///
   /// \return The field index of the newly added field.
   FieldIndex_t AddField(const std::string &qualifiedFieldName, std::unique_ptr<ROOT::RFieldBase> field, void *valuePtr,
                         const RNTupleProcessorProvenance &provenance, bool isJoinField);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Update a field in the entry, preserving the value pointer.
   ///
   /// \param[in] fieldIdx Index of the field to update.
   /// \param[in] field The new field to use in the entry.
   void UpdateField(FieldIndex_t fieldIdx, std::unique_ptr<ROOT::RFieldBase> field);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get a field by name (and optionally type)
   ///
   /// \param[in] canonicalFieldName The name of the field in the entry, including its processor name prefixes and
   /// parent field names, if applicable.
   /// \param[in] typeName Type of the field, if relevant. If no type name is provided, the first field corresponding to
   /// the provided name is returned.
   ///
   /// \return A pointer to the field, or a `nullptr` if the field does not exist in the entry.
   const ROOT::RFieldBase &GetField(FieldIndex_t fieldIdx) const;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Bind a new value pointer to a field in the entry.
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   /// \param[in] valuePtr Pointer to the value to bind to the field.
   void BindRawPtr(FieldIndex_t fieldIdx, void *valuePtr);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Read the field value corresponding to the given field index for the provided entry index.
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   /// \param[in] entryIdx The entry number to read.
   void ReadValue(FieldIndex_t fieldIdx, ROOT::NTupleSize_t entryIdx);

   const ROOT::RFieldBase::RValue &GetValue(FieldIndex_t fieldIdx) const
   {
      assert(fieldIdx < fProcessorValues.size());
      return fProcessorValues[fieldIdx].fValue;
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
   /// \brief Get the name of a field in the entry, including its parent fields.
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   std::string GetQualifiedFieldName(FieldIndex_t fieldIdx) const
   {
      assert(fieldIdx < fProcessorValues.size());
      return fProcessorValues[fieldIdx].fQualifiedFieldName;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get all field indices of this entry.
   std::unordered_set<FieldIndex_t> GetFieldIndices() const;
};
} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif // ROOT_RNTupleProcessorEntry
