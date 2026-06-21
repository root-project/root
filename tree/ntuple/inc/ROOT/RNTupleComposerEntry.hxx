/// \file ROOT/RNTupleComposer.hxx
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

#ifndef ROOT_RNTupleComposerEntry
#define ROOT_RNTupleComposerEntry

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
\class ROOT::Experimental::RNTupleCompositionProvenance
\ingroup NTuple
\brief Identifies the provenance of a composed RNTuple.

The composition provenance is used in RNTupleComposerEntry to identify how an (auxiliary) field in a composed RNTuple
can be accessed.
*/
// clang-format on
class RNTupleCompositionProvenance {
private:
   std::string fProvenance{};

public:
   RNTupleCompositionProvenance() = default;
   RNTupleCompositionProvenance(const std::string &provenance) : fProvenance(provenance) {}

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the full composer provenance, in the form of "x.y.z".
   std::string Get() const { return fProvenance; }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Add a new composer to the provenance.
   ///
   /// \param[in] ntupleName Name of the composed RNTuple to add.
   ///
   /// \return The updated provenance.
   RNTupleCompositionProvenance Evolve(const std::string &ntupleName) const
   {
      if (fProvenance.empty())
         return RNTupleCompositionProvenance(ntupleName);

      return RNTupleCompositionProvenance(fProvenance + "." + ntupleName);
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Check whether the provenance subsumes the provenance in `other`.
   ///
   /// \param[in] other The other provenance
   bool Contains(const RNTupleCompositionProvenance &other) const
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
\class ROOT::Experimental::Internal::RNTupleComposerEntry
\ingroup NTuple
\brief Collection of values in an RNTupleComposer, analogous to REntry, with checks and support for missing values.
*/
// clang-format on
class RNTupleComposerEntry {
public:
   // We don't use RFieldTokens here, because it (semantically) does not make sense for the entry to be fixed to the
   // schema ID of a particular model.
   using FieldIndex_t = std::uint64_t;

private:
   struct RComposerValue {
      std::unique_ptr<ROOT::RFieldBase> fField;
      std::string fQualifiedFieldName;
      ROOT::RFieldBase::RValue fValue;
      bool fIsValid;
      RNTupleCompositionProvenance fCompositionProvenance;

      RComposerValue(std::unique_ptr<ROOT::RFieldBase> field, std::string_view qualifiedFieldName,
                     ROOT::RFieldBase::RValue &&value, bool isValid, RNTupleCompositionProvenance provenance)
         : fField(std::move(field)),
           fQualifiedFieldName(qualifiedFieldName),
           fValue(std::move(value)),
           fIsValid(isValid),
           fCompositionProvenance(provenance)
      {
      }
   };

   std::vector<RComposerValue> fComposerValues;
   // Maps from the field name to all type alternatives for that field that have been added to the entry.
   std::unordered_map<std::string, std::vector<FieldIndex_t>> fFieldName2Index;

public:
   /////////////////////////////////////////////////////////////////////////////
   /// \brief Clear all fields from the entry.
   void Clear()
   {
      fComposerValues.clear();
      fFieldName2Index.clear();
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Set the validity of a field, i.e. whether it is possible to read its value in the current entry.
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   /// \param[in] isValid The new validity of the field.
   void SetFieldValidity(FieldIndex_t fieldIdx, bool isValid)
   {
      assert(fieldIdx < fComposerValues.size());
      fComposerValues[fieldIdx].fIsValid = isValid;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Check whether a field is valid for reading.
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   bool IsValidField(FieldIndex_t fieldIdx) const
   {
      assert(fieldIdx < fComposerValues.size());
      return fComposerValues[fieldIdx].fIsValid;
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
   /// \param[in] canonicalFieldName The name of the field in the entry, including its composition prefixes and
   /// parent field names, if applicable.
   ///
   /// \return A `std::optional` containing the field index if it was found.
   std::optional<FieldIndex_t> FindFieldIndex(std::string_view canonicalFieldName, std::string_view typeName) const;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Add a new field to the entry.
   ///
   /// \param[in] qualifiedFieldName Name of the field to add, including its parent field if applicable.
   /// \param[in] field Reference to the field to add, used to to create its corresponding RValue.
   /// \param[in] valuePtr Pointer to an object corresponding to the field's type to bind to its value. If this is a
   /// `nullptr`, a pointer will be created.
   /// \param[in] provenance Composition provenance of the field.
   ///
   /// \return The field index of the newly added field.
   FieldIndex_t AddField(const std::string &qualifiedFieldName, std::unique_ptr<ROOT::RFieldBase> field, void *valuePtr,
                         const RNTupleCompositionProvenance &provenance);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Update a field in the entry, preserving the value pointer.
   ///
   /// \param[in] fieldIdx Index of the field to update.
   /// \param[in] field The new field to use in the entry.
   void UpdateField(FieldIndex_t fieldIdx, std::unique_ptr<ROOT::RFieldBase> field);

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
      assert(fieldIdx < fComposerValues.size());
      return fComposerValues[fieldIdx].fValue;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the composition provenance of a field in the entry.
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   const RNTupleCompositionProvenance &GetCompositionProvenance(FieldIndex_t fieldIdx) const
   {
      assert(fieldIdx < fComposerValues.size());
      return fComposerValues[fieldIdx].fCompositionProvenance;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the name of a field in the entry, including its parent fields.
   ///
   /// \param[in] fieldIdx The index of the field in the entry.
   std::string GetQualifiedFieldName(FieldIndex_t fieldIdx) const
   {
      assert(fieldIdx < fComposerValues.size());
      return fComposerValues[fieldIdx].fQualifiedFieldName;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get all field indices of this entry.
   std::unordered_set<FieldIndex_t> GetFieldIndices() const;
};
} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif // ROOT_RNTupleComposerEntry
