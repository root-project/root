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
#include <ROOT/RFieldToken.hxx>

#include <cassert>
#include <cstddef>
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
private:
   /// The proto model contains all available fields. The entry itself will only contain a subset of these fields (i.e.,
   /// the ones actually requested by the user).
   ROOT::RNTupleModel *fProtoModel;
   /// Corresponds to the fields of the proto model
   std::vector<ROOT::RFieldBase::RValue> fValues;
   /// Marks whether fields are valid for reading
   std::vector<bool> fFieldValidities;
   /// For fast lookup of token IDs given a (sub)field name present in the entry
   std::unordered_map<std::string, std::size_t> fFieldName2Token;
   /// When the entry is frozen, no fields can be added to it anymore
   bool fIsFrozen = false;

public:
   RNTupleProcessorEntry(ROOT::RNTupleModel &protoModel) : fProtoModel(&protoModel) {}

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Set whether the entry is frozen.
   ///
   /// When the entry is frozen, no fields can be added to it anymore.
   void SetFrozen(bool frozen) { fIsFrozen = frozen; }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Check whether the entry is frozen.
   bool IsFrozen() const { return fIsFrozen; }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Set the validity of a field, i.e. whether it is possible to read its value in the current entry.
   ///
   /// \param[in] token The token representing the field in the entry.
   /// \param[in] isValid The new validity of the field.
   void SetFieldValidity(ROOT::RFieldToken token, bool isValid)
   {
      assert(token.fSchemaId == fProtoModel->fSchemaId && token.fIndex < fValues.size());
      fFieldValidities[token.fIndex] = isValid;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Set the validity of a field, i.e. whether it is possible to read its value in the current entry.
   ///
   /// \param[in] fieldName The name of the field in the entry.
   /// \param[in] isValid The new validity of the field.
   void SetFieldValidity(std::string_view fieldName, bool isValid) { SetFieldValidity(GetToken(fieldName), isValid); }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Check whether a field is valid for reading.
   ///
   /// \param[in] token The token representing the field in the entry.
   bool FieldIsValid(ROOT::RFieldToken token) const { return fFieldValidities[token.fIndex]; }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Check whether a field is valid for reading.
   ///
   /// \param[in] fieldName The name of the field in the entry.
   bool FieldIsValid(std::string_view fieldName) const { return FieldIsValid(GetToken(fieldName)); }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Find the name of a field from its token
   ///
   /// \param[in] token The token representing the field in the entry.
   ///
   /// \warning This function has linear complexity, only use it for more helpful error messages!
   const std::string &FindFieldName(ROOT::RFieldToken token) const
   {
      assert(token.fSchemaId == fProtoModel->fSchemaId && token.fIndex < fValues.size());
      for (const auto &[fieldName, index] : fFieldName2Token) {
         if (index == token.fIndex) {
            return fieldName;
         }
      }
      // Should never happen, but avoid compiler warning about "returning reference to local temporary object".
      static const std::string empty = "";
      return empty;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Find the token representing the provided field in the entry.
   ///
   /// \param[in] fieldName The name of the field in the entry.
   ///
   /// \return A `std::optional` containing the token if it was found.
   std::optional<ROOT::RFieldToken> FindToken(std::string_view fieldName) const
   {
      auto it = fFieldName2Token.find(std::string(fieldName));
      if (it == fFieldName2Token.end()) {
         return std::nullopt;
      }
      return ROOT::RFieldToken(it->second, fProtoModel->GetSchemaId());
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the token representing the provided field in the entry.
   ///
   /// \param[in] fieldName The name of the field in the entry.
   ///
   /// \return The token corresponding to the provided fieldName.
   ///
   /// \throws An RException in case the token could not be retrieved from the entry.
   ROOT::RFieldToken GetToken(std::string_view fieldName) const
   {
      if (auto token = FindToken(fieldName))
         return *token;

      throw RException(R__FAIL("invalid field name: " + std::string(fieldName)));
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Add a new field to the entry or update the value pointer for an existing field.
   ///
   /// \param[in] fieldName The name of the field to add or update.
   /// \param[in] valuePtr Pointer to the value for the field to add or update.
   ///
   /// \return `true` if the field was added successfully or if was already present and its value has been updated,
   /// `false` otherwise.
   ///
   /// A field will only be updated if `valuePtr` is not a `nullptr`.
   bool AddOrUpdateField(std::string_view fieldName, std::shared_ptr<void> valuePtr)
   {
      if (auto token = FindToken(fieldName)) {
         if (valuePtr)
            BindValue(*token, valuePtr);
         return true;
      }
      ROOT::RFieldBase *field = fProtoModel->FindField(fieldName);
      if (field) {
         auto value = field->CreateValue();
         if (valuePtr)
            value.Bind(valuePtr);
         fFieldName2Token[value.GetField().GetQualifiedFieldName()] = fValues.size();
         fValues.emplace_back(std::move(value));
         fFieldValidities.push_back(true);

         return true;
      }

      return false;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Bind a new value pointer to a field in the entry.
   ///
   /// \param[in] fieldName The name of the field in the entry.
   /// \param[in] objPtr Pointer to the value to bind to the field.
   void BindValue(ROOT::RFieldToken token, std::shared_ptr<void> objPtr) { fValues[token.fIndex].Bind(objPtr); }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Bind a new value pointer to a field in the entry.
   ///
   /// \param[in] fieldName The name of the field in the entry.
   /// \param[in] objPtr Pointer to the value to bind to the field.
   void BindValue(std::string_view fieldName, std::shared_ptr<void> objPtr) { BindValue(GetToken(fieldName), objPtr); }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Read the field values for the entry corresponding to the provided index.
   ///
   /// \param[in] index The entry number to read.
   void Read(ROOT::NTupleSize_t index)
   {
      for (unsigned i = 0; i < fValues.size(); i++) {
         if (fFieldValidities[i])
            fValues[i].Read(index);
      }
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get a pointer to the value for the field represented by the provided token.
   ///
   /// \tparam T The type of the pointer
   ///
   /// \param[in] token The token representing the field in the entry.
   template <typename T>
   std::shared_ptr<T> GetPtr(ROOT::RFieldToken token) const
   {
      assert(token.fSchemaId == fProtoModel->fSchemaId && token.fIndex < fValues.size());

      if (fFieldValidities[token.fIndex])
         return fValues[token.fIndex].GetPtr<T>();

      return nullptr;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get a pointer to the value for the field represented by the provided token.
   ///
   /// \tparam T The type of the pointer
   ///
   /// \param[in] fieldName The name of the field in the entry.
   template <typename T>
   std::shared_ptr<T> GetPtr(std::string_view fieldName) const
   {
      return GetPtr<T>(GetToken(fieldName));
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get a reference to a field in the entry.
   ///
   /// \param[in] token The token representing the field in the entry.
   const ROOT::RFieldBase &GetField(ROOT::RFieldToken token)
   {
      assert(token.fSchemaId == fProtoModel->fSchemaId && token.fIndex < fValues.size());
      return fValues[token.fIndex].GetField();
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get a reference to a field in the entry.
   ///
   /// \param[in] fieldName The name of the field in the entry.
   const ROOT::RFieldBase &GetField(std::string_view fieldName) { return GetField(GetToken(fieldName)); }

   ROOT::REntry::ConstIterator_t begin() const { return fValues.begin(); }
   ROOT::REntry::ConstIterator_t end() const { return fValues.end(); }
};
} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif // ROOT_RNTupleProcessorEntry
