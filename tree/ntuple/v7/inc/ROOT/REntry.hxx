/// \file ROOT/REntry.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-07-19
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_REntry
#define ROOT7_REntry

#include <ROOT/RError.hxx>
#include <ROOT/RField.hxx>
#include <string_view>

#include <TError.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>
#include <unordered_map>

namespace ROOT {
namespace Experimental {

class RNTupleProcessor;
class RNTupleSingleProcessor;
class RNTupleChainProcessor;
class RNTupleJoinProcessor;

// clang-format off
/**
\class ROOT::Experimental::REntry
\ingroup NTuple
\brief The REntry is a collection of values in an ntuple corresponding to a complete row in the data set

The entry provides a memory-managed binder for a set of values. Through shared pointers, the memory locations
that are associated to values are managed.
*/
// clang-format on
class REntry {
   friend class RNTupleModel;
   friend class RNTupleReader;
   friend class RNTupleFillContext;
   friend class RNTupleProcessor;
   friend class RNTupleSingleProcessor;
   friend class RNTupleChainProcessor;
   friend class RNTupleJoinProcessor;

public:
   /// The field token identifies a (sub)field in this entry. It can be used for fast indexing in REntry's methods, e.g.
   /// BindValue. The field token can also be created by the model.
   class RFieldToken {
      friend class REntry;
      friend class RNTupleModel;

      std::size_t fIndex = 0;                      ///< The index in fValues that belongs to the field
      std::uint64_t fSchemaId = std::uint64_t(-1); ///< Safety check to prevent tokens from other models being used
      RFieldToken(std::size_t index, std::uint64_t schemaId) : fIndex(index), fSchemaId(schemaId) {}

   public:
      RFieldToken() = default; // The default constructed token cannot be used by any entry
   };

private:
   /// The entry must be linked to a specific model, identified by a model ID
   std::uint64_t fModelId = 0;
   /// The entry and its tokens are also linked to a specific schema, identified by a schema ID
   std::uint64_t fSchemaId = 0;
   /// Corresponds to the fields of the linked model
   std::vector<RFieldBase::RValue> fValues;
   /// For fast lookup of token IDs given a (sub)field name present in the entry
   std::unordered_map<std::string, std::size_t> fFieldName2Token;
   /// To ensure that the entry is standalone, a copy of all field types
   std::vector<std::string> fFieldTypes;

   // Creation of entries is done by the RNTupleModel class

   REntry() = default;
   explicit REntry(std::uint64_t modelId, std::uint64_t schemaId) : fModelId(modelId), fSchemaId(schemaId) {}

   void AddValue(RFieldBase::RValue &&value)
   {
      fFieldName2Token[value.GetField().GetQualifiedFieldName()] = fValues.size();
      fFieldTypes.push_back(value.GetField().GetTypeName());
      fValues.emplace_back(std::move(value));
   }

   /// While building the entry, adds a new value to the list and return the value's shared pointer
   template <typename T>
   std::shared_ptr<T> AddValue(RField<T> &field)
   {
      fFieldName2Token[field.GetQualifiedFieldName()] = fValues.size();
      fFieldTypes.push_back(field.GetTypeName());
      auto value = field.CreateValue();
      fValues.emplace_back(value);
      return value.template GetPtr<T>();
   }

   /// Update the RValue for a field in the entry. To be used when its underlying RFieldBase changes, which typically
   /// happens when page source the field values are read from changes.
   void UpdateValue(RFieldToken token, RFieldBase::RValue &&value) { std::swap(fValues.at(token.fIndex), value); }
   void UpdateValue(RFieldToken token, RFieldBase::RValue &value) { std::swap(fValues.at(token.fIndex), value); }

   /// Return the RValue currently bound to the provided field.
   RFieldBase::RValue &GetValue(RFieldToken token) { return fValues.at(token.fIndex); }
   RFieldBase::RValue &GetValue(std::string_view fieldName) { return GetValue(GetToken(fieldName)); }

   void Read(NTupleSize_t index)
   {
      for (auto &v : fValues) {
         v.Read(index);
      }
   }

   std::size_t Append()
   {
      std::size_t bytesWritten = 0;
      for (auto &v : fValues) {
         bytesWritten += v.Append();
      }
      return bytesWritten;
   }

   void EnsureMatchingModel(RFieldToken token) const
   {
      if (fSchemaId != token.fSchemaId) {
         throw RException(R__FAIL("invalid token for this entry, "
                                  "make sure to use a token from a model with the same schema as this entry."));
      }
   }

   /// This function has linear complexity, only use for more helpful error messages!
   const std::string &FindFieldName(RFieldToken token) const
   {
      for (const auto &[fieldName, index] : fFieldName2Token) {
         if (index == token.fIndex) {
            return fieldName;
         }
      }
      // Should never happen, but avoid compiler warning about "returning reference to local temporary object".
      static const std::string empty = "";
      return empty;
   }

   template <typename T>
   void EnsureMatchingType(RFieldToken token [[maybe_unused]]) const
   {
      if constexpr (!std::is_void_v<T>) {
         if (fFieldTypes[token.fIndex] != RField<T>::TypeName()) {
            throw RException(R__FAIL("type mismatch for field " + FindFieldName(token) + ": " +
                                     fFieldTypes[token.fIndex] + " vs. " + RField<T>::TypeName()));
         }
      }
   }

public:
   using ConstIterator_t = decltype(fValues)::const_iterator;

   REntry(const REntry &other) = delete;
   REntry &operator=(const REntry &other) = delete;
   REntry(REntry &&other) = default;
   REntry &operator=(REntry &&other) = default;
   ~REntry() = default;

   /// The ordinal of the (sub)field fieldName; can be used in other methods to address the corresponding value
   RFieldToken GetToken(std::string_view fieldName) const
   {
      auto it = fFieldName2Token.find(std::string(fieldName));
      if (it == fFieldName2Token.end()) {
         throw RException(R__FAIL("invalid field name: " + std::string(fieldName)));
      }
      return RFieldToken(it->second, fSchemaId);
   }

   void EmplaceNewValue(RFieldToken token)
   {
      EnsureMatchingModel(token);
      fValues[token.fIndex].EmplaceNew();
   }

   void EmplaceNewValue(std::string_view fieldName) { EmplaceNewValue(GetToken(fieldName)); }

   template <typename T>
   void BindValue(RFieldToken token, std::shared_ptr<T> objPtr)
   {
      EnsureMatchingModel(token);
      EnsureMatchingType<T>(token);
      fValues[token.fIndex].Bind(objPtr);
   }

   template <typename T>
   void BindValue(std::string_view fieldName, std::shared_ptr<T> objPtr)
   {
      BindValue<T>(GetToken(fieldName), objPtr);
   }

   template <typename T>
   void BindRawPtr(RFieldToken token, T *rawPtr)
   {
      EnsureMatchingModel(token);
      EnsureMatchingType<T>(token);
      fValues[token.fIndex].BindRawPtr(rawPtr);
   }

   template <typename T>
   void BindRawPtr(std::string_view fieldName, T *rawPtr)
   {
      BindRawPtr<void>(GetToken(fieldName), rawPtr);
   }

   template <typename T>
   std::shared_ptr<T> GetPtr(RFieldToken token) const
   {
      EnsureMatchingModel(token);
      EnsureMatchingType<T>(token);
      return std::static_pointer_cast<T>(fValues[token.fIndex].GetPtr<void>());
   }

   template <typename T>
   std::shared_ptr<T> GetPtr(std::string_view fieldName) const
   {
      return GetPtr<T>(GetToken(fieldName));
   }

   const std::string &GetTypeName(RFieldToken token) const
   {
      EnsureMatchingModel(token);
      return fFieldTypes[token.fIndex];
   }

   const std::string &GetTypeName(std::string_view fieldName) const { return GetTypeName(GetToken(fieldName)); }

   std::uint64_t GetModelId() const { return fModelId; }
   std::uint64_t GetSchemaId() const { return fSchemaId; }

   ConstIterator_t begin() const { return fValues.cbegin(); }
   ConstIterator_t end() const { return fValues.cend(); }
};

} // namespace Experimental
} // namespace ROOT

#endif
