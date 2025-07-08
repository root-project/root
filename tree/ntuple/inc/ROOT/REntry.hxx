/// \file ROOT/REntry.hxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-07-19

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_REntry
#define ROOT_REntry

#include <ROOT/RError.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RFieldToken.hxx>
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

class RNTupleReader;

namespace Experimental {
class RNTupleFillContext;
class RNTupleProcessor;
class RNTupleSingleProcessor;
class RNTupleChainProcessor;
class RNTupleJoinProcessor;
} // namespace Experimental

// clang-format off
/**
\class ROOT::REntry
\ingroup NTuple
\brief The REntry is a collection of values in an RNTuple corresponding to a complete row in the data set.

The entry provides a memory-managed binder for a set of values read from fields in an RNTuple. The memory locations that are associated
with values are managed through shared pointers.
*/
// clang-format on
class REntry {
   friend class RNTupleModel;
   friend class RNTupleReader;
   friend class Experimental::RNTupleFillContext;
   friend class Experimental::RNTupleProcessor;
   friend class Experimental::RNTupleSingleProcessor;
   friend class Experimental::RNTupleChainProcessor;
   friend class Experimental::RNTupleJoinProcessor;

private:
   /// The entry must be linked to a specific model, identified by a model ID
   std::uint64_t fModelId = 0;
   /// The entry and its tokens are also linked to a specific schema, identified by a schema ID
   std::uint64_t fSchemaId = 0;
   /// Corresponds to the fields of the linked model
   std::vector<ROOT::RFieldBase::RValue> fValues;
   /// For fast lookup of token IDs given a (sub)field name present in the entry
   std::unordered_map<std::string, std::size_t> fFieldName2Token;
   /// To ensure that the entry is standalone, a copy of all field types
   std::vector<std::string> fFieldTypes;

   /// Creation of entries can be done by the RNTupleModel, the RNTupleReader, or the RNTupleWriter.
   REntry() = default;
   explicit REntry(std::uint64_t modelId, std::uint64_t schemaId) : fModelId(modelId), fSchemaId(schemaId) {}

   void AddValue(ROOT::RFieldBase::RValue &&value)
   {
      fFieldName2Token[value.GetField().GetQualifiedFieldName()] = fValues.size();
      fFieldTypes.push_back(value.GetField().GetTypeName());
      fValues.emplace_back(std::move(value));
   }

   /// While building the entry, adds a new value for the field and returns the value's shared pointer
   template <typename T>
   std::shared_ptr<T> AddValue(ROOT::RField<T> &field)
   {
      fFieldName2Token[field.GetQualifiedFieldName()] = fValues.size();
      fFieldTypes.push_back(field.GetTypeName());
      auto value = field.CreateValue();
      fValues.emplace_back(value);
      return value.template GetPtr<T>();
   }

   void Read(ROOT::NTupleSize_t index)
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

   void EnsureMatchingModel(ROOT::RFieldToken token) const
   {
      if (fSchemaId != token.fSchemaId) {
         throw RException(R__FAIL("invalid token for this entry, "
                                  "make sure to use a token from a model with the same schema as this entry."));
      }
   }

   /// This function has linear complexity, only use it for more helpful error messages!
   const std::string &FindFieldName(ROOT::RFieldToken token) const
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
   void EnsureMatchingType(ROOT::RFieldToken token [[maybe_unused]]) const
   {
      if constexpr (!std::is_void_v<T>) {
         if (!Internal::IsMatchingFieldType<T>(fFieldTypes[token.fIndex])) {
            throw RException(R__FAIL("type mismatch for field " + FindFieldName(token) + ": " +
                                     fFieldTypes[token.fIndex] + " vs. " + ROOT::RField<T>::TypeName()));
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
   ROOT::RFieldToken GetToken(std::string_view fieldName) const
   {
      auto it = fFieldName2Token.find(std::string(fieldName));
      if (it == fFieldName2Token.end()) {
         throw RException(R__FAIL("invalid field name: " + std::string(fieldName)));
      }
      return ROOT::RFieldToken(it->second, fSchemaId);
   }

   /// Create a new value for the field referenced by `token`.
   void EmplaceNewValue(ROOT::RFieldToken token)
   {
      EnsureMatchingModel(token);
      fValues[token.fIndex].EmplaceNew();
   }

   /// Create a new value for the field referenced by its name.
   void EmplaceNewValue(std::string_view fieldName) { EmplaceNewValue(GetToken(fieldName)); }

   /// Bind the value for the field, referenced by `token`, to `objPtr`.
   ///
   /// \sa BindValue(std::string_view, std::shared_ptr<T>)
   template <typename T>
   void BindValue(ROOT::RFieldToken token, std::shared_ptr<T> objPtr)
   {
      EnsureMatchingModel(token);
      EnsureMatchingType<T>(token);
      fValues[token.fIndex].Bind(objPtr);
   }

   /// Bind the value for the field, referenced by its name, to `objPtr`.
   ///
   /// Ownership is shared with the caller and the object will be kept alive until it is replaced (by a call to
   /// EmplaceNewValue, BindValue, or BindRawPtr) or the entry is destructed.
   ///
   /// **Note**: if `T = void`, type checks are disabled. It is the caller's responsibility to match the field and
   /// object types.
   template <typename T>
   void BindValue(std::string_view fieldName, std::shared_ptr<T> objPtr)
   {
      BindValue<T>(GetToken(fieldName), objPtr);
   }

   /// Bind the value for the field, referenced by `token`, to `rawPtr`.
   ///
   /// \sa BindRawPtr(std::string_view, T *)
   template <typename T>
   void BindRawPtr(ROOT::RFieldToken token, T *rawPtr)
   {
      EnsureMatchingModel(token);
      EnsureMatchingType<T>(token);
      fValues[token.fIndex].BindRawPtr(rawPtr);
   }

   /// Bind the value for the field, referenced by its name, to `rawPtr`.
   ///
   /// The caller retains ownership of the object and must ensure it is kept alive when reading or writing using the
   /// entry.
   ///
   /// **Note**: if `T = void`, type checks are disabled. It is the caller's responsibility to match the field and
   /// object types.
   template <typename T>
   void BindRawPtr(std::string_view fieldName, T *rawPtr)
   {
      BindRawPtr<void>(GetToken(fieldName), rawPtr);
   }

   /// Get the (typed) pointer to the value for the field referenced by `token`.
   ///
   /// \sa GetPtr(std::string_view)
   template <typename T>
   std::shared_ptr<T> GetPtr(ROOT::RFieldToken token) const
   {
      EnsureMatchingModel(token);
      EnsureMatchingType<T>(token);
      return std::static_pointer_cast<T>(fValues[token.fIndex].GetPtr<void>());
   }

   /// Get the (typed) pointer to the value for the field referenced by `token`.
   ///
   /// Ownership is shared and the caller can continue to use the object after the entry is destructed.
   ///
   /// **Note**: if `T = void`, type checks are disabled. It is the caller's responsibility to use the returned pointer
   /// according to the field type.
   template <typename T>
   std::shared_ptr<T> GetPtr(std::string_view fieldName) const
   {
      return GetPtr<T>(GetToken(fieldName));
   }

   const std::string &GetTypeName(ROOT::RFieldToken token) const
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

} // namespace ROOT

#endif
