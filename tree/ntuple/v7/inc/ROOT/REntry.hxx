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

namespace ROOT {
namespace Experimental {

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
   friend class RNTupleCollectionWriter;
   friend class RNTupleModel;
   friend class RNTupleReader;
   friend class RNTupleFillContext;

public:
   /// The field token identifies a top-level field in this entry. It can be used for fast indexing in REntry's
   /// methods, e.g. BindValue
   class RFieldToken {
      friend class REntry;
      std::size_t fIndex;     ///< the index in fValues that belongs to the top-level field
      std::uint64_t fModelId; ///< Safety check to prevent tokens from other models being used
      RFieldToken(std::size_t index, std::uint64_t modelId) : fIndex(index), fModelId(modelId) {}
   };

private:
   /// The entry must be linked to a specific model (or one if its clones), identified by a model ID
   std::uint64_t fModelId = 0;
   /// Corresponds to the top-level fields of the linked model
   std::vector<RFieldBase::RValue> fValues;

   // Creation of entries is done by the RNTupleModel class

   REntry() = default;
   explicit REntry(std::uint64_t modelId) : fModelId(modelId) {}

   void AddValue(RFieldBase::RValue &&value) { fValues.emplace_back(std::move(value)); }

   /// While building the entry, adds a new value to the list and return the value's shared pointer
   template <typename T, typename... ArgsT>
   std::shared_ptr<T> AddValue(RField<T> &field, ArgsT &&...args)
   {
      auto ptr = std::make_shared<T>(std::forward<ArgsT>(args)...);
      fValues.emplace_back(field.BindValue(ptr));
      return ptr;
   }

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
      if (fModelId != token.fModelId) {
         throw RException(R__FAIL("invalid token for this entry, "
                                  "make sure to use a token from the same model as this entry."));
      }
   }

   template <typename T>
   void EnsureMatchingType(RFieldToken token [[maybe_unused]]) const
   {
      if constexpr (!std::is_void_v<T>) {
         const auto &v = fValues[token.fIndex];
         if (v.GetField().GetTypeName() != RField<T>::TypeName()) {
            throw RException(R__FAIL("type mismatch for field " + v.GetField().GetFieldName() + ": " +
                                     v.GetField().GetTypeName() + " vs. " + RField<T>::TypeName()));
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

   /// The ordinal of the top-level field fieldName; can be used in other methods to address the corresponding value
   RFieldToken GetToken(std::string_view fieldName) const
   {
      auto it = std::find_if(fValues.begin(), fValues.end(),
         [&fieldName] (const RFieldBase::RValue &value) { return value.GetField().GetFieldName() == fieldName; });

      if ( it == fValues.end() ) {
         throw RException(R__FAIL("invalid field name: " + std::string(fieldName)));
      }
      return RFieldToken(std::distance(fValues.begin(), it), fModelId);
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

   std::uint64_t GetModelId() const { return fModelId; }

   ConstIterator_t begin() const { return fValues.cbegin(); }
   ConstIterator_t end() const { return fValues.cend(); }
};

} // namespace Experimental
} // namespace ROOT

#endif
