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
   friend class RCollectionNTupleWriter;
   friend class RNTupleModel;
   friend class RNTupleReader;
   friend class RNTupleFillContext;

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

public:
   using ConstIterator_t = decltype(fValues)::const_iterator;

   REntry(const REntry &other) = delete;
   REntry &operator=(const REntry &other) = delete;
   REntry(REntry &&other) = default;
   REntry &operator=(REntry &&other) = default;
   ~REntry() = default;

   template <typename T>
   void BindValue(std::string_view fieldName, std::shared_ptr<T> objPtr)
   {
      for (auto &v : fValues) {
         if (v.GetField().GetFieldName() != fieldName)
            continue;

         if constexpr (!std::is_void_v<T>) {
            if (v.GetField().GetTypeName() != RField<T>::TypeName()) {
               throw RException(R__FAIL("type mismatch for field " + std::string(fieldName) + ": " +
                                        v.GetField().GetTypeName() + " vs. " + RField<T>::TypeName()));
            }
         }
         v.Bind(objPtr);
         return;
      }
      throw RException(R__FAIL("invalid field name: " + std::string(fieldName)));
   }

   template <typename T>
   void BindRawPtr(std::string_view fieldName, T *rawPtr)
   {
      BindValue(fieldName, std::shared_ptr<T>(rawPtr, [](T *) {}));
   }

   template <typename T>
   std::shared_ptr<T> GetPtr(std::string_view fieldName) const
   {
      for (auto &v : fValues) {
         if (v.GetField().GetFieldName() != fieldName)
            continue;

         if constexpr (std::is_void_v<T>)
            return v.GetPtr<void>();

         if (v.GetField().GetTypeName() != RField<T>::TypeName()) {
            throw RException(R__FAIL("type mismatch for field " + std::string(fieldName) + ": " +
                                     v.GetField().GetTypeName() + " vs. " + RField<T>::TypeName()));
         }
         return std::static_pointer_cast<T>(v.GetPtr<void>());
      }
      throw RException(R__FAIL("invalid field name: " + std::string(fieldName)));
   }

   std::uint64_t GetModelId() const { return fModelId; }

   ConstIterator_t begin() const { return fValues.cbegin(); }
   ConstIterator_t end() const { return fValues.cend(); }
};

} // namespace Experimental
} // namespace ROOT

#endif
