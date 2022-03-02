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
#include <ROOT/RFieldValue.hxx>
#include <ROOT/RStringView.hxx>

#include <TError.h>

#include <memory>
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
   friend class RNTupleModel;

   /// The entry must be linked to a specific model (or one if its clones), identified by a model ID
   std::uint64_t fModelId = 0;
   /// Corresponds to the top-level fields of the linked model
   std::vector<Detail::RFieldValue> fValues;
   /// The objects involed in serialization and deserialization might be used long after the entry is gone:
   /// hence the shared pointer
   std::vector<std::shared_ptr<void>> fValuePtrs;
   /// Points into fValues and indicates the values that are owned by the entry and need to be destructed
   std::vector<std::size_t> fManagedValues;

   // Creation of entries is done by the RNTupleModel class

   REntry() = default;
   explicit REntry(std::uint64_t modelId) : fModelId(modelId) {}

   /// Adds a value whose storage is managed by the entry
   void AddValue(const Detail::RFieldValue& value);

   /// Adds a value whose storage is _not_ managed by the entry
   void CaptureValue(const Detail::RFieldValue& value);

   /// While building the entry, adds a new value to the list and return the value's shared pointer
   template<typename T, typename... ArgsT>
   std::shared_ptr<T> AddValue(RField<T>* field, ArgsT&&... args) {
      auto ptr = std::make_shared<T>(std::forward<ArgsT>(args)...);
      fValues.emplace_back(Detail::RFieldValue(field->CaptureValue(ptr.get())));
      fValuePtrs.emplace_back(ptr);
      return ptr;
   }

public:
   using Iterator_t = decltype(fValues)::iterator;

   REntry(const REntry &other) = delete;
   REntry &operator=(const REntry &other) = delete;
   REntry(REntry &&other) = default;
   REntry &operator=(REntry &&other) = default;
   ~REntry();

   void CaptureValueUnsafe(std::string_view fieldName, void *where);

   Detail::RFieldValue GetValue(std::string_view fieldName) const
   {
      for (auto& v : fValues) {
         if (v.GetField()->GetName() == fieldName)
            return v;
      }
      throw RException(R__FAIL("invalid field name: " + std::string(fieldName)));
   }

   template <typename T>
   T *Get(std::string_view fieldName) const
   {
      for (auto& v : fValues) {
         if (v.GetField()->GetName() == fieldName) {
            R__ASSERT(v.GetField()->GetType() == RField<T>::TypeName());
            return v.Get<T>();
         }
      }
      throw RException(R__FAIL("invalid field name: " + std::string(fieldName)));
   }

   std::uint64_t GetModelId() const { return fModelId; }

   Iterator_t begin() { return fValues.begin(); }
   Iterator_t end() { return fValues.end(); }
};

} // namespace Experimental
} // namespace ROOT

#endif
