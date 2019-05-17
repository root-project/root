/// \file ROOT/RForestEntry.hxx
/// \ingroup Forest ROOT7
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

#ifndef ROOT7_RForestEntry
#define ROOT7_RForestEntry

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
\class ROOT::Experimental::RForestEntry
\ingroup Forest
\brief The RForestEntry is a collection of values in a forest corresponding to a complete row in the data set

The entry provides a memory-managed binder for a set of values. Through shared pointers, the memory locations
that are associated to values are managed.
*/
// clang-format on
class RForestEntry {
   std::vector<Detail::RFieldValue> fValues;
   /// The objects involed in serialization and deserialization might be used long after the entry is gone:
   /// hence the shared pointer
   std::vector<std::shared_ptr<void>> fValuePtrs;
   /// Points into fValues and indicates the values that are owned by the entry and need to be destructed
   std::vector<std::size_t> fManagedValues;

public:
   RForestEntry() = default;
   RForestEntry(const RForestEntry& other) = delete;
   RForestEntry& operator=(const RForestEntry& other) = delete;
   ~RForestEntry();

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

   Detail::RFieldValue GetValue(std::string_view fieldName) {
      for (auto& v : fValues) {
         if (v.GetField()->GetName() == fieldName)
            return v;
      }
      return Detail::RFieldValue();
   }

   template<typename T>
   T* Get(std::string_view fieldName) {
      for (auto& v : fValues) {
         if (v.GetField()->GetName() == fieldName) {
            R__ASSERT(v.GetField()->GetType() == RField<T>::MyTypeName());
            return static_cast<T*>(v.GetRawPtr());
         }
      }
      return nullptr;
   }

   decltype(fValues)::iterator begin() { return fValues.begin(); }
   decltype(fValues)::iterator end() { return fValues.end(); }
};

} // namespace Experimental
} // namespace ROOT

#endif
