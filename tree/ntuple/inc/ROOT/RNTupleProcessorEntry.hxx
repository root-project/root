/// \file ROOT/RNTupleProcessorEntry.hxx
/// \ingroup NTuple
/// \author Florine de Geus <florine.de.geus@cern.ch>
/// \date 2025-08-05
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

namespace ROOT {
namespace Experimental {

class RNTupleSingleProcessor;
class RNTupleChainProcessor;
class RNTupleJoinProcessor;
template <typename T>
class RNTupleProcessorView;

namespace Internal {
class RNTupleProcessorValue {
   friend class Internal::RNTupleProcessorEntry;
   friend class ROOT::Experimental::RNTupleChainProcessor;
   friend class ROOT::Experimental::RNTupleJoinProcessor;
   template <typename T>
   friend class ROOT::Experimental::RNTupleProcessorView;

private:
   /// The field that created the RValue
   ROOT::RFieldBase *fField = nullptr;
   std::shared_ptr<void> fObjPtr;
   mutable std::atomic<const std::type_info *> fTypeInfo = nullptr;
   /// The qualified field name, prefixed with auxiliary processor names if applicable.
   std::string fProcessorFieldName;
   bool fIsValid;

   RNTupleProcessorValue(RFieldBase &field, std::shared_ptr<void> objPtr, std::string_view fieldName,
                         bool isValid = true)
      : fField(&field), fObjPtr(objPtr), fProcessorFieldName(fieldName), fIsValid(isValid)
   {
   }

   template <typename T>
   void EnsureMatchingType() const
   {
      if constexpr (!std::is_void_v<T>) {
         const std::type_info &ti = typeid(T);
         // Fast path: if we had a matching type before, try comparing the type_info's. This may still fail in case the
         // type has a suppressed template argument that may change the typeid.
         auto *cachedTypeInfo = fTypeInfo.load();
         if (cachedTypeInfo != nullptr && *cachedTypeInfo == ti) {
            return;
         }
         std::string renormalizedTypeName = ROOT::Internal::GetRenormalizedTypeName(ti);
         if (ROOT::Internal::IsMatchingFieldType(fField->GetTypeName(), renormalizedTypeName, ti)) {
            fTypeInfo.store(&ti);
            return;
         }
         throw RException(R__FAIL("type mismatch for field \"" + fField->GetFieldName() + "\": expected " +
                                  fField->GetTypeName() + ", got " + renormalizedTypeName));
      }
   }

   std::size_t Append() { return fField->Append(fObjPtr.get()); }

   void SetIsValid(bool isValid) { fIsValid = isValid; }

   /// Replace the field backing this value
   void ResetField(RFieldBase &newField) { fField = &newField; }

   template <typename T>
   std::shared_ptr<T> GetPtr() const
   {
      EnsureMatchingType<T>();
      return std::static_pointer_cast<T>(fObjPtr);
   }

   template <typename T>
   const T &GetRef() const
   {
      EnsureMatchingType<T>();
      return *static_cast<T *>(fObjPtr.get());
   }

public:
   RNTupleProcessorValue(const RNTupleProcessorValue &other)
      : fField(other.fField),
        fObjPtr(other.fObjPtr),
        fProcessorFieldName(other.fProcessorFieldName),
        fIsValid(other.fIsValid)
   {
   }
   RNTupleProcessorValue &operator=(const RNTupleProcessorValue &other)
   {
      fField = other.fField;
      fObjPtr = other.fObjPtr;
      // We could copy over the cached type info, or just start with a fresh state...
      fTypeInfo = nullptr;
      fProcessorFieldName = other.fProcessorFieldName;
      fIsValid = other.fIsValid;
      return *this;
   }
   RNTupleProcessorValue(RNTupleProcessorValue &&other)
      : fField(other.fField),
        fObjPtr(other.fObjPtr),
        fProcessorFieldName(other.fProcessorFieldName),
        fIsValid(other.fIsValid)
   {
   }
   RNTupleProcessorValue &operator=(RNTupleProcessorValue &&other)
   {
      fField = other.fField;
      fObjPtr = other.fObjPtr;
      // We could copy over the cached type info, or just start with a fresh state...
      fTypeInfo = nullptr;
      fProcessorFieldName = other.fProcessorFieldName;
      fIsValid = other.fIsValid;
      return *this;
   }
   ~RNTupleProcessorValue() = default;

   /// Read the field at the provided index into the fObjPtr.
   void Read(ROOT::NTupleSize_t globalIndex) { fField->Read(globalIndex, fObjPtr.get()); }

   /// Bind a new pointer to the value.
   void Bind(std::shared_ptr<void> objPtr) { fObjPtr = objPtr; }

   bool IsValid() const { return fIsValid; }

   const RFieldBase &GetField() const { return *fField; }

   /// Get the value's qualified field name, prefixed with auxiliary processor names if applicable.
   const std::string &GetProcessorFieldName() const { return fProcessorFieldName; }
};

// clang-format off
/**
\class ROOT::Experimental::Internal::RNTupleProcessorEntry
\ingroup NTuple
\brief Collection of values in an RNTupleProcessor, analogous to REntry, with checks and support for missing values.
*/
// clang-format on
class RNTupleProcessorEntry {
   friend class ROOT::Experimental::RNTupleSingleProcessor;
   friend class ROOT::Experimental::RNTupleChainProcessor;
   friend class ROOT::Experimental::RNTupleJoinProcessor;

private:
   std::unordered_map<std::string, std::unique_ptr<RFieldBase>> fFields;
   std::unordered_map<std::string, RNTupleProcessorValue> fValues;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get a value in the entry, or throw an exception if it does not exist.
   ///
   /// \param[in] fieldName Name of the field in the entry.
   ///
   /// \return Reference to the RNTupleProcessorValue corresponding to the field, if it exists.
   RNTupleProcessorValue &GetValueOrThrow(std::string_view fieldName)
   {
      auto value = fValues.find(std::string(fieldName));

      if (value == fValues.end()) {
         throw RException(R__FAIL("invalid field name: " + std::string(fieldName)));
      }

      return value->second;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \sa RNTupleProcessorEntry::GetValueOrThrow()
   const RNTupleProcessorValue &GetValueOrThrow(std::string_view fieldName) const
   {
      auto value = fValues.find(std::string(fieldName));

      if (value == fValues.end()) {
         throw RException(R__FAIL("invalid field name: " + std::string(fieldName)));
      }

      return value->second;
   }

   /// Clear the entire entry.
   void Clear()
   {
      fFields.clear();
      fValues.clear();
   }

public:
   using Iterator_t = decltype(fValues)::iterator;
   using ConstIterator_t = decltype(fValues)::const_iterator;

   RNTupleProcessorEntry() = default;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Add a new value to the entry or return its already existing value.
   ///
   /// \param[in] fieldName Name of the field to add or get the value for.
   /// \param[in] field Field to add the value for.
   ///
   /// \return Reference to the newly added or already existing RNTupleProcessorValue.
   RNTupleProcessorValue &AddOrGetValue(std::string_view fieldName, std::unique_ptr<RFieldBase> field)
   {
      auto obj =
         std::shared_ptr<void>(field->CreateObjectRawPtr(), ROOT::RFieldBase::RSharedPtrDeleter(field->GetDeleter()));
      auto [value, newlyAdded] =
         fValues.emplace(field->GetQualifiedFieldName(), RNTupleProcessorValue(*field, obj, fieldName));
      if (!newlyAdded)
         return value->second;
      fFields.emplace(field->GetQualifiedFieldName(), std::move(field));
      return value->second;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Update the field of a value in the entry.
   ///
   /// \param[in] fieldName Name of the field to update.
   /// \param[in] newField New field to use in its corresponding value.
   void UpdateField(std::string_view fieldName, std::unique_ptr<RFieldBase> newField)
   {
      auto &value = GetValueOrThrow(fieldName);
      value.ResetField(*newField);
      fFields[std::string(fieldName)] = std::move(newField);
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Bind a new pointer to a value in the entry.
   ///
   /// \param[in] fieldName Name of the field in the entry.
   /// \param[in] objPtr Pointer to the value to bind to the value.
   void BindValue(std::string_view fieldName, std::shared_ptr<void> objPtr) { GetValueOrThrow(fieldName).Bind(objPtr); }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Read the field values for the entry corresponding to the provided index.
   ///
   /// \param[in] index The entry number to read.
   void Read(ROOT::NTupleSize_t index)
   {
      for (auto &[_, value] : fValues) {
         if (value.IsValid())
            value.Read(index);
      }
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get a pointer to the value represented by the field specified.
   ///
   /// \tparam T The type of the pointer.
   ///
   /// \param[in] fieldName Name of the field to get the pointer for.
   template <typename T>
   std::shared_ptr<T> GetPtr(std::string_view fieldName) const
   {
      auto &value = GetValueOrThrow(fieldName);
      if (value.IsValid())
         return value.GetPtr<T>();

      return nullptr;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get a reference to the field of a value in the entry.
   ///
   /// \param[in] fieldName Name of the field in the entry.
   const ROOT::RFieldBase &GetField(std::string_view fieldName)
   {
      if (auto field = fFields.find(std::string(fieldName)); field != fFields.end())
         return *field->second;

      throw RException(R__FAIL("invalid field name: " + std::string(fieldName)));
   }

   Iterator_t begin() { return fValues.begin(); }
   Iterator_t end() { return fValues.end(); }

   ConstIterator_t begin() const { return fValues.begin(); }
   ConstIterator_t end() const { return fValues.end(); }
};
} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif // ROOT_RNTupleProcessorEntry
