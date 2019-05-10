/// \file ROOT/RFieldValue.hxx
/// \ingroup Forest ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-09
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RFieldValue
#define ROOT7_RFieldValue

#include <ROOT/RColumnElement.hxx>

#include <new>

namespace ROOT {
namespace Experimental {

namespace Detail {

class RFieldBase;

// clang-format off
/**
\class ROOT::Experimental::TreeValueBase
\ingroup Forest
\brief Represents transient storage of simple or complex C++ values.

The data carried by the tree value is used by the computational code and it is supposed to be serialized on Fill
or deserialized into by tree reading.  Only fields can generate their corresponding tree values. This class is a mere
wrapper around the memory location, it does not own it.  Memory ownership is managed through the RForestEntry.
*/
// clang-format on
class RFieldValueBase {
   friend class RFieldBase;

protected:
   /// Every value is connected to a field of the corresponding type that has created the value.
   RFieldBase* fField;
   /// The memory location containing (constructed) data of a certain C++ type
   void* fRawPtr;

   /// For simple types, the mapped element drills through the layers from the C++ data representation
   /// to the primitive columns.  Otherwise, using fMappedElements is undefined.
   /// Only RFieldBase uses fMappedElement
   RColumnElementBase fMappedElement;

public:
   RFieldValueBase() : fField(nullptr), fRawPtr(nullptr) {}
   RFieldValueBase(RFieldBase* field, void* rawPtr) : fField(field), fRawPtr(rawPtr) {}
   RFieldValueBase(RFieldBase* field, void* rawPtr, const RColumnElementBase& mappedElement)
      : fField(field), fRawPtr(rawPtr), fMappedElement(mappedElement) {}

   void* GetRawPtr() const { return fRawPtr; }
   RFieldBase* GetField() const { return fField; }
};

} // namespace Detail


// clang-format off
/**
\class ROOT::Experimental::RFieldValue
\ingroup Forest
\brief A type-safe front for RFieldValueBase

Used when types are available at compile time by RForestModel::AddField()
*/
// clang-format on
template <typename T>
class RFieldValue : public Detail::RFieldValueBase {
public:
   RFieldValue() : Detail::RFieldValueBase(nullptr, nullptr) {}
   RFieldValue(const Detail::RFieldValueBase &other) : Detail::RFieldValueBase(other) {}
   template <typename... ArgsT>
   RFieldValue(Detail::RFieldBase* field, T* where, ArgsT&&... args) : Detail::RFieldValueBase(field, where)
   {
      new (where) T(std::forward<ArgsT>(args)...);
   }
   template <typename... ArgsT>
   RFieldValue(const Detail::RColumnElementBase& elem, Detail::RFieldBase* field, T* where, ArgsT&&... args)
      : Detail::RFieldValueBase(field, where, elem)
   {
      new (where) T(std::forward<ArgsT>(args)...);
   }
   RFieldValue(bool /*captureTag*/, Detail::RFieldBase* field, T* value) : Detail::RFieldValueBase(field, value) {}
   RFieldValue(bool /*captureTag*/, const Detail::RColumnElementBase& elem, Detail::RFieldBase* field, T* value)
      : Detail::RFieldValueBase(field, value, elem) {}

   T* Get() const { return static_cast<T*>(fRawPtr); }
};

} // namespace Experimental
} // namespace ROOT

#endif
