/// \file ROOT/RFieldValue.hxx
/// \ingroup NTuple ROOT7
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
\class ROOT::Experimental::RFieldValue
\ingroup NTuple
\brief Represents transient storage of simple or complex C++ values.

The data carried by the value is used by the computational code and it is supposed to be serialized on Fill
or deserialized into by reading.  Only fields can generate their corresponding values. This class is a mere
wrapper around the memory location, it does not own it.  Memory ownership is managed through the REntry.
*/
// clang-format on
class RFieldValue {
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
   RFieldValue() : fField(nullptr), fRawPtr(nullptr) {}

   // Constructors that wrap around existing objects; prefixed with a bool in order to help the constructor overload
   // selection.
   RFieldValue(bool /*captureTag*/, Detail::RFieldBase* field, void* value) : fField(field), fRawPtr(value) {}
   RFieldValue(bool /*captureTag*/, const Detail::RColumnElementBase& elem, Detail::RFieldBase* field, void* value)
      : fField(field), fRawPtr(value), fMappedElement(elem) {}

   // Typed constructors that initialize the given memory location
   template <typename T, typename... ArgsT>
   RFieldValue(RFieldBase* field, T* where, ArgsT&&... args) : fField(field), fRawPtr(where)
   {
      new (where) T(std::forward<ArgsT>(args)...);
   }

   template <typename T, typename... ArgsT>
   RFieldValue(const Detail::RColumnElementBase& elem, Detail::RFieldBase* field, T* where, ArgsT&&... args)
      : fField(field), fRawPtr(where), fMappedElement(elem)
   {
      new (where) T(std::forward<ArgsT>(args)...);
   }

   template <typename T>
   T* Get() const { return static_cast<T*>(fRawPtr); }

   void* GetRawPtr() const { return fRawPtr; }
   RFieldBase* GetField() const { return fField; }
};

} // namespace Detail
} // namespace Experimental
} // namespace ROOT

#endif
