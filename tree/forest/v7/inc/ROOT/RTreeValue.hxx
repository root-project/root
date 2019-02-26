/// \file ROOT/RTreeValue.hxx
/// \ingroup Forest ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-09
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RTreeValue
#define ROOT7_RTreeValue

#include <ROOT/RColumnElement.hxx>

#include <new>

namespace ROOT {
namespace Experimental {

namespace Detail {

class RTreeFieldBase;

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
class RTreeValueBase {
   friend class RTreeFieldBase;

protected:
   /// Every value is connected to a field of the corresponding type that has created the value.
   RTreeFieldBase* fField;
   /// The memory location containing (constructed) data of a certain C++ type
   void* fRawPtr;

   /// For simple types, the mapped element drills through the layers from the C++ data representation
   /// to the primitive columns.  Otherwise, using fMappedElements is undefined.
   /// Only RTreeFieldBase uses fMappedElement
   RColumnElementBase fMappedElement;

public:
   RTreeValueBase() : fField(nullptr), fRawPtr(nullptr) {}
   RTreeValueBase(RTreeFieldBase* field, void* rawPtr) : fField(field), fRawPtr(rawPtr) {}
   RTreeValueBase(RTreeFieldBase* field, void* rawPtr, const RColumnElementBase& mappedElement)
      : fField(field), fRawPtr(rawPtr), fMappedElement(mappedElement) {}

   void* GetRawPtr() const { return fRawPtr; }
   RTreeFieldBase* GetField() const { return fField; }
};

} // namespace Detail


// clang-format off
/**
\class ROOT::Experimental::RTreeValue
\ingroup Forest
\brief A type-safe front for RTreeValueBase

Used when types are available at compile time by RTreeModel::AddField()
*/
// clang-format on
template <typename T>
class RTreeValue : public Detail::RTreeValueBase {
public:
   RTreeValue() : Detail::RTreeValueBase(nullptr, nullptr) {}
   RTreeValue(const Detail::RTreeValueBase &other) : Detail::RTreeValueBase(other) {}
   template <typename... ArgsT>
   RTreeValue(Detail::RTreeFieldBase* field, T* where, ArgsT&&... args) : Detail::RTreeValueBase(field, where)
   {
      new (where) T(std::forward<ArgsT>(args)...);
   }
   template <typename... ArgsT>
   RTreeValue(const Detail::RColumnElementBase& elem, Detail::RTreeFieldBase* field, T* where, ArgsT&&... args)
      : Detail::RTreeValueBase(field, where, elem)
   {
      new (where) T(std::forward<ArgsT>(args)...);
   }
   template <typename... ArgsT>
   RTreeValue(bool /*captureTag*/, Detail::RTreeFieldBase* field, T* value) : Detail::RTreeValueBase(field, value) {}
   template <typename... ArgsT>
   RTreeValue(bool /*captureTag*/, const Detail::RColumnElementBase& elem, Detail::RTreeFieldBase* field, T* value)
      : Detail::RTreeValueBase(field, value, elem) {}

   T* Get() const { return static_cast<T*>(fRawPtr); }
};

} // namespace Experimental
} // namespace ROOT

#endif
