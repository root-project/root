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

#include <memory>
#include <new>
#include <utility>

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
or deserialized into by tree reading.  Only fields can generate their corresponding tree values.
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
   /// Only RTreeFieldBase used fMappedElement
   RColumnElementBase fMappedElement;

   /// To be called only by RTreeFieldBase
   void SetMappedElement(const Detail::RColumnElementBase &element) {
      fMappedElement = element;
   }

public:
   RTreeValueBase(RTreeFieldBase *field) : fField(field), fRawPtr(nullptr) {}
   RTreeValueBase(RTreeFieldBase *field, void* rawPtr) : fField(field), fRawPtr(rawPtr) {}
   virtual ~RTreeValueBase() = default; // Prevent slicing

   void* GetRawPtr() const { return fRawPtr; }
   RTreeFieldBase* GetField() const { return fField; }
};

} // namespace Detail


// clang-format off
/**
\class ROOT::Experimental::RTreeValue
\ingroup Forest
\brief A type-safe and memory-managed front for RTreeValueBase

Used when types are available at compile time by RTreeModel::AddField()
*/
// clang-format on
template <typename T>
class RTreeValue : public Detail::RTreeValueBase {
private:
   std::shared_ptr<T> fValue;

public:
   template <typename... ArgsT>
   RTreeValue(Detail::RTreeFieldBase* field, T* where, ArgsT&&... args)
      : Detail::RTreeValueBase(field)
      , fValue(std::shared_ptr<T>(new (where) T(std::forward<ArgsT>(args)...)))
   {
      fRawPtr = fValue.get();
   }

   RTreeValue(bool /*captureTag*/, Detail::RTreeFieldBase* field, T* value)
      : Detail::RTreeValueBase(field), fValue(nullptr)
   {
      fRawPtr = value;
   }

   T* Get() const { return static_cast<T*>(fRawPtr); }
   std::shared_ptr<T> GetSharedPtr() const { return fValue; }
};

} // namespace Experimental
} // namespace ROOT

#endif
