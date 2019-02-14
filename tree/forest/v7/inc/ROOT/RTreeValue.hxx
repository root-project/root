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

// TODO(jblomer): only give RTreeField and descendants access to RTreeValue and descendants

class RTreeFieldBase;

// clang-format off
/**
\class ROOT::Experimental::TreeValueBase
\ingroup Forest
\brief Represents transient storage of simple or complex C++ values.

The data carried by the tree value is used by the computational code and it is supposed to be serialized on Fill
or deserialized into by tree reading.
*/
// clang-format on
class RTreeValueBase {
protected:
   /// The memory location containing (constructed) data of a certain C++ type
   void* fRawPtr;

public:
   RTreeValueBase(RTreeFieldBase *field) : fRawPtr(nullptr), fField(field) {}
   RTreeValueBase(RTreeFieldBase *field, void* rawPtr) : fRawPtr(rawPtr), fField(field) {}
   virtual ~RTreeValueBase() = default; // Prevent slicing

   /// Every value is connected to a field of the corresponding type.
   RTreeFieldBase* fField;
   /// For simple types, the mapped element drills through the layers from the C++ data representation
   /// to the primitive columns.  Otherwise, using fMappedElements is undefined.
   RColumnElementBase fMappedElement;

   void* GetRawPtr() const { return fRawPtr; }
};

} // namespace Detail


// clang-format off
/**
\class ROOT::Experimental::RTreeValue
\ingroup Forest

For simple C++ types, the tree value maps directly to a column element.
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
      // fMappedElement is only used for simple types
      fMappedElement = Detail::RColumnElementBase(fRawPtr, sizeof(T), true);
   }

   RTreeValue(bool /*captureTag*/, Detail::RTreeFieldBase* field, T* value)
      : Detail::RTreeValueBase(field), fValue(nullptr)
   {
      fRawPtr = value;
      // fMappedElement is only used for simple types
      fMappedElement = Detail::RColumnElementBase(fRawPtr, sizeof(T), true);
   }

   T* Get() const { return static_cast<T*>(fRawPtr); }
   std::shared_ptr<T> GetSharedPtr() const { return fValue; }
};


// clang-format off
/**
\class ROOT::Experimental::RTreeValueCollection
\ingroup Forest

A value that behaves like a tree and represents the entire collection in its subtree
*/
// clang-format on
class RTreeValueCollection : public Detail::RTreeValueBase {
public:
   RTreeValueCollection(Detail::RTreeFieldBase* field) : Detail::RTreeValueBase(field) {}
};

} // namespace Experimental
} // namespace ROOT

#endif
