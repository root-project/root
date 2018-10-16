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
#include <utility>

namespace ROOT {
namespace Experimental {

namespace Detail {

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
public:
   RColumnElementBase* fPrincipalElement;
};

} // namespace Detail


// clang-format off
/**
\class ROOT::Experimental::RTreeValue
\ingroup Forest

For simple C++ types, no more template specialization is necessary.
*/
// clang-format on
template <typename T>
class RTreeValue : public Detail::RTreeValueBase {
private:
   std::shared_ptr<T> fValue;

public:
   template <typename... ArgsT>
   RTreeValue(ArgsT&&... args) : fValue(std::make_shared<T>(std::forward<ArgsT>(args)...))
   {
   }

   std::shared_ptr<T> Get() { return fValue; }
};

// clang-format off
/**
\class ROOT::Experimental::RTreeValueCaptured
\ingroup Forest

Allows the user to handle storage allocation.
*/
// clang-format on
template <typename T>
class RTreeValueCaptured : public Detail::RTreeValueBase {
private:
   T *fValue;

public:
   RTreeValueCaptured(T *value) : fValue(value)
   {
   }

   T *Get() { return fValue; }
};

// clang-format off
/**
\class ROOT::Experimental::RTreeValueCollection
\ingroup Forest

A value that behaves like a tree and represents the entire collection in its subtree
*/
// clang-format on
class RTreeValueCollection : public Detail::RTreeValueBase {
};

} // namespace Experimental
} // namespace ROOT

#endif
