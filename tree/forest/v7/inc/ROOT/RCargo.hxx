/// \file ROOT/RCargo.hxx
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

#ifndef ROOT7_RCargo
#define ROOT7_RCargo

#include <ROOT/RColumnElement.hxx>

#include <memory>
#include <utility>

namespace ROOT {
namespace Experimental {

namespace Detail {

// clang-format off
/**
\class ROOT::Experimental::RCargoBase
\ingroup Forest
\brief The "cargo" represents transient storage of simple or complex C++ values.

The data carried by the cargo is used by the computational code and it supposed to be serialized on Fill
or deserialized into by tree reading.
*/
// clang-format on
class RCargoBase {
public:
   RColumnElementBase* fPrincipalElement;
};

} // namespace Detail


// clang-format off
/**
\class ROOT::Experimental::RCargo
\ingroup Forest

For simple cargo types, no more template specialization is necessary.
*/
// clang-format on
template <typename T>
class RCargo : public Detail::RCargoBase {
private:
   std::shared_ptr<T> fValue;

public:
   template <typename... ArgsT>
   RCargo(ArgsT&&... args) : fValue(std::make_shared<T>(std::forward<ArgsT>(args)...))
   {
   }

   std::shared_ptr<T> Get() { return fValue; }
};

// clang-format off
/**
\class ROOT::Experimental::RCargoCaptured
\ingroup Forest

Allows the user to handle storage allocation.
*/
// clang-format on
template <typename T>
class RCargoCaptured : public Detail::RCargoBase {
private:
   T *fValue;

public:
   RCargoCaptured(T *value) : fValue(value)
   {
   }

   T *Get() { return fValue; }
};

// clang-format off
/**
\class ROOT::Experimental::RCargoSubtree
\ingroup Forest

A cargo object that behaves like a tree.
*/
// clang-format on
class RCargoSubtree : public Detail::RCargoBase {
};

} // namespace Experimental
} // namespace ROOT

#endif
