/// \file ROOT/RTreeEntry.hxx
/// \ingroup Forest ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-07-19
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RTreeEntry
#define ROOT7_RTreeEntry

#include <ROOT/RTreeValue.hxx>

#include <memory>
#include <utility>

namespace ROOT {
namespace Experimental {

// clang-format off
/**
\class ROOT::Experimental::RTreeEntry
\ingroup Forest
\brief The RTreeEntry is a collection of tree values corresponding to a complete row in the data set

*/
// clang-format on
class RTreeEntry {
   std::vector<Detail::RTreeValueBase> fTreeValues;

public:
   RTreeEntry() = default;

   /// While building the entry, adds a new value to the list and return the value's shared pointer
   template <typename T, typename... ArgsT>
   std::shared_ptr<T> AddField(ArgsT&&... args) {
     auto value = std::make_shared<RTreeValue<T>>(std::forward<ArgsT>(args)...);
     auto value_ptr = value->Get();
     fTreeValues.emplace_back(std::move(value));
     return value_ptr;
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
