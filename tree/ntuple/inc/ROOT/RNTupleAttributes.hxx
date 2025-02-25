/// \file ROOT/RNTupleAttributes.hxx
/// \ingroup NTuple ROOT7
/// \author Giacomo Parolini <giacomo.parolini@cern.ch>
/// \date 2025-02-25
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RNTuple_Attributes
#define ROOT7_RNTuple_Attributes

#include <memory>
#include <string>
#include <string_view>

#include <ROOT/REntry.hxx>

namespace ROOT {

class RNTupleModel;

namespace Experimental {

class RNTupleAttributeSet;
class RNTupleFillContext;

class RNTupleAttributeRange final {
   friend class RNTupleAttributeSet;

   std::unique_ptr<REntry> fEntry;
   ROOT::NTupleSize_t fStart;

   RNTupleAttributeRange(std::unique_ptr<REntry> entry, ROOT::NTupleSize_t start);

public:
   template <typename T>
   std::shared_ptr<T> GetPtr(std::string_view name)
   {
      return fEntry->GetPtr<T>(name);
   }
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleAttributeSet
\ingroup NTuple
\brief A Grouping of RNTuple Attributes, characterized by a common schema

TODO: description here

~~~ {.cpp}
TODO: code sample here
~~~
*/
// clang-format on
class RNTupleAttributeSet final {
   static constexpr const char *const kEntryRangeFieldName = "__ROOT_entryRange";

   using REntryRange = std::pair<NTupleSize_t, NTupleSize_t>;

   /// Our own fill context.
   std::unique_ptr<RNTupleFillContext> fFillContext;
   /// Fill context of the main RNTuple being written (i.e. the RNTuple whose attributes we are).
   const RNTupleFillContext *fMainFillContext = nullptr;

   RNTupleAttributeSet() = default;

public:
   RNTupleAttributeSet(std::string_view name, std::unique_ptr<RNTupleModel> model,
                       const RNTupleFillContext *fillContext, TDirectory &dir);

   RNTupleAttributeRange BeginRange();
   void EndRange(RNTupleAttributeRange range);
};

} // namespace Experimental
} // namespace ROOT

#endif
