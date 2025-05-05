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
#include <ROOT/RNTupleFillContext.hxx>

namespace ROOT {

class RNTupleModel;

namespace Experimental {

class RNTupleAttributeSet;
class RNTupleFillContext;

namespace Internal {
class RNTupleAttributeRange final {
   friend class ROOT::Experimental::RNTupleAttributeSet;

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
} // namespace Internal

class RNTupleAttributeRangeHandle final {
   friend class RNTupleAttributeSet;

   Internal::RNTupleAttributeRange &fRange;

   RNTupleAttributeRangeHandle(Internal::RNTupleAttributeRange &range) : fRange(range) {}

public:
   template <typename T>
   std::shared_ptr<T> GetPtr(std::string_view name)
   {
      return fRange.GetPtr<T>(name);
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
   friend class ::ROOT::Experimental::RNTupleFillContext;

   static constexpr const char *const kEntryRangeFieldName = "__ROOT_entryRange";

   using REntryRange = std::pair<NTupleSize_t, NTupleSize_t>;

   /// Our own fill context.
   RNTupleFillContext fFillContext;
   /// Fill context of the main RNTuple being written (i.e. the RNTuple whose attributes we are).
   const RNTupleFillContext *fMainFillContext = nullptr;
   /// The currently open range, existing from BeginRange() to EndRange()
   std::optional<Internal::RNTupleAttributeRange> fOpenRange;

   static ROOT::RResult<RNTupleAttributeSet> Create(std::string_view name, std::unique_ptr<RNTupleModel> model,
                                                    const RNTupleFillContext *mainFillContext, TDirectory &dir);

   RNTupleAttributeSet(const RNTupleFillContext *mainFillContext, RNTupleFillContext fillContext);

   void EndRangeInternal();

public:
   RNTupleAttributeSet(const RNTupleAttributeSet &) = delete;
   RNTupleAttributeSet &operator=(const RNTupleAttributeSet &) = delete;
   RNTupleAttributeSet(RNTupleAttributeSet &&) = default;
   RNTupleAttributeSet &operator=(RNTupleAttributeSet &&) = default;
   ~RNTupleAttributeSet();

   const std::string &GetName() const;

   RNTupleAttributeRangeHandle BeginRange();
   void EndRange(RNTupleAttributeRangeHandle rangeHandle);
};

} // namespace Experimental
} // namespace ROOT

#endif
