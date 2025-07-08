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
#include <ROOT/RNTupleAttributeEntry.hxx>

namespace ROOT {

class RNTupleModel;
class RNTuple;

namespace Experimental {

// clang-format off
/**
\class ROOT::Experimental::RNTupleAttributeSetWriter
\ingroup NTuple
\brief Class used to write a RNTupleAttributeSet in the context of a RNTupleWriter.

TODO: description here

~~~ {.cpp}
TODO: code sample here
~~~
*/
// clang-format on
class RNTupleAttributeSetWriter final {
   friend class ::ROOT::Experimental::RNTupleFillContext;

   /// Our own fill context.
   RNTupleFillContext fFillContext;
   /// Fill context of the main RNTuple being written (i.e. the RNTuple whose attributes we are).
   const RNTupleFillContext *fMainFillContext = nullptr;
   /// The currently open entry, existing from BeginRange() to CommitRange()
   std::optional<RNTupleAttributeEntry> fOpenEntry;

   /// Creates a RNTupleAttributeSetWriter associated to the RNTupleWriter owning `mainFillContext` and writing
   /// in `dir`. `model` is the schema of the AttributeSet.
   static ROOT::RResult<std::unique_ptr<RNTupleAttributeSetWriter>>
   Create(std::string_view name, std::unique_ptr<RNTupleModel> model, const RNTupleFillContext *mainFillContext,
          TDirectory &dir);

   RNTupleAttributeSetWriter(const RNTupleFillContext *mainFillContext, RNTupleFillContext fillContext);

   void CommitRangeInternal();
   /// Flushes any remaining open range and writes the Attribute RNTuple to storage.
   void Commit();

public:
   RNTupleAttributeSetWriter(const RNTupleAttributeSetWriter &) = delete;
   RNTupleAttributeSetWriter &operator=(const RNTupleAttributeSetWriter &) = delete;
   RNTupleAttributeSetWriter(RNTupleAttributeSetWriter &&) = default;
   RNTupleAttributeSetWriter &operator=(RNTupleAttributeSetWriter &&) = default;
   ~RNTupleAttributeSetWriter() = default;

   const std::string &GetName() const;

   RNTupleAttributeEntryHandle BeginRange();
   void CommitRange(RNTupleAttributeEntryHandle rangeHandle);
};

class RNTupleAttributeSetWriterHandle final {
   friend class ::ROOT::Experimental::RNTupleFillContext;

   RNTupleAttributeSetWriter *fWriter = nullptr;

   explicit RNTupleAttributeSetWriterHandle(RNTupleAttributeSetWriter &range) : fWriter(&range) {}

public:
   RNTupleAttributeSetWriterHandle(const RNTupleAttributeSetWriterHandle &) = delete;
   RNTupleAttributeSetWriterHandle &operator=(const RNTupleAttributeSetWriterHandle &) = delete;
   RNTupleAttributeSetWriterHandle(RNTupleAttributeSetWriterHandle &&other) { std::swap(fWriter, other.fWriter); }
   RNTupleAttributeSetWriterHandle &operator=(RNTupleAttributeSetWriterHandle &&other)
   {
      std::swap(fWriter, other.fWriter);
      return *this;
   }

   RNTupleAttributeSetWriter *operator->()
   {
      if (R__unlikely(!fWriter))
         throw ROOT::RException(R__FAIL("Tried to access invalid RNTupleAttributeSetWriterHandle"));
      return fWriter;
   }
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleAttributeSetReader
\ingroup NTuple
\brief Class used to read a RNTupleAttributeSet in the context of a RNTupleReader

TODO: description here

~~~ {.cpp}
TODO: code sample here
~~~
*/
// clang-format on
class RNTupleAttributeSetReader final {
   friend class ROOT::RNTupleReader;
   friend class RAttributeEntryIterable;

   // List containing pairs { entryRange, entryIndex }, used to quickly find out which entries in the Attribute
   // RNTuple contain entries that overlap a given range. The list is sorted by range start, i.e.
   // entryRange.first.Start().
   std::vector<std::pair<RNTupleAttributeRange, NTupleSize_t>> fEntryRanges;
   std::unique_ptr<RNTupleReader> fReader;

   /// Creates a RNTupleAttributeSetReader associated to the RNTupleReader owning `mainFillContext` and writing
   /// in `dir`. `model` is the schema of the AttributeSet.
   static ROOT::RResult<RNTupleAttributeSetReader> Create(std::string_view name, std::unique_ptr<RNTupleModel> model,
                                                          const RNTupleFillContext *mainFillContext, TDirectory &dir);

   static bool EntryRangesAreSorted(const decltype(fEntryRanges) &ranges);

   RNTupleAttributeSetReader(std::unique_ptr<RNTupleReader> reader);

   // Used for GetAttributesContainingRange (with `rangeIsContained == false`) and GetAttributesInRange (with
   // `rangeIsContained == true`).
   std::vector<RNTupleAttributeEntry>
   GetAttributesRangeInternal(NTupleSize_t startEntry, NTupleSize_t endEntry, bool rangeIsContained);

public:
   RNTupleAttributeSetReader(const RNTupleAttributeSetReader &) = delete;
   RNTupleAttributeSetReader &operator=(const RNTupleAttributeSetReader &) = delete;
   RNTupleAttributeSetReader(RNTupleAttributeSetReader &&) = default;
   RNTupleAttributeSetReader &operator=(RNTupleAttributeSetReader &&) = default;
   ~RNTupleAttributeSetReader() = default;

   const std::string &GetName() const;

   /// Returns all the attributes whose range fully contains `[startEntry, endEntry)`
   std::vector<RNTupleAttributeEntry> GetAttributesContainingRange(NTupleSize_t startEntry, NTupleSize_t endEntry);
   /// Returns all the attributes whose range is fully contained in `[startEntry, endEntry)`
   std::vector<RNTupleAttributeEntry> GetAttributesInRange(NTupleSize_t startEntry, NTupleSize_t endEntry);
   /// Returns all the attributes whose range contains index `entryIndex`.
   std::vector<RNTupleAttributeEntry> GetAttributes(NTupleSize_t entryIndex);
   /// Returns all the attributes in this Set. The returned attributes are sorted by entry range start.
   std::vector<RNTupleAttributeEntry> GetAttributes();
};

} // namespace Experimental
} // namespace ROOT

#endif
