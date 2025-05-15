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
class RNTuple;

namespace Experimental {

class RNTupleAttributeSetWriter;
class RNTupleAttributeSetReader;
class RNTupleFillContext;

namespace Internal::RNTupleAttributes {
static constexpr const char *const kEntryRangeFieldName = "__ROOT_entryRange";
using REntryRange = std::pair<NTupleSize_t, NTupleSize_t>;
} // namespace Internal::RNTupleAttributes

class RNTupleAttributeRange final {
   friend class ROOT::Experimental::RNTupleAttributeSetWriter;
   friend class ROOT::Experimental::RNTupleAttributeSetReader;

   std::unique_ptr<REntry> fEntry;
   ROOT::NTupleSize_t fStart, fEnd;

   RNTupleAttributeRange(std::unique_ptr<REntry> entry, ROOT::NTupleSize_t start, ROOT::NTupleSize_t end = 0)
      : fEntry(std::move(entry)), fStart(start), fEnd(end)
   {
   }

public:
   RNTupleAttributeRange(RNTupleAttributeRange &&) = default;
   RNTupleAttributeRange &operator=(RNTupleAttributeRange &&) = default;

   template <typename T>
   std::shared_ptr<T> GetPtr(std::string_view name) const
   {
      return fEntry->GetPtr<T>(name);
   }

   std::pair<ROOT::NTupleSize_t, ROOT::NTupleSize_t> GetRange() const { return {fStart, fEnd}; }
};

class RNTupleAttributeRangeHandle final {
   friend class RNTupleAttributeSetWriter;

   RNTupleAttributeRange *fRange = nullptr;

   explicit RNTupleAttributeRangeHandle(RNTupleAttributeRange &range) : fRange(&range) {}

public:
   RNTupleAttributeRangeHandle(const RNTupleAttributeRangeHandle &) = delete;
   RNTupleAttributeRangeHandle &operator=(const RNTupleAttributeRangeHandle &) = delete;
   RNTupleAttributeRangeHandle(RNTupleAttributeRangeHandle &&other) { std::swap(fRange, other.fRange); }
   RNTupleAttributeRangeHandle &operator=(RNTupleAttributeRangeHandle &&other)
   {
      std::swap(fRange, other.fRange);
      return *this;
   }

   template <typename T>
   std::shared_ptr<T> GetPtr(std::string_view name)
   {
      if (R__unlikely(!fRange))
         throw ROOT::RException(R__FAIL("Called GetPtr() on invalid RNTupleAttributeRangeHandle"));
      return fRange->GetPtr<T>(name);
   }
};

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
   /// The currently open range, existing from BeginRange() to EndRange()
   std::optional<RNTupleAttributeRange> fOpenRange;

   /// Creates a RNTupleAttributeSetWriter associated to the RNTupleWriter owning `mainFillContext` and writing
   /// in `dir`. `model` is the schema of the AttributeSet.
   static ROOT::RResult<RNTupleAttributeSetWriter> Create(std::string_view name, std::unique_ptr<RNTupleModel> model,
                                                          const RNTupleFillContext *mainFillContext, TDirectory &dir);

   RNTupleAttributeSetWriter(const RNTupleFillContext *mainFillContext, RNTupleFillContext fillContext);

   void EndRangeInternal();
   /// Flushes any remaining open range and writes the Attribute RNTuple to storage.
   void Commit();

public:
   RNTupleAttributeSetWriter(const RNTupleAttributeSetWriter &) = delete;
   RNTupleAttributeSetWriter &operator=(const RNTupleAttributeSetWriter &) = delete;
   RNTupleAttributeSetWriter(RNTupleAttributeSetWriter &&) = default;
   RNTupleAttributeSetWriter &operator=(RNTupleAttributeSetWriter &&) = default;
   ~RNTupleAttributeSetWriter() = default;

   const std::string &GetName() const;

   RNTupleAttributeRangeHandle BeginRange();
   void EndRange(RNTupleAttributeRangeHandle rangeHandle);
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
   friend class RAttributeRangeIterable;

   // List containing pairs { entryRange, entryIndex }, used to quickly find out which entries in the Attribute
   // RNTuple contain entries that overlap a given range. The list is sorted by range start, i.e. entryRange.first.
   std::vector<std::pair<Internal::RNTupleAttributes::REntryRange, NTupleSize_t>> fEntryRanges;
   std::unique_ptr<RNTupleReader> fReader;

   /// Creates a RNTupleAttributeSetReader associated to the RNTupleReader owning `mainFillContext` and writing
   /// in `dir`. `model` is the schema of the AttributeSet.
   static ROOT::RResult<RNTupleAttributeSetReader> Create(std::string_view name, std::unique_ptr<RNTupleModel> model,
                                                          const RNTupleFillContext *mainFillContext, TDirectory &dir);

   static bool EntryRangesAreSorted(const decltype(fEntryRanges) &ranges);

   RNTupleAttributeSetReader(std::unique_ptr<RNTupleReader> reader);

   // Used for GetAttributesContainingRange (with `rangeIsContained == false`) and GetAttributesInRange (with
   // `rangeIsContained == true`).
   std::vector<RNTupleAttributeRange>
   GetAttributesRangeInternal(NTupleSize_t startEntry, NTupleSize_t endEntry, bool rangeIsContained);

public:
   RNTupleAttributeSetReader(const RNTupleAttributeSetReader &) = delete;
   RNTupleAttributeSetReader &operator=(const RNTupleAttributeSetReader &) = delete;
   RNTupleAttributeSetReader(RNTupleAttributeSetReader &&) = default;
   RNTupleAttributeSetReader &operator=(RNTupleAttributeSetReader &&) = default;
   ~RNTupleAttributeSetReader() = default;

   const std::string &GetName() const;

   /// Returns all the attributes whose range fully contains `[startEntry, endEntry]`
   std::vector<RNTupleAttributeRange> GetAttributesContainingRange(NTupleSize_t startEntry, NTupleSize_t endEntry);
   /// Returns all the attributes whose range is fully contained in `[startEntry, endEntry]`
   std::vector<RNTupleAttributeRange> GetAttributesInRange(NTupleSize_t startEntry, NTupleSize_t endEntry);
   /// Returns all the attributes whose range contains index `entryIndex`.
   std::vector<RNTupleAttributeRange> GetAttributes(NTupleSize_t entryIndex);
   /// Returns all the attributes in this Set. The returned attributes are sorted by entry range start.
   std::vector<RNTupleAttributeRange> GetAttributes();
};

} // namespace Experimental
} // namespace ROOT

#endif
