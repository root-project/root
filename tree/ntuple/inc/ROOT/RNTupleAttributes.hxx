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
#include <ROOT/RNTupleAttrEntry.hxx>

namespace ROOT {

class RNTupleModel;
class RNTuple;

namespace Experimental {

// clang-format off
/**
\class ROOT::Experimental::RNTupleAttrSetWriter
\ingroup NTuple
\brief Class used to write a RNTupleAttrSet in the context of a RNTupleWriter.

TODO: description here

~~~ {.cpp}
TODO: code sample here
~~~
*/
// clang-format on
class RNTupleAttrSetWriter final {
   friend class ::ROOT::Experimental::RNTupleFillContext;

   /// Our own fill context.
   RNTupleFillContext fFillContext;
   /// Fill context of the main RNTuple being written (i.e. the RNTuple whose attributes we are).
   const RNTupleFillContext *fMainFillContext = nullptr;

   /// Creates a RNTupleAttrSetWriter associated to the RNTupleWriter owning `mainFillContext` and writing
   /// in `dir`. `model` is the schema of the AttributeSet.
   static std::unique_ptr<RNTupleAttrSetWriter> Create(std::string_view name, std::unique_ptr<RNTupleModel> model,
                                                       const RNTupleFillContext *mainFillContext, TDirectory &dir);

   RNTupleAttrSetWriter(const RNTupleFillContext *mainFillContext, RNTupleFillContext fillContext);

   /// Flushes any remaining open range and writes the Attribute RNTuple to storage.
   void Commit();

public:
   RNTupleAttrSetWriter(const RNTupleAttrSetWriter &) = delete;
   RNTupleAttrSetWriter &operator=(const RNTupleAttrSetWriter &) = delete;
   RNTupleAttrSetWriter(RNTupleAttrSetWriter &&) = default;
   RNTupleAttrSetWriter &operator=(RNTupleAttrSetWriter &&) = default;
   ~RNTupleAttrSetWriter() = default;

   const ROOT::RNTupleDescriptor &GetDescriptor() const;

   RNTupleAttrEntry BeginRange();
   void CommitRange(RNTupleAttrEntry entry);
};

class RNTupleAttrSetWriterHandle final {
   friend class ::ROOT::Experimental::RNTupleFillContext;

   RNTupleAttrSetWriter *fWriter = nullptr;

   explicit RNTupleAttrSetWriterHandle(RNTupleAttrSetWriter &range) : fWriter(&range) {}

public:
   RNTupleAttrSetWriterHandle(const RNTupleAttrSetWriterHandle &) = delete;
   RNTupleAttrSetWriterHandle &operator=(const RNTupleAttrSetWriterHandle &) = delete;
   RNTupleAttrSetWriterHandle(RNTupleAttrSetWriterHandle &&other) { std::swap(fWriter, other.fWriter); }
   RNTupleAttrSetWriterHandle &operator=(RNTupleAttrSetWriterHandle &&other)
   {
      std::swap(fWriter, other.fWriter);
      return *this;
   }

   RNTupleAttrSetWriter *operator->()
   {
      if (R__unlikely(!fWriter))
         throw ROOT::RException(R__FAIL("Tried to access invalid RNTupleAttrSetWriterHandle"));
      return fWriter;
   }
};

// class RNTupleAttrEntryIterable final {

// };

// clang-format off
/**
\class ROOT::Experimental::RNTupleAttrSetReader
\ingroup NTuple
\brief Class used to read a RNTupleAttrSet in the context of a RNTupleReader

TODO: description here

~~~ {.cpp}
TODO: code sample here
~~~
*/
// clang-format on
class RNTupleAttrSetReader final {
   friend class ROOT::RNTupleReader;

   // List containing pairs { entryRange, entryIndex }, used to quickly find out which entries in the Attribute
   // RNTuple contain entries that overlap a given range. The list is sorted by range start, i.e.
   // entryRange.first.Start().
   std::vector<std::pair<RNTupleAttrRange, NTupleSize_t>> fEntryRanges;
   std::unique_ptr<RNTupleReader> fReader;

   static bool EntryRangesAreSorted(const decltype(fEntryRanges) &ranges);

   explicit RNTupleAttrSetReader(std::unique_ptr<RNTupleReader> reader);

   // Used for GetAttributesContainingRange (with `rangeIsContained == false`) and GetAttributesInRange (with
   // `rangeIsContained == true`).
   std::vector<NTupleSize_t>
   GetAttributesRangeInternal(NTupleSize_t startEntry, NTupleSize_t endEntry, bool rangeIsContained);

public:
   RNTupleAttrSetReader(const RNTupleAttrSetReader &) = delete;
   RNTupleAttrSetReader &operator=(const RNTupleAttrSetReader &) = delete;
   RNTupleAttrSetReader(RNTupleAttrSetReader &&) = default;
   RNTupleAttrSetReader &operator=(RNTupleAttrSetReader &&) = default;
   ~RNTupleAttrSetReader() = default;

   const ROOT::RNTupleDescriptor &GetDescriptor() const;

   RNTupleAttrEntry CreateAttrEntry();
   void LoadAttrEntry(NTupleSize_t index, RNTupleAttrEntry &entry);

   /// Returns all the attributes whose range fully contains `[startEntry, endEntry)`
   std::vector<NTupleSize_t> GetAttributesContainingRange(NTupleSize_t startEntry, NTupleSize_t endEntry);
   /// Returns all the attributes whose range is fully contained in `[startEntry, endEntry)`
   std::vector<NTupleSize_t> GetAttributesInRange(NTupleSize_t startEntry, NTupleSize_t endEntry);
   /// Returns all the attributes whose range contains index `entryIndex`.
   std::vector<NTupleSize_t> GetAttributes(NTupleSize_t entryIndex);
   /// Returns all the attributes in this Set. The returned attributes are sorted by entry range start.
   std::vector<NTupleSize_t> GetAttributes();
};

bool IsReservedRNTupleAttrSetName(std::string_view name);

} // namespace Experimental
} // namespace ROOT

#endif
