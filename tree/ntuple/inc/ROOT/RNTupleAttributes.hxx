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
   /// The model that the user provided on creation. Used to create user-visible entries.
   std::unique_ptr<RNTupleModel> fUserModel;

   /// Creates a RNTupleAttrSetWriter associated to the RNTupleWriter owning `mainFillContext` and writing
   /// in `dir`. `model` is the schema of the AttributeSet.
   static std::unique_ptr<RNTupleAttrSetWriter> Create(std::string_view name, std::unique_ptr<RNTupleModel> model,
                                                       const RNTupleFillContext *mainFillContext, TDirectory &dir);

   RNTupleAttrSetWriter(const RNTupleFillContext *mainFillContext, RNTupleFillContext fillContext,
                        std::unique_ptr<RNTupleModel> userModel);

   /// Flushes any remaining open range and writes the Attribute RNTuple to storage.
   void Commit();

public:
   RNTupleAttrSetWriter(const RNTupleAttrSetWriter &) = delete;
   RNTupleAttrSetWriter &operator=(const RNTupleAttrSetWriter &) = delete;
   RNTupleAttrSetWriter(RNTupleAttrSetWriter &&) = default;
   RNTupleAttrSetWriter &operator=(RNTupleAttrSetWriter &&) = default;
   ~RNTupleAttrSetWriter() = default;

   // XXX: should this be exposed?
   const ROOT::RNTupleDescriptor &GetDescriptor() const { return fFillContext.fSink->GetDescriptor(); }
   const ROOT::RNTupleModel &GetModel() const { return *fUserModel; }

   [[nodiscard]] RNTupleAttrPendingRange BeginRange();
   void CommitRange(RNTupleAttrPendingRange range);
   void CommitRange(RNTupleAttrPendingRange range, REntry &entry);

   std::unique_ptr<REntry> CreateEntry() { return fUserModel->CreateEntry(); }
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

class RNTupleAttrEntryIterable;

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
   friend class RNTupleAttrEntryIterable;

   // List containing pairs { entryRange, entryIndex }, used to quickly find out which entries in the Attribute
   // RNTuple contain entries that overlap a given range. The list is sorted by range start, i.e.
   // entryRange.first.Start().
   std::vector<std::pair<RNTupleAttrRange, NTupleSize_t>> fEntryRanges;
   std::unique_ptr<RNTupleReader> fReader;
   // The reconstructed user model
   std::unique_ptr<ROOT::RNTupleModel> fUserModel;

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
   const ROOT::RNTupleModel &GetModel() const { return *fUserModel; }

   std::unique_ptr<REntry> CreateEntry() { return fUserModel->CreateEntry(); }
   RNTupleAttrRange LoadAttrEntry(NTupleSize_t index);
   RNTupleAttrRange LoadAttrEntry(NTupleSize_t index, REntry &entry);

   /// Returns all the attributes whose range fully contains `[startEntry, endEntry)`
   RNTupleAttrEntryIterable GetAttributesContainingRange(NTupleSize_t startEntry, NTupleSize_t endEntry);
   /// Returns all the attributes whose range is fully contained in `[startEntry, endEntry)`
   RNTupleAttrEntryIterable GetAttributesInRange(NTupleSize_t startEntry, NTupleSize_t endEntry);
   /// Returns all the attributes whose range contains index `entryIndex`.
   RNTupleAttrEntryIterable GetAttributes(NTupleSize_t entryIndex);
   /// Returns all the attributes in this Set. The returned attributes are sorted by entry range start.
   RNTupleAttrEntryIterable GetAttributes();

   /// Returns the number of all attribute entries in this Attribute Set.
   std::size_t GetNAttrEntries() const { return fEntryRanges.size(); }
};

class RNTupleAttrEntryIterable final {
public:
   struct RFilter {
      RNTupleAttrRange fRange;
      bool fIsContained;
   };

private:
   RNTupleAttrSetReader &fReader;
   std::optional<RFilter> fFilter;

public:
   class RIterator final {
   private:
      using Iter_t = decltype(std::declval<RNTupleAttrSetReader>().fEntryRanges.begin());
      Iter_t fCur, fEnd;
      std::optional<RFilter> fFilter;

      Iter_t Next() const;
      bool FullyContained(RNTupleAttrRange range) const;

   public:
      using iterator_category = std::forward_iterator_tag;
      using iterator = RIterator;
      using value_type = NTupleSize_t;
      using difference_type = std::ptrdiff_t;
      using pointer = const value_type *;
      using reference = const value_type &;

      RIterator(Iter_t iter, Iter_t end, std::optional<RFilter> filter) : fCur(iter), fEnd(end), fFilter(filter)
      {
         if (fFilter) {
            if (fFilter->fRange.Length() == 0)
               fCur = end;
            else
               fCur = Next();
         }
      }
      iterator operator++()
      {
         ++fCur;
         fCur = Next();
         return *this;
      }
      iterator operator++(int)
      {
         iterator it = *this;
         ++fCur;
         fCur = Next();
         return it;
      }
      reference operator*() { return fCur->second; }
      bool operator!=(const iterator &rh) const { return !operator==(rh); }
      bool operator==(const iterator &rh) const { return fCur == rh.fCur; }
   };

   explicit RNTupleAttrEntryIterable(RNTupleAttrSetReader &reader, std::optional<RFilter> filter = {})
      : fReader(reader), fFilter(filter)
   {
   }

   RIterator begin() { return RIterator{fReader.fEntryRanges.begin(), fReader.fEntryRanges.end(), fFilter}; }
   RIterator end() { return RIterator{fReader.fEntryRanges.end(), fReader.fEntryRanges.end(), fFilter}; }
};

bool IsReservedRNTupleAttrSetName(std::string_view name);

} // namespace Experimental
} // namespace ROOT

#endif
