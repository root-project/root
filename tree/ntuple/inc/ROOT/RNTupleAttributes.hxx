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

class RNTupleAttrSetWriter;

namespace Internal {

struct RNTupleAttrEntryPair {
   REntry &fMetaEntry;
   REntry &fScopedEntry;
   ROOT::RNTupleModel &fMetaModel;

   std::size_t Append();
   ROOT::DescriptorId_t GetModelId() const { return fMetaEntry.GetModelId(); }
};

namespace RNTupleAttributes {

const char *const kRangeStartName = "_rangeStart";
const char *const kRangeLenName = "_rangeLen";

} // namespace RNTupleAttributes

} // namespace Internal

/// An entry range linked to an Attribute
class RNTupleAttrRange final {
   ROOT::NTupleSize_t fStart = 0;
   ROOT::NTupleSize_t fLength = 0;

   RNTupleAttrRange(ROOT::NTupleSize_t start, ROOT::NTupleSize_t length) : fStart(start), fLength(length) {}

public:
   static RNTupleAttrRange FromStartLength(ROOT::NTupleSize_t start, ROOT::NTupleSize_t length)
   {
      return RNTupleAttrRange{start, length};
   }

   /// Creates an AttributeRange from [start, end), where `end` is one past the last valid entry of the range
   /// (`FromStartEnd(0, 10)` will create a range whose last valid index is 9).
   static RNTupleAttrRange FromStartEnd(ROOT::NTupleSize_t start, ROOT::NTupleSize_t end)
   {
      R__ASSERT(end >= start);
      return RNTupleAttrRange{start, end - start};
   }

   RNTupleAttrRange() = default;

   /// Returns the first valid entry index in the range. Returns nullopt if the range has zero length.
   std::optional<ROOT::NTupleSize_t> First() const { return fLength ? std::make_optional(fStart) : std::nullopt; }
   /// Returns the start of the range. Note that this is *not* a valid index in the range if the range has zero length.
   ROOT::NTupleSize_t Start() const { return fStart; }
   /// Returns the last valid entry index in the range. Returns nullopt if the range has zero length.
   std::optional<ROOT::NTupleSize_t> Last() const
   {
      return fLength ? std::make_optional(fStart + fLength - 1) : std::nullopt;
   }
   /// Returns one past the last valid index of the range, equal to `Start() + Length()`.
   ROOT::NTupleSize_t End() const { return fStart + fLength; }
   ROOT::NTupleSize_t Length() const { return fLength; }

   /// Returns the pair { firstEntryIdx, lastEntryIdx } (inclusive). Returns nullopt if the range has zero length.
   std::optional<std::pair<ROOT::NTupleSize_t, ROOT::NTupleSize_t>> GetFirstLast() const
   {
      return fLength ? std::make_optional(std::make_pair(fStart, fStart + fLength - 1)) : std::nullopt;
   }
   /// Returns the pair { start, length }.
   std::pair<ROOT::NTupleSize_t, ROOT::NTupleSize_t> GetStartLength() const { return {Start(), Length()}; }
};

/// A range used for writing. It has a well-defined start but not a length/end yet.
/// It is artificially made non-copyable in order to clarify the semantics of Begin/CommitRange.
/// For the same reason, it can only be created by the AttrSetWriter.
class RNTupleAttrPendingRange final {
   friend class ROOT::Experimental::RNTupleAttrSetWriter;

   ROOT::NTupleSize_t fStart = 0;
   ROOT::DescriptorId_t fModelId = kInvalidDescriptorId;

   explicit RNTupleAttrPendingRange(ROOT::NTupleSize_t start, ROOT::DescriptorId_t modelId)
      : fStart(start), fModelId(modelId)
   {
   }

public:
   RNTupleAttrPendingRange(const RNTupleAttrPendingRange &) = delete;
   RNTupleAttrPendingRange &operator=(const RNTupleAttrPendingRange &) = delete;

   // NOTE: explicitly implemented to make sure that 'other' gets invalidated upon move.
   RNTupleAttrPendingRange(RNTupleAttrPendingRange &&other) { *this = std::move(other); }

   // NOTE: explicitly implemented to make sure that 'other' gets invalidated upon move.
   RNTupleAttrPendingRange &operator=(RNTupleAttrPendingRange &&other)
   {
      std::swap(fStart, other.fStart);
      std::swap(fModelId, other.fModelId);
      return *this;
   }

   ROOT::NTupleSize_t Start() const
   {
      if (fModelId == kInvalidDescriptorId)
         throw ROOT::RException(R__FAIL("Tried to commit an already-committed attribute range."));
      return fStart;
   }

   ROOT::DescriptorId_t GetModelId() const { return fModelId; }
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleAttrSetWriter
\ingroup NTuple
\brief Class used to write a RNTupleAttrSet in the context of a RNTupleWriter.

An Attribute Set is written as a separate RNTuple linked to the "main" RNTuple that created it.
A RNTupleAttrSetWriter only lives as long as the RNTupleWriter that created it (or until CloseAttributeSet() is called).
Users should not use this class directly but rather via RNTupleAttrSetWriterHandle, which is the type returned by
RNTupleWriter::CreateAttributeSet().

~~~ {.cpp}
// Writing attributes via RNTupleAttrSetWriter
// -------------------------------------------

// First define the schema of your Attribute Set:
auto attrModel = ROOT::RNTupleModel::Create();
auto pMyAttr = attrModel->MakeField<std::string>("myAttr");

// Then, assuming `writer` is an RNTupleWriter, create it:
auto attrSet = writer->CreateAttributeSet(std::move(attrModel), "MyAttrSet");

// Attributes are assigned to entry ranges. A range is started via BeginRange():
auto range = attrSet->BeginRange();

// To assign actual attributes, you use the same interface as the main RNTuple:
*pMyAttr = "This is my attribute for this range";

// ... here you can fill your main RNTuple with data ...

// Once you're done, close the range. This will commit the attribute data and bind it to all data written
// between BeginRange() and CommitRange().
attrSet->CommitRange(std::move(range));

// You don't need to explicitly close the AttributeSet, but if you want to do so, use:
// writer->CloseAttributeSet(std::move(attrSet));
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
                                                       const RNTupleFillContext &mainFillContext, TDirectory &dir);

   RNTupleAttrSetWriter(const RNTupleFillContext &mainFillContext, RNTupleFillContext fillContext,
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

   std::weak_ptr<RNTupleAttrSetWriter> fWriter;

   explicit RNTupleAttrSetWriterHandle(const std::shared_ptr<RNTupleAttrSetWriter> &range) : fWriter(range) {}

public:
   RNTupleAttrSetWriterHandle(const RNTupleAttrSetWriterHandle &) = delete;
   RNTupleAttrSetWriterHandle &operator=(const RNTupleAttrSetWriterHandle &) = delete;
   RNTupleAttrSetWriterHandle(RNTupleAttrSetWriterHandle &&) = default;
   RNTupleAttrSetWriterHandle &operator=(RNTupleAttrSetWriterHandle &&other) = default;

   /// Retrieves the underlying pointer to the AttrSetWriter, throwing if it's invalid.
   /// This is NOT thread-safe and must be called from the same thread that created the AttrSetWriter.
   RNTupleAttrSetWriter *operator->()
   {
      if (R__unlikely(fWriter.expired()))
         throw ROOT::RException(R__FAIL("Tried to access invalid RNTupleAttrSetWriterHandle"));
      return fWriter.lock().get();
   }
};

class RNTupleAttrEntryIterable;

// clang-format off
/**
\class ROOT::Experimental::RNTupleAttrSetReader
\ingroup NTuple
\brief Class used to read a RNTupleAttrSet in the context of a RNTupleReader

An RNTupleAttrSetReader is created via RNTupleReader::OpenAttributeSet. Once created, it may outlive its parent Reader.
Reading Attributes works similarly to reading regular RNTuple entries: you can either create entries or just use the
AttrSetReader Model's default entry and load data into it via LoadAttrEntry.

~~~ {.cpp}
// Reading Attributes via RNTupleAttrSetReader
// -------------------------------------------

// Assuming `reader` is a RNTupleReader:
auto attrSet = reader->OpenAttributeSet("MyAttrSet");

// Just like how you would read a regular RNTuple, first get the pointer to the fields you want to read:
auto &attrEntry = attrSet->GetModel().GetDefaultEntry();
auto pAttr = attrEntry->GetPtr<std::string>("myAttr");

// Then select which attributes you want to read. E.g. read all attributes linked to the entry at index 10:
for (auto idx : attrSet->GetAttributes(10)) {
   attrSet->LoadAttrEntry(idx);
   cout << "entry " << idx << " has attribute " << *pAttr << "\n";
}
~~~
*/
// clang-format on
class RNTupleAttrSetReader final {
   friend class ROOT::RNTupleReader;
   friend class RNTupleAttrEntryIterable;

   /// List containing pairs { entryRange, entryIndex }, used to quickly find out which entries in the Attribute
   /// RNTuple contain entries that overlap a given range. The list is sorted by range start, i.e.
   /// entryRange.first.Start().
   std::vector<std::pair<RNTupleAttrRange, NTupleSize_t>> fEntryRanges;
   /// The internal Reader used to read the AttributeSet RNTuple
   std::unique_ptr<RNTupleReader> fReader;
   /// The reconstructed user model
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
