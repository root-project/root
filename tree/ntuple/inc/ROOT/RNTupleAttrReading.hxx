/// \file ROOT/RNTupleAttrReading.hxx
/// \ingroup NTuple
/// \author Giacomo Parolini <giacomo.parolini@cern.ch>
/// \date 2026-04-01
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

#ifndef ROOT7_RNTuple_Attr_Reading
#define ROOT7_RNTuple_Attr_Reading

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <ROOT/RNTupleFillContext.hxx>
#include <ROOT/RNTupleAttrUtils.hxx>
#include <ROOT/RNTupleUtils.hxx>

namespace ROOT {

class REntry;
class RNTupleDescriptor;
class RNTupleModel;

namespace Experimental {

class RNTupleAttrEntryIterable;

// clang-format off
/**
\class ROOT::Experimental::RNTupleAttrRange
\ingroup NTuple
\brief A range of main entries referred to by an attribute entry

Each attribute entry contains a set of values referring to 0 or more contiguous entries in the main RNTuple.
This class represents that contiguous range of entries.
*/
// clang-format on
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
   std::optional<ROOT::NTupleSize_t> GetFirst() const { return fLength ? std::make_optional(fStart) : std::nullopt; }
   /// Returns the beginning of the range. Note that this is *not* a valid index in the range if the range has zero
   /// length.
   ROOT::NTupleSize_t GetStart() const { return fStart; }
   /// Returns the last valid entry index in the range. Returns nullopt if the range has zero length.
   std::optional<ROOT::NTupleSize_t> GetLast() const
   {
      return fLength ? std::make_optional(fStart + fLength - 1) : std::nullopt;
   }
   /// Returns one past the last valid index of the range, equal to `GetStart() + GetLength()`.
   ROOT::NTupleSize_t GetEnd() const { return fStart + fLength; }
   ROOT::NTupleSize_t GetLength() const { return fLength; }

   /// Returns the pair { firstEntryIdx, lastEntryIdx } (inclusive). Returns nullopt if the range has zero length.
   std::optional<std::pair<ROOT::NTupleSize_t, ROOT::NTupleSize_t>> GetFirstLast() const
   {
      return fLength ? std::make_optional(std::make_pair(fStart, fStart + fLength - 1)) : std::nullopt;
   }
   /// Returns the pair { start, length }.
   std::pair<ROOT::NTupleSize_t, ROOT::NTupleSize_t> GetStartLength() const { return {fStart, fLength}; }
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleAttrSetReader
\ingroup NTuple
\brief Class used to read a RNTupleAttrSet in the context of a RNTupleReader

An RNTupleAttrSetReader is created via RNTupleReader::OpenAttributeSet. Once created, it may outlive its parent Reader.
Reading Attributes works similarly to reading regular RNTuple entries: you can either create entries or just use the
AttrSetReader Model's default entry and load data into it via LoadEntry.

~~ {.cpp}
// Reading Attributes via RNTupleAttrSetReader
// -------------------------------------------

// Assuming `reader` is a RNTupleReader:
auto attrSet = reader->OpenAttributeSet("MyAttrSet");

// Just like how you would read a regular RNTuple, first get the pointer to the fields you want to read:
auto &attrEntry = attrSet->GetModel().GetDefaultEntry();
auto pAttr = attrEntry->GetPtr<std::string>("myAttr");

// Then select which attributes you want to read. E.g. read all attributes linked to the entry at index 10:
for (auto idx : attrSet->GetAttributes(10)) {
   attrSet->LoadEntry(idx);
   cout << "entry " << idx << " has attribute " << *pAttr << "\n";
}
~~
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

   RNTupleAttrSetReader(std::unique_ptr<RNTupleReader> reader, std::uint16_t vSchemaMajor);

public:
   RNTupleAttrSetReader(const RNTupleAttrSetReader &) = delete;
   RNTupleAttrSetReader &operator=(const RNTupleAttrSetReader &) = delete;
   RNTupleAttrSetReader(RNTupleAttrSetReader &&) = default;
   RNTupleAttrSetReader &operator=(RNTupleAttrSetReader &&) = default;
   ~RNTupleAttrSetReader() = default;

   /// Returns the read-only descriptor of this attribute set
   const ROOT::RNTupleDescriptor &GetDescriptor() const;
   /// Returns the read-only model of this attribute set
   const ROOT::RNTupleModel &GetModel() const { return *fUserModel; }

   /// Creates an entry suitable for use with LoadEntry.
   /// This is a convenience method equivalent to GetModel().CreateEntry().
   std::unique_ptr<REntry> CreateEntry();

   /// Loads the attribute entry at position `index` into the default entry.
   /// Returns the range of main RNTuple entries that the loaded set of attributes refers to.
   RNTupleAttrRange LoadEntry(NTupleSize_t index);
   /// Loads the attribute entry at position `index` into the given entry.
   /// Returns the range of main RNTuple entries that the loaded set of attributes refers to.
   RNTupleAttrRange LoadEntry(NTupleSize_t index, REntry &entry);

   /// Returns the number of all attribute entries in this attribute set.
   std::size_t GetNEntries() const { return fEntryRanges.size(); }

   /// Returns all the attributes in this Set. The returned attributes are sorted by entry range start.
   RNTupleAttrEntryIterable GetAttributes();
   /// Returns all the attributes whose range contains index `entryIndex`.
   RNTupleAttrEntryIterable GetAttributes(NTupleSize_t entryIndex);
   /// Returns all the attributes whose range fully contains `[startEntry, endEntry)`
   RNTupleAttrEntryIterable GetAttributesContainingRange(NTupleSize_t startEntry, NTupleSize_t endEntry);
   /// Returns all the attributes whose range is fully contained in `[startEntry, endEntry)`
   RNTupleAttrEntryIterable GetAttributesInRange(NTupleSize_t startEntry, NTupleSize_t endEntry);
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleAttrEntryIterable
\ingroup NTuple
\brief Iterable class used to loop over attribute entries.

This class allows to perform range-for iteration on some set of attributes, typically returned by the
RNTupleAttrSetReader::GetAttributes family of methods.

See the documentation of RNTupleAttrSetReader for example usage.
*/
// clang-format on
class RNTupleAttrEntryIterable final {
public:
   struct RFilter {
      RNTupleAttrRange fRange;
      bool fIsContained;
   };

private:
   RNTupleAttrSetReader *fReader = nullptr;
   std::optional<RFilter> fFilter;

public:
   class RIterator final {
   private:
      using Iter_t = decltype(std::declval<RNTupleAttrSetReader>().fEntryRanges.begin());
      Iter_t fCur, fEnd;
      std::optional<RFilter> fFilter;

      Iter_t SkipFiltered() const;
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
            if (fFilter->fRange.GetLength() == 0)
               fCur = end;
            else
               fCur = SkipFiltered();
         }
      }
      iterator operator++()
      {
         ++fCur;
         fCur = SkipFiltered();
         return *this;
      }
      iterator operator++(int)
      {
         iterator it = *this;
         operator++();
         return it;
      }
      reference operator*() { return fCur->second; }
      bool operator!=(const iterator &rh) const { return !operator==(rh); }
      bool operator==(const iterator &rh) const { return fCur == rh.fCur; }
   };

   explicit RNTupleAttrEntryIterable(RNTupleAttrSetReader &reader, std::optional<RFilter> filter = {})
      : fReader(&reader), fFilter(filter)
   {
   }

   RIterator begin() { return RIterator{fReader->fEntryRanges.begin(), fReader->fEntryRanges.end(), fFilter}; }
   RIterator end() { return RIterator{fReader->fEntryRanges.end(), fReader->fEntryRanges.end(), fFilter}; }
};

} // namespace Experimental
} // namespace ROOT

#endif
