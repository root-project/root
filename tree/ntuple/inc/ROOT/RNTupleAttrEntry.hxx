/// \file ROOT/RNTupleAttrEntry.hxx
/// \ingroup NTuple ROOT7
/// \author Giacomo Parolini <giacomo.parolini@cern.ch>
/// \date 2025-05-19
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RNTuple_AttrEntry
#define ROOT7_RNTuple_AttrEntry

#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/REntry.hxx>

#include <TError.h>

namespace ROOT::Experimental {

class RNTupleAttrSetWriter;
class RNTupleAttrSetReader;
class RNTupleFillContext;

namespace Internal::RNTupleAttributes {

const std::string kRangeStartName = "_rangeStart";
const std::string kRangeLenName = "_rangeLen";

} // namespace Internal::RNTupleAttributes

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

namespace Internal {
struct RNTupleAttrEntryPair {
   REntry &fMetaEntry;
   REntry &fScopedEntry;
   ROOT::RNTupleModel &fMetaModel;

   std::size_t Append();
   ROOT::DescriptorId_t GetModelId() const { return fMetaEntry.GetModelId(); }
};
} // namespace Internal

class RNTupleAttrEntry final {
   friend class ROOT::Experimental::RNTupleAttrSetWriter;
   friend class ROOT::Experimental::RNTupleAttrSetReader;
   friend class ROOT::Experimental::RNTupleFillContext;

   /// Entry containing the Attribute-specific fields (such as the entry range)
   std::unique_ptr<REntry> fMetaEntry;
   /* Entry containing to user-defined fields. It is "scoped" because the Attribute Model is organized like this:
    *
    *                      FieldZero
    *                          |
    *                _________/ \_________
    *               /        |            \
    *      _entryStart   _entryLen    RecordField
    *                                    / | \
    *                            (user defined fields)
    *
    *  and the ScopedEntry is scoped under RecordField, as if it were its top-level field.
    */
   std::unique_ptr<REntry> fScopedEntry;
   RNTupleAttrRange fRange;

   static std::unique_ptr<REntry> CreateScopedEntry(ROOT::RNTupleModel &model);
   static std::pair<std::unique_ptr<REntry>, std::unique_ptr<REntry>> CreateInternalEntries(ROOT::RNTupleModel &model);

   /// Creates a pending AttrEntry whose length is not determined yet.
   /// `metaEntry` is the entry containing the range data, `scopedEntry` contains the user-defined values.
   RNTupleAttrEntry(std::unique_ptr<REntry> metaEntry, std::unique_ptr<REntry> scopedEntry, ROOT::NTupleSize_t start)
      : fMetaEntry(std::move(metaEntry)),
        fScopedEntry(std::move(scopedEntry)),
        fRange(RNTupleAttrRange::FromStartLength(start, 0))
   {
   }

   /// Creates an AttrEntry with the given range.
   /// `metaEntry` is the entry containing the range data, `scopedEntry` contains the user-defined values.
   RNTupleAttrEntry(std::unique_ptr<REntry> entry, std::unique_ptr<REntry> scopedEntry, RNTupleAttrRange range)
      : fMetaEntry(std::move(entry)), fScopedEntry(std::move(scopedEntry)), fRange(range)
   {
   }

   std::size_t Append();
   // Required for RNTupleFillContext::FillNoFlushImpl()
   std::uint64_t GetModelId() const
   {
      R__ASSERT(fScopedEntry);
      return fScopedEntry->GetModelId();
   }

public:
   RNTupleAttrEntry(RNTupleAttrEntry &&) = default;
   RNTupleAttrEntry &operator=(RNTupleAttrEntry &&) = default;

   REntry *operator->()
   {
      R__ASSERT(fScopedEntry);
      return fScopedEntry.get();
   }
   const REntry *operator->() const
   {
      R__ASSERT(fScopedEntry);
      return fScopedEntry.get();
   }

   operator bool() const { return fScopedEntry && fMetaEntry; }

   RNTupleAttrRange GetRange() const { return fRange; }
};

} // namespace ROOT::Experimental
#endif
