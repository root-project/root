/// \file ROOT/RNTupleAttributeEntry.hxx
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

#ifndef ROOT7_RNTuple_AttributeEntry
#define ROOT7_RNTuple_AttributeEntry

#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/REntry.hxx>

#include <TError.h>

namespace ROOT::Experimental {

class RNTupleAttributeSetWriter;
class RNTupleAttributeSetReader;
class RNTupleFillContext;

class RNTupleAttributeRange final {
   ROOT::NTupleSize_t fStart = 0;
   ROOT::NTupleSize_t fLength = 0;

   RNTupleAttributeRange(ROOT::NTupleSize_t start, ROOT::NTupleSize_t length) : fStart(start), fLength(length) {}

public:
   static RNTupleAttributeRange FromStartLength(ROOT::NTupleSize_t start, ROOT::NTupleSize_t length)
   {
      return RNTupleAttributeRange{start, length};
   }

   /// Creates an AttributeRange from [start, end), where `end` is one past the last valid entry of the range
   /// (`FromStartEnd(0, 10)` will create a range whose last valid index is 9).
   static RNTupleAttributeRange FromStartEnd(ROOT::NTupleSize_t start, ROOT::NTupleSize_t end)
   {
      R__ASSERT(end >= start);
      return RNTupleAttributeRange{start, end - start};
   }

   RNTupleAttributeRange() = default;

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

class RNTupleAttributeEntry final {
   friend class ROOT::Experimental::RNTupleAttributeSetWriter;
   friend class ROOT::Experimental::RNTupleAttributeSetReader;

   std::unique_ptr<REntry> fEntry;
   std::unique_ptr<REntry> fScopedEntry;
   RNTupleAttributeRange fRange;

   static std::unique_ptr<REntry> CreateScopedEntry(ROOT::RNTupleModel &model);
   static std::pair<std::unique_ptr<REntry>, std::unique_ptr<REntry>> CreateInternalEntries(ROOT::RNTupleModel &model);

   /// Creates a pending AttributeEntry whose length is not determined yet.
   /// `entry` is the "real" entry containing all the attribute data including the range, `scopedEntry` only contains
   /// the values of the user-defined values.
   RNTupleAttributeEntry(std::unique_ptr<REntry> entry, std::unique_ptr<REntry> scopedEntry, ROOT::NTupleSize_t start)
      : fEntry(std::move(entry)),
        fScopedEntry(std::move(scopedEntry)),
        fRange(RNTupleAttributeRange::FromStartLength(start, 0))
   {
   }

   /// Creates an AttributeEntry with the given range.
   /// `entry` is the "real" entry containing all the attribute data including the range, `scopedEntry` only contains
   /// the values of the user-defined values.
   RNTupleAttributeEntry(std::unique_ptr<REntry> entry, std::unique_ptr<REntry> scopedEntry,
                         RNTupleAttributeRange range)
      : fEntry(std::move(entry)), fScopedEntry(std::move(scopedEntry)), fRange(range)
   {
   }

public:
   RNTupleAttributeEntry(RNTupleAttributeEntry &&) = default;
   RNTupleAttributeEntry &operator=(RNTupleAttributeEntry &&) = default;

   std::size_t Append();

   ROOT::DescriptorId_t GetModelId() const { return fScopedEntry->GetModelId(); }

   template <typename T>
   std::shared_ptr<T> GetPtr(std::string_view name) const
   {
      return fScopedEntry->GetPtr<T>(name);
   }

   RNTupleAttributeRange GetRange() const { return fRange; }
};

class RNTupleAttributeEntryHandle final {
   friend class RNTupleAttributeSetWriter;

   RNTupleAttributeEntry *fRange = nullptr;

   explicit RNTupleAttributeEntryHandle(RNTupleAttributeEntry &range) : fRange(&range) {}

public:
   RNTupleAttributeEntryHandle(const RNTupleAttributeEntryHandle &) = delete;
   RNTupleAttributeEntryHandle &operator=(const RNTupleAttributeEntryHandle &) = delete;
   RNTupleAttributeEntryHandle(RNTupleAttributeEntryHandle &&other) { std::swap(fRange, other.fRange); }
   RNTupleAttributeEntryHandle &operator=(RNTupleAttributeEntryHandle &&other)
   {
      std::swap(fRange, other.fRange);
      return *this;
   }

   template <typename T>
   std::shared_ptr<T> GetPtr(std::string_view name)
   {
      if (R__unlikely(!fRange))
         throw ROOT::RException(R__FAIL("Called GetPtr() on invalid RNTupleAttributeEntryHandle"));
      return fRange->GetPtr<T>(name);
   }
};

} // namespace ROOT::Experimental
#endif
