// Author: Enrico Guiraud, Danilo Piparo CERN  9/2017

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RTRIVIALDS
#define ROOT_RTRIVIALDS

#include "ROOT/RDF/RInterface.hxx"
#include "ROOT/RDataSource.hxx"

namespace ROOT::Internal::RDF {
class R__CLING_PTRCHECK(off) RTrivialDSColumnReader final : public ROOT::Detail::RDF::RColumnReaderBase {
   ULong64_t *fValuePtr;
   ROOT::RVec<ULong64_t> fCachedValues;
   std::uint64_t fLastEntry{std::numeric_limits<std::uint64_t>::max()};
   bool fSkipEvenEntries;
   ROOT::RVec<bool> fIsCachedValueValidAtIdx;
   void *GetImpl(std::size_t entryInBulk) final
   {
      if (fIsCachedValueValidAtIdx[entryInBulk])
         return &fCachedValues[entryInBulk];

      return nullptr;
   }
   void LoadImpl(const ROOT::Internal::RDF::RMaskedEntryRange &mask) final
   {
      if (fLastEntry == mask.GetFirstEntry())
         return;

      if (fSkipEvenEntries && mask.GetBulkSize() > 1)
         throw std::runtime_error(
            "Execution of RTrivialDS skipping even entries with bulk size greater than one is not supported.");

      fLastEntry = mask.GetFirstEntry();
      fCachedValues = ROOT::RVec<ULong64_t>(mask.GetBulkSize());
      fIsCachedValueValidAtIdx = ROOT::RVec<bool>(mask.GetBulkSize());
      for (std::uint64_t entry = fLastEntry; entry < fLastEntry + static_cast<std::uint64_t>(mask.GetBulkSize());
           entry++) {
         fIsCachedValueValidAtIdx[entry - fLastEntry] = (fSkipEvenEntries && entry % 2 == 0) ? false : true;
      }
      std::iota(fCachedValues.begin(), fCachedValues.end(), *fValuePtr);
   }

public:
   RTrivialDSColumnReader(ULong64_t *valuePtr, bool skipEvenEntries)
      : fValuePtr(valuePtr), fSkipEvenEntries(skipEvenEntries)
   {
   }
};
} // namespace ROOT::Internal::RDF

namespace ROOT {

namespace RDF {

/// \brief A simple data-source implementation, for demo purposes.
///
/// Constructing an RDataFrame as `RDataFrame(nEntries)` is a superior alternative.
/// If size is std::numeric_limits<ULong64_t>::max(), this acts as an infinite data-source:
/// it returns entries from GetEntryRanges forever or until a Range stops the event loop (for test purposes).
class RTrivialDS final : public ROOT::RDF::RDataSource {
private:
   ULong64_t fSize = 0ULL;
   bool fSkipEvenEntries = false;
   std::vector<std::pair<ULong64_t, ULong64_t>> fEntryRanges;
   std::vector<std::string> fColNames{"col0"};
   std::vector<ULong64_t> fCounter;
   std::vector<ULong64_t *> fCounterAddr;
   std::vector<void *> GetColumnReadersImpl(std::string_view name, const std::type_info &) final;

protected:
   std::string AsString() final { return "trivial data source"; };

public:
   RTrivialDS(ULong64_t size, bool skipEvenEntries = false);
   /// This ctor produces a data-source that returns infinite entries
   RTrivialDS();
   // Rule of five
   RTrivialDS(const RTrivialDS &) = delete;
   RTrivialDS &operator=(const RTrivialDS &) = delete;
   RTrivialDS(RTrivialDS &&) = delete;
   RTrivialDS &operator=(RTrivialDS &&) = delete;
   ~RTrivialDS() final = default;

   const std::vector<std::string> &GetColumnNames() const final;
   bool HasColumn(std::string_view colName) const final;
   std::string GetTypeName(std::string_view) const final;
   std::vector<std::pair<ULong64_t, ULong64_t>> GetEntryRanges() final;
   bool SetEntry(unsigned int slot, ULong64_t entry) final;
   void SetNSlots(unsigned int nSlots) final;
   void Initialize() final;
   std::string GetLabel() final;

   std::unique_ptr<ROOT::Detail::RDF::RColumnReaderBase>
   GetColumnReaders(unsigned int slot, std::string_view colName, const std::type_info &tid) final;
};

/// \brief Make a RDF wrapping a RTrivialDS with the specified amount of entries.
///
/// Constructing an RDataFrame as `RDataFrame(nEntries)` is a superior alternative.
/// If size is std::numeric_limits<ULong64_t>::max(), this acts as an infinite data-source:
/// it returns entries from GetEntryRanges forever or until a Range stops the event loop (for test purposes).
RInterface<RDFDetail::RLoopManager> MakeTrivialDataFrame(ULong64_t size, bool skipEvenEntries = false);
/// \brief Make a RDF wrapping a RTrivialDS with infinite entries, for demo purposes.
RInterface<RDFDetail::RLoopManager> MakeTrivialDataFrame();

} // namespace RDF

} // namespace ROOT

#endif
