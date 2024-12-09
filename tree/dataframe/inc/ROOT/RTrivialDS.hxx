// Author: Enrico Guiraud, Danilo Piparo CERN  9/2017

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RTRIVIALTDS
#define ROOT_RTRIVIALTDS

#include "ROOT/RDF/RInterface.hxx"
#include "ROOT/RDataSource.hxx"

namespace ROOT {

namespace RDF {

/// \brief A simple data-source implementation, for demo purposes.
///
/// Constructing an RDataFrame as `RDataFrame(nEntries)` is a superior alternative.
/// If size is std::numeric_limits<ULong64_t>::max(), this acts as an infinite data-source:
/// it returns entries from GetEntryRanges forever or until a Range stops the event loop (for test purposes).
class RTrivialDS final : public ROOT::RDF::RDataSource {
private:
   unsigned int fNSlots = 0U;
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
};

/// \brief Make a RDF wrapping a RTrivialDS with the specified amount of entries.
///
/// Constructing an RDataFrame as `RDataFrame(nEntries)` is a superior alternative.
/// If size is std::numeric_limits<ULong64_t>::max(), this acts as an infinite data-source:
/// it returns entries from GetEntryRanges forever or until a Range stops the event loop (for test purposes).
RInterface<RDFDetail::RLoopManager> MakeTrivialDataFrame(ULong64_t size, bool skipEvenEntries = false);
/// \brief Make a RDF wrapping a RTrivialDS with infinite entries, for demo purposes.
RInterface<RDFDetail::RLoopManager> MakeTrivialDataFrame();

} // ns RDF

} // ns ROOT

#endif
