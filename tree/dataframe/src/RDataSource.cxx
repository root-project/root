#include <ROOT/RDataSource.hxx>
#include <ROOT/RDF/RSample.hxx>
#include <ROOT/RDF/RSampleInfo.hxx>
#include <ROOT/RDF/RLoopManager.hxx>

#ifdef R__USE_IMT
#include <ROOT/RSlotStack.hxx>
#include <ROOT/TThreadExecutor.hxx>
#endif

ROOT::RDF::RSampleInfo ROOT::RDF::RDataSource::CreateSampleInfo(
   unsigned int, const std::unordered_map<std::string, ROOT::RDF::Experimental::RSample *> &) const
{
   return ROOT::RDF::RSampleInfo{};
}

void ROOT::RDF::RDataSource::ProcessMT(ROOT::Detail::RDF::RLoopManager &lm)
{
#ifdef R__USE_IMT
   ROOT::Internal::RSlotStack slotStack(fNSlots);
   std::atomic<ULong64_t> entryCount(0ull);
   ROOT::TThreadExecutor pool;

   auto ranges = GetEntryRanges();
   while (!ranges.empty()) {
      pool.Foreach(
         [this, &lm, &slotStack, &entryCount](const std::pair<ULong64_t, ULong64_t> &range) {
            lm.DataSourceThreadTask(range, slotStack, entryCount,
                                    [this](std::size_t maxBulkSize, std::uint64_t bulkBeginEntry,
                                           std::uint64_t rangeEndEntry, void *extraArg) {
                                       return this->PickBulkSize(maxBulkSize, bulkBeginEntry, rangeEndEntry, extraArg);
                                    });
         },
         ranges);
      ranges = GetEntryRanges();
   }

   if (fGlobalEntryRange.has_value()) {
      auto &&[begin, end] = fGlobalEntryRange.value();
      auto &&processedEntries = entryCount.load();
      if ((end - begin) > processedEntries) {
         Warning("RDataFrame::Run",
                 "RDataFrame stopped processing after %lld entries, whereas an entry range (begin=%lld,end=%lld) was "
                 "requested. Consider adjusting the end value of the entry range to a maximum of %lld.",
                 processedEntries, begin, end, begin + processedEntries);
      }
   }
#else
   (void)lm;
#endif
}

std::size_t ROOT::Internal::RDF::PickBulkSize(ROOT::RDF::RDataSource &ds, std::size_t maxBulkSize,
                                              std::uint64_t &bulkBeginEntry, std::uint64_t &rangeEndEntry,
                                              void *extraArg)
{
   return ds.PickBulkSize(maxBulkSize, bulkBeginEntry, rangeEndEntry, extraArg);
}

std::size_t ROOT::RDF::RDataSource::PickBulkSize(std::size_t maxBulkSize, std::uint64_t &bulkBeginEntry,
                                                 std::uint64_t &rangeEndEntry, void *)
{
   return std::min(maxBulkSize, static_cast<std::size_t>(rangeEndEntry - bulkBeginEntry));
}

bool ROOT::Internal::RDF::CheckValidRead(ROOT::RDF::RDataSource &ds, unsigned int slot, std::uint64_t entry)
{
   return ds.CheckValidRead(slot, entry);
}

ROOT::Internal::RDF::RMaskedEntryRange ROOT::Internal::RDF::CheckFilters(ROOT::RDF::RDataSource &ds, unsigned int slot,
                                                                         Long64_t bulkBeginEntry, std::size_t bulkSize)
{
   return ds.CheckFilters(slot, bulkBeginEntry, bulkSize);
}

ROOT::Internal::RDF::RMaskedEntryRange
ROOT::RDF::RDataSource::CheckFilters(unsigned int, Long64_t bulkBeginEntry, std::size_t bulkSize)
{
   return ROOT::Internal::RDF::RMaskedEntryRange{bulkSize, true, static_cast<std::uint64_t>(bulkBeginEntry)};
}
