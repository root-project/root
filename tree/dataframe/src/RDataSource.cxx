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
         [&lm, &slotStack, &entryCount](const std::pair<ULong64_t, ULong64_t> &range) {
            lm.DataSourceThreadTask(range, slotStack, entryCount);
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
