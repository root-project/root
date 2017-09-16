#ifndef ROOT_TTRIVIALTDS
#define ROOT_TTRIVIALTDS

#include "ROOT/TDataFrame.hxx"
#include "ROOT/TDataSource.hxx"
#include "ROOT/TSeq.hxx"

namespace ROOT {
namespace Experimental {
namespace TDF {

class TTrivialDS final : public ROOT::Experimental::TDF::TDataSource {
private:
   unsigned int fNSlots = 0U;
   ULong64_t fSize = 0ULL;
   std::vector<std::pair<ULong64_t, ULong64_t>> fEntryRanges;
   std::vector<std::string> fColNames{"col0"};
   std::vector<ULong64_t> fCounter;
   std::vector<ULong64_t *> fCounterAddr;
   std::vector<void *> GetColumnReadersImpl(std::string_view name, const std::type_info &)
   {
      std::vector<void *> ret;
      for (auto i : ROOT::TSeqU(fNSlots)) {
         fCounterAddr[i] = &fCounter[i];
         ret.emplace_back((void *)(&fCounterAddr[i]));
      }
      return ret;
   }

public:
   TTrivialDS(ULong64_t size) : fSize(size) {}

   ~TTrivialDS() {}

   const std::vector<std::string> &GetColumnNames() const { return fColNames; }

   bool HasColumn(std::string_view colName) const { return colName == fColNames[0]; }

   std::string GetTypeName(std::string_view) const { return "ULong64_t"; }

   const std::vector<std::pair<ULong64_t, ULong64_t>> &GetEntryRanges() const
   {
      if (fEntryRanges.empty()) {
         throw std::runtime_error("No ranges are available. Did you set the number of slots?");
      }
      return fEntryRanges;
   }
   void SetEntry(ULong64_t entry, unsigned int slot) { fCounter[slot] = entry; }

   void SetNSlots(unsigned int nSlots)
   {
       assert(0U == fNSlots && "Setting the number of slots even if the number of slots is different from zero.");

      fNSlots = nSlots;
      fCounter.resize(fNSlots);
      fCounterAddr.resize(fNSlots);

      auto chunkSize = fSize / fNSlots;
      auto start = 0UL;
      auto end = 0UL;
      for (auto i : ROOT::TSeqUL(fNSlots)) {
         start = end;
         end += chunkSize;
         fEntryRanges.emplace_back(start, end);
      }
      // TODO: redistribute reminder to all slots
      fEntryRanges.back().second += fSize % fNSlots;
   };
};
}
}
}

#endif
