#include <ROOT/TDFUtils.hxx>
#include <ROOT/TSeq.hxx>
#include <ROOT/TTrivialDS.hxx>

namespace ROOT {
namespace Experimental {
namespace TDF {

std::vector<void *> TTrivialDS::GetColumnReadersImpl(std::string_view, const std::type_info &)
{
   std::vector<void *> ret;
   for (auto i : ROOT::TSeqU(fNSlots)) {
      fCounterAddr[i] = &fCounter[i];
      ret.emplace_back((void *)(&fCounterAddr[i]));
   }
   return ret;
}

TTrivialDS::TTrivialDS(ULong64_t size) : fSize(size)
{
}

TTrivialDS::~TTrivialDS()
{
}

const std::vector<std::string> &TTrivialDS::GetColumnNames() const
{
   return fColNames;
}

bool TTrivialDS::HasColumn(std::string_view colName) const
{
   return colName == fColNames[0];
}

std::string TTrivialDS::GetTypeName(std::string_view) const
{
   return "ULong64_t";
}

const std::vector<std::pair<ULong64_t, ULong64_t>> &TTrivialDS::GetEntryRanges() const
{
   if (fEntryRanges.empty()) {
      throw std::runtime_error("No ranges are available. Did you set the number of slots?");
   }
   return fEntryRanges;
}

void TTrivialDS::SetEntry(unsigned int slot, ULong64_t entry)
{
   fCounter[slot] = entry;
}

void TTrivialDS::SetNSlots(unsigned int nSlots)
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
      (void)i;
   }
   // TODO: redistribute reminder to all slots
   fEntryRanges.back().second += fSize % fNSlots;
}

TDataFrame MakeTrivialDataFrame(ULong64_t size)
{
   std::unique_ptr<TTrivialDS> tds(new TTrivialDS(size));
   ROOT::Experimental::TDataFrame tdf(std::move(tds));
   return tdf;
}

} // ns TDF
} // ns Experimental
} // ns ROOT
