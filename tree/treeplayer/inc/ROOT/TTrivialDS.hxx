#ifndef ROOT_TTRIVIALTDS
#define ROOT_TTRIVIALTDS

#include "ROOT/TDataFrame.hxx"
#include "ROOT/TDataSource.hxx"

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
   std::vector<void *> GetColumnReadersImpl(std::string_view name, const std::type_info &);

public:
   TTrivialDS(ULong64_t size);
   ~TTrivialDS();
   const std::vector<std::string> &GetColumnNames() const;
   bool HasColumn(std::string_view colName) const;
   std::string GetTypeName(std::string_view) const;
   const std::vector<std::pair<ULong64_t, ULong64_t>> &GetEntryRanges() const;
   void SetEntry(unsigned int slot, ULong64_t entry);
   void SetNSlots(unsigned int nSlots);
};

TDataFrame MakeTrivialDataFrame(ULong64_t size);

} // ns TDF
} // ns Experimental
} // ns ROOT

#endif
