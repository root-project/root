#ifndef ROOT_RSTREAMINGDS
#define ROOT_RSTREAMINGDS

#include "ROOT/RDataSource.hxx"
#include "ROOT/RMakeUnique.hxx"
#include <chrono>
#include <thread>
#include <type_traits>

/// A RDataSource that provides multiple entry ranges
class RStreamingDS : public ROOT::RDF::RDataSource {
   unsigned int fNSlots = 0u;
   unsigned int fCounter = 0u;
   const int fAns = 42;
   const int *fAnsPtr = &fAns;
   const std::vector<std::string> fColumnNames = {"ans"};

public:
   void SetNSlots(unsigned int nSlots) { fNSlots = nSlots; }
   const std::vector<std::string> &GetColumnNames() const { return fColumnNames; }
   bool HasColumn(std::string_view name) const { return std::string(name) == "ans" ? true : false; }
   std::string GetTypeName(std::string_view) const { return "int"; }
   std::vector<std::pair<ULong64_t, ULong64_t>> GetEntryRanges()
   {
      std::this_thread::sleep_for(std::chrono::milliseconds(10)); // simulate some latency
      if (fCounter == 4)
         return {};
      auto ranges = std::vector<std::pair<ULong64_t, ULong64_t>>(fNSlots);
      for (auto i = 0u; i < fNSlots; ++i)
         ranges[i] = std::make_pair(fCounter*fNSlots + i, fCounter*fNSlots + i + 1);
      ++fCounter;
      return ranges;
   }
   bool SetEntry(unsigned int, ULong64_t) {return true;}
   void Initialise() { fCounter = 0; }

   std::string GetDataSourceType() override {
      return "Streaming";
   }

protected:
   std::vector<void *> GetColumnReadersImpl(std::string_view name, const std::type_info &t) {
      if (t != typeid(int) && std::string(name) != "ans")
         throw;
      return std::vector<void *>(fNSlots, &fAnsPtr);
   }
};

#endif
