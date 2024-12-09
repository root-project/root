#ifndef ROOT_RNONCOPIABLECOLUMNDS
#define ROOT_RNONCOPIABLECOLUMNDS

#include "ROOT/RDataSource.hxx"
#include <string>
#include <vector>

class RNonCopiable {
public:
   using type = unsigned int;
   unsigned int fValue = 42;
   RNonCopiable(const RNonCopiable &) = delete;
   RNonCopiable() = default;
};

class RNonCopiableColumnDS final : public ROOT::RDF::RDataSource {
private:
   std::vector<std::pair<ULong64_t, ULong64_t>> fEntryRanges = {{0UL, 1UL}};
   std::vector<std::string> fColNames{fgColumnName};
   RNonCopiable fNonCopiable;
   RNonCopiable *fCounterAddr = &fNonCopiable;
   std::vector<void *> GetColumnReadersImpl(std::string_view, const std::type_info &) final
   {
      std::vector<void *> ret{(void *)(&fCounterAddr)};
      return ret;
   }

public:
   using NonCopiable_t = RNonCopiable;
   constexpr const static auto fgColumnName = "nonCopiable";

   RNonCopiableColumnDS() = default;
   // Rule of five
   RNonCopiableColumnDS(const RNonCopiableColumnDS &) = delete;
   RNonCopiableColumnDS &operator=(const RNonCopiableColumnDS &) = delete;
   RNonCopiableColumnDS(RNonCopiableColumnDS &&) = delete;
   RNonCopiableColumnDS &operator=(RNonCopiableColumnDS &&) = delete;
   ~RNonCopiableColumnDS() final = default;

   const std::vector<std::string> &GetColumnNames() const final { return fColNames; };
   bool HasColumn(std::string_view colName) const final { return colName == fColNames[0]; };
   std::string GetTypeName(std::string_view) const final { return "RNonCopiable"; };
   std::vector<std::pair<ULong64_t, ULong64_t>> GetEntryRanges() final
   {
      auto entryRanges(std::move(fEntryRanges)); // empty fEntryRanges
      return entryRanges;
   };
   bool SetEntry(unsigned int, ULong64_t) final { return true;};
   void SetNSlots(unsigned int) final {};
   std::string GetLabel() final {
      return "NonCopiableColumnDS";
   }
};

#endif
