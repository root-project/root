#include "ROOT/TDataSource.hxx"
#include <string>
#include <vector>
#include <map>

class TNonCopiable {
public:
   using type = unsigned int;
   unsigned int fValue = 42;
   TNonCopiable(const TNonCopiable &) = delete;
   TNonCopiable() = default;
};

class NonCopiableDS final : public ROOT::Experimental::TDF::TDataSource {
private:
   std::vector<std::pair<ULong64_t, ULong64_t>> fEntryRanges = {{0UL, 1UL}};
   std::vector<std::string> fColNames{fgColumnName};
   TNonCopiable fNonCopiable;
   TNonCopiable *fCounterAddr = &fNonCopiable;
   std::vector<void *> GetColumnReadersImpl(std::string_view, const std::type_info &)
   {
      std::vector<void *> ret{(void *)(&fCounterAddr)};
      return ret;
   }

public:
   using NonCopiable_t = TNonCopiable;
   constexpr const static auto fgColumnName = "nonCopiable";
   NonCopiableDS(){};
   ~NonCopiableDS(){};
   const std::vector<std::string> &GetColumnNames() const { return fColNames; };
   bool HasColumn(std::string_view colName) const { return colName == fColNames[0]; };
   std::string GetTypeName(std::string_view) const { return "TNonCopiable"; };
   std::vector<std::pair<ULong64_t, ULong64_t>> GetEntryRanges()
   {
      auto entryRanges(std::move(fEntryRanges)); // empty fEntryRanges
      return entryRanges;
   };
   void SetEntry(unsigned int, ULong64_t){};
   void SetNSlots(unsigned int){};
};
