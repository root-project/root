#ifndef ROOT_RARRAYSDS
#define ROOT_RARRAYSDS

#include "ROOT/RDataSource.hxx"
#include "ROOT/RDF/RColumnReaderBase.hxx"
#include <Rtypes.h>

#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>

class R__CLING_PTRCHECK(off) RArraysDSVarReader final : public ROOT::Detail::RDF::RColumnReaderBase {
   std::vector<int> *fPtr = nullptr;
   void *GetImpl(Long64_t) final { return fPtr; }

public:
   RArraysDSVarReader(std::vector<int> &v) : fPtr(&v) {}
};

class R__CLING_PTRCHECK(off) RArraysDSVarSizeReader final : public ROOT::Detail::RDF::RColumnReaderBase {
   std::vector<int> *fPtr = nullptr;
   std::size_t fSize = 0;
   void *GetImpl(Long64_t) final
   {
      fSize = fPtr->size();
      return &fSize;
   }

public:
   RArraysDSVarSizeReader(std::vector<int> &v) : fPtr(&v) {}
};

/// A RDataSource to test the `#var` feature
class RArraysDS : public ROOT::RDF::RDataSource {
   std::vector<int> fVar = {42};
   std::vector<std::string> fColumnNames = {"R_rdf_sizeof_var", "var"};
   std::vector<std::pair<ULong64_t, ULong64_t>> fRanges = {{0ull, 1ull}};

   bool IsSizeColumn(std::string_view colName) const { return colName.substr(0, 13) == "R_rdf_sizeof_"; }

public:
   void SetNSlots(unsigned int) final { }

   const std::vector<std::string> &GetColumnNames() const final { return fColumnNames; }

   bool HasColumn(std::string_view name) const final
   {
      return name == "var" || name == "R_rdf_sizeof_var";
   }

   std::string GetTypeName(std::string_view name) const final
   {
      return IsSizeColumn(name) ? "std::size_t" : "std::vector<int>";
   }

   std::vector<std::pair<ULong64_t, ULong64_t>> GetEntryRanges() final
   {
      return std::move(fRanges);
   }

   bool SetEntry(unsigned int, ULong64_t) final { return true; }

   void Initialize() final { fRanges = {{0ull, 1ull}}; }

   std::string GetLabel() final { return "ArraysDS"; }

protected:
   std::vector<void *> GetColumnReadersImpl(std::string_view, const std::type_info &) final
   {
      // we use the new column reader API
      return {};
   }

   std::unique_ptr<ROOT::Detail::RDF::RColumnReaderBase>
   GetColumnReaders(unsigned int /*slot*/, std::string_view name, const std::type_info &) final
   {
      return IsSizeColumn(name)
                ? std::unique_ptr<ROOT::Detail::RDF::RColumnReaderBase>(new RArraysDSVarSizeReader(fVar))
                : std::unique_ptr<ROOT::Detail::RDF::RColumnReaderBase>(new RArraysDSVarReader(fVar));
   }
};

#endif
