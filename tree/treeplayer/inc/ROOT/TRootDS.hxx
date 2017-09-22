#ifndef ROOT_TROOTTDS
#define ROOT_TROOTTDS

#include "ROOT/TDataFrame.hxx"
#include "ROOT/TDataSource.hxx"
#include <TChain.h>

#include <memory>

namespace ROOT {
namespace Experimental {
namespace TDF {

class TRootDS final : public ROOT::Experimental::TDF::TDataSource {
private:
   unsigned int fNSlots = 0U;
   std::string fTreeName;
   std::string fFileNameGlob;
   mutable TChain fModelChain; // Mutable needed for getting the column type name
   std::vector<double *> fAddressesToFree;
   std::vector<std::string> fListOfBranches;
   std::vector<std::pair<ULong64_t, ULong64_t>> fEntryRanges;
   std::vector<std::vector<void *>> fBranchAddresses; // first container-> slot, second -> column;
   std::vector<std::unique_ptr<TChain>> fChains;

   std::vector<void *> GetColumnReadersImpl(std::string_view, const std::type_info &);

public:
   TRootDS(std::string_view treeName, std::string_view fileNameGlob);
   ~TRootDS();
   std::string GetTypeName(std::string_view colName) const;
   const std::vector<std::string> &GetColumnNames() const;
   bool HasColumn(std::string_view colName) const;
   void InitSlot(unsigned int slot, ULong64_t firstEntry);
   const std::vector<std::pair<ULong64_t, ULong64_t>> &GetEntryRanges() const;
   void SetEntry(unsigned int slot, ULong64_t entry);
   void SetNSlots(unsigned int nSlots);
};

TDataFrame MakeRootDataFrame(std::string_view treeName, std::string_view fileNameGlob);

} // ns TDF
} // ns Experimental
} // ns ROOT

#endif
