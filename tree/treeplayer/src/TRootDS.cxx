#include <TClass.h>
#include <ROOT/TDFUtils.hxx>
#include <ROOT/TRootDS.hxx>
#include <ROOT/TSeq.hxx>

#include <algorithm>
#include <vector>

namespace ROOT {
namespace Experimental {
namespace TDF {

std::vector<void *> TRootDS::GetColumnReadersImpl(std::string_view name, const std::type_info &)
{

   const auto &colNames = GetColumnNames();

   if (fBranchAddresses.empty()) {
      auto nColumns = colNames.size();
      // Initialise the entire set of addresses
      fBranchAddresses.resize(nColumns, std::vector<void *>(fNSlots));
   }

   const auto index = std::distance(colNames.begin(), std::find(colNames.begin(), colNames.end(), name));
   std::vector<void *> ret(fNSlots);
   for (auto slot : ROOT::TSeqU(fNSlots)) {
      ret[slot] = (void *)&fBranchAddresses[index][slot];
   }
   return ret;
}

TRootDS::TRootDS(std::string_view treeName, std::string_view fileNameGlob)
   : fTreeName(treeName), fFileNameGlob(fileNameGlob), fModelChain(std::string(treeName).c_str())
{
   fModelChain.Add(fFileNameGlob.c_str());

   auto &lob = *fModelChain.GetListOfBranches();
   fListOfBranches.resize(lob.GetEntries());
   std::transform(lob.begin(), lob.end(), fListOfBranches.begin(), [](TObject *o) { return o->GetName(); });
}

TRootDS::~TRootDS()
{
}

std::string TRootDS::GetTypeName(std::string_view colName) const
{
   if (!HasColumn(colName)) {
      std::string e = "The dataset does not have column ";
      e += colName;
      throw std::runtime_error(e);
   }
   // TODO: we need to factor out the routine for the branch alone...
   // Maybe a cache for the names?
   auto typeName = ROOT::Internal::TDF::ColumnName2ColumnTypeName(std::string(colName).c_str(), &fModelChain,
                                                                  nullptr /*TCustomColumnBase here*/);
   // We may not have yet loaded the library where the dictionary of this type
   // is
   TClass::GetClass(typeName.c_str());
   return typeName;
}

const std::vector<std::string> &TRootDS::GetColumnNames() const
{
   return fListOfBranches;
}

bool TRootDS::HasColumn(std::string_view colName) const
{
   if (!fListOfBranches.empty())
      GetColumnNames();
   return fListOfBranches.end() != std::find(fListOfBranches.begin(), fListOfBranches.end(), colName);
}

void TRootDS::InitSlot(unsigned int slot, ULong64_t firstEntry)
{
   auto chain = new TChain(fTreeName.c_str());
   fChains[slot].reset(chain);
   chain->Add(fFileNameGlob.c_str());
   chain->GetEntry(firstEntry);
   for (auto i : ROOT::TSeqU(fListOfBranches.size())) {
      auto colName = fListOfBranches[i].c_str();
      auto &addr = fBranchAddresses[i][slot];
      auto typeName = GetTypeName(colName);
      auto isClass = nullptr != TClass::GetClass(typeName.c_str());
      if (isClass) {
         chain->SetBranchAddress(colName, &addr);
      } else {
         if (!addr) {
            addr = new double(); // who frees this :) ?
         }
         chain->SetBranchAddress(colName, addr);
      }
   }
}

const std::vector<std::pair<ULong64_t, ULong64_t>> &TRootDS::GetEntryRanges() const
{
   if (fEntryRanges.empty()) {
      throw std::runtime_error("No ranges are available. Did you set the number of slots?");
   }
   return fEntryRanges;
}

void TRootDS::SetEntry(ULong64_t entry, unsigned int slot)
{
   fChains[slot]->GetEntry(entry);
}

void TRootDS::SetNSlots(unsigned int nSlots)
{
   assert(0U == fNSlots && "Setting the number of slots even if the number of slots is different from zero.");

   fNSlots = nSlots;
   fChains.resize(fNSlots);
   auto nentries = fModelChain.GetEntries();
   auto chunkSize = nentries / fNSlots;
   auto reminder = 1U == fNSlots ? 0 : nentries % fNSlots;
   auto start = 0UL;
   auto end = 0UL;
   for (auto i : ROOT::TSeqU(fNSlots)) {
      start = end;
      end += chunkSize;
      fEntryRanges.emplace_back(start, end);
      (void)i;
   }
   fEntryRanges.back().second += reminder;
}
} // ns TDF
} // ns Experimental
} // ns ROOT
