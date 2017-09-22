#include <ROOT/TDFUtils.hxx>
#include <ROOT/TRootDS.hxx>
#include <ROOT/TSeq.hxx>
#include <TClass.h>
#include <TROOT.h>         // For the gROOTMutex
#include <TVirtualMutex.h> // For the R__LOCKGUARD

#include <algorithm>
#include <vector>

namespace ROOT {
namespace Experimental {
namespace TDF {

std::vector<void *> TRootDS::GetColumnReadersImpl(std::string_view name, const std::type_info &)
{
   const auto index =
      std::distance(fListOfBranches.begin(), std::find(fListOfBranches.begin(), fListOfBranches.end(), name));
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
   for (auto addr : fAddressesToFree) {
      delete addr;
   }
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
   TChain *chain;
   {
      R__LOCKGUARD(gROOTMutex);
      chain = new TChain(fTreeName.c_str());
   }
   chain->ResetBit(kMustCleanup);
   chain->Add(fFileNameGlob.c_str());
   chain->GetEntry(firstEntry);
   TString setBranches;
   for (auto i : ROOT::TSeqU(fListOfBranches.size())) {
      auto colName = fListOfBranches[i].c_str();
      auto &addr = fBranchAddresses.at(i).at(slot);
      auto typeName = GetTypeName(colName);
      auto typeClass = TClass::GetClass(typeName.c_str());
      if (typeClass) {
         chain->SetBranchAddress(colName, &addr, nullptr, typeClass, EDataType(0), true);
      } else {
         if (!addr) {
            addr = new double();
            fAddressesToFree.emplace_back((double *)addr);
         }
         chain->SetBranchAddress(colName, addr);
      }
   }
   fChains[slot].reset(chain);
}

const std::vector<std::pair<ULong64_t, ULong64_t>> &TRootDS::GetEntryRanges() const
{
   if (fEntryRanges.empty()) {
      throw std::runtime_error("No ranges are available. Did you set the number of slots?");
   }
   return fEntryRanges;
}

void TRootDS::SetEntry(unsigned int slot, ULong64_t entry)
{
   fChains[slot]->GetEntry(entry);
}

void TRootDS::SetNSlots(unsigned int nSlots)
{
   assert(0U == fNSlots && "Setting the number of slots even if the number of slots is different from zero.");

   fNSlots = nSlots;

   const auto nColumns = fListOfBranches.size();
   // Initialise the entire set of addresses
   fBranchAddresses.resize(nColumns, std::vector<void *>(fNSlots, nullptr));

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

TDataFrame MakeRootDataFrame(std::string_view treeName, std::string_view fileNameGlob)
{
   std::unique_ptr<TDF::TRootDS> tds(new TDF::TRootDS(treeName, fileNameGlob));
   ROOT::Experimental::TDataFrame tdf(std::move(tds));
   return tdf;
}

} // ns TDF
} // ns Experimental
} // ns ROOT
