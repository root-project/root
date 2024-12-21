#include <ROOT/RDF/ColumnReaderUtils.hxx>

ROOT::Detail::RDF::RColumnReaderBase *
ROOT::Internal::RDF::GetUntypedColumnReader(unsigned int slot,
                                            ROOT::Internal::RDF::RColumnReaderBase *defineOrVariationReader,
                                            ROOT::Detail::RDF::RLoopManager &lm, TTreeReader *r,
                                            const std::string &colName)
{
   if (defineOrVariationReader != nullptr)
      return defineOrVariationReader;

   assert(r != nullptr && "We could not find a reader for this column, this should never happen at this point.");
   assert(r->GetTree() != nullptr &&
          "There is no tree associated with the TTreeReader, this should never happen at this point.");

   auto branchTypeName = ROOT::Internal::RDF::GetBranchOrLeafTypeName(*r->GetTree(), colName);
   auto treeColReader = std::make_unique<RTreeUntypedColumnReader>(*r, colName, branchTypeName);
   return lm.AddTreeColumnReader(slot, colName, std::move(treeColReader),
                                 ROOT::Internal::RDF::TypeName2TypeID(branchTypeName));
}

std::vector<ROOT::Detail::RDF::RColumnReaderBase *>
ROOT::Internal::RDF::GetUntypedColumnReaders(unsigned int slot, TTreeReader *r,
                                             const ROOT::Internal::RDF::RColumnReadersUntypedSnapshotInfo &colInfo,
                                             const std::string &variationName)
{
   // see RColumnReadersInfo for why we pass these arguments like this rather than directly as function arguments
   const auto &colNames = colInfo.fColNames;
   auto &lm = colInfo.fLoopManager;
   auto &colRegister = colInfo.fColRegister;

   std::vector<ROOT::Detail::RDF::RColumnReaderBase *> readers;
   auto &&nCols = colNames.size();
   readers.reserve(nCols);
   for (decltype(nCols) i{}; i < nCols; i++) {
      readers.push_back(ROOT::Internal::RDF::GetUntypedColumnReader(
         slot, colRegister.GetReaderUnchecked(slot, colNames[i], variationName), lm, r, colNames[i]));
   }

   return readers;
}