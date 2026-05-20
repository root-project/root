#include "ROOT/RDF/ColumnReaderUtils.hxx"

ROOT::Detail::RDF::RColumnReaderBase *
ROOT::Internal::RDF::GetColumnReader(unsigned int slot, ROOT::Detail::RDF::RColumnReaderBase *defineOrVariationReader,
                                     ROOT::Detail::RDF::RLoopManager &lm, TTreeReader *treeReader,
                                     std::string_view colName, const std::type_info &ti)
{
   if (defineOrVariationReader != nullptr)
      return defineOrVariationReader;

   // Check if we already inserted a reader for this column in the dataset column readers (RDataSource or Tree/TChain
   // readers)
   auto *datasetColReader = lm.GetDatasetColumnReader(slot, std::string(colName), ti);
   if (datasetColReader != nullptr)
      return datasetColReader;

   return lm.AddDataSourceColumnReader(slot, colName, ti, treeReader);
}

std::vector<ROOT::Detail::RDF::RColumnReaderBase *> ROOT::Internal::RDF::GetUntypedColumnReaders(
   unsigned int slot, TTreeReader *treeReader, ROOT::Internal::RDF::RColumnRegister &colRegister,
   ROOT::Detail::RDF::RLoopManager &lm, const std::vector<std::string> &colNames,
   const std::vector<const std::type_info *> &colTypeIDs, const std::string &variationName)
{

   std::vector<ROOT::Detail::RDF::RColumnReaderBase *> readers;
   auto nCols = colNames.size();
   readers.reserve(nCols);
   for (decltype(nCols) i{}; i < nCols; i++) {
      readers.push_back(
         ROOT::Internal::RDF::GetColumnReader(slot, colRegister.GetReaderUnchecked(slot, colNames[i], variationName),
                                              lm, treeReader, colNames[i], *colTypeIDs[i]));
   }

   return readers;
}
