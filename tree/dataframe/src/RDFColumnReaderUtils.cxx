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
