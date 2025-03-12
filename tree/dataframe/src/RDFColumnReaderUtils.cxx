#include "ROOT/RDF/ColumnReaderUtils.hxx"
#include "ROOT/RDF/RTreeColumnReader.hxx"

namespace {
std::tuple<bool, std::string, ROOT::Internal::RDF::RTreeUntypedArrayColumnReader::ECollectionType>
GetCollectionInfo(const std::string &typeName)
{
   const auto beginType = typeName.substr(0, typeName.find_first_of('<') + 1);

   // Find TYPE from ROOT::RVec<TYPE>
   if (auto pos = beginType.find("RVec<"); pos != std::string::npos) {
      const auto begin = pos + 5;
      const auto end = typeName.find_last_of('>');
      const auto innerTypeName = typeName.substr(begin, end - begin);
      if (innerTypeName == "bool")
         return {true, innerTypeName, ROOT::Internal::RDF::RTreeUntypedArrayColumnReader::ECollectionType::kRVecBool};
      else
         return {true, innerTypeName, ROOT::Internal::RDF::RTreeUntypedArrayColumnReader::ECollectionType::kRVec};
   }

   // Find TYPE from std::array<TYPE,N>
   if (auto pos = beginType.find("array<"); pos != std::string::npos) {
      const auto begin = pos + 6;
      const auto end = typeName.find_last_of('>');
      const auto arrTemplArgs = typeName.substr(begin, end - begin);
      const auto lastComma = arrTemplArgs.find_last_of(',');
      return {true, arrTemplArgs.substr(0, lastComma),
              ROOT::Internal::RDF::RTreeUntypedArrayColumnReader::ECollectionType::kStdArray};
   }

   return {false, "", ROOT::Internal::RDF::RTreeUntypedArrayColumnReader::ECollectionType::kRVec};
}
} // namespace

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

   assert(treeReader != nullptr &&
          "We could not find a reader for this column, this should never happen at this point.");

   // Make a RTreeColumnReader for this column and insert it in RLoopManager's map
   auto createColReader = [&]() -> std::unique_ptr<ROOT::Detail::RDF::RColumnReaderBase> {
      if (ti == typeid(void))
         return std::make_unique<ROOT::Internal::RDF::RTreeOpaqueColumnReader>(*treeReader, colName);

      const auto typeName = ROOT::Internal::RDF::TypeID2TypeName(ti);
      if (auto &&[toConvert, innerTypeName, collType] = GetCollectionInfo(typeName); toConvert)
         return std::make_unique<ROOT::Internal::RDF::RTreeUntypedArrayColumnReader>(*treeReader, colName,
                                                                                     innerTypeName, collType);
      else
         return std::make_unique<ROOT::Internal::RDF::RTreeUntypedValueColumnReader>(*treeReader, colName, typeName);
   };

   return lm.AddTreeColumnReader(slot, std::string(colName), createColReader(), ti);
}
