#include "ROOT/ML/RDatasetLoader.hxx"

namespace ROOT::Experimental::Internal::ML {

RDatasetLoader::RDatasetLoader(const std::vector<ROOT::RDF::RNode> &rdfs, const float validationSplit,
                               const std::vector<std::string> &cols, const std::vector<std::size_t> &vecSizes,
                               const float vecPadding, bool shuffle, const std::size_t setSeed)
   : f_rdfs(rdfs),
     fCols(cols),
     fVecSizes(vecSizes),
     fVecPadding(vecPadding),
     fValidationSplit(validationSplit),
     fShuffle(shuffle),
     fSetSeed(setSeed)
{
   fTensorOperators = std::make_unique<RFlat2DMatrixOperators>(fShuffle, fSetSeed);
   fNumCols = fCols.size();
   fSumVecSizes = std::accumulate(fVecSizes.begin(), fVecSizes.end(), 0);

   fNumDatasetCols = fNumCols + fSumVecSizes - fVecSizes.size();
}

//////////////////////////////////////////////////////////////////////////
/// \brief Split an individual dataframe into a training and validation dataset
/// \param[in] rdf Dataframe that will be split into training and validation
/// \param[in] TrainingDataset Tensor for the training dataset
/// \param[in] ValidationDataset Tensor for the validation dataset
void RDatasetLoader::SplitDataframe(ROOT::RDF::RNode &rdf, RFlat2DMatrix &TrainingDataset,
                                    RFlat2DMatrix &ValidationDataset)
{
   const bool NotFiltered = rdf.GetFilterNames().empty();

   // size the buffer from the cluster metadata, Count() is only the fallback for sources without it
   ROOT::RDF::RResultPtr<std::vector<ULong64_t>> Entries;
   std::size_t NumEntries = 0;
   if (NotFiltered) {
      try {
         for (const auto &cluster : ROOT::Internal::RDF::GetDatasetGlobalClusterBoundaries(rdf)) {
            NumEntries += cluster.second - cluster.first;
         }
      } catch (const std::runtime_error &) {
         // GetDatasetGlobalClusterBoundaries() throws when the RDataFrame has no cluster
         // metadata to query (a source other than TTree/RNTuple or no data source at all).
         // Those paths fall back to Count() below.
      }
      if (NumEntries == 0) {
         NumEntries = static_cast<std::size_t>(*rdf.Count());
      }
   } else {
      Entries = rdf.Take<ULong64_t>("rdfentry_");
      NumEntries = Entries->size();
      // add the last element in entries to not go out of range when filling chunks
      Entries->push_back((*Entries)[NumEntries - 1] + 1);
   }

   // number of training and validation entries after the split
   std::size_t NumValidationEntries = static_cast<std::size_t>(fValidationSplit * NumEntries);
   std::size_t NumTrainingEntries = NumEntries - NumValidationEntries;

   RFlat2DMatrix Dataset({NumEntries, fNumDatasetCols});

   if (NotFiltered) {
      auto values = ROOT::Internal::RDF::LoadCustomValues(rdf, fCols, fVecSizes, fVecPadding);
      std::copy(values->begin(), values->end(), Dataset.GetData());
   }

   else {
      std::size_t datasetEntry = 0;
      for (std::size_t j = 0; j < NumEntries; j++) {
         ROOT::Internal::RDF::ChangeBeginAndEndEntries(rdf, (*Entries)[j], (*Entries)[j + 1]);
         auto values = ROOT::Internal::RDF::LoadCustomValues(rdf, fCols, fVecSizes, fVecPadding);
         std::copy(values->begin(), values->end(), Dataset.GetData() + datasetEntry * fNumDatasetCols);
         datasetEntry++;
      }
   }

   // reset dataframe, only the filtered path changed the entry range
   if (!NotFiltered) {
      ROOT::Internal::RDF::ChangeBeginAndEndEntries(rdf, (*Entries)[0], (*Entries)[NumEntries]);
   }

   // copy out the validation tail, then shrink the (shuffled) buffer to the training rows and move it
   RFlat2DMatrix ShuffledDataset;
   RFlat2DMatrix &Source = fTensorOperators->ShuffleTensor(ShuffledDataset, Dataset);
   fTensorOperators->SliceTensor(ValidationDataset, Source, {{NumTrainingEntries, NumEntries}, {0, fNumDatasetCols}});
   Source.Resize(NumTrainingEntries, fNumDatasetCols);
   TrainingDataset = std::move(Source);
}

//////////////////////////////////////////////////////////////////////////
/// \brief Split the dataframes in a training and validation dataset
void RDatasetLoader::SplitDatasets()
{
   fNumTrainingEntries = 0;
   fNumValidationEntries = 0;

   for (auto &rdf : f_rdfs) {
      RFlat2DMatrix TrainingDataset;
      RFlat2DMatrix ValidationDataset;

      SplitDataframe(rdf, TrainingDataset, ValidationDataset);

      fNumTrainingEntries += TrainingDataset.GetRows();
      fNumValidationEntries += ValidationDataset.GetRows();

      fTrainingDatasets.push_back(std::move(TrainingDataset));
      fValidationDatasets.push_back(std::move(ValidationDataset));
   }
}

//////////////////////////////////////////////////////////////////////////
/// \brief Concatenate the datasets to a dataset
void RDatasetLoader::ConcatenateDatasets()
{
   if (fTrainingDatasets.size() == 1) {
      fTrainingDataset = std::move(fTrainingDatasets[0]);
      fValidationDataset = std::move(fValidationDatasets[0]);
   } else {
      fTensorOperators->ConcatenateTensors(fTrainingDataset, fTrainingDatasets);
      fTensorOperators->ConcatenateTensors(fValidationDataset, fValidationDatasets);
   }
   fTrainingDatasets.clear();
   fValidationDatasets.clear();
}

} // namespace ROOT::Experimental::Internal::ML
