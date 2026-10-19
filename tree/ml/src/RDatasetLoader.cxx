#include "RDatasetLoader.hxx"

namespace ROOT::Experimental::Internal::ML {

RDatasetLoader::RDatasetLoader(const std::vector<ROOT::RDF::RNode> &rdfs, const float validationSplit,
                               const std::vector<std::string> &cols, const std::vector<std::size_t> &vecSizes,
                               const float vecPadding, bool shuffle, const std::size_t setSeed)
   : fValidationSplit(validationSplit),
     fVecSizes(vecSizes),
     fVecPadding(vecPadding),
     f_rdfs(rdfs),
     fCols(cols),
     fSetSeed(setSeed),
     fShuffle(shuffle)
{
   fTensorOperators = std::make_unique<RFlat2DMatrixOperators>(fShuffle, fSetSeed);
   fNumCols = fCols.size();
   fSumVecSizes = std::accumulate(fVecSizes.begin(), fVecSizes.end(), 0);

   fNumDatasetCols = fNumCols + fSumVecSizes - fVecSizes.size();
}

//////////////////////////////////////////////////////////////////////////
/// \brief Load all the entries of an individual dataframe
/// \param[in] rdf Dataframe to load
/// \return Tensor with the dataset
RFlat2DMatrix RDatasetLoader::LoadDataframe(ROOT::RDF::RNode &rdf)
{
   const bool NotFiltered = rdf.GetFilterNames().empty();

   // size the buffer from the cluster metadata, Count() is only the fallback for sources without it. With filters
   // this is an upper bound: the memory reserved for the entries that don't pass them is never touched
   std::size_t NumEntries = 0;
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

   // reserve the dataset and let the values be appended to it
   RFlat2DMatrix Dataset;
   Dataset.fRVec.reserve(NumEntries * fNumDatasetCols);

   // a single event loop, the filters only let the passing entries through
   ROOT::Internal::RDF::LoadCustomValues(rdf, fCols, fVecSizes, fVecPadding, Dataset.fRVec);
   if (!NotFiltered) {
      NumEntries = Dataset.GetSize() / fNumDatasetCols;
   }
   Dataset.Resize(NumEntries, fNumDatasetCols);

   return Dataset;
}

//////////////////////////////////////////////////////////////////////////
/// \brief Split a loaded dataset into training and validation entries, appended to the given datasets
/// \param[in] Dataset Tensor with the loaded dataset
/// \param[in] TrainingDataset Tensor for the training dataset
/// \param[in] ValidationDataset Tensor for the validation dataset
void RDatasetLoader::SplitDataset(RFlat2DMatrix &Dataset, RFlat2DMatrix &TrainingDataset,
                                  RFlat2DMatrix &ValidationDataset)
{
   // number of training and validation entries after the split
   std::size_t NumEntries = Dataset.GetRows();
   std::size_t NumValidationEntries = static_cast<std::size_t>(fValidationSplit * NumEntries);
   std::size_t NumTrainingEntries = NumEntries - NumValidationEntries;

   // read the entries in shuffled order, instead of copying the dataset into a shuffled one first
   const auto permutation = fTensorOperators->MakePermutation(NumEntries);

   ValidationDataset.fRVec.reserve(ValidationDataset.GetSize() + NumValidationEntries * fNumDatasetCols);
   RFlat2DMatrixOperators::AppendRows(ValidationDataset, Dataset, permutation, NumTrainingEntries, NumEntries);

   TrainingDataset.fRVec.reserve(TrainingDataset.GetSize() + NumTrainingEntries * fNumDatasetCols);
   RFlat2DMatrixOperators::AppendRows(TrainingDataset, Dataset, permutation, 0, NumTrainingEntries);
}

//////////////////////////////////////////////////////////////////////////
/// \brief Split the dataframes in a training and validation dataset
/// \param[in] concatenate Whether to concatenate the datasets of all the dataframes into one
void RDatasetLoader::SplitDatasets(bool concatenate)
{
   std::vector<RFlat2DMatrix> Datasets;
   for (auto &rdf : f_rdfs) {
      Datasets.push_back(LoadDataframe(rdf));
   }

   if (concatenate && Datasets.size() > 1) {
      // split every dataset directly into the concatenated ones, sized for all of them
      std::size_t NumValidationEntries = 0;
      std::size_t NumEntries = 0;
      for (const auto &Dataset : Datasets) {
         NumEntries += Dataset.GetRows();
         NumValidationEntries += static_cast<std::size_t>(fValidationSplit * Dataset.GetRows());
      }
      fTrainingDataset.fRVec.reserve((NumEntries - NumValidationEntries) * fNumDatasetCols);
      fValidationDataset.fRVec.reserve(NumValidationEntries * fNumDatasetCols);

      for (auto &Dataset : Datasets) {
         SplitDataset(Dataset, fTrainingDataset, fValidationDataset);
         Dataset = RFlat2DMatrix();
      }
   } else {
      for (auto &Dataset : Datasets) {
         RFlat2DMatrix TrainingDataset;
         RFlat2DMatrix ValidationDataset;
         SplitDataset(Dataset, TrainingDataset, ValidationDataset);
         Dataset = RFlat2DMatrix();
         fTrainingDatasets.push_back(std::move(TrainingDataset));
         fValidationDatasets.push_back(std::move(ValidationDataset));
      }

      if (concatenate) {
         fTrainingDataset = std::move(fTrainingDatasets[0]);
         fValidationDataset = std::move(fValidationDatasets[0]);
         fTrainingDatasets.clear();
         fValidationDatasets.clear();
      }
   }

   fNumTrainingEntries = 0;
   fNumValidationEntries = 0;
   if (concatenate) {
      fNumTrainingEntries = fTrainingDataset.GetRows();
      fNumValidationEntries = fValidationDataset.GetRows();
   } else {
      for (std::size_t i = 0; i < fTrainingDatasets.size(); i++) {
         fNumTrainingEntries += fTrainingDatasets[i].GetRows();
         fNumValidationEntries += fValidationDatasets[i].GetRows();
      }
   }
}

} // namespace ROOT::Experimental::Internal::ML
