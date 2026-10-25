#include "RDatasetLoader.hxx"

#include "ROOT/RDFHelpers.hxx"

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
/// \brief Number of entries to reserve for each dataframe
/// \param[in] rdfs Dataframes to load
/// \return Number of entries of each dataframe, an upper bound for a filtered dataframe
std::vector<std::size_t> RDatasetLoader::CountEntries(std::vector<ROOT::RDF::RNode> &rdfs)
{
   // size the buffer from the cluster metadata, Count() is only the fallback for sources without it. With filters
   // this is an upper bound: the memory reserved for the entries that don't pass them is never touched
   std::vector<std::size_t> NumEntries(rdfs.size(), 0);
   std::vector<ROOT::RDF::RResultPtr<ULong64_t>> Counts;
   for (std::size_t i = 0; i < rdfs.size(); i++) {
      try {
         for (const auto &cluster : ROOT::Internal::RDF::GetDatasetGlobalClusterBoundaries(rdfs[i])) {
            NumEntries[i] += cluster.second - cluster.first;
         }
      } catch (const std::runtime_error &) {
         // GetDatasetGlobalClusterBoundaries() throws when the RDataFrame has no cluster
         // metadata to query (a source other than TTree/RNTuple or no data source at all).
         // Those paths fall back to Count() below.
      }
      if (NumEntries[i] == 0) {
         Counts.push_back(rdfs[i].Count());
      }
   }

   // the dataframes without cluster metadata are counted together
   if (!Counts.empty()) {
      ROOT::RDF::RunGraphs({Counts.begin(), Counts.end()});
   }
   for (std::size_t i = 0, c = 0; i < rdfs.size(); i++) {
      if (NumEntries[i] == 0) {
         NumEntries[i] = static_cast<std::size_t>(*Counts[c++]);
      }
   }

   return NumEntries;
}

//////////////////////////////////////////////////////////////////////////
/// \brief Load all the entries of an individual dataframe at the end of the dataset
/// \param[in] rdf Dataframe to load
/// \param[in] NumEntries Number of entries of the dataframe, an upper bound for a filtered dataframe
void RDatasetLoader::LoadDataframe(ROOT::RDF::RNode &rdf, std::size_t NumEntries)
{
   const bool NotFiltered = rdf.GetFilterNames().empty();
   const std::size_t firstRow = fDataset.GetRows();

   // a single event loop, the filters only let the passing entries through
   ROOT::Internal::RDF::LoadCustomValues(rdf, fCols, fVecSizes, fVecPadding, fDataset.fRVec);
   if (!NotFiltered) {
      NumEntries = fDataset.GetSize() / fNumDatasetCols - firstRow;
   }
   fDataset.Resize(firstRow + NumEntries, fNumDatasetCols);
}

//////////////////////////////////////////////////////////////////////////
/// \brief Split the rows of the dataset from \p firstRow on into training and validation rows
/// \param[in] firstRow First row of the dataframe in the dataset
/// \param[in] TrainingRows Rows of the training set, appended to
/// \param[in] ValidationRows Rows of the validation set, appended to
void RDatasetLoader::SplitRows(std::size_t firstRow, std::vector<std::size_t> &TrainingRows,
                               std::vector<std::size_t> &ValidationRows)
{
   // number of training and validation entries after the split
   std::size_t NumEntries = fDataset.GetRows() - firstRow;
   std::size_t NumValidationEntries = static_cast<std::size_t>(fValidationSplit * NumEntries);
   std::size_t NumTrainingEntries = NumEntries - NumValidationEntries;

   // the sets refer to the rows of the dataset in shuffled order, instead of copying them
   const auto permutation = fTensorOperators->MakePermutation(NumEntries);
   TrainingRows.reserve(TrainingRows.size() + NumTrainingEntries);
   ValidationRows.reserve(ValidationRows.size() + NumValidationEntries);
   for (std::size_t i = 0; i < NumEntries; i++) {
      const std::size_t row = firstRow + (permutation.empty() ? i : permutation[i]);
      (i < NumTrainingEntries ? TrainingRows : ValidationRows).push_back(row);
   }
}

//////////////////////////////////////////////////////////////////////////
/// \brief Load the dataframes into a single dataset and split it in training and validation rows
/// \param[in] concatenate Whether to concatenate the rows of all the dataframes into a single set
void RDatasetLoader::SplitDatasets(bool concatenate)
{
   // reserve the dataset for all the dataframes and let their values be appended to it
   const std::vector<std::size_t> NumEntries = CountEntries(f_rdfs);
   fDataset.Reserve(std::accumulate(NumEntries.begin(), NumEntries.end(), std::size_t{0}), fNumDatasetCols);

   const std::size_t numSets = concatenate ? 1 : f_rdfs.size();
   fTrainingRows.resize(numSets);
   fValidationRows.resize(numSets);
   for (std::size_t i = 0; i < f_rdfs.size(); i++) {
      const std::size_t firstRow = fDataset.GetRows();
      LoadDataframe(f_rdfs[i], NumEntries[i]);
      SplitRows(firstRow, fTrainingRows[concatenate ? 0 : i], fValidationRows[concatenate ? 0 : i]);
   }

   fNumTrainingEntries = 0;
   fNumValidationEntries = 0;
   for (std::size_t i = 0; i < numSets; i++) {
      fNumTrainingEntries += fTrainingRows[i].size();
      fNumValidationEntries += fValidationRows[i].size();
   }
}

} // namespace ROOT::Experimental::Internal::ML
