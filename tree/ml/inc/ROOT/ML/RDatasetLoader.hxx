// Author: Martin Føll, University of Oslo (UiO) & CERN 01/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_INTERNAL_ML_RDATASETLOADER
#define ROOT_INTERNAL_ML_RDATASETLOADER

#include <algorithm>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#include "ROOT/ML/RFlat2DMatrix.hxx"
#include "ROOT/ML/RFlat2DMatrixOperators.hxx"
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDF/Utils.hxx"

namespace ROOT::Experimental::Internal::ML {

/**
\class ROOT::Experimental::Internal::ML::RDatasetLoaderFunctor

\brief Writes one row of RDataFrame column values into an RFlat2DMatrix, padding or truncating vector columns.
*/

template <typename... ColTypes>
class RDatasetLoaderFunctor {
   float fVecPadding{};
   std::vector<std::size_t> fMaxVecSizes{};
   RFlat2DMatrix &fDatasetTensor;
   std::size_t fNumColumns;

   //////////////////////////////////////////////////////////////////////////
   /// \brief Copy the content of a column into RFlat2DMatrix when the column consists of vectors
   template <typename T, std::enable_if_t<ROOT::Internal::RDF::IsDataContainer<T>::value, int> = 0>
   void AssignToTensor(const T &vec, float *&dst, std::size_t &vecSizeIdx) const
   {
      std::size_t max_vec_size = fMaxVecSizes[vecSizeIdx++];
      std::size_t vec_size = vec.size();
      if (vec_size < max_vec_size) // Padding vector column to max_vec_size with fVecPadding
      {
         std::copy(vec.begin(), vec.end(), dst);
         std::fill(dst + vec_size, dst + max_vec_size, fVecPadding);
      } else // Copy only max_vec_size length from vector column
      {
         std::copy(vec.begin(), vec.begin() + max_vec_size, dst);
      }
      dst += max_vec_size;
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Copy the content of a column into RFlat2DMatrix when the column consists of single values
   template <typename T, std::enable_if_t<!ROOT::Internal::RDF::IsDataContainer<T>::value, int> = 0>
   void AssignToTensor(const T &val, float *&dst, std::size_t & /*vecSizeIdx*/) const
   {
      *dst++ = val;
   }

public:
   RDatasetLoaderFunctor(RFlat2DMatrix &datasetTensor, std::size_t numColumns,
                         const std::vector<std::size_t> &maxVecSizes, float vecPadding)
      : fVecPadding(vecPadding), fMaxVecSizes(maxVecSizes), fDatasetTensor(datasetTensor), fNumColumns(numColumns)
   {
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Fill row \p row with the column values. Only local state is used, so distinct rows can be filled
   /// concurrently.
   void FillRow(std::size_t row, const ColTypes &...cols) const
   {
      float *dst = fDatasetTensor.GetData() + row * fNumColumns;
      std::size_t vecSizeIdx = 0;
      (AssignToTensor(cols, dst, vecSizeIdx), ...);
   }
};

/**
\class ROOT::Experimental::Internal::ML::RDatasetLoader

\brief Load the whole dataset into memory.

In this class the whole dataset is loaded into memory. The dataset is further shuffled and spit into training and
validation sets with the user-defined validation split fraction.
*/

template <typename... Args>
class RDatasetLoader {
private:
   std::size_t fNumEntries;
   float fValidationSplit;

   std::vector<std::size_t> fVecSizes;
   std::size_t fSumVecSizes;
   std::size_t fVecPadding;
   std::size_t fNumDatasetCols;

   std::vector<RFlat2DMatrix> fTrainingDatasets;
   std::vector<RFlat2DMatrix> fValidationDatasets;

   RFlat2DMatrix fTrainingDataset;
   RFlat2DMatrix fValidationDataset;

   std::size_t fNumTrainingEntries;
   std::size_t fNumValidationEntries;
   std::unique_ptr<RFlat2DMatrixOperators> fTensorOperators;

   std::vector<ROOT::RDF::RNode> f_rdfs;
   std::vector<std::string> fCols;
   std::size_t fNumCols;
   std::size_t fSetSeed;

   bool fNotFiltered;
   bool fShuffle;

   ROOT::RDF::RResultPtr<std::vector<ULong64_t>> fEntries;

public:
   RDatasetLoader(const std::vector<ROOT::RDF::RNode> &rdfs, const float validationSplit,
                  const std::vector<std::string> &cols, const std::vector<std::size_t> &vecSizes = {},
                  const float vecPadding = 0.0, bool shuffle = true, const std::size_t setSeed = 0)
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
   void SplitDataframe(ROOT::RDF::RNode &rdf, RFlat2DMatrix &TrainingDataset, RFlat2DMatrix &ValidationDataset)
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
            // no cluster information for this data source
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
         // one row per rdfentry_: thread-safe under implicit multi-threading, rows follow the event loop order
         RDatasetLoaderFunctor<Args...> func(Dataset, fNumDatasetCols, fVecSizes, fVecPadding);
         std::vector<std::string> colsWithEntry{"rdfentry_"};
         colsWithEntry.insert(colsWithEntry.end(), fCols.begin(), fCols.end());
         rdf.Foreach([&func](ULong64_t entry, const Args &...cols) { func.FillRow(entry, cols...); }, colsWithEntry);
      }

      else {
         RDatasetLoaderFunctor<Args...> func(Dataset, fNumDatasetCols, fVecSizes, fVecPadding);
         for (std::size_t j = 0; j < NumEntries; j++) {
            ROOT::Internal::RDF::ChangeBeginAndEndEntries(rdf, (*Entries)[j], (*Entries)[j + 1]);
            rdf.Foreach([&func, j](const Args &...cols) { func.FillRow(j, cols...); }, fCols);
         }
      }

      // reset dataframe, only the filtered path changed the entry range
      if (!NotFiltered) {
         ROOT::Internal::RDF::ChangeBeginAndEndEntries(rdf, (*Entries)[0], (*Entries)[NumEntries]);
      }

      // copy out the validation tail, then shrink the (shuffled) buffer to the training rows and move it
      RFlat2DMatrix ShuffledDataset;
      RFlat2DMatrix &Source = fTensorOperators->ShuffleTensor(ShuffledDataset, Dataset);
      fTensorOperators->SliceTensor(ValidationDataset, Source,
                                    {{NumTrainingEntries, NumEntries}, {0, fNumDatasetCols}});
      Source.Resize(NumTrainingEntries, fNumDatasetCols);
      TrainingDataset = std::move(Source);
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Split the dataframes in a training and validation dataset
   void SplitDatasets()
   {
      fNumEntries = 0;
      fNumTrainingEntries = 0;
      fNumValidationEntries = 0;

      for (auto &rdf : f_rdfs) {
         RFlat2DMatrix TrainingDataset;
         RFlat2DMatrix ValidationDataset;

         SplitDataframe(rdf, TrainingDataset, ValidationDataset);

         fNumTrainingEntries += TrainingDataset.GetRows();
         fNumValidationEntries += ValidationDataset.GetRows();
         fNumEntries += TrainingDataset.GetRows() + ValidationDataset.GetRows();

         fTrainingDatasets.push_back(std::move(TrainingDataset));
         fValidationDatasets.push_back(std::move(ValidationDataset));
      }
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Concatenate the datasets to a dataset
   void ConcatenateDatasets()
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

   std::vector<RFlat2DMatrix> GetTrainingDatasets() { return std::move(fTrainingDatasets); }
   std::vector<RFlat2DMatrix> GetValidationDatasets() { return std::move(fValidationDatasets); }

   RFlat2DMatrix GetTrainingDataset() { return std::move(fTrainingDataset); }
   RFlat2DMatrix GetValidationDataset() { return std::move(fValidationDataset); }

   std::size_t GetNumTrainingEntries() { return fNumTrainingEntries; }
   std::size_t GetNumValidationEntries() { return fNumValidationEntries; }
};

} // namespace ROOT::Experimental::Internal::ML
#endif // ROOT_INTERNAL_ML_RDATASETLOADER
