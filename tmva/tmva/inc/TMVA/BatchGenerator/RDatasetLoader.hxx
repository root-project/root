// Author: Martin Føll, University of Oslo (UiO) & CERN 01/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef TMVA_RDATASETLOADER
#define TMVA_RDATASETLOADER

#include <vector>
#include <random>

#include "TMVA/RTensor.hxx"
#include "ROOT/RDataFrame.hxx"
#include "TMVA/BatchGenerator/RFlat2DMatrix.hxx"
#include "TMVA/BatchGenerator/RFlat2DMatrixOperators.hxx"
#include "ROOT/RDF/Utils.hxx"
#include "ROOT/RVec.hxx"

#include "ROOT/RLogger.hxx"

namespace TMVA {
namespace Experimental {
namespace Internal {

// clang-format off
/**
\class ROOT::TMVA::Experimental::Internal::RDatasetLoaderFunctor
\ingroup tmva
\brief Loading chunks made in RDatasetLoader into tensors from data from RDataFrame.
*/

template <typename... ColTypes>
class RDatasetLoaderFunctor {
   // clang-format on   
   std::size_t fOffset{};
   std::size_t fVecSizeIdx{};
   float fVecPadding{};
   std::vector<std::size_t> fMaxVecSizes{};
   RFlat2DMatrix &fDatasetTensor;

   std::size_t fNumDatasetCols;

   int fI;
   int fNumColumns;

   //////////////////////////////////////////////////////////////////////////
   /// \brief Copy the content of a column into RTensor when the column consits of vectors 
   template <typename T, std::enable_if_t<ROOT::Internal::RDF::IsDataContainer<T>::value, int> = 0>
   void AssignToTensor(const T &vec, int i, int numColumns)
   {
      std::size_t max_vec_size = fMaxVecSizes[fVecSizeIdx++];
      std::size_t vec_size = vec.size();
      if (vec_size < max_vec_size) // Padding vector column to max_vec_size with fVecPadding
      {
         std::copy(vec.begin(), vec.end(), &fDatasetTensor.GetData()[fOffset + numColumns * i]);
         std::fill(&fDatasetTensor.GetData()[fOffset + numColumns * i + vec_size],
                   &fDatasetTensor.GetData()[fOffset + numColumns * i + max_vec_size], fVecPadding);
      } else // Copy only max_vec_size length from vector column
      {
         std::copy(vec.begin(), vec.begin() + max_vec_size, &fDatasetTensor.GetData()[fOffset + numColumns * i]);
      }
      fOffset += max_vec_size;
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Copy the content of a column into RTensor when the column consits of single values 
   template <typename T, std::enable_if_t<!ROOT::Internal::RDF::IsDataContainer<T>::value, int> = 0>
   void AssignToTensor(const T &val, int i, int numColumns)
   {
      fDatasetTensor.GetData()[fOffset + numColumns * i] = val;
      fOffset++;
   }

public:
   RDatasetLoaderFunctor(RFlat2DMatrix &datasetTensor, std::size_t numColumns,
                       const std::vector<std::size_t> &maxVecSizes, float vecPadding, int i)
      : fDatasetTensor(datasetTensor), fMaxVecSizes(maxVecSizes), fVecPadding(vecPadding), fI(i), fNumColumns(numColumns)
   {
   }

   void operator()(const ColTypes &...cols)
   {
      fVecSizeIdx = 0;
      (AssignToTensor(cols, fI, fNumColumns), ...);
   }
};

// clang-format off
/**
\class ROOT::TMVA::Experimental::Internal::RDatasetLoader
\ingroup tmva
\brief Load the whole dataset into memory.

In this class the whole dataset is loaded into memory. The dataset is further shuffled and spit into training and validation sets with the user-defined validation split fraction.
*/

template <typename... Args>
class RDatasetLoader {
private:
   // clang-format on   
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
      std::size_t NumEntries = rdf.Count().GetValue();
      ROOT::RDF::RResultPtr<std::vector<ULong64_t>> Entries = rdf.Take<ULong64_t>("rdfentry_");

      // add the last element in entries to not go out of range when filling chunks
      Entries->push_back((*Entries)[NumEntries - 1] + 1);
      
      // number of training and validation entries after the split
      std::size_t NumValidationEntries = static_cast<std::size_t>(fValidationSplit * NumEntries);
      std::size_t NumTrainingEntries = NumEntries - NumValidationEntries;
      
      RFlat2DMatrix Dataset({NumEntries, fNumDatasetCols});

      bool NotFiltered = rdf.GetFilterNames().empty();
      if (NotFiltered) {
         RDatasetLoaderFunctor<Args...> func(Dataset, fNumDatasetCols, fVecSizes, fVecPadding, 0);
         rdf.Foreach(func, fCols);
      }

      else {
         std::size_t datasetEntry = 0;
         for (std::size_t j = 0; j < NumEntries; j++) {
            RDatasetLoaderFunctor<Args...> func(Dataset, fNumDatasetCols, fVecSizes, fVecPadding, datasetEntry);
            ROOT::Internal::RDF::ChangeBeginAndEndEntries(rdf, (*Entries)[j], (*Entries)[j + 1]);
            rdf.Foreach(func, fCols);
            datasetEntry++;
         }
      }

      RFlat2DMatrix ShuffledDataset({NumEntries, fNumDatasetCols});
      fTensorOperators->ShuffleTensor(ShuffledDataset, Dataset);
      fTensorOperators->SliceTensor(TrainingDataset, ShuffledDataset, {{0, NumTrainingEntries}, {0, fNumDatasetCols}});
      fTensorOperators->SliceTensor(ValidationDataset, ShuffledDataset, {{NumTrainingEntries, NumEntries}, {0, fNumDatasetCols}});
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Split the dataframes in a training and validation dataset
   void SplitDatasets()
   {
     fNumEntries = 0;
     fNumTrainingEntries = 0;
     fNumValidationEntries = 0;
     
     for (auto& rdf : f_rdfs) {
       RFlat2DMatrix TrainingDataset;
       RFlat2DMatrix ValidationDataset;
       
       SplitDataframe(rdf, TrainingDataset, ValidationDataset);
       fTrainingDatasets.push_back(TrainingDataset);
       fValidationDatasets.push_back(ValidationDataset);
       
       fNumTrainingEntries += TrainingDataset.GetRows();
       fNumValidationEntries += ValidationDataset.GetRows();
       fNumEntries += TrainingDataset.GetRows() + ValidationDataset.GetRows();       
     }
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Concatenate the datasets to a dataset
   void ConcatenateDatasets()
   {
      fTensorOperators->ConcatenateTensors(fTrainingDataset, fTrainingDatasets);
      fTensorOperators->ConcatenateTensors(fValidationDataset, fValidationDatasets);      
   }
   
   std::vector<RFlat2DMatrix> GetTrainingDatasets() {return fTrainingDatasets;}
   std::vector<RFlat2DMatrix> GetValidationDatasets() {return fValidationDatasets;}

   RFlat2DMatrix GetTrainingDataset() {return fTrainingDataset;}
   RFlat2DMatrix GetValidationDataset() {return fValidationDataset;}
   
   std::size_t GetNumTrainingEntries() {return fTrainingDataset.GetRows();}
   std::size_t GetNumValidationEntries() {return fValidationDataset.GetRows();}   
};

} // namespace Internal
} // namespace Experimental
} // namespace TMVA
#endif // TMVA_RDATASETLOADER
