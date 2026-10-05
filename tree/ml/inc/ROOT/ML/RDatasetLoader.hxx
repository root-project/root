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
\class ROOT::Experimental::Internal::ML::RDatasetLoader

\brief Load the whole dataset into memory.

In this class the whole dataset is loaded into memory. The dataset is further shuffled and spit into training and
validation sets with the user-defined validation split fraction.
*/

class RDatasetLoader {
private:
   float fValidationSplit;

   std::vector<std::size_t> fVecSizes;
   std::size_t fSumVecSizes;
   float fVecPadding;
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

   bool fShuffle;

public:
   RDatasetLoader(const std::vector<ROOT::RDF::RNode> &rdfs, const float validationSplit,
                  const std::vector<std::string> &cols, const std::vector<std::size_t> &vecSizes = {},
                  const float vecPadding = 0.0, bool shuffle = true, const std::size_t setSeed = 0);

   void SplitDataframe(ROOT::RDF::RNode &rdf, RFlat2DMatrix &TrainingDataset, RFlat2DMatrix &ValidationDataset);

   void SplitDatasets();

   void ConcatenateDatasets();

   std::vector<RFlat2DMatrix> ReleaseTrainingDatasets() { return std::move(fTrainingDatasets); }
   std::vector<RFlat2DMatrix> ReleaseValidationDatasets() { return std::move(fValidationDatasets); }

   RFlat2DMatrix ReleaseTrainingDataset() { return std::move(fTrainingDataset); }
   RFlat2DMatrix ReleaseValidationDataset() { return std::move(fValidationDataset); }

   std::size_t GetNumTrainingEntries() { return fNumTrainingEntries; }
   std::size_t GetNumValidationEntries() { return fNumValidationEntries; }
};

} // namespace ROOT::Experimental::Internal::ML
#endif // ROOT_INTERNAL_ML_RDATASETLOADER
