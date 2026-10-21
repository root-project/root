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
#include "RFlat2DMatrixOperators.hxx"
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

   RFlat2DMatrix fDataset;

   // rows of fDataset in the training and validation sets, one set per dataframe or a single one if concatenated
   std::vector<std::vector<std::size_t>> fTrainingRows;
   std::vector<std::vector<std::size_t>> fValidationRows;

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

   std::vector<std::size_t> CountEntries(std::vector<ROOT::RDF::RNode> &rdfs);

   void LoadDataframe(ROOT::RDF::RNode &rdf, std::size_t NumEntries);

   void
   SplitRows(std::size_t firstRow, std::vector<std::size_t> &TrainingRows, std::vector<std::size_t> &ValidationRows);

   void SplitDatasets(bool concatenate);

   RFlat2DMatrix ReleaseDataset() { return std::move(fDataset); }

   std::vector<std::vector<std::size_t>> ReleaseTrainingRows() { return std::move(fTrainingRows); }
   std::vector<std::vector<std::size_t>> ReleaseValidationRows() { return std::move(fValidationRows); }

   std::size_t GetNumTrainingEntries() { return fNumTrainingEntries; }
   std::size_t GetNumValidationEntries() { return fNumValidationEntries; }
};

} // namespace ROOT::Experimental::Internal::ML
#endif // ROOT_INTERNAL_ML_RDATASETLOADER
