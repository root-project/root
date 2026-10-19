// Author: Martin Føll, University of Oslo (UiO) & CERN 01/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_INTERNAL_ML_RSAMPLER
#define ROOT_INTERNAL_ML_RSAMPLER

#include <memory>
#include <string>
#include <vector>

#include "ROOT/ML/RFlat2DMatrix.hxx"

// Forward decls
namespace ROOT::Experimental::Internal::ML {
class RFlat2DMatrixOperators;
}

namespace ROOT::Experimental::Internal::ML {
/**
\class ROOT::Experimental::Internal::ML::RSampler

\brief Implementation of different sampling strategies.
*/

class RSampler {
private:
   const RFlat2DMatrix &fDataset;
   // rows of fDataset in each class
   std::vector<std::vector<std::size_t>> &fRows;
   std::string fSampleType;
   float fSampleRatio;
   bool fReplacement;
   bool fShuffle;
   std::size_t fSetSeed;
   std::size_t fNumEntries;

   std::size_t fMajor;
   std::size_t fMinor;
   std::size_t fNumMajor;
   std::size_t fNumMinor;
   std::size_t fNumResampledMajor;
   std::size_t fNumResampledMinor;

   std::vector<std::size_t> fSamples;

   std::unique_ptr<RFlat2DMatrixOperators> fTensorOperators;

public:
   RSampler(const RFlat2DMatrix &dataset, std::vector<std::vector<std::size_t>> &rows, const std::string &sampleType,
            float sampleRatio, bool replacement = false, bool shuffle = true, std::size_t setSeed = 0);

   ~RSampler();

   void SetupSampler();

   void Sampler(RFlat2DMatrix &SampledTensor);

   void SetupRandomUndersampler();

   void SetupRandomOversampler();

   void RandomUndersampler(RFlat2DMatrix &ShuffledTensor);

   void RandomOversampler(RFlat2DMatrix &ShuffledTensor);

   void GatherSamples(RFlat2DMatrix &ShuffledTensor, const std::vector<std::size_t> &ResampledRows,
                      std::size_t numResampled, const std::vector<std::size_t> &OtherRows);

   void SampleWithReplacement(std::size_t n_samples, std::size_t max);

   void SampleWithoutReplacement(std::size_t n_samples, std::size_t max);

   std::size_t GetNumEntries() { return fNumEntries; }
};

} // namespace ROOT::Experimental::Internal::ML
#endif // ROOT_INTERNAL_ML_RSAMPLER
