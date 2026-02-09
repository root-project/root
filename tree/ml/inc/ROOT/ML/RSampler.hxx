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

#include <vector>
#include <random>
#include <algorithm>

#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDF/Utils.hxx"
#include "ROOT/RVec.hxx"
#include "ROOT/ML/RFlat2DMatrixOperators.hxx"
#include "ROOT/RLogger.hxx"

namespace ROOT::Experimental::Internal::ML {
/**
\class ROOT::Experimental::Internal::ML::RSampler

\brief Implementation of different sampling strategies.
*/

class RSampler {
private:
   std::vector<RFlat2DMatrix> &fDatasets;
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
   RSampler(std::vector<RFlat2DMatrix> &datasets, const std::string &sampleType, float sampleRatio,
            bool replacement = false, bool shuffle = true, std::size_t setSeed = 0)
      : fDatasets(datasets),
        fSampleType(sampleType),
        fSampleRatio(sampleRatio),
        fReplacement(replacement),
        fShuffle(shuffle),
        fSetSeed(setSeed)
   {
      fTensorOperators = std::make_unique<RFlat2DMatrixOperators>(fShuffle, fSetSeed);

      // setup the sampler for the datasets
      SetupSampler();
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Calculate fNumEntries and major/minor variables
   void SetupSampler()
   {
      if (fSampleType == "undersampling") {
         SetupRandomUndersampler();
      } else if (fSampleType == "oversampling") {
         SetupRandomOversampler();
      }
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Collection of sampling types
   /// \param[in] SampledTensor Tensor with all the sampled entries
   void Sampler(RFlat2DMatrix &SampledTensor)
   {
      if (fSampleType == "undersampling") {
         RandomUndersampler(SampledTensor);
      } else if (fSampleType == "oversampling") {
         RandomOversampler(SampledTensor);
      }
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Calculate fNumEntries and major/minor variables for the random undersampler
   void SetupRandomUndersampler()
   {
      if (fDatasets[0].GetRows() > fDatasets[1].GetRows()) {
         fMajor = 0;
         fMinor = 1;
      } else {
         fMajor = 1;
         fMinor = 0;
      }

      fNumMajor = fDatasets[fMajor].GetRows();
      fNumMinor = fDatasets[fMinor].GetRows();
      fNumResampledMajor = static_cast<std::size_t>(fNumMinor / fSampleRatio);
      if (!fReplacement && fNumResampledMajor > fNumMajor) {
         auto minRatio = std::to_string(std::round(double(fNumMinor) / double(fNumMajor) * 100.0) / 100.0);
         minRatio.erase(minRatio.find('.') + 3);
         throw std::invalid_argument(
            "The sampling_ratio is too low: not enough entries in the majority class to sample from.\n"
            "Choose sampling_ratio > " +
            minRatio + " or set replacement to True.");
      }
      fNumEntries = fNumMinor + fNumResampledMajor;
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Calculate fNumEntries and major/minor variables for the random oversampler
   void SetupRandomOversampler()
   {
      if (fDatasets[0].GetRows() > fDatasets[1].GetRows()) {
         fMajor = 0;
         fMinor = 1;
      } else {
         fMajor = 1;
         fMinor = 0;
      }

      fNumMajor = fDatasets[fMajor].GetRows();
      fNumMinor = fDatasets[fMinor].GetRows();
      fNumResampledMinor = static_cast<std::size_t>(fSampleRatio * fNumMajor);
      fNumEntries = fNumMajor + fNumResampledMinor;
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Undersample entries randomly from the majority dataset
   /// \param[in] SampledTensor Tensor with all the sampled entries
   void RandomUndersampler(RFlat2DMatrix &ShuffledTensor)
   {
      if (fReplacement) {
         SampleWithReplacement(fNumResampledMajor, fNumMajor);
      }

      else {
         SampleWithoutReplacement(fNumResampledMajor, fNumMajor);
      }

      std::size_t cols = fDatasets[0].GetCols();
      ShuffledTensor.Reshape(fNumEntries, cols);
      RFlat2DMatrix SampledTensor(fNumEntries, cols);
      RFlat2DMatrix UndersampledMajorTensor(fNumResampledMajor, cols);

      std::size_t index = 0;
      for (std::size_t i = 0; i < fNumResampledMajor; i++) {
         std::copy(fDatasets[fMajor].GetData() + fSamples[i] * cols,
                   fDatasets[fMajor].GetData() + (fSamples[i] + 1) * cols,
                   UndersampledMajorTensor.GetData() + index * cols);
         index++;
      }

      fTensorOperators->ConcatenateTensors(SampledTensor, {UndersampledMajorTensor, fDatasets[fMinor]});
      fTensorOperators->ShuffleTensor(ShuffledTensor, SampledTensor);
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Oversample entries randomly from the minority dataset
   /// \param[in] SampledTensor Tensor with all the sampled entries
   void RandomOversampler(RFlat2DMatrix &ShuffledTensor)
   {
      SampleWithReplacement(fNumResampledMinor, fNumMinor);

      std::size_t cols = fDatasets[0].GetCols();
      ShuffledTensor.Reshape(fNumEntries, cols);
      RFlat2DMatrix SampledTensor(fNumEntries, cols);
      RFlat2DMatrix OversampledMinorTensor(fNumResampledMinor, cols);

      std::size_t index = 0;
      for (std::size_t i = 0; i < fNumResampledMinor; i++) {
         std::copy(fDatasets[fMinor].GetData() + fSamples[i] * cols,
                   fDatasets[fMinor].GetData() + (fSamples[i] + 1) * cols,
                   OversampledMinorTensor.GetData() + index * cols);
         index++;
      }

      fTensorOperators->ConcatenateTensors(SampledTensor, {OversampledMinorTensor, fDatasets[fMajor]});
      fTensorOperators->ShuffleTensor(ShuffledTensor, SampledTensor);
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Add indices with replacement to fSamples
   /// \param[in] n_samples Number of indices to sample
   /// \param[in] max Max index of the sample distribution
   void SampleWithReplacement(std::size_t n_samples, std::size_t max)
   {
      std::uniform_int_distribution<> dist(0, max - 1);
      fSamples.clear();
      fSamples.reserve(n_samples);
      for (std::size_t i = 0; i < n_samples; ++i) {
         std::size_t sample;
         if (fShuffle) {
            std::random_device rd;
            std::mt19937 g;

            if (fSetSeed == 0) {
               g.seed(rd());
            } else {
               g.seed(fSetSeed);
            }

            sample = dist(g);
         }

         else {
            sample = i % max;
         }
         fSamples.push_back(sample);
      }
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Add indices without replacement to fSamples
   /// \param[in] n_samples Number of indices to sample
   /// \param[in] max Max index of the sample distribution
   void SampleWithoutReplacement(std::size_t n_samples, std::size_t max)
   {
      std::vector<std::size_t> UniqueSamples;
      UniqueSamples.reserve(max);
      fSamples.clear();
      fSamples.reserve(n_samples);

      for (std::size_t i = 0; i < max; ++i)
         UniqueSamples.push_back(i);

      if (fShuffle) {
         std::random_device rd;
         std::mt19937 g;

         if (fSetSeed == 0) {
            g.seed(rd());
         } else {
            g.seed(fSetSeed);
         }
         std::shuffle(UniqueSamples.begin(), UniqueSamples.end(), g);
      }

      for (std::size_t i = 0; i < n_samples; ++i) {
         fSamples.push_back(UniqueSamples[i]);
      }
   }

   std::size_t GetNumEntries() { return fNumEntries; }
};

} // namespace ROOT::Experimental::Internal::ML
#endif // ROOT_INTERNAL_ML_RSAMPLER
