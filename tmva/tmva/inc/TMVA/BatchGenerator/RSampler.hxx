// Author: Martin Føll, University of Oslo (UiO) & CERN 01/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef TMVA_RSAMPLER
#define TMVA_RSAMPLER

#include <vector>
#include <random>
#include <algorithm>

#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDF/Utils.hxx"
#include "ROOT/RVec.hxx"
#include "TMVA/BatchGenerator/RFlat2DMatrixOperators.hxx"
#include "ROOT/RLogger.hxx"

namespace TMVA::Experimental::Internal {
// clang-format off
/**
\class ROOT::TMVA::Experimental::Internal::RSampler
\ingroup tmva
\brief Implementation of different sampling strategies.
*/

class RSampler {
private:
   // clang-format on   
   std::vector<RFlat2DMatrix> &fDatasets;
   std::string fSampleType;
   bool fShuffle;
   std::size_t fSetSeed;
   std::size_t fNumEntries;

   std::unique_ptr<RFlat2DMatrixOperators> fTensorOperators;   
public:
  RSampler(std::vector<RFlat2DMatrix> &datasets, const std::string &sampleType, bool shuffle = true, const std::size_t setSeed = 0)
    : fDatasets(datasets),
      fSampleType(sampleType),
      fShuffle(shuffle),
      fSetSeed(setSeed)
   {
      fTensorOperators = std::make_unique<RFlat2DMatrixOperators>(fShuffle, fSetSeed);
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Collection of sampling types
   /// \param[in] SampledTensor Tensor with all the sampled entries
   void Sampler(RFlat2DMatrix &SampledTensor)
   {
      if (fSampleType == "random") {
         RandomSampler(SampledTensor);
      }
   }
   
   //////////////////////////////////////////////////////////////////////////
   /// \brief Sample all entries randomly from the datasets
   /// \param[in] SampledTensor Tensor with all the sampled entries
   void RandomSampler(RFlat2DMatrix &SampledTensor) {
         RFlat2DMatrix ConcatTensor;
         fTensorOperators->ConcatenateTensors(ConcatTensor, fDatasets);
         fTensorOperators->ShuffleTensor(SampledTensor, ConcatTensor);
         fNumEntries = SampledTensor.GetRows();
   }

   std::size_t GetNumEntries() { return fNumEntries;}
};

} // namespace TMVA::Experimental::Internal
#endif // TMVA_RSAMPLER
