#include "RSampler.hxx"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>

#include "RFlat2DMatrixOperators.hxx"

namespace ROOT::Experimental::Internal::ML {

RSampler::~RSampler() = default;

RSampler::RSampler(std::vector<RFlat2DMatrix> &datasets, const std::string &sampleType, float sampleRatio,
                   bool replacement, bool shuffle, std::size_t setSeed)
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
void RSampler::SetupSampler()
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
void RSampler::Sampler(RFlat2DMatrix &SampledTensor)
{
   if (fSampleType == "undersampling") {
      RandomUndersampler(SampledTensor);
   } else if (fSampleType == "oversampling") {
      RandomOversampler(SampledTensor);
   }
}

//////////////////////////////////////////////////////////////////////////
/// \brief Calculate fNumEntries and major/minor variables for the random undersampler
void RSampler::SetupRandomUndersampler()
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
void RSampler::SetupRandomOversampler()
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
void RSampler::RandomUndersampler(RFlat2DMatrix &ShuffledTensor)
{
   if (fReplacement) {
      SampleWithReplacement(fNumResampledMajor, fNumMajor);
   }

   else {
      SampleWithoutReplacement(fNumResampledMajor, fNumMajor);
   }

   GatherSamples(ShuffledTensor, fDatasets[fMajor], fNumResampledMajor, fDatasets[fMinor]);
}

//////////////////////////////////////////////////////////////////////////
/// \brief Oversample entries randomly from the minority dataset
/// \param[in] SampledTensor Tensor with all the sampled entries
void RSampler::RandomOversampler(RFlat2DMatrix &ShuffledTensor)
{
   SampleWithReplacement(fNumResampledMinor, fNumMinor);

   GatherSamples(ShuffledTensor, fDatasets[fMinor], fNumResampledMinor, fDatasets[fMajor]);
}

//////////////////////////////////////////////////////////////////////////
/// \brief Fill the sampled entries: the resampled entries followed by all the other ones, in shuffled order
/// \param[in] ShuffledTensor Tensor with all the sampled entries
/// \param[in] ResampledDataset Dataset the entries in fSamples are taken from
/// \param[in] numResampled Number of resampled entries
/// \param[in] OtherDataset Dataset whose entries are all taken
void RSampler::GatherSamples(RFlat2DMatrix &ShuffledTensor, const RFlat2DMatrix &ResampledDataset,
                             std::size_t numResampled, const RFlat2DMatrix &OtherDataset)
{
   std::size_t cols = fDatasets[0].GetCols();
   const auto permutation = fTensorOperators->MakePermutation(fNumEntries);

   // read the entries straight from the datasets, instead of concatenating and shuffling copies of them
   ShuffledTensor.Resize(fNumEntries, cols);
   float *out = ShuffledTensor.GetData();
   for (std::size_t i = 0; i < fNumEntries; i++) {
      const std::size_t entry = permutation.empty() ? i : permutation[i];
      const float *row = entry < numResampled ? ResampledDataset.GetData() + fSamples[entry] * cols
                                              : OtherDataset.GetData() + (entry - numResampled) * cols;
      out = std::copy_n(row, cols, out);
   }
}

//////////////////////////////////////////////////////////////////////////
/// \brief Add indices with replacement to fSamples
/// \param[in] n_samples Number of indices to sample
/// \param[in] max Max index of the sample distribution
void RSampler::SampleWithReplacement(std::size_t n_samples, std::size_t max)
{
   std::uniform_int_distribution<> dist(0, max - 1);
   // a single generator for all the samples: seeding one per sample was slow, and with a fixed seed it drew the same
   // sample every time
   std::random_device rd;
   std::mt19937 g;

   if (fSetSeed == 0) {
      g.seed(rd());
   } else {
      g.seed(fSetSeed);
   }

   fSamples.clear();
   fSamples.reserve(n_samples);
   for (std::size_t i = 0; i < n_samples; ++i) {
      std::size_t sample;
      if (fShuffle) {
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
void RSampler::SampleWithoutReplacement(std::size_t n_samples, std::size_t max)
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

} // namespace ROOT::Experimental::Internal::ML
