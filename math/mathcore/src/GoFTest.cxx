// @(#)root/mathcore:$Id$
// Authors: Bartolomeu Rabacal    05/2010 
/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/
// implementation file for GoFTest


#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <string.h>

#include "Math/Integrator.h"
#include "Math/ProbFuncMathCore.h"

#include "Math/GoFTest.h"


/* Note: The references mentioned here are stated in GoFTest.h */

namespace ROOT {
namespace Math {
  
#if !defined(__CINT__) && !defined(__MAKECINT__)
GoFTest::BadSampleArgument::BadSampleArgument(std::string type) : std::invalid_argument(type + " cannot be zero or zero-length!") {}

GoFTest::DegenerateSamples::DegenerateSamples(std::string type) : std::domain_error(type + " Sampling values all identical.") {}
#else
#include <cassert>
#endif

   GoFTest::GoFTest(const Double_t* sample1, UInt_t sample1Size, const Double_t* sample2, UInt_t sample2Size)
#if !defined(__CINT__) && !defined(__MAKECINT__)
   throw(BadSampleArgument, std::bad_exception) try 
#endif 
      : fCDF(0), fSamples(std::vector<std::vector<Double_t> >(2)), fTestSampleFromH0(kFALSE)
      {
#if !defined(__CINT__) && !defined(__MAKECINT__)
         if (sample1 == 0 || sample1Size == 0) {
            std::string msg = "'sample1";
            msg += !sample1Size ? "Size'" : "'";
            throw BadSampleArgument(msg);
         } else if (sample2 == 0 || sample2Size == 0) {
            std::string msg = "'sample2";
            msg += !sample2Size ? "Size'" : "'";
            throw BadSampleArgument(msg);
         }  
#else
         assert(sample2 != 0 && sample2Size != 0);
         assert(sample2 != 0 && sample2Size != 0); 
#endif 
         std::vector<const Double_t*> samples(2);
         std::vector<UInt_t> samplesSizes(2);
         samples[0] = sample1;
         samples[1] = sample2;
         samplesSizes[0] = sample1Size;
         samplesSizes[1] = sample2Size;
         SetSamples(samples, samplesSizes);
         SetParameters();
#if !defined(__CINT__) && !defined(__MAKECINT__)
      } catch (const BadSampleArgument& bsa) {
      delete fCDF;
      throw;
   } catch (const std::bad_exception& be) {
      delete fCDF;
      throw;
#endif 
   }

   GoFTest::GoFTest(const Double_t* sample, UInt_t sampleSize, EDistribution dist) 
#if !defined(__CINT__) && !defined(__MAKECINT__)
   throw(BadSampleArgument, std::bad_exception) try
#endif  
      : fCDF(0), fDist(dist), fSamples(std::vector<std::vector<Double_t> >(1)), fTestSampleFromH0(kTRUE)
      {
#if !defined(__CINT__) && !defined(__MAKECINT__)
         if (sample == 0 || sampleSize == 0) {
            std::string msg = "'sample";
            msg += !sampleSize ? "Size'" : "'"; 
            throw BadSampleArgument(msg);
         }
#else
         assert(sample != 0 && sampleSize != 0);
#endif
         std::vector<const Double_t*> samples(1, sample);
         std::vector<UInt_t> samplesSizes(1, sampleSize);
         SetSamples(samples, samplesSizes);
         SetParameters();
         SetCDF();
#if !defined(__CINT__) && !defined(__MAKECINT__)
      } catch (const BadSampleArgument& bsa) {
      delete fCDF;
      throw;
   } catch (const std::bad_exception& be) {
      delete fCDF;
      throw;
#endif 
   }

   GoFTest::~GoFTest() {
      delete fCDF;
   }

   void GoFTest::SetSamples(std::vector<const Double_t*> samples, const std::vector<UInt_t> samplesSizes) {
      fCombinedSamples.assign(std::accumulate(samplesSizes.begin(), samplesSizes.end(), 0), 0.0);
      UInt_t combinedSamplesSize = 0;
      for (UInt_t i = 0; i < samples.size(); ++i) {
         fSamples[i].assign(samples[i], samples[i] + samplesSizes[i]);
         std::sort(fSamples[i].begin(), fSamples[i].end());
         for (UInt_t j = 0; j < samplesSizes[i]; ++j) {
            fCombinedSamples[combinedSamplesSize + j] = samples[i][j];
         }
         combinedSamplesSize += samplesSizes[i];
      }   
      std::sort(fCombinedSamples.begin(), fCombinedSamples.end());
#if !defined(__CINT__) && !defined(__MAKECINT__)
      if (*fCombinedSamples.begin() == *(fCombinedSamples.end() - 1)) {
         std::string msg = "Degenerate sample";
         msg += samplesSizes.size() > 1 ? "s!" : "!";
         throw DegenerateSamples(msg);
      }
#else
      assert(*fCombinedSamples.begin() != *(--fCombinedSamples.end()));
#endif
   }

   void GoFTest::SetParameters() {
      fMean = std::accumulate(fSamples[0].begin(), fSamples[0].end(), 0.0) / fSamples[0].size();
      fSigma = TMath::Sqrt(1. / (fSamples[0].size() - 1) * (std::inner_product(fSamples[0].begin(), fSamples[0].end(),     fSamples[0].begin(), 0.0) - fSamples[0].size() * TMath::Power(fMean, 2)));
   }
   
   Double_t GoFTest::operator()(ETestType test, const Char_t* option) const {
      Double_t result = 0.0;
      switch (test) {
         default:
         case kAD:
            result = AndersonDarlingTest(option);
            break;
         case kAD2s:
            result = AndersonDarling2SamplesTest(option);
            break;
         case kKS:
            result = KolmogorovSmirnovTest(option);
            break;
         case kKS2s:
            result = KolmogorovSmirnov2SamplesTest(option);
      }
      return result;
   }

   void GoFTest::SetCDF(CDF_Ptr cdf) { //  Setting parameter-free distributions 
      switch (fDist) {
      case kLogNormal:
         LogSample();
      case kGaussian :
         fCDF = new ROOT::Math::WrappedMemFunction<GoFTest, Double_t (GoFTest::*)(Double_t) const>(*this, &GoFTest::GaussianCDF);
         break;
      case kExponential:
         fCDF = new ROOT::Math::WrappedMemFunction<GoFTest, Double_t (GoFTest::*)(Double_t) const>(*this, &GoFTest::ExponentialCDF);
         break;
      case kUserDefined:
      default:
         fCDF = cdf;
      }
   }

   void GoFTest::Instantiate(const Double_t* sample, UInt_t sampleSize)
#if !defined(__CINT__) && !defined(__MAKECINT__)
      throw(BadSampleArgument, std::bad_exception) try
#endif
   {
#if ! defined(__CINT__) && !defined(__MAKECINT__)
      if (sample == 0 || sampleSize == 0) {
         std::string msg = "'sample";
         msg += !sampleSize ? "Size'" : "'"; 
         throw BadSampleArgument(msg);
      }
#else
      assert(sample != 0 && sampleSize != 0);
#endif
      fCDF = 0;
      fDist = kUserDefined;
      fSamples = std::vector<std::vector<Double_t> >(1);
      fTestSampleFromH0 = kTRUE;
      SetSamples(std::vector<const Double_t*>(1, sample), std::vector<UInt_t>(1, sampleSize));
#if ! defined(__CINT__) && !defined(__MAKECINT__)
   } catch (const BadSampleArgument& bsa) {
      delete fCDF;
      throw;
   } catch (const std::bad_exception& be) {
      delete fCDF;
      throw;
#endif 
   }

   Double_t GoFTest::ComputeIntegral(Double_t* parms) const { 
      ROOT::Math::IntegratorOneDim ig;
      Integrand func(parms);
      ig.SetFunction(func);
      Double_t result = ig.IntegralUp( 0);
      return result;
   }

   Double_t GoFTest::GaussianCDF(Double_t x) const {
      x -= fMean;
      x /= fSigma;
      return ROOT::Math::normal_cdf(x);
   }

   Double_t GoFTest::ExponentialCDF(Double_t x) const {
      x /= fMean;
      return ROOT::Math::exponential_cdf(x, 1.0);
   }

   void GoFTest::LogSample() {
      transform(fSamples[0].begin(), fSamples[0].end(), fSamples[0].begin(), std::ptr_fun<Double_t, Double_t>(TMath::Log));
      SetParameters();
   }

   GoFTest::Integrand::Integrand(Double_t* p) : parms(p) {}

/*
  Taken from (2)
*/
   Double_t GoFTest::Integrand::operator()(Double_t y) const {
      Double_t z = parms[0];
      Double_t t_j = parms[1];
      return TMath::Exp(z / (8 * (1 + TMath::Power(y, 2))) - TMath::Power(y, 2) * t_j);
   }

/*
  Taken from (1)
*/
   Double_t GoFTest::GetSigmaN(UInt_t N) const {
      Double_t sigmaN = 0.0, h = 0.0, H = 0.0, g = 0.0, a, b, c, d, k = fSamples.size();
      for (UInt_t i = 0; i < k; ++i) {
         H += 1.0 / fSamples[i].size();
      }
      for (UInt_t i = 1; i <= N - 1; ++i) {
         h += 1.0 / i;
      }
      for (UInt_t i = 1; i <= N - 2; ++i) {
         for (UInt_t j = i + 1; j <= N - 1; ++j) {
            g += 1.0 / ((N - i) * j);
         }
      }
      a = (4 * g - 6) * k + (10 - 6 * g) * H - 4 * g + 6;
      b = (2 * g - 4) * TMath::Power(k, 2) + 8 * h * k + (2 * g - 14 * h - 4) * H - 8 * h + 4 * g - 6;
      c = (6 * h + 2 * g - 2) * TMath::Power(k, 2) + (4 * h - 4 *g + 6) * k + (2 * h - 6) * H + 4 * h;
      d = (2 * h + 6) * TMath::Power(k, 2) - 4 * h * k;
      sigmaN +=  a * TMath::Power(N, 3) + b * TMath::Power(N, 2) + c * N + d;
      sigmaN /= (N - 1) * (N - 2) * (N - 3);
      sigmaN = TMath::Sqrt(sigmaN);
      return sigmaN;
   }

   Double_t GoFTest::InterpolatePValues(Double_t dA2, Int_t bin) const {
      static const Double_t pvalue[450] = { // The p-value table for the 2-sample Anderson-Darling Anderson-Darling test statistic's asymtotic distribution
         1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9999, 0.9996, 0.9987, 0.9968, 0.9936,
         0.9900, 0.9836, 0.9747, 0.9638, 0.9505, 0.9363, 0.9182, 0.9003, 0.8802, 0.8608,
         0.8437, 0.8251, 0.8033, 0.7839, 0.7643, 0.7452, 0.7273, 0.7081, 0.6900, 0.6704,
         0.6544, 0.6370, 0.6196, 0.6021, 0.5845, 0.5677, 0.5518, 0.5357, 0.5219, 0.5083,
         0.4917, 0.4779, 0.4632, 0.4507, 0.4405, 0.4263, 0.4131, 0.4010, 0.3892, 0.3780,
         0.3680, 0.3563, 0.3467, 0.3359, 0.3277, 0.3191, 0.3096, 0.3019, 0.2937, 0.2858,
         0.2793, 0.2720, 0.2642, 0.2564, 0.2490, 0.2415, 0.2348, 0.2288, 0.2219, 0.2176,
         0.2126, 0.2068, 0.2020, 0.1963, 0.1907, 0.1856, 0.1803, 0.1752, 0.1708, 0.1659,
         0.1615, 0.1569, 0.1515, 0.1463, 0.1431, 0.1405, 0.1377, 0.1338, 0.1303, 0.1278,
         0.1250, 0.1218, 0.1190, 0.1161, 0.1138, 0.1112, 0.1082, 0.1055, 0.1033, 0.1007,
         0.0981, 0.0951, 0.0930, 0.0907, 0.0888, 0.0865, 0.0844, 0.0829, 0.0812, 0.0795,
         0.0774, 0.0754, 0.0739, 0.0723, 0.0709, 0.0692, 0.0673, 0.0658, 0.0645, 0.0626,
         0.0614, 0.0601, 0.0591, 0.0578, 0.0566, 0.0551, 0.0541, 0.0531, 0.0516, 0.0506,
         0.0494, 0.0486, 0.0473, 0.0457, 0.0448, 0.0439, 0.0428, 0.0417, 0.0408, 0.0392,
         0.0385, 0.0376, 0.0364, 0.0352, 0.0347, 0.0338, 0.0334, 0.0330, 0.0324, 0.0318,
         0.0310, 0.0304, 0.0300, 0.0293, 0.0290, 0.0285, 0.0280, 0.0273, 0.0268, 0.0259,
         0.0256, 0.0250, 0.0242, 0.0236, 0.0225, 0.0224, 0.0219, 0.0214, 0.0211, 0.0203,
         0.0196, 0.0193, 0.0188, 0.0182, 0.0179, 0.0175, 0.0172, 0.0168, 0.0164, 0.0160,
         0.0154, 0.0151, 0.0144, 0.0140, 0.0137, 0.0137, 0.0133, 0.0130, 0.0128, 0.0126,
         0.0126, 0.0125, 0.0123, 0.0122, 0.0120, 0.0120, 0.0117, 0.0114, 0.0112, 0.0111,
         0.0109, 0.0107, 0.0106, 0.0106, 0.0105, 0.0105, 0.0102, 0.0100, 0.0097, 0.0096,
         0.0095, 0.0092, 0.0089, 0.0087, 0.0084, 0.0082, 0.0079, 0.0078, 0.0077, 0.0074,
         0.0073, 0.0073, 0.0069, 0.0069, 0.0067, 0.0066, 0.0065, 0.0064, 0.0063, 0.0062,
         0.0060, 0.0060, 0.0057, 0.0057, 0.0056, 0.0054, 0.0054, 0.0053, 0.0052, 0.0050,
         0.0050, 0.0049, 0.0048, 0.0046, 0.0042, 0.0041, 0.0039, 0.0038, 0.0036, 0.0036,
         0.0036, 0.0036, 0.0033, 0.0032, 0.0031, 0.0030, 0.0029, 0.0028, 0.0027, 0.0027,
         0.0027, 0.0027, 0.0027, 0.0027, 0.0027, 0.0027, 0.0027, 0.0027, 0.0027, 0.0027,
         0.0027, 0.0026, 0.0026, 0.0025, 0.0025, 0.0024, 0.0024, 0.0024, 0.0023, 0.0023,
         0.0022, 0.0022, 0.0022, 0.0022, 0.0022, 0.0022, 0.0022, 0.0021, 0.0021, 0.0021,
         0.0021, 0.0017, 0.0016, 0.0015, 0.0015, 0.0015, 0.0015, 0.0015, 0.0015, 0.0015,
         0.0014, 0.0014, 0.0014, 0.0014, 0.0014, 0.0013, 0.0011, 0.0011, 0.0011, 0.0011,
         0.0011, 0.0011, 0.0011, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0009, 0.0009,
         0.0009, 0.0009, 0.0009, 0.0009, 0.0009, 0.0009, 0.0009, 0.0009, 0.0009, 0.0009,
         0.0009, 0.0008, 0.0008, 0.0007, 0.0007, 0.0007, 0.0007, 0.0007, 0.0007, 0.0006,
         0.0006, 0.0006, 0.0006, 0.0006, 0.0006, 0.0006, 0.0006, 0.0005, 0.0004, 0.0004,
         0.0004, 0.0004, 0.0004, 0.0004, 0.0003, 0.0003, 0.0003, 0.0002, 0.0002, 0.0002,
         0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002,
         0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002,
         0.0002, 0.0002, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
         0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
         0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
         0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
         0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
         0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
         0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0000
      };
      Double_t pvl, pvr;
      if (dA2 >= 0.0) {
         pvl = pvalue[bin];
         pvr = pvalue[bin - 1];
      } else {
         dA2 *= -1;
         pvl = pvalue[bin + 1];
         pvr = pvalue[bin];
      }
      return pvl + dA2 * (pvr - pvl);
   }

   Double_t GoFTest::PValueAD2Samples(Double_t& A2, UInt_t N) const {
      Double_t pvalue, dA2, W2 = A2, sigmaN = GetSigmaN(N);
      A2 -= fSamples.size() - 1;
      A2 /= sigmaN; // standartized test statistic
      if (W2 >= 8.0)
         return 0.0;
      else if (W2 <= 0.0)
         return 1.0;
      if (A2 <= 0.0) A2 = W2;
      Int_t bin = Int_t(50 * A2);
      dA2 = Double_t(bin) / 50 + 0.01 - A2; // Difference between the bin center and A2 
      pvalue = InterpolatePValues(dA2, bin);
      return pvalue;
   }

/*
  Taken from (2)
*/Double_t GoFTest::PValueAD1Sample(Double_t A2) const {
   if (A2 == 0)
      return 0.0;
   Double_t t0 = TMath::Power(TMath::Pi(), 2) / (8 * A2);
   Double_t t1 = 25 * t0;
   Double_t parms[2] = {A2, t0};
   Double_t f0Integral = ComputeIntegral(parms);
   parms[1] = t1;
   Double_t f1Integral = ComputeIntegral(parms);
   Double_t pvalue = 1 - (TMath::Sqrt(2 * TMath::Pi()) / A2 * (TMath::Exp(-t0) * f0Integral - 5.0 / 2.0 * TMath::Exp(-t1) * f1Integral));
   return pvalue;
}

/*
  Taken from (1) -- Named for 2 samples but implemented for K. Restricted to K = 2 by the class's constructors 
*/Double_t GoFTest::AndersonDarling2SamplesTest(const Char_t* option) const { 
   if (fTestSampleFromH0) {
      std::cerr << "Only 1-sample tests can be issued!" << std::endl;
      return -1;
   }
   std::vector<Double_t> z(fCombinedSamples); 
   std::vector<Double_t>::iterator endUnique = std::unique(z.begin(), z.end()); //z_j's in (1)
   std::vector<UInt_t> h; // h_j's in (1)
   std::vector<Double_t> H; // H_j's in (1)
   UInt_t N = fCombinedSamples.size();
   for (std::vector<Double_t>::iterator data = z.begin(); data != endUnique; ++data) {
      UInt_t n = std::count(fCombinedSamples.begin(), fCombinedSamples.end(), *data);
      h.push_back(n);
      H.push_back(std::count_if(fCombinedSamples.begin(), fCombinedSamples.end(), bind2nd(std::less<Double_t>(), *data)) + n / 2.);
   }
   std::vector<std::vector<Double_t> > F(fSamples.size()); // F_ij's in (1)
   for (UInt_t i = 0; i < fSamples.size(); ++i) {
      for (std::vector<Double_t>::iterator data = z.begin(); data != endUnique; ++data) {
         UInt_t n = std::count(fSamples[i].begin(), fSamples[i].end(), *data);
         F[i].push_back(std::count_if(fSamples[i].begin(), fSamples[i].end(), bind2nd(std::less<Double_t>(), *data)) + n / 2.);
      }
   }
   Double_t A2 = 0.0; // Anderson-Darling A^2 Test Statistic
   for (UInt_t i = 0; i < fSamples.size(); ++i) {
      Double_t sum_result = 0.0;
      UInt_t j = 0;
      for (std::vector<Double_t>::iterator data = z.begin(); data != endUnique; ++data) {
         sum_result += h[j] *  TMath::Power(N * F[i][j]- fSamples[i].size() * H[j], 2) / (H[j] * (N - H[j]) - N * h[j] / 4.0); 
         ++j;
      }
      A2 += 1.0 / fSamples[i].size() * sum_result;
   }
   A2 *= (N - 1) / (TMath::Power(N, 2)); // A2_akN in (1)
   Double_t pvalue = PValueAD2Samples(A2, N); // standartized A2
   return (strncmp(option, "p", 1) == 0 || strncmp(option, "t", 1) != 0) ? pvalue : A2;
}

/*
  Taken from (3)
*/Double_t GoFTest::AndersonDarlingTest(const Char_t* option) const {
   if (!fTestSampleFromH0) {
      std::cerr << "Only 2-samples tests can be issued!" << std::endl;
      return -1;
   }
   Double_t A2 = 0.0;
   Int_t n = fSamples[0].size();
   for (Int_t i = 0; i < n ; ++i) {
      Double_t x1 = fSamples[0][i];
      Double_t w1 = (*fCDF)(x1);
      Double_t result = (2 * (i + 1) - 1) * TMath::Log(w1) + (2 * (n - (i + 1)) + 1) * TMath::Log(1 - w1);
      A2 += result; 
   }
   (A2 /= -n) -= n;
   Double_t pvalue = PValueAD1Sample(A2);
   return (strncmp(option, "p", 1) == 0 || strncmp(option, "t", 1) != 0) ? pvalue : A2;
}

   Double_t GoFTest::KolmogorovSmirnov2SamplesTest(const Char_t* option) const {
      if (fTestSampleFromH0) {
         std::cerr << "Only 1-sample tests can be issued!" << std::endl;
         return -1;
      }
      const UInt_t na = fSamples[0].size();
      const UInt_t nb = fSamples[1].size();
      Double_t* a = new Double_t[na];
      Double_t* b = new Double_t[nb]; 
      std::copy(fSamples[0].begin(), fSamples[0].end(), a);
      std::copy(fSamples[1].begin(), fSamples[1].end(), b);
      Double_t result = TMath::KolmogorovTest(na, a, nb, b, (strncmp(option, "p", 1) == 0 || strncmp(option, "t", 1) != 0 ? 0 : "M"));
      return result;
   }

/* 
   Algorithm taken from (3) in page 737
*/Double_t GoFTest::KolmogorovSmirnovTest(const Char_t* option) const {
   if (!fTestSampleFromH0) {
      std::cerr << "Only 2-samples tests can be issued!" << std::endl;
      return -1;
   }
   Double_t Fo = 0.0, Dn = 0.0;
   UInt_t n = fSamples[0].size();
   for (UInt_t i = 0; i < n; ++i) {
      Double_t Fn = (i + 1.0) / n;
      Double_t F = (*fCDF)(fSamples[0][i]);
      Double_t result = std::max(TMath::Abs(Fn - F), TMath::Abs(Fo - Fn));
      if (result > Dn) Dn = result;
      Fo = Fn;
   }
   Double_t pvalue = TMath::KolmogorovProb(Dn * (TMath::Sqrt(n) + 0.12 + 0.11 / TMath::Sqrt(n)));
   return (strncmp(option, "p", 1) == 0 || strncmp(option, "t", 1) != 0) ? pvalue : Dn;
}

} // ROOT namespace
} // Math namespace

