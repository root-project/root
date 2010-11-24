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
#include <cassert>

#include "Math/Error.h"
#include "Math/Integrator.h"
#include "Math/ProbFuncMathCore.h"

#include "Math/GoFTest.h"


/* Note: The references mentioned here are stated in GoFTest.h */

namespace ROOT {
namespace Math {
   
   struct CDFWrapper : public IGenFunction { 
      // wrapper around a cdf funciton to re-scale for the range
      Double_t fXmin; // lower range for x 
      Double_t fXmax; // lower range for x 
      Double_t fNorm; // normalization
      const IGenFunction* fCDF; // cdf pointer (owned by the class) 

      
      virtual ~CDFWrapper() { if (fCDF) delete fCDF; }

      CDFWrapper(const IGenFunction& cdf, Double_t xmin=0, Double_t xmax=-1) : 
         fCDF(cdf.Clone())
      {
         if (xmin >= xmax) { 
            fNorm = 1; 
            fXmin = -std::numeric_limits<double>::infinity();
            fXmax = std::numeric_limits<double>::infinity();
         }
         else {
            fNorm = cdf(xmax) - cdf(xmin); 
            fXmin = xmin; 
            fXmax = xmax; 
         }
      }

      Double_t DoEval(Double_t x) const {
         if (x <= fXmin) return 0; 
         if (x >= fXmax) return 1.0; 
         return (*fCDF)(x)/fNorm;
      }
      
      IGenFunction* Clone() const {
         return new CDFWrapper(*fCDF,fXmin,fXmax);
      }
   };


   class PDFIntegral : public IGenFunction { 
      Double_t fXmin; // lower range for x 
      Double_t fXmax; // lower range for x 
      Double_t fNorm; // normalization
      mutable IntegratorOneDim fIntegral;
      const IGenFunction* fPDF; // pdf pointer (owned by the class) 
   public:

      virtual ~PDFIntegral() { if (fPDF) delete fPDF; }

      PDFIntegral(const IGenFunction& pdf, Double_t xmin = 0, Double_t xmax = -1) : 
         fXmin(xmin), 
         fXmax(xmax),
         fNorm(1),
         fPDF(pdf.Clone())
      {
         // compute normalization
         fIntegral.SetFunction(*fPDF);  // N.B. must be fPDF (the cloned copy) and not pdf which can disappear
         if (fXmin >= fXmax) {  
            fXmin = -std::numeric_limits<double>::infinity();
            fXmax = std::numeric_limits<double>::infinity();
         }
         if (fXmin == -std::numeric_limits<double>::infinity() && fXmax == std::numeric_limits<double>::infinity() ) { 
            fNorm = fIntegral.Integral();
         }
         else if (fXmin == -std::numeric_limits<double>::infinity() )
            fNorm = fIntegral.IntegralLow(fXmax);
         else if (fXmax == std::numeric_limits<double>::infinity() )
            fNorm = fIntegral.IntegralUp(fXmin);
         else
            fNorm = fIntegral.Integral(fXmin, fXmax);
      }
            
      Double_t DoEval(Double_t x) const {
         if (x <= fXmin) return 0; 
         if (x >= fXmax) return 1.0; 
         if (fXmin == -std::numeric_limits<double>::infinity() )
            return fIntegral.IntegralLow(x)/fNorm;
         else 
            return fIntegral.Integral(fXmin,x)/fNorm;
      }
      
      IGenFunction* Clone() const {
         return new PDFIntegral(*fPDF, fXmin, fXmax);
      }
   };

   void GoFTest::SetDistribution(EDistribution dist) {
      if (!(kGaussian <= dist && dist <= kExponential)) {
         MATH_ERROR_MSG("SetDistribution", "Cannot set distribution type! Distribution type option must be ennabled.");
         return;
      }
      fDist = dist;
      SetCDF();
   }
  
   GoFTest::GoFTest( UInt_t sample1Size, const Double_t* sample1, UInt_t sample2Size, const Double_t* sample2 )
   : fCDF(std::auto_ptr<IGenFunction>((IGenFunction*)0)), 
     fDist(kUndefined), 
     fSamples(std::vector<std::vector<Double_t> >(2)), 
     fTestSampleFromH0(kFALSE) {
      Bool_t badSampleArg = sample1 == 0 || sample1Size == 0;
      if (badSampleArg) { 
         std::string msg = "'sample1";
         msg += !sample1Size ? "Size' cannot be zero" : "' cannot be zero-length";
         MATH_ERROR_MSG("GoFTest", msg.c_str());
         assert(!badSampleArg);
      }
      badSampleArg = sample2 == 0 || sample2Size == 0;
      if (badSampleArg) { 
         std::string msg = "'sample2";         
         msg += !sample2Size ? "Size' cannot be zero" : "' cannot be zero-length";
         MATH_ERROR_MSG("GoFTest", msg.c_str());
         assert(!badSampleArg);
      }
      std::vector<const Double_t*> samples(2);
      std::vector<UInt_t> samplesSizes(2);
      samples[0] = sample1;
      samples[1] = sample2;
      samplesSizes[0] = sample1Size;
      samplesSizes[1] = sample2Size;
      SetSamples(samples, samplesSizes);
      SetParameters();
   }

   GoFTest::GoFTest(UInt_t sampleSize, const Double_t* sample, EDistribution dist)   
   : fCDF(std::auto_ptr<IGenFunction>((IGenFunction*)0)), 
     fDist(dist), 
     fSamples(std::vector<std::vector<Double_t> >(1)), 
     fTestSampleFromH0(kTRUE) {
      Bool_t badSampleArg = sample == 0 || sampleSize == 0;
      if (badSampleArg) { 
         std::string msg = "'sample";
         msg += !sampleSize ? "Size' cannot be zero" : "' cannot be zero-length";
         MATH_ERROR_MSG("GoFTest", msg.c_str());
         assert(!badSampleArg);
      }
      std::vector<const Double_t*> samples(1, sample);
      std::vector<UInt_t> samplesSizes(1, sampleSize);
      SetSamples(samples, samplesSizes);
      SetParameters();
      SetCDF();
   }

   GoFTest::~GoFTest() {}

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
   
      Bool_t degenerateSamples = *(fCombinedSamples.begin()) == *(fCombinedSamples.end() - 1);
      if (degenerateSamples) { 
         std::string msg = "Degenerate sample";
         msg += samplesSizes.size() > 1 ? "s!" : "!";
         msg += " Sampling values all identical.";
         MATH_ERROR_MSG("SetSamples", msg.c_str());
         assert(!degenerateSamples);
      }
   }

   void GoFTest::SetParameters() {
      fMean = std::accumulate(fSamples[0].begin(), fSamples[0].end(), 0.0) / fSamples[0].size();
      fSigma = TMath::Sqrt(1. / (fSamples[0].size() - 1) * (std::inner_product(fSamples[0].begin(), fSamples[0].end(),     fSamples[0].begin(), 0.0) - fSamples[0].size() * TMath::Power(fMean, 2)));
   }
   
   void GoFTest::operator()(ETestType test, Double_t& pvalue, Double_t& testStat) const {
      switch (test) {
         default:
         case kAD:
            AndersonDarlingTest(pvalue, testStat);
            break;
         case kAD2s:
            AndersonDarling2SamplesTest(pvalue, testStat);
            break;
         case kKS:
            KolmogorovSmirnovTest(pvalue, testStat);
            break;
         case kKS2s:
            KolmogorovSmirnov2SamplesTest(pvalue, testStat);
      }
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

   void GoFTest::SetCDF() { // Setting parameter-free distributions 
      IGenFunction* cdf = 0;
      switch (fDist) {
      case kLogNormal:
         LogSample();
      case kGaussian :
         cdf = new ROOT::Math::WrappedMemFunction<GoFTest, Double_t (GoFTest::*)(Double_t) const>(*this, &GoFTest::GaussianCDF);
         break;
      case kExponential:
         cdf = new ROOT::Math::WrappedMemFunction<GoFTest, Double_t (GoFTest::*)(Double_t) const>(*this, &GoFTest::ExponentialCDF);
         break;
      case kUserDefined:
      case kUndefined:
      default:
         break;   
      }
      fCDF = std::auto_ptr<IGenFunction>(cdf);
   }
   
   void GoFTest::SetDistributionFunction(const IGenFunction& f, Bool_t isPDF, Double_t xmin, Double_t xmax) {
      if (fDist > kUserDefined) {
         MATH_WARN_MSG("SetDistributionFunction","Distribution type is changed to user defined");
      }
      fDist = kUserDefined; 
      // function will be cloned inside the wrapper PDFIntegral of CDFWrapper classes
      if (isPDF) 
         fCDF = std::auto_ptr<IGenFunction>(new PDFIntegral(f, xmin, xmax) ); 
      else 
         fCDF = std::auto_ptr<IGenFunction>(new CDFWrapper(f, xmin, xmax) ); 
   }

   void GoFTest::Instantiate(const Double_t* sample, UInt_t sampleSize) {
      // initialization function for the template constructors
      Bool_t badSampleArg = sample == 0 || sampleSize == 0;
      if (badSampleArg) { 
         std::string msg = "'sample";
         msg += !sampleSize ? "Size' cannot be zero" : "' cannot be zero-length";
         MATH_ERROR_MSG("GoFTest", msg.c_str());
         assert(!badSampleArg);
      }
      fCDF = std::auto_ptr<IGenFunction>((IGenFunction*)0);
      fDist = kUserDefined;
      fMean = 0;
      fSigma = 0; 
      fSamples = std::vector<std::vector<Double_t> >(1);
      fTestSampleFromH0 = kTRUE;
      SetSamples(std::vector<const Double_t*>(1, sample), std::vector<UInt_t>(1, sampleSize));
   }
   
   Double_t GoFTest::GaussianCDF(Double_t x) const {
      return ROOT::Math::normal_cdf(x, fSigma, fMean);
   }

   Double_t GoFTest::ExponentialCDF(Double_t x) const {
      return ROOT::Math::exponential_cdf(x, 1.0 / fMean);
   }

   void GoFTest::LogSample() {
      transform(fSamples[0].begin(), fSamples[0].end(), fSamples[0].begin(), std::ptr_fun<Double_t, Double_t>(TMath::Log));
      SetParameters();
   }

/*
  Taken from (1)
*/ Double_t GoFTest::GetSigmaN(UInt_t N) const {
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
*/ Double_t GoFTest::PValueAD1Sample(Double_t A2) const {
      Double_t pvalue = 0.0;
      if (A2 <= 0.0) {
         return pvalue;
      } else if (A2 < 2.) {
         pvalue = std::pow(A2, -0.5) * std::exp(-1.2337141 / A2) * (2.00012 + (0.247105 - (0.0649821 - (0.0347962 - (0.011672 - 0.00168691 * A2) * A2) * A2) * A2) * A2);
      } else {
         pvalue = std::exp(-1. * std::exp(1.0776 - (2.30695 - (0.43424 - (.082433 - (0.008056 - 0.0003146 * A2) * A2) * A2) * A2) * A2));
      }   
      return 1. - pvalue;
   }

/*
  Taken from (1) -- Named for 2 samples but implemented for K. Restricted to K = 2 by the class's constructors 
*/ void GoFTest::AndersonDarling2SamplesTest(Double_t& pvalue, Double_t& testStat) const {
      pvalue = -1;
      testStat = -1;
      if (fTestSampleFromH0) {
         MATH_ERROR_MSG("AndersonDarling2SamplesTest", "Only 1-sample tests can be issued with a 1-sample constructed GoFTest object!");
         return;
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
      pvalue = PValueAD2Samples(A2, N); // standartized A2
      testStat = A2;
   }
   
   Double_t GoFTest::AndersonDarling2SamplesTest(const Char_t* option) const {
      Double_t pvalue, testStat;
      AndersonDarling2SamplesTest(pvalue, testStat);
      return (strncmp(option, "p", 1) == 0 || strncmp(option, "t", 1) != 0) ? pvalue : testStat;   
   }
   
/*
  Taken from (3)
*/ void GoFTest::AndersonDarlingTest(Double_t& pvalue, Double_t& testStat) const {
      pvalue = -1;
      testStat = -1;
      if (!fTestSampleFromH0) {
         MATH_ERROR_MSG("AndersonDarlingTest", "Only 2-sample tests can be issued with a 2-sample constructed GoFTest object!");
         return;
      } 
      if (fDist == kUndefined) {
         MATH_ERROR_MSG("AndersonDarlingTest", "Distribution type is undefined! Please use SetDistribution(GoFTest::EDistribution).");
         return;
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
      if (A2 != A2) {
         MATH_ERROR_MSG("AndersonDarlingTest", "Cannot compute p-value: data below or above the distribution's thresholds. Check sample consistency.");
         return;
      }
      pvalue = PValueAD1Sample(A2);
      testStat = A2;
   }
   
   Double_t GoFTest::AndersonDarlingTest(const Char_t* option) const {
      Double_t pvalue, testStat;
      AndersonDarlingTest(pvalue, testStat);
      return (strncmp(option, "p", 1) == 0 || strncmp(option, "t", 1) != 0) ? pvalue : testStat;
   }

   void GoFTest::KolmogorovSmirnov2SamplesTest(Double_t& pvalue, Double_t& testStat) const {
      pvalue = -1;
      testStat = -1;
      if (fTestSampleFromH0) {
         MATH_ERROR_MSG("KolmogorovSmirnov2SamplesTest", "Only 1-sample tests can be issued with a 1-sample constructed GoFTest object!");
         return;
      }
      const UInt_t na = fSamples[0].size();
      const UInt_t nb = fSamples[1].size();
      Double_t* a = new Double_t[na];
      Double_t* b = new Double_t[nb]; 
      std::copy(fSamples[0].begin(), fSamples[0].end(), a);
      std::copy(fSamples[1].begin(), fSamples[1].end(), b);
      pvalue = TMath::KolmogorovTest(na, a, nb, b, 0);
      testStat = TMath::KolmogorovTest(na, a, nb, b, "M");
   }
   
   Double_t GoFTest::KolmogorovSmirnov2SamplesTest(const Char_t* option) const {
      Double_t pvalue, testStat;
      KolmogorovSmirnov2SamplesTest(pvalue, testStat);
      return (strncmp(option, "p", 1) == 0 || strncmp(option, "t", 1) != 0) ? pvalue : testStat;    
   }

/* 
   Algorithm taken from (3) in page 737
*/ void GoFTest::KolmogorovSmirnovTest(Double_t& pvalue, Double_t& testStat) const {
      pvalue = -1;
      testStat = -1;
      if (!fTestSampleFromH0) {
         MATH_ERROR_MSG("KolmogorovSmirnovTest", "Only 2-sample tests can be issued with a 2-sample constructed GoFTest object!");
         return;
      }
      if (fDist == kUndefined) {
         MATH_ERROR_MSG("KolmogorovSmirnovTest", "Distribution type is undefined! Please use SetDistribution(GoFTest::EDistribution).");
         return;
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
      pvalue = TMath::KolmogorovProb(Dn * (TMath::Sqrt(n) + 0.12 + 0.11 / TMath::Sqrt(n)));
      testStat = Dn;
   }
   
   Double_t GoFTest::KolmogorovSmirnovTest(const Char_t* option) const {
      Double_t pvalue, testStat;
      KolmogorovSmirnovTest(pvalue, testStat);
      return (strncmp(option, "p", 1) == 0 || strncmp(option, "t", 1) != 0) ? pvalue : testStat;
   }

} // ROOT namespace
} // Math namespace

