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
#include "Math/Math.h"
#include "Math/IFunction.h"
#include "Math/IFunctionfwd.h"
#include "Math/Integrator.h"
#include "Math/ProbFuncMathCore.h"
#include "Math/WrappedFunction.h"

#include "Math/GoFTest.h"

#include "Fit/BinData.h"

#include "TStopwatch.h"

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
   : fDist(kUndefined),
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
   : fDist(dist),
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
      fCombinedSamples.assign(std::accumulate(samplesSizes.begin(), samplesSizes.end(), 0u), 0.0);
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
         /* fall through */
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
      fCDF.reset(cdf);
   }

   void GoFTest::SetDistributionFunction(const IGenFunction& f, Bool_t isPDF, Double_t xmin, Double_t xmax) {
      if (fDist > kUserDefined) {
         MATH_WARN_MSG("SetDistributionFunction","Distribution type is changed to user defined");
      }
      fDist = kUserDefined;
      // function will be cloned inside the wrapper PDFIntegral of CDFWrapper classes
      if (isPDF)
         fCDF.reset(new PDFIntegral(f, xmin, xmax) );
      else
         fCDF.reset(new CDFWrapper(f, xmin, xmax) );
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
      fCDF.reset((IGenFunction*)0);
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
      transform(fSamples[0].begin(), fSamples[0].end(), fSamples[0].begin(),
                std::function<Double_t(Double_t)>(TMath::Log));
      SetParameters();
   }

/* 
  Taken from (1)
*/ 
   Double_t GoFTest::GetSigmaN(const std::vector<UInt_t> & ns, UInt_t N) {
      // compute moments of AD distribution (from Scholz-Stephen paper, paragraph 3)

      Double_t sigmaN = 0.0, h = 0.0, H = 0.0, g = 0.0, a, b, c, d, k = ns.size();

      for (UInt_t i = 0; i < ns.size(); ++i) {
         H += 1.0 /  double( ns[i] );
      }

      // use approximate formulas for large N
      // cache Sum( 1 / i)
      if (N < 2000) { 
         std::vector<double> invI(N); 
         for (UInt_t i = 1; i <= N - 1; ++i) {
            invI[i] = 1.0 / i; 
            h += invI[i]; 
         }
         for (UInt_t i = 1; i <= N - 2; ++i) {
            double tmp = invI[N-i];
            for (UInt_t j = i + 1; j <= N - 1; ++j) {
               g += tmp * invI[j];
            }
         }
      }
      else {
         // for N larger than 2000 error difference in g is ~ 5 10^-3 while in h is at the level of 10^-5
         const double emc = 0.5772156649015328606065120900824024; // Euler-Mascheroni constant
         h = std::log(double(N-1) ) + emc;
         g = (M_PI)*(M_PI)/6.0;
      }
      double k2 = std::pow(k,2);
      a = (4 * g - 6) * k + (10 - 6 * g) * H - 4 * g + 6;
      b = (2 * g - 4) * k2 + 8 * h * k + (2 * g - 14 * h - 4) * H - 8 * h + 4 * g - 6;
      c = (6 * h + 2 * g - 2) * k2 + (4 * h - 4 *g + 6) * k + (2 * h - 6) * H + 4 * h;
      d = (2 * h + 6) * k2 - 4 * h * k;
      sigmaN +=  a * std::pow(double(N),3) + b * std::pow(double(N),2) + c * N + d;
      sigmaN /= ( double(N - 1) * double(N - 2) * double(N - 3) );
      sigmaN = TMath::Sqrt(sigmaN);
      return sigmaN;
   }


   Double_t GoFTest::PValueADKSamples(UInt_t nsamples, Double_t tx)  {

      /*
       Computation of p-values according to 
       "K-Sample Anderson-Darling Tests" by F.W. Scholz 
       and M.A. Stephens (1987), Journal of the American Statistical Association, 
       Vol 82, No. 399, pp 918-924.
       Code from kSamples package from R (author F. Scholtz)

       This function uses the upper T_m quantiles as obtained via simulation of
       the Anderson-Darling test statistics (Nsim = 2*10^6) with sample sizes n=500
       for each sample, and after standardization, in order to emulate the Table 1 
       values given in the above reference. However, here we estimate p-quantiles
       for p = .00001,.00005,.0001,.0005,.001,.005,.01,.025,.05,.075,
       .1,.2,.3,.4,.5,.6,.7,.8,.9,.925,.95,.975,.99,.9925,.995,.9975,.999,
       .99925,.9995,.99975,.9999,.999925,.99995,.999975,.99999
       First the appropriate p-quantiles are determined from those simulated
       for ms = 1,2,3,4,6,8,10, Inf, interpolating to the given value of m. 
       Since we use only m=2 we avoid this interpolation. 

       Next linear inetrpolation to find the observed p value given the observed test statistic value. 
       We use interpolation in the test statistic -> log((1-p)/p) domain
       and we extrapolatelinearly) beyond p = .00001 and .99999.
      */
      
      // sample values 
      //double ms[] = { 1, 2, 3, 4, 6, 8, 10, TMath::Infinity() };
      //int ns = ms.size();
      const int ns = 8;
      double ts[ ]           = { -1.1954, -1.5806, -1.8172, 
                                 -2.0032, -2.2526, -2.4204, -2.5283, -4.2649, -1.1786, -1.5394, 
                                 -1.7728, -1.9426, -2.1685, -2.3288, -2.4374, -3.8906, -1.166, 
                                 -1.5193, -1.7462, -1.9067, -2.126, -2.2818, -2.3926, -3.719, 
                                 -1.1407, -1.4659, -1.671, -1.8105, -2.0048, -2.1356, -2.2348, 
                                 -3.2905, -1.1253, -1.4371, -1.6314, -1.7619, -1.9396, -2.0637, 
                                 -2.1521, -3.0902, -1.0777, -1.3503, -1.5102, -1.6177, -1.761, 
                                 -1.8537, -1.9178, -2.5758, -1.0489, -1.2984, -1.4415, -1.5355, 
                                 -1.6625, -1.738, -1.7936, -2.3263, -0.9978, -1.2098, -1.3251, 
                                 -1.4007, -1.4977, -1.5555, -1.5941, -1.96, -0.9417, -1.1187, 
                                 -1.209, -1.2671, -1.3382, -1.379, -1.405, -1.6449, -0.8981, -1.0491, 
                                 -1.1235, -1.1692, -1.2249, -1.2552, -1.2755, -1.4395, -0.8598, 
                                 -0.9904, -1.0513, -1.0879, -1.1317, -1.155, -1.1694, -1.2816, 
                                 -0.7258, -0.7938, -0.8188, -0.8312, -0.8435, -0.8471, -0.8496, 
                                 -0.8416, -0.5966, -0.617, -0.6177, -0.6139, -0.6073, -0.5987, 
                                 -0.5941, -0.5244, -0.4572, -0.4383, -0.419, -0.4033, -0.3834, 
                                 -0.3676, -0.3587, -0.2533, -0.2966, -0.2428, -0.2078, -0.1844, 
                                 -0.1548, -0.1346, -0.1224, 0, -0.1009, -0.0169, 0.0304, 0.0596, 
                                 0.0933, 0.1156, 0.1294, 0.2533, 0.1571, 0.2635, 0.3169, 0.348, 
                                 0.3823, 0.4038, 0.4166, 0.5244, 0.5357, 0.6496, 0.6992, 0.7246, 
                                 0.7528, 0.7683, 0.7771, 0.8416, 1.2255, 1.2989, 1.3202, 1.3254, 
                                 1.3305, 1.3286, 1.3257, 1.2816, 1.5262, 1.5677, 1.5709, 1.5663, 
                                 1.5561, 1.5449, 1.5356, 1.4395, 1.9633, 1.943, 1.919, 1.8975, 
                                 1.8641, 1.8389, 1.8212, 1.6449, 2.7314, 2.5899, 2.5, 2.4451, 
                                 2.3664, 2.3155, 2.2823, 1.96, 3.7825, 3.4425, 3.2582, 3.1423, 
                                 3.0036, 2.9101, 2.8579, 2.3263, 4.1241, 3.716, 3.4984, 3.3651, 
                                 3.2003, 3.0928, 3.0311, 2.4324, 4.6044, 4.0847, 3.8348, 3.6714, 
                                 3.4721, 3.3453, 3.2777, 2.5758, 5.409, 4.7223, 4.4022, 4.1791, 
                                 3.9357, 3.7809, 3.6963, 2.807, 6.4954, 5.5823, 5.1456, 4.8657, 
                                 4.5506, 4.3275, 4.2228, 3.0902, 6.8279, 5.8282, 5.3658, 5.0749, 
                                 4.7318, 4.4923, 4.3642, 3.1747, 7.2755, 6.197, 5.6715, 5.3642, 
                                 4.9991, 4.7135, 4.5945, 3.2905, 8.1885, 6.8537, 6.2077, 5.8499, 
                                 5.4246, 5.1137, 4.9555, 3.4808, 9.3061, 7.6592, 6.85, 6.4806, 
                                 5.9919, 5.6122, 5.5136, 3.719, 9.6132, 7.9234, 7.1025, 6.6731, 
                                 6.1549, 5.8217, 5.7345, 3.7911, 10.0989, 8.2395, 7.4326, 6.9567, 
                                 6.3908, 6.011, 5.9566, 3.8906, 10.8825, 8.8994, 7.8934, 7.4501, 
                                 6.9009, 6.4538, 6.2705, 4.0556, 11.8537, 9.5482, 8.5568, 8.0283, 
                                 7.4418, 6.9524, 6.6195, 4.2649 };
   


   
   
      // p values bins 
      double p[] = { .00001,.00005,.0001,.0005,.001,.005,.01,.025,.05,.075,.1,.2,.3,.4,.5,.6,.7,.8,.9,
                       .925,.95,.975,.99,.9925,.995,.9975,.999,.99925,.9995,.99975,.9999,.999925,.99995,.999975,.99999 };

      //int nbins = p.size();
      const int nbins = 35;
      //assert ( nbins*ns == ts.size() ); 

      // get ts values for nsamples = 2
      // corresponding value is for m=nsamples-1
      int offset = 0;  // for m = 1 (i.e. for nsamples = 2)
      if (nsamples != 2) { 
         MATH_ERROR_MSG("InterpolatePValues", "Interpolation not implemented for nsamples not equal to  2");
         return 0;
      }
      std::vector<double> ts2(nbins); // ts values for nsamples = 2
      std::vector<double> lp(nbins);   // log ( p / (1-p) )
      for (int i = 0; i < nbins; ++i)  
      { 
         ts2[i] = ts[offset+ i * ns];                 
         p[i] = 1.-p[i];
         lp[i] = std::log( p[i]/(1.-p[i] ) ); 
      }
      // do linear interpolation to find right lp value for given observed test staistic value
      //auto it = std::lower_bound(ts2.begin(), ts2.end(), tx ); 
      int i1 = std::distance(ts2.begin(),  std::lower_bound(ts2.begin(), ts2.end(), tx ) ) - 1; 
      int i2 = i1+1;
      // if tx is before min of tabluated data
      if (i1 < 0) { 
         i1 = 0;
         i2 = 1;
      }
      // if tx is after max of tabulated data
      if (i2 >= int(ts2.size()) ) { 
         i1 = ts2.size()-2; 
         i2 = ts2.size()-1;
      }

      //std::cout << i1 << " , " << i2 << std::endl;
      assert(i1 < (int) lp.size() && i2 < (int) lp.size() ); 
      double lp1 = lp[i1]; 
      double lp2 = lp[i2];
      double tx1 = ts2[i1];
      double tx2 = ts2[i2];

      //std::cout << " tx1,2 " << tx1 << "  " << tx2 << std::endl;
      /// find interpolated (or extrapolated value)( 
      double lp0 = (lp1-lp2) * (tx - tx2)/ ( tx1-tx2) + lp2; 


      double p0 = exp(lp0)/(1. + exp(lp0) );
      return p0; 

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


// code from kSamples (R) F. Scholz

/* computes the k-sample Anderson-Darling test statistics in both original 
   and alternative versions for the nonparametric (rank) test described in
   Scholz F.W. and Stephens M.A. (1987), K-sample Anderson-Darling Tests,
   Journal of the American Statistical Association, Vol 82, No. 399,
   pp. 918-924

   Arguments:
   adk: double array with length 2, stores AkN2 and AakN2
   k: integer, number of samples being compared
   x: double array storing the concatenated samples in the same order as ns
   ns: integer array storing the k sample sizes, corresponding to x
   zstar: double array storing the l distinct ordered observations in the
      pooled sample
   l: integer, length of zstar

   Outputs:
   when the computation ends, AkN2 and AakN2 are stored in the given memory
   pointed by adk
*/

/* counts and returns the number of occurrence of a given number 
   in a double array */
int getCount(double z, const double *dat, int n) {
   int i;
   int count = 0;

   for (i = 0; i < n; i++) {
      if (dat[i] == z) {
         count++;
      }
   }

   return(count);
}

/* computes and returns the sum of elements in a given integer array */ 
int getSum(const int *x, int n) {
   int i;
   int sum = 0;

   for (i = 0; i < n; i++) {
      sum += x[i];
   }

   return(sum);
}


void adkTestStat(double *adk, const std::vector<std::vector<double> > & samples, const std::vector<double> & zstar) {

   int i;
   int j;

   int nsum; /* total sample size = n_1 + ... + n_k */
   int k = samples.size();
   int l = zstar.size();

   /* fij records the number of observations in the ith sample coinciding
      with zstar[j], where i = 1, ..., k, and j = 1, ..., l */
   std::vector<int> fij (k*l);
   /* lvec is an integer vector with length l,
      whose jth entry = \sum_{i=1}^{k} f_{ij}, i.e., the multiplicity
      of zstar[j] */
   std::vector<int> lvec(l);

   /* for computation */
   double mij;
   double maij;
   double innerSum;
   double aInnerSum;
   double bj;
   double baj;
   double tmp;

   /* samples is a two-dimensional double array with length k;
      it stores an array of k pointers to double arrays which are
      the k samples beeing compared */
// double **samples;

   /* dynamically allocate memory */
        //std::vector< std::vector<double> > samples(k);
   std::vector<int> ns(k);
   nsum = 0;
   for (i = 0; i < k; i++) {
      ns[i] = samples[i].size();
      nsum += ns[i];
   }

   /* fij: k*l integer matrix, where l is the length of zstar and
      k is the number of samples being compared
      lvec: integer vector of length l, records the multiplicity of
      each element of zstar */
   for (j = 0; j < l; j++) {
      lvec[j] = 0;
      for (i = 0; i < k; i++) {
         fij[i + j*k] = getCount(zstar[j], &samples[i][0], ns[i]);
         lvec[j] += fij[i + j*k];
      }
   }

   // loop on samples to compute the adk's
   // Formula (6) and (7) of the paper
   adk[0] = adk[1] = 0;
   for (i = 0; i < k; i++) {
      mij = 0;
      maij = 0;
      innerSum = 0;
      aInnerSum = 0;

      for (j = 0; j < l; j++) {
         mij += fij[i + j*k];
         maij = mij - (double) fij[i + j*k] / 2.0;
         bj = getSum(&lvec[0], j + 1);
         baj = bj - (double) lvec[j] / 2.0;

         if (j < l - 1) {
            tmp = (double) nsum * mij - (double) ns[i] * bj;
            innerSum = innerSum + (double) lvec[j] * tmp * tmp /
                       (bj * ((double) nsum - bj));
         }

         tmp = (double) nsum * maij - (double) ns[i] * baj;
         aInnerSum = aInnerSum + (double) lvec[j] * tmp * tmp /
                     (baj * (nsum - baj) - nsum * (double) lvec[j] / 4.0);
      }

      adk[0] = adk[0] + innerSum / ns[i]; /* AkN2*/
      adk[1] = adk[1] + aInnerSum / ns[i]; /* AakN2 */
   }

   /* k-sample Anderson-Darling test statistics in both original and
      alternative versions, AkN2 and AakN2, are stored in the given
      double array adk */
   adk[0] = adk[0] / (double) nsum; /* AkN2*/
   adk[1] = (nsum - 1) * adk[1] / ((double) nsum * (double) nsum); /* AakN2 */

   // /* free pointers */
   // for (i = 0; i < k; i++) {
   //    free(samples[i]);
   // }
   // free(samples);

}


/*
  Taken from (1) -- Named for 2 samples but implemented for K. Restricted to K = 2 by the class's constructors
*/
void GoFTest::AndersonDarling2SamplesTest(Double_t& pvalue, Double_t& testStat) const {
      pvalue = -1;
      testStat = -1;
      if (fTestSampleFromH0) {
         MATH_ERROR_MSG("AndersonDarling2SamplesTest", "Only 1-sample tests can be issued with a 1-sample constructed GoFTest object!");
         return;
      }
      std::vector<Double_t> z(fCombinedSamples);
      // unique removes all consecutives duplicates elements. This is exactly what we wants 
      // for example unique of v={1,2,2,3,1,2,3,3} results in {1,2,3,1,2,3}  which is exactly what we wants 
      std::vector<Double_t>::iterator endUnique = std::unique(z.begin(), z.end()); //z_j's in (1)
      z.erase(endUnique, z.end() ); 
      std::vector<UInt_t> h; // h_j's in (1)
      std::vector<Double_t> H; // H_j's in (1)
      UInt_t N = fCombinedSamples.size();
      Double_t A2 = 0.0; // Anderson-Darling A^2 Test Statistic

#ifdef USE_OLDIMPL      

      TStopwatch w; w.Start();

      unsigned int nSamples = fSamples.size();

      // old implementation 
      for (std::vector<Double_t>::iterator data = z.begin(); data != endUnique; ++data) {
         UInt_t n = std::count(fCombinedSamples.begin(), fCombinedSamples.end(), *data);
         h.push_back(n);
         H.push_back(std::count_if(fCombinedSamples.begin(), fCombinedSamples.end(),
                     std::bind(std::less<Double_t>(), std::placeholders::_1, *data)) + n / 2.);
      }
      std::cout << "time for H";
      w.Print();
      w.Reset(); w.Start();
      std::vector<std::vector<Double_t> > F(nSamples); // F_ij's in (1)
      for (UInt_t i = 0; i < nSamples; ++i) {
         for (std::vector<Double_t>::iterator data = z.begin(); data != endUnique; ++data) {
            UInt_t n = std::count(fSamples[i].begin(), fSamples[i].end(), *data);
            F[i].push_back(std::count_if(fSamples[i].begin(), fSamples[i].end(),
                           std::bind(std::less<Double_t>(), std::placeholders::_1, *data)) + n / 2.);
         }
      }
      std::cout << "time for F";
      w.Print();
      for (UInt_t i = 0; i < nSamples; ++i) {
         Double_t sum_result = 0.0;
         UInt_t j = 0;
         w.Reset(); w.Start();      
         for (std::vector<Double_t>::iterator data = z.begin(); data != endUnique; ++data) {
            sum_result += h[j] *  TMath::Power(N * F[i][j]- fSamples[i].size() * H[j], 2) / (H[j] * (N - H[j]) - N * h[j] / 4.0);
            ++j;
         }
         std::cout << "time for sum_resut"; 
         w.Print(); 
         std::cout << "sum_result " << sum_result << std::endl;
         A2 += 1.0 / fSamples[i].size() * sum_result;
      }
      A2 *= (N - 1) / (TMath::Power(N, 2)); // A2_akN in (1)

      std::cout << "A2 - old Bartolomeo code " << A2 << std::endl;
#endif
      // w.Reset();
      // w.Start();

      double adk[2] = {0,0};

      //debug
      // std::cout << "combined samples\n";
      // for (int i = 0; i < fCombinedSamples.size(); ++i)
      //    std::cout << fCombinedSamples[i] << " ,";
      // std::cout << std::endl;
      // std::cout << ns[0] << "  " << ns[1] << std::endl;
      // std::cout << "Z\n";
      // for (int i = 0; i < z.size(); ++i)
      //    std::cout << z[i] << " ,";
      // std::cout << std::endl;

      // use function from kSamples code
      adkTestStat(adk, fSamples, z );
      // w.Print();
      // std::cout << "A2 - new kSamples  code " << adk[0] << "  " << adk[1]  << std::endl;

      A2 = adk[0]; 

      // compute the normalized test statistic 

      std::vector<UInt_t> ns(fSamples.size());
      for (unsigned int k = 0; k < ns.size(); ++k) ns[k] = fSamples[k].size();
      Double_t sigmaN = GetSigmaN(ns, N);
      A2 -= fSamples.size() - 1;
      A2 /= sigmaN; // standartized test statistic

      pvalue = PValueADKSamples(2,A2); 
      testStat = A2;
      return;
   }


/*
   Compute Anderson Darling test for two binned data set. 
   A binned data set can be seen as many identical observation happening at the center of the bin
   In this way it is trivial to apply the formula (6) in the paper of W. Scholz, M. Stephens, "K-Sample Anderson-Darling Tests"
   to the case of histograms. See also http://arxiv.org/pdf/0804.0380v1.pdf paragraph  3.3.5
   It is importat that empty bins are not present 
*/
   void GoFTest::AndersonDarling2SamplesTest(const ROOT::Fit::BinData &data1, const ROOT::Fit::BinData & data2, Double_t& pvalue, Double_t& testStat)  {
      pvalue = -1;
      testStat = -1;
      // 
      // compute cumulative sum of bin counts 
      // std::vector<double> sum1(data1.Size() ); 
      // std::vector<double> sum2(data2.Size() ); 
      // std::vector<double> sumAll(data1.Size() + data2.Size() ); 
      
      if (data1.NDim() != 1 && data2.NDim() != 1) {
            MATH_ERROR_MSG("AndersonDarling2SamplesTest", "Bin Data set must be one-dimensional ");
            return;
      }
      unsigned int n1 = data1.Size(); 
      unsigned int n2 = data2.Size(); 
      double ntot1 = 0; 
      double ntot2 = 0;
      

      // make a combined data set and sort it 
      std::vector<double> xdata(n1+n2); 
      for (unsigned int i = 0; i < n1; ++i) {
         double value = 0; 
         const double * x = data1.GetPoint(i, value);
         xdata[i] = *x; 
         ntot1 += value; 
      }
      for (unsigned int i = 0; i < n2; ++i) {
         double value = 0;
         const double * x = data2.GetPoint(i, value);
         xdata[n1+i] = *x;
         ntot2 += value; 
      }
      double nall = ntot1+ntot2; 
      // sort the combined data 
      std::vector<unsigned int> index(n1+n2);
      TMath::Sort(n1+n2, &xdata[0], &index[0], false );  

      // now compute the sums for the tests 
      double sum1 = 0; 
      double sum2 = 0;
      double sumAll = 0; 
      double adsum = 0;
      unsigned int j = 0; 

      while( j < n1+n2 ) { 
//      for (unsigned int j = 0; j < n1+n2; ++j) { 
         // skip equal observations
         double x = xdata[ index[j] ]; 
         unsigned int k = j; 
         // loop on the bins with the same center value 
         double t = 0;
         do { 
            unsigned int i = index[k];
            double value = 0; 
            if (i < n1 ) {
               value = data1.Value(i); 
               sum1 += value;
            }
            else { 
               // from data2
               i -= n1;
               assert(i < n2);
               value = data2.Value(i); 
               sum2 += value; 
            }
            sumAll += value;
            t += value; 
            //std::cout << "j " << j << " k " << k << " data " << x << " index " << index[k] << " value " << value << std::endl;
            k++;
         } while ( k < n1+n2 && xdata[ index[k] ] == x  );


         j = k; 
         // skip last point
         if (j < n1+n2) {
            double tmp1 =  ( nall * sum1 - ntot1 * sumAll );
            double tmp2 =  ( nall * sum2 - ntot2 * sumAll );
            adsum += t * (tmp1*tmp1/ntot1 + tmp2*tmp2/ntot2) / ( sumAll *  (nall - sumAll) ) ;

            //std::cout << "comp sum " << adsum << "  " << t << "  " << sumAll << " s1 " << sum1 << " s2 " << sum2 << " tmp1 " << tmp1 << " tmp2 " << tmp2 << std::endl;
         }
      }
      double A2 = adsum / nall; 

      // compute the normalized test statistic 
      std::vector<unsigned int> ns(2); 
      ns[0] = ntot1; 
      ns[1] = ntot2;
      //std::cout << " ad2 = " << A2 << " nall " << nall;

      Double_t sigmaN = GetSigmaN(ns,nall);
      A2 -= 1;
      A2 /= sigmaN; // standartized test statistic

      //std::cout << " sigmaN " << sigmaN << " new A2 " << A2;

      pvalue = PValueADKSamples(2,A2); 
      //std::cout << " pvalue = " << pvalue << std::endl;
      testStat = A2;
      return;
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
      std::vector<Double_t> a(na);
      std::vector<Double_t> b(nb);
      std::copy(fSamples[0].begin(), fSamples[0].end(), a.begin());
      std::copy(fSamples[1].begin(), fSamples[1].end(), b.begin());
      pvalue = TMath::KolmogorovTest(na, a.data(), nb, b.data(), 0);
      testStat = TMath::KolmogorovTest(na, a.data(), nb, b.data(), "M");
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

