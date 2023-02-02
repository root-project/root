// @(#)root/mathcore:$Id$
// Authors: Bartolomeu Rabacal    05/2010
/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/
// Header file for GoFTest

#ifndef ROOT_Math_GoFTest
#define ROOT_Math_GoFTest

#include "Math/WrappedFunction.h"
#include "TMath.h"

#include <memory>
#include <vector>

/*
*/

namespace ROOT {

   namespace Fit {
      class BinData;
   }
namespace Math {


/**
  @defgroup GoFClasses Goodness of Fit Tests
  Classical one-dimensional goodness of git tests for unbinned data.
  ROOT provides 1 sample goodness of fit test (comparison of data with a theoretical distribution) and
  2-sample test (comparison of two data sets) through the class ROOT::Math::GoFTest
  The algorithms provided are the Kolmogorov-Smirnov and Anderson-Darling.
  These tests could be applied approximately also to binned data, assuming the bin size is much smaller than the intrinsic
  data variations. It is assumed than a bin is like many data at the same bin center value.
  For these binned version tests look at `TH1::KolmogorovTest` and `TH1::AndersonDarlingTest`
  @ingroup MathCore
 */

/**
 * GoFTest class implementing the 1 sample and 2 sample goodness of fit tests
 * for uni-variate distributions and data.
 * The class implements the AndersonDarling and the KolmogorovSmirnov tests
 *
 * In the case of the 1-sample test the user needs to  provide:
 *   - input data
 *   - theoretical distribution. The distribution can be provided as a function object (functor) or an object implementing
 *     the `ROOT::Math::IGenFunction` interface. One can provide either the PDF (default) of the CDF (cumulative distribution)
 *     One can also provide a pre-defined function. In that case one needs to give also the distribution parameters otherwise the default values will be used.
 *     The pre-defined distributions are:
 *     - kGaussian  with default parameter mean=0, sigma=1
 *     - kExponential with default parameter rate=1
 *     - kLogNormal with default parameter meanlog=0, sigmalog=1
 *
 *     Note that one should not use data computed distribution parameters, otherwise the test will be biased.
 *     The 1-sample KS test using data computed quantities is called Lilliefors test (see https://en.wikipedia.org/wiki/Lilliefors_test)
 *
 *  @ingroup GoFClasses
 */


class GoFTest {
public:

   /// H0 distributions for using only with 1-sample tests.
   /// One should provide the distribution parameters otherwise the default values will be used
   enum EDistribution {
      kUndefined,       /// Default value for non templated 1-sample test. Set with SetDistribution
      kUserDefined,     /// For internal use only within the class's template constructor
      kGaussian,        /// Gaussian distribution with default  mean=0, sigma=1
      kLogNormal,       /// Lognormal distribution with default  meanlog=0, sigmalog=1
      kExponential      /// Exponential distribution with default rate=1
   };

   /// User input distribution option
   enum EUserDistribution {
      kCDF,             /// Input distribution is a CDF : cumulative distribution function
      kPDF              /// Input distribution is a PDF (Default value)
   };

   /// Goodness of Fit test types for using with the class's unary functions as a shorthand for the in-built methods
   enum ETestType {
      kAD,   /// Anderson-Darling Test. Default value
      kAD2s, /// Anderson-Darling 2-Samples Test
      kKS,   /// Kolmogorov-Smirnov Test
      kKS2s  /// Kolmogorov-Smirnov 2-Samples Test
   };

   /// Constructor for  2-samples tests
   GoFTest(size_t sample1Size, const Double_t* sample1, size_t sample2Size, const Double_t* sample2);

   /// Constructor for 1-sample tests with a specified distribution.
   /// If a specific distribution is not specified it can be set later using SetDistribution.
   GoFTest(size_t sampleSize, const Double_t* sample, EDistribution dist = kUndefined, const std::vector<double>  & distParams = {});

   /// Templated constructor for 1-sample tests with a user specified distribution as a functor object implementing `double operator()(double x)`.
   template<class Dist>
   GoFTest(size_t sampleSize, const Double_t* sample, Dist& dist, EUserDistribution userDist = kPDF,
           Double_t xmin = 1, Double_t xmax = 0)
   {
      Instantiate(sample, sampleSize);
      SetUserDistribution<Dist>(dist, userDist, xmin, xmax);
   }

   /// Constructor for 1-sample tests with a user specified distribution implementing the ROOT::Math::IGenFunction interface.
   GoFTest(size_t sampleSize, const Double_t* sample, const IGenFunction& dist, EUserDistribution userDist = kPDF,
           Double_t xmin = 1, Double_t xmax = 0)
   {
      Instantiate(sample, sampleSize);
      SetUserDistribution(dist, userDist, xmin, xmax);
   }

   /// Sets the user input distribution function for 1-sample test as a generic functor object.
   template<class Dist>
   void SetUserDistribution(Dist& dist, EUserDistribution userDist = kPDF, Double_t xmin = 1, Double_t xmax = 0) {
      WrappedFunction<Dist&> wdist(dist);
      SetDistributionFunction(wdist, userDist, xmin, xmax);
   }

   ///  Sets the user input distribution function for 1-sample test using the ROOT::Math::IGenFunction interface.
   void SetUserDistribution(const IGenFunction& dist, GoFTest::EUserDistribution userDist = kPDF, Double_t xmin = 1, Double_t xmax = 0) {
      SetDistributionFunction(dist, userDist, xmin, xmax);
   }

   /// Sets the user input distribution as a probability density function for 1-sample tests.
   template<class Dist>
   void SetUserPDF(Dist& pdf, Double_t xmin = 1, Double_t xmax = 0) {
      SetUserDistribution<Dist>(pdf, kPDF, xmin, xmax);
   }

   /// Specialization to set the user input distribution as a probability density function for 1-sample tests using the ROOT::Math::IGenFunction interface.
   void SetUserPDF(const IGenFunction& pdf, Double_t xmin = 1, Double_t xmax = 0) {
      SetUserDistribution(pdf, kPDF, xmin, xmax);
   }

   /// Sets the user input distribution as a cumulative distribution function for 1-sample tests.
   /// The CDF must return zero for x=xmin and 1 for x=xmax.
   template<class Dist>
   void SetUserCDF(Dist& cdf, Double_t xmin = 1, Double_t xmax = 0) {
      SetUserDistribution<Dist>(cdf, kCDF, xmin, xmax);
   }

   /// Specialization to set the user input distribution as a cumulative distribution function for 1-sample tests.
   void SetUserCDF(const IGenFunction& cdf, Double_t xmin = 1, Double_t xmax = 0)  {
      SetUserDistribution(cdf, kCDF, xmin, xmax);
   }


   /// Sets the distribution for the predefined distribution types and optionally its parameters for 1-sample tests.
   void SetDistribution(EDistribution dist, const std::vector<double>  & distParams = {});


   virtual ~GoFTest();

   /// Performs the Anderson-Darling 2-Sample Test.
   ///  The Anderson-Darling K-Sample Test algorithm is described and taken from
   ///  http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/andeksam.htm
   ///  and from
   ///   (1) Scholz F.W., Stephens M.A. (1987), K-sample Anderson-Darling Tests, Journal of the American Statistical Association, 82, 918–924.
   ///   (2-samples variant implemented).
   void AndersonDarling2SamplesTest(Double_t& pvalue, Double_t& testStat) const;

   ///  Anderson-Darling 2-Sample Test.
   ///  Returns by default the p-value; when using option "t" returns the test statistic value "A2".
   Double_t AndersonDarling2SamplesTest(const Char_t* option = "p") const;

   /**
   Performs the Anderson-Darling 1-Sample Test.
   The Anderson-Darling 1-Sample Test algorithm for a specific distribution is described at
   http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/andedarl.htm
   and described and taken from (2)
   Marsaglia J.C.W., Marsaglia G. (2004), Evaluating the Anderson-Darling Distribution, Journal of Statistical Software, Volume 09, Issue i02.
   and described and taken from (3)
   Lewis P.A.W. (1961), The Annals of Mathematical Statistics, Distribution of the Anderson-Darling Statistic, Volume 32, Number 4, 1118-1124.
   */
   void AndersonDarlingTest(Double_t& pvalue, Double_t& testStat) const;

   /// Anderson-Darling 2-Sample Test.
   /// Returns default p-value; option "t" returns the test statistic value "A2"
   Double_t AndersonDarlingTest(const Char_t* option = "p") const;

   /**
   * @brief Kolmogorov-Smirnov 2-Samples Test.
   The Kolmogorov-Smirnov 2-Samples Test algorithm is described at
   http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/ks2samp.htm
   and described and taken from
   http://root.cern.ch/root/html/TMath.html#TMath:KolmogorovTest
   */
   void KolmogorovSmirnov2SamplesTest(Double_t& pvalue, Double_t& testStat) const;

   /// Kolmogorov-Smirnov 2-Samples Test.
   /// Returns by default the p-value; option "t" returns the test statistic value "Dn".
   Double_t KolmogorovSmirnov2SamplesTest(const Char_t* option = "p") const;

  /**
   * @brief  Kolmogorov-Smirnov 1-Sample Test.
   *
     The Kolmogorov-Smirnov 1-Sample Test algorithm for a specific distribution is described at
     http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/kstest.htm
     and described and taken from (4)
     Press W. H., Teukolsky S.A., Vetterling W.T., Flannery B.P. (2007), Numerical Recipes -
     The Art of Scientific Computing (Third Edition), Cambridge University Press
   */
   void KolmogorovSmirnovTest(Double_t& pvalue, Double_t& testStat) const;

   /// Kolmogorov-Smirnov 1-Sample Test.
   /// Returns default p-value; option "t" returns the test statistic value "Dn".
   Double_t KolmogorovSmirnovTest(const Char_t* option = "p") const;

   /// The class's unary functions performing the gif test according to the ETestType provided.
   void operator()(ETestType test, Double_t& pvalue, Double_t& testStat) const;

   /// Returns default Anderson Darling 1-Sample Test and default p-value; option "t" returns the test statistic value
   /// specific to the test type.
   Double_t operator()(ETestType test = kAD, const Char_t* option = "p") const;

   /// Computation of the K-Sample Anderson-Darling Test's p-value as described in (1)
   // given a normalized test statistic. The first variant described in the paper is used.
   static Double_t PValueADKSamples(size_t nsamples, Double_t A2 );

   /// Compute the 2-Sample Anderson Darling test for binned data
   /// assuming equal data are present at the bin center values.
   /// Used by `TH1::AndersonDarling`
   static void  AndersonDarling2SamplesTest(const ROOT::Fit::BinData & data1, const ROOT::Fit::BinData & data2, Double_t& pvalue, Double_t& testStat);

private:

   GoFTest();                       ///< Disallowed default constructor
   GoFTest(GoFTest& gof);           ///< Disallowed copy constructor
   GoFTest operator=(GoFTest& gof); ///< Disallowed assign operator

   std::unique_ptr<IGenFunction> fCDF;  ///< Pointer to CDF used in 1-sample test


   EDistribution fDist;                ///< Type of distribution
   std::vector<Double_t> fParams;      ///< The distribution parameters (e.g. fParams[0] = mean, fParams[1] = sigma for a Gaussian)

   std::vector<Double_t> fCombinedSamples;       ///< The combined data

   std::vector<std::vector<Double_t> > fSamples;  ///< The input data

   Bool_t fTestSampleFromH0;

   void SetCDF();
   void SetDistributionFunction(const IGenFunction& cdf, Bool_t isPDF, Double_t xmin, Double_t xmax);

   void Instantiate(const Double_t* sample, size_t sampleSize);


   Double_t LogNormalCDF(Double_t x) const;
   Double_t GaussianCDF(Double_t x) const;
   Double_t ExponentialCDF(Double_t x) const;

   /// Computation of sigma_N as described in (1)
   static Double_t GetSigmaN(const std::vector<size_t> & ns, size_t N);

   /// Linear interpolation used in GoFTest::PValueAD2Samples
   static Double_t InterpolatePValues(int nsamples,Double_t A2);

   /// Computation of the 1-Sample Anderson-Darling Test's p-value
   Double_t PValueAD1Sample(Double_t A2) const;

   /// Applies the logarithm to the sample when the specified distribution to test is LogNormal
   void LogSample();

   /// set a vector of samples
   void SetSamples(std::vector<const Double_t*> samples, const std::vector<size_t> samplesSizes);

   /// Sets the distribution parameters
   void SetParameters(const std::vector<double> & params);

}; // end GoFTest class


} // ROOT namespace
} // Math namespace
#endif
