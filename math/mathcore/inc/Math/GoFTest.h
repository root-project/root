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

#if !defined(__CINT__) && !defined(__MAKECINT__)
#include <stdexcept>
#endif


#ifndef ROOT_Math_WrappedFunction
#include "Math/WrappedFunction.h"
#endif
#ifndef ROOT_TMath
#include "TMath.h"
#endif


/*
  Goodness of Fit Statistical Tests Toolkit -- Anderson-Darling and Kolmogorov-Smirnov 1- and 2-Samples Tests
*/
                   
namespace ROOT {
namespace Math {

class GoFTest {
public:

   enum EDistribution { // H0 distributions for using only with 2-samples tests
      kUserDefined = -1, // Internal use only for the class's template constructor
      kGaussian,         // Default value
      kLogNormal,
      kExponential
   };
  
   enum ETestType { // Goodness of Fit test types for using with the class's unary funtion as a shorthand for the in-built methods
      kAD,   // Anderson-Darling Test. Default value
      kAD2s, // Anderson-Darling 2-Samples Test
      kKS,   // Kolmogorov-Smirnov Test
      kKS2s // Kolmogorov-Smirnov 2-Samples Test
   };
  
#if !defined(__CINT__) && !defined(__MAKECINT__)
   struct BadSampleArgument : public std::invalid_argument {
      BadSampleArgument(std::string type);
   };
  
   struct DegenerateSamples : public std::domain_error {
      DegenerateSamples(std::string type);
   };
#endif
  
   /* Constructor for using only with 2-samples tests */
   GoFTest(const Double_t* sample1, UInt_t sample1Size, const Double_t* sample2, UInt_t sample2Size)
#if !defined(__CINT__) && !defined(__MAKECINT__)
      throw(BadSampleArgument, std::bad_exception) 
#endif
      ;
  
   /* Constructor for using only with 1-sample tests with a specified distribution */
   GoFTest(const Double_t* sample, UInt_t sampleSize, EDistribution dist = kGaussian)
#if !defined(__CINT__) && !defined(__MAKECINT__)
      throw(BadSampleArgument, std::bad_exception)
#endif  
      ;
  
   /* Templated constructor for using only with 1-sample tests with a user specified distribution */
   template<class Dist>
   GoFTest(const Double_t* sample, UInt_t sampleSize, const Dist& cdf) {
      Instantiate(sample, sampleSize);
      SetCDF(new ROOT::Math::WrappedFunction<const Dist&>(cdf));
   }
   
   virtual ~GoFTest();

/*
  The Anderson-Darling K-Sample Test algorithm is described and taken from 
  http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/andeksam.htm
  and described and taken from (1)
  Scholz F.W., Stephens M.A. (1987), K-sample Anderson-Darling Tests, Journal of the American Statistical Association, 82, 918–924. (2-samples variant implemented)
*/
   Double_t AndersonDarling2SamplesTest(const Char_t* option = "p") const; // Returns default p-value; option "t" returns the test statistic value "A2"
  
/*
  The Anderson-Darling 1-Sample Test algorithm for a specific distribution is described at 
  http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/andedarl.htm
  and described and taken from (2)
  Marsaglia J.C.W., Marsaglia G. (2004), Evaluating the Anderson-Darling Distribution, Journal of Statistical Software, Volume 09, Issue i02.
  and described and taken from (3)
  Lewis P.A.W. (1961), The Annals of Mathematical Statistics, Distribution of the Anderson-Darling Statistic, Volume 32, Number 4, 1118-1124. 
*/
   Double_t AndersonDarlingTest(const Char_t* option = "p") const; // Returns default p-value; option "t" returns the test statistic value "A2"
  
/*
  The Kolmogorov-Smirnov 2-Samples Test algorithm is described at
  http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/ks2samp.htm
  and described and taken from
  http://root.cern.ch/root/html/TMath.html#TMath:KolmogorovTest
*/
   Double_t KolmogorovSmirnov2SamplesTest(const Char_t* option = "p") const; // Returns default p-value; option "t" returns the test statistic value "Dn"

/*
  The Kolmogorov-Smirnov 1-Sample Test algorithm for a specific distribution is described at
  http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/kstest.htm
  and described and taken from (4)
  Press W. H., Teukolsky S.A., Vetterling W.T., Flannery B.P. (2007), Numerical Recipes - The Art of Scientific Computing (Third Edition), Cambridge Univerdity Press
*/
   Double_t KolmogorovSmirnovTest(const Char_t* option = "p") const; // Returns default p-value; option "t" returns the test statistic value "Dn"
  
   // The class's unary function
   Double_t operator()(ETestType test = kAD, const Char_t* option = "p") const; // The class's unary function: returns default Anderson Darling 1-Sample Test and default p-value; option "t" returns the test statistic value specific to the test type

private:
  
   GoFTest();                       // Disallowed default constructor
   GoFTest(GoFTest& gof);           // Disallowed copy constructor
   GoFTest operator=(GoFTest& gof); // Disallowed assign operator
  
   typedef ROOT::Math::IBaseFunctionOneDim* CDF_Ptr;
   CDF_Ptr fCDF;
  
   class Integrand { // Integrand of the integral term of the Anderson-Darling test statistic's asymtotic distribution as described in (2)
      Double_t* parms;
   public:
      Integrand(Double_t* parms);
      Double_t operator()(Double_t y) const;
   };
  
   EDistribution fDist;
  
   Double_t fMean;
   Double_t fSigma;
  
   std::vector<Double_t> fCombinedSamples;
  
   std::vector<std::vector<Double_t> > fSamples;
  
   Bool_t fTestSampleFromH0;
  
   void SetCDF(CDF_Ptr cdf = 0);
  
   void Instantiate(const Double_t* sample, UInt_t sampleSize)
#if !defined(__CINT__) && !defined(__MAKECINT__)
      throw(BadSampleArgument, std::bad_exception) 
#endif
      ; 
   Double_t ComputeIntegral(Double_t* parms) const; // Computation of the integral term of the 1-Sample Anderson-Darling test statistic's asymtotic distribution as described in (2)
  
   Double_t GaussianCDF(Double_t x) const;
   Double_t ExponentialCDF(Double_t x) const;
  
   Double_t GetSigmaN(UInt_t N) const; // Computation of sigma_N as described in (1) 
  
   Double_t InterpolatePValues(Double_t dA2, Int_t bin) const; // Linear interpolation used in GoFTest::PValueAD2Samples
  
   Double_t PValueAD2Samples(Double_t& A2, UInt_t N) const; // Computation of the 2-Sample Anderson-Darling Test's p-value as described in (1)
  
   Double_t PValueAD1Sample(Double_t A2) const; // Computation of the 1-Sample Anderson-Darling Test's p-value 
    
   void LogSample(); // Applies the logarithm to the sample when the specified distribution to test is LogNormal
    
   void SetSamples(std::vector<const Double_t*> samples, const std::vector<UInt_t> samplesSizes);
  
   void SetParameters(); // Sets the estimated mean and standard-deviation from the samples 
}; // end GoFTest class

} // ROOT namespace
} // Math namespace
#endif
