// Author: Stephan Hageboeck, CERN  01 Jul 2019

/*****************************************************************************
 * RooFit
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2019, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#include "VectorisedPDFTests.h"

#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "RooPolynomial.h"
#include "TMath.h"
#include "RooRandom.h"
#include "RooFormulaVar.h"
#include <math.h>

using namespace std;

class TestRooPolynomial : public PDFTest {
protected:
   TestRooPolynomial() : PDFTest("Polynomial(...)")
   {
      auto x = std::make_unique<RooRealVar>("x", "x", 0, 10);
      auto a1 = std::make_unique<RooRealVar>("a1", "First coefficient", 5, 0, 10);
      auto a2 = std::make_unique<RooRealVar>("a2", "Second coefficient", 1, 0, 10);
      auto a3 = std::make_unique<RooFormulaVar>("a3", "Third coefficient", "a1+a2", RooArgList(*a1, *a2));

      _pdf = std::make_unique<RooPolynomial>("pol", "Polynomial", *x, RooArgList(*a1, *a2, *a3));

      _variables.addOwned(std::move(x));
      _variables.addOwned(std::move(a1));
      //        _variablesToPlot.add(var);

      _parameters.addOwned(std::move(a2));

      _otherObjects.addOwned(std::move(a3));
   }
};

COMPARE_FIXED_VALUES_UNNORM(TestRooPolynomial, CompareFixedUnnorm)
COMPARE_FIXED_VALUES_NORM(TestRooPolynomial, CompareFixedNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestRooPolynomial, CompareFixedNormLog)

FIT_TEST_SCALAR(TestRooPolynomial, RunScalar)
FIT_TEST_BATCH(TestRooPolynomial, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestRooPolynomial, CompareBatchScalar)

class RooNonVecGaussian : public RooAbsPdf {
public:
   RooNonVecGaussian() {};
   RooNonVecGaussian(const char *name, const char *title, RooAbsReal &_x, RooAbsReal &_mean, RooAbsReal &_sigma)
      : RooAbsPdf(name, title),
        x("x", "Observable", this, _x),
        mean("mean", "Mean", this, _mean),
        sigma("sigma", "Width", this, _sigma)
   {
   }
   virtual TObject *clone(const char *newname) const override { return new RooNonVecGaussian(*this, newname); }
   RooNonVecGaussian(const RooNonVecGaussian &other, const char *name)
      : RooAbsPdf(other, name), x("x", this, other.x), mean("mean", this, other.mean), sigma("sigma", this, other.sigma)
   {
   }

   virtual ~RooNonVecGaussian() = default;

   Int_t getAnalyticalIntegral(RooArgSet &allVars, RooArgSet &analVars, const char *) const override
   {
      if (matchArgs(allVars, analVars, x))
         return 1;
      if (matchArgs(allVars, analVars, mean))
         return 2;
      return 0;
   }

   Double_t analyticalIntegral(Int_t code, const char *rangeName = 0) const override
   {
      assert(code == 1 || code == 2);

      // The normalisation constant 1./sqrt(2*pi*sigma^2) is left out in evaluate().
      // Therefore, the integral is scaled up by that amount to make RooFit normalise
      // correctly.
      const double resultScale = sqrt(TMath::TwoPi()) * sigma;

      // Here everything is scaled and shifted into a standard normal distribution:
      const double xscale = TMath::Sqrt2() * sigma;
      double max = 0.;
      double min = 0.;
      if (code == 1) {
         max = (x.max(rangeName) - mean) / xscale;
         min = (x.min(rangeName) - mean) / xscale;
      } else { // No == 2 test because of assert
         max = (mean.max(rangeName) - x) / xscale;
         min = (mean.min(rangeName) - x) / xscale;
      }

      // Here we go for maximum precision: We compute all integrals in the UPPER
      // tail of the Gaussian, because erfc has the highest precision there.
      // Therefore, the different cases for range limits in the negative hemisphere are mapped onto
      // the equivalent points in the upper hemisphere using erfc(-x) = 2. - erfc(x)
      const double ecmin = std::erfc(std::abs(min));
      const double ecmax = std::erfc(std::abs(max));

      const double result = resultScale * 0.5 *
                            (min * max < 0.0 ? 2.0 - (ecmin + ecmax)
                             : max <= 0.     ? ecmax - ecmin
                                             : ecmin - ecmax);

      return result != 0. ? result : 1.E-300;
   }

   Int_t getGenerator(const RooArgSet &directVars, RooArgSet &generateVars, Bool_t) const override
   {
      if (matchArgs(directVars, generateVars, x))
         return 1;
      if (matchArgs(directVars, generateVars, mean))
         return 2;
      return 0;
   }

   void generateEvent(Int_t code) override
   {
      assert(code == 1 || code == 2);
      Double_t xgen;
      if (code == 1) {
         while (1) {
            xgen = RooRandom::randomGenerator()->Gaus(mean, sigma);
            if (xgen < x.max() && xgen > x.min()) {
               x = xgen;
               break;
            }
         }
      } else if (code == 2) {
         while (1) {
            xgen = RooRandom::randomGenerator()->Gaus(x, sigma);
            if (xgen < mean.max() && xgen > mean.min()) {
               mean = xgen;
               break;
            }
         }
      } else {
         std::cout << "error in RooNonVecGaussian generateEvent" << std::endl;
      }

      return;
   }

protected:
   RooRealProxy x;
   RooRealProxy mean;
   RooRealProxy sigma;

   Double_t evaluate() const override
   {
      const double arg = x - mean;
      const double sig = sigma;
      return exp(-0.5 * arg * arg / (sig * sig));
   }
};

class TestNonVecGauss : public PDFTest {
protected:
   TestNonVecGauss() : PDFTest("GaussNoBatches")
   {
      auto x = std::make_unique<RooRealVar>("x", "x", -10, 10);
      auto mean = std::make_unique<RooRealVar>("mean", "mean of gaussian", 1, -10, 10);
      auto sigma = std::make_unique<RooRealVar>("sigma", "width of gaussian", 1, 0.1, 10);

      // Build gaussian p.d.f in terms of x,mean and sigma
      _pdf = std::make_unique<RooNonVecGaussian>("gauss", "gaussian PDF", *x, *mean, *sigma);

      _variables.addOwned(std::move(x));

      _parameters.addOwned(std::move(mean));
      _parameters.addOwned(std::move(sigma));

      _toleranceCompareLogs = 2.5E-14;
   }
};

COMPARE_FIXED_VALUES_UNNORM(TestNonVecGauss, CompareFixedUnnorm)
COMPARE_FIXED_VALUES_NORM(TestNonVecGauss, CompareFixedNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestNonVecGauss, CompareFixedNormLog)

FIT_TEST_SCALAR(TestNonVecGauss, RunScalar)
FIT_TEST_BATCH(TestNonVecGauss, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestNonVecGauss, CompareBatchScalar)

class TestNonVecGaussWeighted : public PDFTestWeightedData {
protected:
   TestNonVecGaussWeighted() : PDFTestWeightedData("GaussNoBatchesWithWeights", 50000)
   {
      // Declare variables x,mean,sigma with associated name, title, initial value and allowed range
      auto x = std::make_unique<RooRealVar>("x", "x", -10, 10);
      auto mean = std::make_unique<RooRealVar>("mean", "mean of gaussian", 1, -10, 10);
      auto sigma = std::make_unique<RooRealVar>("sigma", "width of gaussian", 2.3, 0.1, 10);

      // Build gaussian p.d.f in terms of x,mean and sigma
      _pdf = std::make_unique<RooNonVecGaussian>("gauss", "gaussian PDF", *x, *mean, *sigma);

      _variables.addOwned(std::move(x));

      _parameters.addOwned(std::move(mean));
      _parameters.addOwned(std::move(sigma));

      _toleranceCompareLogs = 2.5E-14;
   }
};

COMPARE_FIXED_VALUES_UNNORM(TestNonVecGaussWeighted, CompareFixedUnnorm)
COMPARE_FIXED_VALUES_NORM(TestNonVecGaussWeighted, CompareFixedNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestNonVecGaussWeighted, CompareFixedNormLog)

FIT_TEST_SCALAR(TestNonVecGaussWeighted,
                DISABLED_RunScalar) // Would need SumW2 error matrix correction, but no done in macro
FIT_TEST_BATCH(TestNonVecGaussWeighted, DISABLED_RunBatch) // As above
FIT_TEST_BATCH_VS_SCALAR(TestNonVecGaussWeighted, CompareBatchScalar)

class TestNonVecGaussInMeanAndX : public PDFTest {
protected:
   TestNonVecGaussInMeanAndX() : PDFTest("GaussNoBatches(x, mean)")
   {
      // Declare variables x,mean,sigma with associated name, title, initial value and allowed range
      auto x = std::make_unique<RooRealVar>("x", "x", -10, 10);
      auto mean = std::make_unique<RooRealVar>("mean", "mean of gaussian", 1, -10, 10);
      auto sigma = std::make_unique<RooRealVar>("sigma", "width of gaussian", 1, 0.1, 10);

      // Build gaussian p.d.f in terms of x,mean and sigma
      _pdf = std::make_unique<RooNonVecGaussian>("gauss", "gaussian PDF", *x, *mean, *sigma);

      _variables.addOwned(std::move(x));
      _variables.addOwned(std::move(mean));
      //        _variablesToPlot.add(var);

      _parameters.addOwned(std::move(sigma));
   }
};

COMPARE_FIXED_VALUES_UNNORM(TestNonVecGaussInMeanAndX, CompareFixedUnnorm)
COMPARE_FIXED_VALUES_NORM(TestNonVecGaussInMeanAndX, CompareFixedNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestNonVecGaussInMeanAndX, CompareFixedNormLog)

FIT_TEST_SCALAR(TestNonVecGaussInMeanAndX, RunScalar)
FIT_TEST_BATCH(TestNonVecGaussInMeanAndX, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestNonVecGaussInMeanAndX, CompareBatchScalar)
