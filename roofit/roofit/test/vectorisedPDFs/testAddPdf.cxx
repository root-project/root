// Author: Stephan Hageboeck, CERN  26 Apr 2019

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

#include "RooAddPdf.h"
#include "RooGaussian.h"
#include "RooPoisson.h"
#include "RooExponential.h"

class TestGaussPlusPoisson : public PDFTest {
protected:
   TestGaussPlusPoisson() : PDFTest("Gauss + Poisson")
   {
      // Declare variables x,mean,sigma with associated name, title, initial value and allowed range
      auto x = std::make_unique<RooRealVar>("x", "x", -1.5, 40.5);
      x->setBins(42); // Prettier plots for Poisson

      auto mean = std::make_unique<RooRealVar>("mean", "mean of gaussian", 20., -10, 30);
      auto sigma = std::make_unique<RooRealVar>("sigma", "width of gaussian", 4., 0.5, 10);

      // Build gaussian p.d.f in terms of x,mean and sigma
      auto gauss = std::make_unique<RooGaussian>("gauss", "gaussian PDF", *x, *mean, *sigma);

      auto meanPois = std::make_unique<RooRealVar>("meanPois", "Mean of Poisson", 10.3, 0, 30);
      auto pois = std::make_unique<RooPoisson>("Pois", "Poisson PDF", *x, *meanPois, true);

      auto fractionGaus = std::make_unique<RooRealVar>("fractionGaus", "Fraction of Gauss component", 0.5, 0., 1.);
      auto sumGausPois =
         std::make_unique<RooAddPdf>("SumGausPois", "Sum of Gaus and Poisson", RooArgSet(*gauss, *pois), *fractionGaus);
      sumGausPois->fixCoefNormalization(*x);
      _pdf = std::move(sumGausPois);

      _variables.addOwned(std::move(x));

      //      _variablesToPlot.add(x);

      _parameters.addOwned(std::move(mean));
      _parameters.addOwned(std::move(sigma));
      _parameters.addOwned(std::move(meanPois));
      _parameters.addOwned(std::move(fractionGaus));

      _otherObjects.addOwned(std::move(gauss));
      _otherObjects.addOwned(std::move(pois));

      // Gauss is slightly less accurate
      _toleranceCompareBatches = 2.E-13;
      _toleranceParameter = 5.E-5;
      _toleranceCorrelation = 5.E-4;
   }
};

COMPARE_FIXED_VALUES_UNNORM(TestGaussPlusPoisson, CompareFixedValuesUnnorm)
COMPARE_FIXED_VALUES_NORM(TestGaussPlusPoisson, CompareFixedValuesNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestGaussPlusPoisson, CompareFixedValuesNormLog)

FIT_TEST_SCALAR(TestGaussPlusPoisson, DISABLED_Scalar) // Save time
FIT_TEST_BATCH(TestGaussPlusPoisson, DISABLED_Batch)   // Save time
FIT_TEST_BATCH_VS_SCALAR(TestGaussPlusPoisson, CompareBatchScalar)

class TestGaussPlusGaussPlusExp : public PDFTest {
protected:
   TestGaussPlusGaussPlusExp() : PDFTest("Gauss + Gauss + Exp")
   {
      auto x = std::make_unique<RooRealVar>("x", "x", 0., 100.);

      auto c = std::make_unique<RooRealVar>("c", "c", -0.05, -100., -0.005);
      auto expo = std::make_unique<RooExponential>("expo", "expo", *x, *c);

      auto mean = std::make_unique<RooRealVar>("mean1", "mean of gaussian", 30., -10, 100);
      auto sigma = std::make_unique<RooRealVar>("sigma1", "width of gaussian", 4., 0.1, 20);
      auto gauss = std::make_unique<RooGaussian>("gauss1", "gaussian PDF", *x, *mean, *sigma);

      auto mean2 = std::make_unique<RooRealVar>("mean2", "mean of gaussian", 60., 50, 100);
      auto sigma2 = std::make_unique<RooRealVar>("sigma2", "width of gaussian", 10., 0.1, 20);
      auto gauss2 = std::make_unique<RooGaussian>("gauss2", "gaussian PDF", *x, *mean2, *sigma2);

      auto nGauss = std::make_unique<RooRealVar>("nGauss", "Fraction of Gauss component", 800., 0., 1.E6);
      auto nGauss2 = std::make_unique<RooRealVar>("nGauss2", "Fraction of Gauss component", 600., 0., 1.E6);
      auto nExp = std::make_unique<RooRealVar>("nExp", "Number of events in exp", 1000, 0, 1.E6);
      auto sum2GausExp =
         std::make_unique<RooAddPdf>("Sum2GausExp", "Sum of Gaus and Exponentials", RooArgSet(*gauss, *gauss2, *expo),
                                     RooArgSet(*nGauss, *nGauss2, *nExp));
      sum2GausExp->fixCoefNormalization(*x);
      _pdf = std::move(sum2GausExp);

      _variables.addOwned(std::move(x));

      _parameters.addOwned(std::move(c));
      _parameters.addOwned(std::move(mean));
      _parameters.addOwned(std::move(sigma));
      _parameters.addOwned(std::move(mean2));
      _parameters.addOwned(std::move(sigma2));

      _yields.addOwned(std::move(nGauss));
      _yields.addOwned(std::move(nGauss2));
      _yields.addOwned(std::move(nExp));

      _otherObjects.addOwned(std::move(expo));
      _otherObjects.addOwned(std::move(gauss));
      _otherObjects.addOwned(std::move(gauss2));

      _toleranceCompareLogs = 4.3E-14;

      // VDT stops computing exponentials below exp(-708) = 3.3075530e-308
      // Since this test runs Gaussians far from their mean, we need to be a bit more forgiving
      _toleranceParameter = 5.E-5;
      _toleranceCorrelation = 5.E-4;
   }
};

COMPARE_FIXED_VALUES_UNNORM(TestGaussPlusGaussPlusExp, CompareFixedValuesUnnorm)
COMPARE_FIXED_VALUES_NORM(TestGaussPlusGaussPlusExp, CompareFixedValuesNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestGaussPlusGaussPlusExp, CompareFixedValuesNormLog)

FIT_TEST_SCALAR(TestGaussPlusGaussPlusExp, DISABLED_Scalar) // Save time
FIT_TEST_BATCH(TestGaussPlusGaussPlusExp, DISABLED_Batch)   // Save time
FIT_TEST_BATCH_VS_SCALAR(TestGaussPlusGaussPlusExp, CompareBatchScalar)

#if !defined(_MSC_VER) // RooFit multiprocessing doesn't work on Windows

class TestGaussPlusGaussPlusExp_MP : public TestGaussPlusGaussPlusExp {
public:
   TestGaussPlusGaussPlusExp_MP() : TestGaussPlusGaussPlusExp() { _multiProcess = 2; }
};

COMPARE_FIXED_VALUES_UNNORM(TestGaussPlusGaussPlusExp_MP, CompareFixedValuesUnnorm)
COMPARE_FIXED_VALUES_NORM(TestGaussPlusGaussPlusExp_MP, CompareFixedValuesNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestGaussPlusGaussPlusExp_MP, CompareFixedValuesNormLog)

FIT_TEST_SCALAR(TestGaussPlusGaussPlusExp_MP, DISABLED_Scalar) // Save time
FIT_TEST_BATCH(TestGaussPlusGaussPlusExp_MP, DISABLED_Batch)   // Save time
FIT_TEST_BATCH_VS_SCALAR(TestGaussPlusGaussPlusExp_MP, CompareBatchScalar)

#endif // !defined(_MSC_VER)
