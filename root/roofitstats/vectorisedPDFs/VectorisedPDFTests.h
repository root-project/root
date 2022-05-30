// Author: Stephan Hageboeck, CERN  26 Jul 2019

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

#include "RooArgSet.h"
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooFitResult.h"

#include "gtest/gtest.h"

#include <memory>

class RooAbsPdf;

class PDFTest : public ::testing::Test
{
  protected:
    PDFTest(std::string&& name, std::size_t nEvt = 100000);

    void SetUp() override;

    virtual void makeFitData();

    virtual void makeUniformData();

    void randomiseParameters(ULong_t seed);

    void makePlots(std::string&& fitStage) const;

    void setValuesConstant(const RooAbsCollection& coll, bool constant) const;

    void resetParameters();

    void kickParameters();

    void compareFixedValues(double& maximalError, bool normalise, bool compareLogs, bool runTimer = false, unsigned int nChunks = 1);

    void checkParameters();

    void runBatchVsScalar(bool clonePDF = false);

    std::unique_ptr<RooFitResult> runBatchFit(RooAbsPdf* pdf);

    std::unique_ptr<RooFitResult> runScalarFit(RooAbsPdf* pdf);


    std::unique_ptr<RooAbsPdf> _pdf;
    std::unique_ptr<RooDataSet> _dataUniform;
    std::unique_ptr<RooDataSet> _dataFit;

    std::string _name;
    std::string _plotDirectory{"/tmp/"};
    RooArgSet _variables;
    RooArgSet _variablesToPlot;
    RooArgSet _parameters;
    RooArgSet _yields;
    RooArgSet _origYields;
    RooArgSet _origParameters;
    RooArgSet _otherObjects;
    const std::size_t _nEvents;
    double _toleranceParameter{1.E-6};
    double _toleranceCorrelation{1.E-4};
    double _toleranceCompareBatches{5.E-14};
    double _toleranceCompareLogs{2.E-14};
    int _printLevel{-1};
    unsigned int _multiProcess{0};
};


class PDFTestWeightedData : public PDFTest {
  protected:
    PDFTestWeightedData(const char* name, std::size_t events = 100000) :
      PDFTest(name, events) { }

    void makeFitData() override;
};

/// Test batch against scalar code for fixed values of observable. Don't run normalisation.
#define COMPARE_FIXED_VALUES_UNNORM(TEST_CLASS, TEST_NAME) \
    TEST_F(TEST_CLASS, DISABLED_##TEST_NAME) {\
  resetParameters();\
  double relativeError, maximalRelativeError=0.0;\
  compareFixedValues(relativeError, false, false, false);\
  maximalRelativeError = std::max(maximalRelativeError,relativeError);\
  \
  for (unsigned int i=0; i<5 && !HasFailure(); ++i) {\
    std::stringstream str;\
    str << "Parameter set " << i;\
    for (auto par : _parameters) {\
      auto p = static_cast<RooAbsReal*>(par);\
      str << "\n\t" << p->GetName() << "\t" << p->getVal();\
    }\
    SCOPED_TRACE(str.str());\
    compareFixedValues(relativeError, false, false, false, _multiProcess);\
    maximalRelativeError = std::max(maximalRelativeError,relativeError);\
    \
    randomiseParameters(1337+i);\
  }\
  std::cout << "\nMaximal relative error (scalar vs batch) is: " << maximalRelativeError << "\n\n";\
}

/// Test batch against scalar code for fixed values of observable with normalisation.
#define COMPARE_FIXED_VALUES_NORM(TEST_CLASS, TEST_NAME)\
    TEST_F(TEST_CLASS, TEST_NAME) {\
  resetParameters();\
  double relativeError, maximalRelativeError=0.0;\
  \
  for (unsigned int i=0; i<5 && !HasFailure(); ++i) {\
    std::stringstream str;\
    str << "Parameter set " << i;\
    for (auto par : _parameters) {\
      auto p = static_cast<RooAbsReal*>(par);\
      str << "\n\t" << p->GetName() << "\t" << p->getVal();\
    }\
    SCOPED_TRACE(str.str());\
    compareFixedValues(relativeError, true, false, false, _multiProcess);\
    maximalRelativeError = std::max(maximalRelativeError,relativeError);\
    \
    randomiseParameters(1337+i);\
  }\
  std::cout << "\nMaximal relative error (scalar vs batch) is: " << maximalRelativeError << "\n\n";\
}

/// Test batch against scalar code for fixed values of observable. Compute log probabilities.
#define COMPARE_FIXED_VALUES_NORM_LOG(TEST_CLASS, TEST_NAME) \
    TEST_F(TEST_CLASS, DISABLED_##TEST_NAME) {\
  resetParameters();\
  double relativeError, maximalRelativeError=0.0;\
  \
  for (unsigned int i=0; i<5 && !HasFailure(); ++i) {\
    std::stringstream str;\
    str << "Parameter set " << i;\
    for (auto par : _parameters) {\
      auto p = static_cast<RooAbsReal*>(par);\
      str << "\n\t" << p->GetName() << "\t" << p->getVal();\
    }\
    SCOPED_TRACE(str.str());\
    compareFixedValues(relativeError, true, true, false, _multiProcess);\
    maximalRelativeError = std::max(maximalRelativeError,relativeError);\
    \
    randomiseParameters(1337+i);\
  }\
  std::cout << "\nMaximal relative error (scalar vs batch) is: " << maximalRelativeError << "\n\n";\
}

/// Run a fit for batch and scalar code and compare results.
#define FIT_TEST_BATCH_VS_SCALAR(TEST_CLASS, TEST_NAME) \
    TEST_F(TEST_CLASS, TEST_NAME) {\
  runBatchVsScalar();\
}

/// Run a fit for batch and scalar code and compare results.
/// Clone the PDFs before running the tests. This can run the test even if some internal state
/// is propagated / saved wrongly.
#define FIT_TEST_BATCH_VS_SCALAR_CLONE_PDF(TEST_CLASS, TEST_NAME) \
    TEST_F(TEST_CLASS, TEST_NAME) {\
  runBatchVsScalar(true);\
}

/// Run a fit in batch mode and compare results to pre-fit values.
#define FIT_TEST_BATCH(TEST_CLASS, TEST_NAME) \
    TEST_F(TEST_CLASS, TEST_NAME) {\
  auto result = runBatchFit(_pdf.get());\
  ASSERT_NE(result, nullptr);\
  checkParameters();\
}

/// Run a fit in legacy mode and compare results to pre-fit values.
#define FIT_TEST_SCALAR(TEST_CLASS, TEST_NAME) \
    TEST_F(TEST_CLASS, TEST_NAME) {\
  auto result = runScalarFit(_pdf.get());\
  ASSERT_NE(result, nullptr);\
  checkParameters();\
}

