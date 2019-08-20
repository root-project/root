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
#include "RooChiSquarePdf.h"

class TestChiSquarePdfinX: public PDFTest
{
  protected:
    TestChiSquarePdfinX() :
      PDFTest("ChiSquarePdf", 300000)
    {
      auto x = new RooRealVar("x", "x", 0.1, 100);
      auto ndof = new RooRealVar("ndof", "ndof of chiSquarePdf", 2, 1, 5);
      
      // Build chiSquarePdf p.d.f 
      _pdf = std::make_unique<RooChiSquarePdf>("chiSquarePdf", "chiSquarePdf PDF", *x, *ndof);
      
      for (auto var : {x}) {
        _variables.addOwned(*var);
      }
      
      //      _variablesToPlot.add(x);
      
      for (auto par : {ndof}) {
        _parameters.addOwned(*par);
      }
      _toleranceCompareLogs = 5e-14;
    }
};

COMPARE_FIXED_VALUES_UNNORM(TestChiSquarePdfinX, CompareFixedValuesUnnorm)
COMPARE_FIXED_VALUES_NORM(TestChiSquarePdfinX, CompareFixedValuesNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestChiSquarePdfinX, CompareFixedNormLog)
FIT_TEST_SCALAR(TestChiSquarePdfinX, DISABLED_RunScalar)
FIT_TEST_BATCH(TestChiSquarePdfinX, DISABLED_RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestChiSquarePdfinX, DISABLED_CompareBatchScalar)

class TestChiSquarePdfinNdof: public PDFTest
{
  protected:
    TestChiSquarePdfinNdof() :
      PDFTest("ChiSquarePdf", 300000)
    {
      // Declare variables x,mean,sigma with associated name, title, initial value and allowed range
      auto x = new RooRealVar("x", "x", 0.1, 100);
      auto ndof = new RooRealVar("ndof", "ndof of chiSquarePdf", 2, 1, 5);
      
      // Build chiSquarePdf p.d.f 
      _pdf = std::make_unique<RooChiSquarePdf>("chiSquarePdf", "chiSquarePdf PDF", *x, *ndof);
      
      for (auto var : {ndof}) {
        _variables.addOwned(*var);
      }
      
      _variablesToPlot.add(*x);
      
      for (auto par : {x}) {
        _parameters.addOwned(*par);
      }

      _toleranceCompareLogs = 5e-14;
    }
};

COMPARE_FIXED_VALUES_UNNORM(TestChiSquarePdfinNdof, CompareFixedValuesUnnorm)
COMPARE_FIXED_VALUES_NORM(TestChiSquarePdfinNdof, CompareFixedValuesNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestChiSquarePdfinNdof, CompareFixedNormLog)
FIT_TEST_SCALAR(TestChiSquarePdfinNdof, DISABLED_RunScalar)
FIT_TEST_BATCH(TestChiSquarePdfinNdof, DISABLED_RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestChiSquarePdfinNdof, DISABLED_CompareBatchScalar)

class TestChiSquarePdfinXandNdof: public PDFTest
{
  protected:
    TestChiSquarePdfinXandNdof() :
      PDFTest("ChiSquarePdf", 300000)
    {
      // Declare variables x,mean,sigma with associated name, title, initial value and allowed range
      auto x = new RooRealVar("x", "x", 0.1, 100);
      auto ndof = new RooRealVar("ndof", "ndof of chiSquarePdf", 2, 1, 5);
      
      // Build chiSquarePdf p.d.f 
      _pdf = std::make_unique<RooChiSquarePdf>("chiSquarePdf", "chiSquarePdf PDF", *x, *ndof);
      
      for (auto var : {x, ndof}) {
        _variables.addOwned(*var);
      }
      
      //      _variablesToPlot.add(x);
      _toleranceCompareLogs = 5e-14;
    }
};

COMPARE_FIXED_VALUES_UNNORM(TestChiSquarePdfinXandNdof, CompareFixedValuesUnnorm)
COMPARE_FIXED_VALUES_NORM(TestChiSquarePdfinXandNdof, CompareFixedValuesNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestChiSquarePdfinXandNdof, CompareFixedNormLog)
FIT_TEST_SCALAR(TestChiSquarePdfinXandNdof, DISABLED_RunScalar)
FIT_TEST_BATCH(TestChiSquarePdfinXandNdof, DISABLED_RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestChiSquarePdfinXandNdof, DISABLED_CompareBatchScalar)
