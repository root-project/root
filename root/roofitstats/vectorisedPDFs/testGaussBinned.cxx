// Author: Stephan Hageboeck, CERN  21 Jan 2020

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

#include "RooRealVar.h"
#include "RooDataHist.h"
#include "RooGaussian.h"
#include "RooHelpers.h"
#include "RooRandom.h"

#include "gtest/gtest.h"

#include <chrono>

class MyTimer {
  public:
    MyTimer(std::string&& name)
  : m_name(name), m_startTime(clock()), m_endTime(0),
    m_steadyStart(std::chrono::steady_clock::now()), m_steadyEnd() { }

    clock_t diffTime() const {
      return clock() - m_startTime;
    }

    void interval() {
      m_endTime = clock();
      m_steadyEnd = std::chrono::steady_clock::now();
    }

    void print(std::ostream& str) {
      clock_t diff = m_endTime - m_startTime;
      std::chrono::duration<double> diffSteady = m_steadyEnd - m_steadyStart;
      str << "\n" << "Timer '" << m_name << "':\t" << double(diff)/CLOCKS_PER_SEC << "s (CPU) "
          << diffSteady.count() << "s (wall)" << std::endl;
    }

  private:
    std::string m_name;
    clock_t m_startTime;
    clock_t m_endTime;
    std::chrono::time_point<std::chrono::steady_clock> m_steadyStart;
    std::chrono::time_point<std::chrono::steady_clock> m_steadyEnd;
};

std::ostream& operator<<(std::ostream& str, MyTimer& timer) {
  timer.interval();
  timer.print(str);
  return str;
}

class GaussBinnedFit : public testing::TestWithParam<bool> {
public:
  void SetUp() override {
    RooRandom::randomGenerator()->SetSeed(10);
  }

  RooRealVar x{"x", "x", 0, -10, 10};
  RooRealVar m{"m", "m", 1., -10., 10};
  RooRealVar s{"s", "s", 1.5, 0.01, 10};

  RooGaussian gaus{"gaus", "gaus", x, m, s};
};

TEST_P(GaussBinnedFit, BatchFit) {
  x.setBins(50);
  std::unique_ptr<RooDataHist> dataHist( gaus.generateBinned(x, 10000) );

  const bool batchMode = GetParam();
  m.setVal(-1.);
  s.setVal(3.);
  MyTimer timer(batchMode ? "BatchBinned" : "ScalarBinned");
  gaus.fitTo(*dataHist, RooFit::BatchMode(batchMode), RooFit::PrintLevel(-1), RooFit::Optimize(0));
  timer.interval();
  std::cout << timer << std::endl;
  EXPECT_NEAR(m.getVal(), 1., m.getError());
  EXPECT_NEAR(s.getVal(), 1.5, s.getError());
}

/// Test binned fit with a lot of bins. Because of ROOT-3874, it unfortunately
/// has a biased sigma parameter.
TEST_P(GaussBinnedFit, BatchFitFineBinsBiased) {
  x.setBins(1000);
  s.setVal(4.);
  std::unique_ptr<RooDataHist> dataHist( gaus.generateBinned(x, 20000) );

  const bool batchMode = GetParam();
  m.setVal(-1.);
  s.setVal(3.);
  MyTimer timer(batchMode ? "BatchFineBinned" : "ScalarFineBinned");
  gaus.fitTo(*dataHist, RooFit::BatchMode(batchMode), RooFit::PrintLevel(-1));
  timer.interval();
  std::cout << timer << std::endl;
  EXPECT_NEAR(m.getVal(), 1., m.getError());
  EXPECT_NEAR(s.getVal(), 3.95, s.getError())
      << "It is known that binned fits with strong curvatures are biased.\n"
      << "If this fails, the bias was fixed. Enable the test below, and delete this one.";
}


/// Enable instead of BatchFitFineBinsBiased once ROOT-3874 is fixed.
TEST_P(GaussBinnedFit, DISABLED_BatchFitFineBins) {
  x.setBins(1000);
  s.setVal(4.);
  std::unique_ptr<RooDataHist> dataHist( gaus.generateBinned(x, 20000) );

  const bool batchMode = GetParam();
  m.setVal(-1.);
  s.setVal(3.);
  MyTimer timer(batchMode ? "BatchFineBinned" : "ScalarFineBinned");
  gaus.fitTo(*dataHist, RooFit::BatchMode(batchMode), RooFit::PrintLevel(-1));
  timer.interval();
  std::cout << timer << std::endl;
  EXPECT_NEAR(m.getVal(), 1., m.getError());
  EXPECT_NEAR(s.getVal(), 4., s.getError());
}

INSTANTIATE_TEST_SUITE_P(RunFits,
    GaussBinnedFit,
    testing::Bool());

// TODO Test a batch fit that uses categories once categories can be passed through the batch interface.

