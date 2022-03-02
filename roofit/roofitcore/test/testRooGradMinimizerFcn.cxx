/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   PB, Patrick Bos,     NL eScience Center, p.bos@esciencecenter.nl        *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#include <RooWorkspace.h>
#include <RooDataSet.h>
#include <RooRealVar.h>
#include <RooAbsPdf.h>
#include <RooMinimizer.h>
#include <RooFitResult.h>
#include <RooAddPdf.h>
#include <RooRandom.h>
#include "TFile.h" // for loading the workspace file
#include <stdio.h> // remove redundant workspace files

#include "gtest/gtest.h"
#include "test_lib.h"

#include <RooMsgService.h>
#include <RooGlobalFunc.h> // RooFit::ERROR

class GradMinimizerParSeed : public testing::TestWithParam<unsigned long> {};

TEST_P(GradMinimizerParSeed, Gaussian1D)
{
   RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);

   // produce the same random stuff every time
   RooRandom::randomGenerator()->SetSeed(GetParam());

   RooWorkspace w = RooWorkspace();

   std::unique_ptr<RooAbsReal> nll;
   std::unique_ptr<RooArgSet> values;
   RooAbsPdf *pdf;
   RooDataSet *data;
   std::tie(nll, pdf, data, values) = generate_1D_gaussian_pdf_nll(w, 10000);
   // when c++17 support arrives, change to this:
   //  auto [nll, values] = generate_1D_gaussian_pdf_nll(w, 10000);
   RooRealVar *mu = w.var("mu");

   RooArgSet *savedValues = dynamic_cast<RooArgSet *>(values->snapshot());
   if (savedValues == nullptr) {
      throw std::runtime_error("params->snapshot() cannot be casted to RooArgSet!");
   }

   // --------

   RooMinimizer m0(*nll);
   m0.setMinimizerType("Minuit2");

   m0.setStrategy(0);
   m0.setPrintLevel(-1);

   m0.migrad();

   RooFitResult *m0result = m0.lastMinuitFit();
   double minNll0 = m0result->minNll();
   double edm0 = m0result->edm();
   double mu0 = mu->getVal();
   double muerr0 = mu->getError();

   *values = *savedValues;

   RooMinimizer m1(*nll, RooMinimizer::FcnMode::gradient);
   m1.setMinimizerType("Minuit2");

   m1.setStrategy(0);
   m1.setPrintLevel(-1);

   m1.migrad();

   RooFitResult *m1result = m1.lastMinuitFit();
   double minNll1 = m1result->minNll();
   double edm1 = m1result->edm();
   double mu1 = mu->getVal();
   double muerr1 = mu->getError();

   EXPECT_EQ(minNll0, minNll1);
   EXPECT_EQ(mu0, mu1);
   EXPECT_EQ(muerr0, muerr1);
   EXPECT_EQ(edm0, edm1);
}

INSTANTIATE_TEST_SUITE_P(Seeds,
                         GradMinimizerParSeed,
                         testing::Range<unsigned long>(1, 11));

TEST(GradMinimizerDebugging, DISABLED_Gaussian1DNominal)
{
   // produce the same random stuff every time
   RooRandom::randomGenerator()->SetSeed(1);

   RooWorkspace w = RooWorkspace();

   std::unique_ptr<RooAbsReal> nll;
   std::unique_ptr<RooArgSet> _;
   RooAbsPdf *pdf;
   RooDataSet *data;
   std::tie(nll, pdf, data, _) = generate_1D_gaussian_pdf_nll(w, 10000);
   // when c++17 support arrives, change to this:
   //  auto [nll, _] = generate_1D_gaussian_pdf_nll(w, 10000);

   RooMinimizer m0(*nll);
   m0.setMinimizerType("Minuit2");

   m0.setStrategy(0);
   m0.setPrintLevel(100);
   m0.setVerbose(kTRUE);

   m0.migrad();
}

TEST(GradMinimizerDebugging, DISABLED_Gaussian1DGradMinimizer)
{
   // produce the same random stuff every time
   RooRandom::randomGenerator()->SetSeed(1);

   RooWorkspace w = RooWorkspace();

   std::unique_ptr<RooAbsReal> nll;
   std::unique_ptr<RooArgSet> _;
   RooAbsPdf *pdf;
   RooDataSet *data;
   std::tie(nll, pdf, data, _) = generate_1D_gaussian_pdf_nll(w, 10000);
   // when c++17 support arrives, change to this:
   // auto [nll, _] = generate_1D_gaussian_pdf_nll(w, 10000);

   RooMinimizer m1(*nll, RooMinimizer::FcnMode::gradient);
   m1.setMinimizerType("Minuit2");

   m1.setStrategy(0);
   m1.setPrintLevel(100);
   m1.setVerbose(kTRUE);

   m1.migrad();
}

TEST(GradMinimizer, GaussianND)
{
   // test RooMinimizer<RooGradMinimizerFcn> class with simple N-dimensional pdf

   RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);

   unsigned N = 5;
   unsigned N_events = 1000;
   // produce the same random stuff every time
   RooRandom::randomGenerator()->SetSeed(1);

   RooWorkspace w("w", kFALSE);

   std::unique_ptr<RooAbsReal> nll;
   std::unique_ptr<RooArgSet> all_values;
   RooAbsPdf *pdf;
   RooDataSet *data;
   std::tie(nll, pdf, data, all_values) = generate_ND_gaussian_pdf_nll(w, N, N_events);
   // when c++17 support arrives, change to this:
   //  auto [nll, all_values] = generate_ND_gaussian_pdf_nll(w, N, N_events);

   // save initial values for the start of all minimizations
   RooArgSet *savedValues = dynamic_cast<RooArgSet *>(all_values->snapshot());
   if (savedValues == nullptr) {
      throw std::runtime_error("params->snapshot() cannot be casted to RooArgSet!");
   }

   // --------

   RooMinimizer m0(*(nll.get()));
   m0.setMinimizerType("Minuit2");

   m0.setStrategy(0);
   m0.setPrintLevel(-1);

   m0.migrad();

   RooFitResult *m0result = m0.lastMinuitFit();
   double minNll0 = m0result->minNll();
   double edm0 = m0result->edm();
   std::vector<double> mean0(N);
   std::vector<double> std0(N);
   for (unsigned ix = 0; ix < N; ++ix) {
      {
         std::ostringstream os;
         os << "m" << ix;
         mean0[ix] = dynamic_cast<RooRealVar *>(w.arg(os.str().c_str()))->getVal();
      }
      {
         std::ostringstream os;
         os << "s" << ix;
         std0[ix] = dynamic_cast<RooRealVar *>(w.arg(os.str().c_str()))->getVal();
      }
   }

   // --------

   *all_values = *savedValues;

   // --------

   RooMinimizer m1(*(nll.get()), RooMinimizer::FcnMode::gradient);
   m1.setMinimizerType("Minuit2");

   m1.setStrategy(0);
   m1.setPrintLevel(-1);

   m1.migrad();

   RooFitResult *m1result = m1.lastMinuitFit();
   double minNll1 = m1result->minNll();
   double edm1 = m1result->edm();
   std::vector<double> mean1(N);
   std::vector<double> std1(N);
   for (unsigned ix = 0; ix < N; ++ix) {
      {
         std::ostringstream os;
         os << "m" << ix;
         mean1[ix] = dynamic_cast<RooRealVar *>(w.arg(os.str().c_str()))->getVal();
      }
      {
         std::ostringstream os;
         os << "s" << ix;
         std1[ix] = dynamic_cast<RooRealVar *>(w.arg(os.str().c_str()))->getVal();
      }
   }

   EXPECT_EQ(minNll0, minNll1);
   EXPECT_EQ(edm0, edm1);

   for (unsigned ix = 0; ix < N; ++ix) {
      EXPECT_EQ(mean0[ix], mean1[ix]);
      EXPECT_EQ(std0[ix], std1[ix]);
   }
}

TEST(GradMinimizerReverse, GaussianND)
{
   // test RooMinimizer<RooGradMinimizerFcn> class with simple N-dimensional pdf

   RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);

   unsigned N = 5;
   unsigned N_events = 1000;
   // produce the same random stuff every time
   RooRandom::randomGenerator()->SetSeed(1);

   RooWorkspace w("w", kFALSE);

   std::unique_ptr<RooAbsReal> nll;
   std::unique_ptr<RooArgSet> all_values;
   RooAbsPdf *pdf;
   RooDataSet *data;
   std::tie(nll, pdf, data, all_values) = generate_ND_gaussian_pdf_nll(w, N, N_events);
   // when c++17 support arrives, change to this:
   //  auto [nll, all_values] = generate_ND_gaussian_pdf_nll(w, N, N_events);

   // save initial values for the start of all minimizations
   RooArgSet *savedValues = dynamic_cast<RooArgSet *>(all_values->snapshot());
   if (savedValues == nullptr) {
      throw std::runtime_error("params->snapshot() cannot be casted to RooArgSet!");
   }

   // --------

   RooMinimizer m0(*nll, RooMinimizer::FcnMode::gradient);

   m0.setMinimizerType("Minuit2");

   m0.setStrategy(0);
   m0.setPrintLevel(-1);

   m0.migrad();

   RooFitResult *m0result = m0.lastMinuitFit();
   double minNll0 = m0result->minNll();
   double edm0 = m0result->edm();
   std::vector<double> mean0(N);
   std::vector<double> std0(N);
   for (unsigned ix = 0; ix < N; ++ix) {
      {
         std::ostringstream os;
         os << "m" << ix;
         mean0[ix] = dynamic_cast<RooRealVar *>(w.arg(os.str().c_str()))->getVal();
      }
      {
         std::ostringstream os;
         os << "s" << ix;
         std0[ix] = dynamic_cast<RooRealVar *>(w.arg(os.str().c_str()))->getVal();
      }
   }

   // --------

   *all_values = *savedValues;

   // --------

   RooMinimizer m1(*nll);
   m1.setMinimizerType("Minuit2");

   m1.setStrategy(0);
   m1.setPrintLevel(-1);

   m1.migrad();

   RooFitResult *m1result = m1.lastMinuitFit();
   double minNll1 = m1result->minNll();
   double edm1 = m1result->edm();
   std::vector<double> mean1(N);
   std::vector<double> std1(N);
   for (unsigned ix = 0; ix < N; ++ix) {
      {
         std::ostringstream os;
         os << "m" << ix;
         mean1[ix] = dynamic_cast<RooRealVar *>(w.arg(os.str().c_str()))->getVal();
      }
      {
         std::ostringstream os;
         os << "s" << ix;
         std1[ix] = dynamic_cast<RooRealVar *>(w.arg(os.str().c_str()))->getVal();
      }
   }

   EXPECT_EQ(minNll0, minNll1);
   EXPECT_EQ(edm0, edm1);

   for (unsigned ix = 0; ix < N; ++ix) {
      EXPECT_EQ(mean0[ix], mean1[ix]);
      EXPECT_EQ(std0[ix], std1[ix]);
   }
}

TEST(GradMinimizer, BranchingPDF)
{
   // test RooMinimizer<RooGradMinimizerFcn> class with an N-dimensional pdf that forms a tree of
   // pdfs, where one subpdf is the parameter of a higher level pdf

   RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);

   int N_events = 1000;
   // produce the same random stuff every time
   RooRandom::randomGenerator()->SetSeed(1);

   RooWorkspace w("w", false);

   // 3rd level
   w.factory("Gamma::ga0_0_1(k0_0_1[3,2,10],u[1,20],1,0)"); // leaf pdf
   // Gamma(mu,N+1,1,0) ~ Pois(N,mu), so this is a "continuous Poissonian"

   // 2nd level that will be linked to from 3rd level
   w.factory("Gamma::ga1_0(k1_0[4,2,10],z[1,20],1,0)"); // leaf pdf

   // rest of 3rd level
   w.factory("Gaussian::g0_0_0(v[-10,10],m0_0_0[0.6,-10,10],ga1_0)"); // two branch pdf, one up a level to different 1st
                                                                      // level branch

   // rest of 2nd level
   w.factory("Gaussian::g0_0(g0_0_0,m0_0[6,-10,10],ga0_0_1)"); // branch pdf

   // 1st level
   w.factory("Gaussian::g0(x[-10,10],g0_0,s0[3,0.1,10])");   // branch pdf
   w.factory("Gaussian::g1(y[-10,10],m1[-2,-10,10],ga1_0)"); // branch pdf
   RooArgSet level1_pdfs{*w.arg("g0"), *w.arg("g1")};

   // event counts for 1st level pdfs
   RooRealVar N_g0("N_g0", "#events g0", N_events / 10, 0., 10 * N_events);
   RooRealVar N_g1("N_g1", "#events g1", N_events / 10, 0., 10 * N_events);
   w.import(N_g0);
   w.import(N_g1);
   // gather in count_set
   RooArgSet level1_counts{N_g0, N_g1};

   // finally, sum the top level pdfs
   RooAddPdf sum("sum", "gaussian tree", level1_pdfs, level1_counts);

   // gather observables
   RooArgSet obs_set;
   for (auto obs : {"x", "y", "v"}) {
      obs_set.add(*w.arg(obs));
   }

   // --- Generate a toyMC sample from composite PDF ---
   std::unique_ptr<RooDataSet> data{sum.generate(obs_set, N_events)};

   auto nll = sum.createNLL(*data);

   // gather all values of parameters, observables, pdfs and nll here for easy
   // saving and restoring
   RooArgSet some_values{obs_set, w.allPdfs(), "some_values"};
   RooArgSet most_values{some_values, level1_counts, "most_values"};
   most_values.add(*nll);
   most_values.add(sum);

   std::unique_ptr<RooArgSet> param_set{nll->getParameters(obs_set)};

   RooArgSet all_values{most_values, *param_set, "all_values"};

   // set parameter values randomly so that they actually need to do some fitting
   auto it = all_values.fwdIterator();
   while (auto *val = dynamic_cast<RooRealVar *>(it.next())) {
      val->setVal(RooRandom::randomGenerator()->Uniform(val->getMin(), val->getMax()));
   }

   // save initial values for the start of all minimizations
   std::unique_ptr<RooArgSet> savedValues{static_cast<RooArgSet *>(all_values.snapshot())};
   if (savedValues == nullptr) {
      throw std::runtime_error("params->snapshot() cannot be casted to RooArgSet!");
   }

   // write the workspace state to file, but remove it again if everything was
   // successful (at the end of the test)
   w.import(*data);
   w.import(sum);
   w.writeToFile("failed_testRooGradMinimizer_BranchingPDF_workspace.root");

   // --------

   RooMinimizer m0(*nll);
   m0.setMinimizerType("Minuit2");

   m0.setStrategy(0);
   m0.setPrintLevel(-1);

   m0.migrad();

   RooFitResult *m0result = m0.lastMinuitFit();
   double minNll0 = m0result->minNll();
   double edm0 = m0result->edm();

   double N_g0__0 = N_g0.getVal();
   double N_g1__0 = N_g1.getVal();
   double k0_0_1__0 = dynamic_cast<RooRealVar *>(w.arg("k0_0_1"))->getVal();
   double k1_0__0 = dynamic_cast<RooRealVar *>(w.arg("k1_0"))->getVal();
   double m0_0__0 = dynamic_cast<RooRealVar *>(w.arg("m0_0"))->getVal();
   double m0_0_0__0 = dynamic_cast<RooRealVar *>(w.arg("m0_0_0"))->getVal();
   double m1__0 = dynamic_cast<RooRealVar *>(w.arg("m1"))->getVal();
   double s0__0 = dynamic_cast<RooRealVar *>(w.arg("s0"))->getVal();

   // --------

   all_values = *savedValues;

   // --------

   RooMinimizer m1(*nll, RooMinimizer::FcnMode::gradient);
   m1.setMinimizerType("Minuit2");

   m1.setStrategy(0);
   m1.setPrintLevel(-1);

   m1.migrad();

   RooFitResult *m1result = m1.lastMinuitFit();
   double minNll1 = m1result->minNll();
   double edm1 = m1result->edm();

   EXPECT_EQ(minNll0, minNll1);
   EXPECT_EQ(edm0, edm1);

   double N_g0__1 = N_g0.getVal();
   double N_g1__1 = N_g1.getVal();
   double k0_0_1__1 = dynamic_cast<RooRealVar *>(w.arg("k0_0_1"))->getVal();
   double k1_0__1 = dynamic_cast<RooRealVar *>(w.arg("k1_0"))->getVal();
   double m0_0__1 = dynamic_cast<RooRealVar *>(w.arg("m0_0"))->getVal();
   double m0_0_0__1 = dynamic_cast<RooRealVar *>(w.arg("m0_0_0"))->getVal();
   double m1__1 = dynamic_cast<RooRealVar *>(w.arg("m1"))->getVal();
   double s0__1 = dynamic_cast<RooRealVar *>(w.arg("s0"))->getVal();

   EXPECT_EQ(N_g0__0, N_g0__1);
   EXPECT_EQ(N_g1__0, N_g1__1);
   EXPECT_EQ(k0_0_1__0, k0_0_1__1);
   EXPECT_EQ(k1_0__0, k1_0__1);
   EXPECT_EQ(m0_0__0, m0_0__1);
   EXPECT_EQ(m0_0_0__0, m0_0_0__1);
   EXPECT_EQ(m1__0, m1__1);
   EXPECT_EQ(s0__0, s0__1);

   // N_g0    = 494.514  +/-  18.8621 (limited)
   // N_g1    = 505.817  +/-  24.6705 (limited)
   // k0_0_1    = 2.96883  +/-  0.00561152  (limited)
   // k1_0    = 4.12068  +/-  0.0565994 (limited)
   // m0_0    = 8.09563  +/-  1.30395 (limited)
   // m0_0_0    = 0.411472   +/-  0.183239  (limited)
   // m1    = -1.99988   +/-  0.00194089  (limited)
   // s0    = 3.04623  +/-  0.0982477 (limited)

   if (!HasFailure()) {
      if (remove("failed_testRooGradMinimizer_BranchingPDF_workspace.root") != 0) {
         std::cout << "Failed to remove failed_testRooGradMinimizer_BranchingPDF_workspace.root workspace file, sorry. "
                      "There were no failures though, so manually remove at your leisure."
                   << std::endl;
      }
   }
}

TEST(GradMinimizerDebugging, DISABLED_BranchingPDFLoadFromWorkspace)
{
   // test RooMinimizer<RooGradMinimizerFcn> class with an N-dimensional pdf that forms a tree of
   // pdfs, where one subpdf is the parameter of a higher level pdf

   // This version of the BranchingPDF test loads the random parameters written
   // to a workspace file by the original BranchingPDF test at some point.

   RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);

   TFile *f = new TFile("failed_testRooGradMinimizer_BranchingPDF_workspace.root");

   // Retrieve workspace from file
   RooWorkspace w = *static_cast<RooWorkspace *>(f->Get("w"));

   RooAddPdf sum = *static_cast<RooAddPdf *>(w.pdf("sum"));
   RooDataSet *data = static_cast<RooDataSet *>(w.data(""));

   auto nll = sum.createNLL(*data);

   RooArgSet all_values = w.allVars();
   //  RooArgSet all_values;
   //  for (auto var_name : {"x", "y", "z", "u", "v", "ga0_0_1", "ga1_0", "g0_0_0", "g0_0", "g0", "g1", "N_g0", "N_g1",
   //  "nll_sum_sumData", "sum", "k0_0_1", "k1_0", "m0_0", "m0_0_0", "m1", "s0"}) {
   //    all_values.add(*w.arg(var_name));
   //  }

   // save initial values for the start of all minimizations
   RooArgSet *savedValues = dynamic_cast<RooArgSet *>(all_values.snapshot());
   if (savedValues == nullptr) {
      throw std::runtime_error("params->snapshot() cannot be casted to RooArgSet!");
   }

   // --------

   all_values.Print("v");
   RooMinimizer m0(*nll);
   m0.setMinimizerType("Minuit2");

   m0.setStrategy(0);
   m0.setPrintLevel(-1);

   m0.migrad();

   RooFitResult *m0result = m0.lastMinuitFit();
   double minNll0 = m0result->minNll();
   double edm0 = m0result->edm();

   double N_g0__0 = dynamic_cast<RooRealVar *>(w.arg("N_g0"))->getVal();
   double N_g1__0 = dynamic_cast<RooRealVar *>(w.arg("N_g1"))->getVal();
   double k0_0_1__0 = dynamic_cast<RooRealVar *>(w.arg("k0_0_1"))->getVal();
   double k1_0__0 = dynamic_cast<RooRealVar *>(w.arg("k1_0"))->getVal();
   double m0_0__0 = dynamic_cast<RooRealVar *>(w.arg("m0_0"))->getVal();
   double m0_0_0__0 = dynamic_cast<RooRealVar *>(w.arg("m0_0_0"))->getVal();
   double m1__0 = dynamic_cast<RooRealVar *>(w.arg("m1"))->getVal();
   double s0__0 = dynamic_cast<RooRealVar *>(w.arg("s0"))->getVal();

   all_values.Print("v");

   // --------

   all_values = *savedValues;

   // --------

   all_values.Print("v");

   RooMinimizer m1(*nll, RooMinimizer::FcnMode::gradient);
   m1.setMinimizerType("Minuit2");

   m1.setStrategy(0);
   m1.setPrintLevel(-1);

   m1.migrad();

   RooFitResult *m1result = m1.lastMinuitFit();
   double minNll1 = m1result->minNll();
   double edm1 = m1result->edm();

   EXPECT_EQ(minNll0, minNll1);
   EXPECT_EQ(edm0, edm1);

   double N_g0__1 = dynamic_cast<RooRealVar *>(w.arg("N_g0"))->getVal();
   double N_g1__1 = dynamic_cast<RooRealVar *>(w.arg("N_g1"))->getVal();
   double k0_0_1__1 = dynamic_cast<RooRealVar *>(w.arg("k0_0_1"))->getVal();
   double k1_0__1 = dynamic_cast<RooRealVar *>(w.arg("k1_0"))->getVal();
   double m0_0__1 = dynamic_cast<RooRealVar *>(w.arg("m0_0"))->getVal();
   double m0_0_0__1 = dynamic_cast<RooRealVar *>(w.arg("m0_0_0"))->getVal();
   double m1__1 = dynamic_cast<RooRealVar *>(w.arg("m1"))->getVal();
   double s0__1 = dynamic_cast<RooRealVar *>(w.arg("s0"))->getVal();

   EXPECT_EQ(N_g0__0, N_g0__1);
   EXPECT_EQ(N_g1__0, N_g1__1);
   EXPECT_EQ(k0_0_1__0, k0_0_1__1);
   EXPECT_EQ(k1_0__0, k1_0__1);
   EXPECT_EQ(m0_0__0, m0_0__1);
   EXPECT_EQ(m0_0_0__0, m0_0_0__1);
   EXPECT_EQ(m1__0, m1__1);
   EXPECT_EQ(s0__0, s0__1);

   all_values.Print("v");

   // N_g0    = 494.514  +/-  18.8621 (limited)
   // N_g1    = 505.817  +/-  24.6705 (limited)
   // k0_0_1    = 2.96883  +/-  0.00561152  (limited)
   // k1_0    = 4.12068  +/-  0.0565994 (limited)
   // m0_0    = 8.09563  +/-  1.30395 (limited)
   // m0_0_0    = 0.411472   +/-  0.183239  (limited)
   // m1    = -1.99988   +/-  0.00194089  (limited)
   // s0    = 3.04623  +/-  0.0982477 (limited)
}

TEST(GradMinimizerDebugging, DISABLED_BranchingPDFLoadFromWorkspaceNominal)
{
   // only run the nominal minimizer of the BranchingPDF test and print results

   TFile *f = new TFile("failed_testRooGradMinimizer_BranchingPDF_workspace.root");
   RooWorkspace w = *static_cast<RooWorkspace *>(f->Get("w"));
   RooAddPdf sum = *static_cast<RooAddPdf *>(w.pdf("sum"));
   RooDataSet *data = static_cast<RooDataSet *>(w.data(""));
   auto nll = sum.createNLL(*data);

   RooMinimizer m0(*nll);
   m0.setMinimizerType("Minuit2");
   m0.setStrategy(0);
   m0.migrad();
}

TEST(GradMinimizerDebugging, DISABLED_BranchingPDFLoadFromWorkspaceGradMinimizer)
{
   // only run the GradMinimizer from the BranchingPDF test and print results

   TFile *f = new TFile("failed_testRooGradMinimizer_BranchingPDF_workspace.root");
   RooWorkspace w = *static_cast<RooWorkspace *>(f->Get("w"));
   RooAddPdf sum = *static_cast<RooAddPdf *>(w.pdf("sum"));
   RooDataSet *data = static_cast<RooDataSet *>(w.data(""));
   auto nll = sum.createNLL(*data);

   RooMinimizer m0(*nll, RooMinimizer::FcnMode::gradient);
   m0.setMinimizerType("Minuit2");
   m0.setStrategy(0);
   m0.migrad();
}
