// Tests for the RooSimPdfBuilder
// Authors: Jonas Rembser, CERN  03/2021

#include "RooCategory.h"
#include "RooRealVar.h"
#include "RooArgSet.h"
#include "RooDataSet.h"
#include "RooGenericPdf.h"
#include "RooFitResult.h"
#include "RooExtendPdf.h"
#include "RooSimultaneous.h"
#include "RooSimPdfBuilder.h"
#include "RooStringVar.h"
#include "RooMsgService.h"

#include <memory>

#include "gtest/gtest.h"

TEST(TestRooSimPdfBuilder, Example)
{
   using namespace RooFit;

   // silence log output
   RooMsgService::instance().setGlobalKillBelow(RooFit::WARNING);

   RooCategory C("C", "C");
   C.defineType("C1");
   C.defineType("C2");

   RooRealVar X("X", "X", -10., 10.);

   RooArgSet varSet{X, C};
   std::unique_ptr<RooDataSet> data = nullptr;

   // reference fit results
   double fit_m_ref = 0.0;
   double fit_s_C1_ref = 0.0;
   double fit_s_C2_ref = 0.0;

   {
      // Build PDF manually, do refernce fit and generate dataset

      RooRealVar m("m", "mean of gaussian", 0, -10, 10);
      RooRealVar s_C1("s_C1", "sigma of gaussian C1", 3, 0.1, 10);
      RooRealVar s_C2("s_C2", "sigma of gaussian C2", 5, 0.1, 10);
      RooGenericPdf gauss_C1("gauss_C1", "gaussian C1", "std::exp(-0.5*(X - m)^2/s_C1^2)", {X, m, s_C1});
      RooGenericPdf gauss_C2("gauss_C2", "gaussian C2", "std::exp(-0.5*(X - m)^2/s_C2^2)", {X, m, s_C2});

      RooRealVar N_C1("N_C1", "Extended term C1", 1., 0., 10.);
      RooRealVar N_C2("N_C2", "Extended term C2", 1., 0., 10.);

      RooExtendPdf pdf_C1("pdf_C1", "Extended model C1", gauss_C1, N_C1, "FULL");
      RooExtendPdf pdf_C2("pdf_C2", "Extended model C2", gauss_C2, N_C2, "FULL");

      RooSimultaneous simPdf("simPdf", "simPdf", C);
      simPdf.addPdf(pdf_C1, "C1");
      simPdf.addPdf(pdf_C2, "C2");

      data.reset(simPdf.generate(varSet, 2000));

      std::unique_ptr<RooFitResult> result{simPdf.fitTo(*data, Verbose(false), PrintLevel(-1), Save())};

      auto floatResults = result->floatParsFinal();

      fit_m_ref = static_cast<RooAbsReal *>(floatResults.find(m))->getVal();
      fit_s_C1_ref = static_cast<RooAbsReal *>(floatResults.find(s_C1))->getVal();
      fit_s_C2_ref = static_cast<RooAbsReal *>(floatResults.find(s_C2))->getVal();
   }

   {
      // Fit the dataset with a PDF that was generated with RooSimPdfBuilder

      RooRealVar m("m", "mean of gaussian", -10, 10);
      RooRealVar s("s", "sigma of gaussian", 0.1, 10);
      RooGenericPdf gauss("gauss", "gaussian", "std::exp(-0.5*(X - m)^2/s^2)", {X, m, s});
      RooRealVar N("N", "Extended term", 1., 0., 10.);
      RooExtendPdf pdf("pdf", "Extended model", gauss, N, "FULL");

      RooSimPdfBuilder builder(pdf);
      std::unique_ptr<RooArgSet> cfg{builder.createProtoBuildConfig()};
      dynamic_cast<RooStringVar &>((*cfg)["physModels"]) = "pdf"; // Name of the PDF we are going to work with
      dynamic_cast<RooStringVar &>((*cfg)["splitCats"]) = "C";      // Category used to differentiate sub-datasets
      dynamic_cast<RooStringVar &>((*cfg)["pdf"]) = "C : s, N";      // Prescription to taylor PDF parameters k and s
                                                                    // for each data subset designated by C states
      RooSimultaneous *simPdf = builder.buildPdf(*cfg, data.get());

      std::unique_ptr<RooFitResult> result{simPdf->fitTo(*data, Verbose(false), PrintLevel(-1), Save())};

      auto floatResults = result->floatParsFinal();

      double fit_m = static_cast<RooAbsReal *>(floatResults.find(m))->getVal();
      double fit_s_C1 = static_cast<RooAbsReal *>(floatResults.find("s_C1"))->getVal();
      double fit_s_C2 = static_cast<RooAbsReal *>(floatResults.find("s_C2"))->getVal();

      // The fit results should be basically identical no matter if you build
      // the PDF yourself or if you use the RooSimPdfBuilder.
      double accuracy = 0.03;
      EXPECT_NEAR(fit_m, fit_m_ref, accuracy*std::abs(fit_m_ref));
      EXPECT_NEAR(fit_s_C1, fit_s_C1_ref, accuracy*std::abs(fit_s_C1_ref));
      EXPECT_NEAR(fit_s_C2, fit_s_C2_ref, accuracy*std::abs(fit_s_C2_ref));
   }
}
