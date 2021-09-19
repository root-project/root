// Tests for the GenericPdf
// Authors: Stephan Hageboeck, CERN  05/2019
//          Jonas Rembser, CERN 06/2022

#include <RooExponential.h>
#include <RooRealVar.h>
#include <RooGenericPdf.h>
#include <RooWorkspace.h>

#include <gtest/gtest.h>

#define MAKE_JOHNSON_AND_VARS RooRealVar mass("mass", "mass", 0., -200., 200.);\
RooRealVar mu("mu", "Location parameter of normal distribution", 100., -200., 200.);\
RooRealVar sigma("sigma", "Two sigma of normal distribution", 2., 0., 100.);\
RooRealVar gamma("gamma", "gamma", -10., -100., 100.);\
RooRealVar delta("delta", "delta", 3., 0., 100.);

const char* fixedFormula = "delta/(sigma*TMath::Sqrt(TMath::TwoPi()))"
    "*TMath::Exp(-0.5*(gamma+delta*TMath::ASinH((mass-mu)/sigma))"
                    "*(gamma+delta*TMath::ASinH((mass-mu)/sigma)))"
    "/TMath::Sqrt(1+(mass-mu)*(mass-mu)/(sigma*sigma))";



TEST(GenericPdf, CrashWhenRunningJohnson)
{
  MAKE_JOHNSON_AND_VARS
  RooGenericPdf johnsonRef("johnsonRef",
      fixedFormula,
      RooArgSet( mass, mu, sigma, gamma, delta));

  for (double theMass : {1., 10., 50., 100.}) {
    mass = theMass;
    EXPECT_NE(johnsonRef.getVal(), 0.) << theMass;
  }
}

// ROOT-10411
TEST(GenericPdf, CrashWhenRenamingArguments) {
  RooRealVar var("var", "var", 0.1, 0, 1);
  RooRealVar par("par", "par", 0.5, 0, 1);
  RooGenericPdf genPdf("genPdf", "var*(par + 1)", RooArgSet(var, par));

  par.SetName("par_test");

  auto pdfClone = (RooAbsPdf*)genPdf.clone("clone");
  // Would crash:
  EXPECT_NEAR(pdfClone->getVal(), 0.15, 1.E-6);

  par.SetName("par");

  RooWorkspace ws("ws");
  ws.import(genPdf, RooFit::RenameAllVariablesExcept("new", "var"));
  auto impPdf = ws.pdf("genPdf");
  // Would crash:
  EXPECT_NEAR(impPdf->getVal(), 0.15, 1.E-6);
}

// ROOT-5101: Identity PDF affects normalization
TEST(GenericPdf, IdentidyPdfNormalization) {
  RooRealVar x{"x", "x", 0.0, 100.0};
  RooRealVar s{"s", "s", -0.5, -10.0,  0.0};

  RooExponential exp{"exp", "exp", x, s};

  // This RooGenericPdf should behave exactly the same as the exponential, only
  // that the integration will be done numerically.
  RooGenericPdf pdf{"pdf", "pdf", "@0", {exp}};

  RooArgSet normSet{x};

  // Check that the values with and without normalization are almost identical.
  // They are not exactly identical for the normalized case, because the
  // RooGenericPdf doesn't do analytic integration.
  constexpr double tol = 1e-6;
  EXPECT_NEAR(exp.getVal(), pdf.getVal(), tol);
  EXPECT_NEAR(exp.getVal(normSet), pdf.getVal(normSet), tol);
}
