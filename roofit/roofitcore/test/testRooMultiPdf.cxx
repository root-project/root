#include <RooCategory.h>
#include <RooConstVar.h>
#include <RooGaussian.h>
#include <RooMultiPdf.h>
#include <RooRealVar.h>
#include <RooBernstein.h>
#include <RooExponential.h>
#include <RooMinimizer.h>
#include <gtest/gtest.h>

#include <memory>
// Helper function to count parameters including x
int countFloatingParametersIncludingObservable(const RooAbsPdf &pdf)
{
   std::unique_ptr<RooArgSet> params{pdf.getParameters(RooArgSet())};
   int count = 0;
   for (auto *obj : *params) {
      auto *var = dynamic_cast<RooRealVar *>(obj);
      if (var && !var->isConstant()) {
         count++;
      }
   }

   return count;
}

TEST(RooMultiPdf, SelectsCorrectPdf)
{

   RooRealVar x("x", "x", -10, 10);
   x.setVal(2.0);

   RooRealVar m1("mean1", "mean1", 0.);
   RooRealVar s1("sigma1", "sigma1", 1., 0.001, 10.);
   RooRealVar m2("mean2", "mean2", 2.);
   RooRealVar s2("sigma2", "sigma2", 1., 0.001, 10.);
   RooGaussian gaus1("gaus1", "gaus1", x, m1, s1);
   RooGaussian gaus2("gaus2", "gaus2", x, m2, s2);

   RooCategory indx("my_special_index", "my_index");

   RooArgList list{gaus1, gaus2};

   RooMultiPdf pdf("mult", "multi_pdf", indx, list);

   RooArgSet normSet{x};

   indx.setIndex(0);
   EXPECT_EQ(pdf.getVal(normSet), gaus1.getVal(normSet));

   indx.setIndex(1);
   EXPECT_EQ(pdf.getVal(normSet), gaus2.getVal(normSet));
}

TEST(RooMultiPdfTest, FitConvergesAndReturnsReasonableResult)

{
   RooRealVar x("x", "x", -10, 10);
   RooRealVar m1("mean1", "mean1", 0., -10, 10);
   RooRealVar s1("sigma1", "sigma1", 1., 0.001, 10.);
   RooGaussian gaus1("gaus1", "gaus1", x, m1, s1);

   RooRealVar expo_1("expo_1", "slope of exponential", -0.01, -0.2, -0.0003);
   RooExponential exponential("exponential", "exponential pdf", x, expo_1);

   RooRealVar poly_1("poly_1", "T1 of chebychev polynomial", 0, 0, 2);
   RooRealVar poly_2("poly_2", "T2 of chebychev polynomial", 0, 0, 2);
   RooBernstein polynomial("polynomial", "polynomial pdf", x, RooArgList(poly_1, poly_2));

   RooCategory indx("my_special_index", "my_index");
   indx.setConstant();

   RooArgList mypdfs;
   mypdfs.add(gaus1);
   mypdfs.add(polynomial);
   mypdfs.add(exponential);

   RooMultiPdf multipdf("roomultipdf", "pdfs", indx, mypdfs);
   indx.setIndex(0); // choose Gaussian initially

   std::unique_ptr<RooDataSet> data{gaus1.generate(x, 1000)};

   // Fit 1 - RooMultiPdf fit

   std::unique_ptr<RooAbsReal> nll{multipdf.createNLL(*data, RooFit::EvalBackend("codegen"))};

   RooMinimizer minim{*nll};
   minim.setStrategy(0);
   int status = minim.minimize("Minuit2", "");

   // Reseting the initial parameters

   m1.setVal(0.);
   s1.setVal(1.);

   m1.setError(0.0);
   s1.setError(0.0);
   // Fit 2 - Reference fit
   std::unique_ptr<RooAbsReal> nll1{gaus1.createNLL(*data, RooFit::EvalBackend("codegen"))};

   RooMinimizer minim1{*nll1};
   minim1.setStrategy(0);
   int status1 = minim1.minimize("Minuit2", "");

   int n_param_gaus1 = countFloatingParametersIncludingObservable(gaus1);

   double first_fit = nll->getVal();
   double ref_fit = nll1->getVal() + 0.5 * n_param_gaus1; // 1.5 because the gaussian has 3 param*0.5
   // Now test the results
   EXPECT_EQ(status, 0) << "Fit 1 did not converge.";
   EXPECT_EQ(status1, 0) << "Fit 2 did not converge.";
   EXPECT_TRUE(std::isfinite(first_fit)) << "NLL is not finite.";
   EXPECT_TRUE(std::isfinite(ref_fit)) << "NLL1 is not finite.";
   EXPECT_DOUBLE_EQ(first_fit, ref_fit);

   // Check that the correct number of PDFs are present
   EXPECT_EQ(multipdf.getNumPdfs(), 3);

   //  check fitted parameter
   EXPECT_NEAR(m1.getVal(), 0.0, 0.2);
   EXPECT_NEAR(s1.getVal(), 1.0, 0.2);

   // Check whether RooMultiPdf chooses the correct index

   for (int i = 0; i < multipdf.getNumPdfs(); ++i) {

      indx.setIndex(i);

      std::unique_ptr<RooAbsReal> nll_multi{multipdf.createNLL(*data, RooFit::EvalBackend("codegen"))};

      RooAbsPdf *selectedPdf = multipdf.getPdf(i);
      std::unique_ptr<RooAbsReal> nll_direct{selectedPdf->createNLL(*data, RooFit::EvalBackend("codegen"))};

      int n_param = countFloatingParametersIncludingObservable(*selectedPdf);

      double multi = nll_multi->getVal();
      double direct = nll_direct->getVal() + 0.5 * n_param;

      std::cout << "PDF index " << i << ": n_param = " << n_param << ", direct+penalty = " << direct
                << ", multipdf = " << multi << std::endl;

      EXPECT_NEAR(multi, direct, 1e-6) << "Mismatch at index " << i;
   }
}
TEST(RooMultiPdfTest, PenaltyTermIsAppliedCorrectly)
{
   using namespace RooFit;

   RooRealVar x("x", "x", -10, 10);

   RooRealVar mean("mean", "mean", 0, -5, 5);
   RooRealVar sigma("sigma", "sigma", 1, 0.1, 10);

   RooGaussian gauss1("gauss1", "gauss1", x, mean, sigma);
   RooGaussian gauss2("gauss2", "gauss2", x, mean, sigma);

   RooCategory index("index", "index");

   RooArgList pdfList(gauss1, gauss2);

   RooMultiPdf multiPdf("multiPdf", "multiPdf", index, pdfList);
   index.setConstant();

   index.setIndex(0);

   std::unique_ptr<RooDataSet> data{gauss1.generate(x, 100)};

   std::unique_ptr<RooAbsReal> nll_gauss1{gauss1.createNLL(*data, EvalBackend("codegen"))};

   std::unique_ptr<RooAbsReal> nll_multi{multiPdf.createNLL(*data, EvalBackend("codegen"))};

   double val_gauss1 = nll_gauss1->getVal();
   double val_multi = nll_multi->getVal();
   int n_params = countFloatingParametersIncludingObservable(gauss1);

   const double expected_penalty = 0.5 * n_params;
   const double delta = val_multi - val_gauss1;

   std::cout << "NLL(gauss1):     " << val_gauss1 << std::endl;
   std::cout << "NLL(multiPdf):   " << val_multi << std::endl;
   std::cout << "Expected penalty: " << expected_penalty << std::endl;
   std::cout << "Delta:           " << delta << std::endl;

   EXPECT_TRUE(std::isfinite(val_gauss1));
   EXPECT_TRUE(std::isfinite(val_multi));

   EXPECT_NEAR(delta, expected_penalty, 1e-6) << "Penalty term not correctly applied.";
}