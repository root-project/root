#include <RooCategory.h>
#include <RooGaussian.h>
#include <RooMultiPdf.h>
#include <RooRealVar.h>

#include <gtest/gtest.h>

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
