// Tests for the RooProdPdf
// Authors: Stephan Hageboeck, CERN  02/2019
//          Jonas Rembser, CERN, June 2021

#include <RooArgList.h>
#include <RooArgSet.h>
#include <RooGenericPdf.h>
#include <RooProdPdf.h>
#include <RooRealVar.h>
#include <RooDataSet.h>
#include <RooGaussModel.h>
#include <RooDecay.h>
#include <RooFitResult.h>
#include <RooConstVar.h>
#include <RooGamma.h>

#include <gtest/gtest.h>

#include <memory>
#include <sstream>
#include <string>

class TestProdPdf : public ::testing::Test {
protected:
   TestProdPdf() : Test()
   {
      datap.reset(prod.generate(RooArgSet(x), 1000));
      a.setConstant(true);
   }

   constexpr static double bTruth = -0.5;

   RooRealVar x{"x", "x", 2., 0., 5.};
   RooRealVar a{"a", "a", -0.2, -5., 0.};
   RooRealVar b{"b", "b", bTruth, -5., 0.};

   RooGenericPdf c1{"c1", "exp(x[0]*x[1])", RooArgSet(x, a)};
   RooGenericPdf c2{"c2", "exp(x[0]*x[1])", RooArgSet(x, b)};
   RooProdPdf prod{"mypdf", "mypdf", RooArgList(c1, c2)};
   std::unique_ptr<RooDataSet> datap{nullptr};
};

TEST_F(TestProdPdf, CachingOpt2)
{
   prod.fitTo(*datap, RooFit::Optimize(2), RooFit::PrintLevel(-1));
   EXPECT_LT(fabs(b.getVal() - bTruth), b.getError() * 1.1)
      << "b=" << b.getVal() << " +- " << b.getError() << " doesn't match truth value with O2.";
}

TEST_F(TestProdPdf, CachingOpt1)
{
   prod.fitTo(*datap, RooFit::Optimize(1), RooFit::PrintLevel(-1));
   EXPECT_LT(fabs(b.getVal() - bTruth), b.getError() * 1.1)
      << "b=" << b.getVal() << " +- " << b.getError() << " doesn't match truth value with O1.";
}

TEST_F(TestProdPdf, CachingOpt0)
{
   prod.fitTo(*datap, RooFit::Optimize(0), RooFit::PrintLevel(-1));
   EXPECT_LT(fabs(b.getVal() - bTruth), b.getError() * 1.1)
      << "b=" << b.getVal() << " +- " << b.getError() << " doesn't match truth value with O0.";
}

std::vector<std::vector<RooAbsArg *>> allPossibleSubset(RooAbsCollection const &arr)
{
   std::vector<std::vector<RooAbsArg *>> out;
   std::size_t n = arr.size();

   std::size_t count = std::pow(2, n);
   for (std::size_t i = 0; i < count; i++) {
      out.emplace_back();
      auto &back = out.back();
      for (std::size_t j = 0; j < n; j++) {
         if ((i & (1 << j)) != 0) {
            back.push_back(arr[j]);
         }
      }
   }
   return out;
}

// Hash the integral configuration for all possible normalization sets.
unsigned int hashRooProduct(RooProdPdf const &prod)
{
   RooArgSet params;
   prod.getParameters(nullptr, params);
   auto subsets = allPossibleSubset(params);

   std::stringstream ss;

   for (auto const &subset : subsets) {
      // this can't be on the stack, otherwise we will always get the same
      // address and therefore get wrong cache hits!
      auto nset = std::make_unique<RooArgSet>(subset.begin(), subset.end());
      prod.writeCacheToStream(ss, nset.get());
   }

   std::string s = ss.str();
   return TString::Hash(s.c_str(), s.size());
}

TEST(RooProdPdf, TestGetPartIntList)
{
   // This test checks if RooProdPdf::getPartIntList factorizes the integrals
   // as expected.
   // Instead of trying to construct tests for all possible cases by hand,
   // this test creates a product where the factors have different patters of
   // overlapping parameters. To make sure all possible cases are covered, we
   // are using all possible subsets of the parameters one after the other to
   // create the reference test result.

   RooRealVar x{"x", "x", 1., 0, 10};
   RooRealVar y{"y", "y", 1., 0, 10};
   RooRealVar z{"z", "z", 1., 0, 10};

   RooRealVar m1{"m1", "m1", 1., 0, 10};
   RooRealVar m2{"m2", "m2", 1., 0, 10};
   RooRealVar m3{"m3", "m3", 1., 0, 10};

   RooGenericPdf gauss1{"gauss1", "gauss1", "x+m1", {x, m1}};
   RooGenericPdf gauss2{"gauss2", "gauss2", "x+m2", {x, m2}};
   RooGenericPdf gauss3{"gauss3", "gauss3", "y+m3", {y, m3}};
   RooGenericPdf gauss4{"gauss4", "gauss4", "z+m1", {z, m1}};
   RooGenericPdf gauss5{"gauss5", "gauss5", "x+m1", {x, m1}};

   // Product of all the pdfs.
   RooProdPdf prod{"prod", "prod", RooArgList{gauss1, gauss2, gauss3, gauss4, gauss5}};

   // We hash the string serializations of caches for all possible
   // normalization sets and compare it to the expected hash.
   // This value must be updated if the convention for integral names in
   // RooProdPdf changes.
   EXPECT_EQ(hashRooProduct(prod), 2448666198);
}

TEST(RooProdPdf, TestDepsAreCond)
{
   using namespace RooFit;

   RooRealVar x("x", "", 0, 1);
   RooRealVar xErr("xErr", "", 0.0001, 0.1);

   RooGaussModel gm("gm", "", x, RooConst(0), xErr);

   RooRealVar tau("tau", "", 0.4, 0, 1);
   RooDecay decayPdf("decayPdf", "", x, tau, gm, RooDecay::SingleSided);

   RooGamma errPdf("errPdf", "", xErr, RooConst(4), RooConst(0.005), RooConst(0));

   // What we want: decayPdf(x|xErr)*errPdf(xErr):
   RooProdPdf pdf1("pdf1", "", RooArgSet(errPdf), Conditional(decayPdf, x, false));

   // Should be the same as pdf1:
   RooProdPdf pdf2("pdf2", "", RooArgSet(errPdf), Conditional(decayPdf, xErr, true));

   std::unique_ptr<RooDataSet> data{pdf1.generate(RooArgSet(x, xErr), NumEvents(10000))};

   auto resetParameters = [&]() {
      tau.setVal(0.4);
      tau.setError(0.0);
   };

   using ResultPtr = std::unique_ptr<RooFitResult>;

   ResultPtr result1{pdf1.fitTo(*data, Save(), BatchMode("off"), PrintLevel(-1))};
   resetParameters();
   ResultPtr result2{pdf1.fitTo(*data, Save(), BatchMode("cpu"), PrintLevel(-1))};
   resetParameters();
   ResultPtr result3{pdf2.fitTo(*data, Save(), BatchMode("off"), PrintLevel(-1))};
   resetParameters();
   ResultPtr result4{pdf2.fitTo(*data, Save(), BatchMode("cpu"), PrintLevel(-1))};

   EXPECT_TRUE(result2->isIdentical(*result1)) << "batchmode fit is inconsistent!";
   EXPECT_TRUE(result3->isIdentical(*result1)) << "alternative model fit is inconsistent!";
   EXPECT_TRUE(result4->isIdentical(*result1)) << "alternative model batchmode fit is inconsistent!";
}
