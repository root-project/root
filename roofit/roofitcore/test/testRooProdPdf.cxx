// Tests for the RooProdPdf
// Author: Jonas Rembser, CERN, June 2021

#include "RooArgList.h"
#include "RooArgSet.h"
#include "RooPoisson.h"
#include "RooProdPdf.h"
#include "RooRealVar.h"

#include "gtest/gtest.h"

#include <memory>
#include <sstream>
#include <string>

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

   RooPoisson gauss1{"gauss1", "gauss1", x, m1};
   RooPoisson gauss2{"gauss2", "gauss2", x, m2};
   RooPoisson gauss3{"gauss3", "gauss3", y, m3};
   RooPoisson gauss4{"gauss4", "gauss4", z, m1};
   RooPoisson gauss5{"gauss5", "gauss5", x, m1};

   // Product of all the Gaussians.
   RooProdPdf prod{"prod", "prod", RooArgList{gauss1, gauss2, gauss3, gauss4, gauss5}};

   // We hash the string serializations of caches for all possible
   // normalization sets and compare it to the expected hash.
   // This value must be updated if the convention for integral names in
   // RooProdPdf changes.
   EXPECT_EQ(hashRooProduct(prod), 2448666198);
}
