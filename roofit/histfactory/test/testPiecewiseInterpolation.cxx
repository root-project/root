// Tests for the PiecewiseInterpolation
// Authors: Jonas Rembser, CERN  12/2024

#include <RooStats/HistFactory/PiecewiseInterpolation.h>

#include <RooRealVar.h>

#include <gtest/gtest.h>

/// Validate that the interpolation codes are "additive" or "multiplicative" as documented.
TEST(PiecewiseInterpolation, AdditiveOrMultiplicative)
{
   using namespace RooFit;

   // In the usual use cases, the nominal value is 1.0, but we spice up this
   // test a little bit by changing that.
   double nVal = 3.0;
   RooRealVar nominal{"nominal", "nominal", nVal};

   RooRealVar param1{"param_1", "param_1", -2., 2.};
   RooRealVar low1{"low_1", "low_1", nVal * 0.75};
   RooRealVar high1{"high_1", "high_1", nVal * 1.5};

   RooRealVar param2{"param_2", "param_2", -1.5, 1.5};
   RooRealVar low2{"low_2", "low_2", nVal * 0.8};
   RooRealVar high2{"high_2", "high_2", nVal * 1.25};

   int nBins = 10;

   param1.setBins(nBins);
   param2.setBins(nBins);

   RooArgList paramsSet1{param1};
   RooArgList lowSet1{low1};
   RooArgList highSet1{high1};

   RooArgList paramsSet2{param2};
   RooArgList lowSet2{low2};
   RooArgList highSet2{high2};

   RooArgList paramsSetBoth{param1, param2};
   RooArgList lowSetBoth{low1, low2};
   RooArgList highSetBoth{high1, high2};

   PiecewiseInterpolation pci1{"piecewise1", "", nominal, lowSet1, highSet1, paramsSet1};
   PiecewiseInterpolation pci2{"piecewise2", "", nominal, lowSet2, highSet2, paramsSet2};
   PiecewiseInterpolation pciBoth{"piecewiseBoth", "", nominal, lowSetBoth, highSetBoth, paramsSetBoth};

   std::vector<int> codes{0, 1, 2, 4, 5, 6};
   std::vector<bool> isMultiplicative{false, true, false, false, true, true};

   for (std::size_t iCode = 0; iCode < codes.size(); ++iCode) {

      int code = codes[iCode];

      pci1.setAllInterpCodes(code);
      pci2.setAllInterpCodes(code);
      pciBoth.setAllInterpCodes(code);

      // basic check that when param1 and param2 are equal to 1, pci1 and pci2 are equal to high
      // and pciBoth is equal when the respective parameter is 1
      param2.setVal(0);
      param1.setVal(1);
      EXPECT_FLOAT_EQ(pci1.getVal(), high1.getVal());
      EXPECT_FLOAT_EQ(pciBoth.getVal(), high1.getVal());
      param1.setVal(0);
      param2.setVal(1);
      EXPECT_FLOAT_EQ(pci2.getVal(), high2.getVal());
      EXPECT_FLOAT_EQ(pciBoth.getVal(), high2.getVal());
      param2.setVal(0);
      // and similarly for -1
      param1.setVal(-1);
      EXPECT_FLOAT_EQ(pci1.getVal(), low1.getVal());
      EXPECT_FLOAT_EQ(pciBoth.getVal(), low1.getVal());
      param1.setVal(0);
      param2.setVal(-1);
      EXPECT_FLOAT_EQ(pci2.getVal(), low2.getVal());
      EXPECT_FLOAT_EQ(pciBoth.getVal(), low2.getVal());
      param2.setVal(0);

      for (int ibin1 = 0; ibin1 < param1.numBins(); ++ibin1) {
         for (int ibin2 = 0; ibin2 < param2.numBins(); ++ibin2) {
            param1.setBin(ibin1);
            param2.setBin(ibin2);

            double nom = nominal.getVal();
            double v1 = pci1.getVal();
            double v2 = pci2.getVal();
            double vBoth = pciBoth.getVal();

            // The definition of multiplicative and additive is in this test
            double vBothMultRef = (v1 / nom) * (v2 / nom) * nom;
            double vBothAddiRef = (v1 - nom) + (v2 - nom) + nom;

            if (isMultiplicative[iCode]) {
               EXPECT_FLOAT_EQ(vBoth, vBothMultRef);
            } else {
               EXPECT_FLOAT_EQ(vBoth, vBothAddiRef);
            }
         }
      }
   }
}
