// Tests for the RooSpline

#include <RooRealVar.h>
#include <RooSpline.h>

#include <TGraph.h>

#include <gtest/gtest.h>

#include <vector>

// The constructor from a TGraph swapped the logx and logy flags.
TEST(RooSpline, LogFlagsFromTGraph)
{
   RooRealVar x{"x", "x", 2.0, 1.0, 4.0};

   const std::vector<double> x0{1.0, 2.0, 3.0, 4.0};
   const std::vector<double> y0{1.0, 4.0, 9.0, 16.0};
   TGraph graph{static_cast<int>(x0.size()), x0.data(), y0.data()};

   RooSpline fromSpans{"fromSpans", "", x, x0, y0, 3, /*logx=*/true, /*logy=*/false};
   RooSpline fromGraph{"fromGraph", "", x, graph, 3, /*logx=*/true, /*logy=*/false};

   EXPECT_TRUE(fromGraph.logx());
   EXPECT_FALSE(fromGraph.logy());

   for (double xVal : {1.0, 1.5, 2.7, 4.0}) {
      x.setVal(xVal);
      EXPECT_DOUBLE_EQ(fromGraph.getVal(), fromSpans.getVal()) << "x = " << xVal;
   }
}
