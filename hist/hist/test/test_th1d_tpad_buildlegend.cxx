#include <gtest/gtest.h>
#include <TCanvas.h>
#include <TH1D.h>

TEST(TH1D, DrawWithBuildLegend)
{

   TH1D h1{"h1", "h1", 10, 0, 1};
   auto nFills{10};
   for (auto i = 0; i < nFills; i++) {
      h1.Fill(0.45);
   }

   // Set a range on the Y axis lower cannot contain the maximum value: this triggered a bug in TPad::FillCollideGridTH1
   // which would access the vector to find empty spaces for the legend out of bounds.
   h1.GetYaxis()->SetRangeUser(0, 8);

   TCanvas c;
   h1.Draw();
   // Need to check that BuildLegend can work also when the Y axis does not contain the maximum value
   c.BuildLegend();
}
