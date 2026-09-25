// @(#)root/histpainter/test/test_TGraphErrors_band.cxx
// Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
// All rights reserved.
//
// For the licensing terms see $ROOTSYS/LICENSE.
// For the list of contributors see the CREDITS file.

// Regression test for https://github.com/root-project/root/issues/23491
//
// The vertices of the error band drawn by TGraphErrors with the option "3" (and
// "4") used to be clamped to the user range of the axes. Clamping moves a
// vertex along the axis instead of cutting the band off at the frame, so the
// shape of the band changed with the axis range. The band must keep the
// geometry of the data and let the clipping of the pad (TGraph::kClipFrame, set
// by default) cut it, exactly like the band of a "F" TGraph.

#include "gtest/gtest.h"

#include "TAxis.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TROOT.h"

#include <cstddef>
#include <string>
#include <vector>

namespace {

/// Canvas recording the polygons handed to TPad::PaintFillArea(), which is how
/// TGraphPainter::PaintGraph() fills a graph drawn with the "F" option.
/// PaintGraphErrors() paints the error band with "F" as well, so the band of a
/// TGraphErrors and the band of an equivalent plain TGraph are recorded through
/// the very same code path and can be compared vertex by vertex.
class FillRecorder : public TCanvas {
public:
   FillRecorder(const char *name, Int_t expected)
      : TCanvas(name, name, 400, 400), fExpected(expected)
   {
   }
   ~FillRecorder() override = default;

   void PaintFillArea(Int_t n, Double_t *x, Double_t *y, Option_t *option = "") override
   {
      if (n == fExpected) {
         fX.assign(x, x + n);
         fY.assign(y, y + n);
         fFound = kTRUE;
      }
      TCanvas::PaintFillArea(n, x, y, option);
   }

   void Reset()
   {
      fX.clear();
      fY.clear();
      fFound = kFALSE;
   }

   Bool_t Found() const { return fFound; }
   const std::vector<Double_t> &X() const { return fX; }
   const std::vector<Double_t> &Y() const { return fY; }

private:
   Int_t fExpected;
   std::vector<Double_t> fX, fY;
   Bool_t fFound = kFALSE;
};

const Int_t kN = 5;
const Double_t kX[kN] = {0, 1, 2, 3, 4};
const Double_t kY[kN] = {0, 2, 4, 1, 3};
const Double_t kEY[kN] = {1, 0.5, 1, 0.5, 1};

/// The error band built explicitly as a TGraph, ie the reference of the
/// issue: the points (x, y + ey) followed by (x, y - ey) in reverse order. Its
/// shape does not depend on the axis range, only its visible part is clipped,
/// because the pad clips it to the frame.
TGraph *MakeReferenceBand()
{
   auto band = new TGraph(2 * kN);
   for (Int_t i = 0; i < kN; ++i) band->SetPoint(i, kX[i], kY[i] + kEY[i]);
   for (Int_t i = 0; i < kN; ++i) band->SetPoint(2 * kN - 1 - i, kX[i], kY[i] - kEY[i]);
   band->SetFillColor(kGreen);
   band->SetFillStyle(3005);
   return band;
}

TGraphErrors *MakeGraphErrors()
{
   const Double_t noXError[kN] = {0., 0., 0., 0., 0.};
   auto ge = new TGraphErrors(kN, kX, kY, noXError, kEY);
   ge->SetFillColor(kGreen);
   ge->SetFillStyle(3005);
   return ge;
}

int gCanvasId = 0;

} // namespace

// The band of a TGraphErrors must be the band of the data for every axis range:
// with a range wide enough to show all of it, and with a range cutting it, the
// painted band must always be the band of the reference TGraph.
TEST(TGraphErrors, band_geometry_does_not_depend_on_the_axis_range)
{
   gROOT->SetBatch(true);

   // uxmin, uxmax, uymin, uymax; the second range cuts the band on all sides
   const Double_t ranges[][4] = {{-0.5, 4.5, -2., 6.}, {0.5, 3.5, 0., 4.}};

   for (const auto &range : ranges) {
      const std::string cname = "testTGraphErrorsBand" + std::to_string(++gCanvasId);
      FillRecorder canvas(cname.c_str(), 2 * kN);
      canvas.cd();

      TGraphErrors *ge = MakeGraphErrors();
      ge->GetXaxis()->SetRangeUser(range[0], range[1]);
      ge->GetYaxis()->SetRangeUser(range[2], range[3]);

      canvas.Reset();
      ge->Draw("a3");
      ASSERT_TRUE(canvas.Found()) << "no error band painted";
      const std::vector<Double_t> bandX = canvas.X();
      const std::vector<Double_t> bandY = canvas.Y();

      TGraph *reference = MakeReferenceBand();
      canvas.Reset();
      reference->Draw("same F");
      ASSERT_TRUE(canvas.Found()) << "no reference band painted";

      ASSERT_EQ(bandX.size(), canvas.X().size());
      for (std::size_t i = 0; i < bandX.size(); ++i) {
         EXPECT_NEAR(bandX[i], canvas.X()[i], 1.e-9) << "x of vertex " << i;
         EXPECT_NEAR(bandY[i], canvas.Y()[i], 1.e-9) << "y of vertex " << i;
      }

      delete reference;
      delete ge;
   }
}
