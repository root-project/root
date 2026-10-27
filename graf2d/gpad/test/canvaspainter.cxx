#include "gtest/gtest.h"

#include "TCanvas.h"
#include "TH1D.h"
#include "TROOT.h"
#include "TSystem.h"

#include <string>

TEST(TCanvasPainter, BatchHistogramDoesNotLoadImages)
{
   gROOT->SetBatch(kTRUE);
   ASSERT_EQ(std::string(gSystem->GetLibraries()).find("libASImage"), std::string::npos);
   TCanvas canvas("batch_painter", "batch painter");
   ASSERT_NE(canvas.GetCanvasPainter(), nullptr);

   TH1D histogram("histogram", "Batch histogram;X;Entries", 10, 0., 10.);
   histogram.Fill(5.);
   histogram.Draw();
   canvas.Update();
   EXPECT_FALSE(canvas.IsModified());
   EXPECT_EQ(std::string(gSystem->GetLibraries()).find("libASImage"), std::string::npos);
}
