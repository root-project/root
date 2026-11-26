#include "gtest/gtest.h"

#include "TCanvas.h"
#include "TH1F.h"
#include "TLatex.h"
#include "TROOT.h"
#include "TString.h"
#include "TSystem.h"

// Drawing and saving a canvas in batch mode must not load libraries that are
// only needed to display graphics on screen (pad painter, image, X11, ...).
TEST(TCanvas, BatchDoesNotLoadGraphicsLibraries)
{
   gROOT->SetBatch(kTRUE);

   {
      TCanvas c("c_batchlibs", "c_batchlibs", 600, 400);
      TH1F h("h_batchlibs", "Histogram;x;y", 10, 0, 10);
      for (int i = 0; i < 10; ++i)
         h.Fill(i, i);
      h.Draw();
      TLatex latex(0.2, 0.8, "#alpha + #beta = #gamma");
      latex.SetNDC();
      latex.Draw();
      c.Update();

      for (const char *ext : {"pdf", "svg", "C"}) {
         TString fname = TString::Format("batchlibs.%s", ext);
         c.SaveAs(fname);
         EXPECT_EQ(gSystem->AccessPathName(fname), kFALSE) << fname << " was not created";
         gSystem->Unlink(fname);
      }
   }

   const TString libs = gSystem->GetLibraries();
   TString loaded;
   for (const char *lib : {"ROOTPadPainter", "ASImage", "ASImageUtils", "GX11", "GX11TTF", "GQuartz", "X11", "gif"}) {
      if (libs.Index(TString::Format("lib%s.", lib)) != kNPOS)
         loaded += TString::Format(" lib%s", lib);
   }
   EXPECT_TRUE(loaded.IsNull()) << "Libraries loaded in batch mode:" << loaded << "\nAll loaded libraries: " << libs;
}
