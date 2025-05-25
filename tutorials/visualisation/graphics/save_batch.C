/// \file
/// \ingroup tutorial_graphics
/// This macro demonstrates batch image mode of web canvas
/// When enabled - several images converted into JSON before all together
/// provided to headless browser to produce image files. Let significantly
/// increase performance. Important - disable batch mode for flushing remaining images
///
/// \macro_code
///
/// \author Sergey Linev

void save_batch()
{
   // 37 canvases will be collected together for conversion
   TWebCanvas::BatchImageMode(37);

   auto c = new TCanvas("canvas", "Canvas with histogram");

   auto h1 = new TH1I("hist", "Histogram with random data", 100, -5., 5);
   h1->SetDirectory(nullptr);
   h1->FillRandom("gaus", 10000);
   h1->Draw();

   for (int n = 0; n < 100; ++n) {
      h1->FillRandom("gaus", 10000);
      c->SaveAs(TString::Format("batch_image_%03d.png", n));
   }

   // Important - disabling batch mode also flush remaining images
   TWebCanvas::BatchImageMode(0);
}
