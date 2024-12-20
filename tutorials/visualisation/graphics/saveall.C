/// \file
/// \ingroup tutorial_graphics
/// \notebook
/// Creates many canvases and save as images or pdf.
/// This macro creates 100 canvases and store them in different images files using TCanvas::SaveAll() method.
/// Demonstrated how different output format can be used in batch mode.
///
/// \macro_code
///
/// \author Sergey Linev

void saveall()
{
   gROOT->SetBatch(kTRUE); // enforce batch mode to avoid appearance of multiple canvas windows

   std::vector<TPad *> pads;

   for (int n = 0; n < 100; ++n) {
      auto c = new TCanvas(TString::Format("canvas%d", n), "Canvas with histogram");

      auto h1 = new TH1I(TString::Format("hist%d", n), "Histogram with random data", 100, -5., 5);
      h1->SetDirectory(nullptr);
      h1->FillRandom("gaus", 10000);

      h1->Draw();

      pads.push_back(c);
   }

   TCanvas::SaveAll(pads, "image%03d.png"); // create 100 PNG images

   TCanvas::SaveAll(pads, "image.svg"); // create 100 SVG images, %d pattern will be automatically append

   TCanvas::SaveAll(pads, "images.root"); // create single ROOT file with all canvases

   TCanvas::SaveAll(); // save all existing canvases in allcanvases.pdf file
}
