/// \file
/// \ingroup tutorial_image
/// \notebook
/// Create a canvas and save as png.
///
/// \macro_image
/// \macro_code
///
/// \author Valeriy Onuchin

void pad2png()
{
   TCanvas *c = new TCanvas;
   TH1F *h = new TH1F("gaus", "gaus", 100, -5, 5);
   h->FillRandom("gaus", 10000);
   h->Draw();

   gSystem->ProcessEvents();

   TImage *img = TImage::Create();

   img->FromPad(c);

   img->WriteImage("canvas.png");
}
