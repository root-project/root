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
   TCanvas *c = new TCanvas("c1", "Creating image from histogram drawing", 800, 600);
   TH1F *h = new TH1F("gaus", "gaus", 100, -5, 5);
   h->FillRandom("gaus", 10000);
   c->Add(h);
   c->Update();

   std::unique_ptr<TImage> img(TImage::Create());
   img->FromPad(c);
   img->WriteImage("canvas.png");
}
