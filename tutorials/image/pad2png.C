void pad2png()
{
   // Create a canvas and save as png.
   //Author: Valeriy Onuchin

   TCanvas *c = new TCanvas;
   TH1F *h = new TH1F("gaus", "gaus", 100, -5, 5);
   h->FillRandom("gaus", 10000);
   h->Draw();

   gSystem->ProcessEvents();

   TImage *img = TImage::Create();

   //img->FromPad(c, 10, 10, 300, 200);
   img->FromPad(c);

   img->WriteImage("canvas.png");

   delete h;
   delete c;
   delete img;
}
