void glbox()
{
// Display a 3D histogram using GL (box option).
//Author: Timur Pocheptsov
   gStyle->SetCanvasPreferGL(kTRUE);
   TCanvas *c        = new TCanvas("glc","TH3 Drawing", 100, 10, 850, 400);
   TPaveLabel *title = new TPaveLabel(0.04, 0.86, 0.96, 0.98,
                           "\"glbox\" and \"glbox1\" options for TH3.");
   title->SetFillColor(32);
   title->Draw();

   TPad *boxPad  = new TPad("box", "box", 0.02, 0.02, 0.48, 0.82);
   TPad *box1Pad = new TPad("box1", "box1", 0.52, 0.02, 0.98, 0.82);
   boxPad->Draw();
   box1Pad->Draw();
   
   TH3F *h31 = new TH3F("h31", "h31", 10, -1, 1, 10, -1, 1, 10, -1, 1);
   h31->FillRandom("gaus");
   h31->SetFillColor(2);
   boxPad->cd();
   h31->Draw("glbox");

   TH3F *h32 = new TH3F("h32", "h32", 10, -2, 2, 10, -1, 1, 10, -3, 3);
   h32->FillRandom("gaus");
   h32->SetFillColor(4);
   box1Pad->cd();
   h32->Draw("glbox1");
}
