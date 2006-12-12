void img2pad()
{
   // Display image in canvas and pad.
   //Author: valeriy Onuchin
   
   TImage *img = TImage::Open("rose512.jpg");
   if (!img) {
      printf("Could not create an image... exit\n");
      return;
   }
   img->SetConstRatio(kFALSE);
   img->Draw("N");

   TCanvas *c = gROOT->GetListOfCanvases()->FindObject("rose512jpg");
   c->SetFixedAspectRatio();

   TCanvas *c = new TCanvas("roses", "roses", 800, 800);
   img->Draw("T100,100,yellow");
   //img->Draw("T100,100,#556655");
   //img->Draw("T100,100");

   TImage *i1 = TImage::Open("rose512.jpg");
   i1->SetConstRatio(kFALSE);
   i1->Flip(90);
   TImage *i2 = TImage::Open("rose512.jpg");
   i2->SetConstRatio(kFALSE);
   i2->Flip(180);
   TImage *i3 = TImage::Open("rose512.jpg");
   i3->SetConstRatio(kFALSE);
   i3->Flip(270);
   TImage *i4 = TImage::Open("rose512.jpg");
   i4->SetConstRatio(kFALSE);
   i4->Mirror(kTRUE);

   float d = 0.40;
   TPad *p1 = new TPad("i1", "i1", 0.05, 0.55, 0.05+d*i1->GetWidth()/i1->GetHeight(), 0.95);
   TPad *p2 = new TPad("i2", "i2", 0.55, 0.55, 0.95, 0.55+d*i2->GetHeight()/i2->GetWidth());
   TPad *p3 = new TPad("i3", "i3", 0.55, 0.05, 0.55+d*i3->GetWidth()/i3->GetHeight(), 0.45);
   TPad *p4 = new TPad("i4", "i4", 0.05, 0.05, 0.45, 0.05+d*i4->GetHeight()/i4->GetWidth());

   p1->Draw();
   p1->cd();
   i1->Draw();
   c->cd();

   p2->Draw();
   p2->cd();
   i2->Draw();
   c->cd();

   p3->Draw();
   p3->cd();
   i3->Draw();
   c->cd();

   p4->Draw();
   p4->cd();
   i4->Draw();
   c->cd();
}
