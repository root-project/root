/// \file
/// \ingroup tutorial_hist
/// \notebook
/// Example showing an histogram with reverse axis.
///
/// \macro_image
/// \macro_code
///
/// \author Olivier Couet

void ReverseXAxis (TH1 *h);
void ReverseYAxis (TH1 *h);

void reverseaxis()
{
   TH2F *hpxpy  = new TH2F("hpxpy","py vs px",40,-4,4,40,-4,4);
   Float_t px, py;
   TRandom r;
   for (Int_t i = 0; i < 25000; i++) {
      r.Rannor(px,py);
      hpxpy->Fill(px,py);
   }
   TCanvas *c1 = new TCanvas("c1");
   hpxpy->Draw("colz");
   ReverseXAxis(hpxpy);
   ReverseYAxis(hpxpy);
}

void ReverseXAxis(TH1 *h)
{
   // Remove the current axis
   h->GetXaxis()->SetLabelOffset(999);
   h->GetXaxis()->SetTickLength(0);

   // Redraw the new axis
   gPad->Update();
   TGaxis *newaxis = new TGaxis(gPad->GetUxmax(),
                                gPad->GetUymin(),
                                gPad->GetUxmin(),
                                gPad->GetUymin(),
                                h->GetXaxis()->GetXmin(),
                                h->GetXaxis()->GetXmax(),
                                510,"-");
   newaxis->SetLabelOffset(-0.03);
   newaxis->Draw();
}

void ReverseYAxis(TH1 *h)
{
   // Remove the current axis
   h->GetYaxis()->SetLabelOffset(999);
   h->GetYaxis()->SetTickLength(0);

   // Redraw the new axis
   gPad->Update();
   TGaxis *newaxis = new TGaxis(gPad->GetUxmin(),
                                gPad->GetUymax(),
                                gPad->GetUxmin()-0.001,
                                gPad->GetUymin(),
                                h->GetYaxis()->GetXmin(),
                                h->GetYaxis()->GetXmax(),
                                510,"+");
   newaxis->SetLabelOffset(-0.03);
   newaxis->Draw();
}
