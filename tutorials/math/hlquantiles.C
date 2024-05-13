/// \file
/// \ingroup tutorial_math
/// Demo for quantiles (with highlight mode)
///
/// \macro_image
/// \macro_code
///
/// \authors Rene Brun, Eddy Offermann, Jan Musinsky

TList *lq = nullptr;
TGraph *gr = nullptr;

void HighlightQuantile(TVirtualPad *pad, TObject *obj, Int_t ihp, Int_t y)
{
   // show the evolution of all quantiles in the bottom pad
   if (obj != gr) return;
   if (ihp == -1) return;

   TVirtualPad *savepad = gPad;
   pad->GetCanvas()->cd(3);
   lq->At(ihp)->Draw("alp");
   gPad->Update();
   if (savepad) savepad->cd();
}


void hlquantiles() {
   const Int_t nq = 100;
   const Int_t nshots = 10;
   Double_t xq[nq];  // position where to compute the quantiles in [0,1]
   Double_t yq[nq];  // array to contain the quantiles
   for (Int_t i=0;i<nq;i++) xq[i] = Float_t(i+1)/nq;

   TGraph *gr70 = new TGraph(nshots);
   TGraph *gr90 = new TGraph(nshots);
   TGraph *gr98 = new TGraph(nshots);
   TGraph *grq[nq];
   for (Int_t ig = 0; ig < nq; ig++)
      grq[ig] = new TGraph(nshots);
   TH1F *h = new TH1F("h","demo quantiles",50,-3,3);

   for (Int_t shot=0;shot<nshots;shot++) {
      h->FillRandom("gaus",50);
      h->GetQuantiles(nq,yq,xq);
      gr70->SetPoint(shot,shot+1,yq[70]);
      gr90->SetPoint(shot,shot+1,yq[90]);
      gr98->SetPoint(shot,shot+1,yq[98]);
      for (Int_t ig = 0; ig < nq; ig++)
         grq[ig]->SetPoint(shot,shot+1,yq[ig]);
   }

   //show the original histogram in the top pad
   TCanvas *c1 = new TCanvas("c1","demo quantiles",10,10,600,900);
   c1->HighlightConnect("HighlightQuantile(TVirtualPad*,TObject*,Int_t,Int_t)");
   c1->SetFillColor(41);
   c1->Divide(1,3);
   c1->cd(1);
   h->SetFillColor(38);
   h->Draw();

   // show the final quantiles in the middle pad
   c1->cd(2);
   gPad->SetFrameFillColor(33);
   gPad->SetGrid();
   gr = new TGraph(nq,xq,yq);
   gr->SetTitle("final quantiles");
   gr->SetMarkerStyle(21);
   gr->SetMarkerColor(kRed);
   gr->SetMarkerSize(0.3);
   gr->Draw("ap");

   // prepare quantiles
   lq = new TList();
   for (Int_t ig = 0; ig < nq; ig++) {
      grq[ig]->SetMinimum(gr->GetYaxis()->GetXmin());
      grq[ig]->SetMaximum(gr->GetYaxis()->GetXmax());
      grq[ig]->SetMarkerStyle(23);
      grq[ig]->SetMarkerColor(ig%100);
      grq[ig]->SetTitle(TString::Format("q%02d", ig));
      lq->Add(grq[ig]);
   }

   TText *info = new TText(0.1, 2.4, "please move the mouse over the graph");
   info->SetTextSize(0.08);
   info->SetTextColor(gr->GetMarkerColor());
   info->SetBit(kCannotPick);
   info->Draw();

   gr->SetHighlight();

   // show the evolution of some  quantiles in the bottom pad
   c1->cd(3);
   gPad->SetFrameFillColor(17);
   gPad->DrawFrame(0,0,nshots+1,3.2);
   gPad->SetGrid();
   gr98->SetMarkerStyle(22);
   gr98->SetMarkerColor(kRed);
   gr98->Draw("lp");
   gr90->SetMarkerStyle(21);
   gr90->SetMarkerColor(kBlue);
   gr90->Draw("lp");
   gr70->SetMarkerStyle(20);
   gr70->SetMarkerColor(kMagenta);
   gr70->Draw("lp");
   // add a legend
   TLegend *legend = new TLegend(0.85,0.74,0.95,0.95);
   legend->SetTextFont(72);
   legend->SetTextSize(0.05);
   legend->AddEntry(gr98," q98","lp");
   legend->AddEntry(gr90," q90","lp");
   legend->AddEntry(gr70," q70","lp");
   legend->Draw();
}

