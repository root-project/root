// Example of bar charts
// Author: Rene Brun
void hbars()
{
   if (gSystem->AccessPathName("cernstaff.root")) {
      gROOT->ProcessLine(".x cernbuild.C");
   }
   TFile *f = new TFile("cernstaff.root");
   TTree *T = (TTree*)f->Get("T");
   T->SetFillColor(45);
   TCanvas *c1 = new TCanvas("c1","histograms with bars",10,10,800,900);
   c1->SetFillColor(42);
   c1->Divide(1,2);
   
   //horizontal bar chart
   c1->cd(1); gPad->SetGrid(); gPad->SetLogx(); gPad->SetFrameFillColor(33);
   T->Draw("Nation","","hbar2");
   
   //vertical bar chart
   c1->cd(2); gPad->SetGrid(); gPad->SetFrameFillColor(33);
   T->Draw("Division>>hDiv","","goff");
   TH1F *hDiv   = (TH1F*)gDirectory->Get("hDiv");
   hDiv->SetStats(0);
   TH1F *hDivFR = (TH1F*)hDiv->Clone("hDivFR");
   T->Draw("Division>>hDivFR","Nation==\"FR\"","goff");
   hDiv->SetBarWidth(0.45);
   hDiv->SetBarOffset(0.1);
   hDiv->SetFillColor(49);
   hDiv->Draw("bar2");
   hDivFR->SetBarWidth(0.4);
   hDivFR->SetBarOffset(0.55);
   hDivFR->SetFillColor(50);
   hDivFR->Draw("bar2,same");
  
   TLegend *legend = new TLegend(0.55,0.65,0.76,0.82);
   legend->AddEntry(hDiv,"All nations","f");
   legend->AddEntry(hDivFR,"French only","f");
   legend->Draw();
   
   c1->cd();
}
