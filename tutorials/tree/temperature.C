/// \file
/// \ingroup tutorial_tree
/// Demo for highlight mode
///
/// \macro_code
///
/// \date March 2018
/// \author Jan Musinsky

#include <TTree.h>
#include <TLeaf.h>
#include <TProfile.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TLatex.h>

Int_t year, month, day;
TTree *tree;
TProfile *hYear = 0, *hMonth = 0, *hDay = 0;
TCanvas *c1;
Int_t customhb = -2;
TLatex *info = 0;

// ranges
Int_t rYear[3];   // from tree/data
Int_t rMonth[3]   = { 12, 1, 13 };
Int_t rDay[3]     = { 31, 1, 32 };
Double_t rTemp[3] = { 55.0, -20.0, 35.0 };

void HighlightTemp(TVirtualPad *pad, TObject *obj, Int_t xhb, Int_t yhb);
void HighlightYear(Int_t xhb);
void HighlightMonth(Int_t xhb);
void HighlightDay(Int_t xhb);

void temperature()
{
   // read file (data from Global Historical Climatology Network)
   tree = new TTree("tree", "GHCN-Daily");
   // data format: YEAR/I:MONTH/I:DAY/I:T/F

   // read file $ROOTSYS/tutorials/tree/temperature_Prague.dat
   TString dir = gROOT->GetTutorialDir();
   dir.Append("/tree/");
   dir.ReplaceAll("/./","/");
   if (tree->ReadFile(Form("%stemperature_Prague.dat",dir.Data())) == 0) return;

   // range of years
   tree->GetEntry(0);
   rYear[1] = (Int_t)tree->GetLeaf("YEAR")->GetValue(); // first year
   tree->GetEntry(tree->GetEntries() - 1);
   rYear[2] = (Int_t)tree->GetLeaf("YEAR")->GetValue(); // last year
   rYear[2] = rYear[2] + 1;
   rYear[0] = rYear[2] - rYear[1];

   // temp by years
   hYear = new TProfile("hYear", "temperature (average) by year; year; temp, #circC", rYear[0], rYear[1], rYear[2]);
   tree->Draw("T:YEAR>>hYear", "", "goff");
   hYear->SetMaximum(hYear->GetMean(2)*1.50);
   hYear->SetMinimum(hYear->GetMean(2)*0.50);
   hYear->GetXaxis()->SetNdivisions(410);
   hYear->GetYaxis()->SetNdivisions(309);
   hYear->SetLineColor(kGray+2);
   hYear->SetMarkerStyle(8);
   hYear->SetMarkerSize(0.75);

   // draw
   gStyle->SetOptStat("em");
   c1 = new TCanvas("c1", "c1", 0, 0, 700, 900);
   c1->Divide(1, 3, 0.001, 0.001);
   c1->cd(1);
   hYear->Draw("HIST, LP");
   gPad->Update();

   // highlight
   hYear->SetHighlight();
   c1->HighlightConnect("HighlightTemp(TVirtualPad*,TObject*,Int_t,Int_t)");
}

void HighlightTemp(TVirtualPad *pad, TObject *obj, Int_t xhb, Int_t yhb)
{
   if (obj == hYear)  HighlightYear(xhb);
   if (obj == hMonth) HighlightMonth(xhb);
   if (obj == hDay)   HighlightDay(xhb);
   c1->Update();
}

void HighlightYear(Int_t xhb)
{
   if (!hMonth) {
      hMonth = new TProfile("hMonth", "; month; temp, #circC", rMonth[0], rMonth[1], rMonth[2]);
      hMonth->SetMinimum(rTemp[1]);
      hMonth->SetMaximum(rTemp[2]);
      hMonth->GetXaxis()->SetNdivisions(112);
      hMonth->GetXaxis()->CenterLabels();
      hMonth->GetYaxis()->SetNdivisions(410);
      hMonth->SetFillColor(kGray+1);
      hMonth->SetMarkerStyle(7);
      c1->cd(2)->SetGridx();
      hMonth->Draw("HIST, CP");
      gPad->Update();
      hMonth->SetHighlight();
   }

   year = xhb - 1 + rYear[1];
   tree->Draw("T:MONTH>>hMonth", TString::Format("YEAR==%d", year), "goff");
   hMonth->SetTitle(TString::Format("temperature by month (year = %d)", year));
   c1->GetPad(2)->Modified();

   HighlightMonth(customhb); // custom call HighlightMonth
}

void HighlightMonth(Int_t xhb)
{
   if (!hDay) {
      hDay = new TProfile("hDay", "; day; temp, #circC", rDay[0], rDay[1], rDay[2]);
      hDay->SetMinimum(rTemp[1]);
      hDay->SetMaximum(rTemp[2]);
      hDay->GetYaxis()->SetNdivisions(410);
      hDay->SetFillColor(kGray);
      hDay->SetMarkerStyle(7);
      c1->cd(3);
      hDay->Draw("HIST, CP");
      gPad->Update();
      hDay->SetHighlight();
   }

   if (xhb != customhb) month = xhb;
   tree->Draw("T:DAY>>hDay", TString::Format("MONTH==%d && YEAR==%d", month, year), "goff");
   hDay->SetTitle(TString::Format("temperature by day (month = %02d, year = %d)", month, year));
   c1->GetPad(3)->Modified();

   HighlightDay(customhb); // custom call HighlightDay
}

void HighlightDay(Int_t xhb)
{
   if (!info) {
      info = new TLatex();
      info->SetTextSizePixels(25);
      c1->cd(3);
      info->Draw();
      gPad->Update();
   }

   if (xhb != customhb) day = xhb;
   TString temp = TString::Format(" %5.1f #circC", hDay->GetBinContent(day));
   if (hDay->GetBinEntries(day) == 0) temp = " none";
   info->SetText(12.0, hDay->GetMinimum()*0.8, TString::Format("%4d-%02d-%02d %s", year, month, day, temp.Data()));
   c1->GetPad(3)->Modified();
}
