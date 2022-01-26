/// \file
/// \ingroup tutorial_tree
///
/// This tutorial illustrates how to use the highlight mode with trees.
/// It first creates a TTree from a temperature data set in Prague between 1775
/// and 2004. Then it defines three pads representing the temperature per year,
/// month and day. Thanks to the highlight mechanism it is possible to explore the
/// data set only by moving the mouse on the plots. Movements on the years' plot
/// will update the months' and days' plot. Movements on the months plot will update
/// the days plot. Movements on the days' plot will display the exact temperature
/// for a given day.
///
/// \macro_code
///
/// \date March 2018
/// \author Jan Musinsky

Int_t year, month, day;
TTree *tree = nullptr;
TProfile *hYear = nullptr, *hMonth = nullptr, *hDay = nullptr;
TCanvas *Canvas = nullptr;
Int_t customhb = -2;
TLatex *info = nullptr;

// Ranges for year, month, day and temperature
Int_t rYear[3];   // from tree/data
Int_t rMonth[3]   = { 12, 1, 13 };
Int_t rDay[3]     = { 31, 1, 32 };
Double_t rTemp[3] = { 55.0, -20.0, 35.0 };

void HighlightDay(Int_t xhb)
{
   if (!info) {
      info = new TLatex();
      info->SetTextSizePixels(25);
      Canvas->cd(3);
      info->Draw();
      gPad->Update();
   }

   if (xhb != customhb) day = xhb;
   TString temp = TString::Format(" %5.1f #circC", hDay->GetBinContent(day));
   if (hDay->GetBinEntries(day) == 0) temp = " ";
   TString m = " ";
   if (month>0) m = TString::Format("-%02d",month);
   TString d = " ";
   if (day>0) d = TString::Format("-%02d",day);
   info->SetText(2.0, hDay->GetMinimum()*0.8, TString::Format("%4d%s%s%s", year, m.Data(), d.Data(), temp.Data()));
   Canvas->GetPad(3)->Modified();
}

void HighlightMonth(Int_t xhb)
{
   if (!hDay) {
      hDay = new TProfile("hDay", "; day; temp, #circC", rDay[0], rDay[1], rDay[2]);
      hDay->SetMinimum(rTemp[1]);
      hDay->SetMaximum(rTemp[2]);
      hDay->GetYaxis()->SetNdivisions(410);
      hDay->SetFillColor(kGray);
      hDay->SetMarkerStyle(kFullDotMedium);
      Canvas->cd(3);
      hDay->Draw("HIST, CP");
      gPad->Update();
      hDay->SetHighlight();
   }

   if (xhb != customhb) month = xhb;
   tree->Draw("T:DAY>>hDay", TString::Format("MONTH==%d && YEAR==%d", month, year), "goff");
   hDay->SetTitle(TString::Format("temperature by day (month = %02d, year = %d)", month, year));
   Canvas->GetPad(3)->Modified();

   HighlightDay(customhb); // custom call HighlightDay
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
      hMonth->SetMarkerStyle(kFullDotMedium);
      Canvas->cd(2)->SetGridx();
      hMonth->Draw("HIST, CP");
      gPad->Update();
      hMonth->SetHighlight();
   }

   year = xhb - 1 + rYear[1];
   tree->Draw("T:MONTH>>hMonth", TString::Format("YEAR==%d", year), "goff");
   hMonth->SetTitle(TString::Format("temperature by month (year = %d)", year));
   Canvas->GetPad(2)->Modified();

   HighlightMonth(customhb); // custom call HighlightMonth
}

void HighlightTemp(TVirtualPad *pad, TObject *obj, Int_t xhb, Int_t yhb)
{
   if (obj == hYear)  HighlightYear(xhb);
   if (obj == hMonth) HighlightMonth(xhb);
   if (obj == hDay)   HighlightDay(xhb);
   Canvas->Update();
}

void temperature()
{
   // Read file (data from Global Historical Climatology Network)
   tree = new TTree("tree", "GHCN-Daily");
   // data format: YEAR/I:MONTH/I:DAY/I:T/F

   // Read file $ROOTSYS/tutorials/tree/temperature_Prague.dat
   auto dir = gROOT->GetTutorialDir();
   dir.Append("/tree/");
   dir.ReplaceAll("/./","/");
   if (tree->ReadFile(Form("%stemperature_Prague.dat",dir.Data())) == 0) return;

   // Compute range of years
   tree->GetEntry(0);
   rYear[1] = (Int_t)tree->GetLeaf("YEAR")->GetValue(); // first year
   tree->GetEntry(tree->GetEntries() - 1);
   rYear[2] = (Int_t)tree->GetLeaf("YEAR")->GetValue(); // last year
   rYear[2] = rYear[2] + 1;
   rYear[0] = rYear[2] - rYear[1];

   // Create a TProfile for the average temperature by years
   hYear = new TProfile("hYear", "temperature (average) by year; year; temp, #circC", rYear[0], rYear[1], rYear[2]);
   tree->Draw("T:YEAR>>hYear", "", "goff");
   hYear->SetMaximum(hYear->GetMean(2)*1.50);
   hYear->SetMinimum(hYear->GetMean(2)*0.50);
   hYear->GetXaxis()->SetNdivisions(410);
   hYear->GetYaxis()->SetNdivisions(309);
   hYear->SetLineColor(kGray+2);
   hYear->SetMarkerStyle(8);
   hYear->SetMarkerSize(0.75);

   // Draw the average temperature by years
   gStyle->SetOptStat("em");
   Canvas = new TCanvas("Canvas", "Canvas", 0, 0, 700, 900);
   Canvas->HighlightConnect("HighlightTemp(TVirtualPad*,TObject*,Int_t,Int_t)");
   Canvas->Divide(1, 3, 0.001, 0.001);
   Canvas->cd(1);
   hYear->Draw("HIST, LP");
   gPad->Update();

   // Connect the highlight procedure to the temperature profile
   hYear->SetHighlight();
}
