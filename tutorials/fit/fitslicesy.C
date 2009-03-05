void fitslicesy() {
//
// Illustrates how to use the TH1::FitSlicesY function
// To see the output of this macro, click begin_html <a href="gif/fitslicesy.gif" >here</a> end_html
//    It uses the TH2F histogram generated in macro hsimple.C
//    It invokes FitSlicesY and draw the fitted "mean" and "sigma"
//    in 2 sepate pads.
//    This macro shows also how to annotate a picture, change
//    some pad parameters.
//Author: Rene Brun
   
// Change some default parameters in the current style
   gStyle->SetLabelSize(0.06,"x");
   gStyle->SetLabelSize(0.06,"y");
   gStyle->SetFrameFillColor(38);
   gStyle->SetTitleW(0.6);
   gStyle->SetTitleH(0.1);

// Connect the input file and get the 2-d histogram in memory
   TString dir = gSystem->UnixPathName(gInterpreter->GetCurrentMacroName());
   dir.ReplaceAll("fitslicesy.C","../hsimple.C");
   dir.ReplaceAll("/./","/");
   if (!gInterpreter->IsLoaded(dir.Data())) gInterpreter->LoadMacro(dir.Data());
   TFile *hsimple = (TFile*)gROOT->ProcessLineFast("hsimple(1)");
   if (!hsimple) return;
   TH2F *hpxpy = (TH2F*)hsimple->Get("hpxpy");

// Create a canvas and divide it
   TCanvas *c1 = new TCanvas("c1","c1",700,500);
   c1->SetFillColor(42);
   c1->Divide(2,1);
   c1->cd(1);
   TPad *left = (TPad*)gPad;
   left->Divide(1,2);

// Draw 2-d original histogram
   left->cd(1);
   gPad->SetTopMargin(0.12);
   gPad->SetFillColor(33);
   hpxpy->Draw();
   hpxpy->GetXaxis()->SetLabelSize(0.06);
   hpxpy->GetYaxis()->SetLabelSize(0.06);
   hpxpy->SetMarkerColor(kYellow);

// Fit slices projected along Y fron bins in X [7,32] with more than 20 bins  in Y filled
   hpxpy->FitSlicesY(0,7,32,20);

// Show fitted "mean" for each slice
   left->cd(2);
   gPad->SetFillColor(33);
   hpxpy_0->Draw();
   c1->cd(2);
   TPad *right = (TPad*)gPad;
   right->Divide(1,2);
   right->cd(1);
   gPad->SetTopMargin(0.12);
   gPad->SetLeftMargin(0.15);
   gPad->SetFillColor(33);
   hpxpy_1->Draw();

// Show fitted "sigma" for each slice
   right->cd(2);
   gPad->SetTopMargin(0.12);
   gPad->SetLeftMargin(0.15);
   gPad->SetFillColor(33);
   hpxpy_2->SetMinimum(0.8);
   hpxpy_2->Draw();

//attributes
   hpxpy_0->SetLineColor(kYellow);
   hpxpy_1->SetLineColor(kYellow);
   hpxpy_2->SetLineColor(kYellow);
   hpxpy_0->SetMarkerColor(kRed);
   hpxpy_1->SetMarkerColor(kRed);
   hpxpy_2->SetMarkerColor(kRed);
   hpxpy_0->SetMarkerStyle(21);
   hpxpy_1->SetMarkerStyle(21);
   hpxpy_2->SetMarkerStyle(21);
   hpxpy_0->SetMarkerSize(0.6);
   hpxpy_1->SetMarkerSize(0.6);
   hpxpy_2->SetMarkerSize(0.6);
}
