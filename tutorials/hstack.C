void hstack() {
// Example of stacked histograms: class THStack
//
//  Author: Rene Brun
   
   THStack *hs = new THStack("hs","test stacked histograms");
   //create three 1-d histograms
   TH1F *h1 = new TH1F("h1","test hstack",100,-4,4);
   h1->FillRandom("gaus",20000);
   h1->SetFillColor(kRed);
   h1->SetMarkerStyle(21);
   h1->SetMarkerColor(kRed);
   hs->Add(h1);
   TH1F *h2 = new TH1F("h2","test hstack",100,-4,4);
   h2->FillRandom("gaus",15000);
   h2->SetFillColor(kBlue);
   h2->SetMarkerStyle(21);
   h2->SetMarkerColor(kBlue);
   hs->Add(h2);
   TH1F *h3 = new TH1F("h3","test hstack",100,-4,4);
   h3->FillRandom("gaus",10000);
   h3->SetFillColor(kGreen);
   h3->SetMarkerStyle(21);
   h3->SetMarkerColor(kGreen);
   hs->Add(h3);
   
   TCanvas *c1 = new TCanvas("c1","stacked hists",10,10,1000,800);
   c1->SetFillColor(41);
   c1->Divide(2,2);
   // in top left pad, draw the stack with defaults
   c1->cd(1);
   hs->Draw();
   // in top right pad, draw the stack in non-stack mode and errors option
   c1->cd(2);
   gPad->SetGrid();
   hs->Draw("nostack,e1p");
   //in bottom left, draw in stack mode with "lego1" option
   c1->cd(3);
   gPad->SetFrameFillColor(17);
   gPad->SetTheta(3.77);
   gPad->SetPhi(2.9);
   hs->Draw("lego1");

   c1->cd(4);
   //create two 2-D histograms and draw them in stack mode
   gPad->SetFrameFillColor(17);
   THStack *a = new THStack("a","test legos");
   TF2 *f1 = new TF2("f1","xygaus + xygaus(5) + xylandau(10)",-4,4,-4,4);
   Double_t params[] = {130,-1.4,1.8,1.5,1, 150,2,0.5,-2,0.5, 3600,-2,0.7,-3,0.3};
   f1->SetParameters(params);
   TH2F *h2a = new TH2F("h2a","h2a",20,-4,4,20,-4,4);
   h2a->SetFillColor(38);
   h2a->FillRandom("f1",4000);
   TF2 *f2 = new TF2("f2","xygaus + xygaus(5)",-4,4,-4,4);
   Double_t params[] = {100,-1.4,1.9,1.1,2, 80,2,0.7,-2,0.5};
   f2->SetParameters(params);
   TH2F *h2b = new TH2F("h2b","h2b",20,-4,4,20,-4,4);
   h2b->SetFillColor(46);
   h2b->FillRandom("f2",3000);
   a->Add(h2a);
   a->Add(h2b);
   a->Draw();
}
