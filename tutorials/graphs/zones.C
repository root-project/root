/// \file
/// \ingroup tutorial_graphs
/// \notebook
/// Example of script showing how to divide a canvas
/// into adjacent subpads + axis labels on the top and right side
/// of the pads.
///
/// \macro_image
/// \macro_code
///
/// \author Rene Brun

void zones() {
   TCanvas *c1 = new TCanvas("c1","multipads",900,700);
   gStyle->SetOptStat(0);
   c1->Divide(2,2,0,0);
   TH2F *h1 = new TH2F("h1","test1",10,0,1,20,0,20);
   TH2F *h2 = new TH2F("h2","test2",10,0,1,20,0,100);
   TH2F *h3 = new TH2F("h3","test3",10,0,1,20,-1,1);
   TH2F *h4 = new TH2F("h4","test4",10,0,1,20,0,1000);

   c1->cd(1);
   gPad->SetTickx(2);
   h1->Draw();

   c1->cd(2);
   gPad->SetTickx(2);
   gPad->SetTicky(2);
   h2->GetYaxis()->SetLabelOffset(0.01);
   h2->Draw();

   c1->cd(3);
   h3->Draw();

   c1->cd(4);
   gPad->SetTicky(2);
   h4->Draw();
}
