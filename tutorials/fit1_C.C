
   gROOT->Reset();
   c1 = new TCanvas("c1","The Fit Canvas");
   c1->SetGridx();
   c1->SetGridy();
   TFile fill("fillrandom.root");
   fill.ls();
   sqroot->Print();

   h1f->Fit("sqroot");

   fitlabel = new TPaveText(0.6,0.3,0.9,0.80,"NDC");
   fitlabel->SetTextAlign(12);
   fitlabel->SetFillColor(25);
   fitlabel->ReadFile("fit1_C.C");
   fitlabel->Draw();

