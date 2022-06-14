/// \file
/// \ingroup tutorial_hist
/// \notebook
/// Fill a 1-D histogram from a parametric function.
///
/// \macro_image
/// \macro_code
///
/// \author Rene Brun

void fillrandom(int plot=1) {
   gBenchmark->Start("fillrandom");

   // A function (any dimension) or a formula may reference
   // an already defined formula
   auto form1 = new TFormula("form1","abs(sin(x)/x)");
   auto sqroot = new TF1("sqroot","x*gaus(0) + [3]*form1",0,10);
   sqroot->SetLineColor(4);
   sqroot->SetLineWidth(6);
   sqroot->SetParameters(10,4,1,20);

   // Create a one dimensional histogram (one float per bin)
   // and fill it following the distribution in function sqroot.
   auto h1f = new TH1F("h1f","Test random numbers",200,0,10);
   h1f->FillRandom("sqroot",10000);

   // Open a ROOT file and save the formula, function and histogram
   TFile myfile("fillrandom.root","RECREATE");
   form1->Write();
   sqroot->Write();
   h1f->Write();

   // Produce a plot if requested
   if (plot) {
      TCanvas *c1 = new TCanvas("c1","The FillRandom example",200,10,700,900);

      auto pad1 = new TPad("pad1","The pad with the function",0.05,0.50,0.95,0.95);
      auto pad2 = new TPad("pad2","The pad with the histogram",0.05,0.05,0.95,0.45);
      pad1->Draw();
      pad2->Draw();
      pad1->cd();

      pad1->SetGridx();
      pad1->SetGridy();
      pad1->GetFrame()->SetBorderMode(-1);
      pad1->GetFrame()->SetBorderSize(5);

      sqroot->Draw();
      auto lfunction = new TPaveLabel(5,39,9.8,46,"The sqroot function");
      lfunction->Draw();
      c1->Update();

      pad2->cd();
      pad2->GetFrame()->SetBorderMode(-1);
      pad2->GetFrame()->SetBorderSize(5);

      h1f->SetFillColor(45);
      h1f->Draw();
      c1->Update();
   }

   gBenchmark->Show("fillrandom");
}
