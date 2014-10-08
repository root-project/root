TCanvas *statsEditing() {
// This example shows:
//    - how to remove a stat element from the stat box
//    - how to add a new one
//
//  Author: Olivier Couet

   // Create and plot a test histogram with stats
   TCanvas *se = new TCanvas;
   TH1F *h = new TH1F("h","test",100,-3,3);
   h->FillRandom("gaus",3000);
   gStyle->SetOptStat();
   h->Draw();
   se->Update();

   // Retrieve the stat box
   TPaveStats *ps = (TPaveStats*)se->GetPrimitive("stats");
   ps->SetName("mystats");
   TList *list = ps->GetListOfLines();

   // Remove the RMS line
   TText *tconst = ps->GetLineWith("RMS");
   list->Remove(tconst);

   // Add a new line in the stat box.
   // Note that "=" is a control character
   TLatex *myt = new TLatex(0,0,"Test = 10");
   myt ->SetTextFont(42);
   myt ->SetTextSize(0.04);
   myt ->SetTextColor(kRed);
   list->Add(myt);

   // the following line is needed to avoid that the automatic redrawing of stats
   h->SetStats(0);

   se->Modified();
   return se;
}
