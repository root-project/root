// https://root.cern.ch/root/html528/THistPainter.html
// https://root.cern.ch/root/htmldoc/guides/users-guide/Histograms.html#the-bar-options

{
   TCanvas *cb = new TCanvas("cb", "cb", 600, 400);
   cb->SetGrid();
   gStyle->SetHistMinimumZero();

   TFile f("reduceall.root", "READ");
   Double_t maxvalue = std::max(h2->GetMaximum(), h4->GetMaximum());
   maxvalue = std::max(maxvalue, h8->GetMaximum()) + 20;
   h2->SetBarWidth(0.1);
   h2->SetBarOffset(0.1);
   h2->SetStats(0);
   h2->SetMinimum(0);
   h2->SetMaximum(maxvalue);
   h2->Draw("b");

   h4->SetBarWidth(0.1);
   h4->SetBarOffset(0.2);
   h4->SetStats(0);
   h4->SetMinimum(0);
   h4->SetMaximum(maxvalue);
   h4->Draw("b same");

   h8->SetBarWidth(0.1);
   h8->SetBarOffset(0.3);
   h8->SetStats(0);
   h8->SetMinimum(0);
   h8->SetMaximum(maxvalue);
   h8->Draw("b same");

   TLegend *legend = new TLegend(0.12709, 0.68984, 0.337793, 0.858289);
   legend->AddEntry(h2, "2 processors", "f");
   legend->AddEntry(h4, "4 processors", "f");
   legend->AddEntry(h8, "8 processors", "f");
   legend->Draw();

   return cb;
}
