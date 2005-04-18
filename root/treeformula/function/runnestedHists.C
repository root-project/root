{
   TFile f("prova.root");
   TreeEq->Draw("sectors.fStrips.fHists.size()");
   TreeEq->Draw("sectors.fStrips.fHists.GetNbinsX()");
}
