{
#ifdef ClingWorkAroundIncorrectTearDownOrder
   if (1) {
#endif
   TFile f("prova.root");
#ifdef ClingWorkAroundMissingDynamicScope
   TTree *TreeEq; f.GetObject("TreeEq",TreeEq);
#endif
   TreeEq->Draw("sectors.fStrips.fHists.size()");
   TreeEq->Draw("sectors.fStrips.fHists.GetNbinsX()");
#ifdef ClingWorkAroundIncorrectTearDownOrder
   }
#endif
}
