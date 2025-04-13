{
#ifdef ClingWorkAroundIncorrectTearDownOrder
   if (1) {
#endif
TCanvas *c = new TCanvas;
TChain ch("tRawData");
ch.Add("dat_001.root");
ch.Add("dat_002.root");
ch.Add("dat_003.root");

ch.Draw("Q2");
ch.Draw("Q2shift");

TChain ch2("tCorrectedData");
ch2.Add("dat_001.root");
ch2.Add("dat_002.root");
ch2.Add("dat_003.root");

ch2.Draw("Q2shift"); 

ch.AddFriend(&ch2);
ch.Draw("Q2shift");
#ifdef ClingWorkAroundIncorrectTearDownOrder
   }
#endif
}

