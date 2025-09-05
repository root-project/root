
void testFriend3()
{
  TChain *ch = new TChain("ntp1");
  TChain *as = new TChain("ntp2");
  ch->Add("MC2*.root");
  as->Add("MC2*.root");

  ch->Draw("ntp2.runNumber"); // segmentation violation

  ch->AddFriend("ntp2");

// Intentionally empty
  ch->Draw("ntp2.runNumber"); // segmentation violation
}
