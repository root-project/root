
void testFriend2()
{
  TChain *ch = new TChain("ntp1");
  TChain *as = new TChain("ntp2");
  ch->Add("MC*.root");
  as->Add("MC*.root");
  ch->AddFriend("ntp2");

  ch->Draw("ntp2.runNumber"); // segmentation violation
}
