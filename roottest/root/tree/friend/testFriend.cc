void testFriend()
{
  TChain ch("ntp1");
  TChain as("ntp2");
  ch.Add("MC*.root");
  as.Add("MC*.root");
  ch.AddFriend("ntp2","ntp2");

  ch.Draw("ntp2.runNumber"); // segmentation violation

  int a;
  //scanf("%d",&a);
}
