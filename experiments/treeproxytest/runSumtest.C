{
  // usage: 
  // root[0] .x testsumenergy.C
  //
  // tested with root cvs development version dated August 18, 2004,
  // but root files generated previous to this date can be used as input.

gSystem -> Load("libTreePlayer");

remove("neutrino_pc.txt");
remove("antineu_pc.txt");

TChain chainmc ("NtpMC");
chainmc.Add("n14301001_0000.sntp.R1.12.root");


TChain chainsr ("NtpSR");
chainsr.Add("n14301001_0000.sntp.R1.12.root");


TChain chainth ("NtpTH");
chainth.Add("n14301001_0000.sntp.R1.12.root");


chainmc.AddFriend("NtpSR");
chainmc.AddFriend("NtpTH");

cout<<"came till here "<<endl;

//These two lines will produce the same results as the Draw statement above

chainmc.MakeProxy("withfriend","sum.C","","");

cout<<"reached here"<<endl;

chainmc.Process("withfriend.h+");


}

