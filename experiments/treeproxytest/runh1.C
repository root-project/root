{
   TChain chain("h42");
//   chain.Add("$H1/dstar.root");  //  21330730 bytes  21920 events
//   chain.Add("/home/CSC/dstarmb.root");  //  21330730 bytes  21920 events
   chain.Add("$H1/dstarmb.root");  //  21330730 bytes  21920 events
   chain.Add("$H1/dstarp1a.root"); //  71464503 bytes  73243 events
   chain.Add("$H1/dstarp1b.root"); //  83827959 bytes  85597 events
//   chain.Add("$H1/dstarp2.root");  // 100675234 bytes 103053 events

   //We have to use ACLiC for now because of a limitiation in CINT
   ROOT::draw(&chain,"h1analysis.C+","h1analysisCut.C"); 
   h1analysis_Terminate();
}
