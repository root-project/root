{
   TChain chain("h42");
//   chain.Add("./dstar.root");  //  21330730 bytes  21920 events
//   chain.Add("/home/CSC/dstarmb.root");  //  21330730 bytes  21920 events
   chain.Add("$H1/dstarmb.root");  //  21330730 bytes  21920 events
   chain.Add("$H1/dstarp1a.root"); //  71464503 bytes  73243 events
   chain.Add("$H1/dstarp1b.root"); //  83827959 bytes  85597 events
//   chain.Add("$H1/dstarp2.root");  // 100675234 bytes 103053 events

// was genereated using:
// TGenerateProxy tp(&chain,"h1analysis.C","h1analysisCut.C","h1sel2",99)
// gROOT->ProcessLine(".L h1sel2.h+");
// h1sel2 sel;

gSystem->Load("libTreePlayer.so");
gROOT->Time();      
cout << "run:\nROOT::draw(&chain,\"h1analysis.C+\",\"h1analysisCut.C\"); h1analysis_Terminate();\n";
cout << "compare to:\nchain.Process(\"$ROOTSYS/tutorials/h1analysis.C+\");\n";

}
