{
// Fill out the code of the actual test
   gROOT->ProcessLine(".L loadcode.C+");
   jetwrite();
   TFile *_file0 = TFile::Open("jetclass.root");
   new TCanvas;
   t.Draw("jet[].v_x[jet.nbest]","jet.nbest>=0");
   new TCanvas;
   t.Draw("jet[].v_x[jet.nbest+(jet.nbest==-1)]","");
   return 0;
}
