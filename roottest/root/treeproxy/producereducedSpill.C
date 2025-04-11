{
// Fill out the code of the actual test
   TFile *_file0 = TFile::Open("reduced.N00008257_0002.spill.sntp.R1_18.0.root");
   TTree *cnt;
   _file0->GetObject("cnt",cnt);
   gSystem->Unlink("red.h");
   cnt->MakeProxy("red","sum.C");
}
