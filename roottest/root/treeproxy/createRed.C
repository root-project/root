{
   TFile *f = TFile::Open("reduced.N00008257_0002.spill.sntp.R1_18.0.root");
   TTree *t; f->GetObject("cnt",t);
   t->MakeProxy("red","sum.C");
}