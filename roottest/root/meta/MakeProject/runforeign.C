{
   TFile *_file0 = TFile::Open("foreign.root");
   gFile->MakeProject("foreign","*","RECREATE++");
}
