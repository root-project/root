{
   TFile *_file0 = TFile::Open("alice_ESDs.root");
   gFile->MakeProject("aliceesd","*","RECREATE+");
}
