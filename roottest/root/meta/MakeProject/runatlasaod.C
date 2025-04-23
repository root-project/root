{
   TFile *_file0 = TFile::Open("small_aod.pool.root");
   gFile->MakeProject("small_aod","*","RECREATE+");
}
