{
   TFile *_file0 = TFile::Open("CMSSW_3_1_0_pre11-RelValZTT-default-copy.root");
#if __clang__ || __GNUC__
   gSystem->AddIncludePath("-Wno-deprecated-declarations");
#endif
   gFile->MakeProject("cms310","*","RECREATE+");
}
