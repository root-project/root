{
   TFile *_file0 = TFile::Open("crashed.root");
#ifdef ClingWorkAroundBrokenUnnamedReturn
   if (_file0 == 0) {
      gApplication->Terminate(0);
   } else {
      gApplication->Terminate(1);
   }
#else
   if (_file0 == 0) {
      return 0;
   } else {
      return 1;
   }
#endif
}
