{
   TFile *_file0 = TFile::Open("emulvector.root");
#ifdef ClingWorkAroundMissingDynamicScope
   TTree*tr; _file0->GetObject("tr",tr);
#endif
   tr->Scan();
   return 0;
}
