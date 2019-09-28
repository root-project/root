{
   auto f = TFile::Open("simplepair.root");
   if (!f)
      Fatal("execReadSimplePair","Could not open the ROOT file 'simplepair.root'");
   auto p = f->Get("pair");
   if (!p)
      Fatal("execReadSimplePair","Could not read the pair ...");
   return 0;
}
