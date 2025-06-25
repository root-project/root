{
   auto f = TFile::Open("geodemo.root");
   gSystem->cd("..");
   if (f && (449262 == f->GetSize()))
      return 0;
   else
      return 1;
}
