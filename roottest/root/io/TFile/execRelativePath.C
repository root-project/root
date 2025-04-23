{
   auto f = new TFile("geodemo.root");
   gSystem->cd("..");
   if (449262 == f->GetSize())
      return 0;
   else
      return 1;
}
