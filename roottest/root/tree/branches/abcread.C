void abcread(const char* mode, const char* what)
{
   TFile f(TString::Format("abc_%s.root", mode));
   TTree* t = 0;
   f.GetObject("tree", t);
   Holder* h = 0;
   t->SetBranchAddress("h", &h);
   int e = 0;
   while (t->GetEntry(e++) > 0)
      if ((e % 20) == 0) {
         Derived * d = (Derived*) h->fABC;
         std::cout << e << ": read d.abc==" << d->abc
                   << ", d.derived==" << d->derived << std::endl;
   }
}
