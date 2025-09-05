{
   // Avoid loading the library
   gInterpreter->UnloadLibraryMap("sel01_C");

   // First in interpreted mode.
   gROOT->ProcessLine(".L sel01.C");
# ifdef ClingWorkAroundMissingDynamicScope
   TSelector *isel = (TSelector*)TClass::GetClass("sel01")->New();
# else
   sel01 isel;
# endif
   TFile *f = TFile::Open("Event1.root");
   TTree *tree; f->GetObject("T1",tree);

# ifdef ClingWorkAroundMissingDynamicScope
   tree->Process(isel);
# else
   tree->Process(&isel);
# endif

   auto c = new TChain("T1");
   c->Add("Event1.root");
# ifdef ClingWorkAroundMissingDynamicScope
   c->Process(isel);
# else
   c->Process(&isel);
# endif

   gROOT->ProcessLine(".L sel01.C+");
# ifdef ClingWorkAroundMissingDynamicScope
   TSelector *csel = (TSelector*)TClass::GetClass("sel01")->New();
# else
   sel01 csel;
# endif

   f = TFile::Open("Event1.root");
   f->GetObject("T1",tree);

# ifdef ClingWorkAroundMissingDynamicScope
      tree->Process(csel);
# else
      tree->Process(&csel);
# endif

   c = new TChain("T1");
   c->Add("Event1.root");
# ifdef ClingWorkAroundMissingDynamicScope
   c->Process(csel);
# else
   c->Process(&csel);
# endif

}
