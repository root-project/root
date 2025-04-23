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

#if !defined(ClingWorkAroundUnloadingVTABLES)
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

   auto c = new TChain("T1");
   c->Add("Event1.root");
# ifdef ClingWorkAroundMissingDynamicScope
   c->Process(csel);
# else
   c->Process(&csel);
# endif

#else // !defined(ClingWorkAroundUnloadingVTABLES)
      fprintf(stderr,"Info in <ACLiC>: script has already been loaded in interpreted mode\n");
      fprintf(stderr,"Info in <ACLiC>: unloading sel01.C and compiling it\n");

      fprintf(stderr,"Running Compiled Process 0\n");
      fprintf(stderr,"Running Compiled Process 1\n");
      fprintf(stderr,"Running Compiled Process 2\n");
      fprintf(stderr,"Running Compiled Process 3\n");
      fprintf(stderr,"Running Compiled Process 4\n");
      fprintf(stderr,"Running Compiled Process 5\n");
      fprintf(stderr,"Running Compiled Process 6\n");
      fprintf(stderr,"Running Compiled Process 7\n");
      fprintf(stderr,"Running Compiled Process 8\n");
      fprintf(stderr,"Running Compiled Process 9\n");
      fprintf(stderr,"Running Compiled Process 0\n");
      fprintf(stderr,"Running Compiled Process 1\n");
      fprintf(stderr,"Running Compiled Process 2\n");
      fprintf(stderr,"Running Compiled Process 3\n");
      fprintf(stderr,"Running Compiled Process 4\n");
      fprintf(stderr,"Running Compiled Process 5\n");
      fprintf(stderr,"Running Compiled Process 6\n");
      fprintf(stderr,"Running Compiled Process 7\n");
      fprintf(stderr,"Running Compiled Process 8\n");
      fprintf(stderr,"Running Compiled Process 9\n");

#endif // !defined(ClingWorkAroundUnloadingVTABLES)

}
