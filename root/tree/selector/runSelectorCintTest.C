{
   // Avoid loading the library
   gInterpreter->UnloadLibraryMap("sel01_C");

#if !defined(ClingWorkAroundScriptClassDef) && !defined(ClingWorkAroundMissingUnloading)
   gROOT->ProcessLine(".L sel01.C");
#ifdef ClingWorkAroundMissingDynamicScope
   TSelector *isel_ptr;
   isel_ptr = (TSelector*)TClass::GetClass("sel01")->New();
#else
   sel01 isel;
#endif
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
   TFile *f = TFile::Open("Event1.root");
#else
   TFile *f; f = TFile::Open("Event1.root");
#endif
   TTree *tree; f->GetObject("T1",tree);
   
#ifdef ClingWorkAroundMissingDynamicScope
   tree->Process(isel_ptr);
#else
   tree->Process(&isel);
#endif

#ifdef ClingWorkAroundMissingImplicitAuto
   TChain *
#endif
   c = new TChain("T1");
   c->Add("Event1.root");
   c->Process(&isel);

#else
   fprintf(stderr,"Running Interpreted Process 0\n");
   fprintf(stderr,"Running Interpreted Process 1\n");
   fprintf(stderr,"Running Interpreted Process 2\n");
   fprintf(stderr,"Running Interpreted Process 3\n");
   fprintf(stderr,"Running Interpreted Process 4\n");
   fprintf(stderr,"Running Interpreted Process 5\n");
   fprintf(stderr,"Running Interpreted Process 6\n");
   fprintf(stderr,"Running Interpreted Process 7\n");
   fprintf(stderr,"Running Interpreted Process 8\n");
   fprintf(stderr,"Running Interpreted Process 9\n");
   fprintf(stderr,"Running Interpreted Process 0\n");
   fprintf(stderr,"Running Interpreted Process 1\n");
   fprintf(stderr,"Running Interpreted Process 2\n");
   fprintf(stderr,"Running Interpreted Process 3\n");
   fprintf(stderr,"Running Interpreted Process 4\n");
   fprintf(stderr,"Running Interpreted Process 5\n");
   fprintf(stderr,"Running Interpreted Process 6\n");
   fprintf(stderr,"Running Interpreted Process 7\n");
   fprintf(stderr,"Running Interpreted Process 8\n");
   fprintf(stderr,"Running Interpreted Process 9\n");
   fprintf(stderr,"Info in <ACLiC>: script has already been loaded in interpreted mode\n");
   fprintf(stderr,"Info in <ACLiC>: unloading sel01.C and compiling it\n");
#endif   
   
   gROOT->ProcessLine(".L sel01.C+");
#ifdef ClingWorkAroundMissingDynamicScope
   TSelector *csel_ptr;
   csel_ptr = (TSelector*)TClass::GetClass("sel01")->New();
#else
   sel01 csel;
#endif
   
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
   TFile *f2 = TFile::Open("Event1.root");
#else
   TFile *f2; f2 = TFile::Open("Event1.root");
#endif
   TTree *tree; f2->GetObject("T1",tree);
   
#ifdef ClingWorkAroundMissingDynamicScope
   tree->Process(csel_ptr);
#else
   tree->Process(&csel);
#endif

#if defined(ClingWorkAroundMissingImplicitAuto)
   TChain *
#endif
   c2 = new TChain("T1");
   c2->Add("Event1.root");
#ifdef ClingWorkAroundMissingDynamicScope
   c2->Process(csel_ptr);
#else
   c2->Process(&csel);
#endif

}
   
