{
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
   if (1) {
#endif
   // Avoid loading the library
   gInterpreter->UnloadLibraryMap("sel01_C");

#if !defined(ClingWorkAroundScriptClassDef)
   // First in interpreted mode.
   gROOT->ProcessLine(".L sel01.C");
#ifdef ClingWorkAroundMissingDynamicScope
   TSelector *isel = (TSelector*)TClass::GetClass("sel01")->New();
#else
   sel01 isel;
#endif
   TFile *f = TFile::Open("Event1.root");
   TTree *tree; f->GetObject("T1",tree);
   
#ifdef ClingWorkAroundMissingDynamicScope
   tree->Process(isel);
#else
   tree->Process(&isel);
#endif
      
#ifdef ClingWorkAroundMissingImplicitAuto
   TChain *
#endif
   c = new TChain("T1");
   c->Add("Event1.root");
#ifdef ClingWorkAroundMissingDynamicScope
   c->Process(isel);
#else
   c->Process(&isel);
#endif   

#endif // !defined(ClingWorkAroundScriptClassDef)
      
#if defined(ClingWorkAroundScriptClassDef) || !defined(ClingWorkAroundMissingUnloading)
   gROOT->ProcessLine(".L sel01.C+");
#ifdef ClingWorkAroundMissingDynamicScope
   TSelector *csel = (TSelector*)TClass::GetClass("sel01")->New();
#else
   sel01 csel;
#endif

#ifdef ClingWorkAroundScriptClassDef
   TFile *f; TTree *tree;
#endif
   f = TFile::Open("Event1.root");
   f->GetObject("T1",tree);

#if defined(ClingWorkAroundScriptClassDef)
      // Get the work around printing to happen *after* the opening
      // of the file so that there are after the Warning messages.
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
      
#ifdef ClingWorkAroundMissingDynamicScope
      tree->Process(csel);
#else
      tree->Process(&csel);
#endif
      
#if defined(ClingWorkAroundScriptClassDef)
   TChain *
#endif
   c = new TChain("T1");
   c->Add("Event1.root");
#ifdef ClingWorkAroundMissingDynamicScope
   c->Process(csel);
#else
   c->Process(&csel);
#endif
      
#else // defined(ClingWorkAroundScriptClassDef) || !defined(ClingWorkAroundMissingUnloading)
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
      
#endif // defined(ClingWorkAroundScriptClassDef) || !defined(ClingWorkAroundMissingUnloading)
      
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
   }
#endif
}
   
