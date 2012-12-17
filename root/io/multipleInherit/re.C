{
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
   if (1) {
#endif
   gROOT->ProcessLine(".L na.cxx+");
   gSystem->Load("libHistPainter");
   TFile f("hout.root");
   TH1 * h = (TH1*)f.Get("hpxpy");
   h->SetDirectory(gROOT);
   f.Close();
   TList * l = h->GetListOfFunctions();
//l->Print();
   int last = l->LastIndex();
   for(int i=0; i<=last; i++) {
     fprintf(stdout,"Obj#%d %p ",i,l->At(i));
     fprintf(stdout,"class %p ",l->At(i)->IsA());
     fprintf(stdout,"name %s\n",l->At(i)->IsA()->GetName());

   };
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
   }
#endif
}
