void re()
{
   gROOT->ProcessLine(".L na.cxx+");
   auto f = TFile::Open("hout.root");
   TH1 * h = (TH1*) f->Get("hpxpy");
   h->SetDirectory(gROOT);
   delete f;

   TList * l = h->GetListOfFunctions();
   //l->Print();
   int last = l->LastIndex();
   for(int i=0; i<=last; i++) {
      fprintf(stdout,"Obj#%d class name %s\n",i, l->At(i)->IsA()->GetName());
   }
}
