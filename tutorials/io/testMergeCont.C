TFile *f;

TSeqCollection *GetCollection();

void testMergeCont()
{
   // Macro to test merging of containers.

   TString tutdir = gROOT->GetTutorialsDir();
   gROOT->LoadMacro(tutdir+"/hsimple.C");
   TList *list = (TList *)GetCollection();
   TList *inputs = new TList();
   for (Int_t i=0; i<10; i++) {
      inputs->AddAt(GetCollection(),0);
      list->Merge(inputs);
      inputs->Delete();
      f->Close();
   }
   delete inputs;
   TH1F *hpx = (TH1F*)(((TList*)list->At(1))->At(0));
   printf("============================================\n");
   printf("Total  hpx: %d entries\n", (int)hpx->GetEntries());
   hpx->Draw();
   list->Delete();
   delete list;
}


TSeqCollection *GetCollection()
{
   TObject *obj;
#ifndef ClingWorkAroundMissingDynamicScope
# define ClingWorkAroundMissingDynamicScope
#endif
   f = TFile::Open("hsimple.root");
   if( !f ) {
#ifdef ClingWorkAroundMissingDynamicScope
     f = (TFile*)gROOT->ProcessLine("hsimple(1);");
#else
     f = hsimple(1);
#endif
   }
   gROOT->cd();
   TList *l0 = new TList();
   TList *l01 = new TList();
   TH1 *hpx = (TH1*)f->Get("hpx");
   printf("Adding hpx: %d entries\n", (int)hpx->GetEntries());
   l01->Add(hpx);
   TH1 *hpxpy = (TH1*)f->Get("hpxpy");
   l01->Add(hpxpy);
   TH1 *hprof = (TH1*)f->Get("hprof");
   l0->Add(hprof);
   l0->Add(l01);
   return l0;
}
