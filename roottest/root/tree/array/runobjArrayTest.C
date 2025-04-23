{
#ifdef ClingWorkAroundIncorrectTearDownOrder
   if (1) {
#endif

#ifndef ClingWorkAroundMissingDynamicScope
gROOT->ProcessLine(".L TestObj.cpp+");
#else
TH1F *h1, *h2;
#endif
// gSystem->Load("TestObj.so");

AllReconstructions* pRec = new AllReconstructions;
TTree Tree0("TestTree","Test"); Tree0.Branch("Main", "AllReconstructions", &pRec,32000,0);
pRec->Init();
for (int i = 0; i < 10; ++i) { Tree0.Fill(); }
Tree0.Draw("r.p.nt>>h1");
#ifdef ClingWorkAroundMissingDynamicScope
   h1 = (TH1F*)gROOT->FindObject("h1");
#endif
if (h1->GetMean()!=2) {
   fprintf(stderr,"r.p.nt not properly drawn (mean is %lf)\n",h1->GetMean());
}
Tree0.Draw("r.p.t>>h2");
#ifdef ClingWorkAroundMissingDynamicScope
   h2 = (TH1F*)gROOT->FindObject("h2");
#endif
if (h2->GetMean()!=1 || h2->GetEntries()!=240) {
   fprintf(stderr,"r.p.t not properly drawn (entry %lf, mean is %lf)\n",h2->GetEntries(), h2->GetMean());
}

TTree Tree1("TestTree","Test"); Tree1.Branch("Main", "AllReconstructions", &pRec,32000,1);
pRec->Init();
for (int i = 0; i < 10; ++i) { Tree1.Fill(); }
Tree1.Draw("r.p.nt>>h1");
#ifdef ClingWorkAroundMissingDynamicScope
   h1 = (TH1F*)gROOT->FindObject("h1");
#endif
if (h1->GetMean()!=2) {
   fprintf(stderr,"r.p.nt not properly drawn (mean is %lf)\n",h1->GetMean());
}
Tree1.Draw("r.p.t>>h2");
#ifdef ClingWorkAroundMissingDynamicScope
   h2 = (TH1F*)gROOT->FindObject("h2");
#endif
if (h2->GetMean()!=1 || h2->GetEntries()!=240) {
   fprintf(stderr,"r.p.t not properly drawn (entry %lf, mean is %lf)\n",h2->GetEntries(), h2->GetMean());
}

TTree Tree2("TestTree","Test"); Tree2.Branch("Main", "AllReconstructions", &pRec,32000,2);
pRec->Init();
for (int i = 0; i < 10; ++i) { Tree2.Fill(); }
Tree2.Draw("r.p.nt>>h1");
#ifdef ClingWorkAroundMissingDynamicScope
   h1 = (TH1F*)gROOT->FindObject("h1");
#endif
if (h1->GetMean()!=2) {
   fprintf(stderr,"r.p.nt not properly drawn (mean is %lf)\n",h1->GetMean());
}
Tree2.Draw("r.p.t>>h2");
#ifdef ClingWorkAroundMissingDynamicScope
   h2 = (TH1F*)gROOT->FindObject("h2");
#endif
if (h2->GetMean()!=1 || h2->GetEntries()!=240) {
   fprintf(stderr,"r.p.t not properly drawn (entry %lf, mean is %lf)\n",h2->GetEntries(), h2->GetMean());
}

TTree Tree99("TestTree","Test"); Tree99.Branch("Main", "AllReconstructions", &pRec,32000,99);
pRec->Init();
for (int i = 0; i < 10; ++i) { Tree99.Fill(); }
Tree99.Draw("r.p.nt>>h1");
#ifdef ClingWorkAroundMissingDynamicScope
   h1 = (TH1F*)gROOT->FindObject("h1");
#endif
if (h1->GetMean()!=2) {
   fprintf(stderr,"99. r.p.nt not properly drawn (mean is %lf)\n",h1->GetMean());
}
Tree99.Draw("r.p.t>>h2");
#ifdef ClingWorkAroundMissingDynamicScope
   h2 = (TH1F*)gROOT->FindObject("h2");
#endif
if (h2->GetMean()!=1 || h2->GetEntries()!=240) {
   fprintf(stderr,"99. r.p.t not properly drawn (entry %lf, mean is %lf)\n",h2->GetEntries(), h2->GetMean());
}

#ifdef ClingWorkAroundIncorrectTearDownOrder
   }
#endif
}
