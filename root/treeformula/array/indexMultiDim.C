{
bool result = true;
TFile *f = new TFile("orange.root");
TTree *t = (TTree*)f->Get("h1");
auto c = new TCanvas();
c->Divide(1,2);

#ifdef ClingWorkAroundMissingDynamicScope
   TH1F *histo1, *histo2;
#endif
{
   c->cd(1);
   Long64_t v1 = t->Draw("Vnt_fmcvtx_r[Vnt_vtx_i[0]-1][]>>histo1","");
#ifdef ClingWorkAroundMissingDynamicScope
   histo1 = (TH1F*)gROOT->FindObject("histo1");
#endif
   int m1 = histo1->GetMean();
   c->cd(2);
   Long64_t v2 = t->Draw("Vnt_fmcvtx_r[][]>>histo2","(int(Iteration$/3)==(Vnt_vtx_i[0]-1))");
#ifdef ClingWorkAroundMissingDynamicScope
   histo2 = (TH1F*)gROOT->FindObject("histo2");
#endif
   int m2 = histo2->GetMean();
   if (v1!=v2) {
      result = false;
      fprintf(stderr,"draw return 1 is %lld while draw return 2 is %lld\n",v1,v2);
   }
   if (m1!=m2) {
      result = false;
      fprintf(stderr,"mean 1 is %d while mean 2 is %d\n",m1,m2);
   }
}
{
   c->cd(1);
   delete histo1;
   delete histo2;
   Long64_t v1 = t->Draw("Vnt_fmcvtx_r[Vnt_vtx_i[0]-1][]>>histo1","Vnt_fmckin_id[Vnt_vtx_i[0]-1]<50");
   histo1 = (TH1F*)gDirectory->Get("histo1");
   int e1 = histo1->GetEntries();
   int m1 = histo1->GetMean();
   c->cd(2);
   Long64_t v2 = t->Draw("Vnt_fmcvtx_r[][]>>histo2","Vnt_fmckin_id[]<50&&(int(Iteration$/3)==(Vnt_vtx_i[0]-1))");
   histo2 = (TH1F*)gDirectory->Get("histo2");
   int e2 = histo2->GetEntries();
   int m2 = histo2->GetMean();
   if (v1!=v2) {
      result = false;
      fprintf(stderr,"draw return 1 is %lld while draw return 2 is %lld\n",v1,v2);
   }
   if (e1!=e2) {
      result = false;
      fprintf(stderr,"entries 1 is %d while entries 2 is %d\n",e1,e2);
   }
   if (m1!=m2) {
      result = false;
      fprintf(stderr,"mean 1 is %d while mean 2 is %d\n",m1,m2);
   }
}
if (1) {
   c->cd(1);
   delete histo1;
   delete histo2;
   Long64_t v1 = t->Draw("Vnt_fmcvtx_r[Vnt_vtx_i[]-1][]>>histo1","Vnt_fmckin_id[Vnt_vtx_i[]-1]<50");
   histo1 = (TH1F*)gDirectory->Get("histo1");
   int e1 = histo1->GetEntries();
   int m1 = histo1->GetMean();
   c->cd(2);
   Long64_t v2 = t->Draw("Vnt_fmcvtx_r[][]>>histo2","Vnt_fmckin_id[]<50&&(int(Iteration$/3)==(Vnt_vtx_i[]-1))");
   histo2 = (TH1F*)gDirectory->Get("histo2");
   int e2 = histo2->GetEntries();
   int m2 = histo2->GetMean();
   if (v1!=v2) {
      result = false;
      fprintf(stderr,"draw return 1 is %lld while draw return 2 is %lld\n",v1,v2);
   }
   if (e1!=e2) {
      result = false;
      fprintf(stderr,"entries 1 is %d while entries 2 is %d\n",e1,e2);
   }
   if (m1!=m2) {
      result = false;
      fprintf(stderr,"mean 1 is %d while mean 2 is %d\n",m1,m2);
   }
}
if (1) {
   f = new TFile("shorttrack.root");
   TTree *t2; f->GetObject("T",t2);
   int v11 = (int)t2->Draw("fPx[abs(fNpoint-65)]+fMeasures[]","","",1,0);
   int v12 = (int)t2->Draw("fPx[abs(fNpoint-65)]+fMeasures[fMeasures]","","",1,0);
   if (v11!=1||v12!=1) {
      fprintf(stderr,"For shorttrack v1 is %d and v2 is %d (both should be 1)\n",v11,v12);
      result = false;
   }
}

#ifdef ClingWorkAroundBrokenUnnamedReturn
   bool res = ! result;
#else
return !result; // this is used by the makefile
#endif
}
