{
bool result = true;
TFile *f = new TFile("orange.root");
TTree *t = (TTree*)f->Get("h1");
c = new TCanvas();
c->Divide(1,2);

{
   c->cd(1);
   int v1 = t->Draw("Vnt_fmcvtx_r[Vnt_vtx_i[0]-1][]>>histo1","");
   int m1 = histo1->GetMean();
   c->cd(2);
   int v2 = t->Draw("Vnt_fmcvtx_r[][]>>histo2","(int(Iteration$/3)==(Vnt_vtx_i[0]-1))");
   int m2 = histo2->GetMean();
   if (v1!=v2) {
      result = false;
      fprintf(stderr,"draw return 1 is %d while draw return 2 is %d\n",v1,v2);
   }
   if (m1!=m2) {
      result = false;
      fprintf(stderr,"mean 1 is %d while mean 2 is %d\n",m1,m22);
   }
}
{
   c->cd(1);
   delete histo1;
   delete histo2;
   int v1 = t->Draw("Vnt_fmcvtx_r[Vnt_vtx_i[0]-1][]>>histo1","Vnt_fmckin_id[Vnt_vtx_i[0]-1]<50");
   histo1 = (TH1F*)gDirectory->Get("histo1");
   int e1 = histo1->GetEntries();
   int m1 = histo1->GetMean();
   c->cd(2);
   int v2 = t->Draw("Vnt_fmcvtx_r[][]>>histo2","Vnt_fmckin_id[]<50&&(int(Iteration$/3)==(Vnt_vtx_i[0]-1))");
   histo2 = (TH1F*)gDirectory->Get("histo2");
   int e2 = histo2->GetEntries();
   int m2 = histo2->GetMean();
   if (v1!=v2) {
      result = false;
      fprintf(stderr,"draw return 1 is %d while draw return 2 is %d\n",v1,v2);
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
{
   c->cd(1);
   delete histo1;
   delete histo2;
   int v1 = t->Draw("Vnt_fmcvtx_r[Vnt_vtx_i[]-1][]>>histo1","Vnt_fmckin_id[Vnt_vtx_i[]-1]<50");
   histo1 = (TH1F*)gDirectory->Get("histo1");
   int e1 = histo1->GetEntries();
   int m1 = histo1->GetMean();
   c->cd(2);
   int v2 = t->Draw("Vnt_fmcvtx_r[][]>>histo2","Vnt_fmckin_id[]<50&&(int(Iteration$/3)==(Vnt_vtx_i[]-1))");
   histo2 = (TH1F*)gDirectory->Get("histo2");
   int e2 = histo2->GetEntries();
   int m2 = histo2->GetMean();
   if (v1!=v2) {
      result = false;
      fprintf(stderr,"draw return 1 is %d while draw return 2 is %d\n",v1,v2);
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
{
   f = new TFile("shorttrack.root");
   t2 = (TTree*)f->Get("T");
   int v1 = t2->Draw("fPx[abs(fNpoint-65)]+fMeasures[]","","",1,0);
   int v2 = t2->Draw("fPx[abs(fNpoint-65)]+fMeasures[fMeasures]","","",1,0);
   if (v1!=1||v2!=1) {
      fprintf(stderr,"For shorttrack v1 is %d and v2 is %d (both should be 1)\n",v1,v2);
      result = false;
   }
}

return !result; // this is used by the makefile
}
