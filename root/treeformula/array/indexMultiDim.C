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
   int v1 = t->Draw("Vnt_fmcvtx_r[Vnt_vtx_i[0]-1][]>>histo1","Vnt_fmckin_id<50");
   int e1 = histo1->GetEntries();
   int m1 = histo1->GetMean();
   c->cd(2);
   int v2 = t->Draw("Vnt_fmcvtx_r[][]>>histo2","Vnt_fmckin_id<50&&(int(Iteration$/3)==(Vnt_vtx_i[0]-1))");
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
return !result; // this is used by the makefile
}
