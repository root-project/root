{
gROOT->Reset();
char name[4][50];

for(int i=0; i<4; i++)
{
  snprintf(name[i],50,"hist_array_%d", i);
}


TH1F *hist_array[4];

for(int i=0; i<4; i++)
{

   /*  h[i] = (TH1F*)gROOT->FindObject(name[i]);
  if(h[i]) 
    {
      printf("%s deleted\n", name[i]);
      h[i]->Delete();
    }
 
  */

  
  /*  TH1 *h = (TH1*)(FindObject(name[i]));
  if(h) delete h;
  */

  TH1F *h = (TH1F*)gROOT->FindObject(name[i]);
  if(h) h->Delete();

}



TCanvas *c1 = new TCanvas("c1", "Test Canvas");
c1->Divide(2,2);

for(int i=0; i<4; i++)
{
  hist_array[i] = new TH1F(name[i], name[i], 100, -10, 10);
  hist_array[i]->FillRandom("gaus", 10000);

  c1->cd(i+1);
  hist_array[i]->Draw();

}
}
