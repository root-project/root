//Example illustrating how to draw TGaxis with labels defined by a function.
//Author: Olivier Couet
void gaxis2(){
   gStyle->SetOptStat(0);

   TH2F *h2 = new TH2F("h","Axes",100,0,10,100,-2,2);
   h2->Draw();

   TF1 *f1 = new TF1("f1","-x",-10,10);
   TGaxis *A1 = new TGaxis(0,2,10,2,"f1",510,"-");
   A1->SetTitle("axis with decreasing values");
   A1->Draw();

   TF1 *f2 = new TF1("f2","exp(x)",0,2);
   TGaxis *A2 = new TGaxis(1,1,9,1,"f2");
   A2->SetTitle("exponential axis");
   A2->SetLabelSize(0.03);
   A2->SetTitleSize(0.03);
   A2->SetTitleOffset(1.2);
   A2->Draw();

   TF1 *f3 = new TF1("f3","log10(x)",1,1000);
   TGaxis *A3 = new TGaxis(2,-2,2,0,"f3",505,"G");
   A3->SetTitle("logarithmic axis");
   A3->SetLabelSize(0.03);
   A3->SetTitleSize(0.03);
   A3->SetTitleOffset(1.2);
   A3->Draw();
}
