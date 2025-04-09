#include <TMath.h>
#include <TF1.h>

Double_t gauss(Double_t *x, Double_t *par)
{
return (par[0] * TMath::Gaus(x[0],par[1],par[2]));
}

int NonFormulaMean()
{
   TF1 *g1 = new TF1("hh3", gauss, 0, 10, 3);
   g1->SetParameters(1, 1, 1);
   g1->Draw();
   Double_t m1 = g1->Mean(0, 10);
   return false; // because we know it is failling! 
} 
