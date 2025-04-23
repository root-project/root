#include "MyClass.h"
#include "TMath.h"

ClassImp(MyClass)

TF1*    MyClass::fgWSb            = NULL;     

MyClass::MyClass() : TObject() {
  printf("Default constructor\n");
}

Double_t MyClass::WSb(Double_t* x, Double_t* par)
{
  //
  //  Woods-Saxon Parameterisation
  //  as a function of radius (xx)
  //
  const Double_t kxx  = x[0];   //fm
  const Double_t kr0  = par[0]; //fm
  const Double_t kd   = par[1]; //fm   
  const Double_t kw   = par[2]; //no units
  const Double_t kn   = par[3]; //fm^-3 (used to normalize integral to one)
  Double_t y   = kn * (1.+kw*(kxx/kr0)*(kxx/kr0))/(1.+TMath::Exp((kxx-kr0)/kd));
  return y; //fm^-3
}

void MyClass::Init() {

  Float_t fWSr0 = 6.38;      // Wood-Saxon Parameter r0
  Float_t fWSd  = 0.535;       // Wood-Saxon Parameter d
  Float_t fWSw = 0.;       // Wood-Saxon Parameter w
  Float_t fWSn = 8.59e-4;       // Wood-Saxon Parameter n
  


  fgWSb = new TF1("WSb", WSb, 0, 20, 4);

  fgWSb->SetParameter(0, fWSr0);
  fgWSb->SetParameter(1, fWSd);
  fgWSb->SetParameter(2, fWSw);
  fgWSb->SetParameter(3, fWSn);
}

void MyClass::Integral(Double_t a, Double_t b) {
  printf("Integral: %f\n",fgWSb->Integral(a,b));
}
  

