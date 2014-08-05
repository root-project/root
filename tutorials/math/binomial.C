// tutorial illustrating the use of TMath::Binomial
//  can be run with:
// root > .x binomial.C
// root > .x binomial.C+ with ACLIC
//Author: Federico Carminati

#include <TMath.h>
#include <TRandom.h>

void binomialSimple() {
  //
  // Simple test for the binomial distribution
  //
  printf("\nTMath::Binomial simple test\n");
  printf("Build the Tartaglia triangle\n");
  printf("============================\n");
  const Int_t max=13;
  Int_t j;
  for(Int_t i=0;i<max;i++) {
    printf("n=%2d",i);
    for(j=0;j<(max-i);j++) printf("  ");
    for(j=0;j<i+1;j++) printf("%4d",TMath::Nint(TMath::Binomial(i,j)));
    printf("\n");
  }
}

void binomialFancy() {
  Double_t x;
  Double_t y;
  Double_t res1;
  Double_t res2;
  Double_t err;
  Double_t serr=0;
  const Int_t nmax=10000;
  printf("\nTMath::Binomial fancy test\n");
  printf("Verify Newton formula for (x+y)^n\n");
  printf("x,y in [-2,2] and n from 0 to 9  \n");
  printf("=================================\n");
  TRandom r;
  for(Int_t i=0; i<nmax; i++) {
    do {
        x=2*(1-2*r.Rndm());
        y=2*(1-2*r.Rndm());
    } while (TMath::Abs(x+y)<0.75); //Avoid large cancellations
    for(Int_t j=0; j<10; j++) {
       res1=TMath::Power(x+y,j);
       res2=0;
       for(Int_t k=0; k<=j; k++)
          res2+=TMath::Power(x,k)*TMath::Power(y,j-k)*TMath::Binomial(j,k);
       if((err=TMath::Abs(res1-res2)/TMath::Abs(res1))>1e-10)
          printf("res1=%e res2=%e x=%e y=%e err=%e j=%d\n",res1,res2,x,y,err,j);
       serr +=err;
     }
  }
  printf("Average Error = %e\n",serr/nmax);
}

void binomial () {
   binomialSimple();
   binomialFancy();
}

