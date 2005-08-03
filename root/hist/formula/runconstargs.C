#include "TF1.h"
#include "Riostream.h"
#include "TMath.h"

 //prototype for my Gamma.
double gammaDistribution(const double *yVar, const double gammaPar[]) { 
   return yVar[0]+gammaPar[0];
};

TF1* create() {
   //This line has compiler problems.
   TF1 *gammaDist=new TF1("Gamma_Distribution",gammaDistribution,-2.,10.,4); 
   gammaDist->SetParameter(0,7.1);
   double x = gammaDist->Eval(3);
   double expected = 10.1;
   if (TMath::Abs(x-expected)>0.0001) cout << "x is " << x << " instead of " << expected << "!\nBadly formed TF1 from gammaDistribution\n";
   return gammaDist;
}

bool runconstargs() {
   TF1 *func = create();
   return (func==0);
}
