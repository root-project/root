#include <iostream>

#include "Math/GeneticMinimizer.h"

#include "TMath.h"

using std::cout;
using std::endl;

class RosenBrockFunction : public ROOT::Math::IMultiGenFunction { 

public : 

   RosenBrockFunction() : fNCalls(0) {}
   virtual ~RosenBrockFunction() {}

   unsigned int NDim() const { return 2; } 

   ROOT::Math::IMultiGenFunction * Clone() const { 
      return new RosenBrockFunction();  
   }

   unsigned int getNCalls() { return fNCalls; }
   
   private: 
   mutable unsigned int fNCalls;

   inline double DoEval (const double * x) const { 
      fNCalls++;
      //std::cout << "called!" << std::endl;
      const Double_t xx = x[0];
      const Double_t yy = x[1];
      const Double_t tmp1 = yy-xx*xx;
      const Double_t tmp2 = 1-xx;
      return 100*tmp1*tmp1+tmp2*tmp2;
   }
}; 

class Parabole: public ROOT::Math::IMultiGenFunction {
public:
   virtual ~Parabole() {} 

   unsigned int NDim() const { return 1; }

   ROOT::Math::IMultiGenFunction * Clone() const { 
      return new Parabole();  
   }
   
   private: 

   inline double DoEval (const double * x) const { 
      return x[0] * x[0];
   }
};

class MultiMin: public ROOT::Math::IMultiGenFunction {
private: 
   inline double DoEval (const double * x) const { 
      return 0.6*TMath::Power(x[0],4) + 0.1*TMath::Power(x[0],3) - 2*TMath::Power(x[0],2) + 1;
   }

public:
   virtual ~MultiMin() {} 

   unsigned int NDim() const { return 1; }

   ROOT::Math::IMultiGenFunction * Clone() const { 
      return new MultiMin();  
   }
};

int testGAMinimizer() {
   int status = 0;

   ROOT::Math::GeneticMinimizer gaParabole;
   Parabole parabole;
   gaParabole.SetFunction(parabole);
   gaParabole.SetLimitedVariable(0, "x", 0, 0, -5, +5);
   gaParabole.Minimize();
   std::cout << "Parabole min:" << gaParabole.MinValue() << std::endl;

   ROOT::Math::GeneticMinimizer gaRosenBrock;
   RosenBrockFunction RosenBrock;
   gaRosenBrock.SetFunction(RosenBrock);
   gaRosenBrock.SetLimitedVariable(0, "x", 0, 0, -5, +5);
   gaRosenBrock.SetLimitedVariable(0, "x", 0, 0, -5, +5);
   gaRosenBrock.Minimize();
   const double * xmin = gaRosenBrock.X(); 
   std::cout << "RosenBrock min: [" << xmin[0] << "] [" << xmin[1] << "]" << std::endl;

   ROOT::Math::GeneticMinimizer gaMultiMin;
   MultiMin multimin;
   gaMultiMin.SetFunction(multimin);
   gaMultiMin.SetLimitedVariable(0, "x", 0, 0, -5, +5);
   gaMultiMin.Minimize();
   std::cout << "MultiMin min:" << gaMultiMin.MinValue() << std::endl;

   std::cout << "Done!" << std::endl;

   return status;
}

int main()
{
   int status = 0;

   status = testGAMinimizer();

   return status;
}
