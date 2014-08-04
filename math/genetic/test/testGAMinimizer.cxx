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
      //cout << "called!" << endl;
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

int testGAMinimizer(int verbose = 0) {
   int status = 0;

   if (verbose) {
      cout << "****************************************************\n";
      cout << "Parabola Function Minimization \n";
   }
   ROOT::Math::GeneticMinimizer gaParabole(2);
   Parabole parabole;
   gaParabole.SetFunction(parabole);
   gaParabole.SetLimitedVariable(0, "x", 0, 0, -5, +5);
   gaParabole.SetPrintLevel(verbose);
   gaParabole.Minimize();
   cout << "Parabole min:" << gaParabole.MinValue() << "  x = [" << gaParabole.X()[0] << "]" << endl;
   bool ok =  (std::abs(gaParabole.MinValue() ) < 1.E-3 );
   if (!ok) Error("testGAMinimizer","Test failed for parabola");
   status |= !ok;

   if (verbose) {
      cout << "****************************************************\n";
      cout << "Rosenbrock Function Minimization \n";
   }
   ROOT::Math::GeneticMinimizer gaRosenBrock;
   RosenBrockFunction RosenBrock;
   gaRosenBrock.SetFunction(RosenBrock);
   gaRosenBrock.SetLimitedVariable(0, "x", 0, 0, -5, +5);
   gaRosenBrock.SetLimitedVariable(1, "y", 0, 0, -5, +5);
   gaRosenBrock.SetPrintLevel(verbose);
   gaRosenBrock.SetMaxIterations(500);  // need a large number ot be sure
   gaRosenBrock.SetRandomSeed(111);
   gaRosenBrock.Minimize();
   const double * xmin = gaRosenBrock.X();
   cout << "RosenBrock min: " << gaRosenBrock.MinValue() << " x = [" << xmin[0] << "] [" << xmin[1] << "]" << endl;
   ok =  (std::abs(gaRosenBrock.MinValue() ) < 5.E-2 ); // relax tolerance for Rosenbrock
   if (!ok) Error("testGAMinimizer","Test failed for RosenBrock");
   status |= !ok;

   if (verbose) {
      cout << "****************************************************\n";
      cout << "MultiMinima Function Minimization \n";
   }
   ROOT::Math::GeneticMinimizer gaMultiMin;
   MultiMin multimin;
   gaMultiMin.SetFunction(multimin);
   gaMultiMin.SetLimitedVariable(0, "x", 0, 0, -5, +5);
   gaMultiMin.SetPrintLevel(verbose);
   gaMultiMin.Minimize();
   cout << "MultiMin min:" << gaMultiMin.MinValue() << "  x = [" << gaMultiMin.X()[0] << "]" << endl;
   ok =  (std::abs(gaMultiMin.MinValue() + 0.8982) < 1.E-3 );
   if (!ok) Error("testGAMinimizer","Test failed for MultiMin");
   status |= !ok;

   if (status) cout << "Test Failed !" << endl;
   else cout << "Done!" << endl;

   return status;
}

int main(int argc, char **argv)
{
   int status = 0;
   int verbose = 0;

  // Parse command line arguments
   for (Int_t i=1 ;  i<argc ; i++) {
      std::string arg = argv[i] ;
      if (arg == "-v") {
         verbose = 1;
      }
      if (arg == "-vv") {
         verbose = 3;
      }
      if (arg == "-h") {
         std::cout << "Usage: " << argv[0] << " [-v] [-vv]\n";
         std::cout << "  where:\n";
         std::cout << "     -v  : verbose mode\n";
         std::cout << "     -vv : very verbose mode\n";
         std::cout << std::endl;
         return -1;
      }
   }


   status = testGAMinimizer(verbose);

   return status;
}
