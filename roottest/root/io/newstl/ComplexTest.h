#ifndef Test_hh
#define Test_hh

#include "TObject.h"
#include <complex>

// #include <vector>

class Test: public TObject
{
  public:

   std::complex<double> fMyComplexVector;
   //std::vector<std::complex<double> > fMyComplexVector;

   void Set(int seed) {
      fMyComplexVector = std::complex<double>(seed,seed*2);
   }

   bool TestValue(int seed) {
      return ( seed == (int)fMyComplexVector.real() && seed == (int)(fMyComplexVector.imag()/2) );
   }

  ClassDefOverride(Test, 1);
};

#endif /* Test_hh */
