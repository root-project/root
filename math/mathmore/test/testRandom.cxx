#include "Math/Random.h"
#include "Math/GSLRndmEngines.h"
#include "TStopwatch.h"
#include "TRandom1.h"
#include "TRandom2.h"
#include "TRandom3.h"
#include "TRandomGen.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <typeinfo>
#include <iomanip>
#include <fstream>

#include <limits>

#ifdef HAVE_CLHEP
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RanluxEngine.h"
#include "CLHEP/Random/Ranlux64Engine.h"
#endif



#ifndef PI
#define PI       3.14159265358979323846264338328      /* pi */
#endif

//#define NEVT 10000
#ifndef NEVT
#define NEVT 10000000
#endif

using namespace ROOT::Math;

#ifdef HAVE_CLHEP
using namespace CLHEP;
#endif


// wrapper for stdrand

class RandomStd {
public:
  RandomStd() {
    fScale = 1./(double(RAND_MAX) + 1.);
  }

  inline void RndmArray(int n, double * x) {
    for ( double * itr = x; itr != x+n; ++itr)
      *itr= fScale*double(std::rand());
  }
  inline double Uniform() {
    return fScale*double(std::rand());
  }

  std::string Type() const { return "std::rand"; }
  unsigned int EngineSize() const { return 0; }

  void SetSeed(int seed) { std::srand(seed); }

private:
  double fScale;

};



// wrapper for clhep

template <class Engine>
class RandomCLHEP {
public:
   RandomCLHEP(Engine & e) :
      fRand(e)
   {
   }

  inline void RndmArray(int n, double * x) {
     fRand.flatArray(n,x);
  }
  inline double Uniform() {
     return fRand.flat();
  }

   std::string Type() const { return std::string("CLHEP ") + Engine::engineName(); }
   unsigned int EngineSize() const { return 0; }

  void SetSeed(int seed) { fRand.setSeed(seed); }

private:
   Engine & fRand;

};





template <class R>
void printName( const R & r) {
  std::cout << "\nRandom :\t " << r.Type() << " \t size of state = " << r.EngineSize() << std::endl;
}

// specializations for TRandom's
void printName( const TRandom & r) {
  std::cout << "\nRandom :\t " << r.ClassName() << std::endl;
}
// specializations for TRandom's
void printName( const TRandom1 & r) {
  std::cout << "\nRandom :\t " << r.ClassName() << std::endl;
}
// specializations for TRandom's
void printName( const TRandom2 & r) {
  std::cout << "\nRandom :\t " << r.ClassName() << std::endl;
}
// specializations for TRandom's
void printName( const TRandom3 & r) {
  std::cout << "\nRandom :\t " << r.ClassName() << std::endl;
}
// specialization for TRandomGen
template<class Engine>
void printName(const TRandomGen<Engine> &r) {
   std::cout << "\nRandom :\t " << r.ClassName() << std::endl;
}


template <class R>
void generate( R & r, bool array=true) {

  TStopwatch w;

  int n = NEVT;
  // estimate PI
  double n1=0;
  double x,y;
  w.Start();
  // use default seeds
  // r.SetSeed(0);
  //r.SetSeed(int(pow(2,28)) );
  if (array) {
    double ax[1000];
    double ay[1000];
    for (int i = 0; i < n; i+=1000 ) {
      r.RndmArray(1000,ax);
      r.RndmArray(1000,ay);
      for (int j = 0; j < 1000; ++j)
       if ( ( ax[j]*ax[j] + ay[j]*ay[j] ) <= 1.0 ) n1++;
    }
  }
  else {
    for (int i = 0; i < n; ++i) {
      x=r.Uniform();
      y=r.Uniform();
      if ( ( x*x + y*y ) <= 1.0 ) n1++;
    }
  }
  w.Stop();

  printName(r);
  std::cout << "\tTime = " << w.RealTime()*1.0E9/NEVT << "  "
            << w.CpuTime()*1.0E9/NEVT
            << " (ns/call)" << std::endl;
  double piEstimate = 4.0 * double(n1)/double(n);
  double delta = piEstimate-PI;
  double sigma = std::sqrt( PI * (4 - PI)/double(n) );
  std::cout << "\t\tDeltaPI = " << delta/sigma << " (sigma) " << std::endl;
}

int main() {

  std::cout << "***************************************************\n";
  std::cout << " TEST RANDOM    NEVT = " << NEVT << std::endl;
  std::cout << "***************************************************\n\n";



  Random<GSLRngMT>         r1;
  Random<GSLRngTaus>       r2;
  Random<GSLRngRanLux>     r3;
  Random<GSLRngRanLuxS1>   r3s1;
  Random<GSLRngRanLuxS2>   r3s2;
  Random<GSLRngRanLuxD1>   r3d1;
  Random<GSLRngRanLuxD2>   r3d2;
  Random<GSLRngGFSR4>      r4;
  Random<GSLRngCMRG>       r5;
  Random<GSLRngMRG>        r6;
  Random<GSLRngRand>       r7;
  Random<GSLRngRanMar>     r8;
  Random<GSLRngMinStd>     r9;
  Random<GSLRngMixMax>     r10;

  // std engine
  RandomStd                sr0; 

  // ROOT engine
  TRandom                  tr0;
  TRandom1                 tr1;
  TRandom1                 tr1a(0,0);
  TRandom1                 tr1b(0,1);
  TRandom1                 tr1c(0,2);
  TRandom1                 tr1d(0,3);
  TRandom1                 tr1e(0,4);
  TRandom2                 tr2;
  TRandom3                 tr3;
  TRandomMixMax            tr4;
  TRandomMixMax17          tr5;
  TRandomMixMax256         tr6;

  generate(tr0);
  generate(tr1);
  generate(tr1a);
  generate(tr1b);
  generate(tr1c);
  generate(tr1d);
  generate(tr1e);
  generate(tr2);
  generate(tr3);
  generate(tr4);
  generate(tr5);
  generate(tr6);

  generate(sr0);

  generate(r1);
  generate(r2);
  generate(r3);
  generate(r3s1);
  generate(r3s2);
  generate(r3d1);
  generate(r3d2);
  generate(r4);
  generate(r5);
  generate(r6);
  generate(r7);
  generate(r8);
  generate(r9);
  generate(r10);

#ifdef HAVE_CLHEP
  RanluxEngine             e1(1,3);
  RanluxEngine             e2(1,4);
  Ranlux64Engine           e3(1,0);
  Ranlux64Engine           e4(1,1);
  Ranlux64Engine           e5(1,2);

  RandomCLHEP<RanluxEngine>  crlx3(e1);
  RandomCLHEP<RanluxEngine>  crlx4(e2);

  RandomCLHEP<Ranlux64Engine>  crlx64a(e3);
  RandomCLHEP<Ranlux64Engine>  crlx64b(e4);
  RandomCLHEP<Ranlux64Engine>  crlx64c(e5);

  generate(crlx3);
  generate(crlx4);

  generate(crlx64a);
  generate(crlx64b);
  generate(crlx64c);

#endif

  // generate 1000 number with GSL MT and check with TRandom3
  int n = 1000;
  std::vector<double> v1(n);
  std::vector<double> v2(n);

  Random<GSLRngMT>         gslRndm(4357);
  TRandom3                 rootRndm(4357);


  gslRndm.RndmArray(n,&v1[0]);
  rootRndm.RndmArray(n,&v2[0]);

  int nfail=0;
  for (int i = 0; i < n; ++i) {
     double d = std::fabs(v1[i] - v2[i] );
      if (d > std::numeric_limits<double>::epsilon()*v1[i] ) nfail++;
  }
  if (nfail > 0) {
     std::cout << "ERROR: Test failing comparing TRandom3 with GSL MT" << std::endl;
     return -1;
  }
  // save the generated number
  std::ofstream file("testRandom.out");
  std::ostream & out = file;
  int  j = 0;
  int prec = std::cout.precision(9);
  while ( j < n) {
     for (int l = 0; l < 8; ++l) {
        out << std::setw(12) << v1[j+l] << ",";
//         int nws = int(-log10(v1[j+l]));
//         for (int k = nws; k >= 0; --k)
//            out << " ";
     }
     out << std::endl;
     j+= 8;
  }
  std::cout.precision(prec);

  return 0;

}
