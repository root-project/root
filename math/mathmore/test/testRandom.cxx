#include "Math/Random.h"
#include "Math/GSLRndmEngines.h"
#include "TStopwatch.h"
#include "TRandom1.h"
#include "TRandom2.h"
#include "TRandom3.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <typeinfo>
#include <iomanip>
#include <fstream> 

#include <limits>

#ifndef PI
#define PI       3.14159265358979323846264338328      /* pi */
#endif

//#define NEVT 10000
#ifndef NEVT
#define NEVT 10000000
#endif

using namespace ROOT::Math;

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
    double x[1000];
    double y[1000];
    for (int i = 0; i < n; i+=1000 ) {  
      r.RndmArray(1000,x);
      r.RndmArray(1000,y);
      for (int j = 0; j < 1000; ++j) 
	if ( ( x[j]*x[j] + y[j]*y[j] ) <= 1.0 ) n1++;
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
  Random<GSLRngRanLux2>    r31;
  Random<GSLRngRanLux48>   r32;
  Random<GSLRngGFSR4>      r4;
  Random<GSLRngCMRG>       r5;
  Random<GSLRngMRG>        r6;
  Random<GSLRngRand>       r7;
  Random<GSLRngRanMar>     r8;
  Random<GSLRngMinStd>     r9;
  RandomStd                r10; 

  TRandom                  tr0;
  TRandom1                 tr1;
  TRandom2                 tr2;
  TRandom3                 tr3;


  generate(tr0);
  generate(tr1);
  generate(tr2);
  generate(tr3);

  generate(r10);

  generate(r1);
  generate(r2);
  generate(r3);
  generate(r31);
  generate(r32);
  generate(r4);
  generate(r5);
  generate(r6);
  generate(r7);
  generate(r8);
  generate(r9);


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
  ofstream file("testRandom.out");
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
