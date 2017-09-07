#include "Math/Functor.h"
#include "Math/MultiRootFinder.h"

#ifdef HAVE_ROOTLIBS
#include "TStopwatch.h"
#else
struct TStopwatch {
   void Start(){}
   void Stop(){}
   void Reset(){}
   double RealTime() { return 0; }
   double CpuTime() { return 0; }
};
#endif

#include <iostream>
#include <stdlib.h>

// solve Roots of rosenbrock function
// f1(x,y) = a(1-x)
// f2(x,y) = b(y-x^2)
//  with 1 = 1, b=10


// define system of functions to find the roots
struct FuncSystem {

   double F1(const double *xx) {
      double x = xx[0];
      return a * (1. - x );
   }
   double F2(const double *xx) {
      double x = xx[0]; double y = xx[1];
      return b * (y - x*x );
   }

   // derivative
   double DerivF1(const double *, int icoord) {
      if (icoord == 0) return -a;
      else return 0;
   }
   double DerivF2(const double *xx, int icoord) {
      double x = xx[0];
      if (icoord == 0) return -2 * b * x;
      else return b;
   }

   double a;
   double b;
};

int printlevel = 0;

using namespace ROOT::Math;

int testMultiRootFinder() {

  int status = 0;

  // methods using derivatives
  FuncSystem f;
  f.a = 1;
  f.b = 10;


  MultiRootFinder rf(MultiRootFinder::kHybridSJ);
  GradFunctor g1(&f, &FuncSystem::F1, &FuncSystem::DerivF1,2);
  GradFunctor g2(&f, &FuncSystem::F2, &FuncSystem::DerivF2,2);
  rf.AddFunction(g1);
  rf.AddFunction(g2);
  rf.SetPrintLevel(printlevel);

  double x0[] = {-1.,-1.};

  std::cout << "Testing  Multi-RootFinder - ";
  bool ret = rf.Solve(x0);

  if (!ret) {
     std::cout << rf.Name()  << "\t :   FAILED\n";
     std::cerr << "testMultiRootFinder - Error running derivative algorithm " << std::endl;
     if (printlevel == 0) rf.PrintState(std::cout);
     status += rf.Status();
  }
  else
     std::cout << rf.Name()  << "\t :   OK\n";


  MultiRootFinder rf2(MultiRootFinder::kHybridS);
  Functor f1(&f, &FuncSystem::F1, 2);
  Functor f2(&f, &FuncSystem::F2, 2);
  std::vector<ROOT::Math::IMultiGenFunction*> funlist;
  funlist.push_back(&f1);
  funlist.push_back(&f2);
  rf2.SetFunctionList(funlist.begin(), funlist.end() );

  rf2.SetPrintLevel(printlevel);

  std::cout << "Testing  Multi-RootFinder - "; 
  bool ret2 = rf2.Solve(x0);
  if (!ret2) {
     std::cout << rf2.Name()  << "\t :   FAILED\n";
     std::cout << "\t  FAILED\n";
     std::cerr << "testMultiRootFinder - Error running non-derivative algorithm " << std::endl;
     if (printlevel == 0) rf2.PrintState(std::cout);
     status += 10*rf2.Status();
  }
  else
     std::cout << rf2.Name()  << "\t :   OK\n";

  return status;

}

int main (int argc, char **argv) {
  int status = 0;

  if (argc > 1 )
     printlevel  = atoi(argv[1]);

  status += testMultiRootFinder();
  if (status == 0) {
     std::cout << "testMultiRootFinder --- \t" << "OK" << std::endl;
  }
  else {
     std::cerr << "testMultiRootFinder --- \t" << "FAILED ! " << "\t with status = " << status << std::endl;
  }

  return status;
}
