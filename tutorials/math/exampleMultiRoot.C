/// \file
/// \ingroup tutorial_math
/// \notebook -nodraw
/// Example of using multiroot finder based on GSL algorithm.
/// Find the root of Rosenbrock system of equations:
/// \f[
/// f1(x,y) = a(1-x)
/// \f]
/// \f[
/// f2(x,y) = b(y-x^2)
/// \f]
/// with:
/// \f[
/// a = 1, b=10
/// \f]
///
/// The MultiRootFinder is based on GSL and it requires the MathMore library
/// installed
///
/// Usage:
///
/// ~~~{.cpp}
///  >.x exampleMultiRoot.C()
/// ~~~
///
/// or
///
/// ~~~{.cpp}
/// >.x exampleMultiRoot(algoname,printlevel)
/// ~~~
///
/// where algoname is for an algorithm not using the derivatives:
///  hybridS (default) , hybrid, dnewton, broyden
///
/// \macro_output
/// \macro_code
///
/// \author Lorenzo Moneta

#include "RConfigure.h"

#ifdef R__HAS_MATHMORE
#include "Math/MultiRootFinder.h"
#else
#error libMathMore is not available - cannot run this tutorial
#endif
#include "Math/WrappedMultiTF1.h"
#include "TF2.h"
#include "TError.h"

// example of using multi root finder based on GSL
// need to use an algorithm not requiring the derivative
//like hybrids (default), hybrid, dnewton, broyden

using namespace ROOT::Math;
void exampleMultiRoot(const char * algo = 0, int printlevel = 1) {
   ROOT::Math::MultiRootFinder r(algo);
   //defining the function
   // use Rosenbrock functions
   TF2 * f1 = new TF2("f1","[0]*(1-x)+[1]*y");
   TF2 * f2 = new TF2("f2","[0]*(y-x*x)");
   f1->SetParameters(1,0);
   f2->SetParameter(0,10);
   // wrap the functions
   ROOT::Math::WrappedMultiTF1 g1(*f1,2);
   ROOT::Math::WrappedMultiTF1 g2(*f2,2);
   r.AddFunction(g1);
   r.AddFunction(g2);
   r.SetPrintLevel(printlevel);

   // starting point
   double x0[2]={-1,-1};
   r.Solve(x0);
}
