// @(#)root/mathcore:$Id$
// Authors: David Gonzalez Maline    01/2008

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

#include "Math/GaussIntegrator.h"
#include "Math/IntegratorOptions.h"
#include <cmath>

namespace ROOT {
namespace Math {


bool GaussIntegrator::fgAbsValue = false;

   GaussIntegrator::GaussIntegrator(double epsabs, double epsrel)
{
// Default Constructor. If no relative espsilon is specified assume is equal to the absolute one

   if (epsabs <= 0 ) {
      fEpsAbs = ROOT::Math::IntegratorOneDimOptions::DefaultAbsTolerance();
      fEpsRel = ROOT::Math::IntegratorOneDimOptions::DefaultRelTolerance();
   }
   else {
      fEpsAbs = epsabs;
      if (epsrel <= 0 ) fEpsRel = epsabs;   // use relative tiolerance = to abs if it is given only one
      else fEpsRel = epsrel;
   }
   fLastResult = fLastError = 0;
   fUsedOnce = false;
   fFunction = 0;
}

GaussIntegrator::~GaussIntegrator()
{
   // Destructor. (no - operations)
}

void GaussIntegrator::AbsValue(bool flag)
{   fgAbsValue = flag;  }

double GaussIntegrator::Integral(double a, double b) {
   return DoIntegral(a, b, fFunction);
}

double GaussIntegrator::Integral () {
   IntegrandTransform it(this->fFunction);
   return DoIntegral(0., 1., it.Clone());
}

double GaussIntegrator::IntegralUp (double a) {
   IntegrandTransform it(a, IntegrandTransform::kPlus, this->fFunction);
   return DoIntegral(0., 1., it.Clone());
}

double GaussIntegrator::IntegralLow (double b) {
   IntegrandTransform it(b, IntegrandTransform::kMinus, this->fFunction);
   return DoIntegral(0., 1., it.Clone());
}

double GaussIntegrator::DoIntegral(double a, double b, const IGenFunction* function)
{
   //  Return Integral of function between a and b.

   if (fEpsRel <=0 || fEpsAbs <= 0) {
      if (fEpsRel > 0) fEpsAbs = fEpsRel;
      else if (fEpsAbs > 0) fEpsRel = fEpsAbs;
      else {
         MATH_INFO_MSG("ROOT::Math::GausIntegratorOneDim", "Invalid tolerance given - use default values");
         fEpsRel =  ROOT::Math::IntegratorOneDimOptions::DefaultRelTolerance();
         fEpsAbs =  ROOT::Math::IntegratorOneDimOptions::DefaultAbsTolerance();
      }
   }

   const double kHF = 0.5;
   const double kCST = 5./1000;

   double x[12] = { 0.96028985649753623,  0.79666647741362674,
                      0.52553240991632899,  0.18343464249564980,
                      0.98940093499164993,  0.94457502307323258,
                      0.86563120238783174,  0.75540440835500303,
                      0.61787624440264375,  0.45801677765722739,
                      0.28160355077925891,  0.09501250983763744};

   double w[12] = { 0.10122853629037626,  0.22238103445337447,
                      0.31370664587788729,  0.36268378337836198,
                      0.02715245941175409,  0.06225352393864789,
                      0.09515851168249278,  0.12462897125553387,
                      0.14959598881657673,  0.16915651939500254,
                      0.18260341504492359,  0.18945061045506850};

   double h, aconst, bb, aa, c1, c2, u, s8, s16, f1, f2;
   double xx[1];
   int i;

   if ( fFunction == 0 )
   {
      MATH_ERROR_MSG("ROOT::Math::GausIntegratorOneDim", "A function must be set first!");
      return 0.0;
   }

   h = 0;
   fUsedOnce = true;
   if (b == a) return h;
   aconst = kCST/std::abs(b-a);
   bb = a;
CASE1:
   aa = bb;
   bb = b;
CASE2:
   c1 = kHF*(bb+aa);
   c2 = kHF*(bb-aa);
   s8 = 0;
   for (i=0;i<4;i++) {
      u     = c2*x[i];
      xx[0] = c1+u;
      f1    = (*function)(xx);
      if (fgAbsValue) f1 = std::abs(f1);
      xx[0] = c1-u;
      f2    = (*function) (xx);
      if (fgAbsValue) f2 = std::abs(f2);
      s8   += w[i]*(f1 + f2);
   }
   s16 = 0;
   for (i=4;i<12;i++) {
      u     = c2*x[i];
      xx[0] = c1+u;
      f1    = (*function) (xx);
      if (fgAbsValue) f1 = std::abs(f1);
      xx[0] = c1-u;
      f2    = (*function) (xx);
      if (fgAbsValue) f2 = std::abs(f2);
      s16  += w[i]*(f1 + f2);
   }
   s16 = c2*s16;
   //if (std::abs(s16-c2*s8) <= fEpsilon*(1. + std::abs(s16))) {
   Double_t error = std::abs(s16-c2*s8);
   if (error <= fEpsAbs || error <= fEpsRel*std::abs(s16)) {
      h += s16;
      if(bb != b) goto CASE1;
   } else {
      bb = c1;
      if(1. + aconst*std::abs(c2) != 1) goto CASE2;
      MATH_WARN_MSG("ROOT::Math::GausIntegratorOneDim", "Failed to reach the desired tolerance");
      h = s8;  //this is a crude approximation (cernlib function returned 0 !)
   }

   fLastResult = h;
   fLastError = std::abs(s16-c2*s8);

   return h;
}


double GaussIntegrator::Result () const
{
   // Returns the result of the last Integral calculation.

   if (!fUsedOnce)
      MATH_ERROR_MSG("ROOT::Math::GaussIntegrator", "You must calculate the result at least once!");

   return fLastResult;
}

double GaussIntegrator::Error() const
{   return fLastError;  }

int GaussIntegrator::Status() const
{   return (fUsedOnce) ? 0 : -1;  }

void GaussIntegrator::SetFunction (const IGenFunction & function)
{
   // Set integration function
   fFunction = &function;
   // reset fUsedOne flag
   fUsedOnce = false;
}

double GaussIntegrator::Integral (const std::vector< double > &/*pts*/)
{
   // not impl.
   MATH_WARN_MSG("ROOT::Math::GaussIntegrator", "This method is not implemented in this class !");
   return -1.0;
}

double GaussIntegrator::IntegralCauchy (double /*a*/, double /*b*/, double /*c*/)
{
   // not impl.
   MATH_WARN_MSG("ROOT::Math::GaussIntegrator", "This method is not implemented in this class !");
   return -1.0;
}

void GaussIntegrator::SetOptions(const ROOT::Math::IntegratorOneDimOptions & opt) {
   // set options
   SetRelTolerance(opt.RelTolerance() );
   SetAbsTolerance(opt.AbsTolerance() );
}

ROOT::Math::IntegratorOneDimOptions  GaussIntegrator::Options() const {
   // retrieve options
   ROOT::Math::IntegratorOneDimOptions opt;
   opt.SetIntegrator("Gauss");
   opt.SetAbsTolerance(fEpsAbs);
   opt.SetRelTolerance(fEpsRel);
   opt.SetWKSize(0);
   opt.SetNPoints(0);
   return opt;
}



//implementation of IntegrandTransform class

IntegrandTransform::IntegrandTransform(const IGenFunction* integrand)
   : fSign(kPlus), fIntegrand(integrand), fBoundary(0.), fInfiniteInterval(true) {
}

IntegrandTransform::IntegrandTransform(const double boundary, ESemiInfinitySign sign, const IGenFunction* integrand)
   : fSign(sign), fIntegrand(integrand), fBoundary(boundary), fInfiniteInterval(false)  {
}

double IntegrandTransform::DoEval(double x) const {
   double result = DoEval(x, fBoundary, fSign);
   return (result += (fInfiniteInterval ? DoEval(x, 0., -1) : 0.));
}

double IntegrandTransform::DoEval(double x, double boundary, int sign) const {
   double mappedX = 1. / x - 1.;
   return (*fIntegrand)(boundary + sign * mappedX) * std::pow(mappedX + 1., 2);;
}

double IntegrandTransform::operator()(double x) const {
   return DoEval(x);
}

IGenFunction* IntegrandTransform::Clone() const {
   return (fInfiniteInterval ? new IntegrandTransform(fIntegrand) : new IntegrandTransform(fBoundary, fSign, fIntegrand));
}


} // end namespace Math
} // end namespace ROOT
