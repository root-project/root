/**
 * \class RooStudentT
 * \ingroup Roofit
 *
 * Location-scale Student's t-distribution
 * \see https://en.wikipedia.org/wiki/Student's_t-distribution#Location-scale_t-distribution
 */

#include "RooStudentT.h"
#include "RooHelpers.h"

#include "TMath.h"

RooStudentT::RooStudentT(const char *name, const char *title, RooAbsReal::Ref x, RooAbsReal::Ref mean,
                         RooAbsReal::Ref sigma, RooAbsReal::Ref ndf)
   : RooAbsPdf(name, title),
     _x("x", "Observable", this, x),
     _mean("mean", "Mean", this, mean),
     _sigma("sigma", "Width", this, sigma),
     _ndf("ndf", " Degrees of freedom", this, ndf)
{
   RooHelpers::checkRangeOfParameters(this, {&static_cast<RooAbsReal &>(sigma)}, 0);
   RooHelpers::checkRangeOfParameters(this, {&static_cast<RooAbsReal &>(ndf)}, 1);
}

RooStudentT::RooStudentT(const RooStudentT &other, const char *name)
   : RooAbsPdf(other, name),
     _x("x", this, other._x),
     _mean("mean", this, other._mean),
     _sigma("sigma", this, other._sigma),
     _ndf("ndf", this, other._ndf)
{
}

double RooStudentT::evaluate() const
{
   const double t = (_x - _mean) / _sigma;
   return TMath::Student(t, _ndf);
}

Int_t RooStudentT::getAnalyticalIntegral(RooArgSet &allVars, RooArgSet &analVars, const char * /*rangeName*/) const
{
   return matchArgs(allVars, analVars, _x) ? 1 : 0;
}

double RooStudentT::analyticalIntegral(Int_t code, const char *rangeName) const
{
   R__ASSERT(code == 1);

   double tmin = (_x.min(rangeName) - _mean) / _sigma;
   double tmax = (_x.max(rangeName) - _mean) / _sigma;
   return (TMath::StudentI(tmax, _ndf) - TMath::StudentI(tmin, _ndf)) * _sigma;
}

/// Advertise that we know the maximum of self for given (mean,ndf,sigma).
Int_t RooStudentT::getMaxVal(const RooArgSet &vars) const
{
   RooArgSet dummy;

   if (matchArgs(vars, dummy, _x)) {
      return 1;
   }
   return 0;
}

double RooStudentT::maxVal(Int_t code) const
{
   R__ASSERT(code == 1);

   return TMath::Student(0, _ndf);
}
