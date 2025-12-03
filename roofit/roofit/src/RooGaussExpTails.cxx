/** \class RooGaussExpTails
    \ingroup Roofit

PDF implementing a Gaussian core + double-sided exponential tail distribution
\author Souvik Das (8/1/2013) Initial implementation and
Giovanni Marchiori (30/3/2016) Implemented analytic integral
\see http://arxiv.org/pdf/1603.08591v1.pdf, https://github.com/mpuccio/RooCustomPdfs/blob/master/RooGausDExp.cxx, https://doi.org/10.1142/S0217751X14300440
\note To use the one-sided version, just set the opposite parameter k to a very large value
*/


#include "RooGaussExpTails.h"
#include "RooAbsReal.h"
#include <cmath>
#include "Math/ProbFuncMathCore.h"

//_____________________________________________________________________________
RooGaussExpTails::RooGaussExpTails(const char *name, const char *title, RooAbsReal::Ref x, RooAbsReal::Ref x0,
                                   RooAbsReal::Ref sigma, RooAbsReal::Ref kL, RooAbsReal::Ref kH)
   : RooAbsPdf(name, title),
     _x("x", "x", this, x),
     _x0("x0", "x0", this, x0),
     _sigma("sigma", "sigma", this, sigma),
     _kL("kL", "kL", this, kL),
     _kH("kH", "kH", this, kH)
{
}

//_____________________________________________________________________________
RooGaussExpTails::RooGaussExpTails(const RooGaussExpTails &other, const char* name)
   : RooAbsPdf(other, name),
     _x("x", this, other._x),
     _x0("x0", this, other._x0),
     _sigma("sigma", this, other._sigma),
     _kL("kL", this, other._kL),
     _kH("kH", this, other._kH)
{
}

////////////////////////////////////////////////////////////////////////////////

namespace {

inline double gaussianIntegral(double tmin, double tmax)
{
   constexpr double m_sqrt_2_pi = 2.50662827463; // std::sqrt(TMath::TwoPi())
   return m_sqrt_2_pi * (ROOT::Math::gaussian_cdf(tmax) - ROOT::Math::gaussian_cdf(tmin));
}

inline double tailIntegral(double tmin, double tmax, double k)
{
   double a = std::exp(0.5 * k * k) / k;
   return a * (std::exp(k * tmax) - std::exp(k * tmin));
}

} // namespace

//_____________________________________________________________________________
Double_t RooGaussExpTails::evaluate() const
{
   Double_t t = (_x - _x0) / _sigma;

   if (t <= -_kL)
      return std::exp(0.5 * _kL * _kL + _kL * t);
   else if (t > _kH)
      return std::exp(0.5 * _kH * _kH - _kH * t);
   else
      return std::exp(-0.5 * t * t);
}

//_____________________________________________________________________________
Int_t RooGaussExpTails::getAnalyticalIntegral(RooArgSet &allVars, RooArgSet &analVars, const char * /*rangeName*/) const
{
   if (matchArgs(allVars, analVars, _x))
      return 1;

   return 0;
}

//_____________________________________________________________________________
Double_t RooGaussExpTails::analyticalIntegral(Int_t code, const char *rangeName) const
{
   R__ASSERT(code == 1);
   double result = 0;

   double sig = std::abs((Double_t)_sigma);
   double tmin = (_x.min(rangeName) - _x0) / sig;
   double tmax = (_x.max(rangeName) - _x0) / sig;

   if (tmin <= -_kL)
      result += tailIntegral(tmin, std::min(tmax, -_kL), _kL);
   if (tmin <= _kH && tmax > -_kL)
      result += gaussianIntegral(std::max(tmin, -_kL), std::min(tmax, +_kH));
   if (tmax > _kH)
      result += tailIntegral(std::max(tmin, +_kH), tmax, -_kH);

   return sig * result;
}
