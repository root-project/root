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
RooGaussExpTails::RooGaussExpTails(const char *name, const char *title,
                            RooAbsReal::Ref _x,
                            RooAbsReal::Ref _x0,
                            RooAbsReal::Ref _sigma,
                            RooAbsReal::Ref _kL,
                            RooAbsReal::Ref _kH) :
     RooAbsPdf(name,title),
     x_("x","x",this,_x),
     x0_("x0","x0",this,_x0),
     sigma_("sigma","sigma",this,_sigma),
     kL_("kL","kL",this,_kL),
     kH_("kH","kH",this,_kH)
{
}


//_____________________________________________________________________________
RooGaussExpTails::RooGaussExpTails(const RooGaussExpTails& other, const char* name) :
     RooAbsPdf(other,name),
     x_("x",this,other.x_),
     x0_("x0",this,other.x0_),
     sigma_("sigma",this,other.sigma_),
     kL_("kL",this,other.kL_),
     kH_("kH",this,other.kH_)
{
}

////////////////////////////////////////////////////////////////////////////////

namespace {

inline double gaussianIntegral(double tmin, double tmax)
{
   constexpr double m_sqrt_2_pi = 2.50662827463;//std::sqrt(TMath::TwoPi())
   return m_sqrt_2_pi*(ROOT::Math::gaussian_cdf(tmax) - ROOT::Math::gaussian_cdf(tmin));
}

inline double tailIntegral(double tmin, double tmax, double k)
{
   double a = std::exp(0.5*k*k)/k;
   return (a*(std::exp(k*tmax)-std::exp(k*tmin)));
}

} // namespace



//_____________________________________________________________________________
Double_t RooGaussExpTails::evaluate() const
{
   Double_t t=(x_-x0_)/sigma_;

   if (t<=-kL_)
      return exp(0.5*kL_*kL_+kL_*t);
   else if (t>kH_)
      return exp(0.5*kH_*kH_-kH_*t);
   else
      return exp(-0.5*t*t);
}


//_____________________________________________________________________________
Int_t RooGaussExpTails::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const
{
   if( matchArgs(allVars,analVars,x_) )
      return 1;

   return 0;
}


//_____________________________________________________________________________
Double_t RooGaussExpTails::analyticalIntegral(Int_t code, const char* rangeName) const
{
   R__ASSERT(code == 1);
   double result = 0;

   double sig = std::abs((Double_t)sigma_);
   double tmin = (x_.min(rangeName)-x0_)/sig;
   double tmax = (x_.max(rangeName)-x0_)/sig;

   if (tmin <= -kL_)
      result += tailIntegral(tmin, std::min(tmax, -kL_), kL_);
   if (tmin <= kH_ && tmax > -kL_)
      result += gaussianIntegral(std::max(tmin, -kL_), std::min(tmax, kH_));
   if (tmax > kH_)
      result += tailIntegral(std::max(tmin, kH_), tmax, -kH_);

   return sig*result;
}
