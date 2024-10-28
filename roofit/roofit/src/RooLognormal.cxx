/*****************************************************************************
 * Project: RooFit                                                           *
 * @(#)root/roofit:$Id$ *
 *                                                                           *
 * RooFit Lognormal PDF                                                      *
 *                                                                           *
 * Author: Gregory Schott and Stefan Schmitz                                 *
 *                                                                           *
 *****************************************************************************/

/** \class RooLognormal
    \ingroup Roofit

RooFit Lognormal PDF. The two parameters are:
  - `m0`: the median of the distribution
  - `k = exp(sigma)`: sigma is called the shape parameter in the TMath parameterization

\f[
  \mathrm{RooLognormal}(x \, | \, m_0, k) = \frac{1}{\sqrt{2\pi \cdot \ln(k) \cdot x}} \cdot \exp\left(
\frac{-\ln^2(\frac{x}{m_0})}{2 \ln^2(k)} \right) \f]

The parameterization here is physics driven and differs from the ROOT::Math::lognormal_pdf() in `x,m,s,x0` with:
  - `m = log(m0)`
  - `s = log(k)`
  - `x0 = 0`
**/

#include "RooLognormal.h"
#include "RooRandom.h"
#include "RooMath.h"
#include "RooHelpers.h"
#include "RooBatchCompute.h"

#include <Math/PdfFuncMathCore.h>

ClassImp(RooLognormal);

////////////////////////////////////////////////////////////////////////////////

RooLognormal::RooLognormal(const char *name, const char *title, RooAbsReal &_x, RooAbsReal &_m0, RooAbsReal &_k,
                           bool useStandardParametrization)
   : RooAbsPdf{name, title},
     x{"x", "Observable", this, _x},
     m0{"m0", "m0", this, _m0},
     k{"k", "k", this, _k},
     _useStandardParametrization{useStandardParametrization}
{
   RooHelpers::checkRangeOfParameters(this, {&_x, &_m0, &_k}, 0.);

   auto par = dynamic_cast<const RooAbsRealLValue *>(&_k);
   const double unsafeValue = useStandardParametrization ? 0.0 : 1.0;
   if (par && par->getMin() <= unsafeValue && par->getMax() >= unsafeValue) {
      coutE(InputArguments) << "The parameter '" << par->GetName() << "' with range [" << par->getMin("") << ", "
                            << par->getMax() << "] of the " << this->ClassName() << " '" << this->GetName()
                            << "' can reach the unsafe value " << unsafeValue << " "
                            << ". Advise to limit its range." << std::endl;
   }
}

////////////////////////////////////////////////////////////////////////////////

RooLognormal::RooLognormal(const RooLognormal &other, const char *name)
   : RooAbsPdf(other, name),
     x("x", this, other.x),
     m0("m0", this, other.m0),
     k{"k", this, other.k},
     _useStandardParametrization{other._useStandardParametrization}
{
}

////////////////////////////////////////////////////////////////////////////////
/// ln(k)<1 would correspond to sigma < 0 in the parameterization
/// resulting by transforming a normal random variable in its
/// standard parameterization to a lognormal random variable
/// => treat ln(k) as -ln(k) for k<1

double RooLognormal::evaluate() const
{
   const double ln_k = std::abs(_useStandardParametrization ? k : std::log(k));
   const double ln_m0 = _useStandardParametrization ? m0 : std::log(m0);

   return ROOT::Math::lognormal_pdf(x, ln_m0, ln_k);
}

////////////////////////////////////////////////////////////////////////////////
/// Compute multiple values of Lognormal distribution.
void RooLognormal::doEval(RooFit::EvalContext &ctx) const
{
   auto computer = _useStandardParametrization ? RooBatchCompute::LognormalStandard : RooBatchCompute::Lognormal;
   RooBatchCompute::compute(ctx.config(this), computer, ctx.output(),
                            {ctx.at(x), ctx.at(m0), ctx.at(k)});
}

////////////////////////////////////////////////////////////////////////////////

Int_t RooLognormal::getAnalyticalIntegral(RooArgSet &allVars, RooArgSet &analVars, const char * /*rangeName*/) const
{
   return matchArgs(allVars, analVars, x) ? 1 : 0;
}

////////////////////////////////////////////////////////////////////////////////

double RooLognormal::analyticalIntegral(Int_t /*code*/, const char *rangeName) const
{
   static const double root2 = std::sqrt(2.);

   double ln_k = std::abs(_useStandardParametrization ? k : std::log(k));
   double scaledMin = _useStandardParametrization ? std::log(x.min(rangeName)) - m0 : std::log(x.min(rangeName) / m0);
   double scaledMax = _useStandardParametrization ? std::log(x.max(rangeName)) - m0 : std::log(x.max(rangeName) / m0);
   return 0.5 * (RooMath::erf(scaledMax / (root2 * ln_k)) - RooMath::erf(scaledMin / (root2 * ln_k)));
}

////////////////////////////////////////////////////////////////////////////////

Int_t RooLognormal::getGenerator(const RooArgSet &directVars, RooArgSet &generateVars, bool /*staticInitOK*/) const
{
   return matchArgs(directVars, generateVars, x) ? 1 : 0;
}

////////////////////////////////////////////////////////////////////////////////

void RooLognormal::generateEvent(Int_t /*code*/)
{
   const double ln_k = std::abs(_useStandardParametrization ? k : std::log(k));
   const double ln_m0 = _useStandardParametrization ? m0 : std::log(m0);

   double xgen;
   while (true) {
      xgen = std::exp(RooRandom::randomGenerator()->Gaus(ln_m0, ln_k));
      if (xgen <= x.max() && xgen >= x.min()) {
         x = xgen;
         break;
      }
   }

   return;
}
