 /*****************************************************************************
  * Project: RooFit                                                           *
  *                                                                           *
  * Simple Poisson PDF
  * author: Kyle Cranmer <cranmer@cern.ch>
  *                                                                           *
  *****************************************************************************/

/** \class RooPoisson
    \ingroup Roofit

Poisson pdf
**/

#include "RooPoisson.h"
#include "RooRandom.h"
#include "RooMath.h"
#include "RooNaNPacker.h"
#include "RooBatchCompute.h"

#include "RooFit/Detail/AnalyticalIntegrals.h"
#include "RooFit/Detail/EvaluateFuncs.h"
#include "Math/ProbFuncMathCore.h"

ClassImp(RooPoisson);

////////////////////////////////////////////////////////////////////////////////
/// Constructor

RooPoisson::RooPoisson(const char *name, const char *title,
             RooAbsReal::Ref _x,
             RooAbsReal::Ref _mean,
             bool noRounding) :
  RooAbsPdf(name,title),
  x("x","x",this,_x),
  mean("mean","mean",this,_mean),
  _noRounding(noRounding)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

 RooPoisson::RooPoisson(const RooPoisson& other, const char* name) :
   RooAbsPdf(other,name),
   x("x",this,other.x),
   mean("mean",this,other.mean),
   _noRounding(other._noRounding),
   _protectNegative(other._protectNegative)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Implementation in terms of the TMath::Poisson() function.

double RooPoisson::evaluate() const
{
  double k = _noRounding ? x : floor(x);
  if(_protectNegative && mean<0) {
    RooNaNPacker np;
    np.setPayload(-mean);
    return np._payload;
  }
  return RooFit::Detail::EvaluateFuncs::poissonEvaluate(k, mean);
}

void RooPoisson::translate(RooFit::Detail::CodeSquashContext &ctx) const
{
   std::string xName = ctx.getResult(x);
   if (!_noRounding)
      xName = "std::floor(" + xName + ")";

   ctx.addResult(this, ctx.buildCall("RooFit::Detail::EvaluateFuncs::poissonEvaluate", xName, mean));
}

////////////////////////////////////////////////////////////////////////////////
/// Compute multiple values of the Poisson distribution.
void RooPoisson::computeBatch(cudaStream_t *stream, double *output, size_t nEvents,
                              RooFit::Detail::DataMap const &dataMap) const
{
   auto dispatch = stream ? RooBatchCompute::dispatchCUDA : RooBatchCompute::dispatchCPU;
   RooBatchCompute::ArgVector extraArgs{static_cast<double>(_protectNegative), static_cast<double>(_noRounding)};
   dispatch->compute(stream, RooBatchCompute::Poisson, output, nEvents, {dataMap.at(x), dataMap.at(mean)}, extraArgs);
}

////////////////////////////////////////////////////////////////////////////////

Int_t RooPoisson::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const
{
  if (matchArgs(allVars,analVars,x)) return 1 ;
  if (matchArgs(allVars, analVars, mean)) return 2;
  return 0 ;
}

////////////////////////////////////////////////////////////////////////////////

double RooPoisson::analyticalIntegral(Int_t code, const char* rangeName) const
{
  R__ASSERT(code == 1 || code == 2) ;

  RooRealProxy const &integrand = code == 1 ? x : mean;
  return RooFit::Detail::AnalyticalIntegrals::poissonIntegral(
     code, mean, _noRounding ? x : std::floor(x), integrand.min(rangeName), integrand.max(rangeName), _protectNegative);
}

std::string
RooPoisson::buildCallToAnalyticIntegral(int code, const char *rangeName, RooFit::Detail::CodeSquashContext &ctx) const
{
   R__ASSERT(code == 1 || code == 2);
   std::string xName = ctx.getResult(x);
   if (!_noRounding)
      xName = "std::floor(" + xName + ")";

   RooRealProxy const &integrand = code == 1 ? x : mean;
   // Since the integral function is the same for both codes, we need to make sure the indexed observables do not appear
   // in the function if they are not required.
   xName = code == 1 ? "0" : xName;
   return ctx.buildCall("RooFit::Detail::AnalyticalIntegrals::poissonIntegral", code, mean, xName,
                        integrand.min(rangeName), integrand.max(rangeName), _protectNegative);
}

////////////////////////////////////////////////////////////////////////////////
/// Advertise internal generator in x

Int_t RooPoisson::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, bool /*staticInitOK*/) const
{
  if (matchArgs(directVars,generateVars,x)) return 1 ;
  return 0 ;
}

////////////////////////////////////////////////////////////////////////////////
/// Implement internal generator using TRandom::Poisson

void RooPoisson::generateEvent(Int_t code)
{
  R__ASSERT(code==1) ;
  double xgen ;
  while(1) {
    xgen = RooRandom::randomGenerator()->Poisson(mean);
    if (xgen<=x.max() && xgen>=x.min()) {
      x = xgen ;
      break;
    }
  }
  return;
}
