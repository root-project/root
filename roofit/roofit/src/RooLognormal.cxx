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
  - m0: the median of the distribution
  - k=exp(sigma): sigma is called the shape parameter in the TMath parameterization

\f[ Lognormal(x,m_0,k) = \frac{e^{(-ln^2(x/m_0))/(2ln^2(k))}}{\sqrt{2\pi \cdot ln(k)\cdot x}} \f]

The parameterization here is physics driven and differs from the ROOT::Math::lognormal_pdf(x,m,s,x0) with:
  - m = log(m0)
  - s = log(k)
  - x0 = 0
**/

#include "RooLognormal.h"
#include "RooFit.h"
#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "RooRandom.h"
#include "RooMath.h"
#include "RooVDTHeaders.h"
#include "RooHelpers.h"
#include "BatchHelpers.h"

#include "TMath.h"
#include "TClass.h"
#include <Math/SpecFuncMathCore.h>
#include <Math/PdfFuncMathCore.h>
#include <Math/ProbFuncMathCore.h>

#include <cmath>
using namespace std;

ClassImp(RooLognormal);

////////////////////////////////////////////////////////////////////////////////

RooLognormal::RooLognormal(const char *name, const char *title,
          RooAbsReal& _x, RooAbsReal& _m0,
          RooAbsReal& _k) :
  RooAbsPdf(name,title),
  x("x","Observable",this,_x),
  m0("m0","m0",this,_m0),
  k("k","k",this,_k)
{
    RooHelpers::checkRangeOfParameters(this, {&_x, &_m0, &_k}, 0.);
    
    auto par = dynamic_cast<const RooAbsRealLValue*>(&_k);
    if (par && par->getMin()<=1 && par->getMax()>=1 ) {
      oocoutE(this, InputArguments) << "The parameter '" << par->GetName() << "' with range [" << par->getMin("") << ", "
          << par->getMax() << "] of the " << this->IsA()->GetName() << " '" << this->GetName()
          << "' can reach the unsafe value 1.0 " << ". Advise to limit its range." << std::endl;
    }
}

////////////////////////////////////////////////////////////////////////////////

RooLognormal::RooLognormal(const RooLognormal& other, const char* name) :
  RooAbsPdf(other,name), x("x",this,other.x), m0("m0",this,other.m0),
  k("k",this,other.k)
{
}

////////////////////////////////////////////////////////////////////////////////
/// ln(k)<1 would correspond to sigma < 0 in the parameterization
/// resulting by transforming a normal random variable in its
/// standard parameterization to a lognormal random variable
/// => treat ln(k) as -ln(k) for k<1

Double_t RooLognormal::evaluate() const
{
  Double_t ln_k = TMath::Abs(TMath::Log(k));
  Double_t ln_m0 = TMath::Log(m0);

  Double_t ret = ROOT::Math::lognormal_pdf(x,ln_m0,ln_k);
  return ret ;
}

////////////////////////////////////////////////////////////////////////////////

namespace {
//Author: Emmanouil Michalainas, CERN 10 September 2019

template<class Tx, class Tm0, class Tk>
void compute(	size_t batchSize,
              double * __restrict output,
              Tx X, Tm0 M0, Tk K)
{
  const double rootOf2pi = std::sqrt(2 * M_PI);
  for (size_t i=0; i<batchSize; i++) {
    double lnxOverM0 = _rf_fast_log(X[i]/M0[i]);
    double lnk = _rf_fast_log(K[i]);
    if (lnk<0) lnk = -lnk;
    double arg = lnxOverM0/lnk;
    arg *= -0.5*arg;
    output[i] = _rf_fast_exp(arg) / (X[i]*lnk*rootOf2pi);
  }
}
};

RooSpan<double> RooLognormal::evaluateBatch(std::size_t begin, std::size_t batchSize) const {
  using namespace BatchHelpers;
  auto xData = x.getValBatch(begin, batchSize);
  auto m0Data = m0.getValBatch(begin, batchSize);
  auto kData = k.getValBatch(begin, batchSize);
  const bool batchX = !xData.empty();
  const bool batchM0 = !m0Data.empty();
  const bool batchK = !kData.empty();

  if (!batchX && !batchM0 && !batchK) {
    return {};
  }
  batchSize = findSize({ xData, m0Data, kData });
  auto output = _batchData.makeWritableBatchUnInit(begin, batchSize);

  if (batchX && !batchM0 && !batchK ) {
    compute(batchSize, output.data(), xData, BracketAdapter<double>(m0), BracketAdapter<double>(k));
  }
  else if (!batchX && batchM0 && !batchK ) {
    compute(batchSize, output.data(), BracketAdapter<double>(x), m0Data, BracketAdapter<double>(k));
  }
  else if (batchX && batchM0 && !batchK ) {
    compute(batchSize, output.data(), xData, m0Data, BracketAdapter<double>(k));
  }
  else if (!batchX && !batchM0 && batchK ) {
    compute(batchSize, output.data(), BracketAdapter<double>(x), BracketAdapter<double>(m0), kData);
  }
  else if (batchX && !batchM0 && batchK ) {
    compute(batchSize, output.data(), xData, BracketAdapter<double>(m0), kData);
  }
  else if (!batchX && batchM0 && batchK ) {
    compute(batchSize, output.data(), BracketAdapter<double>(x), m0Data, kData);
  }
  else if (batchX && batchM0 && batchK ) {
    compute(batchSize, output.data(), xData, m0Data, kData);
  }
  return output;
}


////////////////////////////////////////////////////////////////////////////////

Int_t RooLognormal::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const
{
  if (matchArgs(allVars,analVars,x)) return 1 ;
  return 0 ;
}

////////////////////////////////////////////////////////////////////////////////

Double_t RooLognormal::analyticalIntegral(Int_t code, const char* rangeName) const
{
  R__ASSERT(code==1) ;

  static const Double_t root2 = sqrt(2.) ;

  Double_t ln_k = TMath::Abs(TMath::Log(k));
  Double_t ret = 0.5*( RooMath::erf( TMath::Log(x.max(rangeName)/m0)/(root2*ln_k) ) - RooMath::erf( TMath::Log(x.min(rangeName)/m0)/(root2*ln_k) ) ) ;

  return ret ;
}

////////////////////////////////////////////////////////////////////////////////

Int_t RooLognormal::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t /*staticInitOK*/) const
{
  if (matchArgs(directVars,generateVars,x)) return 1 ;
  return 0 ;
}

////////////////////////////////////////////////////////////////////////////////

void RooLognormal::generateEvent(Int_t code)
{
  R__ASSERT(code==1) ;

  Double_t xgen ;
  while(1) {
    xgen = TMath::Exp(RooRandom::randomGenerator()->Gaus(TMath::Log(m0),TMath::Log(k)));
    if (xgen<=x.max() && xgen>=x.min()) {
      x = xgen ;
      break;
    }
  }

  return;
}
