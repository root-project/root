/** \class PiecewiseInterpolation
* \ingroup HistFactory
* The PiecewiseInterpolation is a class that can morph distributions into each other, which
* is useful to estimate systematic uncertainties. Given a nominal distribution and one or
* more altered or distorted ones, it computes a new shape depending on the value of the nuisance
* parameters \f$ \theta_i \f$:
* \f[
*   A = \mathrm{nominal} + \sum_i I_i(\theta_i;\mathrm{low}_i, \mathrm{nominal}, \mathrm{high}_i).
* \f]
* for additive interpolation modes (interpCodes 0, 2, 3, and 4), or:
* \f[
*   A = \mathrm{nominal}\prod_i I_i(\theta_i;\mathrm{low}_i/\mathrm{nominal}, 1, \mathrm{high}_i/\mathrm{nominal}).
* \f]
* for multiplicative interpolation modes (interpCodes 1, 5, and 6). The interpCodes determine the function \f$ I_i \f$ (see table below).
*
* Note that a PiecewiseInterpolation with \f$ \mathrm{nominal}=1 \f$, N variations, and a multiplicative interpolation mode is equivalent to N
* PiecewiseInterpolations each with a single variation and the same interpolation code, all inside a RooProduct.
*
* If an \f$ \theta_i \f$ is zero, the distribution is identical to the nominal distribution, at
* \f$ \pm 1 \f$ it is identical to the up/down distribution for that specific \f$ i \f$.
*
* PiecewiseInterpolation will behave identically (except for differences in the interpCode assignments) to a FlexibleInterpVar if both its nominal, and high and low variation sets
* are all RooRealVar.
*
* The class supports several interpolation methods, which can be selected for each parameter separately
* using setInterpCode(). The default interpolation code is 0. The table below provides details of the interpCodes:

| **interpCode** | **Name** | **Description** |
|----------------|----------|-----------------|
| 0   (default)  | Additive Piecewise Linear | \f$ I_0(\theta;x_{-},x_0,x_{+}) = \theta(x_{+} - x_0) \f$ for \f$ \theta>=0 \f$, otherwise \f$ \theta(x_0 - x_{-}) \f$. Not recommended except if using a symmetric variation, because of discontinuities in derivatives. |
| 1              | Multiplicative Piecewise Exponential | \f$ I_1(\theta;x_{-},x_0,x_{+}) = (x_{+}/x_0)^{\theta} \f$ for \f$ \theta>=0 \f$, otherwise \f$ (x_{-}/x_0)^{-\theta} \f$. |
| 2              | Additive Quadratic Interp. + Linear Extrap. | Deprecated by interpCode 4. |
| 4              | Additive Poly Interp. + Linear Extrap. | \f$ I_4(\theta;x_{-},x_0,x_{+}) = I_0(\theta;x_{-},x_0,x_{+}) \f$ if \f$ |\theta|>=1 \f$, otherwise \f$ \theta(\frac{x_{+}-x_{-}}{2}+\theta\frac{x_{+}+x_{-}-2x_{0}}{16}(15+\theta^2(3\alpha^2-10))) \f$  (6th-order polynomial through origin for with matching 0th,1st,2nd derivatives at boundary). |
| 5              | Multiplicative Poly Interp. + Exponential Extrap. | \f$ I_5(\theta;x_{-},x_0,x_{+}) = I_1(\theta;x_{-},x_0,x_{+}) \f$ if \f$ |\theta|>=1 \f$, otherwise 6th-order polynomial for \f$ |\theta_i|<1 \f$ with matching 0th,1st,2nd derivatives at boundary. Recommended for normalization factors. In FlexibleInterpVar this is interpCode=4. |
| 6              | Multiplicative Poly Interp. + Linear Extrap. | \f$ I_6(\theta;x_{-},x_0,x_{+}) = 1+I_4(\theta;x_{-},x_0,x_{+}). \f$ Recommended for normalization factors that must not have roots (i.e. be equal to 0) outside of \f$ |\theta_i|<1 \f$. |

*/

#include "RooStats/HistFactory/PiecewiseInterpolation.h"

#include <RooFit/Detail/MathFuncs.h>

#include "Riostream.h"
#include "TBuffer.h"

#include "RooAbsReal.h"
#include "RooAbsPdf.h"
#include "RooErrorHandler.h"
#include "RooArgSet.h"
#include "RooRealVar.h"
#include "RooMsgService.h"
#include "RooNumIntConfig.h"
#include "RooTrace.h"
#include "RooDataHist.h"
#include "RooHistFunc.h"

#include <exception>
#include <cmath>
#include <algorithm>


////////////////////////////////////////////////////////////////////////////////

PiecewiseInterpolation::PiecewiseInterpolation() : _normIntMgr(this)
{
  TRACE_CREATE;
}

////////////////////////////////////////////////////////////////////////////////
/// Construct a new interpolation. The value of the function will be
/// \f[
///   A = \sum_i \mathrm{Interpolate}(\mathrm{low}_i, \mathrm{nominal}, \mathrm{high}_i).
/// \f]
/// \param name Name of the object.
/// \param title Title (for e.g. plotting)
/// \param nominal Nominal value of the function.
/// \param lowSet  Set of down variations.
/// \param highSet Set of up variations.
/// \param paramSet Parameters that control the interpolation.
PiecewiseInterpolation::PiecewiseInterpolation(const char *name, const char *title, const RooAbsReal &nominal,
                                               const RooArgList &lowSet, const RooArgList &highSet,
                                               const RooArgList &paramSet)
   : RooAbsReal(name, title),
     _normIntMgr(this),
     _nominal("!nominal", "nominal value", this, (RooAbsReal &)nominal),
     _lowSet("!lowSet", "low-side variation", this),
     _highSet("!highSet", "high-side variation", this),
     _paramSet("!paramSet", "high-side variation", this),
     _positiveDefinite(false)

{
  // KC: check both sizes
  if (lowSet.size() != highSet.size()) {
    coutE(InputArguments) << "PiecewiseInterpolation::ctor(" << GetName() << ") ERROR: input lists should be of equal length" << std::endl ;
    RooErrorHandler::softAbort() ;
  }

  for (auto *comp : lowSet) {
    if (!dynamic_cast<RooAbsReal*>(comp)) {
      coutE(InputArguments) << "PiecewiseInterpolation::ctor(" << GetName() << ") ERROR: component " << comp->GetName()
             << " in first list is not of type RooAbsReal" << std::endl ;
      RooErrorHandler::softAbort() ;
    }
    _lowSet.add(*comp) ;
  }


  for (auto *comp : highSet) {
    if (!dynamic_cast<RooAbsReal*>(comp)) {
      coutE(InputArguments) << "PiecewiseInterpolation::ctor(" << GetName() << ") ERROR: component " << comp->GetName()
             << " in first list is not of type RooAbsReal" << std::endl ;
      RooErrorHandler::softAbort() ;
    }
    _highSet.add(*comp) ;
  }


  for (auto *comp : paramSet) {
    if (!dynamic_cast<RooAbsReal*>(comp)) {
      coutE(InputArguments) << "PiecewiseInterpolation::ctor(" << GetName() << ") ERROR: component " << comp->GetName()
             << " in first list is not of type RooAbsReal" << std::endl ;
      RooErrorHandler::softAbort() ;
    }
    _paramSet.add(*comp) ;
    _interpCode.push_back(0); // default code: linear interpolation
  }


  // Choose special integrator by default
  specialIntegratorConfig(true)->method1D().setLabel("RooBinIntegrator") ;
  TRACE_CREATE;
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

PiecewiseInterpolation::PiecewiseInterpolation(const PiecewiseInterpolation& other, const char* name) :
  RooAbsReal(other, name),
  _normIntMgr(other._normIntMgr, this),
  _nominal("!nominal",this,other._nominal),
  _lowSet("!lowSet",this,other._lowSet),
  _highSet("!highSet",this,other._highSet),
  _paramSet("!paramSet",this,other._paramSet),
  _positiveDefinite(other._positiveDefinite),
  _interpCode(other._interpCode)
{
  // Member _ownedList is intentionally not copy-constructed -- ownership is not transferred
  TRACE_CREATE;
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

PiecewiseInterpolation::~PiecewiseInterpolation()
{
  TRACE_DESTROY;
}




////////////////////////////////////////////////////////////////////////////////
/// Calculate and return current value of self

double PiecewiseInterpolation::evaluate() const
{
  ///////////////////
  double nominal = _nominal;
  double sum(nominal) ;

  for (unsigned int i=0; i < _paramSet.size(); ++i) {
    auto param = static_cast<RooAbsReal*>(_paramSet.at(i));
    auto low   = static_cast<RooAbsReal*>(_lowSet.at(i));
    auto high  = static_cast<RooAbsReal*>(_highSet.at(i));
    using RooFit::Detail::MathFuncs::flexibleInterpSingle;
    sum += flexibleInterpSingle(_interpCode[i], low->getVal(), high->getVal(), 1.0, nominal, param->getVal(), sum);
  }

  if(_positiveDefinite && (sum<0)){
    sum = 0;
    //     std::cout <<"sum < 0 forcing  positive definite"<< std::endl;
    //     int code = 1;
    //     RooArgSet* myset = new RooArgSet();
    //     std::cout << "integral = " << analyticalIntegralWN(code, myset) << std::endl;
  } else if(sum<0){
    cxcoutD(Tracing) <<"PiecewiseInterpolation::evaluate -  sum < 0, not forcing positive definite"<< std::endl;
  }
  return sum;

}

namespace {

inline double broadcast(std::span<const double> const &s, std::size_t i)
{
   return s.size() > 1 ? s[i] : s[0];
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
/// Interpolate between input distributions for all values of the observable in `evalData`.
/// \param[in,out] ctx Struct holding spans pointing to input data. The results of this function will be stored here.
void PiecewiseInterpolation::doEval(RooFit::EvalContext &ctx) const
{
   std::span<double> sum = ctx.output();

   auto nominal = ctx.at(_nominal);

   for (std::size_t j = 0; j < sum.size(); ++j) {
      sum[j] = broadcast(nominal, j);
   }

   for (unsigned int i = 0; i < _paramSet.size(); ++i) {
      auto param = ctx.at(_paramSet.at(i));
      auto low = ctx.at(_lowSet.at(i));
      auto high = ctx.at(_highSet.at(i));

      for (std::size_t j = 0; j < sum.size(); ++j) {
         using RooFit::Detail::MathFuncs::flexibleInterpSingle;
         sum[j] += flexibleInterpSingle(_interpCode[i], broadcast(low, j), broadcast(high, j), 1.0, broadcast(nominal, j),
                                        broadcast(param, j), sum[j]);
      }
   }

   if (_positiveDefinite) {
      for (std::size_t j = 0; j < sum.size(); ++j) {
         if (sum[j] < 0.)
            sum[j] = 0.;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////

bool PiecewiseInterpolation::setBinIntegrator(RooArgSet& allVars)
{
  if(allVars.size()==1){
    RooAbsReal* temp = const_cast<PiecewiseInterpolation*>(this);
    temp->specialIntegratorConfig(true)->method1D().setLabel("RooBinIntegrator")  ;
    int nbins = (static_cast<RooRealVar*>(allVars.first()))->numBins();
    temp->specialIntegratorConfig(true)->getConfigSection("RooBinIntegrator").setRealValue("numBins",nbins);
    return true;
  }else{
    std::cout << "Currently BinIntegrator only knows how to deal with 1-d "<< std::endl;
    return false;
  }
  return false;
}

////////////////////////////////////////////////////////////////////////////////
/// Advertise that all integrals can be handled internally.

Int_t PiecewiseInterpolation::getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars,
                        const RooArgSet* normSet, const char* /*rangeName*/) const
{
  /*
  std::cout << "---------------------------\nin PiecewiseInterpolation get analytic integral " << std::endl;
  std::cout << "all vars = "<< std::endl;
  allVars.Print("v");
  std::cout << "anal vars = "<< std::endl;
  analVars.Print("v");
  std::cout << "normset vars = "<< std::endl;
  if(normSet2)
    normSet2->Print("v");
  */


  // Handle trivial no-integration scenario
  if (allVars.empty()) return 0 ;
  if (_forceNumInt) return 0 ;


  // Force using numeric integration
  // use special numeric integrator
  return 0;


  // KC: check if interCode=0 for all
  for (auto it = _paramSet.begin(); it != _paramSet.end(); ++it) {
    if (!_interpCode.empty() && _interpCode[it - _paramSet.begin()] != 0) {
        // can't factorize integral
        std::cout << "can't factorize integral" << std::endl;
        return 0;
     }
  }

  // Select subset of allVars that are actual dependents
  analVars.add(allVars) ;

  // Check if this configuration was created before
  Int_t sterileIdx(-1) ;
  CacheElem* cache = static_cast<CacheElem*>(_normIntMgr.getObj(normSet,&analVars,&sterileIdx)) ;
  if (cache) {
    return _normIntMgr.lastIndex()+1 ;
  }

  // Create new cache element
  cache = new CacheElem ;

  // Make list of function projection and normalization integrals
  RooAbsReal *func ;

  // do variations
  for (auto it = _paramSet.begin(); it != _paramSet.end(); ++it)
  {
    auto i = it - _paramSet.begin();
    func = static_cast<RooAbsReal *>(_lowSet.at(i));
    cache->_lowIntList.addOwned(std::unique_ptr<RooAbsReal>{func->createIntegral(analVars)});

    func = static_cast<RooAbsReal *>(_highSet.at(i));
    cache->_highIntList.addOwned(std::unique_ptr<RooAbsReal>{func->createIntegral(analVars)});
  }

  // Store cache element
  Int_t code = _normIntMgr.setObj(normSet,&analVars,(RooAbsCacheElement*)cache,nullptr) ;

  return code+1 ;
}




////////////////////////////////////////////////////////////////////////////////
/// Implement analytical integrations by doing appropriate weighting from  component integrals
/// functions to integrators of components

double PiecewiseInterpolation::analyticalIntegralWN(Int_t code, const RooArgSet* /*normSet2*/,const char* /*rangeName*/) const
{
  /*
  std::cout <<"Enter analytic Integral"<< std::endl;
  printDirty(true);
  //  _nominal.arg().setDirtyInhibit(true) ;
  _nominal.arg().setShapeDirty() ;
  RooAbsReal* temp ;
  RooFIter lowIter(_lowSet.fwdIterator()) ;
  while((temp=(RooAbsReal*)lowIter.next())) {
    //    temp->setDirtyInhibit(true) ;
    temp->setShapeDirty() ;
  }
  RooFIter highIter(_highSet.fwdIterator()) ;
  while((temp=(RooAbsReal*)highIter.next())) {
    //    temp->setDirtyInhibit(true) ;
    temp->setShapeDirty() ;
  }
  */

  /*
  RooAbsArg::setDirtyInhibit(true);
  printDirty(true);
  std::cout <<"done setting dirty inhibit = true"<< std::endl;

  // old integral, only works for linear and not positive definite
  CacheElem* cache = (CacheElem*) _normIntMgr.getObjByIndex(code-1) ;


 std::unique_ptr<RooArgSet> vars2( getParameters(RooArgSet()) );
 std::unique_ptr<RooArgSet> iset(  _normIntMgr.nameSet2ByIndex(code-1)->select(*vars2) );
 std::cout <<"iset = "<< std::endl;
 iset->Print("v");

  double sum = 0;
  RooArgSet* vars = getVariables();
  vars->remove(_paramSet);
  _paramSet.Print("v");
  vars->Print("v");
  if(vars->size()==1){
    RooRealVar* obs = (RooRealVar*) vars->first();
    for(int i=0; i<obs->numBins(); ++i){
      obs->setVal( obs->getMin() + (.5+i)*(obs->getMax()-obs->getMin())/obs->numBins());
      sum+=evaluate()*(obs->getMax()-obs->getMin())/obs->numBins();
      std::cout << "obs = " << obs->getVal() << " sum = " << sum << std::endl;
    }
  } else{
    std::cout <<"only know how to deal with 1 observable right now"<< std::endl;
  }
  */

  /*
  _nominal.arg().setDirtyInhibit(false) ;
  RooFIter lowIter2(_lowSet.fwdIterator()) ;
  while((temp=(RooAbsReal*)lowIter2.next())) {
    temp->setDirtyInhibit(false) ;
  }
  RooFIter highIter2(_highSet.fwdIterator()) ;
  while((temp=(RooAbsReal*)highIter2.next())) {
    temp->setDirtyInhibit(false) ;
  }
  */

  /*
  RooAbsArg::setDirtyInhibit(false);
  printDirty(true);
  std::cout <<"done"<< std::endl;
  std::cout << "sum = " <<sum<< std::endl;
  //return sum;
  */

  // old integral, only works for linear and not positive definite
  CacheElem* cache = static_cast<CacheElem*>(_normIntMgr.getObjByIndex(code-1)) ;
  if( cache==nullptr ) {
    std::cout << "Error: Cache Element is nullptr" << std::endl;
    throw std::exception();
  }

  // old integral, only works for linear and not positive definite

  RooAbsReal *low;
  RooAbsReal *high;
  double value(0);
  double nominal(0);

  // get nominal
  int i=0;
  for (auto funcInt : static_range_cast<RooAbsReal*>(cache->_funcIntList)) {
    value += funcInt->getVal() ;
    nominal = value;
    i++;
  }
  if(i==0 || i>1) { std::cout << "problem, wrong number of nominal functions"<< std::endl; }

  // now get low/high variations
  // KC: old interp code with new iterator

  i = 0;
  for (auto const *param : static_range_cast<RooAbsReal *>(_paramSet)) {
    low = static_cast<RooAbsReal *>(cache->_lowIntList.at(i));
    high = static_cast<RooAbsReal *>(cache->_highIntList.at(i));

    if(param->getVal() > 0) {
      value += param->getVal()*(high->getVal() - nominal);
    } else {
      value += param->getVal()*(nominal - low->getVal());
    }
    ++i;
  }

  /* // MB : old bit of interpolation code
  while( (param=(RooAbsReal*)_paramIter->Next()) ) {
    low = (RooAbsReal*)lowIntIter->Next() ;
    high = (RooAbsReal*)highIntIter->Next() ;

    if(param->getVal()>0) {
      value += param->getVal()*(high->getVal() - nominal );
    } else {
      value += param->getVal()*(nominal - low->getVal());
    }
    ++i;
  }
  */

  /* KC: the code below is wrong.  Can't pull out a constant change to a non-linear shape deformation.
  while( (param=(RooAbsReal*)paramIter.next()) ) {
    low = (RooAbsReal*)lowIntIter.next() ;
    high = (RooAbsReal*)highIntIter.next() ;

    if(_interpCode.empty() || _interpCode.at(i)==0){
      // piece-wise linear
      if(param->getVal()>0)
   value +=  param->getVal()*(high->getVal() - nominal );
      else
   value += param->getVal()*(nominal - low->getVal());
    } else if(_interpCode.at(i)==1){
      // piece-wise log
      if(param->getVal()>=0)
   value *= pow(high->getVal()/nominal, +param->getVal());
      else
   value *= pow(low->getVal()/nominal,  -param->getVal());
    } else if(_interpCode.at(i)==2){
      // parabolic with linear
      double a = 0.5*(high->getVal()+low->getVal())-nominal;
      double b = 0.5*(high->getVal()-low->getVal());
      double c = 0;
      if(param->getVal()>1 ){
   value += (2*a+b)*(param->getVal()-1)+high->getVal()-nominal;
      } else if(param->getVal()<-1 ) {
   value += -1*(2*a-b)*(param->getVal()+1)+low->getVal()-nominal;
      } else {
   value +=  a*pow(param->getVal(),2) + b*param->getVal()+c;
      }
    } else if(_interpCode.at(i)==3){
      //parabolic version of log-normal
      double a = 0.5*(high->getVal()+low->getVal())-nominal;
      double b = 0.5*(high->getVal()-low->getVal());
      double c = 0;
      if(param->getVal()>1 ){
   value += (2*a+b)*(param->getVal()-1)+high->getVal()-nominal;
      } else if(param->getVal()<-1 ) {
   value += -1*(2*a-b)*(param->getVal()+1)+low->getVal()-nominal;
      } else {
   value +=  a*pow(param->getVal(),2) + b*param->getVal()+c;
      }

    } else {
      coutE(InputArguments) << "PiecewiseInterpolation::analyticalIntegralWN ERROR:  " << param->GetName()
             << " with unknown interpolation code" << std::endl ;
    }
    ++i;
  }
  */

  //  std::cout << "value = " << value << std::endl;
  return value;
}

void PiecewiseInterpolation::setInterpCode(RooAbsReal &param, int code, bool /*silent*/)
{
   int index = _paramSet.index(&param);
   if (index < 0) {
      coutE(InputArguments) << "PiecewiseInterpolation::setInterpCode ERROR:  " << param.GetName() << " is not in list"
                            << std::endl;
      return;
   }
   setInterpCodeForParam(index, code);
}

void PiecewiseInterpolation::setAllInterpCodes(int code)
{
   for (std::size_t i = 0; i < _interpCode.size(); ++i) {
      setInterpCodeForParam(i, code);
   }
}

void PiecewiseInterpolation::setInterpCodeForParam(int iParam, int code)
{
   RooAbsArg const &param = _paramSet[iParam];
   if (code < 0 || code > 6) {
      coutE(InputArguments) << "PiecewiseInterpolation::setInterpCode ERROR: " << param.GetName()
                            << " with unknown interpolation code " << code << ", keeping current code "
                            << _interpCode[iParam] << std::endl;
      return;
   }
   if (code == 3) {
      // In the past, code 3 was equivalent to code 2, which confused users.
      // Now, we just say that code 3 doesn't exist and default to code 2 in
      // that case for backwards compatible behavior.
      coutE(InputArguments) << "PiecewiseInterpolation::setInterpCode ERROR: " << param.GetName()
                            << " with unknown interpolation code " << code << ", defaulting to code 2" << std::endl;
      code = 2;
   }
   _interpCode.at(iParam) = code;
   setValueDirty();
}

////////////////////////////////////////////////////////////////////////////////

void PiecewiseInterpolation::printAllInterpCodes(){
  for(unsigned int i=0; i<_interpCode.size(); ++i){
    coutI(InputArguments) <<"interp code for " << _paramSet.at(i)->GetName() << " = " << _interpCode.at(i) << std::endl;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// WVE note: assumes nominal and alternates have identical structure, must add explicit check

std::list<double>* PiecewiseInterpolation::binBoundaries(RooAbsRealLValue& obs, double xlo, double xhi) const
{
  return _nominal.arg().binBoundaries(obs,xlo,xhi) ;
}


////////////////////////////////////////////////////////////////////////////////
/// WVE note: assumes nominal and alternates have identical structure, must add explicit check

bool PiecewiseInterpolation::isBinnedDistribution(const RooArgSet& obs) const
{
  return _nominal.arg().isBinnedDistribution(obs) ;
}



////////////////////////////////////////////////////////////////////////////////

std::list<double>* PiecewiseInterpolation::plotSamplingHint(RooAbsRealLValue& obs, double xlo, double xhi) const
{
  return _nominal.arg().plotSamplingHint(obs,xlo,xhi) ;
}

////////////////////////////////////////////////////////////////////////////////
/// Stream an object of class PiecewiseInterpolation.

void PiecewiseInterpolation::Streamer(TBuffer &R__b)
{
   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(PiecewiseInterpolation::Class(),this);
      specialIntegratorConfig(true)->method1D().setLabel("RooBinIntegrator") ;
      if (_interpCode.empty()) _interpCode.resize(_paramSet.size());
   } else {
      R__b.WriteClassBuffer(PiecewiseInterpolation::Class(),this);
   }
}


/*
////////////////////////////////////////////////////////////////////////////////
/// Customized printing of arguments of a PiecewiseInterpolation to more intuitively reflect the contents of the
/// product operator construction

void PiecewiseInterpolation::printMetaArgs(ostream& os) const
{
  _lowIter->Reset() ;
  if (_highIter) {
    _highIter->Reset() ;
  }

  bool first(true) ;

  RooAbsArg* arg1, *arg2 ;
  if (_highSet.size()!=0) {

    while((arg1=(RooAbsArg*)_lowIter->Next())) {
      if (!first) {
   os << " + " ;
      } else {
   first = false ;
      }
      arg2=(RooAbsArg*)_highIter->Next() ;
      os << arg1->GetName() << " * " << arg2->GetName() ;
    }

  } else {

    while((arg1=(RooAbsArg*)_lowIter->Next())) {
      if (!first) {
   os << " + " ;
      } else {
   first = false ;
      }
      os << arg1->GetName() ;
    }

  }

  os << " " ;
}

*/
