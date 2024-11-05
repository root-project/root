/** \class PiecewiseInterpolation
 * \ingroup HistFactory
 * The PiecewiseInterpolation is a class that can morph distributions into each other, which
 * is useful to estimate systematic uncertainties. Given a nominal distribution and one or
 * more altered or distorted ones, it computes a new shape depending on the value of the nuisance
 * parameters \f$ \alpha_i \f$:
 * \f[
 *   A = \sum_i \mathrm{Interpolate}(\mathrm{low}_i, \mathrm{nominal}, \mathrm{high}_i, \alpha_i).
 * \f]
 * If an \f$ \alpha_i \f$ is zero, the distribution is identical to the nominal distribution, at
 * \f$ \pm 1 \f$ it is identical to the up/down distribution for that specific \f$ i \f$.
 *
 * The class supports several interpolation methods, which can be selected for each parameter separately
 * using setInterpCode(). The default interpolation code is 4. This performs
 * - \f$ |\alpha | > 1 \f$: Linear extrapolation.
 * - \f$ |\alpha | < 1 \f$: Polynomial interpolation. A sixth-order polynomial is used. Its coefficients
 * are chosen such that function, first, and second derivative at \f$ \alpha \pm 1 \f$ match the values
 * that the extrapolation procedure uses.
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

using std::endl, std::cout;

ClassImp(PiecewiseInterpolation);

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
    coutE(InputArguments) << "PiecewiseInterpolation::ctor(" << GetName() << ") ERROR: input lists should be of equal length" << endl ;
    RooErrorHandler::softAbort() ;
  }

  for (auto *comp : lowSet) {
    if (!dynamic_cast<RooAbsReal*>(comp)) {
      coutE(InputArguments) << "PiecewiseInterpolation::ctor(" << GetName() << ") ERROR: component " << comp->GetName()
             << " in first list is not of type RooAbsReal" << endl ;
      RooErrorHandler::softAbort() ;
    }
    _lowSet.add(*comp) ;
  }


  for (auto *comp : highSet) {
    if (!dynamic_cast<RooAbsReal*>(comp)) {
      coutE(InputArguments) << "PiecewiseInterpolation::ctor(" << GetName() << ") ERROR: component " << comp->GetName()
             << " in first list is not of type RooAbsReal" << endl ;
      RooErrorHandler::softAbort() ;
    }
    _highSet.add(*comp) ;
  }


  for (auto *comp : paramSet) {
    if (!dynamic_cast<RooAbsReal*>(comp)) {
      coutE(InputArguments) << "PiecewiseInterpolation::ctor(" << GetName() << ") ERROR: component " << comp->GetName()
             << " in first list is not of type RooAbsReal" << endl ;
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
    Int_t icode = _interpCode[i] ;

    if(icode < 0 || icode > 5) {
      coutE(InputArguments) << "PiecewiseInterpolation::evaluate ERROR:  " << param->GetName()
                 << " with unknown interpolation code" << icode << endl ;
    }
    using RooFit::Detail::MathFuncs::flexibleInterpSingle;
    sum += flexibleInterpSingle(icode, low->getVal(), high->getVal(), 1.0, nominal, param->getVal(), sum);
  }

  if(_positiveDefinite && (sum<0)){
    sum = 0;
    //     cout <<"sum < 0 forcing  positive definite"<<endl;
    //     int code = 1;
    //     RooArgSet* myset = new RooArgSet();
    //     cout << "integral = " << analyticalIntegralWN(code, myset) << endl;
  } else if(sum<0){
    cxcoutD(Tracing) <<"PiecewiseInterpolation::evaluate -  sum < 0, not forcing positive definite"<<endl;
  }
  return sum;

}

void PiecewiseInterpolation::translate(RooFit::Detail::CodeSquashContext &ctx) const
{
   std::size_t n = _interpCode.size();

   std::string resName = "total_" + ctx.getTmpVarName();
   for (std::size_t i = 0; i < n; ++i) {
      if (_interpCode[i] < 0 || _interpCode[i] > 5) {
         coutE(InputArguments) << "PiecewiseInterpolation::evaluate ERROR:  " << _paramSet[i].GetName()
                               << " with unknown interpolation code" << _interpCode[i] << endl;
      }
      if (_interpCode[i] != _interpCode[0]) {
         coutE(InputArguments) << "FlexibleInterpVar::evaluate ERROR:  Code Squashing AD does not yet support having "
                                  "different interpolation codes for the same class object "
                               << endl;
      }
   }

   // The PiecewiseInterpolation class is used in the context of HistFactory
   // models, where is is always used the same way: all RooAbsReals in _lowSet,
   // _histSet, and also nominal are 1D RooHistFuncs with with same structure.
   //
   // Therefore, we can make a big optimization: we get the bin index only once
   // here in the generated code for PiecewiseInterpolation. Then, we also
   // rearrange the histogram data in such a way that we can always pass the
   // same arrays to the free function that implements the interpolation, just
   // with a dynamic offset calculated from the bin index.
   RooDataHist const &nomHist = dynamic_cast<RooHistFunc const &>(*_nominal).dataHist();
   int nBins = nomHist.numEntries();
   std::vector<double> valsNominal;
   std::vector<double> valsLow;
   std::vector<double> valsHigh;
   for (int i = 0; i < nBins; ++i) {
      valsNominal.push_back(nomHist.weight(i));
   }
   for (int i = 0; i < nBins; ++i) {
      for (std::size_t iParam = 0; iParam < n; ++iParam) {
         valsLow.push_back(dynamic_cast<RooHistFunc const &>(_lowSet[iParam]).dataHist().weight(i));
         valsHigh.push_back(dynamic_cast<RooHistFunc const &>(_highSet[iParam]).dataHist().weight(i));
      }
   }
   std::string idxName = ctx.getTmpVarName();
   std::string valsNominalStr = ctx.buildArg(valsNominal);
   std::string valsLowStr = ctx.buildArg(valsLow);
   std::string valsHighStr = ctx.buildArg(valsHigh);
   std::string nStr = std::to_string(n);
   std::string code;

   std::string lowName = ctx.getTmpVarName();
   std::string highName = ctx.getTmpVarName();
   std::string nominalName = ctx.getTmpVarName();
   code += "unsigned int " + idxName + " = " + nomHist.calculateTreeIndexForCodeSquash(this, ctx, dynamic_cast<RooHistFunc const &>(*_nominal).variables()) + ";\n";
   code += "double const* " + lowName + " = " + valsLowStr + " + " + nStr + " * " + idxName + ";\n";
   code += "double const* " + highName + " = " + valsHighStr + " + " + nStr + " * " + idxName + ";\n";
   code += "double " + nominalName + " = *(" + valsNominalStr + " + " + idxName + ");\n";

   std::string funcCall = ctx.buildCall("RooFit::Detail::MathFuncs::flexibleInterp", _interpCode[0], _paramSet, n,
                                        lowName, highName, 1.0, nominalName, 0.0);
   code += "double " + resName + " = " + funcCall + ";\n";

   if (_positiveDefinite)
      code += resName + " = " + resName + " < 0 ? 0 : " + resName + ";\n";

   ctx.addToCodeBody(this, code);
   ctx.addResult(this, resName);
}

////////////////////////////////////////////////////////////////////////////////
/// Interpolate between input distributions for all values of the observable in `evalData`.
/// \param[in,out] evalData Struct holding spans pointing to input data. The results of this function will be stored here.
/// \param[in] normSet Arguments to normalise over.
void PiecewiseInterpolation::doEval(RooFit::EvalContext & ctx) const
{
  std::span<double> sum = ctx.output();

  auto nominal = ctx.at(_nominal);
  for(unsigned int j=0; j < nominal.size(); ++j) {
    sum[j] = nominal[j];
  }

  for (unsigned int i=0; i < _paramSet.size(); ++i) {
    const double param = static_cast<RooAbsReal*>(_paramSet.at(i))->getVal();
    auto low   = ctx.at(_lowSet.at(i));
    auto high  = ctx.at(_highSet.at(i));
    const int icode = _interpCode[i];

    if (icode < 0 || icode > 5) {
      coutE(InputArguments) << "PiecewiseInterpolation::doEval(): " << _paramSet[i].GetName()
                       << " with unknown interpolation code" << icode << std::endl;
      throw std::invalid_argument("PiecewiseInterpolation::doEval() got invalid interpolation code " + std::to_string(icode));
    }

    for (unsigned int j=0; j < nominal.size(); ++j) {
       using RooFit::Detail::MathFuncs::flexibleInterpSingle;
       sum[j] += flexibleInterpSingle(icode, low[j], high[j], 1.0, nominal[j], param, sum[j]);
    }
  }

  if (_positiveDefinite) {
    for(unsigned int j=0; j < nominal.size(); ++j) {
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
    cout << "Currently BinIntegrator only knows how to deal with 1-d "<<endl;
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
  cout << "---------------------------\nin PiecewiseInterpolation get analytic integral " <<endl;
  cout << "all vars = "<<endl;
  allVars.Print("v");
  cout << "anal vars = "<<endl;
  analVars.Print("v");
  cout << "normset vars = "<<endl;
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
        cout << "can't factorize integral" << endl;
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
  cout <<"Enter analytic Integral"<<endl;
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
  cout <<"done setting dirty inhibit = true"<<endl;

  // old integral, only works for linear and not positive definite
  CacheElem* cache = (CacheElem*) _normIntMgr.getObjByIndex(code-1) ;


 std::unique_ptr<RooArgSet> vars2( getParameters(RooArgSet()) );
 std::unique_ptr<RooArgSet> iset(  _normIntMgr.nameSet2ByIndex(code-1)->select(*vars2) );
 cout <<"iset = "<<endl;
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
      cout << "obs = " << obs->getVal() << " sum = " << sum << endl;
    }
  } else{
    cout <<"only know how to deal with 1 observable right now"<<endl;
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
  cout <<"done"<<endl;
  cout << "sum = " <<sum<<endl;
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
  if(i==0 || i>1) { cout << "problem, wrong number of nominal functions"<<endl; }

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
             << " with unknown interpolation code" << endl ;
    }
    ++i;
  }
  */

  //  cout << "value = " << value <<endl;
  return value;
}


////////////////////////////////////////////////////////////////////////////////

void PiecewiseInterpolation::setInterpCode(RooAbsReal& param, int code, bool silent){
  int index = _paramSet.index(&param);
  if(index<0){
      coutE(InputArguments) << "PiecewiseInterpolation::setInterpCode ERROR:  " << param.GetName()
             << " is not in list" << endl ;
  } else {
     if(!silent){
       coutW(InputArguments) << "PiecewiseInterpolation::setInterpCode :  " << param.GetName()
                             << " is now " << code << endl ;
     }
    _interpCode.at(index) = code;
  }
}


////////////////////////////////////////////////////////////////////////////////

void PiecewiseInterpolation::setAllInterpCodes(int code){
  for(unsigned int i=0; i<_interpCode.size(); ++i){
    _interpCode.at(i) = code;
  }
}


////////////////////////////////////////////////////////////////////////////////

void PiecewiseInterpolation::printAllInterpCodes(){
  for(unsigned int i=0; i<_interpCode.size(); ++i){
    coutI(InputArguments) <<"interp code for " << _paramSet.at(i)->GetName() << " = " << _interpCode.at(i) <<endl;
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
