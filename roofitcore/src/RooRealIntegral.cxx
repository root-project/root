/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include <iostream.h>
#include "TObjString.h"
#include "TH1.h"
#include "RooFitCore/RooRealIntegral.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooCategory.hh"

ClassImp(RooRealIntegral) 
;


RooRealIntegral::RooRealIntegral(const char *name, const char *title, 
				 RooDerivedReal& function, RooArgSet& depList,
				 Int_t maxSteps, Double_t eps) : 
  RooDerivedReal(name,title), _function(&function), _mode(0),
  _maxSteps(maxSteps), _eps(eps), _intList("intList"), _sumList("sumList")
{
  // Make sublist of dependents that need to be integrated and summed
  TIterator* depIter=depList.MakeIterator() ;
  RooAbsArg *arg ;
  while (arg=(RooAbsArg*)depIter->Next()) {

    if (!function.dependsOn(*arg)) {
      cout << "RooRealIntegral::RooIntegral(" << name << "): integrand " << arg->GetName()
	   << " doesn't depend on function " << function.GetName() << ", ignored" << endl ;
      continue ;
   }

    // Make list of arguments integrated or summed over
    // Register each of them as shape server
    if (arg->IsA()->InheritsFrom(RooRealVar::Class())) {
      _intList.add(*arg) ;
      addServer(*arg,kFALSE,kTRUE) ;
    } else if (arg->IsA()->InheritsFrom(RooCategory::Class())) {
      _sumList.add(*arg) ;
      addServer(*arg,kFALSE,kTRUE) ;
    } else {
      cout << "RooRealIntegral::RooIntegral(" << name << "): integrand " << arg->GetName()
	   << " is neither a RooCategory nor a RooRealVar, ignored" << endl ;
    }
  }
  
  // Register all non-integrands of functions as value servers
  TIterator* sIter = function.serverIterator() ;
  while (arg=(RooAbsArg*)sIter->Next()) {
    if (!_intList.FindObject(arg) && !_sumList.FindObject(arg))
      addServer(*arg,kTRUE,kFALSE) ;
  }

  // Determine if function has analytical integral 
  _mode = _function->getAnalyticalIntegral(_intList) ;

  // Initialize numerical integrator if needed
  if (_mode==0) 
    if (!engineInit()) assert(0) ;
  
}


RooRealIntegral::RooRealIntegral(const char* name, const RooRealIntegral& other) : 
  RooDerivedReal(name,other), _function(other._function), 
  _maxSteps(other._maxSteps), _eps(other._eps), _mode(other._mode),
  _intList("intList"), _sumList("sumList")
{
  copyList(_intList,other._intList) ;
  copyList(_sumList,other._sumList) ;
  if (_mode==0) engineInit() ;
}


RooRealIntegral::RooRealIntegral(const RooRealIntegral& other) :
  RooDerivedReal(other), _function(other._function),
  _maxSteps(other._maxSteps), _eps(other._eps), _mode(other._mode),
  _intList("intList"), _sumList("sumList")
{
  copyList(_intList,other._intList) ;
  copyList(_sumList,other._sumList) ;
  if (_mode==0) engineInit() ;
}



RooRealIntegral::~RooRealIntegral()
{
  if (_mode==0) engineCleanup() ;
}


RooRealIntegral& RooRealIntegral::operator=(const RooRealIntegral& other)
{
  RooDerivedReal::operator=(other) ;
  copyList(_intList,other._intList) ;
  copyList(_sumList,other._sumList) ;
  _function = other._function ;
  setValueDirty(kTRUE) ;
  return *this ;
}


RooAbsArg& RooRealIntegral::operator=(const RooAbsArg& aother)
{
  return operator=((const RooRealIntegral&)aother) ;
}



Double_t RooRealIntegral::evaluate() const 
{
  // Save current integrand values 
  RooArgSet saveInt("saveInt",_intList), saveSum("saveSum",_sumList) ;

  // Evaluate integral
  Double_t retVal = sum(_sumList,_intList) ;

  // Restore integrand values
  _intList=saveInt ;
  _sumList=saveSum ;

  return retVal ;
}



Bool_t RooRealIntegral::engineInit() 
{
  // Allocate workspace for numerical integration engine
  _h= new Double_t[_maxSteps + 1];
  _s= new Double_t[_maxSteps + 2];
  _c= new Double_t[_nPoints + 1];
  _d= new Double_t[_nPoints + 1];
  return kTRUE ;
}



Bool_t RooRealIntegral::engineCleanup() 
{
  // Release integrator workspace
  if(_h) delete[] _h;
  if(_s) delete[] _s;
  if(_c) delete[] _c;
  if(_d) delete[] _d;
  return kFALSE ;
}




Double_t RooRealIntegral::sum(const RooArgSet& sumList, const RooArgSet& intList) const
{
  // Summing should be implemented here using a RooSuperCategory iterator
  return integrate(intList) ;
}



// Default implementation does numerical integration
Double_t RooRealIntegral::integrate(const RooArgSet& intList) const
{
  // Trivial case, no integration required
  if (intList.GetSize()==0) return _function->getVal() ;

  // Call analytical integral function if available
  if (_mode!=0) return _function->analyticalIntegral(_mode) ;  

  // Numerical algorithm can handle only 1 integrand
  if (intList.GetSize()>1) {
    cout << "RooRealIntegral::integrate(" << GetName() 
	 << "): cannot do multi-dimensional integrals" << endl ;
    return 0 ;
  }

  _var = (RooRealVar*) intList.First() ;
  _xmin= _var->getFitMin() ;
  _xmax= _var->getFitMax() ;
  _range= _var->getFitMax() - _var->getFitMin() ;

  Int_t j;
  _h[1]=1.0;
  for(j= 1; j<=_maxSteps; j++) {
    _s[j]= addTrapezoids(j);
    if(j >= _nPoints) {
      extrapolate(j);
      if(fabs(_extrapError) <= _eps*fabs(_extrapValue)) return _extrapValue;
    }
    _h[j+1]=0.25*_h[j];
  }

  cout << "RooIntegrator::Integrate: did not converge after " 
       << _maxSteps << " steps" << endl;
  for(j= 1; j <= _maxSteps; j++) {
    cout << "   [" << j << "] h = " << _h[j] << " , s = " << _s[j] << endl;
  }
  return 0;
}




Double_t RooRealIntegral::evalAt(Double_t x) const
{
  _var->setVal(x) ;
  return _function->getVal();
}



Double_t RooRealIntegral::addTrapezoids(Int_t n) const
{
  Double_t x,tnm,sum,del;
  Int_t it,j;

  if (n == 1) {
    // use a single trapezoid to cover the full range
    return (_savedResult= 0.5*_range*(evalAt(_xmin) + evalAt(_xmax)));
  }
  else {
    // break the range down into several trapezoids using 2**(n-2)
    // equally-spaced interior points
    for(it=1, j=1; j < n-1; j++) it <<= 1;
    tnm= it;
    del= _range/tnm;
    x= _xmin + 0.5*del;
    for(sum=0.0, j=1; j<=it; j++, x+=del) sum += evalAt(x);
    return (_savedResult= 0.5*(_savedResult + _range*sum/tnm));
  }
}


void RooRealIntegral::extrapolate(Int_t n) const
{
  Double_t *xa= &_h[n-_nPoints];
  Double_t *ya= &_s[n-_nPoints];
  Int_t i,m,ns=1;
  Double_t den,dif,dift,ho,hp,w;

  dif=fabs(xa[1]);
  for (i= 1; i <= _nPoints; i++) {
    if ((dift=fabs(xa[i])) < dif) {
      ns=i;
      dif=dift;
    }
    _c[i]=ya[i];
    _d[i]=ya[i];
  }
  _extrapValue= ya[ns--];
  for(m= 1; m < _nPoints; m++) {
    for(i= 1; i <= _nPoints-m; i++) {
      ho=xa[i];
      hp=xa[i+m];
      w=_c[i+1]-_d[i];
      if((den=ho-hp) == 0.0) {
	cout << "RooRealIntegral::extrapolate(" << GetName() << "): internal error" << endl;
      }
      den=w/den;
      _d[i]=hp*den;
      _c[i]=ho*den;
    }
    _extrapValue += (_extrapError=(2*ns < (_nPoints-m) ? _c[ns+1] : _d[ns--]));
  }
}




Bool_t RooRealIntegral::isValid(Double_t value) const 
{
  return kTRUE ;
}



Bool_t RooRealIntegral::redirectServersHook(RooArgSet& newServerList, Bool_t mustReplaceAll)
{
  return kFALSE ;
}


void RooRealIntegral::printToStream(ostream& os, PrintOption opt=Standard) const
{

  if (opt==Verbose) {
    RooAbsArg::printToStream(os,Verbose) ;
    return ;
  }

  //Print object contents
  os << "RooRealIntegral: " << GetName() << " =" ;

  RooAbsArg* arg ;
  Bool_t first(kTRUE) ;

  if (_sumList.First()) {
    TIterator* sIter = _sumList.MakeIterator() ;
    os << " Sum(" ;
    while (arg=(RooAbsArg*)sIter->Next()) {
      os << (first?"":",") << arg->GetName() ;
      first=kFALSE ;
    }
    delete sIter ;
    os << ")" ;
  }

  first=kTRUE ;
  if (_intList.First()) {
    TIterator* iIter = _intList.MakeIterator() ;
    os << " Int(" ;
    while (arg=(RooAbsArg*)iIter->Next()) {
      os << (first?"":",") << arg->GetName() ;
      first=kFALSE ;
    }
    delete iIter ;
    os << ")" ;
  }


  os << " " << _function->GetName() << " = " << getVal();
  if(!_unit.IsNull()) os << ' ' << _unit;
  os << " : \"" << fTitle << "\"" ;

  printAttribList(os) ;
  os << endl ;
} 


