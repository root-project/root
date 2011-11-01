/***************************************************************************** 
 * Project: RooFit                                                           * 
 * author: Max Baak (mbaak@cern.ch)                                          *
 *****************************************************************************/ 

// Written by Max Baak (mbaak@cern.ch)
// 1-dimensional morph function between a list of input functions (varlist) as a function of one input parameter (m).
// The vector mrefpoints assigns an m-number to each function in the function list.
// For example: varlist can contain MC histograms (or single numbers) of a reconstructed mass, for certain 
// true Higgs masses indicated in mrefpoints. the input parameter m is the true (continous) Higgs mass.
// Morphing can be set to be linear or non-linear, or mixture of the two.

#include "Riostream.h" 

#include "Roo1DMomentMorphFunction.h" 
#include "RooAbsCategory.h" 
#include "RooRealIntegral.h"
#include "RooRealConstant.h"
#include "RooRealVar.h"
#include "RooFormulaVar.h"
#include "RooCustomizer.h"
#include "RooAddPdf.h"
#include "RooAddition.h"
#include "RooMoment.h"
#include "RooLinearVar.h"
#include "RooChangeTracker.h"

#include "TMath.h"

ClassImp(Roo1DMomentMorphFunction) 


//_____________________________________________________________________________
  Roo1DMomentMorphFunction::Roo1DMomentMorphFunction() : _mref(0), _frac(0), _M(0) 
{
  _varItr    = _varList.createIterator() ;
}



//_____________________________________________________________________________
Roo1DMomentMorphFunction::Roo1DMomentMorphFunction(const char *name, const char *title, 
						   RooAbsReal& _m,
						   const RooArgList& varList,
						   const TVectorD& mrefpoints,
						   const Setting& setting) :
  RooAbsReal(name,title), 
  m("m","m",this,_m),
  _varList("varList","List of variables",this),
  _setting(setting)
{ 
  // CTOR

  // observables
  TIterator* varItr = varList.createIterator() ;
  RooAbsArg* var ;
  for (Int_t i=0; (var = (RooAbsArg*)varItr->Next()); ++i) {
    if (!dynamic_cast<RooAbsReal*>(var)) {
      coutE(InputArguments) << "Roo1DMomentMorphFunction::ctor(" << GetName() << ") ERROR: variable " << var->GetName() << " is not of type RooAbsReal" << endl ;
      throw string("RooPolyMorh::ctor() ERROR variable is not of type RooAbsReal") ;
    }
    _varList.add(*var) ;
  }
  delete varItr ;

  _mref      = new TVectorD(mrefpoints);
  _varItr    = _varList.createIterator() ;

  // initialization
  initialize();
} 



//_____________________________________________________________________________
Roo1DMomentMorphFunction::Roo1DMomentMorphFunction(const Roo1DMomentMorphFunction& other, const char* name) :  
  RooAbsReal(other,name), 
  m("m",this,other.m),
  _varList("varList",this,other._varList),
  _setting(other._setting)
{ 
  _mref = new TVectorD(*other._mref) ;
  _varItr    = _varList.createIterator() ;

  // initialization
  initialize();
} 

//_____________________________________________________________________________
Roo1DMomentMorphFunction::~Roo1DMomentMorphFunction() 
{
  if (_mref)   delete _mref;
  if (_frac)   delete _frac;
  if (_varItr) delete _varItr;
  if (_M)      delete _M;
}



//_____________________________________________________________________________
void Roo1DMomentMorphFunction::initialize() 
{
  Int_t nVar = _varList.getSize();

  // other quantities needed
  if (nVar!=_mref->GetNrows()) {
    coutE(InputArguments) << "Roo1DMomentMorphFunction::initialize(" << GetName() << ") ERROR: nVar != nRefPoints" << endl ;
    assert(0) ;
  }

  _frac = new TVectorD(nVar);

  TVectorD* dm = new TVectorD(nVar);
  _M = new TMatrixD(nVar,nVar);

  // transformation matrix for non-linear extrapolation, needed in evaluate()
  TMatrixD M(nVar,nVar);
  for (Int_t i=0; i<_mref->GetNrows(); ++i) {
    (*dm)[i] = (*_mref)[i]-(*_mref)[0];
    M(i,0) = 1.;
    if (i>0) M(0,i) = 0.;
  }
  for (Int_t i=1; i<_mref->GetNrows(); ++i) {
    for (Int_t j=1; j<_mref->GetNrows(); ++j) {
      M(i,j) = TMath::Power((*dm)[i],(double)j);
    }
  }
  (*_M) = M.Invert();

  delete dm ;
}


//_____________________________________________________________________________
Double_t Roo1DMomentMorphFunction::evaluate() const 
{ 
  calculateFractions(); // this sets _frac vector, based on function of m

  _varItr->Reset() ;

  Double_t ret(0);
  RooAbsReal* var(0) ;
  for (Int_t i=0; (var = (RooAbsReal*)_varItr->Next()); ++i) {
    ret += (*_frac)(i) * var->getVal();
  }

  return ret ;
} 


//_____________________________________________________________________________
void Roo1DMomentMorphFunction::calculateFractions() const
{
  Int_t nVar = _varList.getSize();

  Double_t dm = m - (*_mref)[0];

  // fully non-linear
  double sumposfrac=0.;
  for (Int_t i=0; i<nVar; ++i) {
    double ffrac=0.;
    for (Int_t j=0; j<nVar; ++j) { ffrac += (*_M)(j,i) * (j==0?1.:TMath::Power(dm,(double)j)); }
    if (ffrac>=0) sumposfrac+=ffrac;
    (*_frac)(i) = ffrac;
  }

  // various mode settings
  int imin = idxmin(m);
  int imax = idxmax(m);
  double mfrac = (m-(*_mref)[imin])/((*_mref)[imax]-(*_mref)[imin]);
  switch (_setting) {
    case NonLinear:
      // default already set above
    break;
    case Linear: 
      for (Int_t i=0; i<nVar; ++i)
        (*_frac)(i) = 0.;      
      if (imax>imin) { // m in between mmin and mmax
        (*_frac)(imin) = (1.-mfrac); 
        (*_frac)(imax) = (mfrac);
      } else if (imax==imin) { // m outside mmin and mmax
        (*_frac)(imin) = (1.);
      }
    break;
    case NonLinearLinFractions:
      for (Int_t i=0; i<nVar; ++i)
        (*_frac)(i) = (0.);
      if (imax>imin) { // m in between mmin and mmax
        (*_frac)(imin) = (1.-mfrac);
        (*_frac)(imax) = (mfrac);
      } else if (imax==imin) { // m outside mmin and mmax
        (*_frac)(imin) = (1.);
      }
    break;
    case NonLinearPosFractions:
      for (Int_t i=0; i<nVar; ++i) {
        if ((*_frac)(i)<0) (*_frac)(i)=(0.);
        (*_frac)(i) = (*_frac)(i)/sumposfrac;
      }
    break;
  } 
}

//_____________________________________________________________________________
int Roo1DMomentMorphFunction::idxmin(const double& mval) const
{
  int imin(0);
  Int_t nVar = _varList.getSize();
  double mmin=-DBL_MAX;
  for (Int_t i=0; i<nVar; ++i) 
    if ( (*_mref)[i]>mmin && (*_mref)[i]<=mval ) { mmin=(*_mref)[i]; imin=i; }
  return imin;
}


//_____________________________________________________________________________
int Roo1DMomentMorphFunction::idxmax(const double& mval) const
{
  int imax(0);
  Int_t nVar = _varList.getSize();
  double mmax=DBL_MAX;
  for (Int_t i=0; i<nVar; ++i) 
    if ( (*_mref)[i]<mmax && (*_mref)[i]>=mval ) { mmax=(*_mref)[i]; imax=i; }
  return imax;
}
