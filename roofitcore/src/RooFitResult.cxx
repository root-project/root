/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   17-Aug-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include "RooFitCore/RooFitResult.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooRealVar.hh"
#include "TMinuit.h"

ClassImp(RooFitResult) 
;

RooFitResult::RooFitResult() : _constPars(0), _initPars(0), _finalPars(0), _globalCorr(0), _corrMatrix(0) 
{
}

RooFitResult::~RooFitResult() 
{
  if (_constPars) delete _constPars ;
  if (_initPars)  delete _initPars ;
  if (_finalPars) delete _finalPars ;
  if (_globalCorr) {
    _globalCorr->Delete() ;
    delete _globalCorr ;
  }
  if (_corrMatrix) {
    Int_t i ;
    for (i=0 ; i<_initPars->GetSize() ; i++) {
      delete _corrMatrix[i] ;      
    }
    delete[] _corrMatrix ;
  }
}

void RooFitResult::setConstParList(const RooArgSet& list) 
{
  if (_constPars) delete _constPars ;
  _constPars = list.snapshot() ;
}


void RooFitResult::setInitParList(const RooArgSet& list)
{
  if (_initPars) delete _initPars ;
  _initPars = list.snapshot() ;
}


void RooFitResult::setFinalParList(const RooArgSet& list)
{
  if (_finalPars) delete _finalPars ;
  _finalPars = list.snapshot() ;
}


Double_t RooFitResult::correlation(const RooAbsArg& par1, const RooAbsArg& par2) const 
{
  const RooArgSet* row = correlation(par1) ;
  if (!row) return 0. ;
  RooAbsArg* arg = _initPars->find(par2.GetName()) ;
  if (!arg) {
    cout << "RooFitResult::correlation: variable " << par2.GetName() << " not a floating parameter in fit" << endl ;
    return 0. ;
  }
  return ((RooRealVar*)row->At(_initPars->IndexOf(arg)))->getVal() ;
}


const RooArgSet* RooFitResult::correlation(const RooAbsArg& par) const 
{
  RooAbsArg* arg = _initPars->find(par.GetName()) ;
  if (!arg) {
    cout << "RooFitResult::correlation: variable " << par.GetName() << " not a floating parameter in fit" << endl ;
    return 0 ;
  }    
  return _corrMatrix[_initPars->IndexOf(arg)] ;
}


void RooFitResult::printToStream(ostream& os, PrintOption opt, TString indent) const
{
  os << "--- RooFitResult --- " << endl ;
  os << " minNLL = " << _minNLL << endl ;
  os << "    EDM = " << _edm << endl ;

  os << " Constant parameters: " << endl ;
  _constPars->printToStream(os,opt,indent) ;

  os << " Initial value of floating parameters: " << endl ;
  _initPars->printToStream(os,opt,indent) ;

  os << " Final value of floating parameters: " << endl ;
  _finalPars->printToStream(os,opt,indent) ;

  os << " Global correlation coefficients: " << endl ;
  _globalCorr->printToStream(os,opt,indent) ;

}


void RooFitResult::fillCorrMatrix()
{
  // Sanity check
  if (gMinuit->fNpar <= 1) {
    cout << "RooFitResult::fillCorrMatrix: number of floating parameters <=1, correlation matrix not filled" << endl ;
    return ;
  }

  if (!_initPars) {
    cout << "RooFitResult::fillCorrMatrix: ERROR: list of initial parameters must be filled first" << endl ;
    return ;
  }

  // Delete eventual prevous correlation data holders
  if (_globalCorr) {
    _globalCorr->Delete() ;
    delete _globalCorr ;
  }

  if (_corrMatrix) {
    Int_t i ;
    for (i=0 ; i<_initPars->GetSize() ; i++) {
      delete _corrMatrix[i] ;      
    }
    delete[] _corrMatrix ;
  }

  // Build holding arrays for correlation coefficients
  _globalCorr = new RooArgSet("globalCorrelations") ;
  _corrMatrix = new pRooArgSet[_initPars->GetSize()] ;
  TIterator* vIter = _initPars->MakeIterator() ;
  RooAbsArg* arg ;
  Int_t idx(0) ;
  while(arg=(RooAbsArg*)vIter->Next()) {
    // Create global correlation value holder
    TString gcName("GC[") ;
    gcName.Append(arg->GetName()) ;
    gcName.Append("]") ;
    TString gcTitle(arg->GetTitle()) ;
    gcTitle.Append(" Global Correlation") ;
    _globalCorr->add(*(new RooRealVar(gcName.Data(),gcTitle.Data(),0.))) ;

    // Create array with correlation holders for this parameter
    TString name("C[") ;
    name.Append(arg->GetName()) ;
    name.Append(",*]") ;
    _corrMatrix[idx] = new RooArgSet(name.Data()) ;
    TIterator* vIter2 = _initPars->MakeIterator() ;
    RooAbsArg* arg2 ;
    while(arg2=(RooAbsArg*)vIter2->Next()) {

      TString cName("C[") ;
      cName.Append(arg->GetName()) ;
      cName.Append(",") ;
      cName.Append(arg2->GetName()) ;
      cName.Append("]") ;
      TString cTitle("Correlation between ") ;
      cTitle.Append(arg->GetName()) ;
      cTitle.Append(" and ") ;
      cTitle.Append(arg2->GetName()) ;
      _corrMatrix[idx]->add(*(new RooRealVar(cName.Data(),cTitle.Data(),0.))) ;      
    }
    delete vIter2 ;
    idx++ ;
  }
  delete vIter ;

  TIterator *gcIter = _globalCorr->MakeIterator() ;

  // Extract correlation information for MINUIT (code taken from TMinuit::mnmatu() )

  // WVE: This code directly manipulates minuit internal workspace, 
  //      if TMinuit code changes this may need updating
  Int_t ndex, i, j, m, n, ncoef, nparm, id, it, ix;
  Int_t ndi, ndj, iso, isw2, isw5;
  ncoef = (gMinuit->fNpagwd - 19) / 6;
  nparm = TMath::Min(gMinuit->fNpar,ncoef);
  RooRealVar* gcVal(0) ;
  for (i = 1; i <= gMinuit->fNpar; ++i) {
    ix  = gMinuit->fNexofi[i-1];
    ndi = i*(i + 1) / 2;
    for (j = 1; j <= gMinuit->fNpar; ++j) {
      m    = TMath::Max(i,j);
      n    = TMath::Min(i,j);
      ndex = m*(m-1) / 2 + n;
      ndj  = j*(j + 1) / 2;
      gMinuit->fMATUvline[j-1] = gMinuit->fVhmat[ndex-1] / TMath::Sqrt(TMath::Abs(gMinuit->fVhmat[ndi-1]*gMinuit->fVhmat[ndj-1]));
    }
    nparm = TMath::Min(gMinuit->fNpar,ncoef);

    gcVal = (RooRealVar*) gcIter->Next() ;
    gcVal->setVal(gMinuit->fGlobcc[i-1]) ;
    TIterator* cIter = _corrMatrix[i-1]->MakeIterator() ;
    for (it = 1; it <= nparm; ++it) {
      RooRealVar* cVal = (RooRealVar*) cIter->Next() ;
      cVal->setVal(gMinuit->fMATUvline[it-1]) ;
    }
    delete cIter ;
  }


  delete gcIter ;
} 

