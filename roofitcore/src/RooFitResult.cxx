/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooFitResult.cc,v 1.14 2002/05/16 19:26:58 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   17-Aug-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [MISC] --
// RooFitResult is a container class to hold the input and output
// of a PDF fit to a dataset. It contains:
//
//   - Values of all constant parameters
//   - Initial and final values of floating parameters with error
//   - Correlation matrix and global correlation coefficients
//   - NLL and EDM at mininum
//
// No references to the fitted PDF and dataset are stored

#include <iomanip.h>
#include "TMinuit.h"
#include "TMarker.h"
#include "TLine.h"
#include "TBox.h"
#include "TGaxis.h"
#include "TMatrix.h"
#include "RooFitCore/RooFitResult.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooArgList.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooPlot.hh"
#include "RooFitCore/RooEllipse.hh"
#include "RooFitCore/RooRandom.hh"

ClassImp(RooFitResult) 
;

RooFitResult::RooFitResult(const char* name, const char* title) : 
  TNamed(name,title), _constPars(0), _initPars(0), _finalPars(0), _globalCorr(0), _randomPars(0), _Lt(0)
{  
  // Constructor
  appendToDir(this) ;
}

RooFitResult::~RooFitResult() 
{
  // Destructor
  if (_constPars) delete _constPars ;
  if (_initPars)  delete _initPars ;
  if (_finalPars) delete _finalPars ;
  if (_globalCorr) delete _globalCorr;
  if (_randomPars) delete _randomPars;
  if (_Lt) delete _Lt;

  _corrMatrix.Delete();

  removeFromDir(this) ;
}

void RooFitResult::setConstParList(const RooArgList& list) 
{
  // Fill the list of constant parameters
  if (_constPars) delete _constPars ;
  _constPars = (RooArgList*) list.snapshot() ;
}


void RooFitResult::setInitParList(const RooArgList& list)
{
  // Fill the list of initial values of the floating parameters 
  if (_initPars) delete _initPars ;
  _initPars = (RooArgList*) list.snapshot() ;
}


void RooFitResult::setFinalParList(const RooArgList& list)
{
  // Fill the list of final values of the floating parameters 
  if (_finalPars) delete _finalPars ;
  _finalPars = (RooArgList*) list.snapshot() ;
}

RooPlot *RooFitResult::plotOn(RooPlot *frame, const char *parName1, const char *parName2,
			      const char *options) const {
  // Add objects to a 2D plot that represent the fit results for the
  // two named parameters.  The input frame with the objects added is
  // returned, or zero in case of an error.  Which objects are added
  // are determined by the options string which should be a concatenation
  // of the following (not case sensitive):
  //
  //   M - a marker at the best fit result
  //   E - an error ellipse calculated at 1-sigma using the error matrix at the minimum
  //   1 - the 1-sigma error bar for parameter 1
  //   2 - the 1-sigma error bar for parameter 2
  //   B - the bounding box for the error ellipse
  //   H - a line and horizontal axis for reading off the correlation coefficient
  //   V - a line and vertical axis for reading off the correlation coefficient
  //   A - draw axes for reading off the correlation coefficients with the H or V options
  //
  // You can change the attributes of objects in the returned RooPlot using the
  // various RooPlot::getAttXxx(name) member functions, e.g.
  //
  //   plot->getAttLine("contour")->SetLineStyle(kDashed);
  //
  // Use plot->Print() for a list of all objects and their names (unfortunately most
  // of the ROOT builtin graphics objects like TLine are unnamed). Drag the left mouse
  // button along the labels of either axis button to interactively zoom in a plot.

  // lookup the input parameters by name: we require that they were floated in our fit
  const RooRealVar *par1= dynamic_cast<const RooRealVar*>(floatParsFinal().find(parName1));
  if(0 == par1) {
    cout << "RooFitResult::correlationPlot: parameter not floated in fit: " << par1->GetName() << endl;
    return 0;
  }
  const RooRealVar *par2= dynamic_cast<const RooRealVar*>(floatParsFinal().find(parName2));
  if(0 == par2) {
    cout << "RooFitResult::correlationPlot: parameter not floated in fit: " << par2->GetName() << endl;
    return 0;
  }

  // options are not case sensitive
  TString opt(options);
  opt.ToUpper();

  // lookup the 2x2 covariance matrix elements for these variables
  Double_t x1= par1->getVal();
  Double_t x2= par2->getVal();
  Double_t s1= par1->getError();
  Double_t s2= par2->getError();
  Double_t rho= correlation(parName1, parName2);

  // add a 1-sigma error ellipse, if requested
  if(opt.Contains("E")) {
    RooEllipse *contour= new RooEllipse("contour",x1,x2,s1,s2,rho);
    frame->addPlotable(contour);
  }

  // add the error bar for parameter 1, if requested
  if(opt.Contains("1")) {
    TLine *hline= new TLine(x1-s1,x2,x1+s1,x2);
    hline->SetLineColor(kRed);
    frame->addObject(hline);
  }

  if(opt.Contains("2")) {
    TLine *vline= new TLine(x1,x2-s2,x1,x2+s2);
    vline->SetLineColor(kRed);
    frame->addObject(vline);
  }

  if(opt.Contains("B")) {
    TBox *box= new TBox(x1-s1,x2-s2,x1+s1,x2+s2);
    box->SetLineStyle(kDashed);
    box->SetLineColor(kRed);
    box->SetFillStyle(0);
    frame->addObject(box);
  }

  if(opt.Contains("H")) {
    TLine *line= new TLine(x1-rho*s1,x2-s2,x1+rho*s1,x2+s2);
    line->SetLineStyle(kDashed);
    line->SetLineColor(kBlue);
    frame->addObject(line);
    if(opt.Contains("A")) {
      TGaxis *axis= new TGaxis(x1-s1,x2-s2,x1+s1,x2-s2,-1.,+1.,502,"-=");
      axis->SetLineColor(kBlue);
      frame->addObject(axis);
    }
  }

  if(opt.Contains("V")) {
    TLine *line= new TLine(x1-s1,x2-rho*s2,x1+s1,x2+rho*s2);
    line->SetLineStyle(kDashed);
    line->SetLineColor(kBlue);
    frame->addObject(line);
    if(opt.Contains("A")) {
      TGaxis *axis= new TGaxis(x1-s1,x2-s2,x1-s1,x2+s2,-1.,+1.,502,"-=");
      axis->SetLineColor(kBlue);
      frame->addObject(axis);
    }
  }

  // add a marker at the fitted value, if requested
  if(opt.Contains("M")) {
    TMarker *marker= new TMarker(x1,x2,20);
    marker->SetMarkerColor(kBlack);
    frame->addObject(marker);
  }

  return frame;
}

const RooArgList& RooFitResult::randomizePars() const {
  // Return a list of floating parameter values that are perturbed from the final
  // fit values by random amounts sampled from the covariance matrix. The returned
  // object is overwritten with each call and belongs to the RooFitResult. Uses
  // the "square root method" to decompose the covariance matrix, which makes inverting
  // it unnecessary.

  Int_t nPar= _finalPars->getSize();
  if(0 == _randomPars) { // first-time initialization
    assert(0 != _finalPars);
    // create the list of random values to fill
    _randomPars= (RooArgList*)_finalPars->snapshot();
    // calculate the elements of the upper-triangular matrix L that gives Lt*L = C
    // where Lt is the transpose of L (the "square-root method")
    TMatrix L(nPar,nPar);
    for(Int_t iPar= 0; iPar < nPar; iPar++) {
      // calculate the diagonal term first
      L(iPar,iPar)= covariance(iPar,iPar);
      for(Int_t k= 0; k < iPar; k++) {
	Double_t tmp= L(k,iPar);
	L(iPar,iPar)-= tmp*tmp;
      }
      L(iPar,iPar)= sqrt(L(iPar,iPar));
      // then the off-diagonal terms
      for(Int_t jPar= iPar+1; jPar < nPar; jPar++) {
	L(iPar,jPar)= covariance(iPar,jPar);
	for(Int_t k= 0; k < iPar; k++) {
	  L(iPar,jPar)-= L(k,iPar)*L(k,jPar);
	}
	L(iPar,jPar)/= L(iPar,iPar);
      }
    }
    // remember Lt
    _Lt= new TMatrix(TMatrix::kTransposed,L);
  }
  else {
    // reset to the final fit values
    *_randomPars= *_finalPars;
  }

  // create a vector of unit Gaussian variables
  TVector g(nPar);
  for(Int_t k= 0; k < nPar; k++) g(k)= RooRandom::gaussian();
  // multiply this vector by Lt to introduce the appropriate correlations
  g*= (*_Lt);
  // add the mean value offsets and store the results
  TIterator *iter= _randomPars->createIterator();
  RooRealVar *par(0);
  Int_t index(0);
  while(0 != (par= (RooRealVar*)iter->Next())) {
    par->setVal(par->getVal() + g(index++));
  }
  delete iter;

  return *_randomPars;
}

Double_t RooFitResult::correlation(const char* parname1, const char* parname2) const 
{
  // Return the correlation between parameters 'par1' and 'par2'

  const RooArgList* row = correlation(parname1) ;
  if (!row) return 0. ;
  RooAbsArg* arg = _initPars->find(parname2) ;
  if (!arg) {
    cout << "RooFitResult::correlation: variable " << parname2 << " not a floating parameter in fit" << endl ;
    return 0. ;
  }
  return ((RooRealVar*)row->at(_initPars->index(arg)))->getVal() ;
}


const RooArgList* RooFitResult::correlation(const char* parname) const 
{
  // Return the set of correlation coefficients of parameter 'par' with
  // all other floating parameters

  RooAbsArg* arg = _initPars->find(parname) ;
  if (!arg) {
    cout << "RooFitResult::correlation: variable " << parname << " not a floating parameter in fit" << endl ;
    return 0 ;
  }    
  return (RooArgList*)_corrMatrix.At(_initPars->index(arg)) ;
}


Double_t RooFitResult::globalCorr(const char* parname) 
{
  // Return the global correlation of the named parameter
  RooAbsArg* arg = _initPars->find(parname) ;
  if (!arg) {
    cout << "RooFitResult::globalCorr: variable " << parname << " not a floating parameter in fit" << endl ;
    return 0 ;
  }    

  if (_globalCorr) {
    return ((RooAbsReal*)_globalCorr->at(_initPars->index(arg)))->getVal() ;
  } else {
    return 1.0 ; 
  }
}


const RooArgList* RooFitResult::globalCorr() 
{
  // Return the list of all global correlations
  return _globalCorr ;
}


Double_t RooFitResult::correlation(Int_t row, Int_t col) const {
  // Return a correlation matrix element addressed with numeric indices.

  const RooArgList *rowVec= (const RooArgList*)_corrMatrix.At(row);
  assert(0 != rowVec);
  const RooRealVar *elem= (const RooRealVar*)rowVec->at(col);
  assert(0 != elem);
  return elem->getVal();
}

Double_t RooFitResult::covariance(Int_t row, Int_t col) const {
  // Return the covariance matrix element addressed with numeric indices.

  const RooRealVar *rowVar= (const RooRealVar*)_finalPars->at(row);
  const RooRealVar *colVar= (const RooRealVar*)_finalPars->at(col);
  assert(0 != rowVar && 0 != colVar);
  return rowVar->getError()*colVar->getError()*correlation(row,col);  
}

void RooFitResult::printToStream(ostream& os, PrintOption opt, TString indent) const
{
  // Print fit result to stream 'os'. In Verbose mode, the contant parameters and
  // the initial and final values of the floating parameters are printed. 
  // Standard mode only the final values of the floating parameters are printed

  os << endl 
     << "  RooFitResult: minimized NLL value: " << _minNLL << ", estimated distance to minimum: " << _edm 
     << endl 
     << endl ;

  Int_t i ;
  if (opt>=Verbose) {
    if (_constPars->getSize()>0) {
      os << "    Constant Parameter    Value     " << endl
	 << "  --------------------  ------------" << endl ;

      for (i=0 ; i<_constPars->getSize() ; i++) {
	os << "  " << setw(20) << ((RooAbsArg*)_constPars->at(i))->GetName()
	   << "  " << setw(12) << Form("%12.4e",((RooRealVar*)_constPars->at(i))->getVal())
	   << endl ;
      }

      os << endl ;
    }


    os << "    Floating Parameter  InitialValue    FinalValue +/-  Error     GblCorr." << endl
       << "  --------------------  ------------  --------------------------  --------" << endl ;

    for (i=0 ; i<_finalPars->getSize() ; i++) {
      os << "  "    << setw(20) << ((RooAbsArg*)_finalPars->at(i))->GetName() ;
      os << "  "    << setw(12) << Form("%12.4e",((RooRealVar*)_initPars->at(i))->getVal())
	 << "  "    << setw(12) << Form("%12.4e",((RooRealVar*)_finalPars->at(i))->getVal())
	 << " +/- " << setw(9)  << Form("%9.2e",((RooRealVar*)_finalPars->at(i))->getError()) ;

      if (_globalCorr) {
	os << "  "    << setw(8)  << Form("%8.6f" ,((RooRealVar*)_globalCorr->at(i))->getVal()) ;
      } else {
	os << "  <none>" ;
      } 

      os << endl ;
    }

  } else {
    os << "    Floating Parameter    FinalValue +/-  Error   " << endl
       << "  --------------------  --------------------------" << endl ;

    for (i=0 ; i<_finalPars->getSize() ; i++) {
      os << "  "    << setw(20) << ((RooAbsArg*)_finalPars->at(i))->GetName()
	 << "  "    << setw(12) << Form("%12.4e",((RooRealVar*)_finalPars->at(i))->getVal())
	 << " +/- " << setw(9)  << Form("%9.2e",((RooRealVar*)_finalPars->at(i))->getError())
	 << endl ;
    }
  }
  

  os << endl ;
}


void RooFitResult::fillCorrMatrix()
{
  // Extract the correlation matrix and the global correlation coefficients from the MINUIT memory buffer 
  // and fill the internal arrays.

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
  if (_globalCorr) delete _globalCorr ;

  _corrMatrix.Delete();

  // Build holding arrays for correlation coefficients
  _globalCorr = new RooArgList("globalCorrelations") ;
  TIterator* vIter = _initPars->createIterator() ;
  RooAbsArg* arg ;
  Int_t idx(0) ;
  while(arg=(RooAbsArg*)vIter->Next()) {
    // Create global correlation value holder
    TString gcName("GC[") ;
    gcName.Append(arg->GetName()) ;
    gcName.Append("]") ;
    TString gcTitle(arg->GetTitle()) ;
    gcTitle.Append(" Global Correlation") ;
    _globalCorr->addOwned(*(new RooRealVar(gcName.Data(),gcTitle.Data(),0.))) ;

    // Create array with correlation holders for this parameter
    TString name("C[") ;
    name.Append(arg->GetName()) ;
    name.Append(",*]") ;
    RooArgList* corrMatrixRow = new RooArgList(name.Data()) ;
    _corrMatrix.Add(corrMatrixRow) ;
    TIterator* vIter2 = _initPars->createIterator() ;
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
      corrMatrixRow->addOwned(*(new RooRealVar(cName.Data(),cTitle.Data(),0.))) ;      
    }
    delete vIter2 ;
    idx++ ;
  }
  delete vIter ;

  TIterator *gcIter = _globalCorr->createIterator() ;

  // Extract correlation information for MINUIT (code taken from TMinuit::mnmatu() )

  // WVE: This code directly manipulates minuit internal workspace, 
  //      if TMinuit code changes this may need updating
  Int_t ndex, i, j, m, n, ncoef, nparm, /*id,*/ it, ix;
  Int_t ndi, ndj /*, iso, isw2, isw5*/;
  ncoef = (gMinuit->fNpagwd - 19) / 6;
  nparm = TMath::Min(gMinuit->fNpar,ncoef);
  RooRealVar* gcVal = 0;
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
    TIterator* cIter = ((RooArgList*)_corrMatrix.At(i-1))->createIterator() ;
    for (it = 1; it <= gMinuit->fNpar ; ++it) {
      RooRealVar* cVal = (RooRealVar*) cIter->Next() ;
      cVal->setVal(gMinuit->fMATUvline[it-1]) ;
    }
    delete cIter ;
  }

  delete gcIter ;
} 




RooFitResult* RooFitResult::lastMinuitFit(const RooArgList& varList) 
{
  // Verify length of supplied varList
  if (varList.getSize()>0 && varList.getSize()!=gMinuit->fNu) {
    cout << "RooFitResult::lastMinuitFit: ERROR: supplied variable list must be either empty " << endl 
	 << "                             or match the number of variables of the last fit (" << gMinuit->fNu << ")" << endl ;
    return 0 ;
  }

  // Verify that all members of varList are of type RooRealVar
  TIterator* iter = varList.createIterator() ;
  RooAbsArg* arg  ;
  while(arg=(RooAbsArg*)iter->Next()) {
    if (!dynamic_cast<RooRealVar*>(arg)) {
      cout << "RooFitResult::lastMinuitFit: ERROR: variable '" << arg->GetName() << "' is not of type RooRealVar" << endl ;
      return 0 ;
    }
  }
  delete iter ;

  RooFitResult* r = new RooFitResult("lastMinuitFit","Last MINUIT fit") ;

  // Extract names of fit parameters from MINUIT 
  // and construct corresponding RooRealVars
  RooArgList constPars("constPars") ;
  RooArgList floatPars("floatPars") ;

  Int_t i ;
  for (i = 1; i <= gMinuit->fNu; ++i) {
    if (gMinuit->fNvarl[i-1] < 0) continue;
    Int_t l = gMinuit->fNiofex[i-1];
    TString varName(gMinuit->fCpnam[i-1]) ;
    Bool_t isConst(l==0) ;
    
    Double_t xlo = gMinuit->fAlim[i-1];
    Double_t xhi = gMinuit->fBlim[i-1];
    Double_t xerr = gMinuit->fWerr[l-1];
    Double_t xval = gMinuit->fU[i-1] ;

    RooRealVar* var ;
    if (varList.getSize()==0) {

      if ((xlo<xhi) && !isConst) {
	var = new RooRealVar(varName,varName,xval,xlo,xhi) ;
      } else {
	var = new RooRealVar(varName,varName,xval) ;
      }
      var->setConstant(isConst) ;
    } else {

      var = (RooRealVar*) varList.at(i-1)->Clone() ;
      var->setConstant(isConst) ;
      var->setVal(xval) ;
      if (xlo<xhi) {
	var->setFitRange(xlo,xhi) ;
      }
      if (varName.CompareTo(var->GetName())) {
	cout << "RooFitResult::lastMinuitFit: fit parameter '" << varName 
	     << "' stored in variable '" << var->GetName() << "'" << endl ;
      }

    }

    if (isConst) {
      constPars.addOwned(*var) ;
    } else {
      var->setError(xerr) ;
      floatPars.addOwned(*var) ;
    }
  }
  
  r->setConstParList(constPars) ;
  r->setInitParList(floatPars) ;
  r->setFinalParList(floatPars) ;
  r->setMinNLL(gMinuit->fAmin) ;
  r->setEDM(gMinuit->fEDM) ; 
  r->setStatus(gMinuit->fStatus) ;
  r->fillCorrMatrix() ;

  return r ;
}
