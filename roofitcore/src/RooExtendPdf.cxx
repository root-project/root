/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooExtendPdf.cc,v 1.3 2001/11/19 07:23:56 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   10-Oct-2001 WV Created initial version
 *
 * Copyright (C) 2000 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [PDF] --
//  RooExtendPdf is a wrappper around an existing PDF that adds a 
//  parameteric extended likelihood term to the PDF, optionally multiplied by a 
//  fractional term from a partial normalization of the PDF:
//
//  nExpected = N   _or Expected = N * frac 
//
//  where N is supplied as a RooAbsReal to RooExtendPdf.
//  The fractional term is defined as
//                          _       _ _   _  _
//            Int(cutRegion[x]) pdf(x,y) dx dy 
//     frac = ---------------_-------_-_---_--_ 
//            Int(normRegion[x]) pdf(x,y) dx dy 
//
//        _                                                               _
//  where x is the set of dependents involved in the selection region and y
//  is the set of remaining dependents.
//            _
//  cutRegion[x] is an limited integration range that is contained in
//  the nominal integration range normRegion[x[]
//

#include "RooFitCore/RooExtendPdf.hh"
#include "RooFitCore/RooArgList.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooFormulaVar.hh"

ClassImp(RooExtendPdf)
;


RooExtendPdf::RooExtendPdf(const char *name, const char *title, const RooAbsPdf& pdf,
			   const RooAbsReal& norm) :
  RooAbsPdf(name,title),
  _pdf("pdf","PDF",this,(RooAbsReal&)pdf),
  _n("n","Normalization",this,(RooAbsReal&)norm),
  _cutDepSet("cutDepSet","Set of dependent with fractional range",this),
  _origDepSet("origDepSet","Set of dependent with full integration range",this),
  _lastFracSet(0),
  _fracIntegral(0),
  _integralCompSet(0),
  _useFrac(kFALSE)
{
  // Constructor. The ExtendedPdf behaves identical to the supplied input pdf,
  // but adds an extended likelihood term. The expected number of events return
  // is 'norm'

  // Copy various setting from pdf
//   setPlotRange(_pdf.arg().getPlotMin(),_pdf.arg().getPlotMax()) ;
//   setPlotBins(_pdf.arg().getPlotBins()) ;
  setUnit(_pdf.arg().getUnit()) ;
  setPlotLabel(_pdf.arg().getPlotLabel()) ;
}



RooExtendPdf::RooExtendPdf(const char *name, const char *title, const RooAbsPdf& pdf,
			   const RooAbsReal& norm, const RooArgList& depList, 
			   const RooArgList& cutDepList) :
  RooAbsPdf(name,title),
  _pdf("pdf","PDF",this,(RooAbsReal&)pdf),
  _n("n","Normalization",this,(RooAbsReal&)norm),
  _cutDepSet("cutDepSet","Set of dependent with fractional range",this),
  _origDepSet("origDepSet","Set of dependent with full integration range",this),
  _lastFracSet(0),
  _fracIntegral(0),
  _integralCompSet(0),
  _useFrac(kTRUE)
{
  // Constructor. The ExtendedPdf behaves identical to the supplied input pdf,
  // but adds an extended likelihood term. The expected number of events return
  // is 'norm' multiplied by a normalization fraction from the pdf. 
  //
  // The dimensions of the cut region should be listed in 'depList'. 
  // The reduced integration range for each dimension in 'depList' should be expressed 
  // in the fit range of a dummy RooRealVar in each matching slot in 'cutDepList'
  //
  // For example, the following code 
  //
  //   RooRealVar x("x","x",-10,10)
  //   RooExamplePdf pdf("pdf","pdf",x)
  //
  //   RooRealVar nevt("nevt","number of expected events")
  //   RooRealVar xcut("xcut","xcut",-3,3)
  //   RooExtendPdf epdf("epdf","...",pdf,nevt,x,cut)
  //
  // constructs an extended pdf with the number of expected events as
  //
  //   nExpected = nevt * Int(-3,3)pdf(x)dx / Int(-10,10)pdf(x)dx
  //

  // Check if dependent and replacement list have same length
  if (depList.getSize() != cutDepList.getSize()) {
    cout << "RooExtendPdf::ctor(" << GetName() 
	 << ") list of cut dependents and their replacements must have equal length" << endl ;
    assert(0) ;
  }
  
  // Copy various setting from pdf
//   setPlotRange(_pdf.arg().getPlotMin(),_pdf.arg().getPlotMax()) ;
//   setPlotBins(_pdf.arg().getPlotBins()) ;
  setUnit(_pdf.arg().getUnit()) ;
  setPlotLabel(_pdf.arg().getPlotLabel()) ;

  // Loop over all dependents to be cut on
  TIterator* dIter = depList.createIterator() ;
  TIterator* cIter = cutDepList.createIterator() ;
  RooAbsArg* dep, *cutDep ;
  while(dep=(RooAbsArg*)dIter->Next()) {
    cutDep= (RooAbsArg*)cIter->Next() ;

    // Check if original and replacement variable are both real lvalues
    RooAbsRealLValue* orig = dynamic_cast<RooAbsRealLValue*>(dep) ;
    RooAbsRealLValue* repl = dynamic_cast<RooAbsRealLValue*>(cutDep) ;
    if (!orig || !repl) {
      cout << "RooExtendPdf::ctor(" << GetName() << "): ERROR: " << dep->GetName() << " and " 
	   << cutDep->GetName() << " must be both RooAbsRealLValues, this pair ignored" << endl ;
      continue ;
    }
    
    // Check if fraction range is fully contained in integration range
    if (repl->getFitMin()<orig->getFitMin() || repl->getFitMax()>orig->getFitMax()) {
      cout << "RooExtendPdf::ctor(" << GetName() << "): WARNING: fit range of " 
	   << repl->GetName() << " is not fully contained in " << orig->GetName() << endl ;
    }
    
    // Affix appropriate name-changing server attribute to each cut variable
    TString origNameLabel("ORIGNAME:") ;
    origNameLabel.Append(orig->GetName()) ;
    repl->setAttribute(origNameLabel,kTRUE);
  }

  // Store cut dependents
  _cutDepSet.add(cutDepList) ;
  _origDepSet.add(depList) ;
}



RooExtendPdf::RooExtendPdf(const RooExtendPdf& other, const char* name) :
  RooAbsPdf(other,name),
  _pdf("pdf",this,other._pdf),
  _n("n",this,other._n),
  _cutDepSet("cutDepSet",this,other._cutDepSet),
  _origDepSet("origDepSet",this,other._origDepSet),
  _lastFracSet(0),
  _fracIntegral(0),
  _integralCompSet(0),
  _useFrac(other._useFrac)
{
  // Copy constructor
}


RooExtendPdf::~RooExtendPdf() 
{
  // Destructor

  // Delete any owned components
  if (_fracIntegral) {
    delete _integralCompSet ;
  }
}



Double_t RooExtendPdf::expectedEvents() const 
{
  // Return the number of expected events, which is
  //
  // n / [ Int(xC,yF) pdf(x,y) / Int(xF,yF) pdf(x,y) ]
  //
  // Where x is the set of dependents with cuts defined
  // and y are the other dependents. xC is the integration
  // of x over the cut range, xF is the integration of
  // x over the full range.

  RooAbsPdf& pdf = (RooAbsPdf&)_pdf.arg() ;

  Double_t nExp = _n ;

  // Optionally multiply with fractional normalization
  if (_useFrac) {
    // Use current PDF normalization, if defined, use cut set otherwise
    const RooArgSet* npset = pdf._lastNormSet ;
    const RooArgSet* nset = npset ? npset : (const RooArgSet*) &_origDepSet ;

    Double_t normInt = pdf.getNorm(nset) ;
    
    // Update fraction integral
    syncFracIntegral() ;
    
    // Evaluate fraction integral and return normalized by full integral
    Double_t fracInt = _fracIntegral->getVal() ;
    if ( fracInt == 0. || normInt == 0. || _n == 0.) {
      cout << "RooExtendPdf(" << GetName() << ") WARNING: nExpected = " << _n << " / ( " 
	   << fracInt << " / " << normInt << " ), for nset = " ;
      if (nset) nset->Print("1") ; else cout << "<none>" << endl ;
    }

    nExp *= (fracInt / normInt) ;
  }

  // Multiply with original Nexpected, if defined
  if (pdf.canBeExtended()) nExp *= pdf.expectedEvents() ;

  return nExp ;
}


void RooExtendPdf::syncFracIntegral() const
{
  // Create the fraction integral by clone the
  // PDFs normalization integral and replacing
  // the regular dependents in the clone by
  // the internal set of dependents that specify
  // a more restrictive integration range
  
  RooAbsPdf& pdf = (RooAbsPdf&) _pdf.arg() ;

  // Check first if any changes are needed
  if (_lastFracSet == pdf._lastNormSet) return ;
  _lastFracSet = pdf._lastNormSet ;

  // Delete existing integral
  if (_fracIntegral) {
    delete _integralCompSet ;
  }

  // Make list of all nodes that need to be cloned:
  // All PDF branch nodes, and the RooRealIntegral object itself
  RooArgSet pdfNodeList ;
  pdf.branchNodeServerList(&pdfNodeList) ;


  //Check if PDF has dummy normalization 
  if (dynamic_cast<RooRealIntegral*>(pdf._norm)) {

    // If not, clone original integral and add to owned set
    pdfNodeList.add(*pdf._norm) ;
    _integralCompSet = (RooArgSet*) pdfNodeList.snapshot(kFALSE) ;
    _fracIntegral = (RooRealIntegral*) _integralCompSet->find(pdf._norm->GetName()) ;

    // Replace dependents involved in cut with internal set
    _fracIntegral->recursiveRedirectServers(_cutDepSet,kFALSE,kTRUE) ;

  } else {
    // If dummy, make both normalization and fraction integral here

    // Cannot dereference _lastFracSet since it may be a dangling ptr
    // Recreate a similar argset by inflating the name set
    RooArgSet pdfLeafList ;
    pdf.leafNodeServerList(&pdfLeafList) ;
    const RooArgSet* fracNormSet = pdf._lastNameSet.select(pdfLeafList) ;
    
    TString fname(pdf.GetName()) ; fname.Append("FracNorm") ;
    TString ftitle(pdf.GetTitle()) ; ftitle.Append(" Fraction Integral") ;

    // Create fractional integral from PDF clone
    _integralCompSet = (RooArgSet*) pdfNodeList.snapshot(kFALSE) ;
    RooAbsPdf* pdfClone = (RooAbsPdf*) _integralCompSet->find(pdf.GetName()) ;
    _fracIntegral = new RooRealIntegral(fname,ftitle,*pdfClone,*fracNormSet) ;
    _integralCompSet->addOwned(*_fracIntegral) ;

    // Replace dependents involved in fractional with internal set
    _fracIntegral->recursiveRedirectServers(_cutDepSet,kFALSE,kTRUE) ;

    TString nname(pdf.GetName()) ; nname.Append("Norm") ;
    TString ntitle(pdf.GetTitle()) ; ntitle.Append(" Integral") ;

    TString rname(pdf.GetName()) ; rname.Append("FracRatio") ;
    TString rtitle(pdf.GetTitle()) ; rtitle.Append(" Integral Ratio") ;
    
    // Create full normalization integral
    RooRealIntegral* normIntegral = new RooRealIntegral(nname,ntitle,pdf,*fracNormSet) ;
    RooFormulaVar* ratio= new RooFormulaVar(rname,rtitle,"@0/@1",RooArgList(*_fracIntegral,*normIntegral)) ;
    _integralCompSet->addOwned(*normIntegral) ;
    _integralCompSet->addOwned(*ratio) ;
    _fracIntegral = ratio ;

    delete fracNormSet ;
  }
}




void RooExtendPdf::getParametersHook(const RooArgSet* nset, RooArgSet* list) const 
{
  // Remove fake dependents from list
  list->remove(_cutDepSet,kTRUE,kTRUE) ;
}



void RooExtendPdf::getDependentsHook(const RooArgSet* nset, RooArgSet* list) const 
{
  // Remove fake dependents from list
  list->remove(_cutDepSet,kTRUE,kTRUE) ;
}
