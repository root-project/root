/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooCutNorm.cc,v 1.2 2001/10/06 07:28:59 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   10-Oct-2001 WV Created initial version
 *
 * Copyright (C) 2000 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [PDF] --
//  RooNorm is a PDF with a flat likelihood distribution that introduces a 
//  parameteric extended likelihood term into a PDF, multiplied by a 
//  fractional term from a partial normalization of a supplied PDF.
//
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
//  Typically RooCutNorm is used to add the extended likelihood term to another 
//  PDF by multiplying RooNorm with that PDF using RooProdPdf.

#include "RooFitCore/RooCutNorm.hh"
#include "RooFitCore/RooArgList.hh"

ClassImp(RooCutNorm)
;


RooCutNorm::RooCutNorm(const char *name, const char *title, const RooAbsReal& norm, 
		       const RooAbsPdf& pdf, const RooArgList& depList, const RooArgList& cutDepList) :
  RooAbsPdf(name,title),
  _pdf("pdf","PDF",this,(RooAbsReal&)pdf),
  _n("n","Normalization",this,(RooAbsReal&)norm),
  _cutDepSet("cutDepSet","Set of dependent with fractional range",this),
  _origDepSet("origDepSet","Set of dependent with full integration range",this),
  _lastFracSet(0),
  _fracIntegral(0),
  _integralCompSet(0)
{
  // Constructor. The extended likelihood is taken from 'norm' multiplied by  
  // a normalization fraction from 'pdf'. The dimensions of the cut region 
  // should be listed in 'depList'. The reduced integration range for each
  // dimension in 'depList' should be expressed in the fit range of a dummy
  // RooRealVar in each matching slot in 'cutDepList'
  //
  // For example, the following code 
  //
  //   RooRealVar x("x","x",-10,10)
  //   RooExamplePdf pdf("pdf","pdf",x)
  //
  //   RooRealVar nevt("nevt","number of expected events")
  //   RooRealVar xcut("xcut","xcut",-3,3)
  //   RooCutNorm cutnorm("cutnorm","...",pdf,nevt,x,cut)
  //
  // constructs a cutnorm with the number of expected events as
  //
  //   nExpected = nevt * Int(-3,3)pdf(x)dx / Int(-10,10)pdf(x)dx
  //
  
  // Check if dependent and replacement list have same length
  if (depList.getSize() != cutDepList.getSize()) {
    cout << "RooCutNorm::ctor(" << GetName() 
	 << ") list of cut dependents and their replacements must have equal length" << endl ;
    assert(0) ;
  }
  
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
      cout << "RooCutNorm::ctor(" << GetName() << "): ERROR: " << dep->GetName() << " and " 
	   << cutDep->GetName() << " must be both RooAbsRealLValues, this pair ignored" << endl ;
      continue ;
    }
    
    // Check if fraction range is fully contained in integration range
    if (repl->getFitMin()<orig->getFitMin() || repl->getFitMax()>orig->getFitMax()) {
      cout << "RooCutNorm::ctor(" << GetName() << "): WARNING: fit range of " 
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



RooCutNorm::RooCutNorm(const RooCutNorm& other, const char* name) :
  RooAbsPdf(other,name),
  _pdf("pdf",this,other._pdf),
  _n("n",this,other._n),
  _cutDepSet("cutDepSet",this,other._cutDepSet),
  _origDepSet("origDepSet",this,other._origDepSet),
  _lastFracSet(0),
  _fracIntegral(0),
  _integralCompSet(0)
{
  // Copy constructor
}


RooCutNorm::~RooCutNorm() 
{
  // Destructor

  // Delete any owned components
  if (_fracIntegral) {
    delete _integralCompSet ;
  }
}



Double_t RooCutNorm::expectedEvents() const 
{
  // Return the number of expected events, which is
  //
  // n * [ Int(xC,yF) pdf(x,y) / Int(xF,yF) pdf(x,y) ]
  //
  // Where x is the set of dependents with cuts defined
  // and y are the other dependents. xC is the integration
  // of x over the cut range, xF is the integration of
  // x over the full range.

  // Use current PDF normalization, if defined, use cut set otherwise
  const RooArgSet* nset = _lastNormSet ? _lastNormSet : (const RooArgSet*) &_origDepSet ;
  Double_t normInt = ((RooAbsPdf&)_pdf.arg()).getNorm(nset) ;

  // Update fraction integral
  syncFracIntegral() ;

  // Evaluate fraction integral and return normalized by full integral
  Double_t fracInt = _fracIntegral->getVal() ;
  return  _n * fracInt / normInt ;
}


void RooCutNorm::syncFracIntegral() const
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

  // Clone integral (including PDF) from present PDF normalization object

  // Make list of all nodes that need to be cloned:
  // All PDF branch nodes, and the RooRealIntegral object itself
  RooArgSet pdfNodeList ;
  pdf.branchNodeServerList(&pdfNodeList) ;
  pdfNodeList.add(*pdf._norm) ;

  // Shallow-clone list of nodes 
  _integralCompSet = (RooArgSet*) pdfNodeList.snapshot(kFALSE) ;
  _fracIntegral = (RooRealIntegral*) _integralCompSet->find(pdf._norm->GetName()) ;

  // Replace dependents involved in cut with internal set
  _fracIntegral->recursiveRedirectServers(_cutDepSet,kFALSE,kTRUE) ;
}
