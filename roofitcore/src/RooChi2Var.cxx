/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, UC Irvine, kirkby@hep.uci.edu
 * History:
 *   25-Mar-2002 WV Created initial version
 *
 * Copyright (C) 2002 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [PDF] --
// Class RooChi2Var implements a simple chi^2 calculation from a binned dataset
// and a PDF. The chi^2 is calculated as 
//
//             / (f_PDF * N_tot/ V_bin) - N_bin \+2
//  Sum[bins] |  ------------------------------ |
//             \         err_bin                /
//
// If no user-defined errors are defined for the dataset, poisson errors
// are used. In extended PDF mode, N_tot is substituted with N_expected.

#include "RooFitCore/RooChi2Var.hh"
#include "RooFitCore/RooDataHist.hh"
#include "RooFitCore/RooAbsPdf.hh"

ClassImp(RooChi2Var)
;

RooChi2Var::RooChi2Var(const char *name, const char *title, RooAbsPdf& pdf, RooDataHist& data,
		     Bool_t extended, Int_t nCPU) : 
  RooAbsOptGoodnessOfFit(name,title,pdf,data,RooArgSet(),nCPU),
  _extended(extended)
{
  
}


RooChi2Var::RooChi2Var(const char *name, const char *title, RooAbsPdf& pdf, RooDataHist& data,
		     const RooArgSet& projDeps, Bool_t extended, Int_t nCPU) : 
  RooAbsOptGoodnessOfFit(name,title,pdf,data,projDeps,nCPU),
  _extended(extended)
{
  
}


RooChi2Var::RooChi2Var(const RooChi2Var& other, const char* name) : 
  RooAbsOptGoodnessOfFit(other,name),
  _extended(other._extended)
{
}


RooChi2Var::~RooChi2Var()
{
}


Double_t RooChi2Var::evaluatePartition(Int_t firstEvent, Int_t lastEvent) const 
{
  Int_t i ;
  Double_t result(0) ;

  // Determine total number of data events to be used for PDF normalization
  Double_t nDataTotal ;
  if (_extended) {
    nDataTotal = _pdfClone->extendedTerm(_dataClone->numEntries(kTRUE));
  } else {
    nDataTotal = _dataClone->numEntries(kTRUE) ;
  }

  // Loop over bins of dataset
  RooDataHist* data = (RooDataHist*) _dataClone ;
    for (i=firstEvent ; i<lastEvent ; i++) {
    
    // get the data values for this event
    data->get(i);
    Double_t nData = data->weight() ;
    Double_t nPdf = _pdfClone->getVal(_normSet) * nDataTotal * data->binVolume() ;

    Double_t eExt = nPdf-nData ;

    Double_t eIntLo,eIntHi ;
    data->weightError(eIntLo,eIntHi) ;
    Double_t eInt = (eExt>0) ? eIntHi : eIntLo ;

    result += eExt*eExt/(eInt*eInt) ;
  }
  

  return result ;
}



