/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooChi2Var.cc,v 1.6 2003/04/09 01:33:58 wverkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2002, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
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
#include "RooFitCore/RooCmdConfig.hh"

ClassImp(RooChi2Var)
;


RooCmdArg Extended() { return RooCmdArg("Extended",1,0,0,0,0,0,0,0) ; }
RooCmdArg DataError(RooDataHist::ErrorType etype) { return RooCmdArg("DataError",(Int_t)etype,0,0,0,0,0,0,0) ; }
RooCmdArg NumCPU(Int_t nCPU) { return RooCmdArg("NumCPU",nCPU,0,0,0,0,0,0,0) ; }


RooChi2Var::RooChi2Var(const char *name, const char* title, RooAbsPdf& pdf, RooDataHist& data,
		       const RooCmdArg& arg1,const RooCmdArg& arg2,const RooCmdArg& arg3,
		       const RooCmdArg& arg4,const RooCmdArg& arg5,const RooCmdArg& arg6,
		       const RooCmdArg& arg7,const RooCmdArg& arg8,const RooCmdArg& arg9) :
  RooAbsOptGoodnessOfFit(name,title,pdf,data,RooArgSet(),
			 RooCmdConfig::decodeIntOnTheFly("RooChi2Var::RooChi2Var","NumCPU",0,1,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9))
{
  RooCmdConfig pc("RooChi2Var::RooChi2Var") ;
  pc.defineInt("extended","Extended",0,kFALSE) ;
  pc.defineInt("etype","DataError",0,(Int_t)RooDataHist::Poisson) ;

  pc.process(arg1) ;  pc.process(arg2) ;  pc.process(arg3) ;
  pc.process(arg4) ;  pc.process(arg5) ;  pc.process(arg6) ;
  pc.process(arg7) ;  pc.process(arg8) ;  pc.process(arg9) ;

  _extended = pc.getInt("extended") ;
  _etype = (RooDataHist::ErrorType) pc.getInt("etype") ;

}


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
    nDataTotal = _pdfClone->expectedEvents() ;
  } else {
    nDataTotal = _dataClone->sumEntries() ;
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
    data->weightError(eIntLo,eIntHi,_etype) ;
    Double_t eInt = (eExt>0) ? eIntHi : eIntLo ;
    
    // Skip cases where pdf=0 and there is no data
    if (eInt==0. && nData==0. && nPdf==0) continue ;

    // Return 0 if eInt=0, special handling in MINUIT will follow
    if (eInt==0.) {
      cout << "RooChi2Var::RooChi2Var(" << GetName() << ") INFINITY ERROR: bin " << i 
	   << " has zero error, but function is not zero (" << nPdf << ")" << endl ;
      return 0 ;
    }

    result += eExt*eExt/(eInt*eInt) ;
  }
  

  return result ;
}



