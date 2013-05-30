/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

//////////////////////////////////////////////////////////////////////////////
// 
// Class RooChi2Var implements a simple chi^2 calculation from a binned dataset
// and a PDF. The chi^2 is calculated as 
//
//             / (f_PDF * N_tot/ V_bin) - N_bin \+2
//  Sum[bins] |  ------------------------------ |
//             \         err_bin                /
//
// If no user-defined errors are defined for the dataset, poisson errors
// are used. In extended PDF mode, N_tot is substituted with N_expected.
//

#include "RooFit.h"

#include "RooChi2Var.h"
#include "RooChi2Var.h"
#include "RooDataHist.h"
#include "RooAbsPdf.h"
#include "RooCmdConfig.h"
#include "RooMsgService.h"

#include "Riostream.h"

#include "RooRealVar.h"
#include "RooAbsDataStore.h"


using namespace std;

ClassImp(RooChi2Var)
;

RooArgSet RooChi2Var::_emptySet ;


//_____________________________________________________________________________
RooChi2Var::RooChi2Var(const char *name, const char* title, RooAbsReal& func, RooDataHist& hdata,
		       const RooCmdArg& arg1,const RooCmdArg& arg2,const RooCmdArg& arg3,
		       const RooCmdArg& arg4,const RooCmdArg& arg5,const RooCmdArg& arg6,
		       const RooCmdArg& arg7,const RooCmdArg& arg8,const RooCmdArg& arg9) :
  RooAbsOptTestStatistic(name,title,func,hdata,_emptySet,
			 RooCmdConfig::decodeStringOnTheFly("RooChi2Var::RooChi2Var","RangeWithName",0,"",arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9),
			 0,
			 RooCmdConfig::decodeIntOnTheFly("RooChi2Var::RooChi2Var","NumCPU",0,1,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9),
			 RooFit::Interleave,
			 RooCmdConfig::decodeIntOnTheFly("RooChi2Var::RooChi2Var","Verbose",0,1,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9),
			 0)
  //  RooChi2Var constructor. Optional arguments taken
  //
  //  DataError()  -- Choose between Poisson errors and Sum-of-weights errors
  //  NumCPU()     -- Activate parallel processing feature
  //  Range()      -- Fit only selected region
  //  Verbose()    -- Verbose output of GOF framework
{
  RooCmdConfig pc("RooChi2Var::RooChi2Var") ;
  pc.defineInt("etype","DataError",0,(Int_t)RooDataHist::Auto) ;  
  pc.defineInt("extended","Extended",0,kFALSE) ;
  pc.allowUndefined() ;

  pc.process(arg1) ;  pc.process(arg2) ;  pc.process(arg3) ;
  pc.process(arg4) ;  pc.process(arg5) ;  pc.process(arg6) ;
  pc.process(arg7) ;  pc.process(arg8) ;  pc.process(arg9) ;

  if (func.IsA()->InheritsFrom(RooAbsPdf::Class())) {
    _funcMode = pc.getInt("extended") ? ExtendedPdf : Pdf ;
  } else {
    _funcMode = Function ;
  }
  _etype = (RooDataHist::ErrorType) pc.getInt("etype") ;

  if (_etype==RooAbsData::Auto) {
    _etype = hdata.isNonPoissonWeighted()? RooAbsData::SumW2 : RooAbsData::Expected ;
  }

}



//_____________________________________________________________________________
RooChi2Var::RooChi2Var(const char *name, const char* title, RooAbsPdf& pdf, RooDataHist& hdata,
		       const RooCmdArg& arg1,const RooCmdArg& arg2,const RooCmdArg& arg3,
		       const RooCmdArg& arg4,const RooCmdArg& arg5,const RooCmdArg& arg6,
		       const RooCmdArg& arg7,const RooCmdArg& arg8,const RooCmdArg& arg9) :
  RooAbsOptTestStatistic(name,title,pdf,hdata,
			 *(const RooArgSet*)RooCmdConfig::decodeObjOnTheFly("RooChi2Var::RooChi2Var","ProjectedObservables",0,&_emptySet
									    ,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9),
			 RooCmdConfig::decodeStringOnTheFly("RooChi2Var::RooChi2Var","RangeWithName",0,"",arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9),
			 RooCmdConfig::decodeStringOnTheFly("RooChi2Var::RooChi2Var","AddCoefRange",0,"",arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9),
			 RooCmdConfig::decodeIntOnTheFly("RooChi2Var::RooChi2Var","NumCPU",0,1,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9),
			 RooFit::Interleave,
			 RooCmdConfig::decodeIntOnTheFly("RooChi2Var::RooChi2Var","Verbose",0,1,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9),
			 RooCmdConfig::decodeIntOnTheFly("RooChi2Var::RooChi2Var","SplitRange",0,0,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9))             
  //  RooChi2Var constructor. Optional arguments taken
  //
  //  Extended()   -- Include extended term in calculation
  //  DataError()  -- Choose between Poisson errors and Sum-of-weights errors
  //  NumCPU()     -- Activate parallel processing feature
  //  Range()      -- Fit only selected region
  //  SumCoefRange() -- Set the range in which to interpret the coefficients of RooAddPdf components 
  //  SplitRange() -- Fit range is split by index catory of simultaneous PDF
  //  ConditionalObservables() -- Define projected observables 
  //  Verbose()    -- Verbose output of GOF framework
{
  RooCmdConfig pc("RooChi2Var::RooChi2Var") ;
  pc.defineInt("extended","Extended",0,kFALSE) ;
  pc.defineInt("etype","DataError",0,(Int_t)RooDataHist::Auto) ;  
  pc.allowUndefined() ;

  pc.process(arg1) ;  pc.process(arg2) ;  pc.process(arg3) ;
  pc.process(arg4) ;  pc.process(arg5) ;  pc.process(arg6) ;
  pc.process(arg7) ;  pc.process(arg8) ;  pc.process(arg9) ;

  _funcMode = pc.getInt("extended") ? ExtendedPdf : Pdf ;
  _etype = (RooDataHist::ErrorType) pc.getInt("etype") ;
  if (_etype==RooAbsData::Auto) {
    _etype = hdata.isNonPoissonWeighted()? RooAbsData::SumW2 : RooAbsData::Expected ;
  }
}



//_____________________________________________________________________________
RooChi2Var::RooChi2Var(const char *name, const char *title, RooAbsPdf& pdf, RooDataHist& hdata,
		       Bool_t extended, const char* cutRange, const char* addCoefRange,
		       Int_t nCPU, RooFit::MPSplit interleave, Bool_t verbose, Bool_t splitCutRange, RooDataHist::ErrorType etype) : 
  RooAbsOptTestStatistic(name,title,pdf,hdata,RooArgSet(),cutRange,addCoefRange,nCPU,interleave,verbose,splitCutRange),
   _etype(etype), _funcMode(extended?ExtendedPdf:Pdf)
{
  // Constructor of a chi2 for given p.d.f. with respect given binned
  // dataset. If cutRange is specified the calculation of the chi2 is
  // restricted to that named range. If addCoefRange is specified, the
  // interpretation of fractions for all component RooAddPdfs that do
  // not have a frozen range interpretation is set to chosen range
  // name. If nCPU is greater than one the chi^2 calculation is
  // paralellized over the specified number of processors. If
  // interleave is true the partitioning of event over processors
  // follows a (i % n == i_set) strategy rather than a bulk
  // partitioning strategy which may result in unequal load balancing
  // in binned datasets with many (adjacent) zero bins. If
  // splitCutRange is true the cutRange is used to construct an
  // individual cutRange for each RooSimultaneous index category state
  // name cutRange_{indexStateName}.
}



//_____________________________________________________________________________
RooChi2Var::RooChi2Var(const char *name, const char *title, RooAbsReal& func, RooDataHist& hdata,
		       const RooArgSet& projDeps, RooChi2Var::FuncMode fmode, const char* cutRange, const char* addCoefRange, 
		       Int_t nCPU, RooFit::MPSplit interleave, Bool_t verbose, Bool_t splitCutRange, RooDataHist::ErrorType etype) : 
  RooAbsOptTestStatistic(name,title,func,hdata,projDeps,cutRange,addCoefRange,nCPU,interleave,verbose,splitCutRange),
  _etype(etype), _funcMode(fmode)
{
  // Constructor of a chi2 for given p.d.f. with respect given binned
  // dataset taking the observables specified in projDeps as projected
  // observables. If cutRange is specified the calculation of the chi2
  // is restricted to that named range. If addCoefRange is specified,
  // the interpretation of fractions for all component RooAddPdfs that
  // do not have a frozen range interpretation is set to chosen range
  // name. If nCPU is greater than one the chi^2 calculation is
  // paralellized over the specified number of processors. If
  // interleave is true the partitioning of event over processors
  // follows a (i % n == i_set) strategy rather than a bulk
  // partitioning strategy which may result in unequal load balancing
  // in binned datasets with many (adjacent) zero bins. If
  // splitCutRange is true the cutRange is used to construct an
  // individual cutRange for each RooSimultaneous index category state
  // name cutRange_{indexStateName}.
}



//_____________________________________________________________________________
RooChi2Var::RooChi2Var(const RooChi2Var& other, const char* name) : 
  RooAbsOptTestStatistic(other,name),
  _etype(other._etype),
  _funcMode(other._funcMode)
{
  // Copy constructor
}



//_____________________________________________________________________________
RooChi2Var::~RooChi2Var()
{
  // Destructor
}



//_____________________________________________________________________________
Double_t RooChi2Var::evaluatePartition(Int_t firstEvent, Int_t lastEvent, Int_t stepSize) const 
{
  // Calculate chi^2 in partition from firstEvent to lastEvent using given stepSize

  // Throughout the calculation, we use Kahan's algorithm for summing to
  // prevent loss of precision - this is a factor four more expensive than
  // straight addition, but since evaluating the PDF is usually much more
  // expensive than that, we tolerate the additional cost...
  Int_t i ;
  Double_t result(0), carry(0);

  _dataClone->store()->recalculateCache( _projDeps, firstEvent, lastEvent, stepSize ) ;


  // Determine normalization factor depending on type of input function
  Double_t normFactor(1) ;
  switch (_funcMode) {
  case Function: normFactor=1 ; break ;
  case Pdf: normFactor = _dataClone->sumEntries() ; break ;
  case ExtendedPdf: normFactor = ((RooAbsPdf*)_funcClone)->expectedEvents(_dataClone->get()) ; break ;
  }

  // Loop over bins of dataset
  RooDataHist* hdata = (RooDataHist*) _dataClone ;
  for (i=firstEvent ; i<lastEvent ; i+=stepSize) {
    
    // get the data values for this event
    hdata->get(i);

    if (!hdata->valid()) continue;

    const Double_t nData = hdata->weight() ;

    const Double_t nPdf = _funcClone->getVal(_normSet) * normFactor * hdata->binVolume() ;

    const Double_t eExt = nPdf-nData ;


    Double_t eInt ;
    if (_etype != RooAbsData::Expected) {
      Double_t eIntLo,eIntHi ;
      hdata->weightError(eIntLo,eIntHi,_etype) ;
      eInt = (eExt>0) ? eIntHi : eIntLo ;
    } else {
      eInt = sqrt(nPdf) ;
    }
    
    // Skip cases where pdf=0 and there is no data
    if (0. == eInt * eInt && 0. == nData * nData && 0. == nPdf * nPdf) continue ;
    
    // Return 0 if eInt=0, special handling in MINUIT will follow
    if (0. == eInt * eInt) {
      coutE(Eval) << "RooChi2Var::RooChi2Var(" << GetName() << ") INFINITY ERROR: bin " << i 
		  << " has zero error" << endl ;
      return 0.;
    }
    
//     cout << "Chi2Var[" << i << "] nData = " << nData << " nPdf = " << nPdf << " errorExt = " << eExt << " errorInt = " << eInt << " contrib = " << eExt*eExt/(eInt*eInt) << endl ;
    
    Double_t term = eExt*eExt/(eInt*eInt) ;
    Double_t y = term - carry;
    Double_t t = result + y;
    carry = (t - result) - y;
    result = t;
  }
    
  return result ;
}



