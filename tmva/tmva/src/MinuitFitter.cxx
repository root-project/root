// @(#)root/tmva $Id$
// Author: Andraes Hoecker

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MinuitFitter                                                          *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation                                                            *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker  <Andreas.Hocker@cern.ch> - CERN, Switzerland             *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

/*! \class TMVA::MinuitFitter
\ingroup TMVA
/Fitter using MINUIT
*/
#include "TMVA/MinuitFitter.h"

#include "TMVA/Configurable.h"
#include "TMVA/FitterBase.h"
#include "TMVA/IFitterTarget.h"
#include "TMVA/Interval.h"
#include "TMVA/MinuitWrapper.h"
#include "TMVA/MsgLogger.h"
#include "TMVA/Timer.h"
#include "TMVA/Types.h"

#include "TFitter.h"

ClassImp(TMVA::MinuitFitter);

////////////////////////////////////////////////////////////////////////////////
/// constructor

TMVA::MinuitFitter::MinuitFitter( IFitterTarget& target,
                                  const TString& name,
                                  std::vector<TMVA::Interval*>& ranges,
                                  const TString& theOption )
: TMVA::FitterBase( target, name, ranges, theOption ),
   TMVA::IFitterTarget( )
{
   // default parameters settings for Simulated Annealing algorithm
   DeclareOptions();
   ParseOptions();

   Init();  // initialise the TFitter
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TMVA::MinuitFitter::~MinuitFitter( )
{
   delete fMinWrap;
}

////////////////////////////////////////////////////////////////////////////////
/// declare SA options

void TMVA::MinuitFitter::DeclareOptions()
{
   DeclareOptionRef(fErrorLevel    =  1,     "ErrorLevel",    "TMinuit: error level: 0.5=logL fit, 1=chi-squared fit" );
   DeclareOptionRef(fPrintLevel    = -1,     "PrintLevel",    "TMinuit: output level: -1=least, 0, +1=all garbage" );
   DeclareOptionRef(fFitStrategy   = 2,      "FitStrategy",   "TMinuit: fit strategy: 2=best" );
   DeclareOptionRef(fPrintWarnings = kFALSE, "PrintWarnings", "TMinuit: suppress warnings" );
   DeclareOptionRef(fUseImprove    = kTRUE,  "UseImprove",    "TMinuit: use IMPROVE" );
   DeclareOptionRef(fUseMinos      = kTRUE,  "UseMinos",      "TMinuit: use MINOS" );
   DeclareOptionRef(fBatch         = kFALSE, "SetBatch",      "TMinuit: use batch mode" );
   DeclareOptionRef(fMaxCalls      = 1000,   "MaxCalls",      "TMinuit: approximate maximum number of function calls" );
   DeclareOptionRef(fTolerance     = 0.1,    "Tolerance",     "TMinuit: tolerance to the function value at the minimum" );
}

////////////////////////////////////////////////////////////////////////////////
/// minuit-specific settings

void TMVA::MinuitFitter::Init()
{
   Double_t args[10];

   // Execute fitting
   if (!fBatch) Log() << kINFO << "<MinuitFitter> Init " << Endl;

   // timing of MC
   Timer timer;

   // initialize first -> prepare the fitter

   // instantiate minuit
   // maximum number of fit parameters is equal to
   // (2xnpar as workaround for TMinuit allocation bug (taken from RooMinuit))
   fMinWrap = new MinuitWrapper( fFitterTarget, 2*GetNpars() );

   // output level
   args[0] = fPrintLevel;
   fMinWrap->ExecuteCommand( "SET PRINTOUT", args, 1 );

   if (fBatch) fMinWrap->ExecuteCommand( "SET BAT", args, 0 );

   // set fitter object, and clear
   fMinWrap->Clear();

   // error level: 1 (2*log(L) fit
   args[0] = fErrorLevel;
   fMinWrap->ExecuteCommand( "SET ERR", args, 1 );

   // print warnings ?
   if (!fPrintWarnings) fMinWrap->ExecuteCommand( "SET NOWARNINGS", args, 0 );

   // define fit strategy
   args[0] = fFitStrategy;
   fMinWrap->ExecuteCommand( "SET STRATEGY", args, 1 );
}

////////////////////////////////////////////////////////////////////////////////
/// performs the fit

Double_t TMVA::MinuitFitter::Run( std::vector<Double_t>& pars )
{
   // minuit-specific settings
   Double_t args[10];

   // Execute fitting
   if ( !fBatch ) Log() << kINFO << "<MinuitFitter> Fitting, please be patient ... " << Endl;

   // sanity check
   if ((Int_t)pars.size() != GetNpars())
      Log() << kFATAL << "<Run> Mismatch in number of parameters: (a)"
            << GetNpars() << " != " << pars.size() << Endl;

   // timing of MC
   Timer* timer = 0;
   if (!fBatch) timer = new Timer();

   // define fit parameters
   for (Int_t ipar=0; ipar<fNpars; ipar++) {
      fMinWrap->SetParameter( ipar, Form( "Par%i",ipar ),
                              pars[ipar], fRanges[ipar]->GetWidth()/100.0,
                              fRanges[ipar]->GetMin(), fRanges[ipar]->GetMax() );
      if (fRanges[ipar]->GetWidth() == 0.0) fMinWrap->FixParameter( ipar );
   }

   // --------- execute the fit

   // continue with usual case
   args[0] = fMaxCalls;
   args[1] = fTolerance;

   // MIGRAD
   fMinWrap->ExecuteCommand( "MIGrad", args, 2 );

   // IMPROVE
   if (fUseImprove) fMinWrap->ExecuteCommand( "IMProve", args, 0 );

   // MINOS
   if (fUseMinos) {
      args[0] = 500;
      fMinWrap->ExecuteCommand( "MINOs", args, 1 );
   }

   // retrieve fit result (statistics)
   Double_t chi2;
   Double_t edm;
   Double_t errdef;
   Int_t    nvpar;
   Int_t    nparx;
   fMinWrap->GetStats( chi2, edm, errdef, nvpar, nparx );

   // sanity check
   if (GetNpars() != nparx) {
      Log() << kFATAL << "<Run> Mismatch in number of parameters: "
            << GetNpars() << " != " << nparx << Endl;
   }

   // retrieve parameters
   for (Int_t ipar=0; ipar<GetNpars(); ipar++) {
      Double_t errp, errm, errsym, globcor, currVal, currErr;
      fMinWrap->GetParameter( ipar, currVal, currErr );
      pars[ipar] = currVal;
      fMinWrap->GetErrors( ipar, errp, errm, errsym, globcor );
   }

   // clean up

   // get elapsed time
   if (!fBatch) {
      Log() << kINFO << "Elapsed time: " << timer->GetElapsedTime()
            << "                            " << Endl;
      delete timer;
   }

   fMinWrap->Clear();

   return chi2;
}

////////////////////////////////////////////////////////////////////////////////
/// performs the fit by calling Run(pars)

Double_t TMVA::MinuitFitter::EstimatorFunction( std::vector<Double_t>& pars )
{
   return Run( pars );
}


