// @(#)root/tmva $Id$ 
// Author: Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MinuitWrapper                                                         *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation                                                            *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Peter Speckmayer <peter.speckmayer@cern.ch> - CERN, Switzerland           *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
//                                                                      
// Wrapper around MINUIT
//_______________________________________________________________________

#include "TMVA/MinuitWrapper.h"

ClassImp(TMVA::MinuitWrapper)

//_______________________________________________________________________
TMVA::MinuitWrapper::MinuitWrapper( IFitterTarget& target, Int_t maxpar )
   : TMinuit( maxpar ),
     fFitterTarget( target ),
     fNumPar( maxpar )
{
   // constructor
   for ( Int_t i=0; i< maxpar; i++ ) {
      fParameters.push_back(0.0);
   }
}

//_______________________________________________________________________
Int_t TMVA::MinuitWrapper::Eval(Int_t /*npar*/, Double_t*, Double_t& f, Double_t* par, Int_t)
{
   // std::vector<Double_t> parameters( npar );
   for (Int_t ipar=0; ipar<fNumPar; ipar++) fParameters[ipar] = par[ipar];
   
   f = fFitterTarget.EstimatorFunction( fParameters );
   return 0;
}

//______________________________________________________________________________
Int_t TMVA::MinuitWrapper::ExecuteCommand(const char *command, Double_t *args, Int_t nargs)
{
   // Execute a fitter command;
   //   command : command string
   //   args    : list of nargs command arguments

   Int_t ierr = 0;
   mnexcm(command,args,nargs,ierr);
   return ierr;
}

//______________________________________________________________________________
void TMVA::MinuitWrapper::Clear(Option_t *)
{
   // reset the fitter environment

   // reset the internal Minuit random generator to its initial state
   Double_t val = 3;
   Int_t inseed = 12345;
   mnrn15(val,inseed);
}

//______________________________________________________________________________
Int_t TMVA::MinuitWrapper::GetStats(Double_t &amin, Double_t &edm, Double_t &errdef, Int_t &nvpar, Int_t &nparx) 
{
   // return global fit parameters
   //   amin     : chisquare
   //   edm      : estimated distance to minimum
   //   errdef
   //   nvpar    : number of variable parameters
   //   nparx    : total number of parameters

   Int_t ierr = 0;
   mnstat(amin,edm,errdef,nvpar,nparx,ierr);
   return ierr;
}

//______________________________________________________________________________
Int_t TMVA::MinuitWrapper::GetErrors(Int_t ipar,Double_t &eplus, Double_t &eminus, Double_t &eparab, Double_t &globcc)
{
   // return current errors for a parameter
   //   ipar     : parameter number
   //   eplus    : upper error
   //   eminus   : lower error
   //   eparab   : parabolic error
   //   globcc   : global correlation coefficient

   Int_t ierr = 0;
   mnerrs(ipar, eplus,eminus,eparab,globcc);
   return ierr;
}

//______________________________________________________________________________
Int_t TMVA::MinuitWrapper::SetParameter(Int_t ipar,const char *parname,Double_t value,Double_t verr,Double_t vlow, Double_t vhigh)
{
   // set initial values for a parameter
   //   ipar     : parameter number
   //   parname  : parameter name
   //   value    : initial parameter value
   //   verr     : initial error for this parameter
   //   vlow     : lower value for the parameter
   //   vhigh    : upper value for the parameter

   //   if (fCovar)  {delete [] fCovar; fCovar = 0;}
   Int_t ierr = 0;
   mnparm(ipar,parname,value,verr,vlow,vhigh,ierr);
   return ierr;
}

//______________________________________________________________________________
TObject *TMVA::MinuitWrapper::Clone(char const* newname) const
{
   // produces a clone of this MinuitWrapper
   MinuitWrapper *named = (MinuitWrapper*)TNamed::Clone(newname);
   named->fFitterTarget = fFitterTarget;
   return 0;
}
