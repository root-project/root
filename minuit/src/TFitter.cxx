// @(#)root/minuit:$Name:  $:$Id: TFitter.cxx,v 1.1.1.1 2000/05/16 17:00:44 rdm Exp $
// Author: Rene Brun   31/08/99
/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TMinuit.h"
#include "TFitter.h"

ClassImp(TFitter)

//______________________________________________________________________________
TFitter::TFitter(Int_t maxpar)
{
//*-*-*-*-*-*-*-*-*-*-*default constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===================

   gMinuit = new TMinuit(maxpar);
   TVirtualFitter::SetFitter(this);
   fNlog = 0;
   fSumLog = 0;
}

//______________________________________________________________________________
TFitter::~TFitter()
{
//*-*-*-*-*-*-*-*-*-*-*default destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ==================

   if (fSumLog) delete [] fSumLog;
}

//______________________________________________________________________________
void TFitter::Clear(Option_t *)
{
   // reset the fitter environment

   gMinuit->mncler();

}

//______________________________________________________________________________
Int_t TFitter::ExecuteCommand(const char *command, Double_t *args, Int_t nargs)
{
   // Execute a fitter command;
   //   command : command string
   //   args    : list of nargs command arguments

   Int_t ierr = 0;
   gMinuit->mnexcm(command,args,nargs,ierr);
   return ierr;
}

//______________________________________________________________________________
void TFitter::FixParameter(Int_t ipar)
{
   // Fix parameter ipar.

   gMinuit->FixParameter(ipar);
}

//______________________________________________________________________________
Int_t TFitter::GetErrors(Int_t ipar,Double_t &eplus, Double_t &eminus, Double_t &eparab, Double_t &globcc)
{
   // return current errors for a parameter
   //   ipar     : parameter number
   //   eplus    : upper error
   //   eminus   : lower error
   //   eparab   : parabolic error
   //   globcc   : global correlation coefficient


   Int_t ierr = 0;
   gMinuit->mnerrs(ipar, eplus,eminus,eparab,globcc);
   return ierr;
}

//______________________________________________________________________________
TObject *TFitter::GetObjectFit() const
{
   // return a pointer to the current object being fitted

   return gMinuit->GetObjectFit();
}

//______________________________________________________________________________
Int_t TFitter::GetParameter(Int_t ipar,char *parname,Double_t &value,Double_t &verr,Double_t &vlow, Double_t &vhigh)
{
   // return current values for a parameter
   //   ipar     : parameter number
   //   parname  : parameter name
   //   value    : initial parameter value
   //   verr     : initial error for this parameter
   //   vlow     : lower value for the parameter
   //   vhigh    : upper value for the parameter

   Int_t ierr = 0;
   TString pname;
   gMinuit->mnpout(ipar, pname,value,verr,vlow,vhigh,ierr);
   strcpy(parname,pname.Data());
   return ierr;
}

//______________________________________________________________________________
Int_t TFitter::GetStats(Double_t &amin, Double_t &edm, Double_t &errdef, Int_t &nvpar, Int_t &nparx)
{
   // return global fit parameters
   //   amin
   //   edm      : estimated distance to minimum
   //   errdef
   //   nvpar    : number of variable parameters
   //   nparx    : total number of parameters

   Int_t ierr = 0;
   gMinuit->mnstat(amin,edm,errdef,nvpar,nparx,ierr);
   return ierr;
}

//______________________________________________________________________________
Double_t TFitter::GetSumLog(Int_t n)
{
   // return Sum(log(i) i=0,n
   // used by log likelihood fits

   if (n < 0) return 0;
   if (n > fNlog) {
      if (fSumLog) delete [] fSumLog;
      fNlog = 2*n+1000;
      fSumLog = new Double_t[fNlog+1];
      Double_t fobs = 0;
      for (Int_t j=0;j<=n;j++) {
         if (j > 1) fobs += TMath::Log(j);
         fSumLog[j] = fobs;
      }
   }
   if (fSumLog) return fSumLog[n];
   return 0;
}

//______________________________________________________________________________
void  TFitter::PrintResults(Int_t level, Double_t amin) const
{
   // Print fit results

   gMinuit->mnprin(level,amin);
}

//______________________________________________________________________________
void TFitter::ReleaseParameter(Int_t ipar)
{
   // Release parameter ipar.

   gMinuit->Release(ipar);
}

//______________________________________________________________________________
void TFitter::SetFCN(void *fcn)
{
   // Specify the address of the fitting algorithm (from the interpreter)

   gMinuit->SetFCN(fcn);
}

//______________________________________________________________________________
void TFitter::SetFCN(void (*fcn)(Int_t &, Double_t *, Double_t &f, Double_t *, Int_t))
{
   // Specify the address of the fitting algorithm

   gMinuit->SetFCN(fcn);
}

//______________________________________________________________________________
void TFitter::SetObjectFit(TObject *obj)
{
   // Specify the current object to be fitted

   gMinuit->SetObjectFit(obj);
}

//______________________________________________________________________________
Int_t TFitter::SetParameter(Int_t ipar,const char *parname,Double_t value,Double_t verr,Double_t vlow, Double_t vhigh)
{
   // set initial values for a parameter
   //   ipar     : parameter number
   //   parname  : parameter name
   //   value    : initial parameter value
   //   verr     : initial error for this parameter
   //   vlow     : lower value for the parameter
   //   vhigh    : upper value for the parameter

   Int_t ierr = 0;
   gMinuit->mnparm(ipar,parname,value,verr,vlow,vhigh,ierr);
   return ierr;
}
