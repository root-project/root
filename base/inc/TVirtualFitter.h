// @(#)root/base:$Name$:$Id$
// Author: Rene Brun   31/08/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TVirtualFitter
#define ROOT_TVirtualFitter


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualFitter                                                       //
//                                                                      //
// Abstract base class for fitting                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TObject
#include "TObject.h"
#endif


class TVirtualFitter : public TObject {


private:
   static TVirtualFitter *fgFitter;    //Current fitter (default TFitter)
   static Int_t           fgMaxpar;    //Maximum number of fit parameters for current fitter
   static Int_t           fgMaxiter;   //Maximum number of iterations
   static Double_t        fgPrecision; //maximum precision

public:
   TVirtualFitter();
   virtual ~TVirtualFitter();
   virtual void     Clear(Option_t *option="") = 0;
   virtual Int_t    ExecuteCommand(const char *command, Double_t *args, Int_t nargs) = 0;
   virtual void     FixParameter(Int_t ipar) = 0;
   virtual Int_t    GetErrors(Int_t ipar,Double_t &eplus, Double_t &eminus, Double_t &eparab, Double_t &globcc) = 0;
   virtual TObject *GetObjectFit() = 0;
   virtual Int_t    GetParameter(Int_t ipar,char *name,Double_t &value,Double_t &verr,Double_t &vlow, Double_t &vhigh) = 0;
   virtual Int_t    GetStats(Double_t &amin, Double_t &edm, Double_t &errdef, Int_t &nvpar, Int_t &nparx) = 0;
   virtual Double_t GetSumLog(Int_t i) = 0;
   virtual void     PrintResults(Int_t level, Double_t amin) = 0;
   virtual void     ReleaseParameter(Int_t ipar) = 0;
   virtual void     SetFCN(void *fcn) = 0;
   virtual void     SetFCN(void (*fcn)(Int_t &, Double_t *, Double_t &f, Double_t *, Int_t)) = 0;
   virtual void     SetObjectFit(TObject *obj) = 0;
   virtual Int_t    SetParameter(Int_t ipar,const char *parname,Double_t value,Double_t verr,Double_t vlow, Double_t vhigh) = 0;

   static  TVirtualFitter *Fitter(TObject *obj, Int_t maxpar = 25);
   static Int_t     GetMaxIterations();
   static Double_t  GetPrecision();
   static void      SetFitter(TVirtualFitter *fitter, Int_t maxpar = 25);
   static void      SetMaxIterations(Int_t niter=5000);
   static void      SetPrecision(Double_t prec=1e-6);

    ClassDef(TVirtualFitter,0)  //Abstract interface for fitting
};

#endif
