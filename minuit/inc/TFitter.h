// @(#)root/minuit:$Name$:$Id$
// Author: Rene Brun   31/08/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TFitter
#define ROOT_TFitter


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFitter                                                              //
//                                                                      //
// The ROOT standard fitter based on TMinuit                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TVirtualFitter
#include "TVirtualFitter.h"
#endif


class TFitter : public TVirtualFitter {

private:
   Int_t      fNlog;       //Number of elements in fSunLog
   Double_t  *fSumLog;     //Sum of logs (array of fNlog elements)

   public:
   TFitter(Int_t maxpar = 25);
   virtual ~TFitter();
   virtual void     Clear(Option_t *option="");
   virtual Int_t    ExecuteCommand(const char *command, Double_t *args, Int_t nargs);
   virtual void     FixParameter(Int_t ipar);
   virtual Int_t    GetErrors(Int_t ipar,Double_t &eplus, Double_t &eminus, Double_t &eparab, Double_t &globcc);
   virtual TObject *GetObjectFit();
   virtual Int_t    GetParameter(Int_t ipar,char *name,Double_t &value,Double_t &verr,Double_t &vlow, Double_t &vhigh);
   virtual Int_t    GetStats(Double_t &amin, Double_t &edm, Double_t &errdef, Int_t &nvpar, Int_t &nparx);
   virtual Double_t GetSumLog(Int_t i);
   virtual void     PrintResults(Int_t level, Double_t amin);
   virtual void     ReleaseParameter(Int_t ipar);
   virtual void     SetFCN(void *fcn);
   virtual void     SetFCN(void (*fcn)(Int_t &, Double_t *, Double_t &f, Double_t *, Int_t));
   virtual void     SetObjectFit(TObject *obj);
   virtual Int_t    SetParameter(Int_t ipar,const char *parname,Double_t value,Double_t verr,Double_t vlow, Double_t vhigh);

    ClassDef(TFitter,0)  //The ROOT standard fitter based on TMinuit
};

#endif
