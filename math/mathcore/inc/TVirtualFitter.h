// @(#)root/mathcore:$Id$
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


#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_TMethodCall
#include "TMethodCall.h"
#endif
#include "Foption.h"


class TVirtualFitter : public TNamed {

protected:
   Foption_t              fOption;     //struct with the fit options
   Int_t                  fXfirst;     //first bin on X axis
   Int_t                  fXlast;      //last  bin on X axis
   Int_t                  fYfirst;     //first bin on Y axis
   Int_t                  fYlast;      //last  bin on Y axis
   Int_t                  fZfirst;     //first bin on Z axis
   Int_t                  fZlast;      //last  bin on Z axis
   Int_t                  fNpoints;    //Number of points to fit
   Int_t                  fPointSize;  //Number of words per point in the cache
   Int_t                  fCacheSize;  //Size of the fCache array
   Double_t              *fCache;      //[fCacheSize] array of points data (fNpoints*fPointSize < fCacheSize words)
   TObject               *fObjectFit;  //pointer to object being fitted
   TObject               *fUserFunc;   //pointer to user theoretical function (a TF1*)
   TMethodCall           *fMethodCall; //Pointer to MethodCall in case of interpreted function
   void                 (*fFCN)(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);

   static TVirtualFitter *fgFitter;    //Current fitter (default TFitter)
   static Int_t           fgMaxpar;    //Maximum number of fit parameters for current fitter
   static Int_t           fgMaxiter;   //Maximum number of iterations
   static Double_t        fgErrorDef;  //Error definition (default=1)
   static Double_t        fgPrecision; //maximum precision
   static TString         fgDefault;   //name of the default fitter ("Minuit","Fumili",etc)

   TVirtualFitter(const TVirtualFitter& tvf);
   TVirtualFitter& operator=(const TVirtualFitter& tvf);

public:
   TVirtualFitter();
   virtual ~TVirtualFitter();
   virtual Double_t  Chisquare(Int_t npar, Double_t *params) const  = 0;

   virtual void      Clear(Option_t *option="") = 0;
   virtual Int_t     ExecuteCommand(const char *command, Double_t *args, Int_t nargs) = 0;
   virtual void      FixParameter(Int_t ipar) = 0;
   virtual void      GetConfidenceIntervals(Int_t n, Int_t ndim, const Double_t *x, Double_t *ci, Double_t cl=0.95);
   virtual void      GetConfidenceIntervals(TObject *obj, Double_t cl=0.95);
   virtual Double_t *GetCovarianceMatrix() const = 0;
   virtual Double_t  GetCovarianceMatrixElement(Int_t i, Int_t j) const = 0;
   virtual Int_t     GetErrors(Int_t ipar,Double_t &eplus, Double_t &eminus, Double_t &eparab, Double_t &globcc) const = 0;
   typedef void   (* FCNFunc_t )(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);
   virtual FCNFunc_t GetFCN() { return fFCN; }
   virtual Foption_t GetFitOption() const {return fOption;}
   TMethodCall      *GetMethodCall() const {return fMethodCall;}
   virtual Int_t     GetNumberTotalParameters() const = 0;
   virtual Int_t     GetNumberFreeParameters() const = 0;
   virtual TObject  *GetObjectFit() const {return fObjectFit;}
   virtual Double_t  GetParError(Int_t ipar) const = 0;
   virtual Double_t  GetParameter(Int_t ipar) const = 0;
   virtual Int_t     GetParameter(Int_t ipar,char *name,Double_t &value,Double_t &verr,Double_t &vlow, Double_t &vhigh) const  = 0;
   virtual const char *GetParName(Int_t ipar) const = 0;
   virtual Int_t     GetStats(Double_t &amin, Double_t &edm, Double_t &errdef, Int_t &nvpar, Int_t &nparx) const  = 0;
   virtual Double_t  GetSumLog(Int_t i) = 0;
   virtual TObject  *GetUserFunc() const {return fUserFunc;}
   virtual Int_t     GetXfirst() const {return fXfirst;}
   virtual Int_t     GetXlast()  const {return fXlast;}
   virtual Int_t     GetYfirst() const {return fYfirst;}
   virtual Int_t     GetYlast()  const {return fYlast;}
   virtual Int_t     GetZfirst() const {return fZfirst;}
   virtual Int_t     GetZlast()  const {return fZlast;}
   virtual Bool_t    IsFixed(Int_t ipar) const = 0;
   virtual void      PrintResults(Int_t level, Double_t amin) const = 0;
   virtual void      ReleaseParameter(Int_t ipar) = 0;
   virtual Double_t *SetCache(Int_t npoints, Int_t psize);
   virtual void      SetFCN(void *fcn);
   virtual void      SetFCN(void (*fcn)(Int_t &, Double_t *, Double_t &f, Double_t *, Int_t));
   virtual void      SetFitMethod(const char *name) = 0;
   virtual void      SetFitOption(Foption_t option) {fOption = option;}
   virtual void      SetObjectFit(TObject *obj) {fObjectFit = obj;}
   virtual Int_t     SetParameter(Int_t ipar,const char *parname,Double_t value,Double_t verr,Double_t vlow, Double_t vhigh) = 0;
   virtual void      SetUserFunc(TObject *userfunc) {fUserFunc = userfunc;}
   virtual void      SetXfirst(Int_t first) {fXfirst = first;}
   virtual void      SetXlast (Int_t last)  {fXlast  = last;}
   virtual void      SetYfirst(Int_t first) {fYfirst = first;}
   virtual void      SetYlast (Int_t last)  {fYlast  = last;}
   virtual void      SetZfirst(Int_t first) {fZfirst = first;}
   virtual void      SetZlast (Int_t last)  {fZlast  = last;}

   static  TVirtualFitter *GetFitter();
   static  TVirtualFitter *Fitter(TObject *obj, Int_t maxpar = 25);
   static const char *GetDefaultFitter();
   static Int_t     GetMaxIterations();
   static Double_t  GetErrorDef();
   static Double_t  GetPrecision();
   static void      SetDefaultFitter(const char* name = "");
   static void      SetFitter(TVirtualFitter *fitter, Int_t maxpar = 25);
   static void      SetMaxIterations(Int_t niter=5000);
   static void      SetErrorDef(Double_t errdef=1);
   static void      SetPrecision(Double_t prec=1e-6);

   ClassDef(TVirtualFitter,0)  //Abstract interface for fitting
};

#endif
