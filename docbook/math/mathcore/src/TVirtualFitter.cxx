// @(#)root/mathcore:$Id$
// Author: Rene Brun   31/08/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


//______________________________________________________________________________
//
//     Abstract Base Class for Fitting

#include "TROOT.h"
#include "TVirtualFitter.h"
#include "TPluginManager.h"
#include "TEnv.h"
#include "TInterpreter.h"
#include "Math/MinimizerOptions.h"


TVirtualFitter *TVirtualFitter::fgFitter    = 0;
Int_t           TVirtualFitter::fgMaxpar    = 0;
// Int_t           TVirtualFitter::fgMaxiter   = 5000;
// Double_t        TVirtualFitter::fgPrecision = 1e-6;
// Double_t        TVirtualFitter::fgErrorDef  = 1;
TString         TVirtualFitter::fgDefault   = "";

ClassImp(TVirtualFitter)

#ifdef R__COMPLETE_MEM_TERMINATION
namespace {
   struct TVirtualFitterCleanup {
      ~TVirtualFitterCleanup() {
         delete TVirtualFitter::GetFitter();
      }
   };
   TVirtualFitterCleanup cleanup;
}
#endif

//______________________________________________________________________________
TVirtualFitter::TVirtualFitter() : 
   fXfirst(0), 
   fXlast(0), 
   fYfirst(0), 
   fYlast(0), 
   fZfirst(0), 
   fZlast(0), 
   fNpoints(0),
   fPointSize(0),
   fCacheSize(0),
   fCache(0),
   fObjectFit(0),
   fUserFunc(0),
   fMethodCall(0),
   fFCN(0)
{
   // Default constructor.
}

//______________________________________________________________________________
TVirtualFitter::TVirtualFitter(const TVirtualFitter& tvf) :
  TNamed(tvf),
  fOption(tvf.fOption),
  fXfirst(tvf.fXfirst),
  fXlast(tvf.fXlast),
  fYfirst(tvf.fYfirst),
  fYlast(tvf.fYlast),
  fZfirst(tvf.fZfirst),
  fZlast(tvf.fZlast),
  fNpoints(tvf.fNpoints),
  fPointSize(tvf.fPointSize),
  fCacheSize(tvf.fCacheSize),
  fCache(tvf.fCache),
  fObjectFit(tvf.fObjectFit),
  fUserFunc(tvf.fUserFunc),
  fMethodCall(tvf.fMethodCall),
  fFCN(tvf.fFCN)
{
   //copy constructor
}

//______________________________________________________________________________
TVirtualFitter& TVirtualFitter::operator=(const TVirtualFitter& tvf)
{
   //assignment operator
   if(this!=&tvf) {
      TNamed::operator=(tvf);
      fOption=tvf.fOption;
      fXfirst=tvf.fXfirst;
      fXlast=tvf.fXlast;
      fYfirst=tvf.fYfirst;
      fYlast=tvf.fYlast;
      fZfirst=tvf.fZfirst;
      fZlast=tvf.fZlast;
      fNpoints=tvf.fNpoints;
      fPointSize=tvf.fPointSize;
      fCacheSize=tvf.fCacheSize;
      fCache=tvf.fCache;
      fObjectFit=tvf.fObjectFit;
      fUserFunc=tvf.fUserFunc;
      fMethodCall=tvf.fMethodCall;
      fFCN=tvf.fFCN;
   }
   return *this;
}

//______________________________________________________________________________
TVirtualFitter::~TVirtualFitter()
{
   // Cleanup virtual fitter.

   delete fMethodCall;
   delete [] fCache;
   if ( fgFitter == this ) { 
      fgFitter    = 0;
      fgMaxpar    = 0;
   }
   fMethodCall = 0;
   fFCN        = 0;
}

//______________________________________________________________________________
TVirtualFitter *TVirtualFitter::Fitter(TObject *obj, Int_t maxpar)
{
   // Static function returning a pointer to the current fitter.
   // If the fitter does not exist, the default TFitter is created.
   // Don't delete the returned fitter object, it will be re-used.

   if (fgFitter && maxpar > fgMaxpar) {
      delete fgFitter;
      fgFitter = 0;
   }

   if (!fgFitter) {
      TPluginHandler *h;
      if (fgDefault.Length() == 0) fgDefault = gEnv->GetValue("Root.Fitter","Minuit");
      if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualFitter",fgDefault))) {
         if (h->LoadPlugin() == -1)
            return 0;
         fgFitter = (TVirtualFitter*) h->ExecPlugin(1, maxpar);
         fgMaxpar = maxpar;
      }
   }

   if (fgFitter) fgFitter->SetObjectFit(obj);
   return fgFitter;
}

//______________________________________________________________________________
void  TVirtualFitter::GetConfidenceIntervals(Int_t /*n*/, Int_t /*ndim*/, const Double_t * /*x*/, Double_t * /*ci*/, Double_t /*cl*/)
{
   //return confidence intervals in array x of dimension ndim
   //implemented in TFitter and TLinearFitter
}

//______________________________________________________________________________
void  TVirtualFitter::GetConfidenceIntervals(TObject * /*obj*/, Double_t /*cl*/)
{
   //return confidence intervals in TObject obj
   //implemented in TFitter and TLinearFitter
}

//______________________________________________________________________________
const char *TVirtualFitter::GetDefaultFitter()
{
   // static: return the name of the default fitter

   //return fgDefault.Data();
   return ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str();
}

//______________________________________________________________________________
TVirtualFitter *TVirtualFitter::GetFitter()
{
   // static: return the current Fitter
   return fgFitter;
}

//______________________________________________________________________________
Int_t TVirtualFitter::GetMaxIterations()
{
   // static: Return the maximum number of iterations
   // actually max number of function calls

   //return fgMaxiter;
   return ROOT::Math::MinimizerOptions::DefaultMaxFunctionCalls();
}

//______________________________________________________________________________
Double_t TVirtualFitter::GetErrorDef()
{
   // static: Return the Error Definition

//   return fgErrorDef;
   return ROOT::Math::MinimizerOptions::DefaultErrorDef();
}

//______________________________________________________________________________
Double_t TVirtualFitter::GetPrecision()
{
   // static: Return the fit relative precision

   //return fgPrecision;
   return ROOT::Math::MinimizerOptions::DefaultTolerance();
}

//______________________________________________________________________________
void TVirtualFitter::SetDefaultFitter(const char *name)
{
   // static: set name of default fitter

   ROOT::Math::MinimizerOptions::SetDefaultMinimizer(name,"");
   if (fgDefault == name) return;
   delete fgFitter;
   fgFitter = 0;
   fgDefault = name;
}

//______________________________________________________________________________
void TVirtualFitter::SetFitter(TVirtualFitter *fitter, Int_t maxpar)
{
   // Static function to set an alternative fitter

   fgFitter = fitter;
   fgMaxpar = maxpar;
}

//______________________________________________________________________________
void TVirtualFitter::SetFCN(void (*fcn)(Int_t &, Double_t *, Double_t &f, Double_t *, Int_t))
{
   // To set the address of the minimization objective function
   // called by the native compiler (see function below when called by CINT)

   fFCN = fcn;
}

//______________________________________________________________________________
void InteractiveFCN(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag)
{
   // Static function called when SetFCN is called in interactive mode

   TMethodCall *m = TVirtualFitter::GetFitter()->GetMethodCall();
   if (!m) return;

   Long_t args[5];
   args[0] = (Long_t)&npar;
   args[1] = (Long_t)gin;
   args[2] = (Long_t)&f;
   args[3] = (Long_t)u;
   args[4] = (Long_t)flag;
   m->SetParamPtrs(args);
   Double_t result;
   m->Execute(result);
}

//______________________________________________________________________________
Double_t *TVirtualFitter::SetCache(Int_t npoints, Int_t psize)
{
   // Initialize the cache array
   // npoints is the number of points to be stored (or already stored) in the cache
   // psize is the number of elements per point
   //
   // if (npoints*psize > fCacheSize) the existing cache is deleted
   // and a new array is created.
   // The function returns a pointer to the cache

   if (npoints*psize > fCacheSize) {
      delete [] fCache;
      fCacheSize = npoints*psize;
      fCache = new Double_t[fCacheSize];
   }
   fNpoints = npoints;
   fPointSize = psize;
   return fCache;
}

//______________________________________________________________________________
void TVirtualFitter::SetFCN(void *fcn)
{
   //  To set the address of the minimization objective function
   //
   //     this function is called by CINT instead of the function above

   if (!fcn) return;

   const char *funcname = gCint->Getp2f2funcname(fcn);
   if (funcname) {
      delete fMethodCall;
      fMethodCall = new TMethodCall();
      fMethodCall->InitWithPrototype(funcname,"Int_t&,Double_t*,Double_t&,Double_t*,Int_t");
   }
   fFCN = InteractiveFCN;
}

//______________________________________________________________________________
void TVirtualFitter::SetMaxIterations(Int_t niter)
{
   // static: Set the maximum number of function calls for the minimization algorithm
   // For example for MIGRAD this is the maxcalls value passed as first argument
   // (see http://wwwasdoc.web.cern.ch/wwwasdoc/minuit/node18.html ) 

   ROOT::Math::MinimizerOptions::SetDefaultMaxFunctionCalls(niter);
}

//______________________________________________________________________________
void TVirtualFitter::SetErrorDef(Double_t errdef)
{
   // static: Set the Error Definition (default=1)
   // For Minuit this is the value passed with the "SET ERR" command
   // (see http://wwwasdoc.web.cern.ch/wwwasdoc/minuit/node18.html)

//    fgErrorDef = errdef;
   ROOT::Math::MinimizerOptions::SetDefaultErrorDef(errdef);
   if (!fgFitter) return;
   Double_t arglist[1];
   arglist[0] = errdef;
   fgFitter->ExecuteCommand("SET ERRORDEF", arglist, 1);
}

//______________________________________________________________________________
void TVirtualFitter::SetPrecision(Double_t prec)
{
   // static: Set the tolerance used in the minimization algorithm
   // For example for MIGRAD this is tolerance value passed as second argument
   // (see http://wwwasdoc.web.cern.ch/wwwasdoc/minuit/node18.html ) 
  
   //fgPrecision = prec;
   ROOT::Math::MinimizerOptions::SetDefaultTolerance(prec);
}
