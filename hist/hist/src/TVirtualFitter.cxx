// @(#)root/hist:$Id$
// Author: Rene Brun   31/08/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


/** \class TVirtualFitter
    \ingroup Hist
  Abstract Base Class for Fitting
*/

#include "TROOT.h"
#include "TVirtualFitter.h"
#include "TPluginManager.h"
#include "TEnv.h"
#include "Math/MinimizerOptions.h"
#include "ThreadLocalStorage.h"


// Implement a thread local static member as a replacement
// for TVirtualFitter::fgFitter
namespace {
   struct FitterGlobals {
      FitterGlobals() : fFitter(nullptr),fMaxPar(0) {}

      TVirtualFitter *fFitter;
      Int_t           fMaxPar;
      TString         fDefault;
   };
   static FitterGlobals &GetGlobals() {
      TTHREAD_TLS_DECL(FitterGlobals,globals);
      return globals;
   }
   static TVirtualFitter *&GetGlobalFitter() {
      return GetGlobals().fFitter;
   }
   static Int_t &GetGlobalMaxPar() {
      return GetGlobals().fMaxPar;
   }
   static TString &GetGlobalDefault() {
      return GetGlobals().fDefault;
   }
}
//Int_t           TVirtualFitter::fgMaxpar    = 0;
// Int_t           TVirtualFitter::fgMaxiter   = 5000;
// Double_t        TVirtualFitter::fgPrecision = 1e-6;
// Double_t        TVirtualFitter::fgErrorDef  = 1;
//TString         TVirtualFitter::fgDefault   = "";

ClassImp(TVirtualFitter);

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

////////////////////////////////////////////////////////////////////////////////
/// Default constructor.

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
}

////////////////////////////////////////////////////////////////////////////////
///copy constructor

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
}

////////////////////////////////////////////////////////////////////////////////
///assignment operator

TVirtualFitter& TVirtualFitter::operator=(const TVirtualFitter& tvf)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Cleanup virtual fitter.

TVirtualFitter::~TVirtualFitter()
{
   delete fMethodCall;
   delete [] fCache;
   if ( GetGlobalFitter() == this ) {
      GetGlobalFitter()    = 0;
      GetGlobalMaxPar()    = 0;
   }
   fMethodCall = 0;
   fFCN        = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Static function returning a pointer to the current fitter.
/// If the fitter does not exist, the default TFitter is created.
/// Don't delete the returned fitter object, it will be re-used.

TVirtualFitter *TVirtualFitter::Fitter(TObject *obj, Int_t maxpar)
{
   if (GetGlobalFitter() && maxpar > GetGlobalMaxPar()) {
      delete GetGlobalFitter();
      GetGlobalFitter() = 0;
   }

   if (!GetGlobalFitter()) {
      TPluginHandler *h;
      if (GetGlobalDefault().Length() == 0) GetGlobalDefault() = gEnv->GetValue("Root.Fitter","Minuit");
      if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualFitter",GetGlobalDefault()))) {
         if (h->LoadPlugin() == -1)
            return 0;
         GetGlobalFitter() = (TVirtualFitter*) h->ExecPlugin(1, maxpar);
         GetGlobalMaxPar() = maxpar;
      }
   }

   if (GetGlobalFitter()) GetGlobalFitter()->SetObjectFit(obj);
   return GetGlobalFitter();
}

////////////////////////////////////////////////////////////////////////////////
///return confidence intervals in array x of dimension ndim
///implemented in TFitter and TLinearFitter

void  TVirtualFitter::GetConfidenceIntervals(Int_t /*n*/, Int_t /*ndim*/, const Double_t * /*x*/, Double_t * /*ci*/, Double_t /*cl*/)
{
}

////////////////////////////////////////////////////////////////////////////////
///return confidence intervals in TObject obj
///implemented in TFitter and TLinearFitter

void  TVirtualFitter::GetConfidenceIntervals(TObject * /*obj*/, Double_t /*cl*/)
{
}

////////////////////////////////////////////////////////////////////////////////
/// static: return the name of the default fitter

const char *TVirtualFitter::GetDefaultFitter()
{
   //return GetGlobalDefault().Data();
   return ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str();
}

////////////////////////////////////////////////////////////////////////////////
/// static: return the current Fitter

TVirtualFitter *TVirtualFitter::GetFitter()
{
   return GetGlobalFitter();
}

////////////////////////////////////////////////////////////////////////////////
/// static: Return the maximum number of iterations
/// actually max number of function calls

Int_t TVirtualFitter::GetMaxIterations()
{
   //return fgMaxiter;
   return ROOT::Math::MinimizerOptions::DefaultMaxFunctionCalls();
}

////////////////////////////////////////////////////////////////////////////////
/// static: Return the Error Definition

Double_t TVirtualFitter::GetErrorDef()
{
//   return fgErrorDef;
   return ROOT::Math::MinimizerOptions::DefaultErrorDef();
}

////////////////////////////////////////////////////////////////////////////////
/// static: Return the fit relative precision

Double_t TVirtualFitter::GetPrecision()
{
   //return fgPrecision;
   return ROOT::Math::MinimizerOptions::DefaultTolerance();
}

////////////////////////////////////////////////////////////////////////////////
/// static: set name of default fitter

void TVirtualFitter::SetDefaultFitter(const char *name)
{
   ROOT::Math::MinimizerOptions::SetDefaultMinimizer(name,"");
   if (GetGlobalDefault() == name) return;
   delete GetGlobalFitter();
   GetGlobalFitter() = 0;
   GetGlobalDefault() = name;
}

////////////////////////////////////////////////////////////////////////////////
/// Static function to set an alternative fitter

void TVirtualFitter::SetFitter(TVirtualFitter *fitter, Int_t maxpar)
{
   GetGlobalFitter() = fitter;
   GetGlobalMaxPar() = maxpar;
}

////////////////////////////////////////////////////////////////////////////////
/// To set the address of the minimization objective function
/// called by the native compiler (see function below when called by CINT)

void TVirtualFitter::SetFCN(void (*fcn)(Int_t &, Double_t *, Double_t &f, Double_t *, Int_t))
{
   fFCN = fcn;
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize the cache array
/// npoints is the number of points to be stored (or already stored) in the cache
/// psize is the number of elements per point
///
/// if (npoints*psize > fCacheSize) the existing cache is deleted
/// and a new array is created.
/// The function returns a pointer to the cache

Double_t *TVirtualFitter::SetCache(Int_t npoints, Int_t psize)
{
   if (npoints*psize > fCacheSize) {
      delete [] fCache;
      fCacheSize = npoints*psize;
      fCache = new Double_t[fCacheSize];
   }
   fNpoints = npoints;
   fPointSize = psize;
   return fCache;
}

////////////////////////////////////////////////////////////////////////////////
/// static: Set the maximum number of function calls for the minimization algorithm
/// For example for MIGRAD this is the maxcalls value passed as first argument
/// (see https://cern-tex.web.cern.ch/cern-tex/minuit/node18.html )

void TVirtualFitter::SetMaxIterations(Int_t niter)
{
   ROOT::Math::MinimizerOptions::SetDefaultMaxFunctionCalls(niter);
}

////////////////////////////////////////////////////////////////////////////////
/// static: Set the Error Definition (default=1)
/// For Minuit this is the value passed with the "SET ERR" command
/// (see https://cern-tex.web.cern.ch/cern-tex/minuit/node18.html)

void TVirtualFitter::SetErrorDef(Double_t errdef)
{
//    fgErrorDef = errdef;
   ROOT::Math::MinimizerOptions::SetDefaultErrorDef(errdef);
   if (!GetGlobalFitter()) return;
   Double_t arglist[1];
   arglist[0] = errdef;
   GetGlobalFitter()->ExecuteCommand("SET ERRORDEF", arglist, 1);
}

////////////////////////////////////////////////////////////////////////////////
/// static: Set the tolerance used in the minimization algorithm
/// For example for MIGRAD this is tolerance value passed as second argument
/// (see https://cern-tex.web.cern.ch/cern-tex/minuit/node18.html )

void TVirtualFitter::SetPrecision(Double_t prec)
{
   //fgPrecision = prec;
   ROOT::Math::MinimizerOptions::SetDefaultTolerance(prec);
}
