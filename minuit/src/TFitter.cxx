// @(#)root/minuit:$Name:  $:$Id: TFitter.cxx,v 1.18 2004/07/09 08:02:46 brun Exp $
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
#include "TH1.h"
#include "TF1.h"
#include "TF2.h"
#include "TGraphAsymmErrors.h"
#include "TGraph2D.h"

extern void H1FitChisquare(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);
extern void H1FitLikelihood(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);
extern void GraphFitChisquare(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);
extern void Graph2DFitChisquare(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);

ClassImp(TFitter)

//______________________________________________________________________________
TFitter::TFitter(Int_t maxpar)
{
//*-*-*-*-*-*-*-*-*-*-*default constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===================

   fMinuit = new TMinuit(maxpar);
   fNlog = 0;
   fSumLog = 0;
   fCovar = 0;
   SetName("MinuitFitter");
}

//______________________________________________________________________________
TFitter::~TFitter()
{
//*-*-*-*-*-*-*-*-*-*-*default destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ==================

   if (fCovar)  delete [] fCovar;
   if (fSumLog) delete [] fSumLog;
}

//______________________________________________________________________________
Double_t TFitter::Chisquare(Int_t npar, Double_t *params)
{
   // return a chisquare equivalent
   
   Double_t amin = 0;
   H1FitChisquare(npar,params,amin,params,1);
   return amin;
}

//______________________________________________________________________________
void TFitter::Clear(Option_t *)
{
   // reset the fitter environment

   fMinuit->mncler();

}

//______________________________________________________________________________
Int_t TFitter::ExecuteCommand(const char *command, Double_t *args, Int_t nargs)
{
   // Execute a fitter command;
   //   command : command string
   //   args    : list of nargs command arguments

   Int_t ierr = 0;
   fMinuit->mnexcm(command,args,nargs,ierr);
   return ierr;
}

//______________________________________________________________________________
void TFitter::FixParameter(Int_t ipar)
{
   // Fix parameter ipar.

   fMinuit->FixParameter(ipar);
}

//______________________________________________________________________________
Double_t *TFitter::GetCovarianceMatrix()
{
   // return a pointer to the covariance matrix 

   delete [] fCovar;
   Int_t npars = fMinuit->GetNumPars();
   fCovar = new Double_t[npars*npars];
   fMinuit->mnemat(fCovar,npars);
   return fCovar;
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
   fMinuit->mnerrs(ipar, eplus,eminus,eparab,globcc);
   return ierr;
}


//______________________________________________________________________________
Double_t TFitter::GetParError(Int_t ipar) const
{
   // return error of parameter ipar

   Int_t ierr = 0;
   TString pname;
   Double_t value,verr,vlow,vhigh;

   fMinuit->mnpout(ipar, pname,value,verr,vlow,vhigh,ierr);
   return verr;
}


//______________________________________________________________________________
Double_t TFitter::GetParameter(Int_t ipar) const
{
   // return current value of parameter ipar

   Int_t ierr = 0;
   TString pname;
   Double_t value,verr,vlow,vhigh;

   fMinuit->mnpout(ipar, pname,value,verr,vlow,vhigh,ierr);
   return value;
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
   fMinuit->mnpout(ipar, pname,value,verr,vlow,vhigh,ierr);
   strcpy(parname,pname.Data());
   return ierr;
}

//______________________________________________________________________________
Int_t TFitter::GetStats(Double_t &amin, Double_t &edm, Double_t &errdef, Int_t &nvpar, Int_t &nparx)
{
   // return global fit parameters
   //   amin     : chisquare
   //   edm      : estimated distance to minimum
   //   errdef
   //   nvpar    : number of variable parameters
   //   nparx    : total number of parameters

   Int_t ierr = 0;
   fMinuit->mnstat(amin,edm,errdef,nvpar,nparx,ierr);
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
      for (Int_t j=0;j<=fNlog;j++) {
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

   fMinuit->mnprin(level,amin);
}

//______________________________________________________________________________
void TFitter::ReleaseParameter(Int_t ipar)
{
   // Release parameter ipar.

   fMinuit->Release(ipar);
}

//______________________________________________________________________________
void TFitter::SetFCN(void *fcn)
{
   // Specify the address of the fitting algorithm (from the interpreter)

   TVirtualFitter::SetFCN(fcn);
   fMinuit->SetFCN(fcn);
   
}

//______________________________________________________________________________
void TFitter::SetFCN(void (*fcn)(Int_t &, Double_t *, Double_t &f, Double_t *, Int_t))
{
   // Specify the address of the fitting algorithm

   TVirtualFitter::SetFCN(fcn);
   fMinuit->SetFCN(fcn);
}

//______________________________________________________________________________
void TFitter::SetFitMethod(const char *name)
{
   // ret fit method (chisquare or loglikelihood)
   
   if (!strcmp(name,"H1FitChisquare"))    SetFCN(H1FitChisquare);
   if (!strcmp(name,"H1FitLikelihood"))   SetFCN(H1FitLikelihood);
   if (!strcmp(name,"GraphFitChisquare")) SetFCN(GraphFitChisquare);
   if (!strcmp(name,"Graph2DFitChisquare")) SetFCN(Graph2DFitChisquare);
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
   fMinuit->mnparm(ipar,parname,value,verr,vlow,vhigh,ierr);
   return ierr;
}


//______________________________________________________________________________
void H1FitChisquare(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag)
{
//           Minimization function for H1s using a Chisquare method
//           ======================================================

   Double_t cu,eu,fu,fsum;
   Double_t dersum[100], grad[100];
   Double_t x[3];
   Int_t bin,binx,biny,binz,k;
   Axis_t binlow, binup, binsize;

   Int_t npfits = 0;


   TVirtualFitter *hFitter = TVirtualFitter::GetFitter();
   TH1 *hfit = (TH1*)hFitter->GetObjectFit();
   TF1 *f1   = (TF1*)hFitter->GetUserFunc();
   Foption_t Foption = hFitter->GetFitOption();
   
   f1->InitArgs(x,u);
   npar = f1->GetNpar();
   if (flag == 2) for (k=0;k<npar;k++) dersum[k] = gin[k] = 0;
   f = 0;
   Int_t hxfirst = hFitter->GetXfirst(); 
   Int_t hxlast  = hFitter->GetXlast(); 
   Int_t hyfirst = hFitter->GetYfirst(); 
   Int_t hylast  = hFitter->GetYlast(); 
   Int_t hzfirst = hFitter->GetZfirst(); 
   Int_t hzlast  = hFitter->GetZlast(); 
   TAxis *xaxis  = hfit->GetXaxis();
   TAxis *yaxis  = hfit->GetYaxis();
   TAxis *zaxis  = hfit->GetZaxis();
   
   for (binz=hzfirst;binz<=hzlast;binz++) {
      x[2]  = zaxis->GetBinCenter(binz);
      for (biny=hyfirst;biny<=hylast;biny++) {
         x[1]  = yaxis->GetBinCenter(biny);
         for (binx=hxfirst;binx<=hxlast;binx++) {
            x[0]  = xaxis->GetBinCenter(binx);
            if (!f1->IsInside(x)) continue;
            bin = hfit->GetBin(binx,biny,binz);
            cu  = hfit->GetBinContent(bin);
            TF1::RejectPoint(kFALSE);
            if (Foption.Integral) {
               binlow  = xaxis->GetBinLowEdge(binx);
               binsize = xaxis->GetBinWidth(binx);
               binup   = binlow + binsize;
               fu      = f1->Integral(binlow,binup,u)/binsize;
            } else {
               fu = f1->EvalPar(x,u);
            }
            if (TF1::RejectedPoint()) continue;
            if (Foption.W1) {
               eu = 1;
            } else {
               eu  = hfit->GetBinError(bin);
               if (eu <= 0) continue;
            }
            if (flag == 2) {
               for (k=0;k<npar;k++) dersum[k] += 1; //should be the derivative
            }
            npfits++;
            if (flag == 2) {
               for (k=0;k<npar;k++) grad[k] += dersum[k]*(fu-cu)/eu; dersum[k] = 0;
            }
            fsum = (cu-fu)/eu;
            f += fsum*fsum;
         }
      }
   }
   f1->SetNumberFitPoints(npfits);
}

//______________________________________________________________________________
void H1FitLikelihood(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag)
{
//   -*-*-*-*Minimization function for H1s using a Likelihood method*-*-*-*-*-*
//           =======================================================
//     Basically, it forms the likelihood by determining the Poisson
//     probability that given a number of entries in a particular bin,
//     the fit would predict it's value.  This is then done for each bin,
//     and the sum of the logs is taken as the likelihood.

   Double_t cu,fu,fobs,fsub;
   Double_t dersum[100];
   Double_t x[3];
   Int_t bin,binx,biny,binz,k,icu;
   Axis_t binlow, binup, binsize;

   Int_t npfits = 0;


   TVirtualFitter *hFitter = TVirtualFitter::GetFitter();
   TH1 *hfit = (TH1*)hFitter->GetObjectFit();
   TF1 *f1   = (TF1*)hFitter->GetUserFunc();
   Foption_t Foption = hFitter->GetFitOption();

   f1->InitArgs(x,u);
   npar = f1->GetNpar();
   if (flag == 2) for (k=0;k<npar;k++) dersum[k] = gin[k] = 0;
   f = 0;
   Int_t hxfirst = hFitter->GetXfirst(); 
   Int_t hxlast  = hFitter->GetXlast(); 
   Int_t hyfirst = hFitter->GetYfirst(); 
   Int_t hylast  = hFitter->GetYlast(); 
   Int_t hzfirst = hFitter->GetZfirst(); 
   Int_t hzlast  = hFitter->GetZlast(); 
   TAxis *xaxis  = hfit->GetXaxis();
   TAxis *yaxis  = hfit->GetYaxis();
   TAxis *zaxis  = hfit->GetZaxis();
   
   for (binz=hzfirst;binz<=hzlast;binz++) {
      x[2]  = zaxis->GetBinCenter(binz);
      for (biny=hyfirst;biny<=hylast;biny++) {
         x[1]  = yaxis->GetBinCenter(biny);
         for (binx=hxfirst;binx<=hxlast;binx++) {
            x[0]  = xaxis->GetBinCenter(binx);
            if (!f1->IsInside(x)) continue;
            TF1::RejectPoint(kFALSE);
            bin = hfit->GetBin(binx,biny,binz);
            cu  = hfit->GetBinContent(bin);
            if (Foption.Integral) {
               binlow  = xaxis->GetBinLowEdge(binx);
               binsize = xaxis->GetBinWidth(binx);
               binup   = binlow + binsize;
               fu      = f1->Integral(binlow,binup,u)/binsize;
            } else {
               fu = f1->EvalPar(x,u);
            }
            if (TF1::RejectedPoint()) continue;
            npfits++;
            if (flag == 2) {
               for (k=0;k<npar;k++) {
                  dersum[k] += 1; //should be the derivative
                  //grad[k] += dersum[k]*(fu-cu)/eu; dersum[k] = 0;
               }
            }
            if (fu < 1.e-9) fu = 1.e-9;
            if (Foption.Like == 1) {
               icu  = Int_t(cu);
               fsub = -fu +icu*TMath::Log(fu);
               fobs = hFitter->GetSumLog(icu);
            } else {
               fsub = -fu +cu*TMath::Log(fu);
               fobs = TMath::LnGamma(cu+1);
            }
            fsub -= fobs;
            f -= fsub;
         }
      }
   }
   f *= 2;
   f1->SetNumberFitPoints(npfits);
}

//______________________________________________________________________________
void GraphFitChisquare(Int_t &npar, Double_t * /*gin*/, Double_t &f,
                       Double_t *u, Int_t /*flag*/)
{
//*-*-*-*-*-*Minimization function for Graphs using a Chisquare method*-*-*-*-*
//*-*        =========================================================
//
// In case of a TGraphErrors object, ex, the error along x,  is projected
// along the y-direction by calculating the function at the points x-exlow and
// x+exhigh.
//
// The chisquare is computed as the sum of the quantity below at each point:
//
//                     (y - f(x))**2
//         -----------------------------------
//         ey**2 + ((f(x+exhigh) - f(x-exlow))/2)**2
//
// where x and y are the point coordinates.
//
// In case the function lies below (above) the data point, ey is ey_low (ey_high).
//
//  thanks to Andy Haas (haas@yahoo.com) for adding the case with TGraphasymmerrors
//            University of Washington
//            February 3, 2004
//
//  NOTE:
//  1) By using the "effective variance" method a simple linear regression
//      becomes suddenly a non-linear case , which takes several iterations
//      instead of 0 as in the linear case .
//
//  2) The effective variance technique assumes that there is no correlation 
//      between the x and y coordinate .
//
//    The book by Sigmund Brandt (Data  Analysis) contains an interesting
//    section how to solve the problem when correclations do exist .

   Double_t cu,eu,exh,exl,ey,eux,fu,fsum;
   Double_t x[1];
   Double_t xm,xp;
   Int_t bin, npfits=0;

   TVirtualFitter *grFitter = TVirtualFitter::GetFitter();
   TGraph *gr     = (TGraph*)grFitter->GetObjectFit();
   TF1 *f1   = (TF1*)grFitter->GetUserFunc();
   Foption_t Foption = grFitter->GetFitOption();
   TGraphAsymmErrors *gra = 0;
   if (gr->InheritsFrom(TGraphAsymmErrors::Class())) gra = (TGraphAsymmErrors*)gr;
   
   
   Int_t n        = gr->GetN();
   Double_t *gx   = gr->GetX();
   Double_t *gy   = gr->GetY();
   Double_t fxmin = f1->GetXmin();
   Double_t fxmax = f1->GetXmax();
   npar           = f1->GetNpar();

   f1->InitArgs(x,u);
   f      = 0;
   for (bin=0;bin<n;bin++) {
      x[0] = gx[bin];
      if (!f1->IsInside(x)) continue;
      cu   = gy[bin];
      TF1::RejectPoint(kFALSE);
      fu   = f1->EvalPar(x,u);
      if (TF1::RejectedPoint()) continue;
      fsum = (cu-fu);
      npfits++;
      if (Foption.W1) {
         f += fsum*fsum;
         continue;
      }
      if (gra) {
         exh  = gra->GetEXhigh()[bin];
         exl  = gra->GetEXlow()[bin];
         if (fsum<0) ey = gra->GetEYhigh()[bin];
         else        ey = gra->GetEYlow()[bin];
      } else {
         exh = gr->GetErrorX(bin);
         exl = exh;
         ey  = gr->GetErrorY(bin);
      }
      if (exl < 0) exl = 0;
      if (exh < 0) exh = 0;
      if (ey < 0)  ey  = 0;
      if (exh > 0 && exl > 0) {
        xm = x[0] - exl; if (xm < fxmin) xm = fxmin;
        xp = x[0] + exh; if (xp > fxmax) xp = fxmax;

        //"Effective Variance" method introduced by Anna Kreshuk 
        // in version 4.00/08.
	
	eux = 0.5*(exl + exh)*f1->Derivative(x[0], u);
	
	//Without the "variance method", we had the 3 next lines instead
        // of the line above.
        //x[0] = xm; fm = f1->EvalPar(x,u);
        //x[0] = xp; fp = f1->EvalPar(x,u);
        //eux = 0.5*(fp-fm);
      } else
        eux = 0.;
      eu = ey*ey+eux*eux;
      if (eu <= 0) eu = 1;
      f += fsum*fsum/eu;
   }
   f1->SetNumberFitPoints(npfits);
}


//______________________________________________________________________________
void Graph2DFitChisquare(Int_t &npar, Double_t * /*gin*/, Double_t &f,
                       Double_t *u, Int_t /*flag*/)
{
//*-*-*-*-*Minimization function for 2D Graphs using a Chisquare method*-*-*-*-*
//*-*      ============================================================

   Double_t cu,eu,ex,ey,ez,eux,euy,fu,fsum,fm,fp;
   Double_t x[2];
   Double_t xm,xp,ym,yp;
   Int_t bin, npfits=0;

   TVirtualFitter *grFitter = TVirtualFitter::GetFitter();
   TGraph2D *gr     = (TGraph2D*)grFitter->GetObjectFit();
   TF2 *f2   = (TF2*)grFitter->GetUserFunc();
   Foption_t Foption = grFitter->GetFitOption();
   
   Int_t n        = gr->GetN();
   Double_t *gx   = gr->GetX();
   Double_t *gy   = gr->GetY();
   Double_t *gz   = gr->GetZ();
   Double_t fxmin = f2->GetXmin();
   Double_t fxmax = f2->GetXmax();
   Double_t fymin = f2->GetYmin();
   Double_t fymax = f2->GetYmax();
   npar           = f2->GetNpar();

   f2->InitArgs(x,u);
   f      = 0;
   for (bin=0;bin<n;bin++) {
      x[0] = gx[bin];
      x[1] = gy[bin];
      if (!f2->IsInside(x)) continue;
      cu   = gz[bin];
      TF2::RejectPoint(kFALSE);
      fu   = f2->EvalPar(x,u);
      if (TF2::RejectedPoint()) continue;
      fsum = (cu-fu);
      npfits++;
      if (Foption.W1) {
         f += fsum*fsum;
         continue;
      }
      ex  = gr->GetErrorX(bin);
      ey  = gr->GetErrorY(bin);
      ez  = gr->GetErrorZ(bin);
      if (ex < 0) ex = 0;
      if (ey < 0) ey = 0;
      if (ez < 0) ez = 0;
      eux = euy = 0;
      if (ex > 0) {
        xm = x[0] - ex; if (xm < fxmin) xm = fxmin;
        xp = x[0] + ex; if (xp > fxmax) xp = fxmax;
        x[0] = xm; fm = f2->EvalPar(x,u);
        x[0] = xp; fp = f2->EvalPar(x,u);
        eux = fp-fm;
      }
      if (ey > 0) {
        x[0] = gx[bin];
        ym = x[1] - ey; if (ym < fymin) ym = fxmin;
        yp = x[1] + ey; if (yp > fymax) yp = fymax;
        x[1] = ym; fm = f2->EvalPar(x,u);
        x[1] = yp; fp = f2->EvalPar(x,u);
        euy = fp-fm;
      }
      eu = ez*ez+eux*eux+euy*euy;
      if (eu <= 0) eu = 1;
      f += fsum*fsum/eu;
   }
   f2->SetNumberFitPoints(npfits);
}
