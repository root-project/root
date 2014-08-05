// @(#)root/minuit:$Id$
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
#include "TF3.h"
#include "TList.h"
#include "TGraph.h"
#include "TGraph2D.h"
#include "TMultiGraph.h"
#include "TMath.h"

extern void H1FitChisquare(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);
extern void H1FitLikelihood(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);
extern void GraphFitChisquare(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);
extern void Graph2DFitChisquare(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);
extern void MultiGraphFitChisquare(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);
extern void F2Fit(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);
extern void F3Fit(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);

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
   delete fMinuit;
}

//______________________________________________________________________________
Double_t TFitter::Chisquare(Int_t npar, Double_t *params) const
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

   if (fCovar)  {delete [] fCovar; fCovar = 0;}
   fMinuit->mncler();

   //reset the internal Minuit random generator to its initial state
   Double_t val = 3;
   Int_t inseed = 12345;
   fMinuit->mnrn15(val,inseed);
}

//______________________________________________________________________________
Int_t TFitter::ExecuteCommand(const char *command, Double_t *args, Int_t nargs)
{
   // Execute a fitter command;
   //   command : command string
   //   args    : list of nargs command arguments

   if (fCovar)  {delete [] fCovar; fCovar = 0;}
   Int_t ierr = 0;
   fMinuit->mnexcm(command,args,nargs,ierr);
   return ierr;
}

//______________________________________________________________________________
void TFitter::FixParameter(Int_t ipar)
{
   // Fix parameter ipar.

   if (fCovar)  {delete [] fCovar; fCovar = 0;}
   fMinuit->FixParameter(ipar);
}

//______________________________________________________________________________
void TFitter::GetConfidenceIntervals(Int_t n, Int_t ndim, const Double_t *x, Double_t *ci, Double_t cl)
{
//Computes point-by-point confidence intervals for the fitted function
//Parameters:
//n - number of points
//ndim - dimensions of points
//x - points, at which to compute the intervals, for ndim > 1
//    should be in order: (x0,y0, x1, y1, ... xn, yn)
//ci - computed intervals are returned in this array
//cl - confidence level, default=0.95
//NOTE, that the intervals are approximate for nonlinear(in parameters) models

   TF1 *f = (TF1*)fUserFunc;
   Int_t npar = f->GetNumberFreeParameters();
   Int_t npar_real = f->GetNpar();
   Double_t *grad = new Double_t[npar_real];
   Double_t *sum_vector = new Double_t[npar];
   Bool_t *fixed=0;
   Double_t al, bl;
   if (npar_real != npar){
      fixed = new Bool_t[npar_real];
      memset(fixed,0,npar_real*sizeof(Bool_t));

      for (Int_t ipar=0; ipar<npar_real; ipar++){
         fixed[ipar]=0;
         f->GetParLimits(ipar,al,bl);
         if (al*bl != 0 && al >= bl) {
            //this parameter is fixed
            fixed[ipar]=1;
         }
      }
   }
   Double_t c=0;

   Double_t *matr = GetCovarianceMatrix();
   if (!matr)
      return;
   Double_t t = TMath::StudentQuantile(0.5 + cl/2, f->GetNDF());
   Double_t chidf = TMath::Sqrt(f->GetChisquare()/f->GetNDF());
   Int_t igrad, ifree=0;
   for (Int_t ipoint=0; ipoint<n; ipoint++){
      c=0;
      f->GradientPar(x+ndim*ipoint, grad);
      //multiply the covariance matrix by gradient
      for (Int_t irow=0; irow<npar; irow++){
         sum_vector[irow]=0;
         igrad = 0;
         for (Int_t icol=0; icol<npar; icol++){
            igrad = 0;
            ifree=0;
            if (fixed) {
               //find the free parameter #icol
               while (ifree<icol+1){
                  if (fixed[igrad]==0) ifree++;
                  igrad++;
               }
               igrad--;
               //now the [igrad] element of gradient corresponds to [icol] element of cov.matrix
            } else {
               igrad = icol;
            }
            sum_vector[irow]+=matr[irow*npar_real+icol]*grad[igrad];
         }
      }
      igrad = 0;
      for (Int_t i=0; i<npar; i++){
         igrad = 0; ifree=0;
         if (fixed) {
            //find the free parameter #icol
            while (ifree<i+1){
               if (fixed[igrad]==0) ifree++;
               igrad++;
            }
            igrad--;
         } else {
            igrad = i;
         }
         c+=grad[igrad]*sum_vector[i];
      }

      c=TMath::Sqrt(c);
      ci[ipoint]=c*t*chidf;
   }

   delete [] grad;
   delete [] sum_vector;
   if (fixed)
      delete [] fixed;
}

//______________________________________________________________________________
void TFitter::GetConfidenceIntervals(TObject *obj, Double_t cl)
{
//Computes confidence intervals at level cl. Default is 0.95
//The TObject parameter can be a TGraphErrors, a TGraph2DErrors or a TH1,2,3.
//For Graphs, confidence intervals are computed for each point,
//the value of the graph at that point is set to the function value at that
//point, and the graph y-errors (or z-errors) are set to the value of
//the confidence interval at that point.
//For Histograms, confidence intervals are computed for each bin center
//The bin content of this bin is then set to the function value at the bin
//center, and the bin error is set to the confidence interval value.
//NOTE: confidence intervals are approximate for nonlinear models!
//
//Allowed combinations:
//Fitted object               Passed object
//TGraph                      TGraphErrors, TH1
//TGraphErrors, AsymmErrors   TGraphErrors, TH1
//TH1                         TGraphErrors, TH1
//TGraph2D                    TGraph2DErrors, TH2
//TGraph2DErrors              TGraph2DErrors, TH2
//TH2                         TGraph2DErrors, TH2
//TH3                         TH3

   if (obj->InheritsFrom(TGraph::Class())) {
      TGraph *gr = (TGraph*)obj;
      if (!gr->GetEY()){
         Error("GetConfidenceIntervals", "A TGraphErrors should be passed instead of a graph");
         return;
      }
      if (fObjectFit->InheritsFrom(TGraph2D::Class())){
         Error("GetConfidenceIntervals", "A TGraph2DErrors should be passed instead of a graph");
         return;
      }
      if (fObjectFit->InheritsFrom(TH1::Class())){
         if (((TH1*)(fObjectFit))->GetDimension()>1){
            Error("GetConfidenceIntervals", "A TGraph2DErrors or a TH23 should be passed instead of a graph");
            return;
         }
      }
      GetConfidenceIntervals(gr->GetN(),1,gr->GetX(), gr->GetEY(), cl);
      for (Int_t i=0; i<gr->GetN(); i++)
         gr->SetPoint(i, gr->GetX()[i], ((TF1*)(fUserFunc))->Eval(gr->GetX()[i]));
   }

   //TGraph2D/////////////////
   else if (obj->InheritsFrom(TGraph2D::Class())) {
      TGraph2D *gr2 = (TGraph2D*)obj;
      if (!gr2->GetEZ()){
         Error("GetConfidenceIntervals", "A TGraph2DErrors should be passed instead of a TGraph2D");
         return;
      }
      if (fObjectFit->InheritsFrom(TGraph::Class())){
         Error("GetConfidenceIntervals", "A TGraphErrors should be passed instead of a TGraph2D");
         return;
      }
      if (fObjectFit->InheritsFrom(TH1::Class())){
         if (((TH1*)(fObjectFit))->GetDimension()==1){
            Error("GetConfidenceIntervals", "A TGraphErrors or a TH1 should be passed instead of a graph");
            return;
         }
      }
      TF2 *f=(TF2*)fUserFunc;
      Double_t xy[2];
      Int_t np = gr2->GetN();
      Int_t npar = f->GetNpar();
      Double_t *grad = new Double_t[npar];
      Double_t *sum_vector = new Double_t[npar];
      Double_t *x = gr2->GetX();
      Double_t *y = gr2->GetY();
      Double_t t = TMath::StudentQuantile(0.5 + cl/2, f->GetNDF());
      Double_t chidf = TMath::Sqrt(f->GetChisquare()/f->GetNDF());
      Double_t *matr=GetCovarianceMatrix();
      Double_t c = 0;
      for (Int_t ipoint=0; ipoint<np; ipoint++){
         xy[0]=x[ipoint];
         xy[1]=y[ipoint];
         f->GradientPar(xy, grad);
         for (Int_t irow=0; irow<f->GetNpar(); irow++){
            sum_vector[irow]=0;
            for (Int_t icol=0; icol<npar; icol++)
               sum_vector[irow]+=matr[irow*npar+icol]*grad[icol];
         }
         c = 0;
         for (Int_t i=0; i<npar; i++)
            c+=grad[i]*sum_vector[i];
         c=TMath::Sqrt(c);
         gr2->SetPoint(ipoint, xy[0], xy[1], f->EvalPar(xy));
         gr2->GetEZ()[ipoint]=c*t*chidf;

      }
      delete [] grad;
      delete [] sum_vector;
   }

   //TH1/////////////////////////
   else if (obj->InheritsFrom(TH1::Class())) {
      if (fObjectFit->InheritsFrom(TGraph::Class())){
         if (((TH1*)obj)->GetDimension()>1){
            Error("GetConfidenceIntervals", "Fitted graph and passed histogram have different number of dimensions");
            return;
         }
      }
      if (fObjectFit->InheritsFrom(TGraph2D::Class())){
         if (((TH1*)obj)->GetDimension()!=2){
            Error("GetConfidenceIntervals", "Fitted graph and passed histogram have different number of dimensions");
            return;
         }
      }
      if (fObjectFit->InheritsFrom(TH1::Class())){
         if (((TH1*)(fObjectFit))->GetDimension()!=((TH1*)(obj))->GetDimension()){
            Error("GetConfidenceIntervals", "Fitted and passed histograms have different number of dimensions");
            return;
         }
      }


      TH1 *hfit = (TH1*)obj;
      TF1 *f = (TF1*)GetUserFunc();
      Int_t npar = f->GetNpar();
      Double_t *grad = new Double_t[npar];
      Double_t *sum_vector = new Double_t[npar];
      Double_t x[3];

      Int_t hxfirst = hfit->GetXaxis()->GetFirst();
      Int_t hxlast  = hfit->GetXaxis()->GetLast();
      Int_t hyfirst = hfit->GetYaxis()->GetFirst();
      Int_t hylast  = hfit->GetYaxis()->GetLast();
      Int_t hzfirst = hfit->GetZaxis()->GetFirst();
      Int_t hzlast  = hfit->GetZaxis()->GetLast();

      TAxis *xaxis  = hfit->GetXaxis();
      TAxis *yaxis  = hfit->GetYaxis();
      TAxis *zaxis  = hfit->GetZaxis();
      Double_t t = TMath::StudentQuantile(0.5 + cl/2, f->GetNDF());
      Double_t chidf = TMath::Sqrt(f->GetChisquare()/f->GetNDF());
      Double_t *matr=GetCovarianceMatrix();
      Double_t c=0;
      for (Int_t binz=hzfirst; binz<=hzlast; binz++){
         x[2]=zaxis->GetBinCenter(binz);
         for (Int_t biny=hyfirst; biny<=hylast; biny++) {
            x[1]=yaxis->GetBinCenter(biny);
            for (Int_t binx=hxfirst; binx<=hxlast; binx++) {
               x[0]=xaxis->GetBinCenter(binx);
               f->GradientPar(x, grad);
               for (Int_t irow=0; irow<npar; irow++){
                  sum_vector[irow]=0;
                  for (Int_t icol=0; icol<npar; icol++)
                     sum_vector[irow]+=matr[irow*npar+icol]*grad[icol];
               }
               c = 0;
               for (Int_t i=0; i<npar; i++)
                  c+=grad[i]*sum_vector[i];
               c=TMath::Sqrt(c);
               hfit->SetBinContent(binx, biny, binz, f->EvalPar(x));
               hfit->SetBinError(binx, biny, binz, c*t*chidf);
            }
         }
      }
      delete [] grad;
      delete [] sum_vector;
   }
   else {
      Error("GetConfidenceIntervals", "This object type is not supported");
      return;
   }

}

//______________________________________________________________________________
Double_t *TFitter::GetCovarianceMatrix() const
{
   // return a pointer to the covariance matrix

   if (fCovar) return fCovar;
   Int_t npars = fMinuit->GetNumPars();
   ((TFitter*)this)->fCovar = new Double_t[npars*npars];
   fMinuit->mnemat(fCovar,npars);
   return fCovar;
}

//______________________________________________________________________________
Double_t TFitter::GetCovarianceMatrixElement(Int_t i, Int_t j) const
{
   // return element i,j from the covariance matrix

   GetCovarianceMatrix();
   Int_t npars = fMinuit->GetNumPars();
   if (i < 0 || i >= npars || j < 0 || j >= npars) {
      Error("GetCovarianceMatrixElement","Illegal arguments i=%d, j=%d",i,j);
      return 0;
   }
   return fCovar[j+npars*i];
}

//______________________________________________________________________________
Int_t TFitter::GetErrors(Int_t ipar,Double_t &eplus, Double_t &eminus, Double_t &eparab, Double_t &globcc) const
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
Int_t TFitter::GetNumberTotalParameters() const
{
   // return the total number of parameters (free + fixed)

   return fMinuit->fNpar + fMinuit->fNpfix;
}

//______________________________________________________________________________
Int_t TFitter::GetNumberFreeParameters() const
{
   // return the number of free parameters

   return fMinuit->fNpar;
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
Int_t TFitter::GetParameter(Int_t ipar, char *parname,Double_t &value,Double_t &verr,Double_t &vlow, Double_t &vhigh) const
{
   // return current values for a parameter
   //   ipar     : parameter number
   //   parname  : parameter name
   //   value    : initial parameter value
   //   verr     : initial error for this parameter
   //   vlow     : lower value for the parameter
   //   vhigh    : upper value for the parameter
   //  WARNING! parname must be suitably dimensionned in the calling function.

   Int_t ierr = 0;
   TString pname;
   fMinuit->mnpout(ipar, pname,value,verr,vlow,vhigh,ierr);
   strcpy(parname,pname.Data());
   return ierr;
}

//______________________________________________________________________________
const char *TFitter::GetParName(Int_t ipar) const
{
   // return name of parameter ipar

   if (!fMinuit || ipar < 0 || ipar > fMinuit->fNu) return "";
   return fMinuit->fCpnam[ipar];
}

//______________________________________________________________________________
Int_t TFitter::GetStats(Double_t &amin, Double_t &edm, Double_t &errdef, Int_t &nvpar, Int_t &nparx) const
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
Bool_t TFitter::IsFixed(Int_t ipar) const
{
   //return kTRUE if parameter ipar is fixed, kFALSE othersise)

   if (fMinuit->fNiofex[ipar] == 0 ) return kTRUE;
   return kFALSE;
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

   if (fCovar)  {delete [] fCovar; fCovar = 0;}
   fMinuit->Release(ipar);
}

//______________________________________________________________________________
void TFitter::SetFCN(void *fcn)
{
   // Specify the address of the fitting algorithm (from the interpreter)

   if (fCovar)  {delete [] fCovar; fCovar = 0;}
   TVirtualFitter::SetFCN(fcn);
   fMinuit->SetFCN(fcn);

}

//______________________________________________________________________________
void TFitter::SetFCN(void (*fcn)(Int_t &, Double_t *, Double_t &f, Double_t *, Int_t))
{
   // Specify the address of the fitting algorithm

   if (fCovar)  {delete [] fCovar; fCovar = 0;}
   TVirtualFitter::SetFCN(fcn);
   fMinuit->SetFCN(fcn);
}

//______________________________________________________________________________
void TFitter::SetFitMethod(const char *name)
{
   // ret fit method (chisquare or loglikelihood)

   if (fCovar)  {delete [] fCovar; fCovar = 0;}
   if (!strcmp(name,"H1FitChisquare"))    SetFCN(H1FitChisquare);
   if (!strcmp(name,"H1FitLikelihood"))   SetFCN(H1FitLikelihood);
   if (!strcmp(name,"GraphFitChisquare")) SetFCN(GraphFitChisquare);
   if (!strcmp(name,"Graph2DFitChisquare")) SetFCN(Graph2DFitChisquare);
   if (!strcmp(name,"MultiGraphFitChisquare")) SetFCN(MultiGraphFitChisquare);
   if (!strcmp(name,"F2Minimizer")) SetFCN(F2Fit);
   if (!strcmp(name,"F3Minimizer")) SetFCN(F3Fit);
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

   if (fCovar)  {delete [] fCovar; fCovar = 0;}
   Int_t ierr = 0;
   fMinuit->mnparm(ipar,parname,value,verr,vlow,vhigh,ierr);
   return ierr;
}

//______________________________________________________________________________
void TFitter::FitChisquare(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag)
{
   //  Minimization function for H1s using a Chisquare method
   //  Default method (function evaluated at center of bin)
   //  for each point the cache contains the following info
   //    -1D : bc,e, xc  (bin content, error, x of center of bin)
   //    -2D : bc,e, xc,yc
   //    -3D : bc,e, xc,yc,zc

   Foption_t fitOption = GetFitOption();
   if (fitOption.Integral) {
      FitChisquareI(npar,gin,f,u,flag);
      return;
   }
   Double_t cu,eu,fu,fsum;
   Double_t dersum[100], grad[100];
   memset(grad,0,800);
   Double_t x[3];

   TH1 *hfit = (TH1*)GetObjectFit();
   TF1 *f1   = (TF1*)GetUserFunc();
   Int_t nd  = hfit->GetDimension();
   Int_t j;

   f1->InitArgs(x,u);
   npar = f1->GetNpar();
   if (flag == 2) for (j=0;j<npar;j++) dersum[j] = gin[j] = 0;
   f = 0;

   Int_t npfit = 0;
   Double_t *cache = fCache;
   for (Int_t i=0;i<fNpoints;i++) {
      if (nd > 2) x[2]     = cache[4];
      if (nd > 1) x[1]     = cache[3];
      x[0]     = cache[2];
      cu  = cache[0];
      TF1::RejectPoint(kFALSE);
      fu = f1->EvalPar(x,u);
      if (TF1::RejectedPoint()) {cache += fPointSize; continue;}
      eu = cache[1];
      if (flag == 2) {
         for (j=0;j<npar;j++) dersum[j] += 1; //should be the derivative
         for (j=0;j<npar;j++) grad[j] += dersum[j]*(fu-cu)/eu; dersum[j] = 0;
      }
      fsum = (cu-fu)/eu;
      f += fsum*fsum;
      npfit++;
      cache += fPointSize;
   }
   f1->SetNumberFitPoints(npfit);
}

//______________________________________________________________________________
void TFitter::FitChisquareI(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag)
{
   //  Minimization function for H1s using a Chisquare method
   //  The "I"ntegral method is used
   //  for each point the cache contains the following info
   //    -1D : bc,e, xc,xw  (bin content, error, x of center of bin, x bin width of bin)
   //    -2D : bc,e, xc,xw,yc,yw
   //    -3D : bc,e, xc,xw,yc,yw,zc,zw

   Double_t cu,eu,fu,fsum;
   Double_t dersum[100], grad[100];
   memset(grad,0,800);
   Double_t x[3];

   TH1 *hfit = (TH1*)GetObjectFit();
   TF1 *f1   = (TF1*)GetUserFunc();
   Int_t nd = hfit->GetDimension();
   Int_t j;

   f1->InitArgs(x,u);
   npar = f1->GetNpar();
   if (flag == 2) for (j=0;j<npar;j++) dersum[j] = gin[j] = 0;
   f = 0;

   Int_t npfit = 0;
   Double_t *cache = fCache;
   for (Int_t i=0;i<fNpoints;i++) {
      cu  = cache[0];
      TF1::RejectPoint(kFALSE);
      f1->SetParameters(u);
      if (nd < 2) {
         fu = f1->Integral(cache[2] - 0.5*cache[3],cache[2] + 0.5*cache[3])/cache[3];
      } else if (nd < 3) {
         fu = ((TF2*)f1)->Integral(cache[2] - 0.5*cache[3],cache[2] + 0.5*cache[3],cache[4] - 0.5*cache[5],cache[4] + 0.5*cache[5])/(cache[3]*cache[5]);
      } else {
         fu = ((TF3*)f1)->Integral(cache[2] - 0.5*cache[3],cache[2] + 0.5*cache[3],cache[4] - 0.5*cache[5],cache[4] + 0.5*cache[5],cache[6] - 0.5*cache[7],cache[6] + 0.5*cache[7])/(cache[3]*cache[5]*cache[7]);
      }
      if (TF1::RejectedPoint()) {cache += fPointSize; continue;}
      eu = cache[1];
      if (flag == 2) {
         for (j=0;j<npar;j++) dersum[j] += 1; //should be the derivative
         for (j=0;j<npar;j++) grad[j] += dersum[j]*(fu-cu)/eu; dersum[j] = 0;
      }
      fsum = (cu-fu)/eu;
      f += fsum*fsum;
      npfit++;
      cache += fPointSize;
   }
   f1->SetNumberFitPoints(npfit);
}


//______________________________________________________________________________
void TFitter::FitLikelihood(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag)
{
   //  Minimization function for H1s using a Likelihood method*-*-*-*-*-*
   //     Basically, it forms the likelihood by determining the Poisson
   //     probability that given a number of entries in a particular bin,
   //     the fit would predict it's value.  This is then done for each bin,
   //     and the sum of the logs is taken as the likelihood.
   //  Default method (function evaluated at center of bin)
   //  for each point the cache contains the following info
   //    -1D : bc,e, xc  (bin content, error, x of center of bin)
   //    -2D : bc,e, xc,yc
   //    -3D : bc,e, xc,yc,zc

   Foption_t fitOption = GetFitOption();
   if (fitOption.Integral) {
      FitLikelihoodI(npar,gin,f,u,flag);
      return;
   }
   Double_t cu,fu,fobs,fsub;
   Double_t dersum[100];
   Double_t x[3];
   Int_t icu;

   TH1 *hfit = (TH1*)GetObjectFit();
   TF1 *f1   = (TF1*)GetUserFunc();
   Int_t nd = hfit->GetDimension();
   Int_t j;

   f1->InitArgs(x,u);
   npar = f1->GetNpar();
   if (flag == 2) for (j=0;j<npar;j++) dersum[j] = gin[j] = 0;
   f = 0;

   Int_t npfit = 0;
   Double_t *cache = fCache;
   for (Int_t i=0;i<fNpoints;i++) {
      if (nd > 2) x[2] = cache[4];
      if (nd > 1) x[1] = cache[3];
      x[0]     = cache[2];
      cu  = cache[0];
      TF1::RejectPoint(kFALSE);
      fu = f1->EvalPar(x,u);
      if (TF1::RejectedPoint()) {cache += fPointSize; continue;}
      if (flag == 2) {
         for (j=0;j<npar;j++) {
            dersum[j] += 1; //should be the derivative
            //grad[j] += dersum[j]*(fu-cu)/eu; dersum[j] = 0;
         }
      }
      if (fu < 1.e-9) fu = 1.e-9;
      if (fitOption.Like == 1) {
         icu  = Int_t(cu);
         fsub = -fu +cu*TMath::Log(fu);
         if (icu <10000) fobs = GetSumLog(icu);
         else            fobs = TMath::LnGamma(cu+1);
      } else {
         fsub = -fu +cu*TMath::Log(fu);
         fobs = TMath::LnGamma(cu+1);
      }
      fsub -= fobs;
      f -= fsub;
      npfit++;
      cache += fPointSize;
   }
   f *= 2;
   f1->SetNumberFitPoints(npfit);
}


//______________________________________________________________________________
void TFitter::FitLikelihoodI(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag)
{
   //  Minimization function for H1s using a Likelihood method*-*-*-*-*-*
   //     Basically, it forms the likelihood by determining the Poisson
   //     probability that given a number of entries in a particular bin,
   //     the fit would predict it's value.  This is then done for each bin,
   //     and the sum of the logs is taken as the likelihood.
   //  The "I"ntegral method is used
   //  for each point the cache contains the following info
   //    -1D : bc,e, xc,xw  (bin content, error, x of center of bin, x bin width of bin)
   //    -2D : bc,e, xc,xw,yc,yw
   //    -3D : bc,e, xc,xw,yc,yw,zc,zw

   Double_t cu,fu,fobs,fsub;
   Double_t dersum[100];
   Double_t x[3];
   Int_t icu;

   TH1 *hfit = (TH1*)GetObjectFit();
   TF1 *f1   = (TF1*)GetUserFunc();
   Foption_t fitOption = GetFitOption();
   Int_t nd = hfit->GetDimension();
   Int_t j;

   f1->InitArgs(x,u);
   npar = f1->GetNpar();
   if (flag == 2) for (j=0;j<npar;j++) dersum[j] = gin[j] = 0;
   f = 0;

   Int_t npfit = 0;
   Double_t *cache = fCache;
   for (Int_t i=0;i<fNpoints;i++) {
      if (nd > 2) x[2] = cache[6];
      if (nd > 1) x[1] = cache[4];
      x[0]     = cache[2];
      cu  = cache[0];
      TF1::RejectPoint(kFALSE);
      f1->SetParameters(u);
      if (nd < 2) {
         fu = f1->Integral(cache[2] - 0.5*cache[3],cache[2] + 0.5*cache[3])/cache[3];
      } else if (nd < 3) {
         fu = ((TF2*)f1)->Integral(cache[2] - 0.5*cache[3],cache[2] + 0.5*cache[3],cache[4] - 0.5*cache[5],cache[4] + 0.5*cache[5])/(cache[3]*cache[5]);
      } else {
         fu = ((TF3*)f1)->Integral(cache[2] - 0.5*cache[3],cache[2] + 0.5*cache[3],cache[4] - 0.5*cache[5],cache[4] + 0.5*cache[5],cache[6] - 0.5*cache[7],cache[6] + 0.5*cache[7])/(cache[3]*cache[5]*cache[7]);
      }
      if (TF1::RejectedPoint()) {cache += fPointSize; continue;}
      if (flag == 2) {
         for (j=0;j<npar;j++) {
            dersum[j] += 1; //should be the derivative
            //grad[j] += dersum[j]*(fu-cu)/eu; dersum[j] = 0;
         }
      }
      if (fu < 1.e-9) fu = 1.e-9;
      if (fitOption.Like == 1) {
         icu  = Int_t(cu);
         fsub = -fu +cu*TMath::Log(fu);
         if (icu <10000) fobs = GetSumLog(icu);
         else            fobs = TMath::LnGamma(cu+1);
      } else {
         fsub = -fu +cu*TMath::Log(fu);
         fobs = TMath::LnGamma(cu+1);
      }
      fsub -= fobs;
      f -= fsub;
      npfit++;
      cache += fPointSize;
   }
   f *= 2;
   f1->SetNumberFitPoints(npfit);
}



//______________________________________________________________________________
void H1FitChisquare(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag)
{
//           Minimization function for H1s using a Chisquare method
//           ======================================================

   TFitter *hFitter = (TFitter*)TVirtualFitter::GetFitter();
   hFitter->FitChisquare(npar, gin, f, u, flag);
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

   TFitter *hFitter = (TFitter*)TVirtualFitter::GetFitter();
   hFitter->FitLikelihood(npar, gin, f, u, flag);
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
//         ey**2 + (0.5*(exl + exh)*f'(x))**2
//
// where x and y are the point coordinates and f'(x) is the derivative of function f(x).
// This method to approximate the uncertainty in y because of the errors in x, is called
// "effective variance" method.
// The improvement, compared to the previously used  method (f(x+ exhigh) - f(x-exlow))/2
// is of (error of x)**2 order.
//
// In case the function lies below (above) the data point, ey is ey_low (ey_high).
//
//  thanks to Andy Haas (haas@yahoo.com) for adding the case with TGraphAsymmErrors
//            University of Washington
//            February 3, 2004
//
//  NOTE:
//  1) By using the "effective variance" method a simple linear regression
//      becomes a non-linear case , which takes several iterations
//      instead of 0 as in the linear case .
//
//  2) The effective variance technique assumes that there is no correlation
//      between the x and y coordinate .
//
//    The book by Sigmund Brandt (Data  Analysis) contains an interesting
//    section how to solve the problem when correclations do exist .

   Double_t cu,eu,exh,exl,ey,eux,fu,fsum;
   Double_t x[1];
   //Double_t xm,xp;
   Int_t bin, npfits=0;

   TVirtualFitter *grFitter = TVirtualFitter::GetFitter();
   TGraph *gr     = (TGraph*)grFitter->GetObjectFit();
   TF1 *f1   = (TF1*)grFitter->GetUserFunc();
   Foption_t fitOption = grFitter->GetFitOption();
   Int_t n        = gr->GetN();
   Double_t *gx   = gr->GetX();
   Double_t *gy   = gr->GetY();
   //Double_t fxmin = f1->GetXmin();
   //Double_t fxmax = f1->GetXmax();
   npar           = f1->GetNpar();

   f      = 0;
   for (bin=0;bin<n;bin++) {
      f1->InitArgs(x,u); //must be inside the loop because of TF1::Derivative calling InitArgs
      x[0] = gx[bin];
      if (!f1->IsInside(x)) continue;
      cu   = gy[bin];
      TF1::RejectPoint(kFALSE);
      fu   = f1->EvalPar(x,u);
      if (TF1::RejectedPoint()) continue;
      fsum = (cu-fu);
      npfits++;
      if (fitOption.W1) {
         f += fsum*fsum;
         continue;
      }

      exh = gr->GetErrorXhigh(bin);
      exl = gr->GetErrorXlow(bin);
      if (fsum < 0)
         ey = gr->GetErrorYhigh(bin);
      else
         ey = gr->GetErrorYlow(bin);
      if (exl < 0) exl = 0;
      if (exh < 0) exh = 0;
      if (ey < 0)  ey  = 0;
      if (exh > 0 || exl > 0) {
         //Without the "variance method", we had the 6 next lines instead
         // of the line above.
          //xm = x[0] - exl; if (xm < fxmin) xm = fxmin;
          //xp = x[0] + exh; if (xp > fxmax) xp = fxmax;
          //Double_t fm,fp;
          //x[0] = xm; fm = f1->EvalPar(x,u);
          //x[0] = xp; fp = f1->EvalPar(x,u);
          //eux = 0.5*(fp-fm);

         //"Effective Variance" method introduced by Anna Kreshuk
         // in version 4.00/08.
         eux = 0.5*(exl + exh)*f1->Derivative(x[0], u);
      } else
         eux = 0.;
      eu = ey*ey+eux*eux;
      if (eu <= 0) eu = 1;
      f += fsum*fsum/eu;
   }
   f1->SetNumberFitPoints(npfits);
   //
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
   Foption_t fitOption = grFitter->GetFitOption();

   Int_t n        = gr->GetN();
   Double_t *gx   = gr->GetX();
   Double_t *gy   = gr->GetY();
   Double_t *gz   = gr->GetZ();
   Double_t fxmin = f2->GetXmin();
   Double_t fxmax = f2->GetXmax();
   Double_t fymin = f2->GetYmin();
   Double_t fymax = f2->GetYmax();
   npar           = f2->GetNpar();
   f      = 0;
   for (bin=0;bin<n;bin++) {
      f2->InitArgs(x,u);
      x[0] = gx[bin];
      x[1] = gy[bin];
      if (!f2->IsInside(x)) continue;
      cu   = gz[bin];
      TF2::RejectPoint(kFALSE);
      fu   = f2->EvalPar(x,u);
      if (TF2::RejectedPoint()) continue;
      fsum = (cu-fu);
      npfits++;
      if (fitOption.W1) {
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

//______________________________________________________________________________
void MultiGraphFitChisquare(Int_t &npar, Double_t * /*gin*/, Double_t &f,
                       Double_t *u, Int_t /*flag*/)
{

   Double_t cu,eu,exh,exl,ey,eux,fu,fsum;
   Double_t x[1];
   //Double_t xm,xp;
   Int_t bin, npfits=0;

   TVirtualFitter *grFitter = TVirtualFitter::GetFitter();
   TMultiGraph *mg     = (TMultiGraph*)grFitter->GetObjectFit();
   TF1 *f1   = (TF1*)grFitter->GetUserFunc();
   Foption_t fitOption = grFitter->GetFitOption();
   TGraph *gr;
   TIter next(mg->GetListOfGraphs());

   Int_t n;
   Double_t *gx;
   Double_t *gy;
   //Double_t fxmin = f1->GetXmin();
   //Double_t fxmax = f1->GetXmax();
   npar           = f1->GetNpar();

   f      = 0;

   while ((gr = (TGraph*) next())) {
      n        = gr->GetN();
      gx   = gr->GetX();
      gy   = gr->GetY();
      for (bin=0;bin<n;bin++) {
         f1->InitArgs(x,u); //must be inside the loop because of TF1::Derivative calling InitArgs
         x[0] = gx[bin];
         if (!f1->IsInside(x)) continue;
         cu   = gy[bin];
         TF1::RejectPoint(kFALSE);
         fu   = f1->EvalPar(x,u);
         if (TF1::RejectedPoint()) continue;
         fsum = (cu-fu);
         npfits++;
         if (fitOption.W1) {
            f += fsum*fsum;
            continue;
         }
         exh = gr->GetErrorXhigh(bin);
         exl = gr->GetErrorXlow(bin);
         ey  = gr->GetErrorY(bin);
         if (exl < 0) exl = 0;
         if (exh < 0) exh = 0;
         if (ey < 0)  ey  = 0;
         if (exh > 0 && exl > 0) {
            //Without the "variance method", we had the 6 next lines instead
            // of the line above.
            //xm = x[0] - exl; if (xm < fxmin) xm = fxmin;
            //xp = x[0] + exh; if (xp > fxmax) xp = fxmax;
            //Double_t fm,fp;
            //x[0] = xm; fm = f1->EvalPar(x,u);
            //x[0] = xp; fp = f1->EvalPar(x,u);
            //eux = 0.5*(fp-fm);

            //"Effective Variance" method introduced by Anna Kreshuk
            // in version 4.00/08.
            eux = 0.5*(exl + exh)*f1->Derivative(x[0], u);
         } else
            eux = 0.;
         eu = ey*ey+eux*eux;
         if (eu <= 0) eu = 1;
         f += fsum*fsum/eu;
      }
   }
   f1->SetNumberFitPoints(npfits);
}

//______________________________________________________________________________
void F2Fit(Int_t &/*npar*/, Double_t * /*gin*/, Double_t &f,Double_t *u, Int_t /*flag*/)
{
   TVirtualFitter *fitter = TVirtualFitter::GetFitter();
   TF2 *f2 = (TF2*)fitter->GetObjectFit();
   f2->InitArgs(u, f2->GetParameters() );
   f = f2->EvalPar(u);
}

//______________________________________________________________________________
void F3Fit(Int_t &/*npar*/, Double_t * /*gin*/, Double_t &f,Double_t *u, Int_t /*flag*/)
{
   TVirtualFitter *fitter = TVirtualFitter::GetFitter();
   TF3 *f3 = (TF3*)fitter->GetObjectFit();
   f3->InitArgs(u, f3->GetParameters() );
   f = f3->EvalPar(u);
}
