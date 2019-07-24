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

////////////////////////////////////////////////////////////////////////////////
/// \class TFitter
///
/// The ROOT standard fitter based on TMinuit
///
////////////////////////////////////////////////////////////////////////////////

// extern void H1FitChisquare(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);
// extern void H1FitLikelihood(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);
// extern void GraphFitChisquare(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);
// extern void Graph2DFitChisquare(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);
// extern void MultiGraphFitChisquare(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);
extern void F2Fit(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);
extern void F3Fit(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);

ClassImp(TFitter);

////////////////////////////////////////////////////////////////////////////////
/// Default constructor

TFitter::TFitter(Int_t maxpar)
{
   fMinuit = new TMinuit(maxpar);
   fNlog = 0;
   fSumLog = 0;
   fCovar = 0;
   SetName("MinuitFitter");
}

////////////////////////////////////////////////////////////////////////////////
/// Default destructor

TFitter::~TFitter()
{
   if (fCovar)  delete [] fCovar;
   if (fSumLog) delete [] fSumLog;
   delete fMinuit;
}

// ////////////////////////////////////////////////////////////////////////////////
// /// return a chisquare equivalent

Double_t TFitter::Chisquare(Int_t , Double_t * )  const
{
    Error("Chisquare","This function is deprecated - use ROOT::Fit::Chisquare class");
    //Double_t amin = 0;
    //H1FitChisquare(npar,params,amin,params,1);
    return TMath::QuietNaN();
}

////////////////////////////////////////////////////////////////////////////////
/// reset the fitter environment

void TFitter::Clear(Option_t *)
{
   if (fCovar)  {delete [] fCovar; fCovar = 0;}
   fMinuit->mncler();

   //reset the internal Minuit random generator to its initial state
   Double_t val = 3;
   Int_t inseed = 12345;
   fMinuit->mnrn15(val,inseed);
}

////////////////////////////////////////////////////////////////////////////////
/// Execute a fitter command;
///   command : command string
///   args    : list of nargs command arguments

Int_t TFitter::ExecuteCommand(const char *command, Double_t *args, Int_t nargs)
{
   if (fCovar)  {delete [] fCovar; fCovar = 0;}
   Int_t ierr = 0;
   fMinuit->mnexcm(command,args,nargs,ierr);
   return ierr;
}

////////////////////////////////////////////////////////////////////////////////
/// Fix parameter ipar.

void TFitter::FixParameter(Int_t ipar)
{
   if (fCovar)  {delete [] fCovar; fCovar = 0;}
   fMinuit->FixParameter(ipar);
}

////////////////////////////////////////////////////////////////////////////////
///Computes point-by-point confidence intervals for the fitted function
///
///Parameters:
/// - n - number of points
/// - ndim - dimensions of points
/// - x - points, at which to compute the intervals, for ndim > 1
///    should be in order: (x0,y0, x1, y1, ... xn, yn)
/// - ci - computed intervals are returned in this array
/// - cl - confidence level, default=0.95
///
///NOTE, that the intervals are approximate for nonlinear(in parameters) models

void TFitter::GetConfidenceIntervals(Int_t n, Int_t ndim, const Double_t *x, Double_t *ci, Double_t cl)
{
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
   if (!matr){
      delete [] grad;
      delete [] sum_vector;
      if (fixed)
         delete [] fixed;
      return;
   }

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

////////////////////////////////////////////////////////////////////////////////
///Computes confidence intervals at level cl. Default is 0.95
///
///The TObject parameter can be a TGraphErrors, a TGraph2DErrors or a TH1,2,3.
///For Graphs, confidence intervals are computed for each point,
///the value of the graph at that point is set to the function value at that
///point, and the graph y-errors (or z-errors) are set to the value of
///the confidence interval at that point.
///For Histograms, confidence intervals are computed for each bin center
///The bin content of this bin is then set to the function value at the bin
///center, and the bin error is set to the confidence interval value.
///NOTE: confidence intervals are approximate for nonlinear models!
///
///Allowed combinations:
///
/// - Fitted object               Passed object
/// - TGraph                      TGraphErrors, TH1
/// - TGraphErrors, AsymmErrors   TGraphErrors, TH1
/// - TH1                         TGraphErrors, TH1
/// - TGraph2D                    TGraph2DErrors, TH2
/// - TGraph2DErrors              TGraph2DErrors, TH2
/// - TH2                         TGraph2DErrors, TH2
/// - TH3                         TH3

void TFitter::GetConfidenceIntervals(TObject *obj, Double_t cl)
{
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

////////////////////////////////////////////////////////////////////////////////
/// return a pointer to the covariance matrix

Double_t *TFitter::GetCovarianceMatrix() const
{
   if (fCovar) return fCovar;
   Int_t npars = fMinuit->GetNumPars();
   ((TFitter*)this)->fCovar = new Double_t[npars*npars];
   fMinuit->mnemat(fCovar,npars);
   return fCovar;
}

////////////////////////////////////////////////////////////////////////////////
/// return element i,j from the covariance matrix

Double_t TFitter::GetCovarianceMatrixElement(Int_t i, Int_t j) const
{
   GetCovarianceMatrix();
   Int_t npars = fMinuit->GetNumPars();
   if (i < 0 || i >= npars || j < 0 || j >= npars) {
      Error("GetCovarianceMatrixElement","Illegal arguments i=%d, j=%d",i,j);
      return 0;
   }
   return fCovar[j+npars*i];
}

////////////////////////////////////////////////////////////////////////////////
/// return current errors for a parameter
///   ipar     : parameter number
///   eplus    : upper error
///   eminus   : lower error
///   eparab   : parabolic error
///   globcc   : global correlation coefficient

Int_t TFitter::GetErrors(Int_t ipar,Double_t &eplus, Double_t &eminus, Double_t &eparab, Double_t &globcc) const
{

   Int_t ierr = 0;
   fMinuit->mnerrs(ipar, eplus,eminus,eparab,globcc);
   return ierr;
}


////////////////////////////////////////////////////////////////////////////////
/// return the total number of parameters (free + fixed)

Int_t TFitter::GetNumberTotalParameters() const
{
   return fMinuit->fNpar + fMinuit->fNpfix;
}

////////////////////////////////////////////////////////////////////////////////
/// return the number of free parameters

Int_t TFitter::GetNumberFreeParameters() const
{
   return fMinuit->fNpar;
}


////////////////////////////////////////////////////////////////////////////////
/// return error of parameter ipar

Double_t TFitter::GetParError(Int_t ipar) const
{
   Int_t ierr = 0;
   TString pname;
   Double_t value,verr,vlow,vhigh;

   fMinuit->mnpout(ipar, pname,value,verr,vlow,vhigh,ierr);
   return verr;
}


////////////////////////////////////////////////////////////////////////////////
/// return current value of parameter ipar

Double_t TFitter::GetParameter(Int_t ipar) const
{
   Int_t ierr = 0;
   TString pname;
   Double_t value,verr,vlow,vhigh;

   fMinuit->mnpout(ipar, pname,value,verr,vlow,vhigh,ierr);
   return value;
}

////////////////////////////////////////////////////////////////////////////////
/// return current values for a parameter
///
///  - ipar     : parameter number
///  - parname  : parameter name
///  - value    : initial parameter value
///  - verr     : initial error for this parameter
///  - vlow     : lower value for the parameter
///  - vhigh    : upper value for the parameter
///
///  WARNING! parname must be suitably dimensionned in the calling function.

Int_t TFitter::GetParameter(Int_t ipar, char *parname,Double_t &value,Double_t &verr,Double_t &vlow, Double_t &vhigh) const
{
   Int_t ierr = 0;
   TString pname;
   fMinuit->mnpout(ipar, pname,value,verr,vlow,vhigh,ierr);
   strcpy(parname,pname.Data());
   return ierr;
}

////////////////////////////////////////////////////////////////////////////////
/// return name of parameter ipar

const char *TFitter::GetParName(Int_t ipar) const
{
   if (!fMinuit || ipar < 0 || ipar > fMinuit->fNu) return "";
   return fMinuit->fCpnam[ipar];
}

////////////////////////////////////////////////////////////////////////////////
/// return global fit parameters
///
///  - amin     : chisquare
///  - edm      : estimated distance to minimum
///  - errdef
///  - nvpar    : number of variable parameters
///  - nparx    : total number of parameters

Int_t TFitter::GetStats(Double_t &amin, Double_t &edm, Double_t &errdef, Int_t &nvpar, Int_t &nparx) const
{
   Int_t ierr = 0;
   fMinuit->mnstat(amin,edm,errdef,nvpar,nparx,ierr);
   return ierr;
}

////////////////////////////////////////////////////////////////////////////////
/// return Sum(log(i) i=0,n
/// used by log likelihood fits

Double_t TFitter::GetSumLog(Int_t n)
{
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


////////////////////////////////////////////////////////////////////////////////
///return kTRUE if parameter ipar is fixed, kFALSE othersise)

Bool_t TFitter::IsFixed(Int_t ipar) const
{
   if (fMinuit->fNiofex[ipar] == 0 ) return kTRUE;
   return kFALSE;
}


////////////////////////////////////////////////////////////////////////////////
/// Print fit results

void  TFitter::PrintResults(Int_t level, Double_t amin) const
{
   fMinuit->mnprin(level,amin);
}

////////////////////////////////////////////////////////////////////////////////
/// Release parameter ipar.

void TFitter::ReleaseParameter(Int_t ipar)
{
   if (fCovar)  {delete [] fCovar; fCovar = 0;}
   fMinuit->Release(ipar);
}

////////////////////////////////////////////////////////////////////////////////
/// Specify the address of the fitting algorithm

void TFitter::SetFCN(void (*fcn)(Int_t &, Double_t *, Double_t &f, Double_t *, Int_t))
{
   if (fCovar)  {delete [] fCovar; fCovar = 0;}
   TVirtualFitter::SetFCN(fcn);
   fMinuit->SetFCN(fcn);
}

////////////////////////////////////////////////////////////////////////////////
/// ret fit method (chisquare or loglikelihood)

void TFitter::SetFitMethod(const char *name)
{
   if (fCovar)  {delete [] fCovar; fCovar = 0;}
   // if (!strcmp(name,"H1FitChisquare"))    SetFCN(H1FitChisquare);
   // if (!strcmp(name,"H1FitLikelihood"))   SetFCN(H1FitLikelihood);
   // if (!strcmp(name,"GraphFitChisquare")) SetFCN(GraphFitChisquare);
   // if (!strcmp(name,"Graph2DFitChisquare")) SetFCN(Graph2DFitChisquare);
   // if (!strcmp(name,"MultiGraphFitChisquare")) SetFCN(MultiGraphFitChisquare);
   if (!strcmp(name,"F2Minimizer")) SetFCN(F2Fit);
   if (!strcmp(name,"F3Minimizer")) SetFCN(F3Fit);
}

////////////////////////////////////////////////////////////////////////////////
/// set initial values for a parameter
///
///  - ipar     : parameter number
///  - parname  : parameter name
///  - value    : initial parameter value
///  - verr     : initial error for this parameter
///  - vlow     : lower value for the parameter
///  - vhigh    : upper value for the parameter

Int_t TFitter::SetParameter(Int_t ipar,const char *parname,Double_t value,Double_t verr,Double_t vlow, Double_t vhigh)
{
   if (fCovar)  {delete [] fCovar; fCovar = 0;}
   Int_t ierr = 0;
   fMinuit->mnparm(ipar,parname,value,verr,vlow,vhigh,ierr);
   return ierr;
}



////////////////////////////////////////////////////////////////////////////////

void F2Fit(Int_t &/*npar*/, Double_t * /*gin*/, Double_t &f,Double_t *u, Int_t /*flag*/)
{
   TVirtualFitter *fitter = TVirtualFitter::GetFitter();
   TF2 *f2 = (TF2*)fitter->GetObjectFit();
   f2->InitArgs(u, f2->GetParameters() );
   f = f2->EvalPar(u);
}

////////////////////////////////////////////////////////////////////////////////

void F3Fit(Int_t &/*npar*/, Double_t * /*gin*/, Double_t &f,Double_t *u, Int_t /*flag*/)
{
   TVirtualFitter *fitter = TVirtualFitter::GetFitter();
   TF3 *f3 = (TF3*)fitter->GetObjectFit();
   f3->InitArgs(u, f3->GetParameters() );
   f = f3->EvalPar(u);
}
