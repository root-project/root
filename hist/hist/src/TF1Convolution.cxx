// @(#)root/hist:$Id$
// Authors: L. Moneta, A. Flandi   08/2014
//
/**********************************************************************
 *                                                                    *
 * Copyright (c) 2015  ROOT  Team, CERN/PH-SFT                        *
 *                                                                    *
 *                                                                    *
 **********************************************************************/
//
//  TF1Convolution.cxx
//  
//
//  Created by Aurélie Flandi on 27.08.14.
//
//
//

#include "TF1Convolution.h"
#include "Riostream.h"
#include "TROOT.h"
#include "TObject.h"
#include "TObjString.h"
#include "TMath.h"
#include "Math/Integrator.h"
#include "Math/IntegratorMultiDim.h"
#include "Math/IntegratorOptions.h"
#include "Math/GaussIntegrator.h"
#include "Math/GaussLegendreIntegrator.h"
#include "Math/AdaptiveIntegratorMultiDim.h"
#include "Math/Functor.h"
#include "TVirtualFFT.h"
#include "TClass.h"

//________________________________________________________________________
//***********************************************************************
// (f*g)(t) = int(f(x)g(x-t)dx)   *
//*********************************
// class wrapping convolution of two function : evaluation of TF1(t) * TF1(x-t)
//
// The convolution is performed by default using FFTW if it is available .
// One can pass optionally the range of the convolution (by default the first function range is used).
// Note that when using Discrete Fouriere Transform (as FFTW), it is a circular transform, so the functions should be
// approximatly zero at the end of the range. If they are significantly different than zero on one side (e.g. the left side)
// a spill over will occur on the other side (e.g right side).
// If no function range is given by default the function1 range + 10% is used 
// One shoud use also a not too small number of points for the DFT (a minimum of 1000).  By default 10000 points are used . 
//
class TF1Convolution_EvalWrapper
{
   std::shared_ptr < TF1 > fFunction1;
   std::shared_ptr < TF1 > fFunction2;
   Double_t fT0;
   
public:

   TF1Convolution_EvalWrapper(std::shared_ptr<TF1> & f1 , std::shared_ptr<TF1> & f2, Double_t t)
      : fFunction1(f1), fFunction2(f2), fT0(t)
   {
   }
   Double_t operator()(Double_t x) const
   {
      return fFunction1->Eval(x) * fFunction2->Eval(x-fT0);
   }
};

//________________________________________________________________________
void TF1Convolution::InitializeDataMembers(TF1* function1, TF1* function2, Bool_t useFFT)
{
   // use copy instead of Clone
   if (function1) { 
      TF1 * fnew1 = (TF1*) function1->IsA()->New();
      function1->Copy(*fnew1); 
      fFunction1  = std::shared_ptr<TF1>(fnew1);
   }
   if (function2) { 
      TF1 * fnew2 = (TF1*) function2->IsA()->New();
      function2->Copy(*fnew2); 
      fFunction2  = std::shared_ptr<TF1>(fnew2);
   }
   if (fFunction1.get() == nullptr|| fFunction2.get() == nullptr)
      Fatal("InitializeDataMembers","Invalid functions - Abort");

   // add by default an extra 10% on  each side
   fFunction1->GetRange(fXmin, fXmax);
   Double_t range = fXmax - fXmin; 
   fXmin       -= 0.1*range;
   fXmax       += 0.1*range; 
   fNofParams1 = fFunction1->GetNpar();
   fNofParams2 = fFunction2->GetNpar();
   fParams1    = std::vector<Double_t>(fNofParams1);
   fParams2    = std::vector<Double_t>(fNofParams2);
   fCstIndex   = fFunction2-> GetParNumber("Constant");
   fFlagFFT    = useFFT;
   fFlagGraph  = false;
   fNofPoints  = 10000;
   
   //std::cout<<"before: NofParams2 = "<<fNofParams2<<std::endl;

   fParNames.reserve( fNofParams1 + fNofParams2);
   for (int i=0; i<fNofParams1; i++)
   {
      fParams1[i] = fFunction1 -> GetParameter(i);
      fParNames.push_back(fFunction1 -> GetParName(i) );
   }
   for (int i=0; i<fNofParams2; i++)
   {
      fParams2[i] = fFunction2 -> GetParameter(i);
      if (i != fCstIndex) fParNames.push_back(fFunction2 -> GetParName(i) );
   }
   if (fCstIndex!=-1)
   {
      fFunction2  -> FixParameter(fCstIndex,1.);
      fNofParams2 =  fNofParams2-1;
      fParams2.erase(fParams2.begin()+fCstIndex);
   }
}
//________________________________________________________________________
TF1Convolution::TF1Convolution(TF1* function1, TF1* function2, Bool_t useFFT)
{
   // constructor from the two function pointer and a flag is using FFT
   InitializeDataMembers(function1,function2, useFFT);   
}

//________________________________________________________________________
TF1Convolution::TF1Convolution(TF1* function1, TF1* function2, Double_t xmin, Double_t xmax, Bool_t useFFT)
{
   // constructor from the two function pointer and the convolution range
   InitializeDataMembers(function1, function2,useFFT);
   if (xmin < xmax) { 
      fXmin      = xmin;
      fXmax      = xmax;
   }
}

//________________________________________________________________________
TF1Convolution::TF1Convolution(TString formula,  Double_t xmin, Double_t xmax, Bool_t useFFT)
{
   // constructor from a formula expression as f1 * f2 where f1 and f2 are two functions known to ROOT
   TF1::InitStandardFunctions();
   
   TObjArray *objarray   = formula.Tokenize("*");
   std::vector < TString > stringarray(2);
   std::vector < TF1*    > funcarray(2);
   for (int i=0; i<2; i++)
   {
      stringarray[i] = ((TObjString*)((*objarray)[i])) -> GetString();
      stringarray[i].ReplaceAll(" ","");
      funcarray[i]   = (TF1*)(gROOT -> GetListOfFunctions() -> FindObject(stringarray[i]));
      // case function is not found try to use as a TFormula
      if (funcarray[i] == nullptr) {
         TF1 * f = new TF1(TString::Format("f_conv_%d",i+1),stringarray[i]);
         if (!f->GetFormula()->IsValid() )
            Error("TF1Convolution","Invalid formula : %s",stringarray[i].Data() );
         if (i == 0)
            fFunction1 = std::shared_ptr<TF1>(f);
         else
            fFunction2 = std::shared_ptr<TF1>(f);
      }
   }
   InitializeDataMembers(funcarray[0], funcarray[1],useFFT);
   if (xmin < xmax) {
      fXmin      = xmin;
      fXmax      = xmax;
   }
}

//________________________________________________________________________
TF1Convolution::TF1Convolution(TString formula1, TString formula2,  Double_t xmin, Double_t xmax, Bool_t useFFT)
{
   // constructor from 2 function names where f1 and f2 are two functions known to ROOT
   // if the function names are not knwon to ROOT then a corresponding 
   TF1::InitStandardFunctions();
   (TString)formula1.ReplaceAll(" ","");
   (TString)formula2.ReplaceAll(" ","");
   TF1* f1 = (TF1*)(gROOT -> GetListOfFunctions() -> FindObject(formula1));
   TF1* f2 = (TF1*)(gROOT -> GetListOfFunctions() -> FindObject(formula2));
   // if function do not exists try using TFormula
   if (!f1) {
      fFunction1 = std::shared_ptr<TF1>(new TF1("f_conv_1",formula1) );
      if (!fFunction1->GetFormula()->IsValid() )
         Error("TF1Convolution","Invalid formula for : %s",formula1.Data() );
   }
   if (!f2) {
      fFunction2 = std::shared_ptr<TF1>(new TF1("f_conv_1",formula2) );
      if (!fFunction2->GetFormula()->IsValid() )
         Error("TF1Convolution","Invalid formula for : %s",formula2.Data() );
   }
   // if f1 or f2 are null ptr are not used in InitializeDataMembers
   InitializeDataMembers(f1, f2,useFFT);
   if (xmin < xmax) {
      fXmin      = xmin;
      fXmax      = xmax;
   }
}

//________________________________________________________________________
void TF1Convolution::MakeFFTConv()
{
   //perform the FFT of the two functions

   if (gDebug)
      Info("MakeFFTConv","Making FFT convolution using %d points in range [%g,%g]",fNofPoints,fXmin,fXmax);
   
   std::vector < Double_t > x  (fNofPoints);
   std::vector < Double_t > in1(fNofPoints);
   std::vector < Double_t > in2(fNofPoints);
   
   TVirtualFFT *fft1 = TVirtualFFT::FFT(1, &fNofPoints, "R2C K");
   TVirtualFFT *fft2 = TVirtualFFT::FFT(1, &fNofPoints, "R2C K");
   if (fft1 == nullptr || fft2 == nullptr) {
      Warning("MakeFFTConv","Cannot use FFT, probably FFTW# package is not available. Switch to numerical convolution");
      fFlagFFT = false; 
   }

   // apply a shift in order to have the second function centered around middle of the range of the convolution
   Double_t shift2 = 0.5*(fXmin+fXmax);
   Double_t x2; 
   for (int i=0; i<fNofPoints; i++)
   {
      x[i]   = fXmin + (fXmax-fXmin)/(fNofPoints-1)*i;
      x2     = x[i] - shift2; 
      in1[i] = fFunction1 -> EvalPar( &x[i], nullptr);
      in2[i] = fFunction2 -> EvalPar( &x2, nullptr);
      fft1  -> SetPoint(i, in1[i]);
      fft2  -> SetPoint(i, in2[i]);
   }
   fft1 -> Transform();
   fft2 -> Transform();
   
   //inverse transformation of the product
   
   TVirtualFFT *fftinverse = TVirtualFFT::FFT(1, &fNofPoints, "C2R K");
   Double_t re1, re2, im1, im2, out_re, out_im;
   
   for (int i=0;i<=fNofPoints/2.;i++)
   {
      fft1 -> GetPointComplex(i,re1,im1);
      fft2 -> GetPointComplex(i,re2,im2);
      out_re = re1*re2 - im1*im2;
      out_im = re1*im2 + re2*im1;
      fftinverse -> SetPoint(i, out_re, out_im);
   }
   fftinverse -> Transform();

   // fill a graph with the result of the convolution
   if (!fGraphConv) fGraphConv  = std::shared_ptr< TGraph >(new TGraph(fNofPoints));

   for (int i=0;i<fNofPoints;i++)
   {
      // we need this since we have applied a shift in the middle of f2
      int j = i + fNofPoints/2;
      if (j >= fNofPoints) j -= fNofPoints; 
      fGraphConv->SetPoint(i, x[i], fftinverse->GetPointReal(j)/fNofPoints);//because it is not normalized
   }
   fFlagGraph = true; // we can use the graph
}

//________________________________________________________________________
Double_t TF1Convolution::EvalFFTConv(Double_t t)
{
   if (!fFlagGraph)  MakeFFTConv();
   // if cannot make FFT use numconv
   if (fGraphConv)
      return  fGraphConv -> Eval(t);
   else
      return EvalNumConv(t); 
}

//________________________________________________________________________
Double_t TF1Convolution::EvalNumConv(Double_t t)
{
   // perform numerical convolution
   // could in principle cache the integral  in a Grap[h as it is done for the FFTW
   TF1Convolution_EvalWrapper fconv( fFunction1, fFunction2, t);
   Double_t result = 0;
   
   if (ROOT::Math::IntegratorOneDimOptions::DefaultIntegratorType() == ROOT::Math::IntegrationOneDim::kGAUSS )
   {
      ROOT::Math::GaussIntegrator integrator(1e-9, 1e-9);
      integrator.SetFunction(ROOT::Math::Functor1D(fconv));
      if      (fXmin != - TMath::Infinity() && fXmax != TMath::Infinity())
         result =  integrator.Integral(fXmin, fXmax);
      else if (fXmin == - TMath::Infinity() && fXmax != TMath::Infinity())
         result = integrator.IntegralLow(fXmax);
      else if (fXmin != - TMath::Infinity() && fXmax == TMath::Infinity())
         result = integrator.IntegralUp(fXmin);
      else if (fXmin == - TMath::Infinity() && fXmax == TMath::Infinity())
         result = integrator.Integral();
   }
   else
   {
      ROOT::Math::IntegratorOneDim integrator(fconv, ROOT::Math::IntegratorOneDimOptions::DefaultIntegratorType(), 1e-9, 1e-9);
      if      (fXmin != - TMath::Infinity() && fXmax != TMath::Infinity() )
         result =  integrator.Integral(fXmin, fXmax);
      else if (fXmin == - TMath::Infinity() && fXmax != TMath::Infinity() )
         result = integrator.IntegralLow(fXmax);
      else if (fXmin != - TMath::Infinity() && fXmax == TMath::Infinity() )
         result = integrator.IntegralUp(fXmin);
      else if (fXmin == - TMath::Infinity() && fXmax == TMath::Infinity() )
         result = integrator.Integral();
   }
   return result;
}

//________________________________________________________________________
Double_t TF1Convolution::operator()(Double_t* t, Double_t* p)//used in TF1 when doing the fit, will be valuated at each point
{
   if (p!=0)   TF1Convolution::SetParameters(p);                           // first refresh the parameters
  
   Double_t result = 0.;
   if (fFlagFFT)  result = EvalFFTConv(t[0]);
   else           result = EvalNumConv(t[0]);
   return result;
}
//________________________________________________________________________
void TF1Convolution::SetNofPointsFFT(Int_t n)
{
   if (n<0) return;
   fNofPoints = n;
   if (fGraphConv) fGraphConv -> Set(fNofPoints); //set nof points of the Tgraph
   fFlagGraph = false; // to indicate we need to re-do the graph
}

//________________________________________________________________________
void TF1Convolution::SetParameters(Double_t* p)
{
   bool equalParams = true;
   for (int i=0; i<fNofParams1; i++) {
      fFunction1 -> SetParameter(i,p[i]);
      equalParams &= ( fParams1[i] == p[i] );   
      fParams1[i] = p[i];      
   }
   Int_t k       = 0;
   Int_t offset  = 0;
   Int_t offset2 = 0;
   if (fCstIndex!=-1)   offset = 1;
   Int_t totalnofparams = fNofParams1+fNofParams2+offset;
   for (int i=fNofParams1; i<totalnofparams; i++)   {
      if (k==fCstIndex)
      {
         k++;
         offset2=1;
         continue;
      }
      fFunction2 -> SetParameter(k,p[i-offset2]);
      equalParams &= ( fParams2[k-offset2] == p[i-offset2] );   
      fParams2[k-offset2] = p[i-offset2];
      k++;
   }
   // std::cout << "parameters for function1   :  ";
   // for (int i = 0; i < fFunction1->GetNpar(); ++i)
   //    std::cout << fFunction1->GetParameter(i) << "  ";
   // std::cout << "\nparameters for function2   :  ";
   // for (int i = 0; i < fFunction2->GetNpar(); ++i)
   //    std::cout << fFunction2->GetParameter(i) << "  ";
   // std::cout << std::endl;

//do the graph for FFT convolution
   if (!equalParams) fFlagGraph = false; // to indicate we need to re-do the convolution
   // if (fFlagFFT)
   // {
   //    MakeFFTConv();
   // }
}

//________________________________________________________________________
void TF1Convolution::SetParameters(Double_t p0, Double_t p1, Double_t p2, Double_t p3,
                                   Double_t p4, Double_t p5, Double_t p6, Double_t p7)
{
   Double_t params[]={p0,p1,p2,p3,p4,p5,p6,p7};
   TF1Convolution::SetParameters(params);
}

//________________________________________________________________________
void TF1Convolution::SetExtraRange(Double_t percentage)
{
   if (percentage<0) return;
   double range = fXmax = fXmin; 
   fXmin -= percentage * range; 
   fXmax += percentage * range;
   fFlagGraph = false;  // to indicate we need to re-do the convolution
}

//________________________________________________________________________
void TF1Convolution::SetRange(Double_t a, Double_t b)
{
   if (a>=b)   return;
   fXmin = a;
   fXmax = b; 
   if (fFlagFFT && ( a==-TMath::Infinity() || b==TMath::Infinity() ) )
   {
      Warning("TF1Convolution::SetRange()","In FFT mode, range can not be infinite. Infinity has been replaced by range of first function plus a bufferzone to avoid spillover.");
      if (a ==-TMath::Infinity()) fXmin = fFunction1 -> GetXmin();
      if ( b== TMath::Infinity()) fXmax = fFunction1 -> GetXmax();
      // add a spill over of 10% in this case
      SetExtraRange(0.1); 
   }
   fFlagGraph = false;  // to indicate we need to re-do the convolution
}
