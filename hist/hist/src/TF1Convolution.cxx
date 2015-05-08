//
//  TF1Convolution.cpp
//  
//
//  Created by Aurélie Flandi on 27.08.14.
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
//#include <fftw3.h>

//*********************************
// (f*g)(t) = int(f(x)g(x-t)dx)   *
//*********************************
//________________________________________________________________________
// class wrapping evaluation of TF1(t) * TF1(x-t)
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
void TF1Convolution::InitializeDataMembers(TF1* function1, TF1* function2)
{
   std::shared_ptr < TF1 > f1((TF1*)function1->Clone());
   std::shared_ptr < TF1 > f2((TF1*)function2->Clone());
   fFunction1  = f1;
   fFunction2  = f2;
   fXmin       = fFunction1->GetXmin()-std::abs(fFunction1->GetXmin())*0.2;
   fXmax       = fFunction1->GetXmax()+std::abs(fFunction1->GetXmax())*0.2;
   fNofParams1 = f1->GetNpar();
   fNofParams2 = f2->GetNpar();
   fParams1    = std::vector<Double_t>(fNofParams1);
   fParams2    = std::vector<Double_t>(fNofParams2);
   fCstIndex   = fFunction2-> GetParNumber("Constant");
   fFlagFFT    = true;
   fFlagGraph  = false;
   fNofPoints  = 10000;
   fGraphConv  = std::shared_ptr< TGraph >(new TGraph(fNofPoints));
   
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
TF1Convolution::TF1Convolution(TF1* function1, TF1* function2)
{
   InitializeDataMembers(function1,function2);
}

//________________________________________________________________________
TF1Convolution::TF1Convolution(TF1* function1, TF1* function2, Double_t xmin, Double_t xmax)
{
   InitializeDataMembers(function1, function2);
   fXmin      = xmin;
   fXmax      = xmax;
}

//________________________________________________________________________
TF1Convolution::TF1Convolution(TString formula)
{
   TF1::InitStandardFunctions();
   
   TObjArray *objarray   = formula.Tokenize("*");
   std::vector < TString > stringarray(2);
   std::vector < TF1*    > funcarray(2);
   for (int i=0; i<2; i++)
   {
      stringarray[i] = ((TObjString*)((*objarray)[i])) -> GetString();
      stringarray[i].ReplaceAll(" ","");
      funcarray[i]   = (TF1*)(gROOT -> GetListOfFunctions() -> FindObject(stringarray[i]));
   }
   InitializeDataMembers(funcarray[0], funcarray[1]);
}

//________________________________________________________________________
TF1Convolution::TF1Convolution(TString formula1, TString formula2)
{
   TF1::InitStandardFunctions();
   (TString)formula1.ReplaceAll(" ","");
   (TString)formula2.ReplaceAll(" ","");
   TF1* f1 = (TF1*)(gROOT -> GetListOfFunctions() -> FindObject(formula1));
   TF1* f2 = (TF1*)(gROOT -> GetListOfFunctions() -> FindObject(formula2));
   InitializeDataMembers(f1, f2);
}

//________________________________________________________________________
void TF1Convolution::MakeGraphConv()
{
   //FFT of the two functions
   //std::cout<<"is called"<<std::endl;
   std::vector < Double_t > x  (fNofPoints);
   std::vector < Double_t > in1(fNofPoints);
   std::vector < Double_t > in2(fNofPoints);
   
   TVirtualFFT *fft1 = TVirtualFFT::FFT(1, &fNofPoints, "R2C K");
   TVirtualFFT *fft2 = TVirtualFFT::FFT(1, &fNofPoints, "R2C K");
   for (int i=0; i<fNofPoints; i++)
   {
      x[i]   = fXmin + (fXmax-fXmin)/(fNofPoints-1)*i;
      in1[i] = fFunction1 -> Eval(x[i]);
      in2[i] = fFunction2 -> Eval(x[i]);
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

   for (int i=0;i<fNofPoints;i++)
   {
     fGraphConv->SetPoint(i, x[i], fftinverse->GetPointReal(i)/fNofPoints);//because not normalized
   }
}

//________________________________________________________________________
Double_t TF1Convolution::MakeFFTConv(Double_t t)
{
   if (!fFlagGraph)  MakeGraphConv();
   return  fGraphConv -> Eval(t);
}

//________________________________________________________________________
Double_t TF1Convolution::MakeNumConv(Double_t t)
{
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
   if (fFlagFFT)  result = MakeFFTConv(t[0]);
   else           result = MakeNumConv(t[0]);
   return result;
}
//________________________________________________________________________
void TF1Convolution::SetNofPointsFFT(Int_t n)
{
   if (n<0) return;
   fNofPoints = n;
   fGraphConv -> Set(fNofPoints); //set nof points of the Tgraph
}

//________________________________________________________________________
void TF1Convolution::SetParameters(Double_t* p)
{
   for (int i=0; i<fNofParams1; i++)
   {
      fFunction1 -> SetParameter(i,p[i]);
      fParams1[i] = p[i];
   }
   Int_t k       = 0;
   Int_t offset  = 0;
   Int_t offset2 = 0;
   if (fCstIndex!=-1)   offset = 1;
   Int_t totalnofparams = fNofParams1+fNofParams2+offset;
   for (int i=fNofParams1; i<totalnofparams; i++)
   {
      if (k==fCstIndex)
      {
         k++;
         offset2=1;
         continue;
      }
      fFunction2 -> SetParameter(k,p[i-offset2]);
      fParams2[k-offset2] = p[i-offset2];
      k++;
   }
   //do the graph for FFT convolution
   if (fFlagFFT)
   {
      MakeGraphConv();
      fFlagGraph = true;
   }
}

//________________________________________________________________________
void TF1Convolution::SetParameters(Double_t p0, Double_t p1, Double_t p2, Double_t p3,
                                   Double_t p4, Double_t p5, Double_t p6, Double_t p7)
{
   Double_t params[]={p0,p1,p2,p3,p4,p5,p6,p7};
   TF1Convolution::SetParameters(params);
}

//________________________________________________________________________
void TF1Convolution::SetRange(Double_t percentage)
{
   if (percentage<0) return;
   fXmin = fFunction1 -> GetXmin() - std::abs(fFunction1 -> GetXmin())*percentage;
   fXmax = fFunction1 -> GetXmax() + std::abs(fFunction1 -> GetXmax())*percentage;
}

//________________________________________________________________________
void TF1Convolution::SetRange(Double_t a, Double_t b)
{
   if (a>=b)   return;
   if (fFlagFFT && ( a==-TMath::Infinity() || b==TMath::Infinity() ) )
   {
      Error("TF1Convolution::SetRange()","In FFT mode, range can not be infinite. Infinity has been replaced by range of first function plus a bufferzone to avoid spillover.");
      if (a ==-TMath::Infinity()) fXmin = fFunction1 -> GetXmin() - std::abs(fFunction1 -> GetXmin())*0.2;
      if ( b== TMath::Infinity()) fXmax = fFunction1 -> GetXmax() + std::abs(fFunction1 -> GetXmax())*0.2;
   }
   fXmin = a;
   fXmax = b;
}
