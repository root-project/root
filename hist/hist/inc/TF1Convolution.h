// @(#)root/hist:$Id$
// Author: L. Moneta May 2015

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2015  ROOT  Team, CERN/PH-SFT                        *
 *                                                                    *
 *                                                                    *
 **********************************************************************/
//
//
//  TF1Convolution.h
//
//  Created by Aurélie Flandi on 27.08.14.
//
//

#ifndef ROOT_TF1Convolution__
#define ROOT_TF1Convolution__

#include <iostream>
#include "TF1.h"
#include "TGraph.h"
#include <memory>

class TF1Convolution
{
   std::shared_ptr <TF1>      fFunction1;
   std::shared_ptr <TF1>      fFunction2;
   std::shared_ptr <TGraph>   fGraphConv;
   
   std::vector < Double_t >   fParams1;
   std::vector < Double_t >   fParams2;

   std::vector< TString >    fParNames;      // parameter names 
   
   Double_t fXmin;            //minimal bound of the range of the convolution
   Double_t fXmax;            //maximal bound of the range of the convolution
   Int_t    fNofParams1;
   Int_t    fNofParams2;
   Int_t    fCstIndex;        //index of the constant parameter f the first function
   Int_t    fNofPoints;       //number of point for FFT array
   Bool_t   fFlagFFT;         //choose fft or numerical convolution
   Bool_t   fFlagGraph; //tells if the graph is already done or not
   
   
   Double_t EvalNumConv(Double_t t);
   Double_t EvalFFTConv(Double_t t);
   void     InitializeDataMembers(TF1* function1, TF1* function2, Bool_t useFFT);
   void     MakeFFTConv();
   
   public:
   
   TF1Convolution(TF1* function1, TF1* function2, Bool_t useFFT = true);
   TF1Convolution(TF1* function1, TF1* function2, Double_t xmin, Double_t xmax, Bool_t useFFT = true);
   TF1Convolution(TString formula, Double_t xmin = 1., Double_t xmax = 0., Bool_t useFFT = true);
   TF1Convolution(TString formula1, TString formula2, Double_t xmin = 1., Double_t xmax = 0., Bool_t useFFT = true);
   
   void     SetParameters(Double_t* p);
   void     SetParameters(Double_t p0,    Double_t p1,    Double_t p2=0., Double_t p3=0.,
                          Double_t p4=0., Double_t p5=0., Double_t p6=0., Double_t p7=0.);
   void     SetRange(Double_t a, Double_t b);
   void     SetExtraRange(Double_t percentage);
   void     SetNofPointsFFT(Int_t n);
   void     SetNumConv(Bool_t flag = true){fFlagFFT=!flag;}
   
   Int_t    GetNpar() const {return (fNofParams1+fNofParams2);}
   Double_t GetXmin() const {return fXmin;}
   Double_t GetXmax() const {return fXmax;}
   const char *    GetParName(Int_t ipar) const { return fParNames.at(ipar).Data(); }
   
   Double_t operator()(Double_t* t, Double_t* p);
};


#endif
