// @(#)root/base:$Name:  $:$Id: TVirtualFFT.cxx,v 1.3 2006/05/11 10:31:41 brun Exp $
// Author: Anna Kreshuk  10/04/2006

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TVirtualFFT.h"
#include "TPluginManager.h"
#include "TEnv.h"
#include "Api.h"

TVirtualFFT *TVirtualFFT::fgFFT    = 0;
TString      TVirtualFFT::fgDefault   = "";

ClassImp(TVirtualFFT)

//_____________________________________________________________________________
TVirtualFFT::~TVirtualFFT()
{
   //destructor
   if (this==fgFFT)
      fgFFT = 0;
}

//_____________________________________________________________________________
TVirtualFFT* TVirtualFFT::FFT(Int_t ndim, Int_t *n, Option_t *option)
{
//Returns a pointer to the FFT of requested size and type.
//Parameters:
// -ndim : number of transform dimensions
// -n    : sizes of each dimension (an array at least ndim long)
// -option : consists of 2 parts - flag option and an option to create a new TVirtualFFT
//         1) transform type option: 
//           Available transform types are:
//           C2CForward, C2CBackward, C2R, R2C, R2HC, HC2R, DHT
//           see class description for details
//         2) flag option: choosing how much time should be spent in planning the transform:
//           Possible options:
//           "ES" (from "estimate") - no time in preparing the transform, 
//                                  but probably sub-optimal  performance
//           "M"  (from "measure")  - some time spend in finding the optimal way 
//                                  to do the transform
//           "P" (from "patient")   - more time spend in finding the optimal way 
//                                  to do the transform
//           "EX" (from "exhaustive") - the most optimal way is found
//           This option should be chosen depending on how many transforms of the 
//           same size and type are going to be done. 
//           Planning is only done once, for the first transform of this size and type.
//         3) option allowing to choose between the global fgFFT and a new TVirtualFFT object
//           ""  - default, changes and returns the global fgFFT variable
//           "K" (from "keep")- without touching the global fgFFT, 
//           creates and returns a new TVirtualFFT*. User is then responsible for deleting it.
// Examples of valid options: "R2C ES K", "C2CF M", "DHT P K", etc.


   Int_t inputtype=0, currenttype=0;
   TString opt = option;
   opt.ToUpper();
   //find the tranform flag
   Option_t *flag;
   if (opt.Contains("ES")) flag = "ES";
   if (opt.Contains("M"))  flag = "M"; 
   if (opt.Contains("P"))  flag = "P";
   if (opt.Contains("EX")) flag = "EX";
   else flag = "ES";
   Int_t ndiff = 0;

   if (!opt.Contains("K")) {
      if (fgFFT){
         //if the global transform exists, check if it should be changed
         if (fgFFT->GetNdim()!=ndim)
            ndiff++;
         else {
            Int_t *ncurrent = fgFFT->GetN();
            for (Int_t i=0; i<ndim; i++){
               if (n[i]!=ncurrent[i])
                  ndiff++;
            }
         }
         Option_t *t = fgFFT->GetType();
         if (!opt.Contains(t)) {
            if (opt.Contains("HC") || opt.Contains("DHT"))
               inputtype = 1;
            if (strcmp(t,"R2HC")==0 || strcmp(t,"HC2R")==0 || strcmp(t,"DHT")==0)
               currenttype=1;
            
            if (!(inputtype==1 && currenttype==1))
               ndiff++;
         }
         if (ndiff>0){
            delete fgFFT;
            fgFFT = 0;
         }
      }
   }
           
   Int_t sign = 0;
   if (opt.Contains("C2CB") || opt.Contains("C2R"))
      sign = 1; 
   if (opt.Contains("C2CF") || opt.Contains("R2C"))
      sign = -1; 
   
   TVirtualFFT *fft = 0;
   if (opt.Contains("K") || !fgFFT) {   
      TPluginHandler *h;
      TString pluginname;
      if (fgDefault.Length()==0) fgDefault="fftw";
      if (strcmp(fgDefault.Data(),"fftw")==0) {
         if (opt.Contains("C2C")) pluginname = "fftwc2c";
         if (opt.Contains("C2R")) pluginname = "fftwc2r";
         if (opt.Contains("R2C")) pluginname = "fftwr2c";
         if (opt.Contains("HC") || opt.Contains("DHT")) pluginname = "fftwr2r";
         if ((h=gROOT->GetPluginManager()->FindHandler("TVirtualFFT", pluginname))) {
            if (h->LoadPlugin()==-1) {
               printf("handler not found\n");
               return 0;
            }
            fft = (TVirtualFFT*)h->ExecPlugin(3, ndim, n, kFALSE);
            Int_t *kind = new Int_t[1];
            if (pluginname=="fftwr2r") { 
               if (opt.Contains("R2HC")) kind[0] = 10;
               if (opt.Contains("HC2R")) kind[0] = 11;
               if (opt.Contains("DHT")) kind[0] = 12;
            }
            fft->Init(flag, sign, kind);
            if (!opt.Contains("K")) {
               fgFFT = fft;
            }
            delete [] kind;
            return fft;
         }
         else {
            printf("plugin not found\n");
            return 0;
         }
      }
   } else {
      //if the global transform already exists and just needs to be reinitialised
      //with different parameters
      if (fgFFT->GetSign()!=sign || !opt.Contains(fgFFT->GetTransformFlag()) || !opt.Contains(fgFFT->GetType())) {
         Int_t *kind = new Int_t[1];
         if (inputtype==1) {
            if (opt.Contains("R2HC")) kind[0] = 10;
            if (opt.Contains("HC2R")) kind[0] = 11;
            if (opt.Contains("DHT")) kind[0] = 12;
         }
         fgFFT->Init(flag, sign, kind);
         delete [] kind;
      }
   }
   return fgFFT;
}

//_____________________________________________________________________________
TVirtualFFT* TVirtualFFT::SineCosine(Int_t ndim, Int_t *n, Int_t *r2rkind, Option_t *option)
{
//Returns a pointer to a sine or cosine transform of requested size and kind
//
//Parameters:
// -ndim    : number of transform dimensions
// -n       : sizes of each dimension (an array at least ndim long)
// -r2rkind : transform kind for each dimension
//     4 different kinds of sine and cosine transforms are available
//     DCT-I    - kind=0
//     DCT-II   - kind=1
//     DCT-III  - kind=2
//     DCT-IV   - kind=3
//     DST-I    - kind=4
//     DST-II   - kind=5
//     DST-III  - kind=6
//     DST-IV   - kind=7
// -option : consists of 2 parts - flag option and an option to create a new TVirtualFFT
//         - flag option: choosing how much time should be spent in planning the transform:
//           Possible options:
//           "ES" (from "estimate") - no time in preparing the transform, 
//                                  but probably sub-optimal  performance
//           "M"  (from "measure")  - some time spend in finding the optimal way 
//                                  to do the transform
//           "P" (from "patient")   - more time spend in finding the optimal way 
//                                  to do the transform
//           "EX" (from "exhaustive") - the most optimal way is found
//           This option should be chosen depending on how many transforms of the 
//           same size and type are going to be done. 
//           Planning is only done once, for the first transform of this size and type.
//         - option allowing to choose between the global fgFFT and a new TVirtualFFT object
//           ""  - default, changes and returns the global fgFFT variable
//           "K" (from "keep")- without touching the global fgFFT, 
//           creates and returns a new TVirtualFFT*. User is then responsible for deleting it.
// Examples of valid options: "ES K", "EX", etc

   TString opt = option;
   //find the tranform flag
   Option_t *flag;
   if (opt.Contains("ES")) flag = "ES";
   if (opt.Contains("M"))  flag = "M"; 
   if (opt.Contains("P"))  flag = "P";
   if (opt.Contains("EX")) flag = "EX";
   else flag = "ES";

   if (!opt.Contains("K")) {
      if (fgFFT){
         Int_t ndiff = 0;
         if (fgFFT->GetNdim()!=ndim || strcmp(fgFFT->GetType(),"R2R")!=0)
            ndiff++;
         else {
            Int_t *ncurrent = fgFFT->GetN();
            for (Int_t i=0; i<ndim; i++) {
               if (n[i] != ncurrent[i])
                  ndiff++;
            }
            
         }
         if (ndiff>0) {
            delete fgFFT;
            fgFFT = 0;
         }
      }
   }
   TVirtualFFT *fft = 0;
   if (!fgFFT || opt.Contains("K")) {   
      TPluginHandler *h;
      TString pluginname;
      if (fgDefault.Length()==0) fgDefault="fftw";
      if (strcmp(fgDefault.Data(),"fftw")==0) {
         pluginname = "fftwr2r";
         if ((h=gROOT->GetPluginManager()->FindHandler("TVirtualFFT", pluginname))) {
            if (h->LoadPlugin()==-1){
               printf("handler not found\n");
               return 0;
            }
            fft = (TVirtualFFT*)h->ExecPlugin(3, ndim, n, kFALSE);
            fft->Init(flag, 0, r2rkind);
            if (!opt.Contains("K"))
               fgFFT = fft;
            return fft;
         } else {
            printf("handler not found\n");
            return 0;
         }
      }
   }
   
   //if (fgFFT->GetTransformFlag()!=flag)
   fgFFT->Init(flag,0, r2rkind);
   return fgFFT;
}

//_____________________________________________________________________________
TVirtualFFT* TVirtualFFT::GetCurrentTransform()
{
// static: return current fgFFT

   if (fgFFT)
      return fgFFT;
   else{
      printf("fgFFT is not defined yet\n");
      return 0;
   }
}

//_____________________________________________________________________________
void TVirtualFFT::SetTransform(TVirtualFFT* fft)
{
// static: set the current transfrom to parameter

   fgFFT = fft;
}

//_____________________________________________________________________________
const char *TVirtualFFT::GetDefaultFFT()
{
// static: return the name of the default fft

   return fgDefault.Data();
}

//______________________________________________________________________________
void TVirtualFFT::SetDefaultFFT(const char *name)
{
   // static: set name of default fft

   if (fgDefault == name) return;
   delete fgFFT;
   fgFFT = 0;
   fgDefault = name;
}

