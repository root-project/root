// @(#)root/base:$Name:  $:$Id: TVirtualFFT.cxx,v 1.16 2006/03/20 08:22:40 brun Exp $
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
   if (this==fgFFT)
      fgFFT = 0;
}

//_____________________________________________________________________________
TVirtualFFT* TVirtualFFT::FFT(Int_t ndim, Int_t *n, Option_t *type, Option_t *flag, Option_t *global_option)
{
//Returns a pointer to the FFT of requested size and type.
//Parameters:
// -ndim : number of transform dimensions
// -n    : sizes of each dimension (an array at least ndim long)
// -type : transform type
//    Available transform types are:
//    C2CForward, C2CBackward, C2R, R2C, R2HC, HC2R, DHT - see class description for details
// -flag : choosing how much time should be spent in planning the transform:
//      Possible flag_options:
//     "ES" (from "estimate") - no time in preparing the transform, but probably sub-optimal
//                              performance
//     "M" (from "measure") - some time spend in finding the optimal way to do the transform
//     "P" (from "patient") - more time spend in finding the optimal way to do the transform
//     "EX" (from "exhaustive") - the most optimal way is found
//     This option should be chosen depending on how many transforms of the same size and
//     type are going to be done. Planning is only done once, for the first transform of 
//     this size and type. Default is "M".
// -global_option : Possible global_options:
//     ""  - default, changes and returns the global fgFFT variable
//     "O" - without touching the global fgFFT, creates and returns a new TVirtualFFT*. 
//           user is then responsible for deleting it.

   Int_t inputtype=0, currenttype=0;
   TString typeopt = type;
   typeopt.ToUpper();
   TString globalopt = global_option;
   globalopt.ToUpper();
   Int_t ndiff = 0;

   if (!globalopt.Contains("O")){
      if (fgFFT){
         if (fgFFT->GetNdim()!=ndim)
            ndiff++;
         else {
            for (Int_t i=0; i<ndim; i++){
               if (n[i]!=(fgFFT->GetN())[i])
                  ndiff++;
            }
         }
         if (fgFFT->GetType()!=type){
            if (typeopt.Contains("HC") || typeopt.Contains("DHT"))
               inputtype = 1;
            if (strcmp(fgFFT->GetType(),"R2HC")==0 || strcmp(fgFFT->GetType(),"HC2R")==0 || strcmp(fgFFT->GetType(),"DHT")==0)
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
   if (typeopt.Contains("C2CB") || typeopt.Contains("C2R"))
      sign = 1; 
   if (typeopt.Contains("C2CF") || typeopt.Contains("R2C"))
      sign = -1; 
   
   TVirtualFFT *fft = 0;
   if (globalopt.Contains("O") || !fgFFT){   
      TPluginHandler *h;
      TString pluginname;
      if (fgDefault.Length()==0) fgDefault="fftw";
      if (strcmp(fgDefault.Data(),"fftw")==0){
         if (typeopt.Contains("C2C")) pluginname = "fftwc2c";
         if (typeopt.Contains("C2R")) pluginname = "fftwc2r";
         if (typeopt.Contains("R2C")) pluginname = "fftwr2c";
         if (typeopt.Contains("HC") || typeopt.Contains("DHT")) pluginname = "fftwr2r";
         if ((h=gROOT->GetPluginManager()->FindHandler("TVirtualFFT", pluginname))) {
            if (h->LoadPlugin()==-1){
               printf("handler not found\n");
               return 0;
            }
            fft = (TVirtualFFT*)h->ExecPlugin(3, ndim, n, kFALSE);
            Int_t *kind = new Int_t[1];
            if (pluginname=="fftwr2r"){
               if (typeopt.Contains("R2HC")) kind[0] = 10;
               if (typeopt.Contains("HC2R")) kind[0] = 11;
               if (typeopt.Contains("DHT")) kind[0] = 12;
            }
            fft->Init(flag, sign, kind);
            if (!globalopt.Contains("O")){
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
      if (fgFFT->GetSign()!=sign || fgFFT->GetTransformFlag()!=flag || fgFFT->GetType()!=type){
         Int_t *kind = new Int_t[1];
         if (inputtype==1){
            if (typeopt.Contains("R2HC")) kind[0] = 10;
            if (typeopt.Contains("HC2R")) kind[0] = 11;
            if (typeopt.Contains("DHT")) kind[0] = 12;
         }
         fgFFT->Init(flag, sign, kind);
         delete [] kind;
      }
   }
   return fgFFT;
}

//_____________________________________________________________________________
TVirtualFFT* TVirtualFFT::SineCosine(Int_t ndim, Int_t *n, Int_t *r2rkind, Option_t *flag, Option_t *global_option)
{
//Returns a pointer to a sine or cosine transform of requested size and kind
//Parameters:
// -ndim    : number of transform dimensions
// -n       : sizes of each dimension (an array at least ndim long)
// -r2rkind : transform kind for each dimension
//     4 different kinds of sine and cosine transforms are available
//     DCT-I   - kind=0
//     DCT-II  - kind=1
//     DCT-III - kind=2
//     DCT-IV  - kind=3
//     DST-I   - kind=4
//     DST-II  - kind=5
//     DSTIII  - kind=6
//     DSTIV   - kind=7
// -flag : choosing how much time should be spent in planning the transform:
//     Possible flag_options:
//     "ES" (from "estimate") - no time in preparing the transform, but probably sub-optimal
//      performance
//     "M" (from "measure") - some time spend in finding the optimal way to do the transform
//     "P" (from "patient") - more time spend in finding the optimal way to do the transform
//     "EX" (from "exhaustive") - the most optimal way is found
//     This option should be chosen depending on how many transforms of the same size and
//     type are going to be done. Planning is only done once, for the first transform of 
//     this size and type.
// -global_tion : Possible global options:
//     ""  - default, changes and returns the global fgFFT variable
//     "O" - without touching the global fgFFT, creates and returns a new TVirtualFFT*. 
//           user is then responsible for deleting it.

   TString globalopt = global_option;
   if (!globalopt.Contains("O")){
      if (fgFFT){
         Int_t ndiff = 0;
         if (fgFFT->GetNdim()!=ndim || strcmp(fgFFT->GetType(),"R2R")!=0)
            ndiff++;
         else {
            for (Int_t i=0; i<ndim; i++){
               if (n[i]!=(fgFFT->GetN())[i])
                  ndiff++;
            }
            
         }
         if (ndiff>0){
            delete fgFFT;
            fgFFT = 0;
         }
      }
   }
   TVirtualFFT *fft = 0;
   if (!fgFFT || globalopt.Contains("O")){   
      TPluginHandler *h;
      TString pluginname;
      //TVirtualFFT *fft=0;
      if (fgDefault.Length()==0) fgDefault="fftw";
      if (strcmp(fgDefault.Data(),"fftw")==0){
         pluginname = "fftwr2r";
         if ((h=gROOT->GetPluginManager()->FindHandler("TVirtualFFT", pluginname))) {
            if (h->LoadPlugin()==-1){
               printf("handler not found\n");
               return 0;
            }
            fft = (TVirtualFFT*)h->ExecPlugin(3, ndim, n, kFALSE);
            fft->Init(flag, 0, r2rkind);
            if (!globalopt.Contains("O"))
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
//return current fgFFT

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
   fgFFT = fft;
}

//_____________________________________________________________________________
const char *TVirtualFFT::GetDefaultFFT()
{
   return fgDefault.Data();
}

//______________________________________________________________________________
void TVirtualFFT::SetDefaultFFT(const char *name)
{
   // static: set name of default fitter

   if (fgDefault == name) return;
   delete fgFFT;
   fgFFT = 0;
   fgDefault = name;
}

