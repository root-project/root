// @(#)root/spectrum:$Id$
// Author: Miroslav Morhac   25/09/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TSpectrum3
#define ROOT_TSpectrum3

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TH1;

class TSpectrum3 : public TNamed {
protected:
   Int_t         fMaxPeaks;       //Maximum number of peaks to be found
   Int_t         fNPeaks;         //number of peaks found
   Float_t      *fPosition;       //[fNPeaks] array of current peak positions
   Float_t      *fPositionX;      //[fNPeaks] X positions of peaks
   Float_t      *fPositionY;      //[fNPeaks] Y positions of peaks
   Float_t      *fPositionZ;      //[fNPeaks] Z positions of peaks
   Float_t       fResolution;     //resolution of the neighboring peaks
   TH1          *fHistogram;      //resulting histogram

public:
   enum {
       kBackIncreasingWindow =0,
       kBackDecreasingWindow =1,
       kBackSuccessiveFiltering =0,
       kBackOneStepFiltering =1
   };

   TSpectrum3();
   TSpectrum3(Int_t maxpositions, Float_t resolution=1);
   virtual ~TSpectrum3();
   virtual const char *Background(const TH1 *hist,int niter, Option_t *option="goff");
   const char         *Background(float ***spectrum,Int_t ssizex, Int_t ssizey, Int_t ssizez, Int_t numberIterationsX,Int_t numberIterationsY, Int_t numberIterationsZ, Int_t direction,Int_t filterType);
   const char         *Deconvolution(float ***source, const float ***resp, Int_t ssizex, Int_t ssizey, Int_t ssizez,Int_t numberIterations, Int_t numberRepetitions, Double_t boost);
   TH1                *GetHistogram() const {return fHistogram;}
   Int_t               GetNPeaks() const {return fNPeaks;}
   Float_t            *GetPositionX() const {return fPositionX;}
   Float_t            *GetPositionY() const {return fPositionY;}
   Float_t            *GetPositionZ() const {return fPositionZ;}
   virtual void        Print(Option_t *option="") const;
   virtual Int_t       Search(const TH1 *hist, Double_t sigma=2, Option_t *option="goff", Double_t threshold=0.05);
   Int_t               SearchFast(const float ***source, float ***dest, Int_t ssizex, Int_t ssizey, Int_t ssizez, Double_t sigma, Double_t threshold, Bool_t markov, Int_t averWindow);
   Int_t               SearchHighRes(const float ***source,float ***dest, Int_t ssizex, Int_t ssizey, Int_t ssizez, Double_t sigma, Double_t threshold, Bool_t backgroundRemove,Int_t deconIterations, Bool_t markov, Int_t averWindow);
   void                SetResolution(Float_t resolution=1);
   const char         *SmoothMarkov(float ***source, Int_t ssizex, Int_t ssizey, Int_t ssizez, Int_t averWindow);

   ClassDef(TSpectrum3,1)  //Peak Finder, Background estimator, Markov smoothing and Deconvolution for 3-D histograms
};

#endif


