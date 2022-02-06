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

#include "TNamed.h"

class TH1;

class TSpectrum3 : public TNamed {
protected:
   Int_t          fMaxPeaks;       ///< Maximum number of peaks to be found
   Int_t          fNPeaks;         ///< number of peaks found
   Double_t      *fPosition;       ///< [fNPeaks] array of current peak positions
   Double_t      *fPositionX;      ///< [fNPeaks] X positions of peaks
   Double_t      *fPositionY;      ///< [fNPeaks] Y positions of peaks
   Double_t      *fPositionZ;      ///< [fNPeaks] Z positions of peaks
   Double_t       fResolution;     ///< *NOT USED* resolution of the neighboring peaks
   TH1           *fHistogram;      ///< resulting histogram

public:
   enum {
       kBackIncreasingWindow =0,
       kBackDecreasingWindow =1,
       kBackSuccessiveFiltering =0,
       kBackOneStepFiltering =1
   };

   TSpectrum3();
   TSpectrum3(Int_t maxpositions, Double_t resolution=1); // resolution is *NOT USED*
   ~TSpectrum3() override;
   virtual const char *Background(const TH1 *hist, Int_t niter, Option_t *option="goff");
   const char         *Background(Double_t ***spectrum, Int_t ssizex, Int_t ssizey, Int_t ssizez, Int_t numberIterationsX,Int_t numberIterationsY, Int_t numberIterationsZ, Int_t direction,Int_t filterType);
   const char         *Deconvolution(Double_t ***source, const Double_t ***resp, Int_t ssizex, Int_t ssizey, Int_t ssizez,Int_t numberIterations, Int_t numberRepetitions, Double_t boost);
   TH1                *GetHistogram() const {return fHistogram;}
   Int_t               GetNPeaks() const {return fNPeaks;}
   Double_t            *GetPositionX() const {return fPositionX;}
   Double_t            *GetPositionY() const {return fPositionY;}
   Double_t            *GetPositionZ() const {return fPositionZ;}
   void        Print(Option_t *option="") const override;
   virtual Int_t       Search(const TH1 *hist, Double_t sigma=2, Option_t *option="goff", Double_t threshold=0.05);
   Int_t               SearchFast(const Double_t ***source, Double_t ***dest, Int_t ssizex, Int_t ssizey, Int_t ssizez, Double_t sigma, Double_t threshold, Bool_t markov, Int_t averWindow);
   Int_t               SearchHighRes(const Double_t ***source,Double_t ***dest, Int_t ssizex, Int_t ssizey, Int_t ssizez, Double_t sigma, Double_t threshold, Bool_t backgroundRemove,Int_t deconIterations, Bool_t markov, Int_t averWindow);
   void                SetResolution(Double_t resolution=1); // *NOT USED*
   const char         *SmoothMarkov(Double_t ***source, Int_t ssizex, Int_t ssizey, Int_t ssizez, Int_t averWindow);

   ClassDefOverride(TSpectrum3,1)  //Peak Finder, Background estimator, Markov smoothing and Deconvolution for 3-D histograms
};

#endif


