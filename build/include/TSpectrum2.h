// @(#)root/spectrum:$Id$
// Author: Miroslav Morhac   17/01/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TSpectrum2
#define ROOT_TSpectrum2

#include "TNamed.h"

class TH1;

class TSpectrum2 : public TNamed {
protected:
   Int_t         fMaxPeaks;         ///< Maximum number of peaks to be found
   Int_t         fNPeaks;           ///< number of peaks found
   Double_t      *fPosition;        ///< [fNPeaks] array of current peak positions
   Double_t      *fPositionX;       ///< [fNPeaks] X position of peaks
   Double_t      *fPositionY;       ///< [fNPeaks] Y position of peaks
   Double_t       fResolution;      ///< *NOT USED* resolution of the neighboring peaks
   TH1           *fHistogram;       ///< resulting histogram
static Int_t      fgAverageWindow;  ///< Average window of searched peaks
static Int_t      fgIterations;     ///< Maximum number of decon iterations (default=3)

public:
   enum {
       kBackIncreasingWindow =0,
       kBackDecreasingWindow =1,
       kBackSuccessiveFiltering =0,
       kBackOneStepFiltering =1
   };

   TSpectrum2();
   TSpectrum2(Int_t maxpositions, Double_t resolution=1); // resolution is *NOT USED*
   ~TSpectrum2() override;
   virtual TH1  *Background(const TH1 *hist, Int_t niter=20, Option_t *option="");
   TH1          *GetHistogram() const {return fHistogram;}
   Int_t         GetNPeaks() const {return fNPeaks;}
   Double_t      *GetPositionX() const {return fPositionX;}
   Double_t      *GetPositionY() const {return fPositionY;}
   void  Print(Option_t *option="") const override;
   virtual Int_t Search(const TH1 *hist, Double_t sigma=2, Option_t *option="", Double_t threshold=0.05);
   static void   SetAverageWindow(Int_t w=3);   //set average window
   static void   SetDeconIterations(Int_t n=3); //set max number of decon iterations
   void          SetResolution(Double_t resolution=1); // *NOT USED*

   //new functions January 2006
   const char   *Background(Double_t **spectrum,Int_t ssizex, Int_t ssizey,Int_t numberIterationsX,Int_t numberIterationsY,Int_t direction,Int_t filterType);
   const char   *SmoothMarkov(Double_t **source, Int_t ssizex, Int_t ssizey, Int_t averWindow);
   const char   *Deconvolution(Double_t **source, Double_t **resp, Int_t ssizex, Int_t ssizey,Int_t numberIterations, Int_t numberRepetitions, Double_t boost);
   Int_t         SearchHighRes(Double_t **source,Double_t **dest, Int_t ssizex, Int_t ssizey, Double_t sigma, Double_t threshold, Bool_t backgroundRemove,Int_t deconIterations, Bool_t markov, Int_t averWindow);

   static Int_t        StaticSearch(const TH1 *hist, Double_t sigma=2, Option_t *option="goff", Double_t threshold=0.05);
   static TH1         *StaticBackground(const TH1 *hist,Int_t niter=20, Option_t *option="");

   ClassDefOverride(TSpectrum2,1)  //Peak Finder, background estimator, Deconvolution for 2-D histograms
};

#endif

