// @(#)root/spectrum:$Id$
// Author: Miroslav Morhac   27/05/99

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TSpectrum
#define ROOT_TSpectrum

#include "TNamed.h"

class TH1;

class TSpectrum : public TNamed {
private:

   TSpectrum(const TSpectrum&); // Not implemented
   TSpectrum& operator=(const TSpectrum&); // Not implemented

protected:
   Int_t          fMaxPeaks;         ///< Maximum number of peaks to be found
   Int_t          fNPeaks;           ///< number of peaks found
   Double_t      *fPosition;         ///< [fNPeaks] array of current peak positions
   Double_t      *fPositionX;        ///< [fNPeaks] X position of peaks
   Double_t      *fPositionY;        ///< [fNPeaks] Y position of peaks
   Double_t       fResolution;       ///< *NOT USED* resolution of the neighboring peaks
   TH1           *fHistogram;        ///< resulting histogram
static Int_t      fgAverageWindow;   ///< Average window of searched peaks
static Int_t      fgIterations;      ///< Maximum number of decon iterations (default=3)

public:
   enum {
       kBackOrder2 =0,
       kBackOrder4 =1,
       kBackOrder6 =2,
       kBackOrder8 =3,
       kBackIncreasingWindow =0,
       kBackDecreasingWindow =1,
       kBackSmoothing3 =3,
       kBackSmoothing5 =5,
       kBackSmoothing7 =7,
       kBackSmoothing9 =9,
       kBackSmoothing11 =11,
       kBackSmoothing13 =13,
       kBackSmoothing15 =15
   };

   TSpectrum();
   TSpectrum(Int_t maxpositions, Double_t resolution=1); // resolution is *NOT USED*
   ~TSpectrum() override;
   virtual TH1        *Background(const TH1 *hist,Int_t niter=20, Option_t *option="");
   TH1                *GetHistogram() const {return fHistogram;}
   Int_t               GetNPeaks() const {return fNPeaks;}
   Double_t            *GetPositionX() const {return fPositionX;}
   Double_t            *GetPositionY() const {return fPositionY;}
   void        Print(Option_t *option="") const override;
   virtual Int_t       Search(const TH1 *hist, Double_t sigma=2, Option_t *option="", Double_t threshold=0.05);
   static void         SetAverageWindow(Int_t w=3);   //set average window
   static void         SetDeconIterations(Int_t n=3); //set max number of decon iterations
   void                SetResolution(Double_t resolution=1); // *NOT USED*

   //new functions January 2006
   const char         *Background(Double_t *spectrum, Int_t ssize,Int_t numberIterations,Int_t direction, Int_t filterOrder,bool smoothing,Int_t smoothWindow,bool compton);
   const char         *SmoothMarkov(Double_t *source, Int_t ssize, Int_t averWindow);
   const char         *Deconvolution(Double_t *source, const Double_t *response,Int_t ssize, Int_t numberIterations,Int_t numberRepetitions, Double_t boost );
   const char         *DeconvolutionRL(Double_t *source, const Double_t *response,Int_t ssize, Int_t numberIterations,Int_t numberRepetitions, Double_t boost );
   const char         *Unfolding(Double_t *source,const Double_t **respMatrix,Int_t ssizex, Int_t ssizey,Int_t numberIterations,Int_t numberRepetitions, Double_t boost);
   Int_t               SearchHighRes(Double_t *source,Double_t *destVector, Int_t ssize,Double_t sigma, Double_t threshold,bool backgroundRemove,Int_t deconIterations,bool markov, Int_t averWindow);
   Int_t               Search1HighRes(Double_t *source,Double_t *destVector, Int_t ssize,Double_t sigma, Double_t threshold,bool backgroundRemove,Int_t deconIterations,bool markov, Int_t averWindow);

   static Int_t        StaticSearch(const TH1 *hist, Double_t sigma=2, Option_t *option="goff", Double_t threshold=0.05);
   static TH1         *StaticBackground(const TH1 *hist,Int_t niter=20, Option_t *option="");

   ClassDefOverride(TSpectrum,3)  //Peak Finder, background estimator, Deconvolution
};

#endif

