// @(#)root/hist:$Name:  $:$Id: TSpectrum.h,v 1.3 2000/12/13 15:13:51 brun Exp $
// Author: Miroslav Morhac   27/05/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TSpectrum
#define ROOT_TSpectrum

/////////////////////////////////////////////////////////////////////////////
//	THIS FILE CONTAINS HEADERS FOR ADVANCED				   //
//	SPECTRA PROCESSING FUNCTIONS.	   				   //
//									   //
//	ONE-DIMENSIONAL BACKGROUND ESTIMATION FUNCTION			   //
//	TWO-DIMENSIONAL BACKGROUND ESTIMATION FUNCTION			   //
//	ONE-DIMENSIONAL DECONVOLUTION FUNCTION				   //
//	TWO-DIMENSIONAL DECONVOLUTION FUNCTION				   //
//	ONE-DIMENSIONAL PEAK SEARCH FUNCTION				   //
//	TWO-DIMENSIONAL PEAK SEARCH FUNCTION				   //
//									   //
//	Miroslav Morhac							   //
//	Institute of Physics						   //
//	Slovak Academy of Sciences					   //
//	Dubravska cesta 9, 842 28 BRATISLAVA				   //
//	SLOVAKIA							   //
//									   //
//	email:fyzimiro@savba.sk,	 fax:+421 7 54772479		   //
//									   //
/////////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TH1
#include "TH1.h"
#endif


class TSpectrum : public TNamed {
protected:
   Int_t         fMaxPeaks;       //Maximum number of peaks to be found
   Int_t         fNPeaks;         //number of peaks found
   Float_t      *fPosition;       //!array of current peak positions
   Float_t      *fPositionX;      //!X position of peaks
   Float_t      *fPositionY;      //!Y position of peaks
   Float_t       fResolution;     //resolution of the neighboring peaks  
   TH1          *fHistogram;      //resulting histogram

public:
   TSpectrum();
   TSpectrum(Int_t maxpositions, Float_t resolution=1);
   virtual          ~TSpectrum();
   virtual  char    *Background(TH1 *hist,int niter, Option_t *option="goff");
   virtual  char    *Background1(float *spectrum,int size,int niter);
   virtual  char    *Background2(float **spectrum,int sizex,int sizey,int niter);
   virtual  char    *Deconvolution1(float *source,float *resp,int size,int niter);
   virtual  char    *Deconvolution2(float **source,float **resp,int sizex,int sizey,int niter);
   virtual  TH1     *GetHistogram() const {return fHistogram;}
   virtual  Int_t    GetNPeaks() const {return fNPeaks;}
   virtual  Float_t *GetPositionX() const {return fPositionX;}
   virtual  Float_t *GetPositionY() const {return fPositionY;}
   virtual  int      PeakEvaluate(double *temp,int size,int xmax,double xmin);
   virtual  Int_t    Search(TH1 *hist, Double_t sigma, Option_t *option="goff");
   virtual  Int_t    Search1(float *spectrum,int size,double sigma);
   virtual  Int_t    Search2(float **source,int sizex,int sizey,double sigma);
   virtual  void     SetResolution(Float_t resolution=1);
   
   ClassDef(TSpectrum,2)  //Peak Finder, background estimator, Deconvolution
};

#endif
