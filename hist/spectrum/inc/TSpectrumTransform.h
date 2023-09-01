// @(#)root/spectrum:$Id$
// Author: Miroslav Morhac   25/09/06

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TSpectrumTransform
#define ROOT_TSpectrumTransform

#include "TNamed.h"

class TH1;

class TSpectrumTransform :  public TNamed {

protected:
   Int_t     fSize;                      ///< length of transformed data
   Int_t     fTransformType;             ///< type of transformation (Haar, Walsh, Cosine, Sine, Fourier, Hartley, Fourier-Walsh, Fourier-Haar, Walsh-Haar, Cosine-Walsh, Cosine-Haar, Sine-Walsh, Sine-Haar)
   Int_t     fDegree;                    ///< degree of mixed transform, applies only for Fourier-Walsh, Fourier-Haar, Walsh-Haar, Cosine-Walsh, Cosine-Haar, Sine-Walsh, Sine-Haar transforms
   Int_t     fDirection;                 ///< forward or inverse transform
   Int_t     fXmin;                      ///< first channel of filtered or enhanced region
   Int_t     fXmax;                      ///< last channel of filtered or enhanced region
   Double_t   fFilterCoeff;              ///< value set in the filtered region
   Double_t   fEnhanceCoeff;             ///< multiplication coefficient applied in enhanced region;

public:
   enum {
       kTransformHaar =0,
       kTransformWalsh =1,
       kTransformCos =2,
       kTransformSin =3,
       kTransformFourier =4,
       kTransformHartley =5,
       kTransformFourierWalsh =6,
       kTransformFourierHaar =7,
       kTransformWalshHaar =8,
       kTransformCosWalsh =9,
       kTransformCosHaar =10,
       kTransformSinWalsh =11,
       kTransformSinHaar =12,
       kTransformForward =0,
       kTransformInverse =1
   };
   TSpectrumTransform();
   TSpectrumTransform(Int_t size);
   ~TSpectrumTransform() override;

protected:
   void                BitReverse(Double_t *working_space,Int_t num);
   void                BitReverseHaar(Double_t *working_space,Int_t shift,Int_t num,Int_t start);
   void                Fourier(Double_t *working_space,Int_t num,Int_t hartley,Int_t direction,Int_t zt_clear);
   Int_t               GeneralExe(Double_t *working_space,Int_t zt_clear,Int_t num,Int_t degree,Int_t type);
   Int_t               GeneralInv(Double_t *working_space,Int_t num,Int_t degree,Int_t type);
   void                Haar(Double_t *working_space,Int_t num,Int_t direction);
   void                Walsh(Double_t *working_space,Int_t num);

public:
   void                Enhance(const Double_t *source, Double_t *destVector);
   void                FilterZonal(const Double_t *source, Double_t *destVector);
   void                SetDirection(Int_t direction);
   void                SetEnhanceCoeff(Double_t enhanceCoeff);
   void                SetFilterCoeff(Double_t filterCoeff);
   void                SetRegion(Int_t xmin, Int_t xmax);
   void                SetTransformType(Int_t transType, Int_t degree);
   void                Transform(const Double_t *source, Double_t *destVector);

   ClassDefOverride(TSpectrumTransform,1)  //Spectrum Transformer, it calculates classic orthogonal 1D transforms
};


#endif

