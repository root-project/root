// @(#)root/tmva $Id$
// Author: Asen Christov

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : KDEKernel                                                             *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      The Probability Density Functions (PDFs) used for the Likelihood analysis *
 *      can suffer from low statistics of the training samples. This can cause    *
 *      the PDFs to fluctuate instead to be smooth. Nonparamatric Kernel Density  *
 *      Estimation is one of the methods to produce "smooth" PDFs.                *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Asen Christov   <christov@physik.uni-freiburg.de> - Freiburg U., Germany  *
 *                                                                                *
 * Copyright (c) 2007:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *      Freiburg U., Germany                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_KDEKernel
#define ROOT_TMVA_KDEKernel

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// KDEKernel                                                            //
//                                                                      //
// KDE Kernel for "smoothing" the PDFs                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "Rtypes.h"

class TH1;
class TH1F;
class TF1;

namespace TMVA {

   class MsgLogger;

   class KDEKernel {

   public:

      enum EKernelType   { kNone = 0, kGauss = 1 };
      enum EKernelIter   { kNonadaptiveKDE = 1, kAdaptiveKDE = 2 };
      enum EKernelBorder { kNoTreatment = 1, kKernelRenorm = 2, kSampleMirror = 3 };

   public:

      KDEKernel( EKernelIter kiter = kNonadaptiveKDE, const TH1* hist = nullptr, Float_t lower_edge=0., Float_t upper_edge=1., EKernelBorder kborder = kNoTreatment, Float_t FineFactor = 1.);

      virtual ~KDEKernel( void );

      // calculates the integral of the Kernel function in the given bin.
      Float_t GetBinKernelIntegral(Float_t lowr, Float_t highr, Float_t mean, Int_t binnum);

      // sets the type of Kernel to be used (Default 1 mean Gaussian)
      void SetKernelType( EKernelType ktype = kGauss );

      // modified name (remove TMVA::)
      const char* GetName() const { return "KDEKernel"; }

   private:

      Float_t       fSigma;             ///< Width of the Kernel function
      EKernelIter   fIter;              ///< iteration number
      Float_t       fLowerEdge;         ///< the lower edge of the PDF
      Float_t       fUpperEdge;         ///< the upper edge of the PDF
      Float_t       fFineFactor;        ///< fine tuning factor for Adaptive KDE: factor to multiply the "width" of the Kernel function
      TF1          *fKernel_integ;      ///< the integral of the Kernel function
      EKernelBorder fKDEborder;         ///< The method to take care about "border" effects
      TH1F         *fHist;              ///< copy of input histogram
      TH1F         *fFirstIterHist;     ///< histogram to be filled in the hidden iteration
      TH1F         *fSigmaHist;         ///< contains the Sigmas Widths for adaptive KDE
      Bool_t        fHiddenIteration;   ///< Defines if whats currently running is the
      // (first) hidden iteration when doing adaptive KDE

      mutable MsgLogger* fLogger;       ///< message logger
      MsgLogger& Log() const { return *fLogger; }

      ClassDef(KDEKernel,0); // Kernel density estimator for PDF smoothing

   };// namespace TMVA
}
#endif // KDEKernel_H
