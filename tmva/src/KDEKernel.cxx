// @(#)root/tmva $Id$ 
// Author: Asen Christov

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate Data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::KDEKernel                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Asen Christov   <christov@physik.uni-freiburg.de> - Freiburg U., Germany  *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *      Freiburg U., Germany                                                      * 
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#include "TH1.h"
#include "TH1F.h"
#include "TF1.h"

// #if ROOT_VERSION_CODE >= 364802
// #ifndef ROOT_TMathBase
// #include "TMathBase.h"
// #endif
// #else
// #ifndef ROOT_TMath
#include "TMath.h"
// #endif
// #endif

#include "TMVA/KDEKernel.h"
#include "TMVA/MsgLogger.h"

ClassImp(TMVA::KDEKernel)

//_______________________________________________________________________
TMVA::KDEKernel::KDEKernel( EKernelIter kiter, const TH1 *hist, Float_t lower_edge, Float_t upper_edge,
                            EKernelBorder kborder, Float_t FineFactor )
   : fSigma( 1. ),
     fIter ( kiter ),
     fLowerEdge (lower_edge ),
     fUpperEdge (upper_edge),
     fFineFactor ( FineFactor ),
     fKernel_integ ( 0 ),
     fKDEborder ( kborder ),
     fLogger( new MsgLogger("KDEKernel") )
{
   // constructor
   // sanity check
   if (hist == NULL) {
      Log() << kFATAL << "Called without valid histogram pointer (hist)!" << Endl;
   } 

   fHist          = (TH1F*)hist->Clone();
   fFirstIterHist = (TH1F*)hist->Clone();
   fFirstIterHist->Reset(); // now it is empty but with the proper binning
   fSigmaHist     = (TH1F*)hist->Clone();
   fSigmaHist->Reset(); // now fSigmaHist is empty but with the proper binning
   
   fHiddenIteration=false;
}

//_______________________________________________________________________
TMVA::KDEKernel::~KDEKernel()
{
   // destructor
   if (fHist           != NULL) delete fHist;
   if (fFirstIterHist  != NULL) delete fFirstIterHist;
   if (fSigmaHist      != NULL) delete fSigmaHist;
   if (fKernel_integ   != NULL) delete fKernel_integ;
   delete fLogger;
}

//_______________________________________________________________________
Double_t GaussIntegral(Double_t *x, Double_t *par)
{
   // when using Gaussian as Kernel function this is faster way to calculate the integrals
   if ( (par[1]<=0) || (x[0]>x[1])) return -1.;
  
   Float_t xs1=(x[0]-par[0])/par[1];
   Float_t xs2=(x[1]-par[0])/par[1];
  
   if (xs1==0) {
      if (xs2==0) return 0.;
      if (xs2>0 ) return 0.5*TMath::Erf(xs2);
   }
   if (xs2==0) return 0.5*TMath::Erf(TMath::Abs(xs1));
   if (xs1>0) return 0.5*(TMath::Erf(xs2)-TMath::Erf(xs1));
   if (xs1<0) {
      if (xs2>0 ) return 0.5*(TMath::Erf(xs2)+TMath::Erf(TMath::Abs(xs1)));
      else return 0.5*(TMath::Erf(TMath::Abs(xs1))-TMath::Erf(TMath::Abs(xs2)));
   }
   return -1.;
}

//_______________________________________________________________________
void TMVA::KDEKernel::SetKernelType( EKernelType ktype )
{
   // fIter == 1 ---> nonadaptive KDE
   // fIter == 2 ---> adaptive KDE
 
   if (ktype == kGauss) {

      // i.e. gauss kernel
      //
      // this is going to be done for both (nonadaptive KDE and adaptive KDE)
      // for nonadaptive KDE this is the only = final thing to do
      // for adaptive KDE this is going to be used in the first (hidden) iteration
      fKernel_integ = new TF1("GaussIntegral",GaussIntegral,fLowerEdge,fUpperEdge,4); 
      fSigma = ( TMath::Sqrt(2.0)
                 *TMath::Power(4./3., 0.2)
                 *fHist->GetRMS()
                 *TMath::Power(fHist->Integral(), -0.2) ); 
      // this formula for sigma is valid for Gaussian Kernel function (nonadaptive KDE).
      // formula found in:
      // Multivariate Density Estimation, Theory, Practice and Visualization D. W. SCOTT, 1992 New York, Wiley
      if (fSigma <= 0 ) {
         Log() << kFATAL << "<SetKernelType> KDE sigma has invalid value ( <=0 ) !" << Endl;
      }
   }

   if (fIter == kAdaptiveKDE) {

      // this is done only for adaptive KDE      
      
      // fill a temporary histo using nonadaptive KDE
      // this histo is identical with the final output when using only nonadaptive KDE
      fHiddenIteration=true;
      
      Float_t histoLowEdge=fHist->GetBinLowEdge(1);
      Float_t histoUpperEdge=fHist->GetBinLowEdge(fHist->GetNbinsX()+1);

      for (Int_t i=1;i<fHist->GetNbinsX();i++) {
         // loop over the bins of the original histo
         for (Int_t j=1;j<fFirstIterHist->GetNbinsX();j++) {
            // loop over the bins of the PDF histo and fill it
            fFirstIterHist->AddBinContent(j,fHist->GetBinContent(i)*
                                    this->GetBinKernelIntegral(fFirstIterHist->GetBinLowEdge(j),
                                                               fFirstIterHist->GetBinLowEdge(j+1),
                                                               fHist->GetBinCenter(i),
                                                               i)
                                    );
         }
         if (fKDEborder == 3) { // mirror the saples and fill them again
         // in order to save time do the mirroring only for the first (the lowwer) 1/5 of the histo to the left; 
         // and the last (the higher) 1/5 of the histo to the right.
         // the middle part of the histo, which is not mirrored, has no influence on the border effects anyway ...
            if (i < fHist->GetNbinsX()/5  ) {  // the first (the lowwer) 1/5 of the histo
               for (Int_t j=1;j<fFirstIterHist->GetNbinsX();j++) {
               // loop over the bins of the PDF histo and fill it
               fFirstIterHist->AddBinContent(j,fHist->GetBinContent(i)*
                                       this->GetBinKernelIntegral(fFirstIterHist->GetBinLowEdge(j),
                                                               fFirstIterHist->GetBinLowEdge(j+1),
                                                               2*histoLowEdge-fHist->GetBinCenter(i), // mirroring to the left
                                                               i)
                                      );
               }
            }
            if (i > 4*fHist->GetNbinsX()/5) { // the last (the higher) 1/5 of the histo
               for (Int_t j=1;j<fFirstIterHist->GetNbinsX();j++) {
               // loop over the bins of the PDF histo and fill it
               fFirstIterHist->AddBinContent(j,fHist->GetBinContent(i)*
                                       this->GetBinKernelIntegral(fFirstIterHist->GetBinLowEdge(j),
                                                               fFirstIterHist->GetBinLowEdge(j+1),
                                                               2*histoUpperEdge-fHist->GetBinCenter(i), // mirroring to the right
                                                               i)
                                      );
               }            
            }
         }
      }
      
      fFirstIterHist->SetEntries(fHist->GetEntries()); //set the number of entries to be the same as the original histo

      // do "function like" integration = sum of (bin_width*bin_content):
      Float_t integ=0;
      for (Int_t j=1;j<fFirstIterHist->GetNbinsX();j++) 
         integ+=fFirstIterHist->GetBinContent(j)*fFirstIterHist->GetBinWidth(j);
      fFirstIterHist->Scale(1./integ);
      
      fHiddenIteration=false;

      // OK, now we have the first iteration, 
      // next: calculate the Sigmas (Widths) for the second (adaptive) iteration 
      // based on the output of the first iteration 
      // these Sigmas will be stored in histo called fSigmaHist
      for (Int_t j=1;j<fFirstIterHist->GetNbinsX();j++) {
         // loop over the bins of the PDF histo and fill fSigmaHist
         if (fSigma*TMath::Sqrt(1.0/fFirstIterHist->GetBinContent(j)) <= 0 ) {
            Log() << kFATAL << "<SetKernelType> KDE sigma has invalid value ( <=0 ) !" << Endl;
         }
         
         fSigmaHist->SetBinContent(j,fFineFactor*fSigma/TMath::Sqrt(fFirstIterHist->GetBinContent(j)));
      }
   }

   if (fKernel_integ ==0 ) {
      Log() << kFATAL << "KDE kernel not correctly initialized!" << Endl;
   }
}

//_______________________________________________________________________
Float_t TMVA::KDEKernel::GetBinKernelIntegral( Float_t lowr, Float_t highr, Float_t mean, Int_t binnum )
{
   // calculates the integral of the Kernel
   if ((fIter == kNonadaptiveKDE) || fHiddenIteration  ) 
      fKernel_integ->SetParameters(mean,fSigma); // non adaptive KDE
   else if ((fIter == kAdaptiveKDE) && !fHiddenIteration ) 
      fKernel_integ->SetParameters(mean,fSigmaHist->GetBinContent(binnum)); // adaptive KDE

   if ( fKDEborder == 2 ) {  // renormalization of the kernel function
      Float_t renormFactor=1.0/fKernel_integ->Eval(fLowerEdge,fUpperEdge);
      return (renormFactor*fKernel_integ->Eval(lowr,highr));
   }
                                   
   // the RenormFactor takes care aboud the "border" effects, i.e. sets the 
   // integral to one inside the histogram borders
   return (fKernel_integ->Eval(lowr,highr));  
}

