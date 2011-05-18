// $Id$
/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : LDA                                                                   *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Local LDA method used by MethodKNN to compute MVA value.                  *
 *      This is experimental code under development. This class computes          *
 *      parameters of signal and background PDFs using Gaussian aproximation.     *
 *                                                                                *
 * Author:                                                                        *
 *      John Alison John.Alison@cern.ch - University of Pennsylvania, USA         *
 *                                                                                *
 * Copyright (c) 2007:                                                            *
 *      CERN, Switzerland                                                         * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *      University of Pennsylvania, USA                                           *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

// Local
#include "TMVA/LDA.h"

// C/C++
#include <iostream>

#ifndef ROOT_TDecompSVD
#include "TDecompSVD.h"       
#endif
#ifndef ROOT_TMatrixF
#include "TMatrixF.h"       
#endif
#ifndef ROOT_TMath
#include "TMath.h"       
#endif

#ifndef ROOT_TMVA_Types
#include "TMVA/Types.h"       
#endif
#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"       
#endif

//_______________________________________________________________________
TMVA::LDA::LDA( Float_t tolerence, Bool_t debug ) 
   : fTolerence(tolerence),
     fNumParams(0),
     fSigma(0),
     fSigmaInverse(0),
     fDebug(debug),
     fLogger( new MsgLogger("LDA", (debug?kINFO:kDEBUG)) )
{
   // constructor
}

//_______________________________________________________________________
TMVA::LDA::~LDA()
{
   // destructor
   delete fLogger;
}

//_______________________________________________________________________
void TMVA::LDA::Initialize(const LDAEvents& inputSignalEvents, const LDAEvents& inputBackgroundEvents)
{
   // Create LDA matrix using local events found by knn method
   Log() << kDEBUG << "There are: " << inputSignalEvents.size() << " input signal events " << Endl;
   Log() << kDEBUG << "There are: " << inputBackgroundEvents.size() << " input background events " << Endl;

   fNumParams = inputSignalEvents[0].size();
  
   UInt_t numSignalEvents = inputSignalEvents.size();
   UInt_t numBackEvents  = inputBackgroundEvents.size();
   UInt_t numTotalEvents = numSignalEvents + numBackEvents;
   fEventFraction[0] = (Float_t)numBackEvents/numTotalEvents;
   fEventFraction[1] = (Float_t)numSignalEvents/numTotalEvents;
   UInt_t K = 2;

   // get the mean of the signal and background for each parameter
   std::vector<Float_t> m_muSignal (fNumParams,0.0);
   std::vector<Float_t> m_muBackground (fNumParams,0.0);
   for (UInt_t param=0; param < fNumParams; ++param) {
      for (UInt_t eventNumber=0; eventNumber < numSignalEvents; ++eventNumber)
         m_muSignal[param] += inputSignalEvents[eventNumber][param];
      for (UInt_t eventNumber=0; eventNumber < numBackEvents; ++eventNumber)
         m_muBackground[param] += inputBackgroundEvents[eventNumber][param]/numBackEvents;
      m_muSignal[param] /= numSignalEvents;
      m_muBackground[param] /= numBackEvents;
   }
   fMu[0] = m_muBackground;
   fMu[1] = m_muSignal;

   if (fDebug) {
      Log() << kDEBUG << "the signal means" << Endl;
      for (UInt_t param=0; param < fNumParams; ++param)
         Log() << kDEBUG << m_muSignal[param] << Endl;
      Log() << kDEBUG << "the background means" << Endl;
      for (UInt_t param=0; param < inputBackgroundEvents[0].size(); ++param)
         Log() << kDEBUG << m_muBackground[param] << Endl;
   }
  
   // sigma is a sum of two symmetric matrices, one for the background and one for signal
   // get the matricies seperately (def not be the best way to do it!)
  
   // the signal, background, and total matrix
   TMatrixF sigmaSignal(fNumParams, fNumParams);
   TMatrixF sigmaBack(fNumParams, fNumParams);
   if (fSigma!=0) delete fSigma;
   fSigma = new TMatrixF(fNumParams, fNumParams);
   for (UInt_t row=0; row < fNumParams; ++row) {
      for (UInt_t col=0; col < fNumParams; ++col) {
         sigmaSignal[row][col] = 0;
         sigmaBack[row][col] = 0;
         (*fSigma)[row][col] = 0;
      }
   }

   for (UInt_t eventNumber=0; eventNumber < numSignalEvents; ++eventNumber) {
      for (UInt_t row=0; row < fNumParams; ++row) {
         for (UInt_t col=0; col < fNumParams; ++col) {
            sigmaSignal[row][col] += (inputSignalEvents[eventNumber][row] - m_muSignal[row]) * (inputSignalEvents[eventNumber][col] - m_muSignal[col] );
         }
      }
   }
  
   for (UInt_t eventNumber=0; eventNumber < numBackEvents; ++eventNumber) {
      for (UInt_t row=0; row < fNumParams; ++row) {
         for (UInt_t col=0; col < fNumParams; ++col) {
            sigmaBack[row][col] += (inputBackgroundEvents[eventNumber][row] - m_muBackground[row]) * (inputBackgroundEvents[eventNumber][col] - m_muBackground[col] );
         }
      }
   }   

   // the total matrix 
   *fSigma = sigmaBack + sigmaSignal;
   *fSigma *= 1.0/(numTotalEvents - K);
  
   if (fDebug) {
      Log() << "after filling sigmaSignal" <<Endl;
      sigmaSignal.Print();
      Log() << "after filling sigmaBack" <<Endl;
      sigmaBack.Print();
      Log() << "after filling total Sigma" <<Endl;
      fSigma->Print();
   }

   TDecompSVD solutionSVD = TDecompSVD( *fSigma );
   TMatrixF   decomposed  = TMatrixF( fNumParams, fNumParams );
   TMatrixF diag  ( fNumParams, fNumParams );
   TMatrixF uTrans( fNumParams, fNumParams );
   TMatrixF vTrans( fNumParams, fNumParams );
   if (solutionSVD.Decompose()) {
      for (UInt_t i = 0; i< fNumParams; ++i) {
         if (solutionSVD.GetSig()[i] > fTolerence)
            diag(i,i) = solutionSVD.GetSig()[i];
         else
            diag(i,i) = fTolerence;
      }

      if (fDebug) {
         Log() << "the diagonal" <<Endl;
         diag.Print();
      }

      decomposed = solutionSVD.GetV();
      decomposed *= diag;
      decomposed *= solutionSVD.GetU();
    
      if (fDebug) {
         Log() << "the decomposition " <<Endl;
         decomposed.Print();
      }
    
      *fSigmaInverse = uTrans.Transpose(solutionSVD.GetU());
      *fSigmaInverse /= diag;
      *fSigmaInverse *= vTrans.Transpose(solutionSVD.GetV());

      if (fDebug) {
         Log() << "the SigmaInverse " <<Endl;
         fSigmaInverse->Print();
        
         Log() << "the real " <<Endl;
         fSigma->Invert();
         fSigma->Print();
      
         Bool_t problem = false;
         for (UInt_t i =0; i< fNumParams; ++i) {
            for (UInt_t j =0; j< fNumParams; ++j) {
               if (TMath::Abs((Float_t)(*fSigma)(i,j) - (Float_t)(*fSigmaInverse)(i,j)) > 0.01) {
                  Log() << "problem, i= "<< i << " j= " << j << Endl; 
                  Log() << "Sigma(i,j)= "<< (*fSigma)(i,j) << " SigmaInverse(i,j)= " << (*fSigmaInverse)(i,j) <<Endl; 
                  Log() << "The difference is : " << TMath::Abs((Float_t)(*fSigma)(i,j) - (Float_t)(*fSigmaInverse)(i,j)) <<Endl;
                  problem = true;
               }
            }
         }
         if (problem) Log() << kWARNING << "Problem with the inversion!" << Endl;
         
      }    
   }
}

//_______________________________________________________________________
Float_t TMVA::LDA::FSub(const std::vector<Float_t>& x, Int_t k)
{
   //
   // Probability value using Gaussian approximation
   //
   Float_t prefactor  = 1.0/(TMath::TwoPi()*TMath::Sqrt(fSigma->Determinant()));
   std::vector<Float_t> m_transPoseTimesSigmaInverse;
  
   for (UInt_t j=0; j < fNumParams; ++j) {
      Float_t m_temp = 0;
      for (UInt_t i=0; i < fNumParams; ++i) {
         m_temp += (x[i] - fMu[k][i]) * (*fSigmaInverse)(j,i);
      }
      m_transPoseTimesSigmaInverse.push_back(m_temp);
   }
  
   Float_t exponent = 0.0;
   for (UInt_t i=0; i< fNumParams; ++i) {
      exponent += m_transPoseTimesSigmaInverse[i]*(x[i] - fMu[k][i]);
   }
  
   exponent *= -0.5;

   return prefactor*TMath::Exp( exponent );
}

//_______________________________________________________________________
Float_t TMVA::LDA::GetProb(const std::vector<Float_t>& x, Int_t k)
{
   //
   // Signal probability with Gaussian approximation
   //
   Float_t m_numerator = FSub(x,k)*fEventFraction[k];
   Float_t m_denominator = FSub(x,0)*fEventFraction[0]+FSub(x,1)*fEventFraction[1];

   return m_numerator/m_denominator;
}

//_______________________________________________________________________
Float_t TMVA::LDA::GetLogLikelihood( const std::vector<Float_t>& x, Int_t k )
{
   //
   // Log likelihood function with Gaussian approximation
   //
   return TMath::Log( FSub(x,k)/FSub(x,!k) ) + TMath::Log( fEventFraction[k]/fEventFraction[!k] );
}
