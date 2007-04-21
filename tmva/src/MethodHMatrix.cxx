// @(#)root/tmva $Id: MethodHMatrix.cxx,v 1.13 2007/04/19 10:32:04 brun Exp $    
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate Data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::MethodHMatrix                                                   *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header file for description)                          *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-K Heidelberg, Germany ,                                               * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#include "TMVA/MethodHMatrix.h"
#include "TMVA/Tools.h"
#include "TMatrix.h"
#include "Riostream.h"
#include <algorithm>

ClassImp(TMVA::MethodHMatrix)

//_______________________________________________________________________
//Begin_Html                                                                      
/*
  H-Matrix method, which is implemented as a simple comparison of      
  chi-squared estimators for signal and background, taking into        
  account the linear correlations between the input variables          

  This MVA approach is used by the D&#216; collaboration (FNAL) for the 
  purpose of electron identification (see, eg., 
  <a href="http://arxiv.org/abs/hep-ex/9507007">hep-ex/9507007</a>). 
  As it is implemented in TMVA, it is usually equivalent or worse than
  the Fisher-Mahalanobis discriminant, and it has only been added for 
  the purpose of completeness.
  Two &chi;<sup>2</sup> estimators are computed for an event, each one
  for signal and background, using the estimates for the means and 
  covariance matrices obtained from the training sample:<br>
  <center>
  <img vspace=6 src="gif/tmva_chi2.gif" align="bottom" > 
  </center>
  TMVA then uses as normalised analyser for event (<i>i</i>) the ratio:
  (<i>&chi;<sub>S</sub>(i)<sup>2</sup> &minus; &chi;<sub>B</sub><sup>2</sup>(i)</i>)
  (<i>&chi;<sub>S</sub><sup>2</sup>(i) + &chi;<sub>B</sub><sup>2</sup>(i)</i>).
*/
//End_Html
//_______________________________________________________________________
 

//_______________________________________________________________________
TMVA::MethodHMatrix::MethodHMatrix( TString jobName, TString methodTitle, DataSet& theData, 
                                    TString theOption, TDirectory* theTargetDir )
   : TMVA::MethodBase( jobName, methodTitle, theData, theOption, theTargetDir )
{
   // standard constructor for the H-Matrix method
   //
   ProcessOptions();
   // HMatrix options: none
   InitHMatrix();
}

//_______________________________________________________________________
TMVA::MethodHMatrix::MethodHMatrix( DataSet & theData, 
                                    TString theWeightFile,  
                                    TDirectory* theTargetDir )
   : TMVA::MethodBase( theData, theWeightFile, theTargetDir ) 
{
   // constructor to calculate the H-Matrix from the weight file
   InitHMatrix();
}

//_______________________________________________________________________
void TMVA::MethodHMatrix::InitHMatrix( void )
{
   // default initialisation called by all constructors
   SetMethodName( "HMatrix" );
   SetMethodType( TMVA::Types::kHMatrix );
   SetTestvarName();

   fNormaliseInputVars = kTRUE;
   fInvHMatrixS = new TMatrixD( GetNvar(), GetNvar() );
   fInvHMatrixB = new TMatrixD( GetNvar(), GetNvar() );
   fVecMeanS    = new TVectorD( GetNvar() );
   fVecMeanB    = new TVectorD( GetNvar() );

   // the minimum requirement to declare an event signal-like
   SetSignalReferenceCut( 0.0 );
}

//_______________________________________________________________________
TMVA::MethodHMatrix::~MethodHMatrix( void )
{
   // destructor
   if (NULL != fInvHMatrixS) delete fInvHMatrixS;
   if (NULL != fInvHMatrixB) delete fInvHMatrixB;
   if (NULL != fVecMeanS   ) delete fVecMeanS;
   if (NULL != fVecMeanB   ) delete fVecMeanB;
}

//_______________________________________________________________________
void TMVA::MethodHMatrix::DeclareOptions() 
{
   // MethodHMatrix options: none (apart from those implemented in MethodBase)
}

//_______________________________________________________________________
void TMVA::MethodHMatrix::ProcessOptions() 
{
   // process user options
   MethodBase::ProcessOptions();
}

//_______________________________________________________________________
void TMVA::MethodHMatrix::Train( void )
{
   // computes H-matrices for signal and background samples

   // default sanity checks
   if (!CheckSanity()) fLogger << kFATAL << "<Train> sanity check failed" << Endl;

   // covariance matrices for signal and background
   ComputeCovariance( kTRUE,  fInvHMatrixS );
   ComputeCovariance( kFALSE, fInvHMatrixB );

   // sanity checks
   if (TMath::Abs(fInvHMatrixS->Determinant()) < 10E-24) {
      fLogger << kWARNING << "<Train> H-matrix  S is almost singular with deterinant= "
              << TMath::Abs(fInvHMatrixS->Determinant())
              << " did you use the variables that are linear combinations or highly correlated ???" 
              << Endl;
   }
   if (TMath::Abs(fInvHMatrixB->Determinant()) < 10E-24) {
      fLogger << kWARNING << "<Train> H-matrix  B is almost singular with deterinant= "
              << TMath::Abs(fInvHMatrixB->Determinant())
              << " did you use the variables that are linear combinations or highly correlated ???" 
              << Endl;
   }

    if (TMath::Abs(fInvHMatrixS->Determinant()) < 10E-120) {
       fLogger << kFATAL << "<Train> H-matrix  S is singular with deterinant= "
               << TMath::Abs(fInvHMatrixS->Determinant())
               << " did you use the variables that are linear combinations ???" 
               << Endl;
    }
    if (TMath::Abs(fInvHMatrixB->Determinant()) < 10E-120) {
       fLogger << kFATAL << "<Train> H-matrix  B is singular with deterinant= "
               << TMath::Abs(fInvHMatrixB->Determinant())
               << " did you use the variables that are linear combinations ???" 
               << Endl;
    }

   // invert matrix
   fInvHMatrixS->Invert();
   fInvHMatrixB->Invert();
}

//_______________________________________________________________________
void TMVA::MethodHMatrix::ComputeCovariance( Bool_t isSignal, TMatrixD* mat )
{
   // compute covariance matrix

   const UInt_t nvar = GetNvar();
   UInt_t ivar, jvar;

   // init matrices
   TVectorD vec(nvar);        vec  *= 0;
   TMatrixD mat2(nvar, nvar); mat2 *= 0;

   // initialise internal sum-of-weights variables
   Double_t sumOfWeights = 0;
   Double_t *xval = new Double_t[nvar];

   // perform event loop
   for (Int_t i=0; i<Data().GetNEvtTrain(); i++) {

      // fill the event
      ReadTrainingEvent(i);      
      if (GetEvent().IsSignal() != isSignal) continue;

      // event is of good type
      Double_t weight = GetEventWeight();
      sumOfWeights += weight;

      // mean values
      for (ivar=0; ivar<nvar; ivar++) {
         xval[ivar] = (fNormaliseInputVars) ? GetEventValNormalized(ivar) : GetEventVal(ivar);
      }

      // covariance matrix         
      for (ivar=0; ivar<nvar; ivar++) {

         vec(ivar)        += xval[ivar]*weight;
         mat2(ivar, ivar) += (xval[ivar]*xval[ivar])*weight;
         
         for (jvar=ivar+1; jvar<nvar; jvar++) {
            mat2(ivar, jvar) += (xval[ivar]*xval[jvar])*weight;
            mat2(jvar, ivar) = mat2(ivar, jvar); // symmetric matrix
         }
      }         
   }

   // variance-covariance
   for (ivar=0; ivar<nvar; ivar++) {

      if (isSignal) (*fVecMeanS)(ivar) = vec(ivar)/sumOfWeights;
      else          (*fVecMeanB)(ivar) = vec(ivar)/sumOfWeights;

      for (jvar=0; jvar<nvar; jvar++) {
         (*mat)(ivar, jvar) = mat2(ivar, jvar)/sumOfWeights - vec(ivar)*vec(jvar)/(sumOfWeights*sumOfWeights);
      }
   }

   delete [] xval;
}

//_______________________________________________________________________
Double_t TMVA::MethodHMatrix::GetMvaValue()
{
   // returns the H-matrix signal estimator
   Double_t s = GetChi2( Types::kSignal     );
   Double_t b = GetChi2( Types::kBackground );
  
   if (s+b < 0) fLogger << kFATAL << "big trouble: s+b: " << s+b << Endl;

   return (b - s)/(s + b);
}

//_______________________________________________________________________
Double_t TMVA::MethodHMatrix::GetChi2( TMVA::Event* e,  Types::ESBType type ) const
{
   // compute chi2-estimator for event according to type (signal/background)

   // loop over variables
   Int_t ivar,jvar;
   vector<Double_t> val( GetNvar() );
   for (ivar=0; ivar<GetNvar(); ivar++) {
      val[ivar] = e->GetVal(ivar);
      if (fNormaliseInputVars) 
         val[ivar] = __N__( val[ivar], GetXmin( ivar ), GetXmax( ivar ) );    
   }

   Double_t chi2 = 0;
   for (ivar=0; ivar<GetNvar(); ivar++) {
      for (jvar=0; jvar<GetNvar(); jvar++) {
         if (type == Types::kSignal) 
            chi2 += ( (val[ivar] - (*fVecMeanS)(ivar))*(val[jvar] - (*fVecMeanS)(jvar))
                      * (*fInvHMatrixS)(ivar,jvar) );
         else
            chi2 += ( (val[ivar] - (*fVecMeanB)(ivar))*(val[jvar] - (*fVecMeanB)(jvar))
                      * (*fInvHMatrixB)(ivar,jvar) );
      }
   }

   // sanity check
   if (chi2 < 0) fLogger << kFATAL << "<GetChi2> negative chi2: " << chi2 << Endl;

   return chi2;
}

//_______________________________________________________________________
Double_t TMVA::MethodHMatrix::GetChi2( Types::ESBType type ) const
{
   // compute chi2-estimator for event according to type (signal/background)

   // loop over variables
   Int_t ivar,jvar;
   vector<Double_t> val( GetNvar() );
   for (ivar=0; ivar<GetNvar(); ivar++)
      val[ivar] = fNormaliseInputVars ? GetEventValNormalized(ivar) : GetEvent().GetVal( ivar );

   Double_t chi2 = 0;
   for (ivar=0; ivar<GetNvar(); ivar++) {
      for (jvar=0; jvar<GetNvar(); jvar++) {
         if (type == Types::kSignal) 
            chi2 += ( (val[ivar] - (*fVecMeanS)(ivar))*(val[jvar] - (*fVecMeanS)(jvar))
                      * (*fInvHMatrixS)(ivar,jvar) );
         else
            chi2 += ( (val[ivar] - (*fVecMeanB)(ivar))*(val[jvar] - (*fVecMeanB)(jvar))
                      * (*fInvHMatrixB)(ivar,jvar) );
      }
   }

   // sanity check
   if (chi2 < 0) fLogger << kFATAL << "<GetChi2> negative chi2: " << chi2 << Endl;

   return chi2;
}
  
//_______________________________________________________________________
void  TMVA::MethodHMatrix::WriteWeightsToStream( ostream& o ) const
{  
   // write variable names and min/max 
   // NOTE: the latter values are mandatory for the normalisation 
   // in the reader application !!!
   Int_t ivar,jvar;
   o << this->GetMethodName() <<endl;

   // mean vectors
   for (ivar=0; ivar<GetNvar(); ivar++) {
      o << (*fVecMeanS)(ivar) << "  " << (*fVecMeanB)(ivar) << endl;
   }

   // inverse covariance matrices (signal)
   for (ivar=0; ivar<GetNvar(); ivar++) {
      for (jvar=0; jvar<GetNvar(); jvar++) {
         o << (*fInvHMatrixS)(ivar,jvar) << "  ";
      }
      o << endl;
   }

   // inverse covariance matrices (background)
   for (ivar=0; ivar<GetNvar(); ivar++) {
      for (jvar=0; jvar<GetNvar(); jvar++) {
         o << (*fInvHMatrixB)(ivar,jvar) << "  ";
      }
      o << endl;
   }
}

//_______________________________________________________________________
void  TMVA::MethodHMatrix::ReadWeightsFromStream( istream& istr )
{
   // read variable names and min/max
   // NOTE: the latter values are mandatory for the normalisation 
   // in the reader application !!!
   Int_t ivar,jvar;
   TString var, dummy;
   istr >> dummy;
   this->SetMethodName(dummy);

   // mean vectors
   for (ivar=0; ivar<GetNvar(); ivar++) 
      istr >> (*fVecMeanS)(ivar) >> (*fVecMeanB)(ivar);

   // inverse covariance matrices (signal)
   for (ivar=0; ivar<GetNvar(); ivar++) 
      for (jvar=0; jvar<GetNvar(); jvar++) 
         istr >> (*fInvHMatrixS)(ivar,jvar);

   // inverse covariance matrices (background)
   for (ivar=0; ivar<GetNvar(); ivar++) 
      for (jvar=0; jvar<GetNvar(); jvar++) 
         istr >> (*fInvHMatrixB)(ivar,jvar);
}

