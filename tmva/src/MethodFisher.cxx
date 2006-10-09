// @(#)root/tmva $Id: MethodFisher.cxx,v 1.48 2006/09/29 23:27:15 andreas.hoecker Exp $
// Author: Andreas Hoecker, Xavier Prudent, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate Data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_MethodFisher                                                     *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Original author of this Fisher-Discriminant implementation:                    *
 *      Andre Gaidot, CEA-France;                                                 *
 *      (Translation from FORTRAN)                                                *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-KP Heidelberg, Germany,                                               * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
// Begin_Html
/*
  Fisher and Mahalanobis Discriminants (Linear Discriminant Analysis) 
  
  <p>
  In the method of Fisher discriminants event selection is performed 
  in a transformed variable space with zero linear correlations, by
  distinguishing the mean values of the signal and background 
  distributions.<br></p>
  
  <p>
  The linear discriminant analysis determines an axis in the (correlated) 
  hyperspace of the input variables 
  such that, when projecting the output classes (signal and background) 
  upon this axis, they are pushed as far as possible away from each other,
  while events of a same class are confined in a close vicinity. 
  The linearity property of this method is reflected in the metric with
  which "far apart" and "close vicinity" are determined: the covariance
  matrix of the discriminant variable space.
  </p>

  <p>
  The classification of the events in signal and background classes 
  relies on the following characteristics (only): overall sample means,    
  <i><my:o>x</my:o><sub>i</sub></i>, for each input variable, <i>i</i>,
  class-specific sample means, <i><my:o>x</my:o><sub>S(B),i</sub></i>, 
  and total covariance matrix <i>T<sub>ij</sub></i>. The covariance matrix 
  can be decomposed into the sum of a <i>within-</i> (<i>W<sub>ij</sub></i>) 
  and a <i>between-class</i> (<i>B<sub>ij</sub></i>) class matrix. They describe
  the dispersion of events relative to the means of their own class (within-class
  matrix), and relative to the overall sample means (between-class matrix).
  The Fisher coefficients, <i>F<sub>i</sub></i>, are then given by <br>
  <center>
  <img vspace=6 src="gif/tmva_fisherC.gif" align="bottom" > 
  </center>
  where in TMVA is set <i>N<sub>S</sub>=N<sub>B</sub></i>, so that the factor 
  in front of the sum simplifies to &frac12;.
  The Fisher discriminant then reads<br>
  <center>
  <img vspace=6 src="gif/tmva_fisherD.gif" align="bottom" > 
  </center>
  The offset <i>F</i><sub>0</sub> centers the sample mean of <i>x</i><sub>Fi</sub>
  at zero. Instead of using the within-class matrix, the Mahalanobis variant
  determines the Fisher coefficients as follows:<br>
  <center>
  <img vspace=6 src="gif/tmva_mahaC.gif" align="bottom" > 
  </center>
  with resulting <i>x</i><sub>Ma</sub> that are very similar to the
  <i>x</i><sub>Fi</sub>. <br></p>

  TMVA provides two outputs for the ranking of the input variables:<br><p></p>
  <ul>
  <li> <u>Fisher test:</u> the Fisher analysis aims at simultaneously maximising 
  the between-class separation, while minimising the within-class dispersion.
  A useful measure of the discrimination power of a variable is hence given 
  by the diagonal quantity: <i>B<sub>ii</sub>/W<sub>ii</sub></i>.
  </li>

  <li> <u>Discrimination power:</u> the value of the Fisher coefficient is a 
  measure of the discriminating power of a variable. The discrimination power 
  of set of input variables can therefore be measured by the scalar
  <center>
  <img vspace=6 src="gif/tmva_discpower.gif" align="bottom" > 
  </center>
  </li>
  </ul>      
  The corresponding numbers are printed on standard output.
*/
// End_Html
//_______________________________________________________________________

#include "Riostream.h"
#include <algorithm>
#include "TMVA/MethodFisher.h"
#include "TMVA/Tools.h"
#include "TMatrix.h"
#include "TMVA/Ranking.h"

ClassImp(TMVA::MethodFisher)

//_______________________________________________________________________
TMVA::MethodFisher::MethodFisher( TString jobName, TString methodTitle, DataSet& theData, 
                                  TString theOption, TDirectory* theTargetDir )
   : TMVA::MethodBase( jobName, methodTitle, theData, theOption, theTargetDir )
   , fTheMethod("Fisher")
{
   // standard constructor for the "Fisher" 
   InitFisher(); // sets default values

   DeclareOptions();

   ParseOptions();

   ProcessOptions();

   // this is the preparation for training
   if (HasTrainingTree()) {

      // count number of signal and background events
      Int_t Nsig = Data().GetNEvtSigTrain();
      Int_t Nbgd = Data().GetNEvtBkgdTrain();

      if (Verbose())
         cout << "--- " << GetName() << " <verbose>: num of events for training (signal, background): "
              << " (" << Nsig << ", " << Nbgd << ")" << endl;

      // Fisher wants same number of events in each species
      if (Nsig != Nbgd) {
         cout << "--- " << GetName() << ":\t--------------------------------------------------"
              << endl;
         cout << "--- " << GetName() << ":\tWarning: different number of signal and background\n"
              << "--- " << GetName() << " \tevents: Fisher training will not be optimal :-("
              << endl;
         cout << "--- " << GetName() << ":\t--------------------------------------------------"
              << endl;
      }      

      // allocate arrays 
      InitMatrices();
   }
}

//_______________________________________________________________________
TMVA::MethodFisher::MethodFisher( DataSet& theData, 
                                  TString theWeightFile,  
                                  TDirectory* theTargetDir )
   : TMVA::MethodBase( theData, theWeightFile, theTargetDir ) 
   , fTheMethod("Fisher")
{
   // constructor to calculate the Fisher-MVA from previously generatad 
   // coefficients (weight file)
   InitFisher();

   DeclareOptions();
}

//_______________________________________________________________________
void TMVA::MethodFisher::InitFisher( void )
{
   // default initialisation called by all constructors
   SetMethodName( "Fisher" );
   SetMethodType( TMVA::Types::Fisher );  
   SetTestvarName();

   fMeanMatx    = 0; 
   fBetw        = 0;
   fWith        = 0;
   fCov         = 0;

   // allocate Fisher coefficients
   fF0          = 0;
   fFisherCoeff = new vector<Double_t>( GetNvar() );

   // the minimum requirement to declare an event signal-like
   SetSignalReferenceCut( 0.0 );
}

void TMVA::MethodFisher::DeclareOptions() 
{
   //
   // MethodFisher options:
   // format and syntax of option string: "type"
   // where type is "Fisher" or "Mahalanobis"
   //
   DeclareOptionRef(fTheMethod,"Method","discrimination method");
   AddPreDefVal(TString("Fisher"));
   AddPreDefVal(TString("Mahalanobis"));
}

void TMVA::MethodFisher::ProcessOptions() 
{
   MethodBase::ProcessOptions();

   if (fTheMethod ==  "Fisher" ) fFisherMethod = kFisher;
   else                          fFisherMethod = kMahalanobis;
}

//_______________________________________________________________________
TMVA::MethodFisher::~MethodFisher( void )
{
   // destructor
   delete fBetw;
   delete fWith;
   delete fCov;
   delete fDiscrimPow;
   delete fFisherCoeff;
}

//_______________________________________________________________________
void TMVA::MethodFisher::Train( void )
{
   // computation of Fisher coefficients by series of matrix operations

   // default sanity checks
   if (!CheckSanity()) { 
      cout << "--- " << GetName() << ": Error: sanity check failed" << endl;
      exit(1);
   }

   // get mean value of each variables for signal, backgd and signal+backgd
   GetMean();   

   // get the matrix of covariance 'within class'
   GetCov_WithinClass();

   // get the matrix of covariance 'between class'
   GetCov_BetweenClass();

   // get the matrix of covariance 'between class'
   GetCov_Full();

   //--------------------------------------------------------------

   // get the Fisher coefficients
   GetFisherCoeff();

   // get the discriminating power of each variables
   GetDiscrimPower();

   // nice output
   PrintCoefficients();
}

//_______________________________________________________________________
Double_t TMVA::MethodFisher::GetMvaValue()
{
   // returns the Fisher value (no fixed range)
   Double_t result = fF0;
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) 
      result += (*fFisherCoeff)[ivar]*GetEventValNormalized(ivar);

   return result;
}

//_______________________________________________________________________
void TMVA::MethodFisher::InitMatrices( void )
{
   // initialisaton method; creates global matrices and vectors
   // should never be called without existing trainingTree
   if (!HasTrainingTree()) {
      cout << "--- " << GetName() << "::InitMatrices(): fatal error: Data().TrainingTree() is zero pointer"
           << " --> exit(1)" << endl;
      exit(1);
   }

   // average value of each variables for S, B, S+B
   fMeanMatx = new TMatrixD( GetNvar(), 3 );

   // the covariance 'within class' and 'between class' matrices
   fBetw = new TMatrixD( GetNvar(), GetNvar() );
   fWith = new TMatrixD( GetNvar(), GetNvar() );
   fCov  = new TMatrixD( GetNvar(), GetNvar() );

   // discriminating power
   fDiscrimPow = new vector<Double_t>( GetNvar() );
}

//_______________________________________________________________________
void TMVA::MethodFisher::GetMean( void )
{
   // compute mean values of variables in each sample, and the overall means
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {   

      Double_t sumS = 0;
      Double_t sumB = 0;
      for (Int_t ievt=0; ievt<Data().GetNEvtTrain(); ievt++) {

         // read the Training Event into "event"
         ReadTrainingEvent(ievt);

         Double_t value = GetEventValNormalized( ivar );

         // for weights ... use: Double_t weight = GetEventWeight();

         if (Data().Event().IsSignal()) sumS += value;
         else                           sumB += value;
      }

      (*fMeanMatx)( ivar, 2 ) = sumS;
      (*fMeanMatx)( ivar, 0 ) = sumS/Data().GetNEvtSigTrain();

      (*fMeanMatx)( ivar, 2 ) += sumB;
      (*fMeanMatx)( ivar, 1 ) = sumB/Data().GetNEvtBkgdTrain();       

      // signal + background
      (*fMeanMatx)( ivar, 2 ) /= Data().GetNEvtTrain();
   }  
}

//_______________________________________________________________________
void TMVA::MethodFisher::GetCov_WithinClass( void )
{
   // the matrix of covariance 'within class' reflects the dispersion of the
   // events relative to the center of gravity of their own class  

   // product matrices (x-<x>)(y-<y>) where x;y are variables

   // 'within class' covariance
   for (Int_t x=0; x<GetNvar(); x++) {
      for (Int_t y=0; y<GetNvar(); y++) {

         Double_t sumSig = 0;
         Double_t sumBgd = 0;

         for (Int_t ievt=0; ievt<Data().GetNEvtTrain(); ievt++) {

            // read the Training Event into "event"
            ReadTrainingEvent(ievt);

            // when using weights....
            //            Double_t weight = GetEventWeight();
            if (Data().Event().IsSignal()) {
               sumSig += ( (GetEventValNormalized( x ) - (*fMeanMatx)(x, 0))* 
                           (GetEventValNormalized( y ) - (*fMeanMatx)(y, 0)) );
            }
            else {
               sumBgd += ( (GetEventValNormalized( x ) - (*fMeanMatx)(x, 1))* 
                           (GetEventValNormalized( y ) - (*fMeanMatx)(y, 1)) );
            }
         }

         (*fWith)(x, y) = (sumSig + sumBgd)/Data().GetNEvtTrain();
      }
   }
}

//_______________________________________________________________________
void TMVA::MethodFisher::GetCov_BetweenClass( void )
{
   // the matrix of covariance 'between class' reflects the dispersion of the
   // events of a class relative to the global center of gravity of all the class
   // hence the separation between classes

   Double_t prodSig, prodBgd;

   for (Int_t x=0; x<GetNvar(); x++) {
      for (Int_t y=0; y<GetNvar(); y++) {

         prodSig = ( ((*fMeanMatx)(x, 0) - (*fMeanMatx)(x, 2))*
                     ((*fMeanMatx)(y, 0) - (*fMeanMatx)(y, 2)) );
         prodBgd = ( ((*fMeanMatx)(x, 1) - (*fMeanMatx)(x, 2))*
                     ((*fMeanMatx)(y, 1) - (*fMeanMatx)(y, 2)) );

         (*fBetw)(x, y) = ( (Data().GetNEvtSigTrain()*prodSig + Data().GetNEvtBkgdTrain()*prodBgd) 
                            / Double_t(Data().GetNEvtTrain()) );
      }
   }
}

//_______________________________________________________________________
void TMVA::MethodFisher::GetCov_Full( void )
{
   // compute full covariance matrix from sum of within and between matrices
   for (Int_t x=0; x<GetNvar(); x++) 
      for (Int_t y=0; y<GetNvar(); y++) 
         (*fCov)(x, y) = (*fWith)(x, y) + (*fBetw)(x, y);
}

//_______________________________________________________________________
void TMVA::MethodFisher::GetFisherCoeff( void )
{
   // Fisher = Sum { [coeff]*[variables] }
   //
   // let Xs be the array of the mean values of variables for signal evts
   // let Xb be the array of the mean values of variables for backgd evts
   // let InvWith be the inverse matrix of the 'within class' correlation matrix
   //
   // then the array of Fisher coefficients is 
   // [coeff] =sqrt(fNsig*fNbgd)/fNevt*transpose{Xs-Xb}*InvWith

   // invert covariance matrix
   TMatrixD* theMat = 0;
   switch (GetFisherMethod()) {
   case kFisher:
      theMat = fWith;
      break;
   case kMahalanobis:
      theMat = fCov;
      break;
   default:
      cout << "--- " << GetName() << ": ERROR: undefined method ==> exit(1)" << endl;
      exit(1);
   }

   TMatrixD invCov( *theMat );
   invCov.Invert();

   // apply rescaling factor
   Double_t xfact = ( sqrt( Double_t(Data().GetNEvtSigTrain()*Data().GetNEvtBkgdTrain()) )
                      / Double_t(Data().GetNEvtTrain()) );

   // compute difference of mean values
   vector<Double_t> diffMeans( GetNvar() );
   Int_t ivar, jvar;
   for (ivar=0; ivar<GetNvar(); ivar++) {
      (*fFisherCoeff)[ivar] = 0;

      for(jvar=0; jvar<GetNvar(); jvar++) {
         Double_t d = (*fMeanMatx)(jvar, 0) - (*fMeanMatx)(jvar, 1);
         (*fFisherCoeff)[ivar] += invCov(ivar, jvar)*d;
      }    
    
      // rescale
      (*fFisherCoeff)[ivar] *= xfact;
   }

   // offset correction
   fF0 = 0.0;
   for(ivar=0; ivar<GetNvar(); ivar++) 
      fF0 += (*fFisherCoeff)[ivar]*((*fMeanMatx)(ivar, 0) + (*fMeanMatx)(ivar, 1));
   fF0 /= -2.0;  
}

//_______________________________________________________________________
void TMVA::MethodFisher::GetDiscrimPower( void )
{
   // computation of discrimination power indicator for each variable
   // small values of "fWith" indicates little compactness of sig & of backgd
   // big values of "fBetw" indicates large separation between sig & backgd
   //
   // we want signal & backgd classes as compact and separated as possible
   // the discriminating power is then defined as the ration "fBetw/fWith"
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
      if ((*fCov)(ivar, ivar) != 0) 
         (*fDiscrimPow)[ivar] = (*fBetw)(ivar, ivar)/(*fCov)(ivar, ivar);
      else
         (*fDiscrimPow)[ivar] = 0;
   }
}

//_______________________________________________________________________
const TMVA::Ranking* TMVA::MethodFisher::CreateRanking() 
{
   // computes ranking of input variables

   // create the ranking object
   fRanking = new Ranking( GetName(), "Discr. power" );

   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
      fRanking->AddRank( *new Rank( GetInputExp(ivar), (*fDiscrimPow)[ivar] ) );
   }

   return fRanking;
}

//_______________________________________________________________________
void TMVA::MethodFisher::PrintCoefficients( void ) 
{
   // display Fisher coefficients and discriminating power for each variable
   // check maximum length of variable name
   Int_t maxL = 0; 
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
      if ((*fInputVars)[ivar].Length() > maxL) maxL = (*fInputVars)[ivar].Length();
   }

   cout << "--- " << GetName() << ": results" << endl;
   cout << "-------------------------------" << endl;
   cout << "--- " << setiosflags(ios::left) << setw(TMath::Max(maxL,10)) << "Variable :"
        << " Coefficient:"
        << resetiosflags(ios::right) << endl;
   cout << "-------------------------------" << endl;
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
      cout << Form( "--- %-11s:   %+.3f", GetInputExp(ivar).Data(), (*fFisherCoeff)[ivar] ) 
           << endl;
   }

   cout << Form( "--- %-11s:   %+.3f", "(offset)", fF0 ) << endl;
   cout << "-------------------------------" << endl;
}
  
//_______________________________________________________________________
void  TMVA::MethodFisher::WriteWeightsToStream( ostream & o ) const
{  
   // save the weights
   o << fF0 << endl;
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) o << (*fFisherCoeff)[ivar] << endl;
}
  

//_______________________________________________________________________
void  TMVA::MethodFisher::ReadWeightsFromStream( istream & istr )
{
   // read Fisher coefficients from weight file
   istr >> fF0;
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) istr >> (*fFisherCoeff)[ivar];
}

//_______________________________________________________________________
void  TMVA::MethodFisher::WriteHistosToFile( void ) const
{
   // write special monitoring histograms to file - not implemented for Fisher
   cout << "--- " << GetName() << ": no monitoring histograms written" << endl;
}
