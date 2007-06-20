// @(#)root/tmva $Id: MethodFisher.cxx,v 1.16 2007/06/19 13:26:21 brun Exp $
// Author: Andreas Hoecker, Xavier Prudent, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate Data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodFisher                                                          *
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
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         * 
 *      U. of Victoria, Canada                                                    * 
 *      MPI-K Heidelberg, Germany                                                 * 
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

#include <assert.h>
#include "TMath.h"
#include "TMVA/MethodFisher.h"
#include "TMVA/Tools.h"
#include "TMatrix.h"
#include "TMVA/Ranking.h"

ClassImp(TMVA::MethodFisher)

//_______________________________________________________________________
TMVA::MethodFisher::MethodFisher( TString jobName, TString methodTitle, DataSet& theData, 
                                  TString theOption, TDirectory* theTargetDir )
   : TMVA::MethodBase( jobName, methodTitle, theData, theOption, theTargetDir )
{
   // standard constructor for the "Fisher" 
   InitFisher(); // sets default values

   // interpretation of configuration option string
   DeclareOptions();
   ParseOptions();
   ProcessOptions();

   // this is the preparation for training
   if (HasTrainingTree()) InitMatrices();
}

//_______________________________________________________________________
TMVA::MethodFisher::MethodFisher( DataSet& theData, 
                                  TString theWeightFile,  
                                  TDirectory* theTargetDir )
   : TMVA::MethodBase( theData, theWeightFile, theTargetDir )
{
   // constructor to calculate the Fisher-MVA from previously generatad 
   // coefficients (weight file)
   InitFisher();

   DeclareOptions();
}

//_______________________________________________________________________
void TMVA::MethodFisher::InitFisher( void )
{
   // default initialization called by all constructors
   SetMethodName( "Fisher" );
   SetMethodType( TMVA::Types::kFisher );  
   SetTestvarName();

   // Fisher prefers normalised input variables
   SetNormalised( kTRUE );

   fMeanMatx    = 0; 
   fBetw        = 0;
   fWith        = 0;
   fCov         = 0;

   fSumOfWeightsS = fSumOfWeightsB = 0;

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
   DeclareOptionRef( fTheMethod = "Fisher", "Method", "Discrimination method" );
   AddPreDefVal(TString("Fisher"));
   AddPreDefVal(TString("Mahalanobis"));
}

void TMVA::MethodFisher::ProcessOptions() 
{
   // process user options
   MethodBase::ProcessOptions();

   if (fTheMethod ==  "Fisher" ) fFisherMethod = kFisher;
   else                          fFisherMethod = kMahalanobis;

   PrintOptions();
   CheckForUnusedOptions();
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
   if (!CheckSanity()) fLogger << kFATAL << "<Train> sanity check failed" << Endl;

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
     result += (*fFisherCoeff)[ivar]*GetEventVal(ivar);
   return result;
}

//_______________________________________________________________________
void TMVA::MethodFisher::InitMatrices( void )
{
   // initializaton method; creates global matrices and vectors
   // should never be called without existing trainingTree
   if (!HasTrainingTree()) {
      fLogger << kFATAL << "<InitMatrices> fatal error: Data().TrainingTree() is zero pointer" << Endl;
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

   // initialize internal sum-of-weights variables
   fSumOfWeightsS = 0;
   fSumOfWeightsB = 0;

   for (Int_t ievt=0; ievt<Data().GetNEvtTrain(); ievt++) {

      // read the training event
      ReadTrainingEvent(ievt);

      if (IsSignalEvent()) fSumOfWeightsS += GetEventWeight();
      else                 fSumOfWeightsB += GetEventWeight();
   }

   // init vectors
   Double_t *sumS = new Double_t[(const Int_t)GetNvar()];
   Double_t *sumB = new Double_t[(const Int_t)GetNvar()];
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) { sumS[ivar] = sumB[ivar] = 0; }   

   // compute sample means
   for (Int_t ievt=0; ievt<Data().GetNEvtTrain(); ievt++) {

      // read the Training Event into "event"
      ReadTrainingEvent(ievt);

      // sum of weights
      Double_t weight = GetEventWeight();
      if (IsSignalEvent()) fSumOfWeightsS += weight;
      else                 fSumOfWeightsB += weight;

      Double_t* sum = IsSignalEvent() ? sumS : sumB;

      for (Int_t ivar=0; ivar<GetNvar(); ivar++) sum[ivar] += GetEventVal( ivar )*weight;
   }

   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {   
      (*fMeanMatx)( ivar, 2 ) = sumS[ivar];
      (*fMeanMatx)( ivar, 0 ) = sumS[ivar]/fSumOfWeightsS;

      (*fMeanMatx)( ivar, 2 ) += sumB[ivar];
      (*fMeanMatx)( ivar, 1 ) = sumB[ivar]/fSumOfWeightsB;

      // signal + background
      (*fMeanMatx)( ivar, 2 ) /= (fSumOfWeightsS + fSumOfWeightsB);
   }  
   delete [] sumS;
   delete [] sumB;
}

//_______________________________________________________________________
void TMVA::MethodFisher::GetCov_WithinClass( void )
{
   // the matrix of covariance 'within class' reflects the dispersion of the
   // events relative to the center of gravity of their own class  

   // assert required
   assert( fSumOfWeightsS > 0 && fSumOfWeightsB > 0 );

   // product matrices (x-<x>)(y-<y>) where x;y are variables

   // init
   const Int_t nvar = GetNvar();
   const Int_t nvar2 = nvar*nvar;
   Double_t *sumSig = new Double_t[nvar2];
   Double_t *sumBgd = new Double_t[nvar2];
   Double_t *xval   = new Double_t[nvar];
   memset(sumSig,0,nvar2*sizeof(Double_t));
   memset(sumBgd,0,nvar2*sizeof(Double_t));

   Int_t k=0;
   
   // 'within class' covariance
   for (Int_t ievt=0; ievt<Data().GetNEvtTrain(); ievt++) {

      // read the Training Event into "event"
      ReadTrainingEvent(ievt);

      Double_t weight = GetEventWeight();

      for (Int_t x=0; x<nvar; x++) xval[x] = GetEventVal( x );
      Int_t k=0;
      for (Int_t x=0; x<nvar; x++) {
         for (Int_t y=0; y<nvar; y++) {            
            Double_t v = ( (xval[x] - (*fMeanMatx)(x, 0))*(xval[y] - (*fMeanMatx)(y, 0)) )*weight;
            if (IsSignalEvent()) sumSig[k] += v;
            else                 sumBgd[k] += v;
            k++;
         }
      }
   }
   k=0;
   for (Int_t x=0; x<nvar; x++) {
      for (Int_t y=0; y<nvar; y++) {
         (*fWith)(x, y) = (sumSig[k] + sumBgd[k])/(fSumOfWeightsS + fSumOfWeightsB);
         k++;
      }
   }

   delete [] sumSig;
   delete [] sumBgd;
   delete [] xval;
}

//_______________________________________________________________________
void TMVA::MethodFisher::GetCov_BetweenClass( void )
{
   // the matrix of covariance 'between class' reflects the dispersion of the
   // events of a class relative to the global center of gravity of all the class
   // hence the separation between classes

   // assert required
   assert( fSumOfWeightsS > 0 && fSumOfWeightsB > 0);

   Double_t prodSig, prodBgd;

   for (Int_t x=0; x<GetNvar(); x++) {
      for (Int_t y=0; y<GetNvar(); y++) {

         prodSig = ( ((*fMeanMatx)(x, 0) - (*fMeanMatx)(x, 2))*
                     ((*fMeanMatx)(y, 0) - (*fMeanMatx)(y, 2)) );
         prodBgd = ( ((*fMeanMatx)(x, 1) - (*fMeanMatx)(x, 2))*
                     ((*fMeanMatx)(y, 1) - (*fMeanMatx)(y, 2)) );

         (*fBetw)(x, y) = (fSumOfWeightsS*prodSig + fSumOfWeightsB*prodBgd) / (fSumOfWeightsS + fSumOfWeightsB);
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

   // assert required
   assert( fSumOfWeightsS > 0 && fSumOfWeightsB > 0);

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
      fLogger << kFATAL << "<GetFisherCoeff> undefined method" << GetFisherMethod() << Endl;
   }

   TMatrixD invCov( *theMat );
   if ( TMath::Abs(invCov.Determinant()) < 10E-24 ) {
      fLogger << kWARNING << "<GetFisherCoeff> matrix is almost singular with deterninant="
              << TMath::Abs(invCov.Determinant()) 
              << " did you use the variables that are linear combinations or highly correlated?" 
              << Endl;
   }
   if ( TMath::Abs(invCov.Determinant()) < 10E-120 ) {
      fLogger << kFATAL << "<GetFisherCoeff> matrix is singular with determinant="
              << TMath::Abs(invCov.Determinant())  
              << " did you use the variables that are linear combinations?" 
              << Endl;
   }

   invCov.Invert();
   
   // apply rescaling factor
   Double_t xfact = TMath::Sqrt( fSumOfWeightsS*fSumOfWeightsB ) / (fSumOfWeightsS + fSumOfWeightsB);

   // compute difference of mean values
   vector<Double_t> diffMeans( GetNvar() );
   Int_t ivar, jvar;
   for (ivar=0; ivar<GetNvar(); ivar++) {
      (*fFisherCoeff)[ivar] = 0;

      for (jvar=0; jvar<GetNvar(); jvar++) {
         Double_t d = (*fMeanMatx)(jvar, 0) - (*fMeanMatx)(jvar, 1);
         (*fFisherCoeff)[ivar] += invCov(ivar, jvar)*d;
      }    
    
      // rescale
      (*fFisherCoeff)[ivar] *= xfact;
   }

   // offset correction
   fF0 = 0.0;
   for (ivar=0; ivar<GetNvar(); ivar++){ 
      fF0 += (*fFisherCoeff)[ivar]*((*fMeanMatx)(ivar, 0) + (*fMeanMatx)(ivar, 1));
   }
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
   fLogger << kINFO << "Results for Fisher coefficients:" << Endl;
   vector<TString>  vars;
   vector<Double_t> coeffs;
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
      vars  .push_back( GetInputExp(ivar) );
      coeffs.push_back(  (*fFisherCoeff)[ivar] );
   }
   vars  .push_back( "(offset)" );
   coeffs.push_back( fF0 );
   TMVA::Tools::FormattedOutput( coeffs, vars, "Variable" , "Coefficient", fLogger );   
}
  
//_______________________________________________________________________
void  TMVA::MethodFisher::WriteWeightsToStream( ostream& o ) const
{  
   // save the weights
   o << fF0 << endl;
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) o << setprecision(12) << (*fFisherCoeff)[ivar] << endl;
}
  
//_______________________________________________________________________
void  TMVA::MethodFisher::ReadWeightsFromStream( istream& istr )
{
   // read Fisher coefficients from weight file
   istr >> fF0;
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) istr >> (*fFisherCoeff)[ivar];
}

//_______________________________________________________________________
void TMVA::MethodFisher::MakeClassSpecific( std::ostream& fout, const TString& className ) const
{
   // write Fisher-specific classifier response
   fout << "   double              fFisher0;" << endl;
   fout << "   std::vector<double> fFisherCoefficients;" << endl;
   fout << "};" << endl;
   fout << "" << endl;
   fout << "inline void " << className << "::Initialize() " << endl;
   fout << "{" << endl;
   fout << "   fFisher0 = " << setprecision(12) << fF0 << ";" << endl;
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
      fout << "   fFisherCoefficients.push_back( " << setprecision(12) << (*fFisherCoeff)[ivar] << " );" << endl;
   }
   fout << endl;
   fout << "   // sanity check" << endl;
   fout << "   if (fFisherCoefficients.size() != fNvars) {" << endl;
   fout << "      std::cout << \"Problem in class \\\"\" << fClassName << \"\\\"::Initialize: mismatch in number of input values\"" << endl;
   fout << "                << fFisherCoefficients.size() << \" != \" << fNvars << std::endl;" << endl;
   fout << "      fStatusIsClean = false;" << endl;
   fout << "   }         " << endl;
   fout << "}" << endl;
   fout << endl;
   fout << "inline double " << className << "::GetMvaValue__( const std::vector<double>& inputValues ) const" << endl;
   fout << "{" << endl;
   fout << "   double retval = fFisher0;" << endl;
   fout << "   for (size_t ivar = 0; ivar < fNvars; ivar++) {" << endl;
   fout << "      retval += fFisherCoefficients[ivar]*inputValues[ivar];" << endl;
   fout << "   }" << endl;
   fout << endl;
   fout << "   return retval;" << endl;
   fout << "}" << endl;
   fout << endl;
   fout << "// Clean up" << endl;
   fout << "inline void " << className << "::Clear() " << endl;
   fout << "{" << endl;
   fout << "   // clear coefficients" << endl;
   fout << "   fFisherCoefficients.clear(); " << endl;
   fout << "}" << endl;
}

//_______________________________________________________________________
void TMVA::MethodFisher::GetHelpMessage() const
{
   // get help message text
   //
   // typical length of text line: 
   //         "|--------------------------------------------------------------|"
   fLogger << Endl;
   fLogger << Tools::Color("bold") << "--- Short description:" << Tools::Color("reset") << Endl;
   fLogger << Endl;
   fLogger << "Fisher discriminants select events by distinguishing the mean " << Endl;
   fLogger << "values of the signal and background distributions in a trans- " << Endl;
   fLogger << "formed variable space where linear correlations are removed." << Endl;
   fLogger << Endl;
   fLogger << "   (More precisely: the \"linear discriminator\" determines" << Endl;
   fLogger << "    an axis in the (correlated) hyperspace of the input " << Endl;
   fLogger << "    variables such that, when projecting the output classes " << Endl;
   fLogger << "    (signal and background) upon this axis, they are pushed " << Endl;
   fLogger << "    as far as possible away from each other, while events" << Endl;
   fLogger << "    of a same class are confined in a close vicinity. The  " << Endl;
   fLogger << "    linearity property of this classifier is reflected in the " << Endl;
   fLogger << "    metric with which \"far apart\" and \"close vicinity\" are " << Endl;
   fLogger << "    determined: the covariance matrix of the discriminating" << Endl;
   fLogger << "    variable space.)" << Endl;
   fLogger << Endl;
   fLogger << Tools::Color("bold") << "--- Performance optimisation:" << Tools::Color("reset") << Endl;
   fLogger << Endl;
   fLogger << "Optimal performance for Fisher discriminants is obtained for " << Endl;
   fLogger << "linearly correlated Gaussian-distributed variables. Any deviation" << Endl;
   fLogger << "from this ideal reduces the achievable separation power. In " << Endl;
   fLogger << "particular, no discrimination at all is achieved for a variable" << Endl;
   fLogger << "that has the same sample mean for signal and background, even if " << Endl;
   fLogger << "the shapes of the distributions are very different. Thus, Fisher " << Endl;
   fLogger << "discriminants often benefit from suitable transformations of the " << Endl;
   fLogger << "input variables. For example, if a variable x in [-1,1] has a " << Endl;
   fLogger << "a parabolic signal distributions, and a uniform background" << Endl;
   fLogger << "distributions, their mean value is zero in both cases, leading " << Endl;
   fLogger << "to no separation. The simple transformation x -> |x| renders this " << Endl;
   fLogger << "variable powerful for the use in a Fisher discriminant." << Endl;
   fLogger << Endl;
   fLogger << Tools::Color("bold") << "--- Performance tuning via configuration options:" << Tools::Color("reset") << Endl;
   fLogger << Endl;
   fLogger << "<None>" << Endl;
}

