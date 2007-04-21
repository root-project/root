// @(#)root/tmva $Id: MethodLikelihood.cxx,v 1.17 2007/04/19 06:53:02 brun Exp $ 
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate Data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::MethodLikelihood                                                *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
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

//_______________________________________________________________________
//Begin_Html
/*
  Likelihood analysis ("non-parametric approach")                      
  
  <p>
  Also implemented is a "diagonalized likelihood approach",            
  which improves over the uncorrelated likelihood ansatz by            
  transforming linearly the input variables into a diagonal space,     
  using the square-root of the covariance matrix                       

  <p>                                                                  
  The method of maximum likelihood is the most straightforward, and 
  certainly among the most elegant multivariate analyser approaches.
  We define the likelihood ratio, <i>R<sub>L</sub></i>, for event 
  <i>i</i>, by:<br>
  <center>
  <img vspace=6 src="gif/tmva_likratio.gif" align="bottom" > 
  </center>
  Here the signal and background likelihoods, <i>L<sub>S</sub></i>, 
  <i>L<sub>B</sub></i>, are products of the corresponding probability 
  densities, <i>p<sub>S</sub></i>, <i>p<sub>B</sub></i>, of the 
  <i>N</i><sub>var</sub> discriminating variables used in the MVA: <br>
  <center>
  <img vspace=6 src="gif/tmva_lik.gif" align="bottom" > 
  </center>
  and accordingly for L<sub>B</sub>.
  In practise, TMVA uses polynomial splines to estimate the probability 
  density functions (PDF) obtained from the distributions of the 
  training variables.<br>

  <p>
  Note that in TMVA the output of the likelihood ratio is transformed
  by<br> 
  <center>
  <img vspace=6 src="gif/tmva_likratio_trans.gif" align="bottom"/> 
  </center>
  to avoid the occurrence of heavy peaks at <i>R<sub>L</sub></i>=0,1.   

  <b>Decorrelated (or "diagonalized") Likelihood</b>

  <p>
  The biggest drawback of the Likelihood approach is that it assumes
  that the discriminant variables are uncorrelated. If it were the case,
  it can be proven that the discrimination obtained by the above likelihood
  ratio is optimal, ie, no other method can beat it. However, in most 
  practical applications of MVAs correlations are present. <br><p></p>

  <p>
  Linear correlations, measured from the training sample, can be taken 
  into account in a straightforward manner through the square-root
  of the covariance matrix. The square-root of a matrix
  <i>C</i> is the matrix <i>C&prime;</i> that multiplied with itself
  yields <i>C</i>: <i>C</i>=<i>C&prime;C&prime;</i>. We compute the 
  square-root matrix (SQM) by means of diagonalising (<i>D</i>) the 
  covariance matrix: <br>
  <center>
  <img vspace=6 src="gif/tmva_sqm.gif" align="bottom" > 
  </center>
  and the linear transformation of the linearly correlated into the 
  uncorrelated variables space is then given by multiplying the measured
  variable tuple by the inverse of the SQM. Note that these transformations
  are performed for both signal and background separately, since the
  correlation pattern is not the same in the two samples.

  <p>
  The above diagonalisation is complete for linearly correlated,
  Gaussian distributed variables only. In real-world examples this 
  is not often the case, so that only little additional information
  may be recovered by the diagonalisation procedure. In these cases,
  non-linear methods must be applied.
*/
//End_Html
//_______________________________________________________________________

#include <cmath>
#include <vector>
#include "TMatrixD.h"
#include "TVector.h"
#include "TMath.h"
#include "TObjString.h"
#include "TFile.h"
#include "TKey.h"
#include "TH1.h"
#include "TClass.h"

#ifndef ROOT_TMVA_MethodLikelihood
#include "TMVA/MethodLikelihood.h"
#endif
#ifndef ROOT_TMVA_Tools
#include "TMVA/Tools.h"
#endif

ClassImp(TMVA::MethodLikelihood)

//_______________________________________________________________________
TMVA::MethodLikelihood::MethodLikelihood( TString jobName, TString methodTitle, DataSet& theData, 
                                          TString theOption, TDirectory* theTargetDir )
      : TMVA::MethodBase( jobName, methodTitle, theData, theOption, theTargetDir )
{
   // standard constructor
   //
   // MethodLikelihood options:
   // format and syntax of option string: "Spline2:0:25:D"
   //
   // where:
   //  SplineI [I=0,12,3,5] - which spline is used for smoothing the pdfs
   //                    0  - how often the input histos are smoothed
   //                    25 - average num of events per PDF bin 
   //                    D  - use square-root-matrix to decorrelate variable space 
   // 
   InitLik();

   DeclareOptions();

   ParseOptions();

   ProcessOptions();
}

//_______________________________________________________________________
TMVA::MethodLikelihood::MethodLikelihood( DataSet& theData, 
                                          TString theWeightFile,  
                                          TDirectory* theTargetDir )
   : TMVA::MethodBase( theData, theWeightFile, theTargetDir ) 
{  
   // construct likelihood references from file
   InitLik();

   DeclareOptions();
}

//_______________________________________________________________________
void TMVA::MethodLikelihood::InitLik( void )
{
   // default initialisation called by all constructors
   fHistBgd        = NULL; 
   fHistSig_smooth = NULL; 
   fHistBgd_smooth = NULL;
   fPDFSig         = NULL;
   fPDFBgd         = NULL;
  
   // no ranking test
   fDropVariable   = -1;

   SetMethodName( "Likelihood" );
   SetMethodType( TMVA::Types::kLikelihood );
   SetTestvarName();

   fEpsilon        = 1e-8;

   fHistSig        = new vector<TH1*>      ( GetNvar() ); 
   fHistBgd        = new vector<TH1*>      ( GetNvar() ); 
   fHistSig_smooth = new vector<TH1*>      ( GetNvar() ); 
   fHistBgd_smooth = new vector<TH1*>      ( GetNvar() );
   fPDFSig         = new vector<TMVA::PDF*>( GetNvar() );
   fPDFBgd         = new vector<TMVA::PDF*>( GetNvar() );

   fSpline         = -1;
   fUseKDE         = kFALSE;
}

//_______________________________________________________________________
void TMVA::MethodLikelihood::DeclareOptions() 
{
   // define the options (their key words) that can be set in the option string 
   // know options:
   // Spline            <int>    spline used to interpolate reference histograms
   //    available values are:        0, 1, 2 <default>, 3, 5
   //
   // NSmooth           <int>    how often the input histos are smoothed
   // NAvEvtPerBin      <int>    minimum average number of events per PDF bin 
   // TransformOutput   <bool>   transform (often strongly peaked) likelihood output through sigmoid inversion
   // fUseKDE           <bool>   T --> use Nonparametric Kernel Density Estimation for smoothing the PDFs
   // fKDEtype          <KernelType>   type of the Kernel to use (1 is Gaussian)
   // fKDEiter          <KerneIter>    number of iterations (1 --> "static KDE", 2 --> "adaptive KDE")
   // fBorderMethod     <KernelBorder> the method to take care about "border" effects (1=no treatment , 2=kernel renormalization, 3=sample mirroring)
   
   DeclareOptionRef(fSpline=2,"Spline","Order of splines used to interpolate reference histograms");
   AddPreDefVal(0); // take histogram
   AddPreDefVal(1); // linear interpolation between bins
   AddPreDefVal(2); // quadratic interpolation
   AddPreDefVal(3); // cubic interpolation
   AddPreDefVal(5); // fifth order polynome interpolation

   // initialize 
   DeclareOptionRef( fNsmooth = 1, "NSmooth",
                     "Number of smoothing iterations for the input histograms");

   fNsmoothVarS = new Int_t[GetNvar()];
   fNsmoothVarB = new Int_t[GetNvar()];
   for(Int_t ivar=0; ivar<GetNvar(); ivar++)
      fNsmoothVarS[ivar] = fNsmoothVarB[ivar] = -1;
   DeclareOptionRef(fNsmoothVarS, GetNvar(), "NSmoothSig",
                    "Number of smoothing iterations for the input histograms");
   DeclareOptionRef(fNsmoothVarB, GetNvar(), "NSmoothBkg",
                    "Number of smoothing iterations for the input histograms");

   DeclareOptionRef( fAverageEvtPerBin = 50, "NAvEvtPerBin",
                     "Average num of events per PDF bin");

   fAverageEvtPerBinVarS = new Int_t[GetNvar()];
   fAverageEvtPerBinVarB = new Int_t[GetNvar()];
   for(int ivar=0; ivar<GetNvar(); ivar++)
      fAverageEvtPerBinVarS[ivar] = fAverageEvtPerBinVarB[ivar] = -1;
   DeclareOptionRef( fAverageEvtPerBinVarS, GetNvar(), "NAvEvtPerBinSig",
                     "Average num of events per PDF bin and variable (signal)");   
   DeclareOptionRef( fAverageEvtPerBinVarB, GetNvar(), "NAvEvtPerBinBkg",
                     "Average num of events per PDF bin and variable (background)");   
   
   DeclareOptionRef( fTransformLikelihoodOutput = kFALSE, "TransformOutput", 
                     "Transform likelihood output by inverse sigmoid function" );

   DeclareOptionRef( fUseKDE        = kFALSE,        "UseKDE",  "Apply kernel density extimation" );

   DeclareOptionRef( fKDEtypeString = "Gauss",       "KDEtype", "KDE kernel type (1=Gauss)" );
   AddPreDefVal(TString("Gauss"));

   DeclareOptionRef( fKDEiterString = "Nonadaptive", "KDEiter", "#iterations (1=non-adaptive, 2=adaptive)" );
   AddPreDefVal(TString("Nonadaptive"));
   AddPreDefVal(TString("Adaptive"));

   DeclareOptionRef( fKDEfineFactor =1. , "KDEFineFactor", "Fine tuning factor for Adaptive KDE: Factor to multyply the width of the kernel");

   DeclareOptionRef( fBorderMethodString = "None", "KDEborder", "Border effects treatment (1=no treatment , 2=kernel renormalization, 3=sample mirroring)" );
   AddPreDefVal(TString("None"));
   AddPreDefVal(TString("Renorm"));
   AddPreDefVal(TString("Mirror"));
}

//_______________________________________________________________________
void TMVA::MethodLikelihood::ProcessOptions() 
{
   // process user options
   MethodBase::ProcessOptions();
   
   // test if to use kernel density estimation, if not fall back to splines
   if      (fUseKDE)      fInterpolateMethod = TMVA::PDF::kKDE;  
   else if (fSpline == 0) fInterpolateMethod = TMVA::PDF::kSpline0;
   else if (fSpline == 1) fInterpolateMethod = TMVA::PDF::kSpline1;
   else if (fSpline == 2) fInterpolateMethod = TMVA::PDF::kSpline2;      
   else if (fSpline == 3) fInterpolateMethod = TMVA::PDF::kSpline3;      
   else if (fSpline == 5) fInterpolateMethod = TMVA::PDF::kSpline5;
   else {
      fLogger << kWARNING << "unknown Spline type! Choose Spline2" << Endl;
      fInterpolateMethod = TMVA::PDF::kSpline2;
   }

   // set variable-specific options
   for(int ivar=0; ivar<GetNvar(); ivar++) {
      if (fNsmoothVarS[ivar]          == -1) fNsmoothVarS[ivar]          = fNsmooth;
      if (fNsmoothVarB[ivar]          == -1) fNsmoothVarB[ivar]          = fNsmooth;
      if (fAverageEvtPerBinVarS[ivar] == -1) fAverageEvtPerBinVarS[ivar] = fAverageEvtPerBin;
      if (fAverageEvtPerBinVarB[ivar] == -1) fAverageEvtPerBinVarB[ivar] = fAverageEvtPerBin;
   }

   // some gossip
   if (fUseKDE) fLogger << kINFO << "choose KDE kernel estimator for PDF interpolation" << Endl;
   
   // init KDE options
   if      (fKDEtypeString == "Gauss"      ) fKDEtype = KDEKernel::kGauss;
   else // nothing more known
      fLogger << kFATAL << "unknown setting for option 'KDEtype': " << fKDEtypeString << Endl;
   if      (fKDEiterString == "Nonadaptive") fKDEiter = KDEKernel::kNonadaptiveKDE;
   else if (fKDEiterString == "Adaptive"   ) fKDEiter = KDEKernel::kAdaptiveKDE;
   else // nothing more known
      fLogger << kFATAL << "unknown setting for option 'KDEiter': " << fKDEiterString << Endl;
   
   if       ( fBorderMethodString == "None"   ) fBorderMethod= KDEKernel::kNoTreatment;
   else if  ( fBorderMethodString == "Renorm" ) fBorderMethod= KDEKernel::kKernelRenorm;
   else if  ( fBorderMethodString == "Mirror" ) fBorderMethod= KDEKernel::kSampleMirror;
   else // nothing more known
      fLogger << kFATAL << "unknown setting for option 'KDEborder': " << fKDEiterString << Endl;
   
   // decorrelate option will be last option, if it is specified
   if      (GetVariableTransform() == Types::kDecorrelated)
      fLogger << kINFO << "use decorrelated variable set" << Endl;
   else if (GetVariableTransform() == Types::kPCA)
      fLogger << kINFO << "use principal component transformation" << Endl;

   // reference cut value to distingiush signal-like from background-like events   
   SetSignalReferenceCut( TransformLikelihoodOutput( 0.5, 0.5 ) );
}

//_______________________________________________________________________
TMVA::MethodLikelihood::~MethodLikelihood( void )
{
   // destructor  
   if (NULL != fHistSig)        delete fHistSig;
   if (NULL != fHistBgd)        delete fHistBgd;
   if (NULL != fHistSig_smooth) delete fHistSig_smooth;
   if (NULL != fHistBgd_smooth) delete fHistBgd_smooth;
   if (NULL != fPDFSig)         delete fPDFSig;
   if (NULL != fPDFBgd)         delete fPDFBgd;
}

//_______________________________________________________________________
void TMVA::MethodLikelihood::Train( void )
{
   // create reference distributions (PDFs) from signal and background events:
   // fill histograms and smooth them; if decorrelation is required, compute 
   // corresponding square-root matrices

   // default sanity checks
   if (!CheckSanity()) { 
      fLogger << kFATAL << "sanity check failed" << Endl;
   }

   // create reference histograms

   // fine binned histos needed for the KDE smoothing
   std::vector<TH1*>* sigFineBinKDE = new vector<TH1*>( GetNvar() );
   std::vector<TH1*>* bgdFineBinKDE = new vector<TH1*>( GetNvar() );   
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {

      TString var = (*fInputVars)[ivar];

      // special treatment for discrete variables
      if (Data().VarTypeOriginal(ivar) == 'I') {
         // special treatment for integer variables
         Int_t xmin = TMath::Nint( GetXmin(ivar) );
         Int_t xmax = TMath::Nint( GetXmax(ivar) + 1 );
         Int_t nbins = xmax - xmin;

         (*fHistSig)[ivar] = new TH1F( var + "_sig", var + " signal training",     nbins, xmin, xmax );
         (*fHistBgd)[ivar] = new TH1F( var + "_bgd", var + " background training", nbins, xmin, xmax );
      }
      else {

         UInt_t minNEvt = TMath::Min(Data().GetNEvtSigTrain(),Data().GetNEvtBkgdTrain());
         UInt_t nbinsS = minNEvt/fAverageEvtPerBinVarS[ivar];
         UInt_t nbinsB = minNEvt/fAverageEvtPerBinVarB[ivar];

         (*fHistSig)[ivar] = new TH1F( var + "_sig", var + " signal training",     nbinsS, GetXmin(ivar), GetXmax(ivar));
         (*fHistBgd)[ivar] = new TH1F( var + "_bgd", var + " background training", nbinsB, GetXmin(ivar), GetXmax(ivar));
      }

      // book the fine binned histos needed for the KDE smoothing                                                      
      if (fInterpolateMethod == TMVA::PDF::kKDE) {
         UInt_t minNEvt = TMath::Min(Data().GetNEvtSigTrain(),Data().GetNEvtBkgdTrain());
         UInt_t nbinsS = minNEvt/fAverageEvtPerBinVarS[ivar];
         UInt_t nbinsB = minNEvt/fAverageEvtPerBinVarB[ivar];

         (*sigFineBinKDE)[ivar] = new TH1F( var + "_sig_KDE", var + " signal training KDE",     5*nbinsS, GetXmin(ivar), GetXmax(ivar));
         (*bgdFineBinKDE)[ivar] = new TH1F( var + "_bgd_KDE", var + " background training KDE", 5*nbinsB, GetXmin(ivar), GetXmax(ivar));
      }
   }

   // ----- fill the reference histograms

   fLogger << kINFO << "Filling reference histograms" << Endl;

   // event loop
   for (Int_t ievt=0; ievt<Data().GetNEvtTrain(); ievt++) {

      // use the true-event-type's transformation
      ReadTrainingEvent( ievt, Types::kTrueType ); 

      // the event weight
      Float_t weight = GetEventWeight();

      // fill variable vector
      for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
         Float_t value  = GetEventVal(ivar);
         if (GetEvent().IsSignal()) {
            (*fHistSig)[ivar]->Fill( value, weight );
            // fill the fine binned signal histos needed for the KDE smoothing
            if (fInterpolateMethod == TMVA::PDF::kKDE) (*sigFineBinKDE)[ivar]->Fill( value, weight );
         } 
         else {
            (*fHistBgd)[ivar]->Fill( value, weight );
            // fill the fine binned signal histos needed for the KDE smoothing
            if (fInterpolateMethod == TMVA::PDF::kKDE) (*bgdFineBinKDE)[ivar]->Fill( value, weight );
         }
      }
   }

   // apply smoothing, and create PDFs
   for (UInt_t itype=0; itype < 2; itype++) { // signal and background

      vector<TH1*>& histV    = itype==0 ? *fHistSig : *fHistBgd;
      vector<TH1*>& histVKDE = itype==0 ? *sigFineBinKDE : *bgdFineBinKDE;

      vector<TH1*>& vHistSmo = itype==0 ? *fHistSig_smooth : *fHistBgd_smooth;
      vector<PDF*>& vPDF     = itype==0 ? *fPDFSig : *fPDFBgd;
      
      for (Int_t ivar=0; ivar<GetNvar(); ivar++) { 

         Int_t nsmooth = (itype==0) ? fNsmoothVarS[ivar] : fNsmoothVarB[ivar];
         TH1*  htmp    = (TH1*)histV[ivar]->Clone( Form("%s_smooth", histV[ivar]->GetName()) );
         htmp->SetTitle( Form("%s smoothed %i times",htmp->GetTitle(), nsmooth) );

         // --- smooth histogram and create PDF

         // if the variable is discrete, use histogram (=kSpline0) as reference 
         // (and no smoothing is applied!)
         TMVA::PDF* ptmp = 0;
         if (Data().VarTypeOriginal(ivar) == 'I') {
            ptmp =  new TMVA::PDF( htmp, PDF::kSpline0, 0 );
         } 
         else {
            if (htmp->GetNbinsX() <= 2 && nsmooth > 0)
               fLogger << kWARNING << "histogram "<< htmp->GetName() 
                       << " does not have enough (" << htmp->GetNbinsX()
                       << ") bins for for smoothing " << Endl;
            if (fInterpolateMethod == TMVA::PDF::kKDE) {
               ptmp = new TMVA::PDF( histVKDE[ivar], fKDEtype, fKDEiter, fBorderMethod, fKDEfineFactor );
            } 
            else {
               ptmp = new TMVA::PDF( htmp, fInterpolateMethod, nsmooth );  
            }
         }

         vPDF[ivar]     = ptmp;
         vHistSmo[ivar] = ptmp->GetSmoothedHist();
          
         // validate histogram
         ptmp->ValidatePDF( histV[ivar] );

         // temporary histogram can be deleted
         delete htmp;
      }
   }

   // clean up the mess
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
      delete (*sigFineBinKDE)[ivar];
      delete (*bgdFineBinKDE)[ivar];
   }   
}

//_______________________________________________________________________
Double_t TMVA::MethodLikelihood::GetMvaValue()
{
   // returns the likelihood estimator for signal

   // fill a new Likelihood branch into the testTree
   Int_t ivar;
    
   // retrieve variables, and transform, if required
   TVector vs( GetNvar() );
   TVector vb( GetNvar() );

   // need to distinguish signal and background in case of variable transformation
   // signal first
   GetVarTransform().ApplyTransformation(Types::kSignal);
   for (ivar=0; ivar<GetNvar(); ivar++) vs(ivar) = GetEventVal(ivar);

   GetVarTransform().ApplyTransformation(Types::kBackground);
   for (ivar=0; ivar<GetNvar(); ivar++) vb(ivar) = GetEventVal(ivar);

   // compute the likelihood (signal)
   Double_t ps(1), pb(1);
   for (ivar=0; ivar<GetNvar(); ivar++) {

      // drop one variable (this is ONLY used for internal variable ranking !)
      if (ivar == fDropVariable) continue;

      Double_t x[2] = { vs(ivar), vb(ivar) };
    
      for (UInt_t itype=0; itype < 2; itype++) {

         // find corresponding histogram from cached indices                 
         PDF* pdf = (itype == 0) ? (*fPDFSig)[ivar]:(*fPDFBgd)[ivar];
         if(pdf==0)
            fLogger << kFATAL << "Reference histograms don't exist" << Endl;
         TH1* hist = pdf->GetPDFHist();

         // interpolate linearly between adjacent bins
         // this is not useful for discrete variables
         Int_t bin = hist->FindBin(x[itype]);
         Double_t p;
         if (fInterpolateMethod == TMVA::PDF::kSpline0) {
            p = TMath::Max( hist->GetBinContent(bin), fEpsilon );
         }
         else { // splined PDF

            Int_t nextbin = bin;
            if ((x[itype] > hist->GetBinCenter(bin)&& bin != hist->GetNbinsX()) || bin == 1) 
               nextbin++;
            else
               nextbin--;  
            
            Double_t dx   = hist->GetBinCenter(bin)  - hist->GetBinCenter(nextbin);
            Double_t dy   = hist->GetBinContent(bin) - hist->GetBinContent(nextbin);
            Double_t like = hist->GetBinContent(bin) + (x[itype] - hist->GetBinCenter(bin)) * dy/dx;
            p = TMath::Max( like, fEpsilon );
         }

         if (itype == 0) ps *= p;
         else            pb *= p;
      }            
   }     

   // the likelihood ratio (transform it ?)
   return TransformLikelihoodOutput( ps, pb );
}

//_______________________________________________________________________
Double_t TMVA::MethodLikelihood::TransformLikelihoodOutput( Double_t ps, Double_t pb )
{
   // returns transformed or non-transformed output
   if (ps + pb < fEpsilon) pb = fEpsilon;
   Double_t r = ps/(ps + pb);

   if (fTransformLikelihoodOutput) {
      // inverse Fermi function

      // sanity check
      if      (r <= 0.0) r = fEpsilon;
      else if (r >= 1.0) r = 1.0 - fEpsilon;

      Double_t tau = 15.0;
      r = - TMath::Log(1.0/r - 1.0)/tau;
   }

   return r;
}

//_______________________________________________________________________
const TMVA::Ranking* TMVA::MethodLikelihood::CreateRanking() 
{
   // computes ranking of input variables

   // create the ranking object
   fRanking = new Ranking( GetName(), "Delta Separation" );

   Double_t sepRef = -1, sep = -1;
   for (Int_t ivar=-1; ivar<GetNvar(); ivar++) {

      // this variable should not be used
      fDropVariable = ivar;
      
      TString nameS = Form( "rS_%i", ivar+1 );
      TString nameB = Form( "rB_%i", ivar+1 );
      TH1* rS = new TH1F( nameS, nameS, 80, 0, 1 );
      TH1* rB = new TH1F( nameB, nameB, 80, 0, 1 );

      // the event loop
      for (Int_t ievt=0; ievt<Data().GetNEvtTrain(); ievt++) {
         
         ReadTrainingEvent( ievt, Types::kTrueType ); 

         Double_t lk = this->GetMvaValue();
         if (GetEvent().IsSignal()) rS->Fill( lk );
         else                       rB->Fill( lk );
      }

      // compute separation
      sep = TMVA::Tools::GetSeparation( rS, rB );
      if (ivar == -1) sepRef = sep;
      sep = sepRef - sep;
      
      // don't need these histograms anymore
      delete rS;
      delete rB;

      if (ivar >= 0) fRanking->AddRank( *new Rank( GetInputExp(ivar), sep ) );
   }

   fDropVariable = -1;
   
   return fRanking;
}

//_______________________________________________________________________
void  TMVA::MethodLikelihood::WriteWeightsToStream( ostream& o ) const
{  
   // write weights to stream 

   if(TxtWeightsOnly()) {
      for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
         if( (*fPDFSig)[ivar]==0 || (*fPDFBgd)[ivar]==0 )
            fLogger << kFATAL << "Reference histograms for variable " << ivar << " don't exist, can't write it to weight file" << Endl;
         o << *(*fPDFSig)[ivar];
         o << *(*fPDFBgd)[ivar];
      }
   } else {
      TString rfname( GetWeightFileName() ); rfname.ReplaceAll( ".txt", ".root" );
      o << "# weights stored in root i/o file: " << rfname << endl;  
   }
}

//_______________________________________________________________________
void  TMVA::MethodLikelihood::WriteWeightsToStream( TFile& ) const
{
   // write reference PDFs to ROOT file
   TString pname = "PDF_";
   for (Int_t ivar=0; ivar<GetNvar(); ivar++){ 
      (*fPDFSig)[ivar]->Write( pname + GetInputVar( ivar ) + "_S" );
      (*fPDFBgd)[ivar]->Write( pname + GetInputVar( ivar ) + "_B" );
   }                  
}
  
//_______________________________________________________________________
void  TMVA::MethodLikelihood::ReadWeightsFromStream( istream & istr )
{
   // read weight info from file
   // nothing to do for this method
   TString pname = "PDF_";
   Bool_t addDirStatus = TH1::AddDirectoryStatus();
   TH1::AddDirectory(0); // this avoids the binding of the hists in TMVA::PDF to the current ROOT file
   for (Int_t ivar=0; ivar<GetNvar(); ivar++){ 
      if( (*fPDFSig)[ivar] !=0 ) delete (*fPDFSig)[ivar];
      if( (*fPDFBgd)[ivar] !=0 ) delete (*fPDFBgd)[ivar];
      (*fPDFSig)[ivar] = new PDF();
      (*fPDFBgd)[ivar] = new PDF();
      istr >> *(*fPDFSig)[ivar];
      istr >> *(*fPDFBgd)[ivar];
   }
   TH1::AddDirectory(addDirStatus);
}

//_______________________________________________________________________
void  TMVA::MethodLikelihood::ReadWeightsFromStream( TFile& rf )
{
   // read reference PDF from ROOT file
   TString pname = "PDF_";
   Bool_t addDirStatus = TH1::AddDirectoryStatus();
   TH1::AddDirectory(0); // this avoids the binding of the hists in TMVA::PDF to the current ROOT file
   for (Int_t ivar=0; ivar<GetNvar(); ivar++){ 
      (*fPDFSig)[ivar] = (TMVA::PDF*)rf.Get( Form( "PDF_%s_S", GetInputVar( ivar ).Data() ) );
      (*fPDFBgd)[ivar] = (TMVA::PDF*)rf.Get( Form( "PDF_%s_B", GetInputVar( ivar ).Data() ) );
   }                     
   TH1::AddDirectory(addDirStatus);

}

//_______________________________________________________________________
void  TMVA::MethodLikelihood::WriteMonitoringHistosToFile( void ) const
{
   // write histograms and PDFs to file for monitoring purposes

   fLogger << kINFO << "write monitoring histograms to file: " << BaseDir()->GetPath() << Endl;
  
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) { 
      (*fHistSig)[ivar]->Write();    
      (*fHistBgd)[ivar]->Write();
      (*fHistSig_smooth)[ivar]->Write();    
      (*fHistBgd_smooth)[ivar]->Write();
      (*fPDFSig)[ivar]->GetPDFHist()->Write();
      (*fPDFBgd)[ivar]->GetPDFHist()->Write();
  
      // add special plots to check the smoothing in the GetVal method
      Float_t xmin=((*fPDFSig)[ivar]->GetPDFHist()->GetXaxis())->GetXmin();
      Float_t xmax=((*fPDFSig)[ivar]->GetPDFHist()->GetXaxis())->GetXmax();
      TH1F* mm = new TH1F( (*fInputVars)[ivar]+"_additional_check",
                           (*fInputVars)[ivar]+"_additional_check", 15000, xmin, xmax );
      Double_t intBin = (xmax-xmin)/15000;
      for (Int_t bin=0; bin < 15000; bin++) {
         Double_t x = (bin + 0.5)*intBin + xmin;
         mm->SetBinContent(bin+1 ,(*fPDFSig)[ivar]->GetVal(x));
      }
      mm->Write();
   }
}
