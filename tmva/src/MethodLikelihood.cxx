// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss, Eckhard von Toerne, Jan Therhaag

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
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *      Jan Therhaag       <Jan.Therhaag@cern.ch>     - U of Bonn, Germany        *
 *      Eckhard v. Toerne  <evt@uni-bonn.de>     - U. of Bonn, Germany            *
 *                                                                                *
 * Copyright (c) 2005-2011:                                                       *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
/* Begin_Html

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

End_Html */
//_______________________________________________________________________

#include <iomanip>
#include <vector>
#include <cstdlib>

#include "TMatrixD.h"
#include "TVector.h"
#include "TMath.h"
#include "TObjString.h"
#include "TFile.h"
#include "TKey.h"
#include "TH1.h"
#include "TClass.h"
#include "Riostream.h"

#include "TMVA/ClassifierFactory.h"
#include "TMVA/MethodLikelihood.h"
#include "TMVA/Tools.h"
#include "TMVA/Ranking.h"

REGISTER_METHOD(Likelihood)

ClassImp(TMVA::MethodLikelihood)

//_______________________________________________________________________
TMVA::MethodLikelihood::MethodLikelihood( const TString& jobName,
                                          const TString& methodTitle,
                                          DataSetInfo& theData,
                                          const TString& theOption,
                                          TDirectory* theTargetDir ) :
   TMVA::MethodBase( jobName, Types::kLikelihood, methodTitle, theData, theOption, theTargetDir ),
   fEpsilon       ( 1.e3 * DBL_MIN ),
   fTransformLikelihoodOutput( kFALSE ),
   fDropVariable  ( 0 ),
   fHistSig       ( 0 ),
   fHistBgd       ( 0 ),
   fHistSig_smooth( 0 ),
   fHistBgd_smooth( 0 ),
   fDefaultPDFLik ( 0 ),
   fPDFSig        ( 0 ),
   fPDFBgd        ( 0 ),
   fNsmooth       ( 2 ),
   fNsmoothVarS   ( 0 ),
   fNsmoothVarB   ( 0 ),
   fAverageEvtPerBin( 0 ),
   fAverageEvtPerBinVarS (0), 
   fAverageEvtPerBinVarB (0),
   fKDEfineFactor ( 0 ),
   fInterpolateString(0)
{
   // standard constructor
}

//_______________________________________________________________________
TMVA::MethodLikelihood::MethodLikelihood( DataSetInfo& theData,
                                          const TString& theWeightFile,
                                          TDirectory* theTargetDir ) :
   TMVA::MethodBase( Types::kLikelihood, theData, theWeightFile, theTargetDir ),
   fEpsilon       ( 1.e3 * DBL_MIN ),
   fTransformLikelihoodOutput( kFALSE ),
   fDropVariable  ( 0 ),
   fHistSig       ( 0 ),
   fHistBgd       ( 0 ),
   fHistSig_smooth( 0 ),
   fHistBgd_smooth( 0 ),
   fDefaultPDFLik ( 0 ),
   fPDFSig        ( 0 ),
   fPDFBgd        ( 0 ),
   fNsmooth       ( 2 ),
   fNsmoothVarS   ( 0 ),
   fNsmoothVarB   ( 0 ),
   fAverageEvtPerBin( 0 ),
   fAverageEvtPerBinVarS (0), 
   fAverageEvtPerBinVarB (0),
   fKDEfineFactor ( 0 ),
   fInterpolateString(0)
{
   // construct likelihood references from file
}

//_______________________________________________________________________
TMVA::MethodLikelihood::~MethodLikelihood( void )
{
   // destructor
   if (NULL != fDefaultPDFLik)  delete fDefaultPDFLik;
   if (NULL != fHistSig)        delete fHistSig;
   if (NULL != fHistBgd)        delete fHistBgd;
   if (NULL != fHistSig_smooth) delete fHistSig_smooth;
   if (NULL != fHistBgd_smooth) delete fHistBgd_smooth;
   for (UInt_t ivar=0; ivar<GetNvar(); ivar++) {
      if ((*fPDFSig)[ivar] !=0) delete (*fPDFSig)[ivar];
      if ((*fPDFBgd)[ivar] !=0) delete (*fPDFBgd)[ivar];
   }
   if (NULL != fPDFSig)         delete fPDFSig;
   if (NULL != fPDFBgd)         delete fPDFBgd;
}

//_______________________________________________________________________
Bool_t TMVA::MethodLikelihood::HasAnalysisType( Types::EAnalysisType type,
                                                UInt_t numberClasses, UInt_t /*numberTargets*/ )
{
   // FDA can handle classification with 2 classes
   if (type == Types::kClassification && numberClasses == 2) return kTRUE;
   return kFALSE;
}

//_______________________________________________________________________
void TMVA::MethodLikelihood::Init( void )
{
   // default initialisation called by all constructors

   // no ranking test
   fDropVariable   = -1;
   fHistSig        = new std::vector<TH1*>      ( GetNvar(), (TH1*)0 );
   fHistBgd        = new std::vector<TH1*>      ( GetNvar(), (TH1*)0 );
   fHistSig_smooth = new std::vector<TH1*>      ( GetNvar(), (TH1*)0 );
   fHistBgd_smooth = new std::vector<TH1*>      ( GetNvar(), (TH1*)0 );
   fPDFSig         = new std::vector<TMVA::PDF*>( GetNvar(), (TMVA::PDF*)0 );
   fPDFBgd         = new std::vector<TMVA::PDF*>( GetNvar(), (TMVA::PDF*)0 );
}

//_______________________________________________________________________
void TMVA::MethodLikelihood::DeclareOptions()
{
   // define the options (their key words) that can be set in the option string
   // TransformOutput   <bool>   transform (often strongly peaked) likelihood output through sigmoid inversion

   DeclareOptionRef( fTransformLikelihoodOutput = kFALSE, "TransformOutput",
                     "Transform likelihood output by inverse sigmoid function" );

   // initialize

   // reading every PDF's definition and passing the option string to the next one to be read and marked
   TString updatedOptions = GetOptions();
   fDefaultPDFLik = new PDF( TString(GetName()) + " PDF", updatedOptions );
   fDefaultPDFLik->DeclareOptions();
   fDefaultPDFLik->ParseOptions();
   updatedOptions = fDefaultPDFLik->GetOptions();
   for (UInt_t ivar = 0; ivar< DataInfo().GetNVariables(); ivar++) {
      (*fPDFSig)[ivar] = new PDF( Form("%s PDF Sig[%d]", GetName(), ivar), updatedOptions,
                                  Form("Sig[%d]",ivar), fDefaultPDFLik );
      (*fPDFSig)[ivar]->DeclareOptions();
      (*fPDFSig)[ivar]->ParseOptions();
      updatedOptions = (*fPDFSig)[ivar]->GetOptions();
      (*fPDFBgd)[ivar] = new PDF( Form("%s PDF Bkg[%d]", GetName(), ivar), updatedOptions,
                                  Form("Bkg[%d]",ivar), fDefaultPDFLik );
      (*fPDFBgd)[ivar]->DeclareOptions();
      (*fPDFBgd)[ivar]->ParseOptions();
      updatedOptions = (*fPDFBgd)[ivar]->GetOptions();
   }

   // the final marked option string is written back to the original likelihood
   SetOptions( updatedOptions );
}


void TMVA::MethodLikelihood::DeclareCompatibilityOptions()
{
   // options that are used ONLY for the READER to ensure backward compatibility

   MethodBase::DeclareCompatibilityOptions();
   DeclareOptionRef( fNsmooth = 1, "NSmooth",
                     "Number of smoothing iterations for the input histograms");
   DeclareOptionRef( fAverageEvtPerBin = 50, "NAvEvtPerBin",
                     "Average number of events per PDF bin");
   DeclareOptionRef( fKDEfineFactor =1. , "KDEFineFactor",
                     "Fine tuning factor for Adaptive KDE: Factor to multyply the width of the kernel");
   DeclareOptionRef( fBorderMethodString = "None", "KDEborder",
                     "Border effects treatment (1=no treatment , 2=kernel renormalization, 3=sample mirroring)" );
   DeclareOptionRef( fKDEiterString = "Nonadaptive", "KDEiter",
                     "Number of iterations (1=non-adaptive, 2=adaptive)" );
   DeclareOptionRef( fKDEtypeString = "Gauss", "KDEtype",
                     "KDE kernel type (1=Gauss)" );
   fAverageEvtPerBinVarS = new Int_t[GetNvar()];
   fAverageEvtPerBinVarB = new Int_t[GetNvar()];
   fNsmoothVarS = new Int_t[GetNvar()];
   fNsmoothVarB = new Int_t[GetNvar()];
   fInterpolateString = new TString[GetNvar()];
   for(UInt_t i=0; i<GetNvar(); ++i) {
      fAverageEvtPerBinVarS[i] = fAverageEvtPerBinVarB[i] = 0;
      fNsmoothVarS[i] = fNsmoothVarB[i] = 0;
      fInterpolateString[i] = "";
   }
   DeclareOptionRef( fAverageEvtPerBinVarS, GetNvar(), "NAvEvtPerBinSig",
                     "Average num of events per PDF bin and variable (signal)");
   DeclareOptionRef( fAverageEvtPerBinVarB, GetNvar(), "NAvEvtPerBinBkg",
                     "Average num of events per PDF bin and variable (background)");
   DeclareOptionRef(fNsmoothVarS, GetNvar(), "NSmoothSig",
                    "Number of smoothing iterations for the input histograms");
   DeclareOptionRef(fNsmoothVarB, GetNvar(), "NSmoothBkg",
                    "Number of smoothing iterations for the input histograms");
   DeclareOptionRef(fInterpolateString, GetNvar(), "PDFInterpol", "Method of interpolating reference histograms (e.g. Spline2 or KDE)");
}




//_______________________________________________________________________
void TMVA::MethodLikelihood::ProcessOptions()
{

   // process user options
   // reference cut value to distingiush signal-like from background-like events
   SetSignalReferenceCut( TransformLikelihoodOutput( 0.5, 0.5 ) );

   fDefaultPDFLik->ProcessOptions();
   for (UInt_t ivar = 0; ivar< DataInfo().GetNVariables(); ivar++) {
      (*fPDFBgd)[ivar]->ProcessOptions();
      (*fPDFSig)[ivar]->ProcessOptions();
   }
}

//_______________________________________________________________________
void TMVA::MethodLikelihood::Train( void )
{
   // create reference distributions (PDFs) from signal and background events:
   // fill histograms and smooth them; if decorrelation is required, compute
   // corresponding square-root matrices
   // the reference histograms require the correct boundaries. Since in Likelihood classification
   // the transformations are applied using both classes, also the corresponding boundaries
   // need to take this into account
   UInt_t nvar=GetNvar();
   std::vector<Double_t> xmin(nvar), xmax(nvar);
   for (UInt_t ivar=0; ivar<nvar; ivar++) {xmin[ivar]=1e30; xmax[ivar]=-1e30;}

   UInt_t nevents=Data()->GetNEvents();
   for (UInt_t ievt=0; ievt<nevents; ievt++) {
      // use the true-event-type's transformation
      // set the event true event types transformation
      const Event* origEv = Data()->GetEvent(ievt);
      if (IgnoreEventsWithNegWeightsInTraining() && origEv->GetWeight()<=0) continue;
      // loop over classes
      for (int cls=0;cls<2;cls++){
         GetTransformationHandler().SetTransformationReferenceClass(cls);
         const Event* ev = GetTransformationHandler().Transform( origEv );
         for (UInt_t ivar=0; ivar<nvar; ivar++) {
            Float_t value  = ev->GetValue(ivar);
            if (value < xmin[ivar]) xmin[ivar] = value;
            if (value > xmax[ivar]) xmax[ivar] = value;
         }
      }
   }

   // create reference histograms
   // (KDE smoothing requires very finely binned reference histograms)
   for (UInt_t ivar=0; ivar<GetNvar(); ivar++) {
      TString var = (*fInputVars)[ivar];

      // the reference histograms require the correct boundaries. Since in Likelihood classification
      // the transformations are applied using both classes, also the corresponding boundaries
      // need to take this into account

      // special treatment for discrete variables
      if (DataInfo().GetVariableInfo(ivar).GetVarType() == 'I') {
         // special treatment for integer variables
         Int_t ixmin = TMath::Nint( xmin[ivar] );
         xmax[ivar]=xmax[ivar]+1; // make sure that all entries are included in histogram
         Int_t ixmax = TMath::Nint( xmax[ivar] );
         Int_t nbins = ixmax - ixmin;

         (*fHistSig)[ivar] = new TH1F( var + "_sig", var + " signal training",     nbins, ixmin, ixmax );
         (*fHistBgd)[ivar] = new TH1F( var + "_bgd", var + " background training", nbins, ixmin, ixmax );
      } else {

         UInt_t minNEvt = TMath::Min(Data()->GetNEvtSigTrain(),Data()->GetNEvtBkgdTrain());
         Int_t nbinsS = (*fPDFSig)[ivar]->GetHistNBins( minNEvt );
         Int_t nbinsB = (*fPDFBgd)[ivar]->GetHistNBins( minNEvt );

         (*fHistSig)[ivar] = new TH1F( Form("%s_sig",var.Data()),
                                       Form("%s signal training",var.Data()), nbinsS, xmin[ivar], xmax[ivar] );
         (*fHistBgd)[ivar] = new TH1F( Form("%s_bgd",var.Data()),
                                       Form("%s background training",var.Data()), nbinsB, xmin[ivar], xmax[ivar] );
      }
   }

   // ----- fill the reference histograms
   Log() << kINFO << "Filling reference histograms" << Endl;

   // event loop
   for (Int_t ievt=0; ievt<Data()->GetNEvents(); ievt++) {

      // use the true-event-type's transformation
      // set the event true event types transformation
      const Event* origEv = Data()->GetEvent(ievt);
      if (IgnoreEventsWithNegWeightsInTraining() && origEv->GetWeight()<=0) continue;
      GetTransformationHandler().SetTransformationReferenceClass( origEv->GetClass() );
      const Event* ev = GetTransformationHandler().Transform( origEv );

      // the event weight
      Float_t weight = ev->GetWeight();

      // fill variable vector
      for (UInt_t ivar=0; ivar<GetNvar(); ivar++) {
         Double_t value  = ev->GetValue(ivar);
         // verify limits
         if (value >= xmax[ivar]) value = xmax[ivar] - 1.0e-10;
         else if (value < xmin[ivar]) value = xmin[ivar] + 1.0e-10;
         // inserting check if there are events in overflow or underflow
         if (value >=(*fHistSig)[ivar]->GetXaxis()->GetXmax() ||
             value <(*fHistSig)[ivar]->GetXaxis()->GetXmin()){
            Log()<<kWARNING
                 <<"error in filling likelihood reference histograms var="
                 <<(*fInputVars)[ivar]
                 << ", xmin="<<(*fHistSig)[ivar]->GetXaxis()->GetXmin()
                 << ", value="<<value
                 << ", xmax="<<(*fHistSig)[ivar]->GetXaxis()->GetXmax()
                 << Endl;
         }
         if (DataInfo().IsSignal(ev)) (*fHistSig)[ivar]->Fill( value, weight );
         else                (*fHistBgd)[ivar]->Fill( value, weight );
      }
   }

   // building the pdfs
   Log() << kINFO << "Building PDF out of reference histograms" << Endl;
   for (UInt_t ivar=0; ivar<GetNvar(); ivar++) {

      // the PDF is built from (binned) reference histograms
      // in case of KDE, this has a large number of bins, which makes it quasi-unbinned
      (*fPDFSig)[ivar]->BuildPDF( (*fHistSig)[ivar] );
      (*fPDFBgd)[ivar]->BuildPDF( (*fHistBgd)[ivar] );

      (*fPDFSig)[ivar]->ValidatePDF( (*fHistSig)[ivar] );
      (*fPDFBgd)[ivar]->ValidatePDF( (*fHistBgd)[ivar] );

      // saving the smoothed histograms
      if ((*fPDFSig)[ivar]->GetSmoothedHist() != 0) (*fHistSig_smooth)[ivar] = (*fPDFSig)[ivar]->GetSmoothedHist();
      if ((*fPDFBgd)[ivar]->GetSmoothedHist() != 0) (*fHistBgd_smooth)[ivar] = (*fPDFBgd)[ivar]->GetSmoothedHist();
   }
}

//_______________________________________________________________________
Double_t TMVA::MethodLikelihood::GetMvaValue( Double_t* err, Double_t* errUpper )
{
   // returns the likelihood estimator for signal
   // fill a new Likelihood branch into the testTree
   UInt_t ivar;

   // cannot determine error
   NoErrorCalc(err, errUpper);

   // retrieve variables, and transform, if required
   TVector vs( GetNvar() );
   TVector vb( GetNvar() );

   // need to distinguish signal and background in case of variable transformation
   // signal first

   GetTransformationHandler().SetTransformationReferenceClass( fSignalClass );
   // temporary: JS  --> FIX
   //GetTransformationHandler().SetTransformationReferenceClass( 0 );
   const Event* ev = GetEvent();
   for (ivar=0; ivar<GetNvar(); ivar++) vs(ivar) = ev->GetValue(ivar);

   GetTransformationHandler().SetTransformationReferenceClass( fBackgroundClass );
   // temporary: JS  --> FIX
   //GetTransformationHandler().SetTransformationReferenceClass( 1 );
   ev = GetEvent();
   for (ivar=0; ivar<GetNvar(); ivar++) vb(ivar) = ev->GetValue(ivar);

   // compute the likelihood (signal)
   Double_t ps(1), pb(1), p(0);
   for (ivar=0; ivar<GetNvar(); ivar++) {

      // drop one variable (this is ONLY used for internal variable ranking !)
      if ((Int_t)ivar == fDropVariable) continue;

      Double_t x[2] = { vs(ivar), vb(ivar) };

      for (UInt_t itype=0; itype < 2; itype++) {

         // verify limits
         if      (x[itype] >= (*fPDFSig)[ivar]->GetXmax()) x[itype] = (*fPDFSig)[ivar]->GetXmax() - 1.0e-10;
         else if (x[itype] <  (*fPDFSig)[ivar]->GetXmin()) x[itype] = (*fPDFSig)[ivar]->GetXmin();

         // find corresponding histogram from cached indices
         PDF* pdf = (itype == 0) ? (*fPDFSig)[ivar] : (*fPDFBgd)[ivar];
         if (pdf == 0) Log() << kFATAL << "<GetMvaValue> Reference histograms don't exist" << Endl;
         TH1* hist = pdf->GetPDFHist();

         // interpolate linearly between adjacent bins
         // this is not useful for discrete variables
         Int_t bin = hist->FindBin(x[itype]);

         // **** POTENTIAL BUG: PREFORMANCE IS WORSE WHEN USING TRUE TYPE ***
         // ==> commented out at present
         if ((*fPDFSig)[ivar]->GetInterpolMethod() == TMVA::PDF::kSpline0 ||
             DataInfo().GetVariableInfo(ivar).GetVarType() == 'N') {
            p = TMath::Max( hist->GetBinContent(bin), fEpsilon );
         } else { // splined PDF
            Int_t nextbin = bin;
            if ((x[itype] > hist->GetBinCenter(bin) && bin != hist->GetNbinsX()) || bin == 1)
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
Double_t TMVA::MethodLikelihood::TransformLikelihoodOutput( Double_t ps, Double_t pb ) const
{
   // returns transformed or non-transformed output
   if (ps < fEpsilon) ps = fEpsilon;
   if (pb < fEpsilon) pb = fEpsilon;
   Double_t r = ps/(ps + pb);
   if (r >= 1.0) r = 1. - 1.e-15;

   if (fTransformLikelihoodOutput) {
      // inverse Fermi function

      // sanity check
      if      (r <= 0.0) r = fEpsilon;
      else if (r >= 1.0) r = 1. - 1.e-15;

      Double_t tau = 15.0;
      r = - TMath::Log(1.0/r - 1.0)/tau;
   }

   return r;
}

//______________________________________________________________________
void TMVA::MethodLikelihood::WriteOptionsToStream( std::ostream& o, const TString& prefix ) const
{
   // write options to stream
   Configurable::WriteOptionsToStream( o, prefix);

   // writing the options defined for the different pdfs
   if (fDefaultPDFLik != 0) {
      o << prefix << std::endl << prefix << "#Default Likelihood PDF Options:" << std::endl << prefix << std::endl;
      fDefaultPDFLik->WriteOptionsToStream( o, prefix );
   }
   for (UInt_t ivar = 0; ivar < fPDFSig->size(); ivar++) {
      if ((*fPDFSig)[ivar] != 0) {
         o << prefix << std::endl << prefix << Form("#Signal[%d] Likelihood PDF Options:",ivar) << std::endl << prefix << std::endl;
         (*fPDFSig)[ivar]->WriteOptionsToStream( o, prefix );
      }
      if ((*fPDFBgd)[ivar] != 0) {
         o << prefix << std::endl << prefix << "#Background[%d] Likelihood PDF Options:" << std::endl << prefix << std::endl;
         (*fPDFBgd)[ivar]->WriteOptionsToStream( o, prefix );
      }
   }
}

//_______________________________________________________________________
void TMVA::MethodLikelihood::AddWeightsXMLTo( void* parent ) const
{
   // write weights to XML
   void* wght = gTools().AddChild(parent, "Weights");
   gTools().AddAttr(wght, "NVariables", GetNvar());
   gTools().AddAttr(wght, "NClasses", 2);
   void* pdfwrap;
   for (UInt_t ivar=0; ivar<GetNvar(); ivar++) {
      if ( (*fPDFSig)[ivar]==0 || (*fPDFBgd)[ivar]==0 )
         Log() << kFATAL << "Reference histograms for variable " << ivar
               << " don't exist, can't write it to weight file" << Endl;
      pdfwrap = gTools().AddChild(wght, "PDFDescriptor");
      gTools().AddAttr(pdfwrap, "VarIndex", ivar);
      gTools().AddAttr(pdfwrap, "ClassIndex", 0);
      (*fPDFSig)[ivar]->AddXMLTo(pdfwrap);
      pdfwrap = gTools().AddChild(wght, "PDFDescriptor");
      gTools().AddAttr(pdfwrap, "VarIndex", ivar);
      gTools().AddAttr(pdfwrap, "ClassIndex", 1);
      (*fPDFBgd)[ivar]->AddXMLTo(pdfwrap);
   }
}

//_______________________________________________________________________
const TMVA::Ranking* TMVA::MethodLikelihood::CreateRanking()
{
   // computes ranking of input variables

   // create the ranking object
   if (fRanking) delete fRanking;
   fRanking = new Ranking( GetName(), "Delta Separation" );

   Double_t sepRef = -1, sep = -1;
   for (Int_t ivar=-1; ivar<(Int_t)GetNvar(); ivar++) {

      // this variable should not be used
      fDropVariable = ivar;

      TString nameS = Form( "rS_%i", ivar+1 );
      TString nameB = Form( "rB_%i", ivar+1 );
      TH1* rS = new TH1F( nameS, nameS, 80, 0, 1 );
      TH1* rB = new TH1F( nameB, nameB, 80, 0, 1 );

      // the event loop
      for (Int_t ievt=0; ievt<Data()->GetNTrainingEvents(); ievt++) {

         const Event* origEv = Data()->GetEvent(ievt);
         GetTransformationHandler().SetTransformationReferenceClass( origEv->GetClass() );
         const Event* ev = GetTransformationHandler().Transform(Data()->GetEvent(ievt));

         Double_t lk = this->GetMvaValue();
         Double_t w  = ev->GetWeight();
         if (DataInfo().IsSignal(ev)) rS->Fill( lk, w );
         else                rB->Fill( lk, w );
      }

      // compute separation
      sep = TMVA::gTools().GetSeparation( rS, rB );
      if (ivar == -1) sepRef = sep;
      sep = sepRef - sep;

      // don't need these histograms anymore
      delete rS;
      delete rB;

      if (ivar >= 0) fRanking->AddRank( Rank( DataInfo().GetVariableInfo(ivar).GetInternalName(), sep ) );
   }

   fDropVariable = -1;

   return fRanking;
}

//_______________________________________________________________________
void  TMVA::MethodLikelihood::WriteWeightsToStream( TFile& ) const
{
   // write reference PDFs to ROOT file
   TString pname = "PDF_";
   for (UInt_t ivar=0; ivar<GetNvar(); ivar++){
      (*fPDFSig)[ivar]->Write( pname + GetInputVar( ivar ) + "_S" );
      (*fPDFBgd)[ivar]->Write( pname + GetInputVar( ivar ) + "_B" );
   }
}
//_______________________________________________________________________
void  TMVA::MethodLikelihood::ReadWeightsFromXML(void* wghtnode)
{
   // read weights from XML
   TString pname = "PDF_";
   Bool_t addDirStatus = TH1::AddDirectoryStatus();
   TH1::AddDirectory(0); // this avoids the binding of the hists in TMVA::PDF to the current ROOT file
   UInt_t nvars=0;
   gTools().ReadAttr(wghtnode, "NVariables",nvars);
   void* descnode = gTools().GetChild(wghtnode);
   for (UInt_t ivar=0; ivar<nvars; ivar++){
      void* pdfnode = gTools().GetChild(descnode);
      Log() << kINFO << "Reading signal and background PDF for variable: " << GetInputVar( ivar ) << Endl;
      if ((*fPDFSig)[ivar] !=0) delete (*fPDFSig)[ivar];
      if ((*fPDFBgd)[ivar] !=0) delete (*fPDFBgd)[ivar];
      (*fPDFSig)[ivar] = new PDF( GetInputVar( ivar ) + " PDF Sig" );
      (*fPDFBgd)[ivar] = new PDF( GetInputVar( ivar ) + " PDF Bkg" );
      (*fPDFSig)[ivar]->SetReadingVersion( GetTrainingTMVAVersionCode() );
      (*fPDFBgd)[ivar]->SetReadingVersion( GetTrainingTMVAVersionCode() );
      (*(*fPDFSig)[ivar]).ReadXML(pdfnode);
      descnode = gTools().GetNextChild(descnode);
      pdfnode  = gTools().GetChild(descnode);
      (*(*fPDFBgd)[ivar]).ReadXML(pdfnode);
      descnode = gTools().GetNextChild(descnode);
   }
   TH1::AddDirectory(addDirStatus);
}
//_______________________________________________________________________
void  TMVA::MethodLikelihood::ReadWeightsFromStream( std::istream & istr )
{
   // read weight info from file
   // nothing to do for this method
   TString pname = "PDF_";
   Bool_t addDirStatus = TH1::AddDirectoryStatus();
   TH1::AddDirectory(0); // this avoids the binding of the hists in TMVA::PDF to the current ROOT file
   for (UInt_t ivar=0; ivar<GetNvar(); ivar++){
      Log() << kINFO << "Reading signal and background PDF for variable: " << GetInputVar( ivar ) << Endl;
      if ((*fPDFSig)[ivar] !=0) delete (*fPDFSig)[ivar];
      if ((*fPDFBgd)[ivar] !=0) delete (*fPDFBgd)[ivar];
      (*fPDFSig)[ivar] = new PDF(GetInputVar( ivar ) + " PDF Sig" );
      (*fPDFBgd)[ivar] = new PDF(GetInputVar( ivar ) + " PDF Bkg");
      (*fPDFSig)[ivar]->SetReadingVersion( GetTrainingTMVAVersionCode() );
      (*fPDFBgd)[ivar]->SetReadingVersion( GetTrainingTMVAVersionCode() );
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
   for (UInt_t ivar=0; ivar<GetNvar(); ivar++){
      (*fPDFSig)[ivar] = (TMVA::PDF*)rf.Get( Form( "PDF_%s_S", GetInputVar( ivar ).Data() ) );
      (*fPDFBgd)[ivar] = (TMVA::PDF*)rf.Get( Form( "PDF_%s_B", GetInputVar( ivar ).Data() ) );
   }
   TH1::AddDirectory(addDirStatus);
}

//_______________________________________________________________________
void  TMVA::MethodLikelihood::WriteMonitoringHistosToFile( void ) const
{
   // write histograms and PDFs to file for monitoring purposes

   Log() << kINFO << "Write monitoring histograms to file: " << BaseDir()->GetPath() << Endl;
   BaseDir()->cd();

   for (UInt_t ivar=0; ivar<GetNvar(); ivar++) {
      (*fHistSig)[ivar]->Write();
      (*fHistBgd)[ivar]->Write();
      if ((*fHistSig_smooth)[ivar] != 0) (*fHistSig_smooth)[ivar]->Write();
      if ((*fHistBgd_smooth)[ivar] != 0) (*fHistBgd_smooth)[ivar]->Write();
      (*fPDFSig)[ivar]->GetPDFHist()->Write();
      (*fPDFBgd)[ivar]->GetPDFHist()->Write();

      if ((*fPDFSig)[ivar]->GetNSmoothHist() != 0) (*fPDFSig)[ivar]->GetNSmoothHist()->Write();
      if ((*fPDFBgd)[ivar]->GetNSmoothHist() != 0) (*fPDFBgd)[ivar]->GetNSmoothHist()->Write();

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

      // ---------- create cloned low-binned histogram for comparison in macros (mainly necessary for KDE)
      TH1* h[2] = { (*fHistSig)[ivar], (*fHistBgd)[ivar] };
      for (UInt_t i=0; i<2; i++) {
         TH1* hclone = (TH1F*)h[i]->Clone( TString(h[i]->GetName()) + "_nice" );
         hclone->SetName ( TString(h[i]->GetName()) + "_nice" );
         hclone->SetTitle( TString(h[i]->GetTitle()) + "" );
         if (hclone->GetNbinsX() > 100) {
            Int_t resFactor = 5;
            hclone->Rebin( resFactor );
            hclone->Scale( 1.0/resFactor );
         }
         hclone->Write();
      }
      // ----------
   }
}

//_______________________________________________________________________
void TMVA::MethodLikelihood::MakeClassSpecificHeader( std::ostream& fout, const TString& ) const
{
   // write specific header of the classifier (mostly include files)
   fout << "#include <math.h>" << std::endl;
   fout << "#include <cstdlib>" << std::endl;
}

//_______________________________________________________________________
void TMVA::MethodLikelihood::MakeClassSpecific( std::ostream& fout, const TString& className ) const
{
   // write specific classifier response
   Int_t dp = fout.precision();
   fout << "   double       fEpsilon;" << std::endl;

   Int_t * nbin = new Int_t[GetNvar()];

   Int_t nbinMax=-1;
   for (UInt_t ivar=0; ivar<GetNvar(); ivar++) {
      nbin[ivar]=(*fPDFSig)[ivar]->GetPDFHist()->GetNbinsX();
      if (nbin[ivar] > nbinMax) nbinMax=nbin[ivar];
   }

   fout << "   static float fRefS[][" << nbinMax << "]; "
        << "// signal reference vector [nvars][max_nbins]" << std::endl;
   fout << "   static float fRefB[][" << nbinMax << "]; "
        << "// backgr reference vector [nvars][max_nbins]" << std::endl << std::endl;
   fout << "// if a variable has its PDF encoded as a spline0 --> treat it like an Integer valued one" <<std::endl;
   fout << "   bool    fHasDiscretPDF[" << GetNvar() <<"]; "<< std::endl;
   fout << "   int    fNbin[" << GetNvar() << "]; "
        << "// number of bins (discrete variables may have less bins)" << std::endl;
   fout << "   double    fHistMin[" << GetNvar() << "]; " << std::endl;
   fout << "   double    fHistMax[" << GetNvar() << "]; " << std::endl;

   fout << "   double TransformLikelihoodOutput( double, double ) const;" << std::endl;
   fout << "};" << std::endl;
   fout << "" << std::endl;
   fout << "inline void " << className << "::Initialize() " << std::endl;
   fout << "{" << std::endl;
   fout << "   fEpsilon = " << fEpsilon << ";" << std::endl;
   for (UInt_t ivar=0; ivar<GetNvar(); ivar++) {
      fout << "   fNbin[" << ivar << "] = " << (*fPDFSig)[ivar]->GetPDFHist()->GetNbinsX() << ";" << std::endl;
      fout << "   fHistMin[" << ivar << "] = " << (*fPDFSig)[ivar]->GetPDFHist()->GetXaxis()->GetXmin() << ";" << std::endl;
      fout << "   fHistMax[" << ivar << "] = " << (*fPDFSig)[ivar]->GetPDFHist()->GetXaxis()->GetXmax() << ";" << std::endl;
      // sanity check (for previous code lines)
      if ((((*fPDFSig)[ivar]->GetPDFHist()->GetNbinsX() != nbin[ivar] ||
            (*fPDFBgd)[ivar]->GetPDFHist()->GetNbinsX() != nbin[ivar])
           ) ||
          (*fPDFSig)[ivar]->GetPDFHist()->GetNbinsX() != (*fPDFBgd)[ivar]->GetPDFHist()->GetNbinsX()) {
         Log() << kFATAL << "<MakeClassSpecific> Mismatch in binning of variable "
               << "\"" << GetOriginalVarName(ivar) << "\" of type: \'" << DataInfo().GetVariableInfo(ivar).GetVarType()
               << "\' : "
               << "nxS = " << (*fPDFSig)[ivar]->GetPDFHist()->GetNbinsX() << ", "
               << "nxB = " << (*fPDFBgd)[ivar]->GetPDFHist()->GetNbinsX()
               << " while we expect " << nbin[ivar]
               << Endl;
      }
   }
   for (UInt_t ivar=0; ivar<GetNvar(); ivar++){
      if ((*fPDFSig)[ivar]->GetInterpolMethod() == TMVA::PDF::kSpline0)
         fout << "   fHasDiscretPDF[" << ivar <<"] = true;  " << std::endl;
      else
         fout << "   fHasDiscretPDF[" << ivar <<"] = false; " << std::endl;
   }

   fout << "}" << std::endl << std::endl;

   fout << "inline double " << className
        << "::GetMvaValue__( const std::vector<double>& inputValues ) const" << std::endl;
   fout << "{" << std::endl;
   fout << "   double ps(1), pb(1);" << std::endl;
   fout << "   std::vector<double> inputValuesSig = inputValues;" << std::endl;
   fout << "   std::vector<double> inputValuesBgd = inputValues;" << std::endl;
   if (GetTransformationHandler().GetTransformationList().GetSize() != 0) {
      fout << "   Transform(inputValuesSig,0);" << std::endl;
      fout << "   Transform(inputValuesBgd,1);" << std::endl;
   }
   fout << "   for (size_t ivar = 0; ivar < GetNvar(); ivar++) {" << std::endl;
   fout << std::endl;
   fout << "      // dummy at present... will be used for variable transforms" << std::endl;
   fout << "      double x[2] = { inputValuesSig[ivar], inputValuesBgd[ivar] };" << std::endl;
   fout << std::endl;
   fout << "      for (int itype=0; itype < 2; itype++) {" << std::endl;
   fout << std::endl;
   fout << "         // interpolate linearly between adjacent bins" << std::endl;
   fout << "         // this is not useful for discrete variables (or forced Spline0)" << std::endl;
   fout << "         int bin = int((x[itype] - fHistMin[ivar])/(fHistMax[ivar] - fHistMin[ivar])*fNbin[ivar]) + 0;" << std::endl;
   fout << std::endl;
   fout << "         // since the test data sample is in general different from the training sample" << std::endl;
   fout << "         // it can happen that the min/max of the training sample are trespassed --> correct this" << std::endl;
   fout << "         if      (bin < 0) {" << std::endl;
   fout << "            bin = 0;" << std::endl;
   fout << "            x[itype] = fHistMin[ivar];" << std::endl;
   fout << "         }" << std::endl;
   fout << "         else if (bin >= fNbin[ivar]) {" << std::endl;
   fout << "            bin = fNbin[ivar]-1;" << std::endl;
   fout << "            x[itype] = fHistMax[ivar];" << std::endl;
   fout << "         }" << std::endl;
   fout << std::endl;
   fout << "         // find corresponding histogram from cached indices" << std::endl;
   fout << "         float ref = (itype == 0) ? fRefS[ivar][bin] : fRefB[ivar][bin];" << std::endl;
   fout << std::endl;
   fout << "         // sanity check" << std::endl;
   fout << "         if (ref < 0) {" << std::endl;
   fout << "            std::cout << \"Fatal error in " << className
        << ": bin entry < 0 ==> abort\" << std::endl;" << std::endl;
   fout << "            std::exit(1);" << std::endl;
   fout << "         }" << std::endl;
   fout << std::endl;
   fout << "         double p = ref;" << std::endl;
   fout << std::endl;
   fout << "         if (GetType(ivar) != 'I' && !fHasDiscretPDF[ivar]) {" << std::endl;
   fout << "            float bincenter = (bin + 0.5)/fNbin[ivar]*(fHistMax[ivar] - fHistMin[ivar]) + fHistMin[ivar];" << std::endl;
   fout << "            int nextbin = bin;" << std::endl;
   fout << "            if ((x[itype] > bincenter && bin != fNbin[ivar]-1) || bin == 0) " << std::endl;
   fout << "               nextbin++;" << std::endl;
   fout << "            else" << std::endl;
   fout << "               nextbin--;  " << std::endl;
   fout << std::endl;
   fout << "            double refnext      = (itype == 0) ? fRefS[ivar][nextbin] : fRefB[ivar][nextbin];" << std::endl;
   fout << "            float nextbincenter = (nextbin + 0.5)/fNbin[ivar]*(fHistMax[ivar] - fHistMin[ivar]) + fHistMin[ivar];" << std::endl;
   fout << std::endl;
   fout << "            double dx = bincenter - nextbincenter;" << std::endl;
   fout << "            double dy = ref - refnext;" << std::endl;
   fout << "            p += (x[itype] - bincenter) * dy/dx;" << std::endl;
   fout << "         }" << std::endl;
   fout << std::endl;
   fout << "         if (p < fEpsilon) p = fEpsilon; // avoid zero response" << std::endl;
   fout << std::endl;
   fout << "         if (itype == 0) ps *= p;" << std::endl;
   fout << "         else            pb *= p;" << std::endl;
   fout << "      }            " << std::endl;
   fout << "   }     " << std::endl;
   fout << std::endl;
   fout << "   // the likelihood ratio (transform it ?)" << std::endl;
   fout << "   return TransformLikelihoodOutput( ps, pb );   " << std::endl;
   fout << "}" << std::endl << std::endl;

   fout << "inline double " << className << "::TransformLikelihoodOutput( double ps, double pb ) const" << std::endl;
   fout << "{" << std::endl;
   fout << "   // returns transformed or non-transformed output" << std::endl;
   fout << "   if (ps < fEpsilon) ps = fEpsilon;" << std::endl;
   fout << "   if (pb < fEpsilon) pb = fEpsilon;" << std::endl;
   fout << "   double r = ps/(ps + pb);" << std::endl;
   fout << "   if (r >= 1.0) r = 1. - 1.e-15;" << std::endl;
   fout << std::endl;
   fout << "   if (" << (fTransformLikelihoodOutput ? "true" : "false") << ") {" << std::endl;
   fout << "      // inverse Fermi function" << std::endl;
   fout << std::endl;
   fout << "      // sanity check" << std::endl;
   fout << "      if      (r <= 0.0) r = fEpsilon;" << std::endl;
   fout << "      else if (r >= 1.0) r = 1. - 1.e-15;" << std::endl;
   fout << std::endl;
   fout << "      double tau = 15.0;" << std::endl;
   fout << "      r = - log(1.0/r - 1.0)/tau;" << std::endl;
   fout << "   }" << std::endl;
   fout << std::endl;
   fout << "   return r;" << std::endl;
   fout << "}" << std::endl;
   fout << std::endl;

   fout << "// Clean up" << std::endl;
   fout << "inline void " << className << "::Clear() " << std::endl;
   fout << "{" << std::endl;
   fout << "   // nothing to clear" << std::endl;
   fout << "}" << std::endl << std::endl;

   fout << "// signal map" << std::endl;
   fout << "float " << className << "::fRefS[][" << nbinMax << "] = " << std::endl;
   fout << "{ " << std::endl;
   for (UInt_t ivar=0; ivar<GetNvar(); ivar++) {
      fout << "   { ";
      for (Int_t ibin=1; ibin<=nbinMax; ibin++) {
         if (ibin-1 < nbin[ivar])
            fout << (*fPDFSig)[ivar]->GetPDFHist()->GetBinContent(ibin);
         else
            fout << -1;

         if (ibin < nbinMax) fout << ", ";
      }
      fout << "   }, " << std::endl;
   }
   fout << "}; " << std::endl;
   fout << std::endl;

   fout << "// background map" << std::endl;
   fout << "float " << className << "::fRefB[][" << nbinMax << "] = " << std::endl;
   fout << "{ " << std::endl;
   for (UInt_t ivar=0; ivar<GetNvar(); ivar++) {
      fout << "   { ";
      fout << std::setprecision(8);
      for (Int_t ibin=1; ibin<=nbinMax; ibin++) {
         if (ibin-1 < nbin[ivar])
            fout << (*fPDFBgd)[ivar]->GetPDFHist()->GetBinContent(ibin);
         else
            fout << -1;

         if (ibin < nbinMax) fout << ", ";
      }
      fout << "   }, " << std::endl;
   }
   fout << "}; " << std::endl;
   fout << std::endl;
   fout << std::setprecision(dp);

   delete[] nbin;
}

//_______________________________________________________________________
void TMVA::MethodLikelihood::GetHelpMessage() const
{
   // get help message text
   //
   // typical length of text line:
   //         "|--------------------------------------------------------------|"
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Short description:" << gTools().Color("reset") << Endl;
   Log() << Endl;
   Log() << "The maximum-likelihood classifier models the data with probability " << Endl;
   Log() << "density functions (PDF) reproducing the signal and background" << Endl;
   Log() << "distributions of the input variables. Correlations among the " << Endl;
   Log() << "variables are ignored." << Endl;
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Performance optimisation:" << gTools().Color("reset") << Endl;
   Log() << Endl;
   Log() << "Required for good performance are decorrelated input variables" << Endl;
   Log() << "(PCA transformation via the option \"VarTransform=Decorrelate\"" << Endl;
   Log() << "may be tried). Irreducible non-linear correlations may be reduced" << Endl;
   Log() << "by precombining strongly correlated input variables, or by simply" << Endl;
   Log() << "removing one of the variables." << Endl;
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Performance tuning via configuration options:" << gTools().Color("reset") << Endl;
   Log() << Endl;
   Log() << "High fidelity PDF estimates are mandatory, i.e., sufficient training " << Endl;
   Log() << "statistics is required to populate the tails of the distributions" << Endl;
   Log() << "It would be a surprise if the default Spline or KDE kernel parameters" << Endl;
   Log() << "provide a satisfying fit to the data. The user is advised to properly" << Endl;
   Log() << "tune the events per bin and smooth options in the spline cases" << Endl;
   Log() << "individually per variable. If the KDE kernel is used, the adaptive" << Endl;
   Log() << "Gaussian kernel may lead to artefacts, so please always also try" << Endl;
   Log() << "the non-adaptive one." << Endl;
   Log() << "" << Endl;
   Log() << "All tuning parameters must be adjusted individually for each input" << Endl;
   Log() << "variable!" << Endl;
}

