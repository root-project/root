// @(#)root/tmva $Id: MethodLikelihood.cxx,v 1.47 2006/10/04 22:29:27 andreas.hoecker Exp $ 
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

#include "TMatrixD.h"
#include "TVector.h"
#include "TObjString.h"
#include "TFile.h"
#include "TKey.h"

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
   //                    25 - average num of events per PDF bin to trigger warning
   //                    D  - use square-root-matrix to decorrelate variable space 
   // 
   InitLik();

   DeclareOptions();

   ParseOptions();

   ProcessOptions();

   // note that one variable is type
   if (HasTrainingTree()) {
    
      // count number of signal and background events
      Int_t Nsig = Data().GetNEvtSigTrain();
      Int_t Nbgd = Data().GetNEvtBkgdTrain();
    
      if (Verbose())
         cout << "--- " << GetName() << " <verbose>: num of events for training (signal, background): "
              << " (" << Nsig << ", " << Nbgd << ")" << endl;
    
      // Likelihood wants same number of events in each species
      if (Nsig != Nbgd) {
         cout << "--- " << GetName() << ":\t--------------------------------------------------"
              << endl;
         cout << "--- " << GetName() << ":\tWarning: different number of signal and background\n"
              << "--- " << GetName() << " \tevents: Likelihood training will not be optimal :-("
              << endl;
         cout << "--- " << GetName() << ":\t--------------------------------------------------"
              << endl;
      }      
   }
}

//_______________________________________________________________________
TMVA::MethodLikelihood::MethodLikelihood( DataSet & theData, 
                                          TString theWeightFile,  
                                          TDirectory* theTargetDir )
   : TMVA::MethodBase( theData, theWeightFile, theTargetDir ) 
{  
   // construct likelihood references from file
   DeclareOptions();

   InitLik();
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
  
   SetMethodName( "Likelihood" );
   SetMethodType( TMVA::Types::Likelihood );
   SetTestvarName();

   fEpsilon        = 1e-5;
   fBgdPDFHist     = new TList();
   fSigPDFHist     = new TList();

   fHistSig        = new vector<TH1*>      ( GetNvar() ); 
   fHistBgd        = new vector<TH1*>      ( GetNvar() ); 
   fHistSig_smooth = new vector<TH1*>      ( GetNvar() ); 
   fHistBgd_smooth = new vector<TH1*>      ( GetNvar() );
   fPDFSig         = new vector<TMVA::PDF*>( GetNvar() );
   fPDFBgd         = new vector<TMVA::PDF*>( GetNvar() );

   fIndexSig       = new vector<UInt_t>    ( GetNvar() );
   fIndexBgd       = new vector<UInt_t>    ( GetNvar() );
}

//_______________________________________________________________________
void TMVA::MethodLikelihood::DeclareOptions() 
{
   DeclareOptionRef(fSpline=2,"Spline","spline used to interpolate reference histograms");
   AddPreDefVal(0); // take histogram
   AddPreDefVal(1); // linear interpolation between bins
   AddPreDefVal(2); // quadratic interpolation
   AddPreDefVal(3); // cubic interpolation
   AddPreDefVal(5); // fifth order polynome interpolation

   // initialize 
   DeclareOptionRef( fNsmooth = 0, "NSmooth",
                     "how often the input histos are smoothed");
   DeclareOptionRef( fAverageEvtPerBin = 25, "NAvEvtPerBin",
                     "average num of events per PDF bin to trigger warning");   
   DeclareOptionRef( fTransformLikelihoodOutput = kFALSE, "TransformOutput", 
                     "transform (often strongly peaked) likelihood output through sigmoid inversion" );
}

//_______________________________________________________________________
void TMVA::MethodLikelihood::ProcessOptions() 
{
   MethodBase::ProcessOptions();

   if      (fSpline == 0) fSmoothMethod = TMVA::PDF::kSpline0;
   else if (fSpline == 1) fSmoothMethod = TMVA::PDF::kSpline1;
   else if (fSpline == 2) fSmoothMethod = TMVA::PDF::kSpline2;      
   else if (fSpline == 3) fSmoothMethod = TMVA::PDF::kSpline3;      
   else if (fSpline == 5) fSmoothMethod = TMVA::PDF::kSpline5;
   else {
      cout  << "--- " << GetName() << ": WARNING unknown Spline type! Choose Spline2" << endl;
      fSmoothMethod = TMVA::PDF::kSpline2;
   }
   
   // decorrelate option will be last option, if it is specified
	if      (GetPreprocessingMethod() == Types::kDecorrelated)
		cout << "--- " << GetName() << ": use decorrelated variable set" << endl;
	else if (GetPreprocessingMethod() == Types::kPCA)
		cout << "--- " << GetName() << ": use principal component preprocessing" << endl;
}

//_______________________________________________________________________
TMVA::MethodLikelihood::~MethodLikelihood( void )
{
   // destructor  
   if (NULL != fHistSig) delete fHistSig;
   if (NULL != fHistBgd) delete fHistBgd;
   if (NULL != fHistSig_smooth) delete fHistSig_smooth;
   if (NULL != fHistBgd_smooth) delete fHistBgd_smooth;
   if (NULL != fPDFSig)  delete  fPDFSig;
   if (NULL != fPDFBgd)  delete  fPDFBgd;
   if (NULL != fIndexSig) delete fIndexSig;
   if (NULL != fIndexBgd) delete fIndexBgd;

   if (NULL != fFin) { fFin->Close(); delete fFin; }

   delete fBgdPDFHist;
   delete fSigPDFHist;
}

//_______________________________________________________________________
void TMVA::MethodLikelihood::Train( void )
{
   // create reference distributions (PDFs) from signal and background events:
   // fill histograms and smooth them; if decorrelation is required, compute 
   // corresponding square-root matrices

   // default sanity checks
   if (!CheckSanity()) { 
      cout << "--- " << GetName() << ": Error: sanity check failed" << endl;
      exit(1);
   }

   // create reference histograms
   UInt_t nbins = (Int_t)(TMath::Min(Data().GetNEvtSigTrain(),Data().GetNEvtBkgdTrain())/fAverageEvtPerBin);
   TString histTitle, histName;

   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {     

      // for signal events
      histTitle = (*fInputVars)[ivar] + " signal training";
      histName  = (*fInputVars)[ivar] + "_sig";
      (*fHistSig)[ivar] = new TH1F( histName, histTitle, nbins, 
                                    GetXmin(ivar,GetPreprocessingMethod()),
                                    GetXmax(ivar,GetPreprocessingMethod()));

      // for background events
      histTitle = (*fInputVars)[ivar] + " background training";
      histName  = (*fInputVars)[ivar] + "_bgd";
      (*fHistBgd)[ivar] = new TH1F( histName, histTitle, nbins, 
                                    GetXmin(ivar, GetPreprocessingMethod()),
                                    GetXmax(ivar, GetPreprocessingMethod()));
   }

   // ----- fill the reference histograms

   // event loop
   for (Int_t ievt=0; ievt<Data().GetNEvtTrain(); ievt++) {

      // use the true-event-type's preprocessing 
      ReadTrainingEvent( ievt, Types::kTrueType ); 

      // the event weight
      Float_t weight = GetEventWeight();

      // fill variable vector
      for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
         Float_t value  = GetEventVal(ivar);
         if (Data().Event().IsSignal()) (*fHistSig)[ivar]->Fill( value, weight );
         else                           (*fHistBgd)[ivar]->Fill( value, weight );
      }
   }

   // apply smoothing, and create PDFs
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) { 

      for (UInt_t itype=0; itype < 2; itype++) {

         TH1F* htmp = 0;         
         if (itype == 0) { // signal
            htmp = (TH1F*)(*fHistSig)[ivar]->Clone();

            histTitle = (*fInputVars)[ivar] + " signal training  smoothed ";
            histName  = (*fInputVars)[ivar] + "_sig_smooth";
         }
         else { // background
            htmp = (TH1F*)(*fHistBgd)[ivar]->Clone();

            histTitle = (*fInputVars)[ivar] + " background training  smoothed ";
            histName  = (*fInputVars)[ivar] + "_bgd_smooth";
         }

         // title continued
         histTitle += fNsmooth;
         histTitle += " times";
         
         htmp->SetName(histName);
         htmp->SetTitle(histTitle);

         // --- smooth histogram and create PDF

         // if the variable is discrete, use histogram (=kSpline0) as reference 
         // also: don't apply any smoothing !
         TMVA::PDF* ptmp;
         if (Data().VarTypeOriginal(ivar) == 'I') {
            ptmp =  new TMVA::PDF( htmp, PDF::kSpline0 );
         }
         else {
            if (htmp->GetNbinsX() > 2 && fNsmooth >0) htmp->Smooth(fNsmooth);
            else {
               if (htmp->GetNbinsX() <=2)
                  cout << "--- " << GetName() << ": WARNING histogram "<< htmp->GetName()
                       << " has not enough (" << htmp->GetNbinsX()
                       << ") bins for for smoothing "<<endl;
            }
            ptmp = new TMVA::PDF( htmp, fSmoothMethod );
         }
         
         if (itype == 0) {
            (*fHistSig_smooth)[ivar] = htmp;
            (*fPDFSig)[ivar]         = ptmp;
            (*fIndexSig)[ivar]       = ivar; // trivial index caching
         }
         else {
            (*fHistBgd_smooth)[ivar] = htmp;
            (*fPDFBgd)[ivar]         = ptmp;
            (*fIndexBgd)[ivar]       = ivar; // trivial index caching
         }
      }
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

   // need to distinguish signal and background in case of preprocessing
   // signal first
   Event eventBackup( Data().Event() );
   Data().ApplyTransformation( GetPreprocessingMethod(), kTRUE );
   for (ivar=0; ivar<GetNvar(); ivar++) vs(ivar) = GetEventVal(ivar);

   // now background ... need to reinitialise original - non-preprocessed event
   Data().Event().CopyVarValues( eventBackup );
   Data().ApplyTransformation( GetPreprocessingMethod(), kFALSE );
   for (ivar=0; ivar<GetNvar(); ivar++) vb(ivar) = GetEventVal(ivar);
   
   // compute the likelihood (signal)
   Double_t ps = 1;
   Double_t pb = 1;
   for (ivar=0; ivar<GetNvar(); ivar++) {
    
      Double_t x[2] = { vs(ivar), vb(ivar) };
    
      for (UInt_t itype=0; itype < 2; itype++) {

         // find corresponding histogram from cached indices                 
         TH1F*   hist = 0;
         if (itype == 0) hist = (TH1F*)fSigPDFHist->At((*fIndexSig)[ivar]);
         else            hist = (TH1F*)fBgdPDFHist->At((*fIndexBgd)[ivar]);
    
         // interpolate linearly between adjacent bins
         // this is not useful for discrete variables
         Int_t bin = hist->FindBin(x[itype]);
         Double_t p;
         if (fSmoothMethod == TMVA::PDF::kSpline0) {         
            p = TMath::Max( hist->GetBinContent(bin), fEpsilon );
         }
         else { // splined PDF

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
  
   // the likelihood
   Double_t myMVA = 0;
   if (fTransformLikelihoodOutput) {
      // inverse Fermi function
      Double_t r   = ps/(pb+ps);

      // sanity check
      if      (r <= 0.0) r = fEpsilon;
      else if (r >= 1.0) r = 1.0 - fEpsilon;

      Double_t tau = 15.0;
      myMVA = - log(1.0/r - 1.0)/tau;
   }
   else myMVA = ps/(pb+ps);
  
   return myMVA;
}

//_______________________________________________________________________
void  TMVA::MethodLikelihood::WriteWeightsToStream( ostream & o ) const
{  
   // write reference PDFs to file

   TString fname = GetWeightFileName() + ".root";
   cout << "--- " << GetName() << ": creating weight file: " << fname << endl;
   TFile *fout = new TFile( fname, "RECREATE" );

   o << "# weights stored in root i/o file: " << fname << endl;

   // build TList of input variables, and TVectors for min/max
   // NOTE: the latter values are mandatory for the normalisation 
   // in the reader application !!!
   TList    lvar;
   TVectorD vmin( GetNvar() ), vmax( GetNvar() );
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
      lvar.Add( new TNamed( (*fInputVars)[ivar], TString() ) );
      vmin[ivar] = this->GetXmin( ivar );
      vmax[ivar] = this->GetXmax( ivar );
   }
   // write to file
   lvar.Write();
   vmin.Write( "vmin" );
   vmax.Write( "vmax" );
   lvar.Delete();

   // save configuration options
   // (best would be to use a TMap here, but found implementation really complicated)
   TVectorD likelOptions( 4 );
   likelOptions(0) = (Double_t)fSmoothMethod;
   likelOptions(1) = (Double_t)fNsmooth;
   likelOptions(2) = (Double_t)fAverageEvtPerBin;
   likelOptions(3) = (Double_t) (GetPreprocessingMethod() == Types::kNone ? 0. : 1.);
   likelOptions.Write( "LikelihoodOptions" );
  
   // now write the histograms
   for(Int_t ivar=0; ivar<GetNvar(); ivar++){ 
      (*fPDFSig)[ivar]->GetPDFHist()->Write();
      (*fPDFBgd)[ivar]->GetPDFHist()->Write();
   }                  

   // the reference histograms to file
   WriteHistosToFile();

   fout->Close();
   delete fout;
}
  
//_______________________________________________________________________
void  TMVA::MethodLikelihood::ReadWeightsFromStream( istream& istr )
{
   // read reference PDFs from file   if (istr.eof());

   if (istr.eof());   // dummy

   TString fname = GetWeightFileName();
   if (!fname.EndsWith( ".root" )) fname += ".root";
   
   cout << "--- " << GetName() << ": reading weight file: " << fname << endl;
   fFin = new TFile(fname);
   
   // build TList of input variables, and TVectors for min/max
   // NOTE: the latter values are mandatory for the normalisation 
   // in the reader application 
   TList lvar;
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {    
      // read variable names
      TNamed t;
      t.Read( (*fInputVars)[ivar] );

      // sanity check
      if (t.GetName() != (*fInputVars)[ivar]) {
         cout << "--- " << GetName() << ": Error while reading weight file; "
              << "unknown variable: " << t.GetName() << " at position: " << ivar << ". "
              << "Expected variable: " << (*fInputVars)[ivar] << " ==> abort" << endl;
         exit(1);
      }
   }

   // read vectors
   TVectorD vmin( GetNvar() ), vmax( GetNvar() );
   // unfortunatly the more elegant vmin/max.Read( "vmin/max" ) crash in ROOT <= V4.04.02
   TVectorD *tmp = (TVectorD*)fFin->Get( "vmin" );
   vmin = *tmp;
   tmp  = (TVectorD*)fFin->Get( "vmax" );
   vmax = *tmp;

   // initialize min/max
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {    
      Data().SetXmin( ivar, vmin[ivar] );
      Data().SetXmax( ivar, vmax[ivar] );
   }

   // now read the histograms
   fSigPDFHist = new TList();
   fBgdPDFHist = new TList();
  
   TIter next(fFin->GetListOfKeys());
   TKey *key;
   while ((key = (TKey*)next())) {
      TClass *cl = gROOT->GetClass(key->GetClassName());
      if (!cl->InheritsFrom("TH1F")) continue;
      TH1F *h = (TH1F*)key->ReadObj();
      TString hname= h->GetName();
      if      (hname.Contains("_sig_")) fSigPDFHist->Add(h);
      else if (hname.Contains("_bgd_")) fBgdPDFHist->Add(h);
      
      // find corresponding variable index and cache it (to spead up likelihood evaluation)
      for (Int_t ivar=0; ivar<GetNvar(); ivar++) {    
         if (hname.Contains( (*fInputVars)[ivar] )) {
            if      (hname.Contains("_sig_")) (*fIndexSig)[ivar] = fSigPDFHist->GetEntries()-1;
            else if (hname.Contains("_bgd_")) (*fIndexBgd)[ivar] = fBgdPDFHist->GetEntries()-1;
         }
      }
   }  
}

//_______________________________________________________________________
void  TMVA::MethodLikelihood::WriteHistosToFile( void ) const
{
   // write histograms and PDFs to file for monitoring purposes

   cout << "--- " << GetName() << ": write"
        <<" monitoring histograms to file: " << BaseDir()->GetPath() << endl;
  
   BaseDir()->cd();
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
