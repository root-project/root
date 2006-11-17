// @(#)root/tmva $Id: MethodPDERS.cxx,v 1.43 2006/11/17 00:21:35 stelzer Exp $
// Author: Andreas Hoecker, Yair Mahalalel, Joerg Stelzer, Helge Voss, Kai Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodPDERS                                                           *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation                                                            *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Yair Mahalalel  <Yair.Mahalalel@cern.ch> - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        *
 *      U. of Victoria, Canada,                                                   *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
// Begin_Html
/*
  This is a generalization of the above Likelihood methods to <i>N</i><sub>var</sub>
  dimensions, where <i>N</i><sub>var</sub> is the number of input variables
  used in the MVA. If the multi-dimensional probability density functions 
  (PDFs) for signal and background were known, this method contains the entire 
  physical information, and is therefore optimal. Usually, kernel estimation 
  methods are used to approximate the PDFs using the events from the 
  training sample. <br><p></p>
   
  A very simple probability density estimator (PDE) has been suggested
  in <a href="http://arxiv.org/abs/hep-ex/0211019">hep-ex/0211019</a>. The
  PDE for a given test event is obtained from counting the (normalized) 
  number of signal and background (training) events that occur in the 
  "vicinity" of the test event. The volume that describes "vicinity" is 
  user-defined. A <a href="http://arxiv.org/abs/hep-ex/0211019">search 
  method based on binary-trees</a> is used to effectively reduce the 
  selection time for the range search. Three different volume definitions
  are optional: <br><p></p>
  <ul>
  <li><u>MinMax:</u>
  the volume is defined in each dimension with respect 
  to the full variable range found in the training sample. </li>
  <li><u>RMS:</u>
  the volume is defined in each dimensions with respect 
  to the RMS estimated from the training sample. </li>
  <li><u>Adaptive:</u>
  a volume element is defined in each dimensions with 
  respect to the RMS estimated from the training sample. The overall 
  scale of the volume element is then determined for each event so 
  that the total number of events confined in the volume be within 
  a user-defined range.</li>
  </ul><p></p>
  The adaptive range search is used by default.
  // End_Html
  */
//_______________________________________________________________________

#include "TMVA/MethodPDERS.h"
#include "TMVA/Tools.h"
#include "TMVA/RootFinder.h"
#include "TFile.h"
#include "TObjString.h"
#include "TMath.h"
#include <stdexcept>

#define TMVA_MethodPDERS__countByHand__Debug__
#undef  TMVA_MethodPDERS__countByHand__Debug__

namespace TMVA {
   const Bool_t MethodPDERS_UseFindRoot = kTRUE;
}

using std::vector;

ClassImp(TMVA::MethodPDERS)
   ;

//_______________________________________________________________________
TMVA::MethodPDERS::MethodPDERS( TString jobName, TString methodTitle, DataSet& theData, 
                                TString theOption, TDirectory* theTargetDir )
   : TMVA::MethodBase( jobName, methodTitle, theData, theOption, theTargetDir )
{
   // standard constructor for the PDERS method
   // format and syntax of option string: "VolumeRangeMode:options"
   // where:
   //    VolumeRangeMode - all methods defined in private enum "VolumeRangeMode" 
   //    options         - deltaFrac in case of VolumeRangeMode=MinMax/RMS
   //                    - nEventsMin/Max, maxVIterations, scale for VolumeRangeMode=Adaptive
   //

   InitPDERS();

   DeclareOptions();

   ParseOptions();

   ProcessOptions();
}

//_______________________________________________________________________
TMVA::MethodPDERS::MethodPDERS( DataSet& theData,
                                TString theWeightFile,
                                TDirectory* theTargetDir )
   : TMVA::MethodBase( theData, theWeightFile, theTargetDir )
{
   // construct MethodPDERS through from file
   InitPDERS();

   DeclareOptions();
}

void TMVA::MethodPDERS::InitPDERS( void )
{
   // default initialisation routine called by all constructors
   SetMethodName( "PDERS" );
   SetMethodType( TMVA::Types::kPDERS );
   SetTestvarName();

   fBinaryTreeS   = fBinaryTreeB = NULL;
   fReferenceTree = NULL;

   UpdateThis();

   // default options
   fDeltaFrac       = 3.0;
   fVRangeMode      = kAdaptive;
   fKernelEstimator = kBox;

   // special options for Adaptive mode
   fNEventsMin      = 100;
   fNEventsMax      = 200;
   fMaxVIterations  = 50;
   fInitialScale    = 0.99;
   fGaussSigma      = 0.2;

   fInitializedVolumeEle = kFALSE;
}

//_______________________________________________________________________
TMVA::MethodPDERS::~MethodPDERS( void )
{
   // destructor
   if (NULL != fBinaryTreeS) delete fBinaryTreeS;
   if (NULL != fBinaryTreeB) delete fBinaryTreeB;

   if (NULL != fFin) { fFin->Close(); delete fFin; }
}


//_______________________________________________________________________
void TMVA::MethodPDERS::DeclareOptions() 
{
   // define the options (their key words) that can be set in the option string 
   // know options:
   // VolumeRangeMode   <string>  Method to determine volume range
   //    available values are:        MinMax <default>
   //                                 Unscaled
   //                                 RMS
   //                                 Adaptive
   //
   // KernelEstimator   <string>  Kernel estimation function
   //    available values are:        Box <default>
   //                                 Sphere
   //                                 Teepee
   //                                 Gauss
   //                                 Sinc3
   //                                 Sinc5
   //                                 Sinc7
   //                                 Sinc9
   //                                 Sinc11
   //                                 Lanczos2
   //                                 Lanczos3
   //                                 Lanczos5
   //                                 Lanczos8
   //
   // DeltaFrac         <float>   Ratio of #EventsMin/#EventsMax for MinMax and RMS volume range
   // NEventsMin        <int>     Minimum number of events for adaptive volume range             
   // NEventsMax        <int>     Maximum number of events for adaptive volume range
   // MaxVIterations    <int>     Maximum number of iterations for adaptive volume range
   // InitialScale      <float>   Initial scale for adaptive volume range           
   // GaussSigma        <float>   Width with respect to the volume size of Gaussian kernel estimator

   DeclareOptionRef(fVolumeRange="MinMax", "VolumeRangeMode", "Method to determine volume range");
   AddPreDefVal(TString("Unscaled"));
   AddPreDefVal(TString("MinMax"));
   AddPreDefVal(TString("RMS"));
   AddPreDefVal(TString("Adaptive"));

   DeclareOptionRef(fKernelString="Box", "KernelEstimator", "Kernel estimation function");
   AddPreDefVal(TString("Box"));
   AddPreDefVal(TString("Sphere"));
   AddPreDefVal(TString("Teepee"));
   AddPreDefVal(TString("Gauss"));
   AddPreDefVal(TString("Sinc3"));
   AddPreDefVal(TString("Sinc5"));
   AddPreDefVal(TString("Sinc7"));
   AddPreDefVal(TString("Sinc9"));
   AddPreDefVal(TString("Sinc11"));
   AddPreDefVal(TString("Lanczos2"));
   AddPreDefVal(TString("Lanczos3"));
   AddPreDefVal(TString("Lanczos5"));
   AddPreDefVal(TString("Lanczos8"));

   DeclareOptionRef(fDeltaFrac     , "DeltaFrac",      "nEventsMin/Max for minmax and rms volume range");
   DeclareOptionRef(fNEventsMin    , "NEventsMin",     "nEventsMin for adaptive volume range");
   DeclareOptionRef(fNEventsMax    , "NEventsMax",     "nEventsMax for adaptive volume range");
   DeclareOptionRef(fMaxVIterations, "MaxVIterations", "MaxVIterations for adaptive volume range");
   DeclareOptionRef(fInitialScale  , "InitialScale",   "InitialScale for adaptive volume range");
   DeclareOptionRef(fGaussSigma    , "GaussSigma",     "Width (wrt volume size) of Gaussian kernel estimator");
}

//_______________________________________________________________________
void TMVA::MethodPDERS::ProcessOptions() 
{
   // process the options specified by the user
   
   MethodBase::ProcessOptions();

   fVRangeMode = TMVA::MethodPDERS::kUnsupported;

   if         (fVolumeRange == "MinMax"    ) fVRangeMode = TMVA::MethodPDERS::kMinMax;
   else if    (fVolumeRange == "RMS"       ) fVRangeMode = TMVA::MethodPDERS::kRMS;
   else if    (fVolumeRange == "Adaptive"  ) fVRangeMode = TMVA::MethodPDERS::kAdaptive;
   else if    (fVolumeRange == "Unscaled"  ) fVRangeMode = TMVA::MethodPDERS::kUnscaled;
   else {
      fLogger << kFATAL << "VolumeRangeMode parameter '" << fVolumeRange << "' unknown" << Endl;
   }

   if        (fKernelString == "Box"      ) fKernelEstimator = TMVA::MethodPDERS::kBox;
   else if   (fKernelString == "Sphere"   ) fKernelEstimator = TMVA::MethodPDERS::kSphere;
   else if   (fKernelString == "Teepee"   ) fKernelEstimator = TMVA::MethodPDERS::kTeepee;
   else if   (fKernelString == "Gauss"    ) fKernelEstimator = TMVA::MethodPDERS::kGauss;
   else if   (fKernelString == "Sinc3"    ) fKernelEstimator = TMVA::MethodPDERS::kSinc3;
   else if   (fKernelString == "Sinc5"    ) fKernelEstimator = TMVA::MethodPDERS::kSinc5;
   else if   (fKernelString == "Sinc7"    ) fKernelEstimator = TMVA::MethodPDERS::kSinc7;
   else if   (fKernelString == "Sinc9"    ) fKernelEstimator = TMVA::MethodPDERS::kSinc9;
   else if   (fKernelString == "Sinc11"   ) fKernelEstimator = TMVA::MethodPDERS::kSinc11;
   else if   (fKernelString == "Lanczos2" ) fKernelEstimator = TMVA::MethodPDERS::kLanczos2;
   else if   (fKernelString == "Lanczos3" ) fKernelEstimator = TMVA::MethodPDERS::kLanczos3;
   else if   (fKernelString == "Lanczos5" ) fKernelEstimator = TMVA::MethodPDERS::kLanczos5;
   else if   (fKernelString == "Lanczos8" ) fKernelEstimator = TMVA::MethodPDERS::kLanczos8;
   else {
      fLogger << kFATAL << "KernelEstimator parameter '" << fKernelString << "' unknown" << Endl;
   }

   // TODO: Add parameter validation

   fLogger << kVERBOSE << "interpreted option string: vRangeMethod: '"
           << (const char*)((fVRangeMode == kMinMax) ? "MinMax" :
                            (fVRangeMode == kUnscaled) ? "Unscaled" :
                            (fVRangeMode == kRMS   ) ? "RMS" : "Adaptive") << "'" << Endl;
   if (fVRangeMode == kMinMax || fVRangeMode == kRMS)
      fLogger << kVERBOSE << "deltaFrac: " << fDeltaFrac << Endl;
   else
      fLogger << kVERBOSE << "nEventsMin/Max, maxVIterations, initialScale: "
              << fNEventsMin << "  " << fNEventsMax
              << "  " << fMaxVIterations << "  " << fInitialScale << Endl;
   fLogger << kVERBOSE << "KernelEstimator = " << fKernelString << Endl;
}

//_______________________________________________________________________
void TMVA::MethodPDERS::Train( void )
{
   // this is a dummy training: the preparation work to do is the construction
   // of the binary tree as a pointer chain. It is easier to directly save the
   // trainingTree in the weight file, and to rebuild the binary tree in the
   // test phase from scratch

   // default sanity checks
   if (!CheckSanity()) fLogger << kFATAL << "<Train> sanity check failed" << Endl;

   Data().GetTrainingTree()->ResetBranchAddresses(); 
   Data().ResetCurrentTree();
   TTree* refTree = (TTree*)Data().GetTrainingTree()->CloneTree(0);
   refTree->SetName("referenceTree");
   Data().Event().SetBranchAddresses( refTree );

   // fill the new ntuple (this is needed, since we may be in decorrelation or 
   // similar preprocessing mode)
   for (Int_t i=0; i<Data().GetNEvtTrain(); i++) {
      ReadTrainingEvent(i);      
      refTree->Fill();
   }   

   SetReferenceTree( refTree );
}

//_______________________________________________________________________
Double_t TMVA::MethodPDERS::GetMvaValue()
{
   // init the size of a volume element using a defined fraction of the
   // volume containing the entire events
   if (fInitializedVolumeEle == kFALSE) {
      fInitializedVolumeEle = kTRUE;
      SetVolumeElement();

      // create binary trees (global member variables) for signal and background
      Int_t nS = 0, nB = 0;      
      fBinaryTreeS = new TMVA::BinarySearchTree();
      nS = fBinaryTreeS->Fill( Data(), GetReferenceTree(), 1, Types::kNone );
      fBinaryTreeB = new TMVA::BinarySearchTree();
      nB = fBinaryTreeB->Fill( Data(), GetReferenceTree(), 0, Types::kNone );

      // sanity check
      if (NULL == fBinaryTreeS || NULL == fBinaryTreeB) {
         fLogger << kFATAL << "<Train> Create(BinaryTree) returned zero "
                 << "binaryTree pointer(s): " << fBinaryTreeS << "  " << fBinaryTreeB << Endl;
      }

      // these are the signal and background scales for the weights
      fScaleS = 1.0/Float_t(nS);
      fScaleB = 1.0/Float_t(nB);
      fLogger << kVERBOSE << "signal and background scales: " << fScaleS << " " << fScaleB << Endl;
   }

   return this->RScalc( Data().Event() );
}

//_______________________________________________________________________
void TMVA::MethodPDERS::SetVolumeElement( void )
{
   // defines volume dimensions

   // init relative scales
   fDelta = (GetNvar() > 0) ? new vector<Float_t>( GetNvar() ) : 0;
   fShift = (GetNvar() > 0) ? new vector<Float_t>( GetNvar() ) : 0;
   if (fDelta != 0) {
      for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
         switch (fVRangeMode) {

         case kRMS:
         case kAdaptive:
            Double_t meanS, meanB, rmsS, rmsB, xmin, xmax;
            TMVA::Tools::ComputeStat( GetReferenceTree(), (*fInputVars)[ivar],
                                      meanS, meanB, rmsS, rmsB, xmin, xmax );

            (*fDelta)[ivar] = (rmsS + rmsB)*0.5*fDeltaFrac;
            fLogger << kVERBOSE << "delta of var[" << (*fInputVars)[ivar]
                    << "\t]: " << (rmsS + rmsB)*0.5
                    << "\t  |  comp with d|norm|: " << (GetXmax( ivar ) - GetXmin( ivar ))
                    << Endl;
            break;

         case kMinMax:
            (*fDelta)[ivar] = (GetXmax( ivar ) - GetXmin( ivar ))*fDeltaFrac;
            break;

         case kUnscaled:
            (*fDelta)[ivar] = fDeltaFrac;
            break;

         default:
            fLogger << kFATAL << "<SetVolumeElement> unknown range-set mode: "
                    << fVRangeMode << Endl;
         }

         (*fShift)[ivar] = 0.5; // volume is centered around test value
      }
   }
   else {
      fLogger << kFATAL << "GetNvar() <= 0: " << GetNvar() << Endl;
   }
}

TMVA::MethodPDERS* TMVA::MethodPDERS::fgThisPDERS = NULL;

//_______________________________________________________________________
Double_t TMVA::MethodPDERS::IGetVolumeContentForRoot( Double_t scale )
{
   // Interface to RootFinder
   return ThisPDERS()->GetVolumeContentForRoot( scale );
}

//_______________________________________________________________________
Double_t TMVA::MethodPDERS::GetVolumeContentForRoot( Double_t scale )
{
   // count number of events in rescaled volume

   Volume v( *fHelpVolume );
   v.ScaleInterval( scale );

   Double_t cS = GetBinaryTreeSig()->SearchVolume( &v );
   Double_t cB = GetBinaryTreeBkg()->SearchVolume( &v );
   v.Delete();
   return cS + cB;
}

//_______________________________________________________________________
Float_t TMVA::MethodPDERS::RScalc( const TMVA::Event& e )
{
   // computes event weight by counting number of signal and background 
   // events (of reference sample) that are found within given volume
   // defined by the event
   vector<Double_t> *lb = new vector<Double_t>( GetNvar() );
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) (*lb)[ivar] = e.GetVal(ivar);

   vector<Double_t> *ub = new vector<Double_t>( *lb );
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
      (*lb)[ivar] -= (*fDelta)[ivar]*(1.0 - (*fShift)[ivar]);
      (*ub)[ivar] += (*fDelta)[ivar]*(*fShift)[ivar];
   }

   TMVA::Volume* volume = new TMVA::Volume( lb, ub );

   Float_t countS = 0;
   Float_t countB = 0;

   // -------------------------------------------------------------------------
   //
   // ==== test of volume search =====
   //
   // #define TMVA::MethodPDERS__countByHand__Debug__

#ifdef  TMVA_MethodPDERS__countByHand__Debug__

   // starting values
   countS = fBinaryTreeS->SearchVolume( volume );
   countB = fBinaryTreeB->SearchVolume( volume );

   Int_t iS = 0, iB = 0;
   for (Int_t ievt_=0; ievt_<Data().GetNEvtTrain(); ievt_++) {
      Data().ReadTrainEvent(ievt_);
      Bool_t inV;
      for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
         Float_t x = Data().Event().GetVal(ivar);
         inV = (x > (*volume->Lower)[ivar] && x <= (*volume->Upper)[ivar]);
         if (!inV) break;
      }
      if (inV) {
         if (Data().Event().IsSignal())
            iS++;
         else
            iB++;
      }
   }
   fLogger << kVERBOSE << "debug: my test: S/B: " << iS << "  " << iB << Endl;
   fLogger << kVERBOSE << "debug: binTree: S/B: " << countS << "  " << countB << Endl << Endl;
#endif
   // -------------------------------------------------------------------------

   if (fVRangeMode == kRMS || fVRangeMode == kUnscaled) { // Constant volume
      vector<Double_t> *lb = new vector<Double_t>( GetNvar() );
      for (Int_t ivar=0; ivar<GetNvar(); ivar++) (*lb)[ivar] = e.GetVal(ivar);
      vector<Double_t> *ub = new vector<Double_t>( *lb );
      for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
         (*lb)[ivar] -= (*fDelta)[ivar]*(1.0 - (*fShift)[ivar]);
         (*ub)[ivar] += (*fDelta)[ivar]*(*fShift)[ivar];
      }
      TMVA::Volume* volume = new TMVA::Volume( lb, ub );
      // starting values
      TMVA::Volume v( *volume );
      
      std::vector<TMVA::Event*> eventsS;
      std::vector<TMVA::Event*> eventsB;
      fBinaryTreeS->SearchVolume( &v, &eventsS );
      fBinaryTreeB->SearchVolume( &v, &eventsB );
      countS = KernelEstimate( e, eventsS, v );
      countB = KernelEstimate( e, eventsB, v );
      
      delete lb;
      delete ub;
   } 
   else if (fVRangeMode == kAdaptive) {      // adaptive volume

      // -----------------------------------------------------------------------

      // TODO: optimize, perhaps multi stage with broadening limits, 
      // or a different root finding method entirely,
      if (TMVA::MethodPDERS_UseFindRoot) { 
         // that won't need to search through large volume, where the bottle neck probably is

         fHelpVolume = new TMVA::Volume( *volume );

         UpdateThis(); // necessary update of static pointer
         TMVA::RootFinder rootFinder( &IGetVolumeContentForRoot, 0.01, 50, 50, 10 );
         Double_t scale = rootFinder.Root( (fNEventsMin + fNEventsMax)/2.0 );

         TMVA::Volume v( *volume );
         v.ScaleInterval( scale );

         std::vector<TMVA::Event*> eventsS;
         std::vector<TMVA::Event*> eventsB;
         fBinaryTreeS->SearchVolume( &v, &eventsS );
         fBinaryTreeB->SearchVolume( &v, &eventsB );
         countS = KernelEstimate( e, eventsS, v );
         countB = KernelEstimate( e, eventsB, v );

         v.Delete();

         fHelpVolume->Delete();
         delete fHelpVolume; fHelpVolume = NULL;

      }
      // -----------------------------------------------------------------------
      else {

         // starting values
         countS = fBinaryTreeS->SearchVolume( volume );
         countB = fBinaryTreeB->SearchVolume( volume );

         Float_t nEventsO = countS + countB;
         Int_t i_=0;
         while (nEventsO < fNEventsMin) { // this isn't a sain start... try again
            volume->ScaleInterval( 1.15 );
            countS = fBinaryTreeS->SearchVolume( volume );
            countB = fBinaryTreeB->SearchVolume( volume );
            nEventsO = countS + countB;
            i_++;
         }
         if (i_ > 50) fLogger << kWARNING << "warning in event: " << e
                              << ": adaptive volume pre-adjustment reached "
                              << ">50 iterations in while loop (" << i_ << ")" << Endl;

         Float_t nEventsN = nEventsO;
         Float_t nEventsE = 0.5*(fNEventsMin + fNEventsMax);
         Float_t scaleO   = 1.0;
         Float_t scaleN   = fInitialScale;
         Float_t scale    = scaleN;

         Float_t cS = countS;
         Float_t cB = countB;

         for (Int_t ic=1; ic<fMaxVIterations; ic++) {
            if (nEventsN < fNEventsMin || nEventsN > fNEventsMax) {

               // search for events in rescaled volume
               TMVA::Volume* v = new TMVA::Volume( *volume );
               v->ScaleInterval( scale );
               cS       = fBinaryTreeS->SearchVolume( v );
               cB       = fBinaryTreeB->SearchVolume( v );
               nEventsN = cS + cB;

               // determine next iteration (linear approximation)
               if (nEventsN > 1 && nEventsN - nEventsO != 0)
                  if (scaleN - scaleO != 0)
                     scale += (scaleN - scaleO)/(nEventsN - nEventsO)*(nEventsE - nEventsN);
                  else
                     scale += (-0.01); // should not actually occur...
               else
                  scale += 0.5; // use much larger volume

               // save old scale
               scaleN   = scale;

               // take if better (don't accept it if too small number of events)
               if (TMath::Abs(cS + cB - nEventsE) < TMath::Abs(countS + countB - nEventsE) &&
                   (cS + cB >= fNEventsMin || countS + countB < cS + cB)) {
                  countS = cS; countB = cB;
               }

               v->Delete();
               delete v;
            }
            else break;
         }

         // last sanity check
         nEventsN = countS + countB;
         // include "1" to cover float precision
         if (nEventsN < fNEventsMin-1 || nEventsN > fNEventsMax+1)
            fLogger << kWARNING << "warning in event " << e
                    << ": adaptive volume adjustment reached "
                    << "max. #iterations (" << fMaxVIterations << ")"
                    << "[ nEvents: " << nEventsN << "  " << fNEventsMin << "  " << fNEventsMax << "]"
                    << Endl;
      }

   } // end of adaptive method
   // -----------------------------------------------------------------------

   volume->Delete();
   delete volume;

   if (countS < 1e-20 && countB < 1e-20) return 0.5;
   if (countB < 1e-20) return 1.0;
   if (countS < 1e-20) return 0.0;

   Float_t r = countB*fScaleB/(countS*fScaleS);
   return 1.0/(r + 1.0);   // TODO: propagate errors from here
}

//_______________________________________________________________________
Double_t TMVA::MethodPDERS::KernelEstimate( const TMVA::Event & event,
                                            vector<TMVA::Event*>& events, TMVA::Volume& v )
{
   // Final estimate
   Double_t pdfSum = 0;
  
   // normalization factors so we can work with radius 1 hyperspheres
   Double_t *dim_normalization = new Double_t[GetNvar()];
   for (Int_t ivar=0; ivar<GetNvar(); ivar++)
      dim_normalization [ivar] = 2 / ((*v.fUpper)[ivar] - (*v.fLower)[ivar]);
   
   // Iteration over sample points
   for (vector<TMVA::Event*>::iterator iev = events.begin(); iev != events.end(); iev++) {
     
      // First switch to the one dimensional distance
      Double_t normalized_distance = GetNormalizedDistance (event, *(*iev), dim_normalization);
      
      // always working within the hyperelipsoid, except for when we don't
      // note that rejection ratio goes to 1 as nvar goes to infinity
      if (normalized_distance > 1 && fKernelEstimator != kBox) continue;      
      
      pdfSum += ApplyKernelFunction (normalized_distance) * (*iev)->GetWeight();
   }
   return KernelNormalization( pdfSum < 0. ? 0. : pdfSum );
}

//_______________________________________________________________________
Double_t TMVA::MethodPDERS::ApplyKernelFunction (Double_t normalized_distance) 
{
   // from the normalized euclidean distance calculate the distance
   // for a certain kernel
   switch (fKernelEstimator) {
   case kBox:
   case kSphere:
      return 1;
      break;
   case kTeepee:
      return (1 - normalized_distance);
      break;
   case kGauss:
      return TMath::Gaus( normalized_distance, 0, fGaussSigma, kFALSE);
      break;
   case kSinc3:
   case kSinc5:
   case kSinc7:
   case kSinc9:
   case kSinc11: {
      Double_t side_crossings = 2 + ((int) fKernelEstimator) - ((int) kSinc3);
      return NormSinc (side_crossings * normalized_distance);
   }
      break;
   case kLanczos2:
      return LanczosFilter (2, normalized_distance);
      break;
   case kLanczos3:
      return LanczosFilter (3, normalized_distance);
      break;
   case kLanczos5:
      return LanczosFilter (5, normalized_distance);
      break;
   case kLanczos8:
      return LanczosFilter (8, normalized_distance);
      break;
   default:
      fLogger << kFATAL << "Kernel estimation function unsupported. Enumerator is " << fKernelEstimator << Endl;
      break;
   }

   return 0;
}
      
//_______________________________________________________________________
Double_t TMVA::MethodPDERS::KernelNormalization (Double_t pdf) 
{
   // Calculating the normalization factor only once (might need a reset at some point. Can the method be restarted with different params?)

   static Double_t ret = 1.; // Caching jammed to disable function. It's not really useful afterall, badly implemented and untested :-)

   if (ret != 0.)
      return ret*pdf; 

   // We first normalize by the volume of the hypersphere.
   switch (fKernelEstimator) {
   case kBox:
   case kSphere:
      ret = 1.;
      break;
   case kTeepee:
      ret =   (GetNvar() * (GetNvar() + 1) * TMath::Gamma (((Double_t) GetNvar()) / 2.)) /
         ( TMath::Power (2., (Double_t) GetNvar() + 1) * TMath::Power (TMath::Pi(), ((Double_t) GetNvar()) / 2.));
      break;
   case kGauss:
      // We use full range integral here. Reasonable because of the fast function decay.
      ret = 1. / TMath::Power ( 2 * TMath::Pi() * fGaussSigma * fGaussSigma, ((Double_t) GetNvar()) / 2.);
      break;
   case kSinc3:
   case kSinc5:
   case kSinc7:
   case kSinc9:
   case kSinc11:
   case kLanczos2:
   case kLanczos3:
   case kLanczos5:
   case kLanczos8:
      // We use the full range integral here. Reasonable because the central lobe domintes it.
      ret = 1 / TMath::Power ( 2., (Double_t) GetNvar() );
      break;
   default:
      fLogger << kFATAL << "Kernel estimation function unsupported. Enumerator is " << fKernelEstimator << Endl;
   }
   // Normalizing by the full volume
   ret *= ( TMath::Power (2., GetNvar()) * TMath::Gamma (1 + (((Double_t) GetNvar()) / 2.)) ) /
      TMath::Power (TMath::Pi(), ((Double_t) GetNvar()) / 2.);
   return ret*pdf;
}

//_______________________________________________________________________
Double_t TMVA::MethodPDERS::GetNormalizedDistance (   const TMVA::Event &base_event,
                                                      const TMVA::Event &sample_event,
                                                      Double_t *dim_normalization) {
   // We use Euclidian metric here. Might not be best or most efficient.
   Double_t ret=0;
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
      Double_t dist = dim_normalization[ivar] * (sample_event.GetVal(ivar) - base_event.GetVal(ivar));
      ret += dist*dist;
   }
   return TMath::Sqrt (ret);
}

//_______________________________________________________________________
Double_t TMVA::MethodPDERS::NormSinc (Double_t x)
{
   // NormSinc
   if (x < 10e-10 && x > -10e-10) {
      return 1; // Poor man's l'Hopital
   }

   Double_t pix = TMath::Pi() * x;
   Double_t sinc = TMath::Sin(pix) / pix;
   Double_t ret;

   if (GetNvar() % 2)
      ret = TMath::Power (sinc, GetNvar());
   else
      ret = TMath::Abs (sinc) * TMath::Power (sinc, GetNvar() - 1);

   return ret;
}

//_______________________________________________________________________
Double_t TMVA::MethodPDERS::LanczosFilter (Int_t level, Double_t x)
{
   // Lanczos Filter
   if (x < 10e-10 && x > -10e-10) {
      return 1; // Poor man's l'Hopital
   }

   Double_t pix = TMath::Pi() * x;
   Double_t pixtimesn = pix * ((Double_t) level);
   Double_t lanczos = (TMath::Sin(pix) / pix) * (TMath::Sin(pixtimesn) / pixtimesn);
   Double_t ret;

   if (GetNvar() % 2)
      ret = TMath::Power (lanczos, GetNvar());
   else
      ret = TMath::Abs (lanczos) * TMath::Power (lanczos, GetNvar() - 1);

   return ret;
}

//_______________________________________________________________________
Float_t TMVA::MethodPDERS::GetError( Float_t countS, Float_t countB,
                                     Float_t sumW2S, Float_t sumW2B ) const
{
   // statistical error estimate for RS estimator

   Float_t c = fScaleB/fScaleS;
   Float_t d = countS + c*countB; d *= d;

   if (d < 1e-10) return 1; // Error is zero because of B = S = 0

   Float_t f = c*c/d/d;
   Float_t err = f*countB*countB*sumW2S + f*countS*countS*sumW2B;

   if (err < 1e-10) return 1; // Error is zero because of B or S = 0

   return sqrt(err);
}

//_______________________________________________________________________
void TMVA::MethodPDERS::WriteWeightsToStream( ostream& o ) const
{
   // write training sample (TTree) to file
   TString fname = GetWeightFileName() + ".root";
   fLogger << kINFO << "creating weight file: " << fname << Endl;
   TFile *fout = new TFile( fname, "RECREATE" );

   // info in txt stream
   o << "# weights stored in root i/o file: " << fname << endl;

   // write trainingTree
   GetReferenceTree()->Write();

   fout->Close();

   delete fout;
}

//_______________________________________________________________________
void TMVA::MethodPDERS::ReadWeightsFromStream( istream& istr )
{
   // read training sample from file
   if (istr.eof()); // dummy

   TString fname = GetWeightFileName();
   if (!fname.EndsWith( ".root" )) fname += ".root";

   fLogger << kINFO << "reading weight file: " << fname << Endl;
   fFin = new TFile( fname );

   // read the trainingTree
   TTree* tree = (TTree*)fFin->Get( "referenceTree" );
   SetReferenceTree( tree );

   if (NULL == GetReferenceTree()) {
      fLogger << kFATAL << "error while reading 'referenceTree': zero pointer " << Endl;
   }   
}


