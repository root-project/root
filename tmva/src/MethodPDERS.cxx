// @(#)root/tmva $Id$
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
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
/* Begin_Html

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

End_Html */
//_______________________________________________________________________

#include <assert.h>
#include <algorithm>

#include "TFile.h"
#include "TObjString.h"
#include "TMath.h"

#include "TMVA/MethodPDERS.h"
#include "TMVA/Tools.h"
#include "TMVA/RootFinder.h"

#define TMVA_MethodPDERS__countByHand__Debug__
#undef  TMVA_MethodPDERS__countByHand__Debug__

namespace TMVA {
   const Bool_t MethodPDERS_UseFindRoot = kTRUE;
}

TMVA::MethodPDERS* TMVA::MethodPDERS::fgThisPDERS = NULL;

ClassImp(TMVA::MethodPDERS)

//_______________________________________________________________________
TMVA::MethodPDERS::MethodPDERS( const TString& jobName, const TString& methodTitle, DataSet& theData, 
                                const TString& theOption, TDirectory* theTargetDir )
   : MethodBase( jobName, methodTitle, theData, theOption, theTargetDir )
   , fDelta(0)
   , fShift(0)
{
   // standard constructor for the PDERS method
   // format and syntax of option string: "VolumeRangeMode:options"
   // where:
   //    VolumeRangeMode - all methods defined in private enum "VolumeRangeMode" 
   //    options         - deltaFrac in case of VolumeRangeMode=MinMax/RMS
   //                    - nEventsMin/Max, maxVIterations, scale for VolumeRangeMode=Adaptive
   InitPDERS();

   // interpretation of configuration option string
   SetConfigName( TString("Method") + GetMethodName() );
   DeclareOptions();
   ParseOptions();
   ProcessOptions();
}

//_______________________________________________________________________
TMVA::MethodPDERS::MethodPDERS( DataSet& theData,
                                const TString& theWeightFile,
                                TDirectory* theTargetDir )
   : MethodBase( theData, theWeightFile, theTargetDir )
   , fDelta(0)
   , fShift(0)
{
   // construct MethodPDERS through from file
   InitPDERS();

   DeclareOptions();
}

//_______________________________________________________________________
void TMVA::MethodPDERS::InitPDERS( void )
{
   // default initialisation routine called by all constructors
   SetMethodName( "PDERS" );
   SetMethodType( Types::kPDERS );
   SetTestvarName();

   fBinaryTreeS = fBinaryTreeB = NULL;

   UpdateThis();

   // default options
   fDeltaFrac       = 3.0;
   fVRangeMode      = kAdaptive;
   fKernelEstimator = kBox;

   // special options for Adaptive mode
   fNEventsMin      = 100;
   fNEventsMax      = 200;
   fMaxVIterations  = 150;
   fInitialScale    = 0.99;
   fGaussSigma      = 0.1;
   fNormTree        = kFALSE;
   
   fkNNTests      = 1000;
    
   fkNNMin      = Int_t(fNEventsMin);
   fkNNMax      = Int_t(fNEventsMax);

   fInitializedVolumeEle = kFALSE;
   fAverageRMS.clear();

   // the minimum requirement to declare an event signal-like
   SetSignalReferenceCut( 0.0 );
}

//_______________________________________________________________________
TMVA::MethodPDERS::~MethodPDERS( void )
{
   // destructor
   if(fDelta) delete fDelta;
   if(fShift) delete fShift;

   if (NULL != fBinaryTreeS) delete fBinaryTreeS;
   if (NULL != fBinaryTreeB) delete fBinaryTreeB;
}

//_______________________________________________________________________
void TMVA::MethodPDERS::DeclareOptions() 
{
   // define the options (their key words) that can be set in the option string 
   // know options:
   // VolumeRangeMode   <string>  Method to determine volume range
   //    available values are:        MinMax 
   //                                 Unscaled
   //                                 RMS
   //                                  kNN
   //                                 Adaptive <default>
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
   //                                 Trim
   //
   // DeltaFrac         <float>   Ratio of #EventsMin/#EventsMax for MinMax and RMS volume range
   // NEventsMin        <int>     Minimum number of events for adaptive volume range             
   // NEventsMax        <int>     Maximum number of events for adaptive volume range
   // MaxVIterations    <int>     Maximum number of iterations for adaptive volume range
   // InitialScale      <float>   Initial scale for adaptive volume range           
   // GaussSigma        <float>   Width with respect to the volume size of Gaussian kernel estimator

   DeclareOptionRef(fVolumeRange="Adaptive", "VolumeRangeMode", "Method to determine volume size");
   AddPreDefVal(TString("Unscaled"));
   AddPreDefVal(TString("MinMax"));
   AddPreDefVal(TString("RMS"));
   AddPreDefVal(TString("Adaptive"));
   AddPreDefVal(TString("kNN"));

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
   AddPreDefVal(TString("Trim"));

   DeclareOptionRef(fDeltaFrac     , "DeltaFrac",      "nEventsMin/Max for minmax and rms volume range");
   DeclareOptionRef(fNEventsMin    , "NEventsMin",     "nEventsMin for adaptive volume range");
   DeclareOptionRef(fNEventsMax    , "NEventsMax",     "nEventsMax for adaptive volume range");
   DeclareOptionRef(fMaxVIterations, "MaxVIterations", "MaxVIterations for adaptive volume range");
   DeclareOptionRef(fInitialScale  , "InitialScale",   "InitialScale for adaptive volume range");
   DeclareOptionRef(fGaussSigma    , "GaussSigma",     "Width (wrt volume size) of Gaussian kernel estimator");
   DeclareOptionRef(fNormTree      , "NormTree",       "Normalize binary search tree");
}

//_______________________________________________________________________
void TMVA::MethodPDERS::ProcessOptions() 
{
   // process the options specified by the user
   
   MethodBase::ProcessOptions();

   fGaussSigmaNorm = fGaussSigma; // * TMath::Sqrt( Double_t(GetNvar()) );

   fVRangeMode = MethodPDERS::kUnsupported;

   if      (fVolumeRange == "MinMax"    ) fVRangeMode = kMinMax;
   else if (fVolumeRange == "RMS"       ) fVRangeMode = kRMS;
   else if (fVolumeRange == "Adaptive"  ) fVRangeMode = kAdaptive;
   else if (fVolumeRange == "Unscaled"  ) fVRangeMode = kUnscaled;
   else if (fVolumeRange == "kNN"   ) fVRangeMode = kkNN;
   else {
      fLogger << kFATAL << "VolumeRangeMode parameter '" << fVolumeRange << "' unknown" << Endl;
   }

   if      (fKernelString == "Box"      ) fKernelEstimator = kBox;
   else if (fKernelString == "Sphere"   ) fKernelEstimator = kSphere;
   else if (fKernelString == "Teepee"   ) fKernelEstimator = kTeepee;
   else if (fKernelString == "Gauss"    ) fKernelEstimator = kGauss;
   else if (fKernelString == "Sinc3"    ) fKernelEstimator = kSinc3;
   else if (fKernelString == "Sinc5"    ) fKernelEstimator = kSinc5;
   else if (fKernelString == "Sinc7"    ) fKernelEstimator = kSinc7;
   else if (fKernelString == "Sinc9"    ) fKernelEstimator = kSinc9;
   else if (fKernelString == "Sinc11"   ) fKernelEstimator = kSinc11;
   else if (fKernelString == "Lanczos2" ) fKernelEstimator = kLanczos2;
   else if (fKernelString == "Lanczos3" ) fKernelEstimator = kLanczos3;
   else if (fKernelString == "Lanczos5" ) fKernelEstimator = kLanczos5;
   else if (fKernelString == "Lanczos8" ) fKernelEstimator = kLanczos8;
   else if (fKernelString == "Trim"     ) fKernelEstimator = kTrim;
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
   if (IsNormalised()) fLogger << kFATAL << "\"Normalise\" option cannot be used with PDERS; " 
                               << "please remove the option from the configuration string, or "
                               << "use \"!Normalise\""
                               << Endl;

   CreateBinarySearchTrees( Data().GetTrainingTree() );
   
   CalcAverages();
   SetVolumeElement();

   fInitializedVolumeEle = kTRUE;
}

//_______________________________________________________________________
Double_t TMVA::MethodPDERS::GetMvaValue()
{
   // init the size of a volume element using a defined fraction of the
   // volume containing the entire events
   if (fInitializedVolumeEle == kFALSE) {
      fInitializedVolumeEle = kTRUE;

      // binary trees must exist
      assert( fBinaryTreeS && fBinaryTreeB );

      CalcAverages();

      SetVolumeElement();
   }

   return this->RScalc( GetEvent() );
}

//_______________________________________________________________________
void TMVA::MethodPDERS::CalcAverages()
{
   // compute also average RMS values required for adaptive Gaussian
   if (fVRangeMode == kAdaptive || fVRangeMode == kRMS || fVRangeMode == kkNN  ) {
      fAverageRMS.clear();
      fBinaryTreeS->CalcStatistics();
      fBinaryTreeB->CalcStatistics();

      for (Int_t ivar=0; ivar<GetNvar(); ivar++) {   
         Float_t rmsS = fBinaryTreeS->RMS(Types::kSignal, ivar);
         Float_t rmsB = fBinaryTreeB->RMS(Types::kBackground, ivar);
         fAverageRMS.push_back( (rmsS + rmsB)*0.5 );
      }
   }
}   

//_______________________________________________________________________
void TMVA::MethodPDERS::CreateBinarySearchTrees( TTree* tree ) 
{
   // create binary search trees for signal and background
   assert( tree != 0 );

   if (NULL != fBinaryTreeS) delete fBinaryTreeS;
   if (NULL != fBinaryTreeB) delete fBinaryTreeB;
   fBinaryTreeS = new BinarySearchTree();
   fBinaryTreeB = new BinarySearchTree();
   if (fNormTree) {
      fBinaryTreeS->SetNormalize( kTRUE );
      fBinaryTreeB->SetNormalize( kTRUE );
   }

   fBinaryTreeS->Fill( *this, tree, 1 );
   fBinaryTreeB->Fill( *this, tree, 0 );

   if (fNormTree) {
      fBinaryTreeS->NormalizeTree();
      fBinaryTreeB->NormalizeTree();
   }

   // these are the signal and background scales for the weights
   fScaleS = 1.0/fBinaryTreeS->GetSumOfWeights();
   fScaleB = 1.0/fBinaryTreeB->GetSumOfWeights();

   fLogger << kVERBOSE << "signal and background scales: " << fScaleS << " " << fScaleB << Endl;
}

//_______________________________________________________________________
void TMVA::MethodPDERS::SetVolumeElement( void )
{
   // defines volume dimensions

   if(GetNvar()<=0) {
      fLogger << kFATAL << "GetNvar() <= 0: " << GetNvar() << Endl;
   }


   // init relative scales
   fkNNMin      = Int_t(fNEventsMin);
   fkNNMax      = Int_t(fNEventsMax);   

   if(fDelta) delete fDelta;
   if(fShift) delete fShift;
   fDelta = new std::vector<Float_t>( GetNvar() );
   fShift = new std::vector<Float_t>( GetNvar() );

   switch (fVRangeMode) {
         
   case kRMS:
   case kkNN:
   case kAdaptive:
      // sanity check
      if ((Int_t)fAverageRMS.size() != GetNvar())
         fLogger << kFATAL << "<SetVolumeElement> RMS not computed: " << fAverageRMS.size() << Endl;
      for (Int_t ivar=0; ivar<GetNvar(); ivar++)
         {
            (*fDelta)[ivar] = fAverageRMS[ivar]*fDeltaFrac;
            fLogger << kVERBOSE << "delta of var[" << (*fInputVars)[ivar]
                    << "\t]: " << fAverageRMS[ivar]
                    << "\t  |  comp with |max - min|: " << (GetXmax( ivar ) - GetXmin( ivar ))
                    << Endl;
         }
      break;
         
   case kMinMax:
      for (Int_t ivar=0; ivar<GetNvar(); ivar++)
         (*fDelta)[ivar] = (GetXmax( ivar ) - GetXmin( ivar ))*fDeltaFrac;
      break;
         
   case kUnscaled:
      for (Int_t ivar=0; ivar<GetNvar(); ivar++)
         (*fDelta)[ivar] = fDeltaFrac;
      break;
   default:
      fLogger << kFATAL << "<SetVolumeElement> unknown range-set mode: "
              << fVRangeMode << Endl;
   }
   for (Int_t ivar=0; ivar<GetNvar(); ivar++)
      (*fShift)[ivar] = 0.5; // volume is centered around test value
   

}

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
Float_t TMVA::MethodPDERS::RScalc( const Event& e )
{
   // computes event weight by counting number of signal and background 
   // events (of reference sample) that are found within given volume
   // defined by the event
   std::vector<Double_t> *lb = new std::vector<Double_t>( GetNvar() );
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) (*lb)[ivar] = e.GetVal(ivar);

   std::vector<Double_t> *ub = new std::vector<Double_t>( *lb );
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
      (*lb)[ivar] -= (*fDelta)[ivar]*(1.0 - (*fShift)[ivar]);
      (*ub)[ivar] += (*fDelta)[ivar]*(*fShift)[ivar];
   }

   Volume *volume = new Volume( lb, ub );   

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
         Float_t x = GetEventVal(ivar);
         inV = (x > (*volume->Lower)[ivar] && x <= (*volume->Upper)[ivar]);
         if (!inV) break;
      }
      if (inV) {
         if (IsSignalEvent()) iS++;
         else                 iB++;
      }
   }
   fLogger << kVERBOSE << "debug: my test: S/B: " << iS << "  " << iB << Endl;
   fLogger << kVERBOSE << "debug: binTree: S/B: " << countS << "  " << countB << Endl << Endl;

#endif

   // -------------------------------------------------------------------------

   if (fVRangeMode == kRMS || fVRangeMode == kMinMax || fVRangeMode == kUnscaled ) { // Constant volume

      std::vector<const BinarySearchTreeNode*> eventsS;
      std::vector<const BinarySearchTreeNode*> eventsB;
      fBinaryTreeS->SearchVolume( volume, &eventsS );
      fBinaryTreeB->SearchVolume( volume, &eventsB );
      countS = KernelEstimate( e, eventsS, *volume );
      countB = KernelEstimate( e, eventsB, *volume );

   }
   else if (fVRangeMode == kAdaptive) {      // adaptive volume

      // -----------------------------------------------------------------------

      // TODO: optimize, perhaps multi stage with broadening limits, 
      // or a different root finding method entirely,
      if (MethodPDERS_UseFindRoot) { 

         // that won't need to search through large volume, where the bottle neck probably is

         fHelpVolume = volume;

         UpdateThis(); // necessary update of static pointer
         RootFinder rootFinder( &IGetVolumeContentForRoot, 0.01, 50, 200, 10 );
         Double_t scale = rootFinder.Root( (fNEventsMin + fNEventsMax)/2.0 );

         Volume v( *volume );
         v.ScaleInterval( scale );

         std::vector<const BinarySearchTreeNode*> eventsS;
         std::vector<const BinarySearchTreeNode*> eventsB;
         fBinaryTreeS->SearchVolume( &v, &eventsS );
         fBinaryTreeB->SearchVolume( &v, &eventsB );
         countS = KernelEstimate( e, eventsS, v );
         countB = KernelEstimate( e, eventsB, v );

         v.Delete();

         fHelpVolume = NULL;

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
               Volume* v = new Volume( *volume );
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
   else if (fVRangeMode == kkNN)
      {
         std::vector< const BinarySearchTreeNode* > eventsS;    //vector for signals
         std::vector< const BinarySearchTreeNode* > eventsB;    //vector for backgrounds
         Volume v(*volume);

         // check number of signals in begining volume
         Int_t kNNcountS = fBinaryTreeS->SearchVolumeWithMaxLimit( &v, &eventsS, fkNNMax+1 );   
         // check number of backgrounds in begining volume
         Int_t kNNcountB = fBinaryTreeB->SearchVolumeWithMaxLimit( &v, &eventsB, fkNNMax+1 );   
         //if this number is too large return fkNNMax+1
         Int_t t_times = 0;  // number of iterations
      
         while ( !(kNNcountS+kNNcountB >= fkNNMin && kNNcountS+kNNcountB <= fkNNMax) ) {
            if (kNNcountS+kNNcountB < fkNNMin) {         //if we have too less points
               Float_t scale = 2;      //better scale needed
               volume->ScaleInterval( scale );
            }
            else if (kNNcountS+kNNcountB > fkNNMax) {    //uf we have too many points
               Float_t scale = 0.1;      //better scale needed
               volume->ScaleInterval( scale );
            }
            eventsS.clear();
            eventsB.clear();
          
            v = *volume ;
         
            kNNcountS = fBinaryTreeS->SearchVolumeWithMaxLimit( &v, &eventsS, fkNNMax+1 );  //new search
            kNNcountB = fBinaryTreeB->SearchVolumeWithMaxLimit( &v, &eventsB, fkNNMax+1 );  //new search
         
            t_times++;
         
            if (t_times == fMaxVIterations) {
               fLogger << kWARNING << "warining in event" << e
                       << ": kNN volume adjustment reached "
                       << "max. #iterations (" << fMaxVIterations << ")"
                       << "[ kNN: " << fkNNMin << " " << fkNNMax << Endl;
               break;
            }
         }
      
         //vector to normalize distance in each dimension
         Double_t *dim_normalization = new Double_t[GetNvar()];
         for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
            dim_normalization [ivar] = 1.0 / ((*v.fUpper)[ivar] - (*v.fLower)[ivar]);
         }      

         std::vector<const BinarySearchTreeNode*> tempVectorS;    // temporary vector for signals
         std::vector<const BinarySearchTreeNode*> tempVectorB;    // temporary vector for backgrounds
      
         if (kNNcountS+kNNcountB >= fkNNMin) {
            std::vector<Double_t> *distances = new std::vector<Double_t>( kNNcountS+kNNcountB );
         
            //counting the distance for earch signal to event
            for (Int_t j=0;j< Int_t(eventsS.size()) ;j++)
               (*distances)[j] = GetNormalizedDistance ( e, *eventsS[j], dim_normalization );
         
            //counting the distance for each background to event
            for (Int_t j=0;j< Int_t(eventsB.size()) ;j++)
               (*distances)[j + Int_t(eventsS.size())] = GetNormalizedDistance( e, *eventsB[j], dim_normalization );
         
            //counting the fkNNMin-th element    
            std::vector<Double_t>::iterator wsk = distances->begin();
            for (Int_t j=0;j<fkNNMin-1;j++) wsk++;
            std::nth_element( distances->begin(), wsk, distances->end() );
         
            //getting all elements that are closer than fkNNMin-th element
            //signals
            for (Int_t j=0;j<Int_t(eventsS.size());j++) {
               Double_t dist = GetNormalizedDistance( e, *eventsS[j], dim_normalization );
               
               if (dist <= (*distances)[fkNNMin-1])        
                  tempVectorS.push_back( eventsS[j] );
            }      
            //backgrounds
            for (Int_t j=0;j<Int_t(eventsB.size());j++) {
               Double_t dist = GetNormalizedDistance( e, *eventsB[j], dim_normalization );
            
               if (dist <= (*distances)[fkNNMin-1]) tempVectorB.push_back( eventsB[j] );
            }      
            fMax_distance = (*distances)[fkNNMin-1];
            delete distances;
         }      
         countS = KernelEstimate( e, tempVectorS, v );
         countB = KernelEstimate( e, tempVectorB, v );
      }
   else {
      // troubles ahead...
      fLogger << kFATAL << "<RScalc> unknown RangeMode: " << fVRangeMode << Endl;
   }
   // -----------------------------------------------------------------------

   delete volume;
   delete lb;
   delete ub;

   if (countS < 1e-20 && countB < 1e-20) return 0.5;
   if (countB < 1e-20) return 1.0;
   if (countS < 1e-20) return 0.0;

   Float_t r = countB*fScaleB/(countS*fScaleS);
   return 1.0/(r + 1.0);   // TODO: propagate errors from here
}

//_______________________________________________________________________
Double_t TMVA::MethodPDERS::KernelEstimate( const Event & event,
                                            std::vector<const BinarySearchTreeNode*>& events, Volume& v )
{
   // Final estimate
   Double_t pdfSum = 0;
  
   // normalization factors so we can work with radius 1 hyperspheres
   Double_t *dim_normalization = new Double_t[GetNvar()];
   for (Int_t ivar=0; ivar<GetNvar(); ivar++)
      dim_normalization [ivar] = 2 / ((*v.fUpper)[ivar] - (*v.fLower)[ivar]);
   
   // Iteration over sample points
   for (std::vector<const BinarySearchTreeNode*>::iterator iev = events.begin(); iev != events.end(); iev++) {
     
      // First switch to the one dimensional distance
      Double_t normalized_distance = GetNormalizedDistance (event, *(*iev), dim_normalization);
            
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
      return TMath::Gaus( normalized_distance, 0, fGaussSigmaNorm, kFALSE);
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
   case kTrim: {
      Double_t x = normalized_distance / fMax_distance;
      x = 1 - x*x*x;
      return x*x*x;
   }
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
   // Calculating the normalization factor only once (might need a reset at some point. 
   // Can the method be restarted with different params?)

   // Caching jammed to disable function. 
   // It's not really useful afterall, badly implemented and untested :-)
   static Double_t ret = 1.0; 
   
   if (ret != 0.0) return ret*pdf; 

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
      ret = 1. / TMath::Power ( 2 * TMath::Pi() * fGaussSigmaNorm * fGaussSigmaNorm, ((Double_t) GetNvar()) / 2.);
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
Double_t TMVA::MethodPDERS::GetNormalizedDistance ( const Event &base_event,
                                                    const BinarySearchTreeNode &sample_event,
                                                    Double_t *dim_normalization) 
{
   // We use Euclidian metric here. Might not be best or most efficient.
   Double_t ret=0;
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
      Double_t dist = dim_normalization[ivar] * (sample_event.GetEventV()[ivar] - base_event.GetVal(ivar));
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

   if (GetNvar() % 2) ret = TMath::Power (lanczos, GetNvar());
   else               ret = TMath::Abs (lanczos) * TMath::Power (lanczos, GetNvar() - 1);

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
   // write only a short comment to file
   if (TxtWeightsOnly()) {
      if (fBinaryTreeS)
         o << *fBinaryTreeS;
      else
         fLogger << kFATAL << "Signal binary search tree not available" << Endl; 
      
      if (fBinaryTreeB)
         o << *fBinaryTreeB;
      else
         fLogger << kFATAL << "Background binary search tree not available" << Endl; 

   } 
   else {
      TString rfname( GetWeightFileName() ); rfname.ReplaceAll( ".txt", ".root" );
      o << "# weights stored in root i/o file: " << rfname << endl;  
   }
}

//_______________________________________________________________________
void TMVA::MethodPDERS::ReadWeightsFromStream( istream& istr)
{
   // read weight info from file
   if (TxtWeightsOnly()) {
      if (NULL != fBinaryTreeS) delete fBinaryTreeS;
      if (NULL != fBinaryTreeB) delete fBinaryTreeB;

      fBinaryTreeS = new BinarySearchTree();
      fBinaryTreeB = new BinarySearchTree();
      istr >> *fBinaryTreeS >> *fBinaryTreeB;

      fBinaryTreeS->SetPeriode( GetVarTransform().Variables().size() );
      fBinaryTreeB->SetPeriode( GetVarTransform().Variables().size() );

      fBinaryTreeS->CalcStatistics();
      fBinaryTreeB->CalcStatistics();

      fBinaryTreeS->CountNodes();
      fBinaryTreeB->CountNodes();

      // these are the signal and background scales for the weights
      fScaleS = 1.0/fBinaryTreeS->GetSumOfWeights();
      fScaleB = 1.0/fBinaryTreeB->GetSumOfWeights();

      fLogger << kVERBOSE << "signal and background scales: " << fScaleS << " " << fScaleB << Endl;

      CalcAverages();

      SetVolumeElement();

      fInitializedVolumeEle = kTRUE;
   }
}

//_______________________________________________________________________
void TMVA::MethodPDERS::WriteWeightsToStream( TFile& ) const
{
   // write training sample (TTree) to file
}

//_______________________________________________________________________
void TMVA::MethodPDERS::ReadWeightsFromStream( TFile& /*rf*/ )
{
   // read training sample from file
}

//_______________________________________________________________________
void TMVA::MethodPDERS::MakeClassSpecific( std::ostream& fout, const TString& className ) const
{
   // write specific classifier response
   fout << "   // not implemented for class: \"" << className << "\"" << endl;
   fout << "};" << endl;
}

//_______________________________________________________________________
void TMVA::MethodPDERS::GetHelpMessage() const
{
   // get help message text
   //
   // typical length of text line: 
   //         "|--------------------------------------------------------------|"
   fLogger << Endl;
   fLogger << gTools().Color("bold") << "--- Short description:" << gTools().Color("reset") << Endl;
   fLogger << Endl;
   fLogger << "PDERS is a generalization of the projective likelihood classifier " << Endl;
   fLogger << "to N dimensions, where N is the number of input variables used." << Endl;
   fLogger << "In its adaptive form it is mostly equivalent to k-Nearest-Neighbor" << Endl;
   fLogger << "(k-NN) methods. If the multidimensional PDF for signal and background" << Endl;
   fLogger << "were known, this classifier would exploit the full information" << Endl;
   fLogger << "contained in the input variables, and would hence be optimal. In " << Endl;
   fLogger << "practice however, huge training samples are necessary to sufficiently " << Endl;
   fLogger << "populate the multidimensional phase space. " << Endl;
   fLogger << Endl;
   fLogger << "The simplest implementation of PDERS counts the number of signal" << Endl;
   fLogger << "and background events in the vicinity of a test event, and returns" << Endl;
   fLogger << "a weight according to the majority species of the neighboring events." << Endl;
   fLogger << "A more involved version of PDERS (selected by the option \"KernelEstimator\")" << Endl;
   fLogger << "uses Kernel estimation methods to approximate the shape of the PDF." << Endl;
   fLogger << Endl;
   fLogger << gTools().Color("bold") << "--- Performance optimisation:" << gTools().Color("reset") << Endl;
   fLogger << Endl;
   fLogger << "PDERS can be very powerful in case of strongly non-linear problems, " << Endl;
   fLogger << "e.g., distinct islands of signal and background regions. Because of " << Endl;
   fLogger << "the exponential growth of the phase space, it is important to restrict" << Endl;
   fLogger << "the number of input variables (dimension) to the strictly necessary." << Endl;
   fLogger << Endl;
   fLogger << "Note that PDERS is a slowly responding classifier. Moreover, the necessity" << Endl;
   fLogger << "to store the entire binary tree in memory, to avoid accessing virtual " << Endl;
   fLogger << "memory, limits the number of training events that can effectively be " << Endl;
   fLogger << "used to model the multidimensional PDF." << Endl;
   fLogger << Endl;
   fLogger << gTools().Color("bold") << "--- Performance tuning via configuration options:" << gTools().Color("reset") << Endl;
   fLogger << Endl;
   fLogger << "If the PDERS response is found too slow when using the adaptive volume " << Endl;
   fLogger << "size (option \"VolumeRangeMode=Adaptive\"), it might be found beneficial" << Endl;
   fLogger << "to reduce the number of events required in the volume, and/or to enlarge" << Endl;
   fLogger << "the allowed range (\"NeventsMin/Max\"). PDERS is relatively insensitive" << Endl;
   fLogger << "to the width (\"GaussSigma\") of the Gaussian kernel (if used)." << Endl;
}

