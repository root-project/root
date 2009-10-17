// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Matt Jachowski, Joerg Stelzer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodMLP                                                             *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      ANN Multilayer Perceptron class for the discrimination of signal          *
 *      from background. BFGS implementation based on TMultiLayerPerceptron       *
 *      class from ROOT (http://root.cern.ch).                                    *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Krzysztof Danielowski <danielow@cern.ch>       - IFJ & AGH, Poland        *
 *      Andreas Hoecker       <Andreas.Hocker@cern.ch> - CERN, Switzerland        *
 *      Matt Jachowski        <jachowski@stanford.edu> - Stanford University, USA *
 *      Kamil Kraszewski      <kalq@cern.ch>           - IFJ & UJ, Poland         *
 *      Maciej Kruk           <mkruk@cern.ch>          - IFJ & AGH, Poland        *
 *      Joerg Stelzer         <stelzer@cern.ch>        - DESY, Germany            *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
//                                                                      
// Multilayer Perceptron class built off of MethodANNBase  
//_______________________________________________________________________

#include "TString.h"
#include <vector>
#include "TTree.h"
#include "Riostream.h"
#include "TFitter.h"
#include "TMatrixD.h"
#include "TMath.h"

#include "TMVA/ClassifierFactory.h"
#include "TMVA/Interval.h"
#include "TMVA/MethodMLP.h"
#include "TMVA/TNeuron.h"
#include "TMVA/TSynapse.h"
#include "TMVA/Timer.h"
#include "TMVA/Types.h"
#include "TMVA/Tools.h"
#include "TMVA/GeneticFitter.h"
#include "TMVA/Config.h"

#ifdef MethodMLP_UseMinuit__
TMVA::MethodMLP* TMVA::MethodMLP::fgThis = 0;
Bool_t MethodMLP_UseMinuit = kTRUE;
#endif

REGISTER_METHOD(MLP)

ClassImp(TMVA::MethodMLP)

using std::vector;

//______________________________________________________________________________
TMVA::MethodMLP::MethodMLP( const TString& jobName,
                            const TString& methodTitle,
                            DataSetInfo& theData, 
                            const TString& theOption,
                            TDirectory* theTargetDir ) 
   : MethodANNBase( jobName, Types::kMLP, methodTitle, theData, theOption, theTargetDir ),
     fSamplingFraction(1.0),
     fSamplingEpoch   (0.0)
{
   // standard constructor
}

//______________________________________________________________________________
TMVA::MethodMLP::MethodMLP( DataSetInfo& theData,
                            const TString& theWeightFile,
                            TDirectory* theTargetDir ) 
   : MethodANNBase( Types::kMLP, theData, theWeightFile, theTargetDir ),
     fSamplingFraction(1.0),
     fSamplingEpoch(0.0)
{
   // constructor from a weight file
}

//______________________________________________________________________________
TMVA::MethodMLP::~MethodMLP()
{
   // destructor
   // nothing to be done
}

//_______________________________________________________________________
Bool_t TMVA::MethodMLP::HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses, UInt_t /*numberTargets*/ )
{
   // MLP can handle classification with 2 classes and regression with one regression-target
   if (type == Types::kClassification && numberClasses == 2 ) return kTRUE;
   //   if (type == Types::kRegression     && numberTargets == 1 ) return kTRUE;
   if (type == Types::kRegression ) return kTRUE;

   return kFALSE;
}

//_______________________________________________________________________
void TMVA::MethodMLP::SetAnalysisType( Types::EAnalysisType type )
{
   MethodBase::SetAnalysisType( type );
   MethodANNBase::SetAnalysisType( type );
}

//______________________________________________________________________________
void TMVA::MethodMLP::Init()
{
   // default initializations

   // the minimum requirement to declare an event signal-like
   SetSignalReferenceCut( 0.5 );
}

//_______________________________________________________________________
void TMVA::MethodMLP::DeclareOptions() 
{
   // define the options (their key words) that can be set in the option string 
   // know options:
   // TrainingMethod  <string>     Training method
   //    available values are:         BP   Back-Propagation <default>
   //                                  GA   Genetic Algorithm (takes a LONG time)
   //
   // LearningRate    <float>      NN learning rate parameter
   // DecayRate       <float>      Decay rate for learning parameter
   // TestRate        <int>        Test for overtraining performed at each #th epochs
   //
   // BPMode          <string>     Back-propagation learning mode
   //    available values are:         sequential <default>
   //                                  batch
   //
   // BatchSize       <int>        Batch size: number of events/batch, only set if in Batch Mode, 
   //                                          -1 for BatchSize=number_of_events

   DeclareOptionRef(fTrainMethodS="BP", "TrainingMethod", 
                    "Train with Back-Propagation (BP), BFGS Algorithm (BFGS), or Genetic Algorithm (GA - slower and worse)");
   AddPreDefVal(TString("BP"));
   AddPreDefVal(TString("GA"));
   AddPreDefVal(TString("BFGS"));

   DeclareOptionRef(fLearnRate=0.02, "LearningRate", "ANN learning rate parameter");
   DeclareOptionRef(fDecayRate=0.01, "DecayRate",    "Decay rate for learning parameter");
   DeclareOptionRef(fTestRate =10,   "TestRate",     "Test for overtraining performed at each #th epochs");

   DeclareOptionRef(fSamplingFraction=1.0, "Sampling","Only 'Sampling' (randomly selected) events are trained each epoch");
   DeclareOptionRef(fSamplingEpoch=1.0,    "SamplingEpoch","Sampling is used for the first 'SamplingEpoch' epochs, afterwards, all events are taken for training");
   DeclareOptionRef(fSamplingWeight=1.0,    "SamplingImportance"," The sampling weights of events in epochs which successful (worse estimator than before) are multiplied with SamplingImportance, else they are divided.");

   DeclareOptionRef(fSamplingTraining=kTRUE,    "SamplingTraining","The training sample is sampled");
   DeclareOptionRef(fSamplingTesting= kFALSE,    "SamplingTesting" ,"The testing sample is sampled");

   DeclareOptionRef(fResetStep=50,   "ResetStep",    "How often BFGS should reset history");
   DeclareOptionRef(fTau      =3.0,  "Tau",          "LineSearch \"size step\"");

   DeclareOptionRef(fBpModeS="sequential", "BPMode", 
                    "Back-propagation learning mode: sequential or batch");
   AddPreDefVal(TString("sequential"));
   AddPreDefVal(TString("batch"));

   DeclareOptionRef(fBatchSize=-1, "BatchSize", 
                    "Batch size: number of events/batch, only set if in Batch Mode, -1 for BatchSize=number_of_events");

   DeclareOptionRef(fImprovement=1e-30, "ConvergenceImprove", 
                    "Minimum improvement which counts as improvement (<0 means automatic convergence check is turned off)");

   DeclareOptionRef(fSteps=-1, "ConvergenceTests", 
                    "Number of steps (without improvement) required for convergence (<0 means automatic convergence check is turned off)");

}

//_______________________________________________________________________
void TMVA::MethodMLP::ProcessOptions() 
{
   // process user options
   MethodANNBase::ProcessOptions();

   if (IgnoreEventsWithNegWeightsInTraining()) {
      Log() << kFATAL << "Mechanism to ignore events with negative weights in training not yet available for method: "
            << GetMethodTypeName() 
            << " --> please remove \"IgnoreNegWeightsInTraining\" option from booking string."
            << Endl;
   }

   if      (fTrainMethodS == "BP"  ) fTrainingMethod = kBP;
   else if (fTrainMethodS == "BFGS") fTrainingMethod = kBFGS;
   else if (fTrainMethodS == "GA"  ) fTrainingMethod = kGA;

   if      (fBpModeS == "sequential") fBPMode = kSequential;
   else if (fBpModeS == "batch")      fBPMode = kBatch;

   //   InitializeLearningRates();

   if (fBPMode == kBatch) {
      Data()->SetCurrentType(Types::kTraining);
      Int_t numEvents = Data()->GetNEvents();
      if (fBatchSize < 1 || fBatchSize > numEvents) fBatchSize = numEvents;
   }
}

//______________________________________________________________________________
void TMVA::MethodMLP::InitializeLearningRates()
{
   // initialize learning rates of synapses, used only by backpropagation
   Log() << kDEBUG << "Initialize learning rates" << Endl;
   TSynapse *synapse;
   Int_t numSynapses = fSynapses->GetEntriesFast();
   for (Int_t i = 0; i < numSynapses; i++) {
      synapse = (TSynapse*)fSynapses->At(i);
      synapse->SetLearningRate(fLearnRate);
   }
}

//______________________________________________________________________________ 
Double_t TMVA::MethodMLP::CalculateEstimator( Types::ETreeType treeType )
{
   // calculate the estimator that training is attempting to minimize

   // sanity check
   if (treeType!=Types::kTraining && treeType!=Types::kTesting)
      Log() << kFATAL << "<CalculateEstimator> fatal error: wrong tree type: " << treeType << Endl;

   Types::ETreeType saveType = Data()->GetCurrentType();
   Data()->SetCurrentType(treeType);

   Double_t estimator = 0;

   // loop over all training events 
   Int_t nEvents = GetNEvents();
   UInt_t nTgts = DataInfo().GetNTargets();
   Double_t d = 0;
   for (Int_t i = 0; i < nEvents; i++) {

      const Event* ev = GetEvent(i);

      ForceNetworkInputs( ev );
      ForceNetworkCalculations();

      d = 0;
      if( DoRegression() ){
         for( UInt_t itgt = 0; itgt < nTgts; itgt++ ){
            Double_t dt = GetOutputNeuron( itgt)->GetActivationValue() - ev->GetTarget( itgt );
            d += (dt*dt);
         }
         d = TMath::Sqrt(d);
      }else{
         Double_t desired = GetDesiredOutput( ev );
         d = GetOutputNeuron()->GetActivationValue() - desired;
      }
      estimator += (d*d);
   }

   if (DoRegression()) estimator = TMath::Sqrt(estimator/Float_t(nEvents));
   else                estimator = estimator*0.5/Float_t(nEvents);

   Data()->SetCurrentType( saveType );

   return estimator;
}

//______________________________________________________________________________
void TMVA::MethodMLP::Train(Int_t nEpochs)
{
   if (fNetwork == 0){
      Log() <<kERROR <<"ANN Network is not initialized, doing it now!"<< Endl;
      SetAnalysisType(GetAnalysisType());
   }
   Log() << kDEBUG << "reinitalize learning rates" << Endl;
   InitializeLearningRates();
   PrintMessage("Training Network");
#ifdef MethodMLP_UseMinuit__  
   if (useMinuit) MinuitMinimize();
#else
   if (fTrainingMethod == kGA)        GeneticMinimize();
   else if (fTrainingMethod == kBFGS) BFGSMinimize(nEpochs);
   else                               BackPropagationMinimize(nEpochs);
#endif

}

//______________________________________________________________________________
void TMVA::MethodMLP::BFGSMinimize( Int_t nEpochs )
{
   // train network with BFGS algorithm

   Timer timer( (fSteps>0?100:nEpochs), GetName() );

   // create histograms for overtraining monitoring
   Int_t nbinTest = Int_t(nEpochs/fTestRate);
   fEstimatorHistTrain = new TH1F( "estimatorHistTrain", "training estimator", 
                                   nbinTest, Int_t(fTestRate/2), nbinTest*fTestRate+Int_t(fTestRate/2) );
   fEstimatorHistTest  = new TH1F( "estimatorHistTest", "test estimator", 
                                   nbinTest, Int_t(fTestRate/2), nbinTest*fTestRate+Int_t(fTestRate/2) );

   Int_t nSynapses = fSynapses->GetEntriesFast();
   Int_t nWeights  = nSynapses;

   for (Int_t i=0;i<nSynapses;i++) {
      TSynapse* synapse = (TSynapse*)fSynapses->At(i);
      synapse->SetDEDw(0.0);
   }

   std::vector<Double_t> buffer( nWeights );
   for (Int_t i=0;i<nWeights;i++) buffer[i] = 0.;

   TMatrixD Dir     ( nWeights, 1 ); 
   TMatrixD Hessian ( nWeights, nWeights );
   TMatrixD Gamma   ( nWeights, 1 );
   TMatrixD Delta   ( nWeights, 1 );

   fLastAlpha = 0.;

   if (fSteps > 0) Log() << kINFO << "Inaccurate progress timing for MLP... " << Endl;
   timer.DrawProgressBar( 0 );

   // start training cycles (epochs)
   for (Int_t i = 0; i < nEpochs; i++) {
      
      if (Float_t(i)/nEpochs < fSamplingEpoch) {
         if ((i+1)%fTestRate == 0 || (i == 0)) {
            if (fSamplingTraining) {
               Data()->SetCurrentType( Types::kTraining );
               Data()->InitSampling(fSamplingFraction,fSamplingWeight);
               Data()->CreateSampling();
            }
            if (fSamplingTesting) {
               Data()->SetCurrentType( Types::kTesting );
               Data()->InitSampling(fSamplingFraction,fSamplingWeight);
               Data()->CreateSampling();
            }
         }
      }
      else {
         Data()->SetCurrentType( Types::kTraining );
         Data()->InitSampling(1.0,1.0);
         Data()->SetCurrentType( Types::kTesting );
         Data()->InitSampling(1.0,1.0);
      }
      Data()->SetCurrentType( Types::kTraining );

      SetGammaDelta( Gamma, Delta, buffer );

      if (i % fResetStep == 0) {
         SteepestDir( Dir );
         Hessian.UnitMatrix();
      }
      else {
         if (GetHessian( Hessian, Gamma, Delta )) {
            SteepestDir( Dir );
            Hessian.UnitMatrix();
         }
         else SetDir( Hessian, Dir );
      }

      if (DerivDir( Dir ) > 0) {
         SteepestDir( Dir );
         Hessian.UnitMatrix();
      }
      if (LineSearch( Dir, buffer )) {
         Hessian.UnitMatrix();
         SteepestDir( Dir );
         if (LineSearch(Dir, buffer)) {
            i = nEpochs;
            Log() << kFATAL << "Line search failed! Huge troubles somewhere..." << Endl;
         }
      }

      // monitor convergence of training and control sample
      if ((i+1)%fTestRate == 0) {
         Double_t trainE = CalculateEstimator( Types::kTraining ); // estimator for training sample
         Double_t testE  = CalculateEstimator( Types::kTesting  );  // estimator for test sample
         fEstimatorHistTrain->Fill( i+1, trainE );
         fEstimatorHistTest ->Fill( i+1, testE );

         Bool_t success = kFALSE;
         if( (testE < GetCurrentValue()) || (GetCurrentValue()<1e-100) ){
            success = kTRUE;
         }
         Data()->EventResult( success );
         
         SetCurrentValue( testE );
         if (HasConverged() ){
            if (Float_t(i)/nEpochs < fSamplingEpoch ){
               Int_t newEpoch = Int_t(fSamplingEpoch*nEpochs);
               i = newEpoch;
               ResetConvergenceCounter();
            }
            else{
               break;
            }
         }
      }
      
      // draw progress
      if (fSteps > 0) {
         Float_t progress = 0;
         if (Float_t(i)/nEpochs < fSamplingEpoch) 
            progress = Progress()*fSamplingEpoch*fSamplingFraction*100;
         else
            progress = 100.0*(fSamplingEpoch*fSamplingFraction+(1.0-fSamplingFraction*fSamplingEpoch)*Progress());
         
         timer.DrawProgressBar( Int_t(progress) );
      }
      else timer.DrawProgressBar( i );

      // some verbose output
      if (fgPRINT_SEQ) {
         PrintNetwork();
         WaitForKeyboard();
      }
   }
}

//______________________________________________________________________________
void TMVA::MethodMLP::SetGammaDelta( TMatrixD &Gamma, TMatrixD &Delta, std::vector<Double_t> &buffer )
{
   Int_t nWeights = fSynapses->GetEntriesFast();

   Int_t IDX = 0;
   Int_t nSynapses = fSynapses->GetEntriesFast();
   for (Int_t i=0;i<nSynapses;i++) {
      TSynapse *synapse = (TSynapse*)fSynapses->At(i);
      Gamma[IDX++][0] = -synapse->GetDEDw();
   }

   for (Int_t i=0;i<nWeights;i++) Delta[i][0] = buffer[i];

   ComputeDEDw();

   IDX = 0;
   for (Int_t i=0;i<nSynapses;i++)
      {
         TSynapse *synapse = (TSynapse*)fSynapses->At(i);
         Gamma[IDX++][0] += synapse->GetDEDw();
      }
}

//______________________________________________________________________________
void TMVA::MethodMLP::ComputeDEDw()
{
   Int_t nSynapses = fSynapses->GetEntriesFast();
   for (Int_t i=0;i<nSynapses;i++) {
      TSynapse *synapse = (TSynapse*)fSynapses->At(i);
      synapse->SetDEDw( 0.0 );
   }

   Int_t nEvents = GetNEvents();
   for (Int_t i=0;i<nEvents;i++) {

      const Event* ev = GetEvent(i);

      SimulateEvent( ev );

      for (Int_t j=0;j<nSynapses;j++) {
         TSynapse *synapse = (TSynapse*)fSynapses->At(j);
         synapse->SetDEDw( synapse->GetDEDw() + synapse->GetDelta() );
      }
   }

   for (Int_t i=0;i<nSynapses;i++) {
      TSynapse *synapse = (TSynapse*)fSynapses->At(i);
      synapse->SetDEDw( synapse->GetDEDw() / nEvents );
   }
}

//______________________________________________________________________________
void TMVA::MethodMLP::SimulateEvent( const Event* ev )
{
   Double_t eventWeight = ev->GetWeight();

   ForceNetworkInputs( ev );
   ForceNetworkCalculations();

   if( DoRegression() ){
      UInt_t ntgt = DataInfo().GetNTargets();
      for( UInt_t itgt = 0; itgt < ntgt; itgt++ ){
         Double_t desired     = ev->GetTarget(itgt);
         Double_t error = ( GetOutputNeuron( itgt )->GetActivationValue() - desired )*eventWeight;
         GetOutputNeuron( itgt )->SetError(error);
      }
   }else{
      Double_t desired     = GetDesiredOutput( ev );
      Double_t error = ( GetOutputNeuron()->GetActivationValue() - desired )*eventWeight;
      GetOutputNeuron()->SetError(error);
   }

   CalculateNeuronDeltas();
   for (Int_t j=0;j<fSynapses->GetEntriesFast();j++) {
      TSynapse *synapse = (TSynapse*)fSynapses->At(j);
      synapse->InitDelta();
      synapse->CalculateDelta();
   }
}

//______________________________________________________________________________
void TMVA::MethodMLP::SteepestDir( TMatrixD &Dir )
{
   Int_t IDX = 0;
   Int_t nSynapses = fSynapses->GetEntriesFast();

   for (Int_t i=0;i<nSynapses;i++) {
      TSynapse *synapse = (TSynapse*)fSynapses->At(i);
      Dir[IDX++][0] = -synapse->GetDEDw();
   }
}

//______________________________________________________________________________
Bool_t TMVA::MethodMLP::GetHessian( TMatrixD &Hessian, TMatrixD &Gamma, TMatrixD &Delta )
{
   TMatrixD gd(Gamma, TMatrixD::kTransposeMult, Delta);
   if ((Double_t) gd[0][0] == 0.) return kTRUE;
   TMatrixD aHg(Hessian, TMatrixD::kMult, Gamma);
   TMatrixD tmp(Gamma,   TMatrixD::kTransposeMult, Hessian);
   TMatrixD gHg(Gamma,   TMatrixD::kTransposeMult, aHg);
   Double_t a = 1 / (Double_t) gd[0][0];
   Double_t f = 1 + ((Double_t)gHg[0][0]*a);
   TMatrixD res(TMatrixD(Delta, TMatrixD::kMult, TMatrixD(TMatrixD::kTransposed,Delta)));
   res *= f;
   res -= (TMatrixD(Delta, TMatrixD::kMult, tmp) + TMatrixD(aHg, TMatrixD::kMult,
                                                            TMatrixD(TMatrixD::kTransposed,Delta)));
   res *= a;
   Hessian += res;

   return kFALSE;
}

//______________________________________________________________________________
void TMVA::MethodMLP::SetDir( TMatrixD &Hessian, TMatrixD &dir )
{
   Int_t IDX = 0;
   Int_t nSynapses = fSynapses->GetEntriesFast();
   TMatrixD DEDw(nSynapses, 1);

   for (Int_t i=0;i<nSynapses;i++) {
      TSynapse *synapse = (TSynapse*)fSynapses->At(i);
      DEDw[IDX++][0] = synapse->GetDEDw();
   }

   dir = Hessian * DEDw;
   for (Int_t i=0;i<IDX;i++) dir[i][0] = -dir[i][0];
}

//______________________________________________________________________________
Double_t TMVA::MethodMLP::DerivDir( TMatrixD &Dir )
{
   Int_t IDX = 0;
   Int_t nSynapses = fSynapses->GetEntriesFast();
   Double_t Result = 0.0;

   for (Int_t i=0;i<nSynapses;i++) {
      TSynapse *synapse = (TSynapse*)fSynapses->At(i);
      Result += Dir[IDX++][0] * synapse->GetDEDw();
   }
	
   return Result;
}

//______________________________________________________________________________
Bool_t TMVA::MethodMLP::LineSearch(TMatrixD &Dir, std::vector<Double_t> &buffer)
{
   Int_t IDX = 0;
   Int_t nSynapses = fSynapses->GetEntriesFast();
   Int_t nWeights = nSynapses;

   std::vector<Double_t> Origin(nWeights);
	
   for (Int_t i=0;i<nSynapses;i++) {
      TSynapse *synapse = (TSynapse*)fSynapses->At(i);
      Origin[i] = synapse->GetWeight();
   }

   Double_t err1 = GetError();
   Double_t alpha1 = 0.;
   Double_t alpha2 = fLastAlpha;


   if      (alpha2 < 0.01) alpha2 = 0.01;
   else if (alpha2 > 2.0)  alpha2 = 2.0;
   Double_t alpha_original = alpha2;
   Double_t alpha3 = alpha2;

   SetDirWeights( Origin, Dir, alpha2 );
   Double_t err2 = GetError();
   //Double_t err2 = err1; 
   Double_t err3 = err2;
   Bool_t bingo = kFALSE;


   if (err1 > err2) {
      for (Int_t i=0;i<100;i++)  {
         alpha3 *= fTau;
         SetDirWeights(Origin, Dir, alpha3);
         err3 = GetError();
         if (err3 > err2) {
            bingo = kTRUE;
            break;
         }
         alpha1 = alpha2;
         err1 = err2;
         alpha2 = alpha3;
         err2 = err3;
      }
      if (!bingo) {
         SetDirWeights(Origin, Dir, 0.);
         return kTRUE;
      }
   }
   else {
      for (Int_t i=0;i<100;i++) {
         alpha2 /= fTau;
         if (i==50){
            Log() << kWARNING << "linesearch, starting to investigate direction opposite of steepestDIR" << Endl;
            alpha2 = -alpha_original;
         }
         SetDirWeights(Origin, Dir, alpha2);
         err2 = GetError();
         if (err1 > err2) {
            bingo = kTRUE;
            break;
         }
         alpha3 = alpha2;
         err3 = err2;
      }
      if (!bingo) {
         SetDirWeights(Origin, Dir, 0.);
         fLastAlpha = 0.05;
         return kTRUE;
      }
   }

   if (alpha1>0 && alpha2>0 && alpha3 > 0){
      fLastAlpha = 0.5 * (alpha1 + alpha3 - 
                          (err3 - err1) / ((err3 - err2) / ( alpha3 - alpha2 )
                                           - ( err2 - err1 ) / (alpha2 - alpha1 )));
   }
   else {
      fLastAlpha = alpha2;
   }

   fLastAlpha = fLastAlpha < 10000 ? fLastAlpha : 10000;

   SetDirWeights(Origin, Dir, fLastAlpha);

   // leaving these lines uncommented is a heavy price to pay for only a warning message 
   // (which shoulnd't appear anyway)
   // --> about 15% of time is spent in the final GetError().
   //    
   Double_t finalError = GetError();
   if (finalError > err1) {
      Log() << kWARNING << "Line search increased error! Something is wrong."
            << "fLastAlpha=" << fLastAlpha << "al123=" << alpha1 << " " 
            << alpha2 << " " << alpha3 << " err1="<< err1 << " errfinal=" << finalError << Endl;	
   } 

   for (Int_t i=0;i<nSynapses;i++) {
      TSynapse *synapse = (TSynapse*)fSynapses->At(i);
      buffer[IDX] = synapse->GetWeight() - Origin[IDX];
      IDX++;
   }
	
   return kFALSE;
}

//______________________________________________________________________________
void TMVA::MethodMLP::SetDirWeights( std::vector<Double_t> &Origin, TMatrixD &Dir, Double_t alpha )
{
   Int_t IDX = 0;
   Int_t nSynapses = fSynapses->GetEntriesFast();

   for (Int_t i=0;i<nSynapses;i++) {
      TSynapse *synapse = (TSynapse*)fSynapses->At(i);
      synapse->SetWeight( Origin[IDX] + Dir[IDX][0] * alpha );
      IDX++;
   }
}


//______________________________________________________________________________
Double_t TMVA::MethodMLP::GetError()
{
   Int_t nEvents = GetNEvents();
   UInt_t ntgts = GetNTargets();
   Double_t Result = 0.;

   for (Int_t i=0;i<nEvents;i++) {
      const Event* ev = GetEvent(i);

      SimulateEvent( ev );

      Double_t error = 0.;
      if( DoRegression() ){
         for( UInt_t itgt = 0; itgt < ntgts; itgt++ ){
            error += GetSqrErr( ev, itgt );
         }
      }else{
         error = GetSqrErr( ev );
      }
      Result += error * ev->GetWeight();   
   }
   return Result;
}

//______________________________________________________________________________
Double_t TMVA::MethodMLP::GetSqrErr( const Event* ev, UInt_t index )
{
   Double_t error = 0;
   Double_t output = GetOutputNeuron( index )->GetActivationValue();
   Double_t target = 0;
   if( DoRegression() ){
      target = ev->GetTarget( index );
   }else{
      target = GetDesiredOutput( ev );  
   }

   error = (output-target)*(output-target);

   return error;
}

//______________________________________________________________________________
void TMVA::MethodMLP::BackPropagationMinimize(Int_t nEpochs)
{
   // minimize estimator / train network with backpropagation algorithm

   //    Timer timer( nEpochs, GetName() );
   Timer timer( (fSteps>0?100:nEpochs), GetName() );
   Int_t lateEpoch = (Int_t)(nEpochs*0.95) - 1;

   // create histograms for overtraining monitoring
   Int_t nbinTest = Int_t(nEpochs/fTestRate);
   fEstimatorHistTrain = new TH1F( "estimatorHistTrain", "training estimator", 
                                   nbinTest, Int_t(fTestRate/2), nbinTest*fTestRate+Int_t(fTestRate/2) );
   fEstimatorHistTest  = new TH1F( "estimatorHistTest", "test estimator", 
                                   nbinTest, Int_t(fTestRate/2), nbinTest*fTestRate+Int_t(fTestRate/2) );

   if (fSteps > 0) Log() << kINFO << "Inaccurate progress timing for MLP... " << Endl;
   timer.DrawProgressBar(0);

   // start training cycles (epochs)
   for (Int_t i = 0; i < nEpochs; i++) {

      if (Float_t(i)/nEpochs < fSamplingEpoch ){
         if ((i+1)%fTestRate == 0 || (i == 0)) {
            if (fSamplingTraining ){
               Data()->SetCurrentType( Types::kTraining );
               Data()->InitSampling(fSamplingFraction,fSamplingWeight);
               Data()->CreateSampling();
            }
            if (fSamplingTesting ){
               Data()->SetCurrentType( Types::kTesting );
               Data()->InitSampling(fSamplingFraction,fSamplingWeight);
               Data()->CreateSampling();
            }
         }
      }
      else {
         Data()->SetCurrentType( Types::kTraining );
         Data()->InitSampling(1.0,1.0);
         Data()->SetCurrentType( Types::kTesting );
         Data()->InitSampling(1.0,1.0);
      }
      Data()->SetCurrentType( Types::kTraining );
            
      TrainOneEpoch();
      DecaySynapseWeights(i >= lateEpoch);
      
      // monitor convergence of training and control sample
      if ((i+1)%fTestRate == 0) {
         Double_t trainE = CalculateEstimator( Types::kTraining ); // estimator for training sample
         Double_t testE  = CalculateEstimator( Types::kTesting  );  // estimator for test samplea
         fEstimatorHistTrain->Fill( i+1, trainE );
         fEstimatorHistTest ->Fill( i+1, testE );

         Bool_t success = kFALSE;
         if( (testE < GetCurrentValue()) || (GetCurrentValue()<1e-100) ){
            success = kTRUE;
         }
         Data()->EventResult( success );
         
         SetCurrentValue( testE );
         if (HasConverged() ){
            if (Float_t(i)/nEpochs < fSamplingEpoch ){
               Int_t newEpoch = Int_t(fSamplingEpoch*nEpochs);
               i = newEpoch;
               ResetConvergenceCounter();
            }
            else{
               if (lateEpoch > i ){
                  lateEpoch = i;
               }else{
                  break;
               }
            }
         }
      }

      // draw progress bar
      if (fSteps > 0) {
         Float_t progress = 0;
         if (Float_t(i)/nEpochs < fSamplingEpoch) 
            progress = Progress()*fSamplingEpoch*fSamplingFraction*100;
         else
            progress = 100*(fSamplingEpoch*fSamplingFraction+(1.0-fSamplingFraction*fSamplingEpoch)*Progress());
         
         timer.DrawProgressBar( Int_t(progress) );
      }
      else {
         timer.DrawProgressBar(i);
      }


   }
}

//______________________________________________________________________________
void TMVA::MethodMLP::TrainOneEpoch()
{     
   // train network over a single epoch/cyle of events

   Int_t nEvents = Data()->GetNEvents();
     
   // randomize the order events will be presented, important for sequential mode
   Int_t* index = new Int_t[nEvents];
   for (Int_t i = 0; i < nEvents; i++) index[i] = i;
   Shuffle(index, nEvents);

   // loop over all training events
   for (Int_t i = 0; i < nEvents; i++) {

      TrainOneEvent(index[i]);

      // do adjustments if in batch mode
      if (fBPMode == kBatch && (i+1)%fBatchSize == 0) {
         AdjustSynapseWeights();
         if (fgPRINT_BATCH) {
            PrintNetwork();
            WaitForKeyboard();
         }
      }
     
      // debug in sequential mode
      if (fgPRINT_SEQ) {
         PrintNetwork();
         WaitForKeyboard();
      }
   }
   
   delete[] index;
}

//______________________________________________________________________________
void TMVA::MethodMLP::Shuffle(Int_t* index, Int_t n) 
{
   // Input:
   //   index: the array to shuffle
   //   n: the size of the array
   // Output:
   //   index: the shuffled indexes
   // This method is used for sequential training

   Int_t j, k;
   Int_t a = n - 1;
   for (Int_t i = 0; i < n; i++) {
      j = (Int_t) (frgen->Rndm() * a);
      k = index[j];
      index[j] = index[i];
      index[i] = k;
   }
}

//______________________________________________________________________________
void TMVA::MethodMLP::DecaySynapseWeights(Bool_t lateEpoch)
{
   // decay synapse weights
   // in last 10 epochs, lower learning rate even more to find a good minimum

   TSynapse* synapse;
   Int_t numSynapses = fSynapses->GetEntriesFast();
   for (Int_t i = 0; i < numSynapses; i++) {
      synapse = (TSynapse*)fSynapses->At(i);
      if (lateEpoch) synapse->DecayLearningRate(fDecayRate*fDecayRate);
      else           synapse->DecayLearningRate(fDecayRate);
   }
}

//______________________________________________________________________________
void TMVA::MethodMLP::TrainOneEventFast(Int_t ievt, Float_t*& branchVar, Int_t& type)
{
   // fast per-event training

   GetEvent(ievt);

   // as soon as we know how to get event weights, get that here
   
   // note: the normalization of event weights will affect the choice
   // of learning rate, one will have to experiment to get the right value.
   // in general, if the "average" event weight is 1, the learning rate
   // should be good if set around 0.02 (a good value if all event weights are 1)
   Double_t eventWeight = 1.0;
   
   // get the desired output of this event
   Double_t desired;
   if (type == 0) desired = fActivation->GetMin();  // background
   else           desired = fActivation->GetMax();  // signal

   // force the value for each input neuron
   Double_t x;
   TNeuron* neuron;
   
   for (UInt_t j = 0; j < GetNvar(); j++) {
      x = branchVar[j];
      if (IsNormalised()) x = gTools().NormVariable( x, GetXmin( j ), GetXmax( j ) );
      neuron = GetInputNeuron(j);
      neuron->ForceValue(x);
   }
   
   ForceNetworkCalculations();
   UpdateNetwork(desired, eventWeight);
}

//______________________________________________________________________________
void TMVA::MethodMLP::TrainOneEvent(Int_t ievt)
{
   // train network over a single event
   // this uses the new event model

   // note: the normalization of event weights will affect the choice
   // of learning rate, one will have to experiment to get the right value.
   // in general, if the "average" event weight is 1, the learning rate
   // should be good if set around 0.02 (a good value if all event weights are 1)

   const Event * ev = GetEvent(ievt);
   Double_t eventWeight = ev->GetWeight();
   ForceNetworkInputs( ev );
   ForceNetworkCalculations();
   if( DoRegression() ){
      UpdateNetwork( ev->GetTargets(), eventWeight );
   }else{
      UpdateNetwork( GetDesiredOutput( ev ), eventWeight );
   }
}

//______________________________________________________________________________
Double_t TMVA::MethodMLP::GetDesiredOutput( const Event* ev )
{
   // get the desired output of this event
   return DataInfo().IsSignal(ev)?fActivation->GetMax():fActivation->GetMin();
}


//______________________________________________________________________________
void TMVA::MethodMLP::UpdateNetwork(Double_t desired, Double_t eventWeight)
{
   // update the network based on how closely
   // the output matched the desired output
   Double_t error = GetOutputNeuron()->GetActivationValue() - desired;
   error *= eventWeight;
   GetOutputNeuron()->SetError(error);
   CalculateNeuronDeltas();
   UpdateSynapses();
}

//______________________________________________________________________________
void TMVA::MethodMLP::UpdateNetwork(std::vector<Float_t>& desired, Double_t eventWeight)
{
   // update the network based on how closely
   // the output matched the desired output
   for( UInt_t i = 0; i < DataInfo().GetNTargets(); i++ ){
      Double_t error = GetOutputNeuron( i )->GetActivationValue() - desired.at(i);
      error *= eventWeight;
      GetOutputNeuron( i )->SetError(error);
   }
   CalculateNeuronDeltas();
   UpdateSynapses();
}


//______________________________________________________________________________
void TMVA::MethodMLP::CalculateNeuronDeltas()
{
   // have each neuron calculate its delta by backpropagation

   TNeuron* neuron;
   Int_t    numNeurons;
   Int_t    numLayers = fNetwork->GetEntriesFast();
   TObjArray* curLayer;

   // step backwards through the network (backpropagation)
   // deltas calculated starting at output layer
   for (Int_t i = numLayers-1; i >= 0; i--) {
      curLayer = (TObjArray*)fNetwork->At(i);
      numNeurons = curLayer->GetEntriesFast();
  
      for (Int_t j = 0; j < numNeurons; j++) {
         neuron = (TNeuron*) curLayer->At(j);
         neuron->CalculateDelta();
      }
   }
}

//______________________________________________________________________________
void TMVA::MethodMLP::GeneticMinimize()
{
   // create genetics class similar to GeneticCut
   // give it vector of parameter ranges (parameters = weights)
   // link fitness function of this class to ComputeEstimator
   // instantiate GA (see MethodCuts)
   // run it
   // then this should exist for GA, Minuit and random sampling

   PrintMessage("Minimizing Estimator with GA");

   // define GA parameters
   fGA_preCalc   = 1;
   fGA_SC_steps  = 10;
   fGA_SC_rate   = 5;
   fGA_SC_factor = 0.95;
   fGA_nsteps    = 30;

   // ranges
   vector<Interval*> ranges;

   Int_t numWeights = fSynapses->GetEntriesFast();
   for (Int_t ivar=0; ivar< numWeights; ivar++) {
      ranges.push_back( new Interval( 0, GetXmax(ivar) - GetXmin(ivar) ));
   }

   FitterBase *gf = new GeneticFitter( *this, Log().GetPrintedSource(), ranges, GetOptions() );
   gf->Run();

   Double_t estimator = CalculateEstimator();
   Log() << kINFO << "GA: estimator after optimization: " << estimator << Endl;
}

//______________________________________________________________________________
Double_t TMVA::MethodMLP::EstimatorFunction( std::vector<Double_t>& parameters)
{
   // interface to the estimate
   return ComputeEstimator( parameters );
}

//______________________________________________________________________________
Double_t TMVA::MethodMLP::ComputeEstimator( std::vector<Double_t>& parameters)
{
   // this function is called by GeneticANN for GA optimization

   TSynapse* synapse;
   Int_t numSynapses = fSynapses->GetEntriesFast();

   for (Int_t i = 0; i < numSynapses; i++) {
      synapse = (TSynapse*)fSynapses->At(i);
      synapse->SetWeight(parameters.at(i));
   }

   Double_t estimator = CalculateEstimator();

   return estimator;
}

//______________________________________________________________________________
void TMVA::MethodMLP::UpdateSynapses()
{
   // update synapse error fields and adjust the weights (if in sequential mode)

   TNeuron* neuron;
   Int_t numNeurons;
   TObjArray* curLayer;
   Int_t numLayers = fNetwork->GetEntriesFast();

   for (Int_t i = 0; i < numLayers; i++) {
      curLayer = (TObjArray*)fNetwork->At(i);
      numNeurons = curLayer->GetEntriesFast();
  
      for (Int_t j = 0; j < numNeurons; j++) {
         neuron = (TNeuron*) curLayer->At(j);
         if (fBPMode == kBatch) neuron->UpdateSynapsesBatch();
         else                neuron->UpdateSynapsesSequential();
      }
   }
}

//______________________________________________________________________________
void TMVA::MethodMLP::AdjustSynapseWeights()
{
   // just adjust the synapse weights (should be called in batch mode)

   TNeuron* neuron;
   Int_t numNeurons;
   TObjArray* curLayer;
   Int_t numLayers = fNetwork->GetEntriesFast();

   for (Int_t i = numLayers-1; i >= 0; i--) {
      curLayer = (TObjArray*)fNetwork->At(i);
      numNeurons = curLayer->GetEntriesFast();
  
      for (Int_t j = 0; j < numNeurons; j++) {
         neuron = (TNeuron*) curLayer->At(j);
         neuron->AdjustSynapseWeights();
      }
   }
}

#ifdef MethodMLP_UseMinuit__

//______________________________________________________________________________
void TMVA::MethodMLP::MinuitMinimize()
{
   // minimize using Minuit
   fNumberOfWeights = fSynapses->GetEntriesFast();

   TFitter* tfitter = new TFitter( fNumberOfWeights );

   // minuit-specific settings
   Double_t args[10];

   // output level      
   args[0] = 2; // put to 0 for results only, or to -1 for no garbage
   tfitter->ExecuteCommand( "SET PRINTOUT", args, 1 );
   tfitter->ExecuteCommand( "SET NOWARNINGS", args, 0 );

   double w[54];

   // init parameters
   for (Int_t ipar=0; ipar < fNumberOfWeights; ipar++) {
      TString parName = Form("w%i", ipar);
      tfitter->SetParameter( ipar, 
                             parName, w[ipar], 0.1, 0, 0 );
   }

   // define the CFN function
   tfitter->SetFCN( &IFCN );
   
   // define fit strategy
   args[0] = 2;
   tfitter->ExecuteCommand( "SET STRATEGY", args, 1 );

   // now do the fit !
   args[0] = 1e-04;
   tfitter->ExecuteCommand( "MIGRAD", args, 1 );

   Bool_t doBetter     = kFALSE;
   Bool_t doEvenBetter = kFALSE;
   if (doBetter) {
      args[0] = 1e-04;
      tfitter->ExecuteCommand( "IMPROVE", args, 1 );

      if (doEvenBetter) {
         args[0] = 500;
         tfitter->ExecuteCommand( "MINOS", args, 1 );
      }
   }
}

_____________________________________________________________________________ 
void TMVA::MethodMLP::IFCN( Int_t& npars, Double_t* grad, Double_t &f, Double_t* fitPars, Int_t iflag )
{
   // Evaluate the minimisation function ----------------------------------------------------
   //
   //  Input parameters:
   //    npars:   number of currently variable parameters
   //             CAUTION: this is not (necessarily) the dimension of the fitPars vector !
   //    fitPars: array of (constant and variable) parameters
   //    iflag:   indicates what is to be calculated (see example below)
   //    grad:    array of gradients
   //
   //  Output parameters:
   //    f:       the calculated function value.
   //    grad:    the (optional) vector of first derivatives).
   // ---------------------------------------------------------------------------------------
   ((MethodMLP*)GetThisPtr())->FCN( npars, grad, f, fitPars, iflag );  
}

static Int_t  nc   = 0;
static double minf = 1000000;

void TMVA::MethodMLP::FCN( Int_t& npars, Double_t* grad, Double_t &f, Double_t* fitPars, Int_t iflag )
{
   // first update the weights
   for (Int_t ipar=0; ipar<fNumberOfWeights; ipar++) {
      TSynapse* synapse = (TSynapse*)fSynapses->At(ipar);
      synapse->SetWeight(fitPars[ipar]);
   }

   // now compute the estimator
   f = CalculateEstimator();

   nc++;
   if (f < minf) minf = f;
   for (Int_t ipar=0; ipar<fNumberOfWeights; ipar++) Log() << kDEBUG << fitPars[ipar] << " ";
   Log() << kDEBUG << Endl;
   Log() << kDEBUG << "***** New estimator: " << f << "  min: " << minf << " --> ncalls: " << nc << Endl;
}

#endif

//_______________________________________________________________________
void TMVA::MethodMLP::MakeClassSpecific( std::ostream& fout, const TString& className ) const
{
   // write specific classifier response
   MethodANNBase::MakeClassSpecific(fout, className);   
}

//_______________________________________________________________________
void TMVA::MethodMLP::GetHelpMessage() const
{
   // get help message text
   //
   // typical length of text line: 
   //         "|--------------------------------------------------------------|"
   TString col    = gConfig().WriteOptionsReference() ? "" : gTools().Color("bold");
   TString colres = gConfig().WriteOptionsReference() ? "" : gTools().Color("reset");

   Log() << Endl;
   Log() << col << "--- Short description:" << colres << Endl;
   Log() << Endl;
   Log() << "The MLP artificial neural network (ANN) is a traditional feed-" << Endl;
   Log() << "forward multilayer perceptron impementation. The MLP has a user-" << Endl;
   Log() << "defined hidden layer architecture, while the number of input (output)" << Endl;
   Log() << "nodes is determined by the input variables (output classes, i.e., " << Endl;
   Log() << "signal and one background). " << Endl;
   Log() << Endl;
   Log() << col << "--- Performance optimisation:" << colres << Endl;
   Log() << Endl;
   Log() << "Neural networks are stable and performing for a large variety of " << Endl;
   Log() << "linear and non-linear classification problems. However, in contrast" << Endl;
   Log() << "to (e.g.) boosted decision trees, the user is advised to reduce the " << Endl;
   Log() << "number of input variables that have only little discrimination power. " << Endl;
   Log() << "" << Endl;
   Log() << "In the tests we have carried out so far, the MLP and ROOT networks" << Endl;
   Log() << "(TMlpANN, interfaced via TMVA) performed equally well, with however" << Endl;
   Log() << "a clear speed advantage for the MLP. The Clermont-Ferrand neural " << Endl;
   Log() << "net (CFMlpANN) exhibited worse classification performance in these" << Endl;
   Log() << "tests, which is partly due to the slow convergence of its training" << Endl;
   Log() << "(at least 10k training cycles are required to achieve approximately" << Endl;
   Log() << "competitive results)." << Endl;
   Log() << Endl;
   Log() << col << "Overtraining: " << colres
         << "only the TMlpANN performs an explicit separation of the" << Endl;
   Log() << "full training sample into independent training and validation samples." << Endl;
   Log() << "We have found that in most high-energy physics applications the " << Endl;
   Log() << "avaliable degrees of freedom (training events) are sufficient to " << Endl;
   Log() << "constrain the weights of the relatively simple architectures required" << Endl;
   Log() << "to achieve good performance. Hence no overtraining should occur, and " << Endl;
   Log() << "the use of validation samples would only reduce the available training" << Endl;
   Log() << "information. However, if the perrormance on the training sample is " << Endl;
   Log() << "found to be significantly better than the one found with the inde-" << Endl;
   Log() << "pendent test sample, caution is needed. The results for these samples " << Endl;
   Log() << "are printed to standard output at the end of each training job." << Endl;
   Log() << Endl;
   Log() << col << "--- Performance tuning via configuration options:" << colres << Endl;
   Log() << Endl;
   Log() << "The hidden layer architecture for all ANNs is defined by the option" << Endl;
   Log() << "\"HiddenLayers=N+1,N,...\", where here the first hidden layer has N+1" << Endl;
   Log() << "neurons and the second N neurons (and so on), and where N is the number  " << Endl;
   Log() << "of input variables. Excessive numbers of hidden layers should be avoided," << Endl;
   Log() << "in favour of more neurons in the first hidden layer." << Endl;
   Log() << "" << Endl;
   Log() << "The number of cycles should be above 500. As said, if the number of" << Endl;
   Log() << "adjustable weights is small compared to the training sample size," << Endl;
   Log() << "using a large number of training samples should not lead to overtraining." << Endl;
}

