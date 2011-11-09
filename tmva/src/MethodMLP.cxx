// @(#)root/tmva $Id$
// Author: Krzysztof Danielowski, Andreas Hoecker, Matt Jachowski, Kamil Kraszewski, Maciej Kruk, Peter Speckmayer, Joerg Stelzer, Eckhard v. Toerne, Jan Therhaag, Jiahang Zhong

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
 *      Peter Speckmayer      <peter.speckmayer@cern.ch> - CERN, Switzerland      *
 *      Joerg Stelzer         <stelzer@cern.ch>        - DESY, Germany            *
 *      Jan Therhaag          <Jan.Therhaag@cern.ch>     - U of Bonn, Germany     *
 *      Eckhard v. Toerne     <evt@uni-bonn.de>          - U of Bonn, Germany     *
 *      Jiahang Zhong         <Jiahang.Zhong@cern.ch>  - Academia Sinica, Taipei  *
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
//
// Multilayer Perceptron class built off of MethodANNBase
//_______________________________________________________________________

#include "TString.h"
#include <vector>
#include <cmath>
#include "TTree.h"
#include "Riostream.h"
#include "TFitter.h"
#include "TMatrixD.h"
#include "TMath.h"
#include "TFile.h"

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
     fUseRegulator(false), fCalculateErrors(false),
     fPrior(0.0), fPriorDev(0), fUpdateLimit(0),
     fTrainingMethod(kBFGS), fTrainMethodS("BFGS"),
     fSamplingFraction(1.0), fSamplingEpoch(0.0), fSamplingWeight(0.0),
     fSamplingTraining(false), fSamplingTesting(false),
     fLastAlpha(0.0), fTau(0.),
     fResetStep(0), fLearnRate(0.0), fDecayRate(0.0),     
     fBPMode(kSequential), fBpModeS("None"),
     fBatchSize(0), fTestRate(0), fEpochMon(false),
     fGA_nsteps(0), fGA_preCalc(0), fGA_SC_steps(0), 
     fGA_SC_rate(0), fGA_SC_factor(0.0),
     fDeviationsFromTargets(0),
     fWeightRange     (1.0)
{
   // standard constructor
}

//______________________________________________________________________________
TMVA::MethodMLP::MethodMLP( DataSetInfo& theData,
                            const TString& theWeightFile,
                            TDirectory* theTargetDir )
   : MethodANNBase( Types::kMLP, theData, theWeightFile, theTargetDir ),
     fUseRegulator(false), fCalculateErrors(false),
     fPrior(0.0), fPriorDev(0), fUpdateLimit(0),
     fTrainingMethod(kBFGS), fTrainMethodS("BFGS"),
     fSamplingFraction(1.0), fSamplingEpoch(0.0), fSamplingWeight(0.0),
     fSamplingTraining(false), fSamplingTesting(false),
     fLastAlpha(0.0), fTau(0.),
     fResetStep(0), fLearnRate(0.0), fDecayRate(0.0),     
     fBPMode(kSequential), fBpModeS("None"),
     fBatchSize(0), fTestRate(0), fEpochMon(false),
     fGA_nsteps(0), fGA_preCalc(0), fGA_SC_steps(0), 
     fGA_SC_rate(0), fGA_SC_factor(0.0),
     fDeviationsFromTargets(0),
     fWeightRange     (1.0)
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
   if (type == Types::kMulticlass ) return kTRUE;
   if (type == Types::kRegression ) return kTRUE;

   return kFALSE;
}

//______________________________________________________________________________
void TMVA::MethodMLP::Init()
{
   // default initializations

   // the minimum requirement to declare an event signal-like
   SetSignalReferenceCut( 0.5 );
#ifdef MethodMLP_UseMinuit__
   fgThis = this;
#endif
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

   DeclareOptionRef(fLearnRate=0.02,    "LearningRate",    "ANN learning rate parameter");
   DeclareOptionRef(fDecayRate=0.01,    "DecayRate",       "Decay rate for learning parameter");
   DeclareOptionRef(fTestRate =10,      "TestRate",        "Test for overtraining performed at each #th epochs");
   DeclareOptionRef(fEpochMon = kFALSE, "EpochMonitoring", "Provide epoch-wise monitoring plots according to TestRate (caution: causes big ROOT output file!)" );

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

   DeclareOptionRef(fUseRegulator=kFALSE, "UseRegulator",
                    "Use regulator to avoid over-training");   //zjh
   DeclareOptionRef(fUpdateLimit=10000, "UpdateLimit",
		    "Maximum times of regulator update");   //zjh
   DeclareOptionRef(fCalculateErrors=kFALSE, "CalculateErrors",
                    "Calculates inverse Hessian matrix at the end of the training to be able to calculate the uncertainties of an MVA value");   //zjh

   DeclareOptionRef(fWeightRange=1.0, "WeightRange",
                    "Take the events for the estimator calculations from small deviations from the desired value to large deviations only over the weight range");

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
Double_t TMVA::MethodMLP::CalculateEstimator( Types::ETreeType treeType, Int_t iEpoch )
{
   // calculate the estimator that training is attempting to minimize

   // sanity check
   if (treeType!=Types::kTraining && treeType!=Types::kTesting) {
      Log() << kFATAL << "<CalculateEstimator> fatal error: wrong tree type: " << treeType << Endl;
   }

   Types::ETreeType saveType = Data()->GetCurrentType();
   Data()->SetCurrentType(treeType);

   // if epochs are counted create monitoring histograms (only available for classification)
   TString type  = (treeType == Types::kTraining ? "train" : "test");
   TString name  = Form("convergencetest___mlp_%s_epoch_%04i", type.Data(), iEpoch);
   TString nameB = name + "_B";
   TString nameS = name + "_S";
   Int_t   nbin  = 100;
   Float_t limit = 2;
   TH1*    histS = 0;
   TH1*    histB = 0;
   if (fEpochMon && iEpoch >= 0 && !DoRegression()) {
      histS = new TH1F( nameS, nameS, nbin, -limit, limit );
      histB = new TH1F( nameB, nameB, nbin, -limit, limit );
   }

   Double_t estimator = 0;

   // loop over all training events
   Int_t  nEvents  = GetNEvents();
   UInt_t nClasses = DataInfo().GetNClasses();
   UInt_t nTgts = DataInfo().GetNTargets();


   Float_t sumOfWeights = 0.f;
   if( fWeightRange < 1.f ){
      fDeviationsFromTargets = new std::vector<std::pair<Float_t,Float_t> >(nEvents);
   }

   for (Int_t i = 0; i < nEvents; i++) {

      const Event* ev = GetEvent(i);
      Double_t     w  = ev->GetWeight();

      ForceNetworkInputs( ev );
      ForceNetworkCalculations();

      Double_t d = 0, v = 0;
      if (DoRegression()) {
         for (UInt_t itgt = 0; itgt < nTgts; itgt++) {
            v = GetOutputNeuron( itgt )->GetActivationValue();
            Double_t targetValue = ev->GetTarget( itgt );
            Double_t dt = v - targetValue;
            d += (dt*dt);
         }
         estimator += d*w;
      } else if (DoMulticlass() ) {
         UInt_t cls = ev->GetClass();
         if (fEstimator==kCE){
            Double_t norm(0);
            for (UInt_t icls = 0; icls < nClasses; icls++) {
	       Float_t activationValue = GetOutputNeuron( icls )->GetActivationValue();
               norm += exp( activationValue );
               if(icls==cls)
                  d = exp( activationValue );
            }
            d = -TMath::Log(d/norm);
         }
         else{
            for (UInt_t icls = 0; icls < nClasses; icls++) {
               Double_t desired = (icls==cls) ? 1.0 : 0.0;
               v = GetOutputNeuron( icls )->GetActivationValue();
               d = (desired-v)*(desired-v);
            }
         }
         estimator += d*w; //zjh
      } else {
         Double_t desired =  DataInfo().IsSignal(ev)?1.:0.;
         v = GetOutputNeuron()->GetActivationValue();
         if (fEstimator==kMSE) d = (desired-v)*(desired-v);                         //zjh
         else if (fEstimator==kCE) d = -2*(desired*TMath::Log(v)+(1-desired)*TMath::Log(1-v));     //zjh
         estimator += d*w; //zjh
      }

      if( fDeviationsFromTargets )
	 fDeviationsFromTargets->push_back(std::pair<Float_t,Float_t>(d,w));

      sumOfWeights += w;


      // fill monitoring histograms
      if (DataInfo().IsSignal(ev) && histS != 0) histS->Fill( float(v), float(w) );
      else if              (histB != 0) histB->Fill( float(v), float(w) );
   }


   if( fDeviationsFromTargets ) {
      std::sort(fDeviationsFromTargets->begin(),fDeviationsFromTargets->end());

      Float_t sumOfWeightsInRange = fWeightRange*sumOfWeights;
      estimator = 0.f;

      Float_t weightRangeCut = fWeightRange*sumOfWeights;
      Float_t weightSum      = 0.f;
      for(std::vector<std::pair<Float_t,Float_t> >::iterator itDev = fDeviationsFromTargets->begin(), itDevEnd = fDeviationsFromTargets->end(); itDev != itDevEnd; ++itDev ){
	 float deviation = (*itDev).first;
	 float devWeight = (*itDev).second;
	 weightSum += devWeight; // add the weight of this event
	 if( weightSum <= weightRangeCut ) { // if within the region defined by fWeightRange
	    estimator += devWeight*deviation;
	 }
      }

      sumOfWeights = sumOfWeightsInRange;
      delete fDeviationsFromTargets;
   }

   if (histS != 0) fEpochMonHistS.push_back( histS );
   if (histB != 0) fEpochMonHistB.push_back( histB );

   //if      (DoRegression()) estimator = TMath::Sqrt(estimator/Float_t(nEvents));
   //else if (DoMulticlass()) estimator = TMath::Sqrt(estimator/Float_t(nEvents));
   //else                     estimator = estimator*0.5/Float_t(nEvents);
   if      (DoRegression()) estimator = estimator/Float_t(sumOfWeights);
   else if (DoMulticlass()) estimator = estimator/Float_t(sumOfWeights);
   else                     estimator = estimator/Float_t(sumOfWeights);


   //if (fUseRegulator) estimator+=fPrior/Float_t(nEvents);  //zjh

   Data()->SetCurrentType( saveType );

   // provide epoch-wise monitoring
   if (fEpochMon && iEpoch >= 0 && !DoRegression() && treeType == Types::kTraining) {
      CreateWeightMonitoringHists( Form("epochmonitoring___epoch_%04i_weights_hist", iEpoch), &fEpochMonHistW );
   }

   return estimator;
}

//______________________________________________________________________________
void TMVA::MethodMLP::Train(Int_t nEpochs)
{
   if (fNetwork == 0) {
      //Log() << kERROR <<"ANN Network is not initialized, doing it now!"<< Endl;
      Log() << kFATAL <<"ANN Network is not initialized, doing it now!"<< Endl;
      SetAnalysisType(GetAnalysisType());
   }
   Log() << kDEBUG << "reinitalize learning rates" << Endl;
   InitializeLearningRates();
   PrintMessage("Training Network");

   Int_t nEvents=GetNEvents();
   Int_t nSynapses=fSynapses->GetEntriesFast();
   if (nSynapses>nEvents) 
      Log()<<kWARNING<<"ANN too complicated: #events="<<nEvents<<"\t#synapses="<<nSynapses<<Endl;

#ifdef MethodMLP_UseMinuit__
   if (useMinuit) MinuitMinimize();
#else
   if (fTrainingMethod == kGA)        GeneticMinimize();
   else if (fTrainingMethod == kBFGS) BFGSMinimize(nEpochs);
   else                               BackPropagationMinimize(nEpochs);
#endif

   float trainE = CalculateEstimator( Types::kTraining, 0 ) ; // estimator for training sample  //zjh
   float testE  = CalculateEstimator( Types::kTesting,  0 ) ; // estimator for test sample //zjh
   if (fUseRegulator){
      Log()<<kINFO<<"Finalizing handling of Regulator terms, trainE="<<trainE<<" testE="<<testE<<Endl;
      UpdateRegulators();
      Log()<<kINFO<<"Done with handling of Regulator terms"<<Endl;
   }

   if( fCalculateErrors || fUseRegulator )
   {
      Int_t numSynapses=fSynapses->GetEntriesFast();
      fInvHessian.ResizeTo(numSynapses,numSynapses);
      GetApproxInvHessian( fInvHessian ,false);
   }
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
   Int_t        RegUpdateCD=0;                  //zjh
   Int_t        RegUpdateTimes=0;               //zjh
   Double_t     AccuError=0;

   Double_t trainE = -1;
   Double_t testE  = -1;

   fLastAlpha = 0.;

   if(fSamplingTraining || fSamplingTesting)
      Data()->InitSampling(1.0,1.0,fRandomSeed); // initialize sampling to initialize the random generator with the given seed

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

      //zjh
      if (fUseRegulator) {
         UpdatePriors();
         RegUpdateCD++;
      }
      //zjh

      SetGammaDelta( Gamma, Delta, buffer );

      if (i % fResetStep == 0 && i<0.5*nEpochs) { //zjh
         SteepestDir( Dir );
         Hessian.UnitMatrix();
         RegUpdateCD=0;    //zjh
      }
      else {
         if (GetHessian( Hessian, Gamma, Delta )) {
            SteepestDir( Dir );
            Hessian.UnitMatrix();
            RegUpdateCD=0;    //zjh
         }
         else SetDir( Hessian, Dir );
      }

      Double_t dError=0;  //zjh
      if (DerivDir( Dir ) > 0) {
         SteepestDir( Dir );
         Hessian.UnitMatrix();
         RegUpdateCD=0;    //zjh
      }
      if (LineSearch( Dir, buffer, &dError )) { //zjh
         Hessian.UnitMatrix();
         SteepestDir( Dir );
         RegUpdateCD=0;    //zjh
         if (LineSearch(Dir, buffer, &dError)) {  //zjh
            i = nEpochs;
            Log() << kFATAL << "Line search failed! Huge troubles somewhere..." << Endl;
         }
      }

      //zjh+
      if (dError<0) Log()<<kWARNING<<"\nnegative dError=" <<dError<<Endl;
      AccuError+=dError;

      if ( fUseRegulator && RegUpdateTimes<fUpdateLimit && RegUpdateCD>=5 && fabs(dError)<0.1*AccuError) {
         Log()<<kDEBUG<<"\n\nUpdate regulators "<<RegUpdateTimes<<" on epoch "<<i<<"\tdError="<<dError<<Endl;
         UpdateRegulators();
         Hessian.UnitMatrix();
         RegUpdateCD=0;
         RegUpdateTimes++;
         AccuError=0;
      }
      //zjh-

      // monitor convergence of training and control sample
      if ((i+1)%fTestRate == 0) {
         //trainE = CalculateEstimator( Types::kTraining, i ) - fPrior/Float_t(GetNEvents()); // estimator for training sample  //zjh
         //testE  = CalculateEstimator( Types::kTesting,  i ) - fPrior/Float_t(GetNEvents()); // estimator for test sample //zjh
         trainE = CalculateEstimator( Types::kTraining, i ) ; // estimator for training sample  //zjh
         testE  = CalculateEstimator( Types::kTesting,  i ) ; // estimator for test sample //zjh
         fEstimatorHistTrain->Fill( i+1, trainE );
         fEstimatorHistTest ->Fill( i+1, testE );

         Bool_t success = kFALSE;
         if ((testE < GetCurrentValue()) || (GetCurrentValue()<1e-100)) {
            success = kTRUE;
         }
         Data()->EventResult( success );

         SetCurrentValue( testE );
         if (HasConverged()) {
            if (Float_t(i)/nEpochs < fSamplingEpoch) {
               Int_t newEpoch = Int_t(fSamplingEpoch*nEpochs);
               i = newEpoch;
               ResetConvergenceCounter();
            }
            else break;
         }
      }

      // draw progress
      TString convText = Form( "<D^2> (train/test/epoch): %.4g/%.4g/%d", trainE, testE,i  ); //zjh
      if (fSteps > 0) {
         Float_t progress = 0;
         if (Float_t(i)/nEpochs < fSamplingEpoch)
//            progress = Progress()*fSamplingEpoch*fSamplingFraction*100;
            progress = Progress()*fSamplingFraction*100*fSamplingEpoch;
         else
	 {
//            progress = 100.0*(fSamplingEpoch*fSamplingFraction+(1.0-fSamplingFraction*fSamplingEpoch)*Progress());
            progress = 100.0*(fSamplingFraction*fSamplingEpoch+(1.0-fSamplingEpoch)*Progress());
	 }
         Float_t progress2= 100.0*RegUpdateTimes/fUpdateLimit; //zjh
         if (progress2>progress) progress=progress2; //zjh
         timer.DrawProgressBar( Int_t(progress), convText );
      }
      else {
         Int_t progress=Int_t(nEpochs*RegUpdateTimes/Float_t(fUpdateLimit)); //zjh
         if (progress<i) progress=i; //zjh
         timer.DrawProgressBar( progress, convText ); //zjh
      }

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
      Double_t DEDw=synapse->GetDEDw();     //zjh
      if (fUseRegulator) DEDw+=fPriorDev[i]; //zjh
      synapse->SetDEDw( DEDw / nEvents );   //zjh
   }
}

//______________________________________________________________________________
void TMVA::MethodMLP::SimulateEvent( const Event* ev )
{
   Double_t eventWeight = ev->GetWeight();

   ForceNetworkInputs( ev );
   ForceNetworkCalculations();

   if (DoRegression()) {
      UInt_t ntgt = DataInfo().GetNTargets();
      for (UInt_t itgt = 0; itgt < ntgt; itgt++) {
         Double_t desired     = ev->GetTarget(itgt);
         Double_t error = ( GetOutputNeuron( itgt )->GetActivationValue() - desired )*eventWeight;
         GetOutputNeuron( itgt )->SetError(error);
      }
   } else if (DoMulticlass()) {
      UInt_t nClasses = DataInfo().GetNClasses();
      UInt_t cls      = ev->GetClass();
      for (UInt_t icls = 0; icls < nClasses; icls++) {
         Double_t desired  = ( cls==icls ? 1.0 : 0.0 );
         Double_t error    = ( GetOutputNeuron( icls )->GetActivationValue() - desired )*eventWeight;
         GetOutputNeuron( icls )->SetError(error);
      }
   } else {
      Double_t desired     = GetDesiredOutput( ev );
      Double_t error=-1;				//zjh
      if (fEstimator==kMSE) error = ( GetOutputNeuron()->GetActivationValue() - desired )*eventWeight;       //zjh
      else if (fEstimator==kCE) error = -eventWeight/(GetOutputNeuron()->GetActivationValue() -1 + desired);  //zjh
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
Bool_t TMVA::MethodMLP::LineSearch(TMatrixD &Dir, std::vector<Double_t> &buffer, Double_t* dError)
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
   Double_t errOrigin=err1;  	//zjh
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
         if (i==50) {
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
         Log() << kWARNING << "linesearch, failed even in opposite direction of steepestDIR" << Endl;
         fLastAlpha = 0.05;
         return kTRUE;
      }
   }

   if (alpha1>0 && alpha2>0 && alpha3 > 0) {
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

   if (dError) (*dError)=(errOrigin-finalError)/finalError; //zjh

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
   if (fUseRegulator) UpdatePriors();	//zjh
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
      if (DoRegression()) {
         for (UInt_t itgt = 0; itgt < ntgts; itgt++) {
            error += GetMSEErr( ev, itgt );	//zjh
         }
      } else if ( DoMulticlass() ){
         for( UInt_t icls = 0, iclsEnd = DataInfo().GetNClasses(); icls < iclsEnd; icls++ ){
            error += GetMSEErr( ev, icls );
         }
      } else {
         if (fEstimator==kMSE) error = GetMSEErr( ev );  //zjh
         else if (fEstimator==kCE) error= GetCEErr( ev ); //zjh
      }
      Result += error * ev->GetWeight();
   }
   if (fUseRegulator) Result+=fPrior;  //zjh
   if (Result<0) Log()<<kWARNING<<"\nNegative Error!!! :"<<Result-fPrior<<"+"<<fPrior<<Endl;
   return Result;
}

//______________________________________________________________________________
Double_t TMVA::MethodMLP::GetMSEErr( const Event* ev, UInt_t index )
{
   Double_t error = 0;
   Double_t output = GetOutputNeuron( index )->GetActivationValue();
   Double_t target = 0;
   if      (DoRegression()) target = ev->GetTarget( index );
   else if (DoMulticlass()) target = (ev->GetClass() == index ? 1.0 : 0.0 );
   else                     target = GetDesiredOutput( ev );

   error = 0.5*(output-target)*(output-target); //zjh

   return error;

}

//______________________________________________________________________________
Double_t TMVA::MethodMLP::GetCEErr( const Event* ev, UInt_t index )  //zjh
{
   Double_t error = 0;
   Double_t output = GetOutputNeuron( index )->GetActivationValue();
   Double_t target = 0;
   if      (DoRegression()) target = ev->GetTarget( index );
   else if (DoMulticlass()) target = (ev->GetClass() == index ? 1.0 : 0.0 );
   else                     target = GetDesiredOutput( ev );

   error = -(target*TMath::Log(output)+(1-target)*TMath::Log(1-output));

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

   if(fSamplingTraining || fSamplingTesting)
      Data()->InitSampling(1.0,1.0,fRandomSeed); // initialize sampling to initialize the random generator with the given seed

   if (fSteps > 0) Log() << kINFO << "Inaccurate progress timing for MLP... " << Endl;
   timer.DrawProgressBar(0);

   // estimators
   Double_t trainE = -1;
   Double_t testE  = -1;

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

      TrainOneEpoch();
      DecaySynapseWeights(i >= lateEpoch);

      // monitor convergence of training and control sample
      if ((i+1)%fTestRate == 0) {
         trainE = CalculateEstimator( Types::kTraining, i ); // estimator for training sample
         testE  = CalculateEstimator( Types::kTesting,  i );  // estimator for test samplea
         fEstimatorHistTrain->Fill( i+1, trainE );
         fEstimatorHistTest ->Fill( i+1, testE );

         Bool_t success = kFALSE;
         if ((testE < GetCurrentValue()) || (GetCurrentValue()<1e-100)) {
            success = kTRUE;
         }
         Data()->EventResult( success );

         SetCurrentValue( testE );
         if (HasConverged()) {
            if (Float_t(i)/nEpochs < fSamplingEpoch) {
               Int_t newEpoch = Int_t(fSamplingEpoch*nEpochs);
               i = newEpoch;
               ResetConvergenceCounter();
            }
            else {
               if (lateEpoch > i) lateEpoch = i;
               else                break;
            }
         }
      }

      // draw progress bar (add convergence value)
      TString convText = Form( "<D^2> (train/test): %.4g/%.4g", trainE, testE );
      if (fSteps > 0) {
         Float_t progress = 0;
         if (Float_t(i)/nEpochs < fSamplingEpoch)
            progress = Progress()*fSamplingEpoch*fSamplingFraction*100;
         else
            progress = 100*(fSamplingEpoch*fSamplingFraction+(1.0-fSamplingFraction*fSamplingEpoch)*Progress());

         timer.DrawProgressBar( Int_t(progress), convText );
      }
      else {
        timer.DrawProgressBar( i, convText );
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
      if (lateEpoch) synapse->DecayLearningRate(TMath::Sqrt(fDecayRate)); // In order to lower the learning rate even more, we need to apply sqrt instead of square.
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
   if (type == 0) desired = fOutput->GetMin();  // background //zjh
   else           desired = fOutput->GetMax();  // signal     //zjh

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
   if (DoRegression()) UpdateNetwork( ev->GetTargets(),       eventWeight );
   if (DoMulticlass()) UpdateNetwork( *DataInfo().GetTargetsForMulticlass( ev ), eventWeight );
   else                UpdateNetwork( GetDesiredOutput( ev ), eventWeight );
}

//______________________________________________________________________________
Double_t TMVA::MethodMLP::GetDesiredOutput( const Event* ev )
{
   // get the desired output of this event
   return DataInfo().IsSignal(ev)?fOutput->GetMax():fOutput->GetMin(); //zjh
}


//______________________________________________________________________________
void TMVA::MethodMLP::UpdateNetwork(Double_t desired, Double_t eventWeight)
{
   // update the network based on how closely
   // the output matched the desired output
   Double_t error = GetOutputNeuron()->GetActivationValue() - desired;
   if (fEstimator==kMSE)  error = GetOutputNeuron()->GetActivationValue() - desired ;  //zjh
   else if (fEstimator==kCE)  error = -1./(GetOutputNeuron()->GetActivationValue() -1 + desired); //zjh
   else  Log() << kFATAL << "Estimator type unspecified!!" << Endl;              //zjh
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
   for (UInt_t i = 0, iEnd = desired.size(); i < iEnd; ++i) {
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
   if (fUseRegulator) UpdatePriors(); //zjh

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

//_______________________________________________________________________
void TMVA::MethodMLP::UpdatePriors()  //zjh
{
   fPrior=0;
   fPriorDev.clear();
   Int_t nSynapses = fSynapses->GetEntriesFast();
   for (Int_t i=0;i<nSynapses;i++) {
      TSynapse* synapse = (TSynapse*)fSynapses->At(i);
      fPrior+=0.5*fRegulators[fRegulatorIdx[i]]*(synapse->GetWeight())*(synapse->GetWeight());
      fPriorDev.push_back(fRegulators[fRegulatorIdx[i]]*(synapse->GetWeight()));
   }
}

//_______________________________________________________________________
void TMVA::MethodMLP::UpdateRegulators()  //zjh
{
   TMatrixD InvH(0,0);
   GetApproxInvHessian(InvH);
   Int_t numSynapses=fSynapses->GetEntriesFast();
   Int_t numRegulators=fRegulators.size();
   Float_t gamma=0,
      variance=1.;    // Gaussian noise
   vector<Int_t> nWDP(numRegulators);
   vector<Double_t> trace(numRegulators),weightSum(numRegulators);
   for (int i=0;i<numSynapses;i++) {
      TSynapse* synapses = (TSynapse*)fSynapses->At(i);
      Int_t idx=fRegulatorIdx[i];
      nWDP[idx]++;
      trace[idx]+=InvH[i][i];
      gamma+=1-fRegulators[idx]*InvH[i][i];
      weightSum[idx]+=(synapses->GetWeight())*(synapses->GetWeight());
   }
   if (fEstimator==kMSE) {
      if (GetNEvents()>gamma) variance=CalculateEstimator( Types::kTraining, 0 )/(1-(gamma/GetNEvents()));
      else variance=CalculateEstimator( Types::kTraining, 0 );
   }

   //Log() << kDEBUG << Endl;
   for (int i=0;i<numRegulators;i++)
      {
         //fRegulators[i]=variance*(nWDP[i]-fRegulators[i]*trace[i])/weightSum[i];
         fRegulators[i]=variance*nWDP[i]/(weightSum[i]+variance*trace[i]);
         if (fRegulators[i]<0) fRegulators[i]=0;
         Log()<<kDEBUG<<"R"<<i<<":"<<fRegulators[i]<<"\t";
      }
   float trainE = CalculateEstimator( Types::kTraining, 0 ) ; // estimator for training sample  //zjh
   float testE  = CalculateEstimator( Types::kTesting,  0 ) ; // estimator for test sample //zjh

   Log()<<kDEBUG<<"\n"<<"trainE:"<<trainE<<"\ttestE:"<<testE<<"\tvariance:"<<variance<<"\tgamma:"<<gamma<<Endl;

}

//_______________________________________________________________________
void TMVA::MethodMLP::GetApproxInvHessian(TMatrixD& InvHessian, bool regulate)  //zjh
{
   Int_t numSynapses=fSynapses->GetEntriesFast();
   InvHessian.ResizeTo( numSynapses, numSynapses );
   InvHessian=0;
   TMatrixD sens(numSynapses,1);
   TMatrixD sensT(1,numSynapses);
   Int_t nEvents = GetNEvents();
   for (Int_t i=0;i<nEvents;i++) {
      GetEvent(i);
      double outputValue=GetMvaValue(); // force calculation
      GetOutputNeuron()->SetError(1./fOutput->EvalDerivative(GetOutputNeuron()->GetValue()));
      CalculateNeuronDeltas();
      for (Int_t j = 0; j < numSynapses; j++){
         TSynapse* synapses = (TSynapse*)fSynapses->At(j);
         synapses->InitDelta();
         synapses->CalculateDelta();
         sens[j][0]=sensT[0][j]=synapses->GetDelta();
      }
      if (fEstimator==kMSE ) InvHessian+=sens*sensT;
      else if (fEstimator==kCE) InvHessian+=(outputValue*(1-outputValue))*sens*sensT;
   }

   // TVectorD eValue(numSynapses);
   if (regulate) {
      for (Int_t i = 0; i < numSynapses; i++){
         InvHessian[i][i]+=fRegulators[fRegulatorIdx[i]];
      }
   }
   else {
      for (Int_t i = 0; i < numSynapses; i++){
         InvHessian[i][i]+=1e-6; //to avoid precision problem that will destroy the pos-def
      }
   }

   InvHessian.Invert();

}

//_______________________________________________________________________
Double_t TMVA::MethodMLP::GetMvaValue( Double_t* errLower, Double_t* errUpper )
{
  Double_t MvaValue = MethodANNBase::GetMvaValue();// contains back propagation

   // no hessian (old training file) or no error reqested
   if (!fCalculateErrors || errLower==0 || errUpper==0)
      return MvaValue;

   Double_t MvaUpper,MvaLower,median,variance;
   Int_t numSynapses=fSynapses->GetEntriesFast();
   if (fInvHessian.GetNcols()!=numSynapses) {
      Log() << kWARNING << "inconsistent dimension " << fInvHessian.GetNcols() << " vs " << numSynapses << Endl;
   }
   TMatrixD sens(numSynapses,1);
   TMatrixD sensT(1,numSynapses);
   GetOutputNeuron()->SetError(1./fOutput->EvalDerivative(GetOutputNeuron()->GetValue()));
   //GetOutputNeuron()->SetError(1.);
   CalculateNeuronDeltas();
   for (Int_t i = 0; i < numSynapses; i++){
      TSynapse* synapses = (TSynapse*)fSynapses->At(i);
      synapses->InitDelta();
      synapses->CalculateDelta();
      sensT[0][i]=synapses->GetDelta();
   }
   sens.Transpose(sensT);
   TMatrixD sig=sensT*fInvHessian*sens;
   variance=sig[0][0];
   median=GetOutputNeuron()->GetValue();

   if (variance<0) {
     Log()<<kWARNING<<"Negative variance!!! median=" << median << "\tvariance(sigma^2)=" << variance <<Endl;
     variance=0;
   }
   variance=sqrt(variance);
 
   //upper
   MvaUpper=fOutput->Eval(median+variance);
   if(errUpper)
      *errUpper=MvaUpper-MvaValue;

   //lower
   MvaLower=fOutput->Eval(median-variance);
   if(errLower)
      *errLower=MvaValue-MvaLower;

   return MvaValue;
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

//_______________________________________________________________________
TMVA::MethodMLP* TMVA::MethodMLP::GetThisPtr()
{
   // global "this" pointer to be used in minuit
   return fgThis;
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
   TString col    = gConfig().WriteOptionsReference() ? TString() : gTools().Color("bold");
   TString colres = gConfig().WriteOptionsReference() ? TString() : gTools().Color("reset");

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

