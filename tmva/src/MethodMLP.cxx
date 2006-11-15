// @(#)root/tmva $Id: MethodMLP.cxx,v 1.29 2006/11/14 23:02:57 stelzer Exp $
// Author: Andreas Hoecker, Matt Jachowski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodMLP                                                             *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      ANN Multilayer Perceptron class for the discrimination of signal          *
 *      from background.                                                          *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker  <Andreas.Hocker@cern.ch> - CERN, Switzerland             *
 *      Matt Jachowski   <jachowski@stanford.edu> - Stanford University, USA      *
 *                                                                               *
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
#include "TRandom3.h"
#include "TFitter.h"

#ifndef ROOT_TMVA_MethodMLP
#include "TMVA/MethodMLP.h"
#endif
#ifndef ROOT_TMVA_TNeuron
#include "TMVA/TNeuron.h"
#endif
#ifndef ROOT_TMVA_TSynapse
#include "TMVA/TSynapse.h"
#endif
#ifndef ROOT_TMVA_Timer
#include "TMVA/Timer.h"
#endif
#ifndef ROOT_TMVA_Types
#include "TMVA/Types.h"
#endif
#ifndef ROOT_TMVA_Tools
#include "TMVA/Tools.h"
#endif
#ifndef ROOT_TMVA_GeneticANN
#include "TMVA/GeneticANN.h"
#endif

#ifdef MethodMLP_UseMinuit__
TMVA::MethodMLP* TMVA::MethodMLP::fgThis = 0;
Bool_t MethodMLP_UseMinuit = kTRUE;
#endif

using std::vector;

ClassImp(TMVA::MethodMLP)
   ;

//______________________________________________________________________________
TMVA::MethodMLP::MethodMLP( TString jobName, TString methodTitle, DataSet& theData, 
                            TString theOption, TDirectory* theTargetDir )
   : TMVA::MethodANNBase( jobName, methodTitle, theData, theOption, theTargetDir )
{
   // standard constructor

   InitMLP();

   DeclareOptions();

   ParseOptions();

   ProcessOptions();

   InitializeLearningRates();

   if (fBPMode == kBatch) {
      Int_t numEvents = Data().GetNEvtTrain();
      if (fBatchSize < 1 || fBatchSize > numEvents) fBatchSize = numEvents;
   }

}

//______________________________________________________________________________
TMVA::MethodMLP::MethodMLP( DataSet& theData, TString theWeightFile, TDirectory* theTargetDir )
   : TMVA::MethodANNBase( theData, theWeightFile, theTargetDir ) 
{
   // construct from a weight file -- most work is done by MethodANNBase constructor
   InitMLP();

   DeclareOptions();
}

//______________________________________________________________________________
TMVA::MethodMLP::~MethodMLP()
{
   // destructor
   // nothing to be done
}

//______________________________________________________________________________
void TMVA::MethodMLP::InitMLP()
{
   // default initializations
   SetMethodName( "MLP" );
   SetMethodType( TMVA::Types::MLP );
   SetTestvarName();
}

//_______________________________________________________________________
void TMVA::MethodMLP::DeclareOptions() 
{
   DeclareOptionRef(fTrainMethodS="BP", "TrainingMethod", 
                    "Train with Back-Propagation (BP) or Genetic Algorithm (GA) (takes a LONG time)");
   AddPreDefVal(TString("BP"));
   AddPreDefVal(TString("GA"));

   DeclareOptionRef(fLearnRate=0.02, "LearningRate", "NN learning rate parameter");
   DeclareOptionRef(fDecayRate=0.01, "DecayRate",    "Decay rate for learning parameter");
   DeclareOptionRef(fTestRate =10,   "TestRate",     "Test for overtraining performed at each #th epochs");

   DeclareOptionRef(fBpModeS="sequential", "BPMode", 
                    "Back-propagation learning mode: sequential or batch");
   AddPreDefVal(TString("sequential"));
   AddPreDefVal(TString("batch"));

   DeclareOptionRef(fBatchSize=-1, "BatchSize", 
                    "Batch size: number of events/batch, only set if in Batch Mode, -1 for BatchSize=number_of_events");
}

//_______________________________________________________________________
void TMVA::MethodMLP::ProcessOptions() 
{
   MethodANNBase::ProcessOptions();

   if      (fTrainMethodS == "BP") fTrainingMethod = kBP;
   else if (fTrainMethodS == "GA") fTrainingMethod = kGA;

   if (fBpModeS == "sequential") fBPMode = kSequential;
   else if (fBpModeS == "batch") fBPMode = kBatch;
}

//______________________________________________________________________________
void TMVA::MethodMLP::InitializeLearningRates()
{

   // initialize learning rates of synapses, used only by backpropagation
   TSynapse *synapse;
   Int_t numSynapses = fSynapses->GetEntriesFast();
   for (Int_t i = 0; i < numSynapses; i++) {
      synapse = (TSynapse*)fSynapses->At(i);
      synapse->SetLearningRate(fLearnRate);
   }
}

//______________________________________________________________________________ 
Double_t TMVA::MethodMLP::CalculateEstimator( TMVA::Types::TreeType treeType )
{
   // calculate the estimator that training is attempting to minimize

   Int_t nEvents = ( (treeType == TMVA::Types::kTest ) ? Data().GetNEvtTest() :
                     (treeType == TMVA::Types::kTrain) ? Data().GetNEvtTrain() : -1 );

   // sanity check
   if (nEvents <=0) 
      fLogger << kFATAL << "<CalculateEstimator> fatal error: wrong tree type: " << treeType << Endl;

   Double_t estimator = 0;

   // loop over all training events 
   for (Int_t i = 0; i < nEvents; i++) {

      if (treeType == TMVA::Types::kTest ) ReadTestEvent(i);
      else                                 ReadTrainingEvent(i);
      
      Double_t desired = GetDesiredOutput();
      ForceNetworkInputs();
      ForceNetworkCalculations();

      Double_t d = GetOutputNeuron()->GetActivationValue() - desired;
      estimator += (d*d);
   }

   estimator = estimator*0.5/Float_t(nEvents);
   return estimator;
}

//______________________________________________________________________________
void TMVA::MethodMLP::Train(Int_t nEpochs)
{
   // train the network
   PrintMessage("Training Network");
   TMVA::Timer timer( nEpochs, GetName() );

#ifdef MethodMLP_UseMinuit__  
   if (useMinuit) MinuitMinimize();
#else
   if (fTrainingMethod == kGA) GeneticMinimize();
   else                        BackPropagationMinimize(nEpochs);
#endif

   PrintMessage("Train: elapsed time: " + timer.GetElapsedTime() +  "                      ", kTRUE);   
}

//______________________________________________________________________________
void TMVA::MethodMLP::BackPropagationMinimize(Int_t nEpochs)
{
   // minimize estimator / train network with backpropagation algorithm

   TMVA::Timer timer( nEpochs, GetName() );
   Int_t lateEpoch = (Int_t)(nEpochs*0.95) - 1;

   // create histograms for overtraining monitoring
   Int_t nbinTest = Int_t(nEpochs/fTestRate);
   fEstimatorHistTrain = new TH1F( "estimatorHistTrain", "training estimator", 
                                   nbinTest, Int_t(fTestRate/2), nbinTest*fTestRate+Int_t(fTestRate/2) );
   fEstimatorHistTest  = new TH1F( "estimatorHistTest", "test estimator", 
                                   nbinTest, Int_t(fTestRate/2), nbinTest*fTestRate+Int_t(fTestRate/2) );

   // start training cycles (epochs)
   for (Int_t i = 0; i < nEpochs; i++) {

      timer.DrawProgressBar(i);
      TrainOneEpoch();
      DecaySynapseWeights(i >= lateEpoch);

      // monitor convergence of training and control sample
      if ((i+1)%fTestRate == 0) {
         Double_t trainE = CalculateEstimator( TMVA::Types::kTrain ); // estimator for training sample
         Double_t testE  = CalculateEstimator( TMVA::Types::kTest  );  // estimator for test samplea
         fEstimatorHistTrain->Fill( i+1, trainE );
         fEstimatorHistTest ->Fill( i+1, testE );
      }
   }
}

//______________________________________________________________________________
void TMVA::MethodMLP::TrainOneEpoch()
{     
   // train network over a single epoch/cyle of events

   Int_t nEvents = Data().GetNEvtTrain();
     
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

   ReadTrainingEvent(ievt);

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
   
   for (Int_t j = 0; j < GetNvar(); j++) {
      x = branchVar[j];
      if (Normalize()) x = Norm(j, x);
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

   ReadTrainingEvent(ievt);
   Double_t eventWeight = Data().Event().GetWeight();
   Double_t desired     = GetDesiredOutput();
   ForceNetworkInputs();
   ForceNetworkCalculations();
   UpdateNetwork(desired, eventWeight);
}

//______________________________________________________________________________
Double_t TMVA::MethodMLP::GetDesiredOutput()
{
   // get the desired output of this event
   Double_t desired;
   if (Data().Event().IsSignal()) desired = fActivation->GetMax(); // signal
   else                           desired = fActivation->GetMin();

   return desired;
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
void TMVA::MethodMLP::CalculateNeuronDeltas()
{
   // have each neuron calculate its delta by backpropagation

   TNeuron* neuron;
   Int_t    numNeurons;
   TObjArray* curLayer;
   Int_t numLayers = fNetwork->GetEntriesFast();

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
   fGA_preCalc        = 1;
   fGA_SC_steps       = 10;
   fGA_SC_offsteps    = 5;
   fGA_SC_factor      = 0.95;
   fGA_nsteps         = 30;

   // ranges
   vector<LowHigh_t*> ranges;

   Int_t numWeights = fSynapses->GetEntriesFast();
   for (Int_t ivar=0; ivar< numWeights; ivar++) {
      ranges.push_back( new LowHigh_t( -3.0, 3.0 ) );
      //ranges.push_back( new LowHigh_t( 0, GetXmax(ivar) - GetXmin(ivar) ));
   }

   GeneticANN *bestResultsStore = new GeneticANN( 0, ranges, this ); 
   GeneticANN *bestResults      = new GeneticANN( 0, ranges, this );

   Timer timer1( fGA_preCalc, GetName() ); 

   // precalculation
   for (Int_t preCalc = 0; preCalc < fGA_preCalc; preCalc++) {
    
      //timer1.DrawProgressBar(preCalc);

      // ---- perform series of fits to achieve best convergence

      GeneticANN ga( ranges.size() * 10, ranges, this ); 

      ga.GetGeneticPopulation().AddPopulation( &bestResults->GetGeneticPopulation() );
      ga.CalculateFitness();
      ga.GetGeneticPopulation().TrimPopulation();

      while (true) {
         ga.Init();
         ga.CalculateFitness();
         ga.SpreadControl( fGA_SC_steps, fGA_SC_offsteps, fGA_SC_factor );
         if (ga.HasConverged( Int_t(fGA_nsteps*0.67), 0.0001 )) break;
      }
      
      bestResultsStore->GetGeneticPopulation().GiveHint( ga.GetGeneticPopulation().GetGenes( 0 )->GetFactors() );
    
      delete bestResults;
      bestResults = bestResultsStore;
      bestResultsStore = new GeneticANN( 0, ranges, this );
   }

   Double_t estimator = CalculateEstimator();

   bestResults->Init();

   // main run
   fLogger << kINFO << "GA: starting main course                                  "  << Endl;

   vector<Double_t> par(2*GetNvar());
  
   // ---- perform series of fits to achieve best convergence

   TMVA::GeneticANN ga( ranges.size() * 10, ranges, this );
   ga.SetSpread( 0.1 );
   ga.GetGeneticPopulation().AddPopulation( &bestResults->GetGeneticPopulation() );
   ga.CalculateFitness();
   ga.GetGeneticPopulation().TrimPopulation();

   while(true) {
      ga.Init();
      ga.CalculateFitness();
      ga.SpreadControl( fGA_SC_steps, fGA_SC_offsteps, fGA_SC_factor );
      if (ga.HasConverged( fGA_nsteps, 0.00001 )) break;
   }

   Int_t n = 0;
   vector< Double_t >::iterator vec = ga.GetGeneticPopulation().GetGenes( 0 )->GetFactors().begin();
   for (; vec < ga.GetGeneticPopulation().GetGenes( 0 )->GetFactors().end(); vec++ ) {
      par[n] = (*vec);
      n++;
   }

   // get elapsed time
   fLogger << kINFO << "GA: elapsed time: " << timer1.GetElapsedTime() << Endl;    

   estimator = CalculateEstimator();
   fLogger << kINFO << "GA: stimator after optimization: " << estimator << Endl;
}

//______________________________________________________________________________
Double_t TMVA::MethodMLP::ComputeEstimator(const vector<Double_t>& parameters)
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
   fNumberOfWeights = fSynapses->GetEntriesFast();

   TFitter* tfitter = new TFitter( fNumberOfWeights );

   double w[54];

   // init parameters
   for (Int_t ipar=0; ipar < fNumberOfWeights; ipar++) {
      TString parName = Form("w%i", ipar);
      tfitter->SetParameter( ipar, 
                             parName, w[ipar], 0.1, 0, 0 );
   }

   // define the CFN function
   tfitter->SetFCN( &IFCN );

   // minuit-specific settings
   Double_t args[10];

   // output level      
   args[0] = 2; // put to 0 for results only, or to -1 for no garbage
   tfitter->ExecuteCommand( "SET PRINTOUT", args, 1 );
   tfitter->ExecuteCommand( "SET NOWARNINGS", args, 0 );
   
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
   for (Int_t ipar=0; ipar<fNumberOfWeights; ipar++) fLogger << kVERBOSE << fitPars[ipar] << " ";
   fLogger << kVERBOSE << Endl;
   fLogger << kVERBOSE << "***** new estimator: " << f << "  min: " << minf << " --> ncalls: " << nc << Endl;
}

#endif
