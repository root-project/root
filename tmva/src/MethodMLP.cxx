// @(#)root/tmva $Id: MethodMLP.cxx,v 1.7 2007/04/19 06:53:02 brun Exp $
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
#include "TFitter.h"

#include "TMVA/Interval.h"
#include "TMVA/MethodMLP.h"
#include "TMVA/TNeuron.h"
#include "TMVA/TSynapse.h"
#include "TMVA/Timer.h"
#include "TMVA/Types.h"
#include "TMVA/Tools.h"
#include "TMVA/GeneticFitter.h"

#ifdef MethodMLP_UseMinuit__
TMVA::MethodMLP* TMVA::MethodMLP::fgThis = 0;
Bool_t MethodMLP_UseMinuit = kTRUE;
#endif

using std::vector;

ClassImp(TMVA::MethodMLP)

//______________________________________________________________________________
TMVA::MethodMLP::MethodMLP( TString jobName, TString methodTitle, DataSet& theData, 
                            TString theOption, TDirectory* theTargetDir )
   : MethodANNBase( jobName, methodTitle, theData, theOption, theTargetDir )
{
   // standard constructor
   InitMLP();

   // interpretation of configuration option string
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
   : MethodANNBase( theData, theWeightFile, theTargetDir ) 
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
   SetMethodType( Types::kMLP );
   SetTestvarName();

   // the minimum requirement to declare an event signal-like
   SetSignalReferenceCut( 0.0 );
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
                    "Train with Back-Propagation (BP - default) or Genetic Algorithm (GA - slower and worse)");
   AddPreDefVal(TString("BP"));
   AddPreDefVal(TString("GA"));

   DeclareOptionRef(fLearnRate=0.02, "LearningRate", "ANN learning rate parameter");
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
   // process user options
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
Double_t TMVA::MethodMLP::CalculateEstimator( Types::ETreeType treeType )
{
   // calculate the estimator that training is attempting to minimize

   Int_t nEvents = ( (treeType == Types::kTesting ) ? Data().GetNEvtTest() :
                     (treeType == Types::kTraining) ? Data().GetNEvtTrain() : -1 );

   // sanity check
   if (nEvents <=0) 
      fLogger << kFATAL << "<CalculateEstimator> fatal error: wrong tree type: " << treeType << Endl;

   Double_t estimator = 0;

   // loop over all training events 
   for (Int_t i = 0; i < nEvents; i++) {

      if (treeType == Types::kTesting )
         ReadTestEvent(i);
      else
         ReadTrainingEvent(i);
      
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
   Timer timer( nEpochs, GetName() );

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

   Timer timer( nEpochs, GetName() );
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
         Double_t trainE = CalculateEstimator( Types::kTraining ); // estimator for training sample
         Double_t testE  = CalculateEstimator( Types::kTesting  );  // estimator for test samplea
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
      if (IsNormalised()) x = Norm(j, x);
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
   Double_t eventWeight = GetEventWeight();
   Double_t desired     = GetDesiredOutput();
   ForceNetworkInputs();
   ForceNetworkCalculations();
   UpdateNetwork( desired, eventWeight );
}

//______________________________________________________________________________
Double_t TMVA::MethodMLP::GetDesiredOutput()
{
   // get the desired output of this event
   Double_t desired;
   if (IsSignalEvent()) desired = fActivation->GetMax(); // signal
   else                 desired = fActivation->GetMin();

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
   fGA_preCalc   = 1;
   fGA_SC_steps  = 10;
   fGA_SC_rate   = 5;
   fGA_SC_factor = 0.95;
   fGA_nsteps    = 30;

   // ranges
   vector<Interval*> ranges;

   Int_t numWeights = fSynapses->GetEntriesFast();
   for (Int_t ivar=0; ivar< numWeights; ivar++) {
      ranges.push_back( new Interval( -3.0, 3.0 ) );
      //ranges.push_back( new Interval( 0, GetXmax(ivar) - GetXmin(ivar) ));
   }

   FitterBase *gf = new GeneticFitter( *this, fLogger.GetPrintedSource(), ranges, GetOptions() );
   gf->Run();

   Double_t estimator = CalculateEstimator();
   fLogger << kINFO << "GA: estimator after optimization: " << estimator << Endl;
}

//______________________________________________________________________________
Double_t TMVA::MethodMLP::EstimatorFunction( std::vector<Double_t>& parameters)
{
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

//_______________________________________________________________________
void TMVA::MethodMLP::MakeClassSpecific( std::ostream& fout, const TString& className ) const
{
   MethodANNBase::MakeClassSpecific(fout, className);   
}

//_______________________________________________________________________
void TMVA::MethodMLP::GetHelpMessage() const
{
   // get help message text
   //
   // typical length of text line: 
   //         "|--------------------------------------------------------------|"
   fLogger << Endl;
   fLogger << Tools::Color("bold") << "--- Short description:" << Tools::Color("reset") << Endl;
   fLogger << Endl;
   fLogger << "The MLP artificial neural network (ANN) is a traditional feed-" << Endl;
   fLogger << "forward multilayer perceptron impementation. The MLP has a user-" << Endl;
   fLogger << "defined hidden layer architecture, while the number of input (output)" << Endl;
   fLogger << "nodes is determined by the input variables (output classes, i.e., " << Endl;
   fLogger << "signal and one background). " << Endl;
   fLogger << Endl;
   fLogger << Tools::Color("bold") << "--- Performance optimisation:" << Tools::Color("reset") << Endl;
   fLogger << Endl;
   fLogger << "Neural networks are stable and performing for a large variety of " << Endl;
   fLogger << "linear and non-linear classification problems. However, in contrast" << Endl;
   fLogger << "to (e.g.) boosted decision trees, the user is advised to reduce the " << Endl;
   fLogger << "number of input variables that have only little discrimination power. " << Endl;
   fLogger << "" << Endl;
   fLogger << "In the tests we have carried out so far, the MLP and ROOT networks" << Endl;
   fLogger << "(TMlpANN, interfaced via TMVA) performed equally well, with however" << Endl;
   fLogger << "a clear speed advantage for the MLP. The Clermont-Ferrand neural " << Endl;
   fLogger << "net (CFMlpANN) exhibited worse classification performance in these" << Endl;
   fLogger << "tests, which is partly due to the slow convergence of its training" << Endl;
   fLogger << "(at least 10k training cycles are required to achieve approximately" << Endl;
   fLogger << "competitive results)." << Endl;
   fLogger << Endl;
   fLogger << Tools::Color("bold") << "Overtraining: " << Tools::Color("reset")
           << "only the TMlpANN performs an explicit separation of the" << Endl;
   fLogger << "full training sample into independent training and validation samples." << Endl;
   fLogger << "We have found that in most high-energy physics applications the " << Endl;
   fLogger << "avaliable degrees of freedom (training events) are sufficient to " << Endl;
   fLogger << "constrain the weights of the relatively simple architectures required" << Endl;
   fLogger << "to achieve good performance. Hence no overtraining should occur, and " << Endl;
   fLogger << "the use of validation samples would only reduce the available training" << Endl;
   fLogger << "information. However, if the perrormance on the training sample is " << Endl;
   fLogger << "found to be significantly better than the one found with the inde-" << Endl;
   fLogger << "pendent test sample, caution is needed. The results for these samples " << Endl;
   fLogger << "are printed to standard output at the end of each training job." << Endl;
   fLogger << Endl;
   fLogger << Tools::Color("bold") << "--- Performance tuning via configuration options:" << Tools::Color("reset") << Endl;
   fLogger << Endl;
   fLogger << "The hidden layer architecture for all ANNs is defined by the option" << Endl;
   fLogger << "\"HiddenLayers=N+1,N,...\", where here the first hidden layer has N+1" << Endl;
   fLogger << "neurons and the second N neurons (and so on), and where N is the number  " << Endl;
   fLogger << "of input variables. Excessive numbers of hidden layers should be avoided," << Endl;
   fLogger << "in favour of more neurons in the first hidden layer." << Endl;
   fLogger << "" << Endl;
   fLogger << "The number of cycles should be above 500. As said, if the number of" << Endl;
   fLogger << "adjustable weights is small compared to the training sample size," << Endl;
   fLogger << "using a large number of training samples should not lead to overtraining." << Endl;
}
