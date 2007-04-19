// @(#)root/tmva $Id: MethodANNBase.cxx,v 1.11 2007/01/30 11:24:16 brun Exp $
// Author: Andreas Hoecker, Matt Jachowski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodANNBase                                                         *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Artificial neural network base class for the discrimination of signal     *
 *      from background.                                                          *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker  <Andreas.Hocker@cern.ch> - CERN, Switzerland             *
 *      Matt Jachowski   <jachowski@stanford.edu> - Stanford University, USA      *
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
// Base class for all TMVA methods using artificial neural networks      
//                                                                      
//_______________________________________________________________________

#include "TString.h"
#include <vector>
#include "TTree.h"
#include "TDirectory.h"
#include "Riostream.h"
#include "TRandom3.h"
#include "TH2F.h"

#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#endif
#ifndef ROOT_TMVA_MethodANNBase
#include "TMVA/MethodANNBase.h"
#endif
#ifndef ROOT_TMVA_TNeuron
#include "TMVA/TNeuron.h"
#endif
#ifndef ROOT_TMVA_TSynapse
#include "TMVA/TSynapse.h"
#endif
#ifndef ROOT_TMVA_TActivationChooser
#include "TMVA/TActivationChooser.h"
#endif
#ifndef ROOT_TMVA_Types
#include "TMVA/Types.h"
#endif
#ifndef ROOT_TMVA_Tools
#include "TMVA/Tools.h"
#endif
#ifndef ROOT_TMVA_TNeuronInputChooser
#include "TMVA/TNeuronInputChooser.h"
#endif
#ifndef ROOT_TMVA_Ranking
#include "TMVA/Ranking.h"
#endif

using std::vector;

ClassImp(TMVA::MethodANNBase)

//______________________________________________________________________________
TMVA::MethodANNBase::MethodANNBase( TString jobName, TString methodTitle, DataSet& theData, 
                                    TString theOption, TDirectory* theTargetDir )
   : TMVA::MethodBase( jobName, methodTitle, theData, theOption, theTargetDir )
{
   // standard constructor
   // Note: Right now it is an option to choose the neuron input function,
   // but only the input function "sum" leads to weight convergence --
   // otherwise the weights go to nan and lead to an ABORT.
   

   InitANNBase();

   DeclareOptions();
}

//______________________________________________________________________________
TMVA::MethodANNBase::MethodANNBase( DataSet & theData, 
                                    TString theWeightFile, TDirectory* theTargetDir )
   : TMVA::MethodBase( theData, theWeightFile, theTargetDir ) 
{
   // construct the Method from the weight file 
   InitANNBase();

   DeclareOptions();
}


//______________________________________________________________________________
void TMVA::MethodANNBase::DeclareOptions()
{
   // define the options (their key words) that can be set in the option string 
   // here the options valid for ALL MVA methods are declared.
   // know options: NCycles=xx              :the number of training cycles
   //               Normalize=kTRUE,kFALSe  :if normalised in put variables should be used
   //               HiddenLayser="N-1,N-2"  :the specification of the hidden layers
   //               NeuronType=sigmoid,tanh,radial,linar  : the type of activation function
   //                                                       used at the neuronn
   //                


   DeclareOptionRef(fNcycles=3000,"NCycles","Number of training cycles");
   DeclareOptionRef(fNormalize=kTRUE, "Normalize", "Normalize input variables");
   DeclareOptionRef(fLayerSpec="N-1,N-2","HiddenLayers","Specification of the hidden layers");
   DeclareOptionRef(fNeuronType="sigmoid",
                    "NeuronType","Neuron activation function type");
   TActivationChooser aChooser;
   vector<TString>* names = aChooser.GetAllActivationNames();
   Int_t nTypes = names->size();
   for (Int_t i = 0; i < nTypes; i++)
      AddPreDefVal(names->at(i));
   delete names;

   DeclareOptionRef(fNeuronInputType="sum", 
                    "NeuronInputType","Neuron input function type");
   TNeuronInputChooser iChooser;
   names = iChooser.GetAllNeuronInputNames();
   nTypes = names->size();
   for (Int_t i = 0; i < nTypes; i++)
      AddPreDefVal(names->at(i));
   delete names;
}

//______________________________________________________________________________
void TMVA::MethodANNBase::ProcessOptions()
{
   // decode the options in the option string

   MethodBase::ProcessOptions();

   vector<Int_t>* layout = ParseLayoutString(fLayerSpec);
   BuildNetwork(layout);
}

//______________________________________________________________________________
vector<Int_t>* TMVA::MethodANNBase::ParseLayoutString(TString layerSpec)
{
   // parse layout specification string and return a vector, each entry
   // containing the number of neurons to go in each successive layer
   vector<Int_t>* layout = new vector<Int_t>();
   layout->push_back(GetNvar());
   while(layerSpec.Length()>0) {
      TString sToAdd="";
      if(layerSpec.First(',')<0) {
         sToAdd = layerSpec;
         layerSpec = "";
      } else {
         sToAdd = layerSpec(0,layerSpec.First(','));
         layerSpec = layerSpec(layerSpec.First(',')+1,layerSpec.Length());
      }
      int nNodes = 0;
      if(sToAdd.BeginsWith("n") || sToAdd.BeginsWith("N")) { sToAdd.Remove(0,1); nNodes = GetNvar(); }
      nNodes += atoi(sToAdd);
      layout->push_back(nNodes);
   }
   layout->push_back(1);  // one output node
  
   return layout;
}

//______________________________________________________________________________
void TMVA::MethodANNBase::InitANNBase()
{
   // initialize ANNBase object
   fNetwork         = NULL;
   frgen            = NULL;
   fActivation      = NULL;
   fIdentity        = NULL;
   fInputCalculator = NULL;
   fSynapses        = NULL;
   fEstimatorHistTrain = NULL;  
   fEstimatorHistTest  = NULL;  

   // these will be set in BuildNetwork()
   fInputLayer = NULL;
   fOutputNeuron = NULL;

   if (fgFIXED_SEED) frgen = new TRandom3(1);   // fix output for debugging
   else              frgen = new TRandom3(0);   // seed = 0 means random seed

   fSynapses = new TObjArray();
}

//______________________________________________________________________________
TMVA::MethodANNBase::~MethodANNBase()
{
   // destructor
   DeleteNetwork();
}

//______________________________________________________________________________
void TMVA::MethodANNBase::DeleteNetwork()
{
   // delete/clear network
   if (fNetwork != NULL) {
      TObjArray *layer;
      Int_t numLayers = fNetwork->GetEntriesFast();
      for (Int_t i = 0; i < numLayers; i++) {
         layer = (TObjArray*)fNetwork->At(i);
         DeleteNetworkLayer(layer);
      }
    
      delete fNetwork;
   }

   if (frgen != NULL)            delete frgen;
   if (fActivation != NULL)      delete fActivation;
   if (fIdentity != NULL)        delete fIdentity;
   if (fInputCalculator != NULL) delete fInputCalculator;
   if (fSynapses != NULL)        delete fSynapses;

   fNetwork         = NULL;
   frgen            = NULL;
   fActivation      = NULL;
   fIdentity        = NULL;
   fInputCalculator = NULL;
   fSynapses        = NULL;
}

//______________________________________________________________________________
void TMVA::MethodANNBase::DeleteNetworkLayer(TObjArray*& layer)
{
   // delete a network layer
   TNeuron* neuron;
   Int_t numNeurons = layer->GetEntriesFast();
   for (Int_t i = 0; i < numNeurons; i++) {
      neuron = (TNeuron*)layer->At(i);
      neuron->DeletePreLinks();
      delete neuron;
   }
   delete layer;
}

//______________________________________________________________________________
void TMVA::MethodANNBase::BuildNetwork(vector<Int_t>* layout, vector<Double_t>* weights)
{
   // build network given a layout (number of neurons in each layer)
   // and optional weights array

   fLogger << kINFO << "Building Network" << Endl;

   DeleteNetwork();
   InitANNBase();

   // set activation and input functions
   TActivationChooser aChooser;
   fActivation = aChooser.CreateActivation(fNeuronType);
   fIdentity   = aChooser.CreateActivation("linear");
   TNeuronInputChooser iChooser;
   fInputCalculator = iChooser.CreateNeuronInput(fNeuronInputType);

   fNetwork = new TObjArray();
   BuildLayers(layout);

   // cache input layer and output neuron for fast access
   fInputLayer   = (TObjArray*)fNetwork->At(0);
   TObjArray* outputLayer = (TObjArray*)fNetwork->At(fNetwork->GetEntriesFast()-1);
   fOutputNeuron = (TNeuron*)outputLayer->At(0);

   if (weights == NULL) InitWeights();
   else                 ForceWeights(weights);
}

//______________________________________________________________________________
void TMVA::MethodANNBase::BuildLayers(vector<Int_t>* layout)
{
   // build the network layers

   TObjArray* curLayer;
   TObjArray* prevLayer = NULL;

   Int_t numLayers = layout->size();

   for (Int_t i = 0; i < numLayers; i++) {
      curLayer = new TObjArray();
      BuildLayer(layout->at(i), curLayer, prevLayer, i, numLayers);
      prevLayer = curLayer;
      fNetwork->Add(curLayer);
   }

   // cache pointers to synapses for fast access, the order matters
   for (Int_t i = 0; i < numLayers; i++) {                                       
      TObjArray* layer = (TObjArray*)fNetwork->At(i);                             
      Int_t numNeurons = layer->GetEntriesFast();                                 
      for (Int_t j = 0; j < numNeurons; j++) {                                    
         TNeuron* neuron = (TNeuron*)layer->At(j);                                 
         Int_t numSynapses = neuron->NumPostLinks();                               
         for (Int_t k = 0; k < numSynapses; k++) {                                 
            TSynapse* synapse = neuron->PostLinkAt(k);                              
            fSynapses->Add(synapse);                                   
         }                                                                         
      }                                                                           
   }  
}

//______________________________________________________________________________
void TMVA::MethodANNBase::BuildLayer(Int_t numNeurons, TObjArray* curLayer, 
                                     TObjArray* prevLayer, Int_t layerIndex, 
                                     Int_t numLayers)
{
   // build a single layer with neurons and synapses connecting this
   // layer to the previous layer

   TNeuron* neuron;
   for (Int_t j = 0; j < numNeurons; j++) {
      neuron = new TNeuron();
      neuron->SetInputCalculator(fInputCalculator);

      // input layer
      if (layerIndex == 0) {
         neuron->SetActivationEqn(fIdentity);
         neuron->SetInputNeuron();
      }
      else {
         // output layer
         if (layerIndex == numLayers-1) {
            neuron->SetOutputNeuron();
            neuron->SetActivationEqn(fIdentity);
         }
         // hidden layers
         else neuron->SetActivationEqn(fActivation);

         AddPreLinks(neuron, prevLayer);
      }

      curLayer->Add(neuron);
   }

   // add bias neutron (except to output layer)
   if (layerIndex != numLayers-1) {
      neuron = new TNeuron();
      neuron->SetActivationEqn(fIdentity);
      neuron->SetBiasNeuron();
      neuron->ForceValue(1.0);
      curLayer->Add(neuron);
   }
}

//______________________________________________________________________________
void TMVA::MethodANNBase::AddPreLinks(TNeuron* neuron, TObjArray* prevLayer)
{
   // add synapses connecting a neuron to its preceding layer

   TSynapse* synapse;
   int numNeurons = prevLayer->GetEntriesFast();
   TNeuron* preNeuron;

   for (Int_t i = 0; i < numNeurons; i++) {
      preNeuron = (TNeuron*)prevLayer->At(i);
      synapse = new TSynapse();
      synapse->SetPreNeuron(preNeuron);
      synapse->SetPostNeuron(neuron);
      preNeuron->AddPostLink(synapse);
      neuron->AddPreLink(synapse);
   }
}

//______________________________________________________________________________
void TMVA::MethodANNBase::InitWeights()
{
   // initialize the synapse weights randomly
   PrintMessage("initializing weights");
   
   // init synapse weights
   Int_t numSynapses = fSynapses->GetEntriesFast();
   TSynapse* synapse;
   for (Int_t i = 0; i < numSynapses; i++) {
      synapse = (TSynapse*)fSynapses->At(i);
      synapse->SetWeight(4.0*frgen->Rndm() - 2.0);
   }
}

//_______________________________________________________________________
void TMVA::MethodANNBase::ForceWeights(vector<Double_t>* weights)
{
   // force the synapse weights
   PrintMessage("forcing weights");

   Int_t numSynapses = fSynapses->GetEntriesFast();
   TSynapse* synapse;
   for (Int_t i = 0; i < numSynapses; i++) {
      synapse = (TSynapse*)fSynapses->At(i);
      synapse->SetWeight(weights->at(i));
   }
}

//______________________________________________________________________________
void TMVA::MethodANNBase::ForceNetworkInputs(Int_t ignoreIndex)
{
   // force the input values of the input neurons
   // force the value for each input neuron

   Double_t x;
   TNeuron* neuron;

   for (Int_t j = 0; j < GetNvar(); j++) {
      if (j == ignoreIndex) x = 0;
      else x = (fNormalize) ? GetEventValNormalized(j) : GetEvent().GetVal(j);

      neuron = GetInputNeuron(j);
      neuron->ForceValue(x);
   }
}

//______________________________________________________________________________
void TMVA::MethodANNBase::ForceNetworkCalculations()
{
   // calculate input values to each neuron

   TObjArray* curLayer;
   TNeuron* neuron;
   Int_t numLayers = fNetwork->GetEntriesFast();
   Int_t numNeurons;

   for (Int_t i = 0; i < numLayers; i++) {
      curLayer = (TObjArray*)fNetwork->At(i);
      numNeurons = curLayer->GetEntriesFast();

      for (Int_t j = 0; j < numNeurons; j++) {
         neuron = (TNeuron*) curLayer->At(j);
         neuron->CalculateValue();
         neuron->CalculateActivationValue();
      }
   }
}

//______________________________________________________________________________
void TMVA::MethodANNBase::PrintMessage(TString message, Bool_t force) const
{
   // print messages, turn off printing by setting verbose and debug flag appropriately
   if (Verbose() || Debug() || force) fLogger << kINFO << message << Endl;
}

//______________________________________________________________________________
void TMVA::MethodANNBase::WaitForKeyboard()
{
   // wait for keyboard input, for debugging
   string dummy;
   fLogger << kINFO << "***Type anything to continue (q to quit): ";
   getline(cin, dummy);
   if (dummy == "q" || dummy == "Q") {
      PrintMessage( "quit" );
      delete this;
      exit(0);
   }
}

//______________________________________________________________________________
void TMVA::MethodANNBase::PrintNetwork()
{
   // print network representation, for debugging

   if (!Debug()) return;

   fLogger << Endl;
   PrintMessage( "printing network " );
   fLogger << kINFO << "-------------------------------------------------------------------" << Endl;

   TObjArray* curLayer;
   Int_t numLayers = fNetwork->GetEntriesFast();

   for (Int_t i = 0; i < numLayers; i++) {

      curLayer = (TObjArray*)fNetwork->At(i);
      Int_t numNeurons = curLayer->GetEntriesFast();

      fLogger << kINFO << "Layer #" << i << " (" << numNeurons << " neurons):" << Endl;
      PrintLayer( curLayer );
   }
}

//______________________________________________________________________________
void TMVA::MethodANNBase::PrintLayer(TObjArray* layer)
{
   // print a single layer, for debugging

   Int_t numNeurons = layer->GetEntriesFast();
   TNeuron* neuron;
  
   for (Int_t j = 0; j < numNeurons; j++) {
      neuron = (TNeuron*) layer->At(j);
      fLogger << kINFO << "\tNeuron #" << j << " (LinksIn: " << neuron->NumPreLinks() 
              << " , LinksOut: " << neuron->NumPostLinks() << ")" << Endl;
      PrintNeuron( neuron );
   }
}

//______________________________________________________________________________
void TMVA::MethodANNBase::PrintNeuron(TNeuron* neuron)
{
   // print a neuron, for debugging
   fLogger << kINFO 
           << "\t\tValue:\t"     << neuron->GetValue()
           << "\t\tActivation: " << neuron->GetActivationValue()
           << "\t\tDelta: "      << neuron->GetDelta() << Endl;
   fLogger << kINFO << "\t\tActivationEquation:\t";
   neuron->PrintActivationEqn();
   fLogger << kINFO << "\t\tLinksIn:" << Endl;
   neuron->PrintPreLinks();
   fLogger << kINFO << "\t\tLinksOut:" << Endl;
   neuron->PrintPostLinks();
}

//_______________________________________________________________________
Double_t TMVA::MethodANNBase::GetMvaValue()
{
   // get the mva value generated by the NN
   TNeuron* neuron;

   TObjArray* inputLayer = (TObjArray*)fNetwork->At(0);

   for (Int_t i = 0; i < GetNvar(); i++) {
      Double_t x = fNormalize?GetEventValNormalized(i):GetEvent().GetVal(i);
      neuron = (TNeuron*)inputLayer->At(i);
      neuron->ForceValue(x);
   }

   ForceNetworkCalculations();

   // check the output of the network
   TObjArray* outputLayer = (TObjArray*)fNetwork->At( fNetwork->GetEntriesFast()-1 );
   neuron = (TNeuron*)outputLayer->At(0);

   return neuron->GetActivationValue();
}

//_______________________________________________________________________
void TMVA::MethodANNBase::WriteWeightsToStream( ostream & o) const
{
   // write the weights stream
   Int_t numLayers = fNetwork->GetEntriesFast();
   o << "Weights" << endl;
   for (Int_t i = 0; i < numLayers; i++) {
      TObjArray* layer = (TObjArray*)fNetwork->At(i);
      Int_t numNeurons = layer->GetEntriesFast();
      for (Int_t j = 0; j < numNeurons; j++) {
         TNeuron* neuron = (TNeuron*)layer->At(j);
         Int_t numSynapses = neuron->NumPostLinks();
         for (Int_t k = 0; k < numSynapses; k++) {
            TSynapse* synapse = neuron->PostLinkAt(k);
            o << "(layer" << i << ",neuron" << j << ")-(layer" 
              << i+1 << ",neuron" << k << "): " 
              << synapse->GetWeight() << endl;
         }
      }
   }
}

//_______________________________________________________________________
void TMVA::MethodANNBase::ReadWeightsFromStream( istream & istr)
{
   // destroy/clear the network then read it back in from the weights file

   // delete network so we can reconstruct network from scratch

   TString dummy;

   // synapse weights
   Double_t weight;
   vector<Double_t>* weights = new vector<Double_t>();
   istr>> dummy;
   while (istr>> dummy >> weight) weights->push_back(weight); // use w/ slower write-out

   ForceWeights(weights);
   
   delete weights;
}

//_______________________________________________________________________
const TMVA::Ranking* TMVA::MethodANNBase::CreateRanking()
{
   // compute ranking of input variables by summing function of weights

   // create the ranking object
   fRanking = new Ranking( GetName(), "Importance" );

   TNeuron*  neuron;
   TSynapse* synapse;
   Double_t  importance, avgVal;
   TString varName;

   for (Int_t i = 0; i < GetNvar(); i++) {

      neuron = GetInputNeuron(i);
      Int_t numSynapses = neuron->NumPostLinks();
      importance = 0;
      varName = GetInputVar(i); // fix this line

      // figure out average value of variable i
      Double_t meanS, meanB, rmsS, rmsB, xmin, xmax;
      Statistics( TMVA::Types::kTraining, varName, 
                  meanS, meanB, rmsS, rmsB, xmin, xmax );

      avgVal = (meanS + meanB) / 2.0; // change this into a real weighted average
      if (Normalize()) avgVal = 0.5*(1 + Norm(i, avgVal));

      for (Int_t j = 0; j < numSynapses; j++) {
         synapse = neuron->PostLinkAt(j);
         importance += synapse->GetWeight() * synapse->GetWeight();
      }
      
      importance *= avgVal * avgVal;

      fRanking->AddRank( *new Rank( varName, importance ) );
   }

   return fRanking;
}

//_______________________________________________________________________
void TMVA::MethodANNBase::WriteMonitoringHistosToFile() const
{
   // write histograms to file
   PrintMessage(Form("write special histos to file: %s", BaseDir()->GetPath()), kTRUE);

   //   BaseDir()->mkdir(GetName()+GetMethodName())->cd(); 

   TH2F*      hist;
   Int_t numLayers = fNetwork->GetEntriesFast();

   if (fEstimatorHistTrain) fEstimatorHistTrain->Write();
   if (fEstimatorHistTest ) fEstimatorHistTest ->Write();

   for (Int_t i = 0; i < numLayers-1; i++) {
      
      TObjArray* layer1 = (TObjArray*)fNetwork->At(i);
      TObjArray* layer2 = (TObjArray*)fNetwork->At(i+1);
      Int_t numNeurons1 = layer1->GetEntriesFast();
      Int_t numNeurons2 = layer2->GetEntriesFast();
      
      TString name = Form("weights_hist%i%i", i, i+1);
      hist = new TH2F(name + "", name + "", 
                      numNeurons1, 0, numNeurons1, numNeurons2, 0, numNeurons2);
      
      for (Int_t j = 0; j < numNeurons1; j++) {
         
         TNeuron* neuron = (TNeuron*)layer1->At(j);
         Int_t numSynapses = neuron->NumPostLinks();

         for (Int_t k = 0; k < numSynapses; k++) {
            
            TSynapse* synapse = neuron->PostLinkAt(k);
            hist->SetBinContent(j+1, k+1, synapse->GetWeight());
            
         }
      }
      
      hist->Write();
      delete hist;
   }
}
