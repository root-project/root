// @(#)root/mlp:$Name:  $:$Id: TMultiLayerPerceptron.h,v 1.00 2003/08/27 13:52:36 brun Exp $
// Author: Christophe.Delaere@cern.ch   20/07/03

#ifndef ROOT_TMultiLayerPerceptron
#define ROOT_TMultiLayerPerceptron

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif
#ifndef ROOT_TMatrix
#include "TMatrix.h"
#endif

class TTree;
class TEventList;

//____________________________________________________________________
//
// TMultiLayerPerceptron
//
// This class decribes a Neural network.
// There are facilities to train the network and use the output.
// 
// The input layer is made of inactive neurons (returning the 
// normalized input), hidden layers are made of sigmoids and output 
// neurons are linear.
//
// The basic input is a TTree and two (training and test) TEventLists.
// For classification jobs, a branch (maybe in a TFriend) must contain 
// the expected output.
// 6 learning methods are available: kStochastic, kBatch, 
// kSteepestDescent, kRibierePolak, kFletcherReeves and kBFGS.
//
// This implementation is *inspired* from the mlpfit package from 
// J.Schwindling et al.
//
//____________________________________________________________________

class TMultiLayerPerceptron : public TObject {
 public:
   enum LearningMethod { kStochastic, kBatch, kSteepestDescent,
                         kRibierePolak, kFletcherReeves, kBFGS };
   enum DataSet { kTraining, kTest };
   TMultiLayerPerceptron();
   TMultiLayerPerceptron(TString layout, TTree* data = NULL, 
                         TEventList* training = NULL, 
                         TEventList* test = NULL);
   virtual ~ TMultiLayerPerceptron() {} 
   void SetData(TTree*);
   void SetTrainingDataSet(TEventList* train);
   void SetTestDataSet(TEventList* test);
   void SetLearningMethod(TMultiLayerPerceptron::LearningMethod method); 
   void SetEventWeight(TString);
   void Train(Int_t nEpoch, Option_t* option = "text");
   Double_t Result(Int_t event, Int_t index = 0);
   Double_t GetError(Int_t event);
   Double_t GetError(TMultiLayerPerceptron::DataSet set);
   void ComputeDEDw();
   void Randomize();
   void SetEta(Double_t eta);
   void SetEpsilon(Double_t eps);
   void SetDelta(Double_t delta);
   void SetEtaDecay(Double_t ed); 
   void SetTau(Double_t tau);
   void SetReset(Int_t reset); 
   inline Double_t GetEta()      { return fEta; }
   inline Double_t GetEpsilon()  { return fEpsilon; }
   inline Double_t GetDelta()    { return fDelta; }
   inline Double_t GetEtaDecay() { return fEtaDecay; }
   inline Double_t GetTau()      { return fTau; }
   inline Int_t GetReset()       { return fReset; }
   void DrawResult(Int_t index = 0, Option_t* option = "");
   void DumpWeights(Option_t* filename = "-");
   void LoadWeights(Option_t* filename = "");
   Double_t Evaluate(Int_t index, ...);
   void Export(Option_t* filename = "NNfunction", Option_t* language = "C++");
   
 protected:
   void BuildNetwork();
   void GetEntry(Int_t);
   void MLP_Stochastic(Double_t*);
   void MLP_Batch(Double_t*);
   Bool_t LineSearch(Double_t*, Double_t*);
   void SteepestDir(Double_t*);
   void ConjugateGradientsDir(Double_t*, Double_t);
   void SetGammaDelta(TMatrix&, TMatrix&, Double_t*);
   bool GetBFGSH(TMatrix&, TMatrix &, TMatrix&);
   void BFGSDir(TMatrix&, Double_t*);
   Double_t DerivDir(Double_t*);
   
 private:
   void BuildFirstLayer(TString&);
   void BuildHiddenLayers(TString&);
   void BuildLastLayer(TString&, Int_t);
   void Shuffle(Int_t*, Int_t);
   void MLP_Line(Double_t*, Double_t*, Double_t);
   
   TTree* fData;                   //! buffer holding a pointer to the tree used as datasource
   TObjArray fNetwork;             // Collection of all the neurons in the network
   TObjArray fFirstLayer;          // Collection of the input neurons. This is a subset of fNetwork
   TObjArray fLastLayer;           // Collection of the output neurons. This is a subset of fNetwork
   TObjArray fSynapses;            // Collection of all the synapses in the network
   TString fStructure;             // String containing the network structure. This is the constructor first argument
   TEventList *fTraining;          //! EventList defining the events in the training dataset
   TEventList *fTest;              //! EventList defining the events in the test dataset
   LearningMethod fLearningMethod; //! The Learning Method
   Double_t fEventWeight;          //! buffer containing the event weight
   Double_t fEta;                  //! Eta - used in stochastic minimisation - Default=0.1
   Double_t fEpsilon;              //! Epsilon - used in stochastic minimisation - Default=0.
   Double_t fDelta;                //! Delta - used in stochastic minimisation - Default=0.
   Double_t fEtaDecay;             //! EtaDecay - Eta *= EtaDecay at each epoch - Default=1.
   Double_t fTau;                  //! Tau - used in line search - Default=3.
   Double_t fLastAlpha;            //! internal parameter used in line search
   Int_t fReset;                   //! number of epochs between two resets of the search direction to the steepest descent. Default=50
   ClassDef(TMultiLayerPerceptron, 1)	// a Neural Network
};

#endif
