// Author: Christophe.Delaere@cern.ch   25/04/04

class TTree;
class TNeuron;
class TSynapse;
class TMultiLayerPerceptron;

//____________________________________________________________________
//
// TMLPAnalyzer
//
// This utility class contains a set of tests usefull when developing 
// a neural network.
// It allows you to check for unneeded variables, and to control 
// the network structure.
//
//--------------------------------------------------------------------

class TMLPAnalyzer : public TObject {
 public:
   TMLPAnalyzer(TMultiLayerPerceptron& net) { network = &net; analysisTree=0; }
   TMLPAnalyzer(TMultiLayerPerceptron* net) { network = net;  analysisTree=0; }
   virtual ~TMLPAnalyzer(); 
   void DrawNetwork(Int_t neuron, const char* signal, const char* bg);
   void DrawDInput(Int_t i);
   void DrawDInputs();
   void CheckNetwork();
   void GatherInformations();
 protected:
   Int_t GetLayers();
   Int_t GetNeurons(Int_t layer);
   TString GetNeuronFormula(Int_t idx);
 
 private:
   TMultiLayerPerceptron* network;
   TTree* analysisTree;

 ClassDef(TMLPAnalyzer, 0)
};

