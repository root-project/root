// @(#)root/mlp:$Id$
// Author: Christophe.Delaere@cern.ch   25/04/04

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMLPAnalyzer
#define ROOT_TMLPAnalyzer


#include "TObject.h"

class TTree;
class TNeuron;
class TSynapse;
class TMultiLayerPerceptron;
class TProfile;
class THStack;

class TMLPAnalyzer : public TObject {

private:
   TMultiLayerPerceptron *fNetwork;
   TTree                 *fAnalysisTree;
   TTree                 *fIOTree;

protected:
   Int_t GetLayers();
   Int_t GetNeurons(Int_t layer);
   TString GetNeuronFormula(Int_t idx);
   const char* GetInputNeuronTitle(Int_t in);
   const char* GetOutputNeuronTitle(Int_t out);

public:
   TMLPAnalyzer(TMultiLayerPerceptron& net):
      fNetwork(&net), fAnalysisTree(0), fIOTree(0) {}
   TMLPAnalyzer(TMultiLayerPerceptron* net):
      fNetwork(net), fAnalysisTree(0), fIOTree(0) {}
   ~TMLPAnalyzer() override;
   void DrawNetwork(Int_t neuron, const char* signal, const char* bg);
   void DrawDInput(Int_t i);
   void DrawDInputs();
   TProfile* DrawTruthDeviation(Int_t outnode=0, Option_t *option="");
   THStack* DrawTruthDeviations(Option_t *option="");
   TProfile* DrawTruthDeviationInOut(Int_t innode, Int_t outnode=0,
                                     Option_t *option="");
   THStack* DrawTruthDeviationInsOut(Int_t outnode=0, Option_t *option="");

   void CheckNetwork();
   void GatherInformations();
   TTree* GetIOTree() const { return fIOTree;}

   ClassDefOverride(TMLPAnalyzer, 0) // A simple analysis class for MLP
};

#endif
