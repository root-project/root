// @(#)root/mlp:$Name:  $:$Id: TSynapse.h,v 1.2 2003/12/16 14:09:38 brun Exp $
// Author: Christophe.Delaere@cern.ch   25/04/04

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TObject
#include "TObject.h"
#endif

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

private:
   TMultiLayerPerceptron *fNetwork;
   TTree                 *fAnalysisTree;

protected:
   Int_t GetLayers();
   Int_t GetNeurons(Int_t layer);
   TString GetNeuronFormula(Int_t idx);

public:
   TMLPAnalyzer(TMultiLayerPerceptron& net) { fNetwork = &net; fAnalysisTree=0; }
   TMLPAnalyzer(TMultiLayerPerceptron* net) { fNetwork = net;  fAnalysisTree=0; }
   virtual ~TMLPAnalyzer();
   void DrawNetwork(Int_t neuron, const char* signal, const char* bg);
   void DrawDInput(Int_t i);
   void DrawDInputs();
   void CheckNetwork();
   void GatherInformations();

   ClassDef(TMLPAnalyzer, 0)
};

