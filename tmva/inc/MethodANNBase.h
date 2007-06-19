// @(#)root/tmva $Id: MethodANNBase.h,v 1.10 2007/04/19 06:53:01 brun Exp $
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
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>   - CERN, Switzerland             *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodANNBase
#define ROOT_TMVA_MethodANNBase

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodANNBase                                                        //
//                                                                      //
// Base class for all TMVA methods using artificial neural networks     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TString.h"
#include <vector>
#include "TTree.h"
#include "TObjArray.h"
#include "TRandom3.h"

#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#endif
#ifndef ROOT_TMVA_TActivation
#include "TMVA/TActivation.h"
#endif
#ifndef ROOT_TMVA_TNeuron
#include "TMVA/TNeuron.h"
#endif
#ifndef ROOT_TMVA_TNeuronInput
#include "TMVA/TNeuronInput.h"
#endif

namespace TMVA {

   class MethodANNBase : public MethodBase {
      
   public:
      
      // constructors dictated by subclassing off of MethodBase
      MethodANNBase( TString jobName, TString methodTitle, DataSet& theData, 
                     TString theOption, TDirectory* theTargetDir );
      
      MethodANNBase( DataSet& theData, TString theWeightFile, 
                     TDirectory* theTargetDir );
      
      virtual ~MethodANNBase();
      
      // this does the real initialization work
      void InitANNBase();
      
      // setters for subclasses
      void SetActivation(TActivation* activation) { 
         if (fActivation != NULL) delete fActivation; fActivation = activation; 
      }
      void SetNeuronInputCalculator(TNeuronInput* inputCalculator) { 
         if (fInputCalculator != NULL) delete fInputCalculator; 
         fInputCalculator = inputCalculator; 
      }
      
      // this will have to be overridden by every subclass
      virtual void Train() = 0;
      
      // print network, for debugging
      virtual void PrintNetwork();
      
      using MethodBase::WriteWeightsToStream;
      using MethodBase::ReadWeightsFromStream;

      // write weights to file
      virtual void WriteWeightsToStream( ostream& o ) const;

      // read weights from file
      virtual void ReadWeightsFromStream( istream& istr );
      
      // calculate the MVA value
      virtual Double_t GetMvaValue();
      
      // write method specific histos to target file
      virtual void WriteMonitoringHistosToFile() const;
     
      // ranking of input variables
      const Ranking* CreateRanking();

      // the option handling methods
      virtual void DeclareOptions();
      virtual void ProcessOptions();
      
      Bool_t Debug() const { return fgDEBUG; }
      
   protected:

      virtual void MakeClassSpecific( std::ostream&, const TString& ) const;
      
      vector<Int_t>* ParseLayoutString(TString layerSpec);
      virtual void BuildNetwork(vector<Int_t>* layout, vector<Double_t>* weights=NULL);
      void     ForceNetworkInputs(Int_t ignoreIndex=-1);
      Double_t GetNetworkOutput() { return GetOutputNeuron()->GetActivationValue(); }
      
      // debugging utilities
      void PrintMessage(TString message, Bool_t force=kFALSE) const;
      void ForceNetworkCalculations();
      void WaitForKeyboard();
      
      // accessors
      Int_t    NumCycles()  { return fNcycles;   }
      TNeuron* GetInputNeuron(Int_t index) { return (TNeuron*)fInputLayer->At(index); }
      TNeuron* GetOutputNeuron()           { return fOutputNeuron; }
      
      // protected variables
      TObjArray*    fNetwork;     // TObjArray of TObjArrays representing network
      TObjArray*    fSynapses;    // array of pointers to synapses, no structural data
      TActivation*  fActivation;  // activation function to be used for hidden layers
      TActivation*  fIdentity;    // activation for input and output layers
      TRandom3*     frgen;        // random number generator for various uses
      TNeuronInput* fInputCalculator; // input calculator for all neurons

      // monitoring histograms
      TH1F* fEstimatorHistTrain; // monitors convergence of training sample
      TH1F* fEstimatorHistTest;  // monitors convergence of independent test sample
      
   private:
      
      // helper functions for building network
      void BuildLayers(std::vector<Int_t>* layout);
      void BuildLayer(Int_t numNeurons, TObjArray* curLayer, TObjArray* prevLayer, 
                      Int_t layerIndex, Int_t numLayers);
      void AddPreLinks(TNeuron* neuron, TObjArray* prevLayer);
     
      // helper functions for weight initialization
      void InitWeights();
      void ForceWeights(std::vector<Double_t>* weights);
      
      // helper functions for deleting network
      void DeleteNetwork();
      void DeleteNetworkLayer(TObjArray*& layer);
      
      // debugging utilities
      void PrintLayer(TObjArray* layer);
      void PrintNeuron(TNeuron* neuron);
      
      // private variables
      Int_t      fNcycles;         // number of epochs to train
      TString    fNeuronType;      // name of neuron activation function class
      TString    fNeuronInputType; // name of neuron input calculator class
      TObjArray* fInputLayer;      // cache this for fast access
      TNeuron*   fOutputNeuron;    // cache this for fast access
      TString    fLayerSpec;       // layout specification option
      
      // some static flags
      static const Bool_t fgDEBUG      = kTRUE;  // debug flag
      static const Bool_t fgFIXED_SEED = kTRUE;  // fix rand generator seed
          
      ClassDef(MethodANNBase,0) // Base class for TMVA ANNs
   };
   
} // namespace TMVA

#endif
