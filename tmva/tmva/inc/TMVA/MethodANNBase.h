// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Peter Speckmayer, Matt Jachowski, Jan Therhaag

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
 *      Peter Speckmayer <Peter.Speckmayer@cern.ch>  - CERN, Switzerland          *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>   - CERN, Switzerland             *
 *      Jan Therhaag       <Jan.Therhaag@cern.ch>     - U of Bonn, Germany        *
 *                                                                                *
 * Small changes (regression):                                                    *
 *      Krzysztof Danielowski <danielow@cern.ch>  - IFJ PAN & AGH, Poland         *
 *      Kamil Kraszewski      <kalq@cern.ch>      - IFJ PAN & UJ , Poland         *
 *      Maciej Kruk           <mkruk@cern.ch>     - IFJ PAN & AGH, Poland         *
 *                                                                                *
 * Copyright (c) 2005-2011:                                                       *
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
#include "TMatrix.h"

#include "TMVA/MethodBase.h"
#include "TMVA/TActivation.h"
#include "TMVA/TNeuron.h"
#include "TMVA/TNeuronInput.h"

class TH1;
class TH1F;

namespace TMVA {

   class MethodANNBase : public MethodBase {

   public:

      // constructors dictated by subclassing off of MethodBase
      MethodANNBase( const TString& jobName,
                     Types::EMVA methodType,
                     const TString& methodTitle,
                     DataSetInfo& theData,
                     const TString& theOption );

      MethodANNBase( Types::EMVA methodType,
                     DataSetInfo& theData,
                     const TString& theWeightFile);

      virtual ~MethodANNBase();

      // this does the real initialization work
      void InitANNBase();

      // setters for subclasses
      void SetActivation(TActivation* activation) {
         if (fActivation != nullptr) delete fActivation;
         fActivation = activation;
      }
      void SetNeuronInputCalculator(TNeuronInput* inputCalculator) {
         if (fInputCalculator != nullptr) delete fInputCalculator;
         fInputCalculator = inputCalculator;
      }

      // this will have to be overridden by every subclass
      virtual void Train() = 0;

      // print network, for debugging
      virtual void PrintNetwork() const;


      // call this function like that:
      // ...
      // MethodMLP* mlp = dynamic_cast<MethodMLP*>(method);
      // std::vector<float> layerValues;
      // mlp->GetLayerActivation (2, std::back_inserter(layerValues));
      // ... do now something with the layerValues
      //
      template <typename WriteIterator>
         void GetLayerActivation (size_t layer, WriteIterator writeIterator);

      using MethodBase::ReadWeightsFromStream;

      // write weights to file
      void AddWeightsXMLTo( void* parent ) const;
      void ReadWeightsFromXML( void* wghtnode );

      // read weights from file
      virtual void ReadWeightsFromStream( std::istream& istr );

      // calculate the MVA value
      virtual Double_t GetMvaValue( Double_t* err = 0, Double_t* errUpper = 0 );

      virtual const std::vector<Float_t> &GetRegressionValues();

      virtual const std::vector<Float_t> &GetMulticlassValues();

      // write method specific histos to target file
      virtual void WriteMonitoringHistosToFile() const;

      // ranking of input variables
      const Ranking* CreateRanking();

      // the option handling methods
      virtual void DeclareOptions();
      virtual void ProcessOptions();

      Bool_t Debug() const;

      enum EEstimator      { kMSE=0,kCE};

      TObjArray*    fNetwork;         // TObjArray of TObjArrays representing network

   protected:

      virtual void MakeClassSpecific( std::ostream&, const TString& ) const;

      std::vector<Int_t>* ParseLayoutString( TString layerSpec );
      virtual void        BuildNetwork( std::vector<Int_t>* layout, std::vector<Double_t>* weights=NULL,
                                        Bool_t fromFile = kFALSE );
      void     ForceNetworkInputs( const Event* ev, Int_t ignoreIndex = -1 );
      Double_t GetNetworkOutput() { return GetOutputNeuron()->GetActivationValue(); }

      // debugging utilities
      void     PrintMessage( TString message, Bool_t force = kFALSE ) const;
      void     ForceNetworkCalculations();
      void     WaitForKeyboard();

      // accessors
      Int_t    NumCycles()  { return fNcycles;   }
      TNeuron* GetInputNeuron (Int_t index)       { return (TNeuron*)fInputLayer->At(index); }
      TNeuron* GetOutputNeuron(Int_t index = 0)   { return fOutputNeurons.at(index); }

      // protected variables
      TObjArray*    fSynapses;        // array of pointers to synapses, no structural data
      TActivation*  fActivation;      // activation function to be used for hidden layers
      TActivation*  fOutput;          // activation function to be used for output layers, depending on estimator
      TActivation*  fIdentity;        // activation for input and output layers
      TRandom3*     frgen;            // random number generator for various uses
      TNeuronInput* fInputCalculator; // input calculator for all neurons

      std::vector<Int_t>        fRegulatorIdx;  //index to different priors from every synapses
      std::vector<Double_t>     fRegulators;    //the priors as regulator
      EEstimator                fEstimator;
      TString                   fEstimatorS;

      // monitoring histograms
      TH1F* fEstimatorHistTrain; // monitors convergence of training sample
      TH1F* fEstimatorHistTest;  // monitors convergence of independent test sample

      // monitoring histograms (not available for regression)
      void CreateWeightMonitoringHists( const TString& bulkname, std::vector<TH1*>* hv = 0 ) const;
      std::vector<TH1*> fEpochMonHistS; // epoch monitoring histograms for signal
      std::vector<TH1*> fEpochMonHistB; // epoch monitoring histograms for background
      std::vector<TH1*> fEpochMonHistW; // epoch monitoring histograms for weights


      // general
      TMatrixD           fInvHessian;           ///< zjh
      bool               fUseRegulator;         ///< zjh

   protected:
      Int_t                   fRandomSeed;      ///< random seed for initial synapse weights

      Int_t                   fNcycles;         ///< number of epochs to train

      TString                 fNeuronType;      ///< name of neuron activation function class
      TString                 fNeuronInputType; ///< name of neuron input calculator class


   private:

      // helper functions for building network
      void BuildLayers(std::vector<Int_t>* layout, Bool_t from_file = false);
      void BuildLayer(Int_t numNeurons, TObjArray* curLayer, TObjArray* prevLayer,
                      Int_t layerIndex, Int_t numLayers, Bool_t from_file = false);
      void AddPreLinks(TNeuron* neuron, TObjArray* prevLayer);

      // helper functions for weight initialization
      void InitWeights();
      void ForceWeights(std::vector<Double_t>* weights);

      // helper functions for deleting network
      void DeleteNetwork();
      void DeleteNetworkLayer(TObjArray*& layer);

      // debugging utilities
      void PrintLayer(TObjArray* layer) const;
      void PrintNeuron(TNeuron* neuron) const;

      // private variables
      TObjArray*              fInputLayer;      ///< cache this for fast access
      std::vector<TNeuron*>   fOutputNeurons;   ///< cache this for fast access
      TString                 fLayerSpec;       ///< layout specification option

      // some static flags
      static const Bool_t fgDEBUG      = kTRUE;  ///< debug flag

      ClassDef(MethodANNBase,0); // Base class for TMVA ANNs
   };



   template <typename WriteIterator>
      inline void MethodANNBase::GetLayerActivation (size_t layerNumber, WriteIterator writeIterator)
      {
         // get the activation values of the nodes in layer "layer"
         // write the node activation values into the writeIterator
         // assumes, that the network has been computed already (by calling
         // "GetRegressionValues")

         if (layerNumber >= (size_t)fNetwork->GetEntriesFast())
            return;

         TObjArray* layer = (TObjArray*)fNetwork->At(layerNumber);
         UInt_t nNodes    = layer->GetEntriesFast();
         for (UInt_t iNode = 0; iNode < nNodes; iNode++)
            {
               (*writeIterator) = ((TNeuron*)layer->At(iNode))->GetActivationValue();
               ++writeIterator;
            }
      }


} // namespace TMVA

#endif
