// @(#)root/tmva $Id: MethodMLP.h,v 1.21 2006/10/10 17:43:52 andreas.hoecker Exp $
// Author: Andreas Hoecker, Matt Jachowski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodMLP                                                             *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      ANN Multilayer Perceptron  class for the discrimination of signal         *
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

#ifndef ROOT_TMVA_MethodMLP
#define ROOT_TMVA_MethodMLP

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodMLP                                                            //
//                                                                      //
// Multilayer Perceptron built off of MethodANNBase                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <vector>
#include "TString.h"
#include "TTree.h"
#include "TObjArray.h"
#include "TRandom3.h"
#include "TH1F.h"

#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#endif
#ifndef ROOT_TMVA_MethodANNBase
#include "TMVA/MethodANNBase.h"
#endif
#ifndef ROOT_TMVA_TNeuron
#include "TMVA/TNeuron.h"
#endif
#ifndef ROOT_TMVA_TActivation
#include "TMVA/TActivation.h"
#endif

#define MethodMLP_UseMinuit__
#undef  MethodMLP_UseMinuit__

namespace TMVA {

   class MethodMLP : public MethodANNBase {

   public:

      // standard constructors
      MethodMLP( TString jobName, 
                 TString  methodTitle,
                 DataSet& theData,
                 TString theOption, 
                 TDirectory* theTargetDir = 0 );

      MethodMLP( DataSet& theData, 
                 TString theWeightFile, 
                 TDirectory* theTargetDir = 0 );

      virtual ~MethodMLP();

      void Train() { Train(NumCycles()); }

      // for GA
      Double_t ComputeEstimator(const std::vector<Double_t>& parameters);

      enum TrainingMethod { kBP=0, kGA };
      enum BPTrainingMode { kSequential=0, kBatch };

   private:

      // the option handling methods
      virtual void DeclareOptions();
      virtual void ProcessOptions();

      // general helper functions
      void     Train( Int_t nEpochs );
      void     InitMLP();
      void     InitializeLearningRates(); // although this is only needed by backprop

      // used as a measure of success in all minimization techniques
      Double_t CalculateEstimator( Types::TreeType treeType = Types::kTrain );

      // backpropagation functions
      void     BackPropagationMinimize( Int_t nEpochs );
      void     TrainOneEpoch();
      void     Shuffle( Int_t* index, Int_t n );
      void     DecaySynapseWeights(Bool_t lateEpoch );
      void     TrainOneEvent( Int_t ievt);
      Double_t GetDesiredOutput();
      void     UpdateNetwork( Double_t desired, Double_t eventWeight=1.0 );
      void     CalculateNeuronDeltas();
      void     UpdateSynapses();
      void     AdjustSynapseWeights();

      // faster backpropagation
      void     TrainOneEventFast( Int_t ievt, Float_t*& branchVar, Int_t& type );

      // genetic algorithm functions
      void GeneticMinimize();

#ifdef MethodMLP_UseMinuit__
      // minuit functions -- commented out because they rely on a static pointer
      void MinuitMinimize();
      static MethodMLP* GetThisPtr() { return fgThis; }
      static void IFCN( Int_t& npars, Double_t* grad, Double_t &f, Double_t* fitPars, Int_t ifl );
      void FCN( Int_t& npars, Double_t* grad, Double_t &f, Double_t* fitPars, Int_t ifl );
#endif

      // general
      TrainingMethod fTrainingMethod; // method of training, BP or GA
      TString        fTrainMethodS;   // training method option param

      // backpropagation variables
      Double_t       fLearnRate;      // learning rate for synapse weight adjustments
      Double_t       fDecayRate;      // decay rate for above learning rate
      BPTrainingMode fBPMode;         // backprop learning mode (sequential or batch)
      TString        fBpModeS;        // backprop learning mode option string (sequential or batch)
      Int_t          fBatchSize;      // batch size, only matters if in batch learning mode
      Int_t          fTestRate;       // test for overtraining performed at each #th epochs
      
      // genetic algorithm variables
      Int_t          fGA_nsteps;      // GA settings: number of steps
      Int_t          fGA_preCalc;     // GA settings: number of pre-calc steps
      Int_t          fGA_SC_steps;    // GA settings: SC_steps
      Int_t          fGA_SC_offsteps; // GA settings: SC_offsteps
      Double_t       fGA_SC_factor;   // GA settings: SC_factor

#ifdef MethodMLP_UseMinuit__
      // minuit variables -- commented out because they rely on a static pointer
      Int_t          fNumberOfWeights; // Minuit: number of weights
      static MethodMLP* fgThis;        // Minuit: this pointer
#endif

      // debugging flags
      static const Int_t  fgPRINT_ESTIMATOR_INC = 10;     // debug flags
      static const Bool_t fgPRINT_SEQ           = kFALSE; // debug flags
      static const Bool_t fgPRINT_BATCH         = kFALSE; // debug flags

      ClassDef(MethodMLP,0) // Multi-layer perceptron implemented specifically for TMVA
         ;
	};

} // namespace TMVA

#endif
