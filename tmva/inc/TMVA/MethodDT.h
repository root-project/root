// @(#)root/tmva $Id$ 
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodDT  (Boosted Decision Trees)                                   *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Analysis of Boosted Decision Trees                                        *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Or Cohen        <orcohenor@gmail.com>    - Weizmann Inst., Israel         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodDT
#define ROOT_TMVA_MethodDT

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodDT                                                             //
//                                                                      //
// Analysis of Single Decision Tree                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <vector>
#ifndef ROOT_TH1
#include "TH1.h"
#endif
#ifndef ROOT_TH2
#include "TH2.h"
#endif
#ifndef ROOT_TTree
#include "TTree.h"
#endif
#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#endif
#ifndef ROOT_TMVA_DecisionTree
#include "TMVA/DecisionTree.h"
#endif
#ifndef ROOT_TMVA_Event
#include "TMVA/Event.h"
#endif

namespace TMVA {
   class MethodBoost;

   class MethodDT : public MethodBase {
   public:
      MethodDT( const TString& jobName, 
                const TString& methodTitle, 
                DataSetInfo& theData,
                const TString& theOption = "",
                TDirectory* theTargetDir = 0 );

      MethodDT( DataSetInfo& dsi, 
                const TString& theWeightFile,  
                TDirectory* theTargetDir = NULL );

      virtual ~MethodDT( void );

      virtual Bool_t HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses, UInt_t numberTargets );

      void Train( void );
      
      using MethodBase::ReadWeightsFromStream;

      // write weights to file
      void AddWeightsXMLTo( void* parent ) const;

      // read weights from file
      void ReadWeightsFromStream( std::istream& istr );
      void ReadWeightsFromXML   ( void* wghtnode );

      // calculate the MVA value
      Double_t GetMvaValue( Double_t* err = 0, Double_t* errUpper = 0 );

      // the option handling methods
      void DeclareOptions();
      void ProcessOptions();
      void DeclareCompatibilityOptions();

      void GetHelpMessage() const;

      // ranking of input variables
      const Ranking* CreateRanking();

      Double_t PruneTree( );

      Double_t TestTreeQuality( DecisionTree *dt );

      Double_t GetPruneStrength () { return fPruneStrength; }

      void SetMinNodeSize(Double_t sizeInPercent);
      void SetMinNodeSize(TString sizeInPercent);

      Int_t GetNNodesBeforePruning(){return fTree->GetNNodesBeforePruning();}
      Int_t GetNNodes(){return fTree->GetNNodes();}

   private:
      // Init used in the various constructors
      void Init( void );

   private:


      std::vector<Event*>             fEventSample;     // the training events

      DecisionTree*                   fTree;            // the decision tree
      //options for the decision Tree
      SeparationBase                 *fSepType;         // the separation used in node splitting
      TString                         fSepTypeS;        // the separation (option string) used in node splitting
      Int_t                           fMinNodeEvents;   // min number of events in node
      Float_t                         fMinNodeSize;     // min percentage of training events in node
      TString                         fMinNodeSizeS;    // string containing min percentage of training events in node
  
      Int_t                           fNCuts;           // grid used in cut applied in node splitting
      Bool_t                          fUseYesNoLeaf;    // use sig or bkg classification in leave nodes or sig/bkg
      Double_t                        fNodePurityLimit; // purity limit for sig/bkg nodes
      UInt_t                          fMaxDepth;        // max depth


      Double_t                         fErrorFraction;   // ntuple var: misclassification error fraction 
      Double_t                         fPruneStrength;   // a parameter to set the "amount" of pruning..needs to be adjusted
      DecisionTree::EPruneMethod       fPruneMethod;     // method used for prunig 
      TString                          fPruneMethodS;    // prune method option String
      Bool_t                           fAutomatic;       // use user given prune strength or automatically determined one using a validation sample 
      Bool_t                           fRandomisedTrees; // choose a random subset of possible cut variables at each node during training
      Int_t                            fUseNvars;        // the number of variables used in the randomised tree splitting
      Bool_t                           fUsePoissonNvars; // fUseNvars is used as a poisson mean, and the actual value of useNvars is at each step drawn form that distribution
      std::vector<Double_t>           fVariableImportance; // the relative importance of the different variables 

      Double_t                        fDeltaPruneStrength; // step size in pruning, is adjusted according to experience of previous trees        
      // debugging flags
      static const Int_t  fgDebugLevel = 0;     // debug level determining some printout/control plots etc.


      Bool_t fPruneBeforeBoost; //aincient variable, only needed for "CompatibilityOptions" 

      ClassDef(MethodDT,0)  // Analysis of Decision Trees 

         };
}

#endif
